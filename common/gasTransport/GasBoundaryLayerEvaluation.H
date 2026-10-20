#pragma once
#include "GasBoundaryLayerState.H"
#include "GasStateView.H"
#include "GasCapabilities.H"
#include "../gasWall/WallModelTypes.H"

namespace ugkwp
{
// Publish physical-wall exchange, not matching flux or auxiliary chemistry.
// normalSweepRate is grid velocity dot MATERIAL->GAS normal, in m/s.
// All stored fluxes are outward integrated rates, matching common gas kernels.
template<class State,class Real,int Ns>
UGKWP_GAS_HD inline bool publishGasBoundaryLayerOutput
(
    const State& state,int slot,const gaswall::WallInput<Real,Ns>& input,
    const gaswall::WallOutput<Real,Ns>& output,Real area,Real normalSweepRate
)
{
    auto& wall=state.gasBoundaryLayer;
    if(!wall.enabled)return false;
    if(slot<0 || slot>=wall.count || !wall.exchange || !wall.sst || !wall.status || !wall.speciesFlux) {
        const auto code=(slot<0 || slot>=wall.count)?ugkp::diagnostics::Code::InvalidIndex:
            ugkp::diagnostics::Code::MissingOperator;
        const int component=!wall.exchange?0:!wall.sst?1:!wall.status?2:!wall.speciesFlux?3:-1;
        ugkp::diagnostics::recordFatal(state.diagnostics,ugkp::diagnostics::makeRecord(
            code,2507,25,0,ugkp::diagnostics::EntityKind::Face,slot,component,
            code==ugkp::diagnostics::Code::InvalidIndex?slot:0,
            code==ugkp::diagnostics::Code::InvalidIndex?wall.count:1));
        return false;
    }
    wall.exchange[slot].ready=false;
    wall.exchange[slot].massReady=false;
    const auto failure=[&](std::uint32_t site,Real value,int component,
        ugkp::diagnostics::Code code=ugkp::diagnostics::Code::NonFinite) {
        auto record=ugkp::diagnostics::makeRecord(code,site,25,0,
            ugkp::diagnostics::EntityKind::Face,slot,component,static_cast<double>(value));
        ugkp::diagnostics::recordFatal(state.diagnostics,record);
        wall.status[slot]=int(GasTransportCode::NonFiniteState);
        return false;
    };
    if(!gasFinite(area))return failure(2501,area,0);
    if(!gasFinite(normalSweepRate))return failure(10054,normalSweepRate,1);
    if(!gasFinite(output.traceTemperature))return failure(2502,output.traceTemperature,0);
    if(output.traceTemperature<0)return failure(2508,output.traceTemperature,0,ugkp::diagnostics::Code::NegativeTemperature);
    if(!gasFinite(input.pressure))return failure(10055,input.pressure,1);
    if(!gasFinite(output.conductiveHeatFlux))return failure(10056,output.conductiveHeatFlux,2);

    GasBoundaryLayerExchange<Real> candidate;
    Real mass=0,enthalpy=0,speed2=0,viscousWork=0,tractionScale=0,scaledSpecies[Ns];
    for(int d=0;d<3;++d)
    {
        if(!gasFinite(output.traceVelocity[d]))return failure(2503,output.traceVelocity[d],d);
        if(!gasFinite(output.traction[d]))return failure(10057,output.traction[d],3+d);
        if(!gasFinite(input.normal[d]))return failure(10058,input.normal[d],6+d);
        speed2+=output.traceVelocity[d]*output.traceVelocity[d];
        tractionScale=gaswall::detail::max(tractionScale,gaswall::detail::abs(output.traction[d]));
        viscousWork+=output.traction[d]*output.traceVelocity[d];
        if(!gasFinite(speed2))return failure(2505,speed2,0);
        if(!gasFinite(viscousWork))return failure(10059,viscousWork,1);
    }
    // Project before taking a max-scaled norm: normal pressure/roundoff must
    // not produce turbulent shear, and squaring tiny/large traction is unsafe.
    Real shear=0;
    if(tractionScale>0)
    {
        Real normalTraction=0,tangential2=0;
        for(int d=0;d<3;++d) {
            normalTraction+=(output.traction[d]/tractionScale)*input.normal[d];
            if(!gasFinite(normalTraction))return failure(10060,normalTraction,4);
        }
        for(int d=0;d<3;++d)
        {
            const Real tangential=output.traction[d]/tractionScale-normalTraction*input.normal[d];
            if(!gasFinite(tangential))return failure(10061,tangential,5);
            tangential2+=tangential*tangential;
            if(!gasFinite(tangential2))return failure(10062,tangential2,6);
        }
        shear=tractionScale*::sqrt(tangential2);
        if(!gasFinite(shear))return failure(10063,shear,7);
    }
    for(int species=0;species<Ns;++species)
    {
        const Real flux=output.wallSpeciesFlux[species];
        if(!gasFinite(flux))return failure(2504,flux,species);
        gasThermoDetail::SpeciesValues<Real> values;bool available=false;
        gasThermoDetail::evaluateSpecies<Real,Ns,Real,true,true>
            (species,output.traceTemperature,input.model.thermo,values,&available);
        if(!available)return failure(10064,output.traceTemperature,species,
            ugkp::diagnostics::Code::InvalidPhysicalState);
        const Real h=values.h;
        if(!gasFinite(h))return failure(10065,h,Ns+species);
        scaledSpecies[species]=-area*flux;
        if(!gasFinite(scaledSpecies[species]))return failure(10066,scaledSpecies[species],2*Ns+species);
        mass+=flux;enthalpy+=flux*h;
        if(!gasFinite(mass))return failure(10067,mass,2);
        if(!gasFinite(enthalpy))return failure(10068,enthalpy,3);
    }
    candidate.matchingPressure=input.pressure;
    candidate.mass=-area*mass;
    candidate.momentumX=-area*(mass*output.traceVelocity[0]+input.pressure*input.normal[0]-output.traction[0]);
    candidate.momentumY=-area*(mass*output.traceVelocity[1]+input.pressure*input.normal[1]-output.traction[1]);
    candidate.momentumZ=-area*(mass*output.traceVelocity[2]+input.pressure*input.normal[2]-output.traction[2]);
    candidate.energy=-area*(enthalpy+Real(.5)*mass*speed2+output.conductiveHeatFlux
        +input.pressure*normalSweepRate-viscousWork);
    candidate.k=input.volumeSstClosure?-area*output.wallKFlux:Real(0);
    const Real published[]={candidate.mass,candidate.momentumX,candidate.momentumY,
        candidate.momentumZ,candidate.energy,candidate.k,shear,output.ownerOmega,
        input.ownerDistance,input.quadrature.volume};
    for(int component=0;component<10;++component)
        if(!gasFinite(published[component]))return failure(2506,published[component],component);
    for(int species=0;species<Ns;++species)
        wall.speciesFlux[species*wall.count+slot]=scaledSpecies[species];
    auto& closure=wall.sst[slot];
    closure={};closure.wallShearStress=shear;closure.ownerOmega=output.ownerOmega;
    closure.volume=input.quadrature.volume;closure.ownerDistance=input.ownerDistance;
    if(input.volumeSstClosure){
        closure.nonDestructiveKSource=(output.integratedKSource+output.ownerKDestruction)/input.quadrature.volume;
        for(int d=0;d<3;++d)closure.normal[d]=input.normal[d];
        closure.kFluxCount=output.kFluxCount;closure.kFluxHeight=output.kFluxHeight;closure.kDiffusiveFlux=output.kDiffusiveFlux;closure.omegaDiffusiveFlux=output.omegaDiffusiveFlux;
    }
    candidate.ready=true;
    candidate.massReady=true;
    wall.exchange[slot]=candidate;
    wall.status[slot]=0;
    return true;
}
}
