#pragma once
#include "GasBoundaryLayerState.H"
#include "GasStateView.H"
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
    if(!wall.enabled || slot<0 || slot>=wall.count || !wall.exchange || !wall.sst
        || !wall.status || !wall.speciesFlux)return false;
    wall.exchange[slot].ready=false;
    if(!gasFinite(area) || !(area>0) || !gasFinite(normalSweepRate)
        || !gasFinite(output.traceTemperature) || !(output.traceTemperature>0)
        || !gasFinite(input.pressure) || !(input.pressure>0))return false;
    GasBoundaryLayerExchange<Real> candidate;
    Real mass=0,enthalpy=0,speed2=0,viscousWork=0;
    for(int d=0;d<3;++d)
    {
        if(!gasFinite(output.traceVelocity[d]) || !gasFinite(output.traction[d])
            || !gasFinite(input.normal[d]))return false;
        speed2+=output.traceVelocity[d]*output.traceVelocity[d];
        viscousWork+=output.traction[d]*output.traceVelocity[d];
    }
    for(int species=0;species<Ns;++species)
    {
        const Real flux=output.wallSpeciesFlux[species];
        const Real h=speciesH(species,output.traceTemperature,input.model.thermo);
        if(!gasFinite(flux) || !gasFinite(h))return false;
        mass+=flux;enthalpy+=flux*h;
    }
    candidate.matchingPressure=input.pressure;
    candidate.mass=-area*mass;
    candidate.momentumX=-area*(mass*output.traceVelocity[0]+input.pressure*input.normal[0]-output.traction[0]);
    candidate.momentumY=-area*(mass*output.traceVelocity[1]+input.pressure*input.normal[1]-output.traction[1]);
    candidate.momentumZ=-area*(mass*output.traceVelocity[2]+input.pressure*input.normal[2]-output.traction[2]);
    candidate.energy=-area*(enthalpy+Real(.5)*mass*speed2+output.conductiveHeatFlux
        +input.pressure*normalSweepRate-viscousWork);
    candidate.k=-area*output.wallKFlux;
    if(!gasFinite(candidate.mass) || !gasFinite(candidate.momentumX) || !gasFinite(candidate.momentumY)
        || !gasFinite(candidate.momentumZ) || !gasFinite(candidate.energy) || !gasFinite(candidate.k)
        || !gasFinite(output.integratedKSource) || !gasFinite(output.ownerOmega)
        || !gasFinite(input.quadrature.volume) || !(input.quadrature.volume>0))return false;
    for(int species=0;species<Ns;++species)
        wall.speciesFlux[species*wall.count+slot]=-area*output.wallSpeciesFlux[species];
    wall.sst[slot]={output.integratedKSource,output.ownerOmega,input.quadrature.volume};
    candidate.ready=true;
    candidate.massReady=true;
    wall.exchange[slot]=candidate;
    wall.status[slot]=0;
    return true;
}
}
