#pragma once
#include "ParticleResidentInterval.H"
#include "gasTransport/MixtureParticleDirectory.cuh"
#include "gasTransport/MixtureCouplingTeam.cuh"
#include "gasTransport/MixtureParticleResponse.cuh"



namespace mixtureThermalParticles {
template<class State>
__device__ bool member(const State& s,int i,int c) {
    return mixtureParticles::member(s,i,c) && s.pStuck[i]==0;
}
template<class State>
__device__ ugkwp::ParticleCaloricHeatResponse response(const State& s,int i,GpuTime dt,
    const mixtureParticles::HeatCellProperties& properties) {
    const auto law=particleCaloricDevice(s,i);
    const float oldT=particleEffectiveTemperatureDevice(s,i);
    const float d=s.pd[i];
    const auto& gas=properties.gas;
    const float dx=gas.ux-s.pux[i],dy=gas.uy-s.puy[i],dz=gas.uz-s.puz[i];
    const float mu=gas.viscosity;
    const float re=gas.rho*d*sqrt(dx*dx+dy*dy+dz*dz)/fmax(mu,1e-30f);
    const float rate=mixtureParticles::caloricHeatRate(properties,re,d,
        s.rhoSolid*law.specificHeat(oldT));
    return {law,oldT,-expm1(-fmax(0.f,rate)*GpuReal(dt)),s.TpMin,s.TpMax};
}
template<class State>
__device__ ugkwp::ParticleCaloricHeatResponse response(const State& s,int c,int i,GpuTime dt) {
    return response(s,i,dt,mixtureParticles::heatCellProperties(s,c));
}



template<class State,class Team>
__device__ void prepareCell(State& s,int c,GpuTime dt,bool csr,bool split,const Team& team) {
    mixtureParticles::HeatCellProperties properties{};
    bool propertiesReady=false;
    const int n=mixtureParticles::count(s,c,csr,split);
    for(int pos=team.rank();pos<n;pos+=team.width()) {
        const int i=mixtureParticles::parcel(s,c,pos,csr,split);
        if(!member(s,i,c))continue;
        if(!propertiesReady) {
            properties=mixtureParticles::heatCellProperties(s,c);
            propertiesReady=true;
        }
        const auto value=response(s,i,dt,properties);
        s.mixtureHeatInitialTemperature[i]=value.initialTemperature;
        s.mixtureHeatFraction[i]=value.fraction;
    }
    team.sync();
}
template<class State>
__device__ ugkwp::ParticleCaloricHeatResponse cachedResponse(const State& s,int i){
    return {particleCaloricDevice(s,i),s.mixtureHeatInitialTemperature[i],
        s.mixtureHeatFraction[i],s.TpMin,s.TpMax};
}
#include "GpuParticleCaloricResponseSum.cuh"
}





template<class State>
__global__ void prepareMixtureThermalParticleHeatKernel(State*sp,GpuTime requestedDt){
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
    State&s=*sp;
    if(!s.solveParticleTemperature||!s.particleGasHeatTransferModelId||dt==0.f)return;
    const int count=mixtureParticles::activeCount(s);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
        const int c=s.pCellId[i];
        if(c<0||c>=s.nCells||!mixtureThermalParticles::member(s,i,c))continue;
        const auto value=mixtureThermalParticles::response(s,c,i,dt);
        s.mixtureHeatInitialTemperature[i]=value.initialTemperature;
        s.mixtureHeatFraction[i]=value.fraction;
    }
}
template<class State,class Team>
__device__ void mixtureThermalParticleHeatCell(State&s,int c,GpuTime dt,bool csr,bool split,const Team&team) {
    mixtureThermalParticles::prepareCell(s,c,dt,csr,split,team);
    constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
    float species[Ns];for(int k=0;k<Ns;++k)species[k]=s.gasSpecies.rho[k*s.nCells+c];
    const float volume=s.epsGPrev[c]*GpuReal(s.V[c]);
    const float kinetic=.5f*(float(s.rhoUx[c])*float(s.rhoUx[c])
        +float(s.rhoUy[c])*float(s.rhoUy[c])+float(s.rhoUz[c])*float(s.rhoUz[c]))/float(s.rho[c]);
    const float oldInternal=s.rhoE[c]-kinetic;
    const mixtureThermalParticles::Response<State,Team> response{s,c,dt,volume,csr,split,team};
    float target=s.Tgas[c];
    const auto status=ugkwp::invertMixtureParticleHeatExchange(species,oldInternal,response,
        s.gasSpecies.thermo,s.gasSpecies.thermoControls,target);
    if(status!=ugkwp::ThermoStatus::Success) {
        if(team.rank()==0){s.gasSpecies.cellStatus[c]=static_cast<int>(status);
        using namespace ugkp::diagnostics;
        auto failure=makeRecord(Code::InvalidPhysicalState,30702,0,0,
            EntityKind::Cell,c,-1,static_cast<int>(status));
        failure.metadata=static_cast<unsigned>(status);recordFatal(s.diagnostics,failure);}
        return;
    }
    float heat=0.f;bool failed=false;
    const int n=mixtureParticles::count(s,c,csr,split);
    for(int pos=team.rank();pos<n;pos+=team.width()) {
        const int i=mixtureParticles::parcel(s,c,pos,csr,split);
        if(!mixtureThermalParticles::member(s,i,c))continue;
        const auto response=mixtureThermalParticles::cachedResponse(s,i);
        float q,derivative;response(target,q,derivative);
        ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i]};
        const auto publication=ugkwp::publishParticleThermalHeat(state,q,s.TpMin,s.TpMax,response.law);
        if(!publication.valid){failed=true;break;}
        s.pT[i]=state.temperature;
        heat+=s.pm[i]*publication.acceptedSpecificHeat;
    }
    heat=team.sum(heat);const bool anyFailure=team.sum(failed?1.f:0.f)>0;
    if(team.rank()==0){
        s.rhoE[c]-=heat/volume;recoverGasPrimitiveCell(s,c);
        if(anyFailure){
            s.gasSpecies.cellStatus[c]=static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput);
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::InvalidPhysicalState,30701,0,0,
                EntityKind::Cell,c,-1,static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput)));
        }
    }
}
