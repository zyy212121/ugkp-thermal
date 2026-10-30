#pragma once
#include "gasTransport/MixtureParticleDirectory.cuh"
#include "gasTransport/MixtureCouplingTeam.cuh"
// Caloric adaptation of the shared frozen-mixture finite-bath source. Free
// parcels alone participate; contact/profile heat has its own accepted ledger.
// One cell owns all candidate evaluations and all publication for its parcels.
namespace mixtureThermalParticles {
template<class State>
__device__ bool member(const State& s,int i,int c) {
    return mixtureParticles::member(s,i,c) && s.pStuck[i]==0;
}
template<class State>
__device__ ugkwp::ParticleCaloricHeatResponse response(const State& s,int c,int i,double dt) {
    const auto law=particleCaloricDevice(s,i);
    const double oldT=particleEffectiveTemperatureDevice(s,i);
    const double d=s.pd[i];
    const double dx=s.Ux[c]-s.pux[i],dy=s.Uy[c]-s.puy[i],dz=s.Uz[c]-s.puz[i];
    const double mu=ugkwp::gasCellMolecularViscosityAt(s,c,s.Tgas[c]);
    const double re=s.rho[c]*d*sqrt(dx*dx+dy*dy+dz*dz)/fmax(mu,1e-30);
    const double rate=particleGasHeatRelaxationRate(s,c,s.Tgas[c],re,d,
        s.rhoSolid*law.specificHeat(oldT));
    return {law,oldT,-expm1(-fmax(0.,rate)*dt),s.TpMin,s.TpMax};
}
template<class State>
__device__ ugkwp::ParticleCaloricHeatResponse cachedResponse(const State& s,int i){
    return {particleCaloricDevice(s,i),s.mixtureHeatInitialTemperature[i],
        s.mixtureHeatFraction[i],s.TpMin,s.TpMax};
}
template<class State,class Team>
struct Response {
    const State& state;int cell;double dt,volume;bool csr,split;Team team;
    __device__ ugkwp::ThermoStatus operator()(double T,double& q,double& derivative)const noexcept {
        q=0.;derivative=0.;
        const int n=mixtureParticles::count(state,cell,csr,split);
        for(int pos=team.rank();pos<n;pos+=team.width()) {
            const int i=mixtureParticles::parcel(state,cell,pos,csr,split);
            if(!member(state,i,cell))continue;
            double h,dh;cachedResponse(state,i)(T,h,dh);
            q+=state.pm[i]*h/volume;derivative+=state.pm[i]*dh/volume;
        }
        q=team.sum(q);derivative=team.sum(derivative);
        return ugkwp::gasFinite(q)&&ugkwp::gasFinite(derivative)?
            ugkwp::ThermoStatus::Success:ugkwp::ThermoStatus::NonFiniteInput;
    }
};
}
// Freeze only quantities independent of the gas root. Parcel enthalpy and its
// derivative still use every trial gas temperature in the original equation.
// This source-epoch kernel runs after drag and before any thermal publication.
template<class State>
__global__ void prepareMixtureThermalParticleHeatKernel(State*sp,double dt){
    State&s=*sp;
    if(!s.solveParticleTemperature||!s.particleGasHeatTransferModelId||dt==0.)return;
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
__device__ void mixtureThermalParticleHeatCell(State&s,int c,double dt,bool csr,bool split,const Team&team) {
    constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
    double species[Ns];for(int k=0;k<Ns;++k)species[k]=s.gasSpecies.rho[k*s.nCells+c];
    const double volume=s.epsGPrev[c]*s.V[c];
    const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
    const double oldInternal=s.rhoE[c]-kinetic;
    const mixtureThermalParticles::Response<State,Team> response{s,c,dt,volume,csr,split,team};
    double target=s.Tgas[c];
    const auto status=ugkwp::invertMixtureParticleHeatExchange(species,oldInternal,response,
        s.gasSpecies.thermo,s.gasSpecies.thermoControls,target);
    if(status!=ugkwp::ThermoStatus::Success) {
        if(team.rank()==0){s.gasSpecies.cellStatus[c]=static_cast<int>(status);
        using namespace ugkp::diagnostics;
        auto failure=makeRecord(Code::InvalidPhysicalState,30700,0,0,
            EntityKind::Cell,c,-1,static_cast<int>(status));
        failure.metadata=static_cast<unsigned>(status);recordFatal(s.diagnostics,failure);}
        return;
    }
    double heat=0.;bool failed=false;
    const int n=mixtureParticles::count(s,c,csr,split);
    for(int pos=team.rank();pos<n;pos+=team.width()) {
        const int i=mixtureParticles::parcel(s,c,pos,csr,split);
        if(!mixtureThermalParticles::member(s,i,c))continue;
        const auto response=mixtureThermalParticles::cachedResponse(s,i);
        double q,derivative;response(target,q,derivative);
        ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i],s.pThermalEnthalpyResidual[i]};
        const auto publication=ugkwp::publishParticleThermalHeat(state,q,s.TpMin,s.TpMax,response.law);
        if(!publication.valid){failed=true;break;}
        s.pT[i]=state.temperature;s.pThermalEnthalpyResidual[i]=state.specificEnthalpyResidual;
        heat+=s.pm[i]*publication.acceptedSpecificHeat;
    }
    heat=team.sum(heat);const bool anyFailure=team.sum(failed?1.:0.)>0;
    if(team.rank()==0){
        s.rhoE[c]-=heat/volume;recoverGasPrimitiveCell(s,c);
        if(anyFailure){
            s.gasSpecies.cellStatus[c]=static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput);
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::InvalidPhysicalState,30700,0,0,
                EntityKind::Cell,c,-1,static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput)));
        }
    }
}

template<bool Cooperative=false,class State>
__global__ void mixtureThermalParticleHeatKernel(State*sp,double dt,bool csr=true,bool split=false,bool skipHeavy=false){
 State&s=*sp;if(!s.solveParticleTemperature||!s.particleGasHeatTransferModelId||dt==0.)return;
 const int thread=blockIdx.x*blockDim.x+threadIdx.x;
#if defined(__CUDACC__)
 if constexpr(Cooperative){
  for(int c=thread/32;c<s.nCells;c+=gridDim.x*(blockDim.x/32))
   if(mixtureParticles::count(s,c,csr,split)>=32)
    mixtureThermalParticleHeatCell(s,c,dt,csr,split,mixtureParticles::WarpCouplingTeam{});
 }else
#endif
 {if(thread<s.nCells&&(!skipHeavy||mixtureParticles::count(s,thread,csr,split)<32))
   mixtureThermalParticleHeatCell(s,thread,dt,csr,split,mixtureParticles::SerialCouplingTeam{});}
}
#if defined(__CUDACC__)
template<class State>
inline cudaError_t launchMixtureThermalParticleHeatCells(State*state,int cells,double dt,bool csr,bool split){
 if(cells<=0)return cudaSuccess;
 mixtureThermalParticleHeatKernel<<<(cells+127)/128,128>>>(state,dt,csr,split,true);
 auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 const int groups=(cells+3)/4;
 mixtureThermalParticleHeatKernel<true><<<groups<256?groups:256,128>>>(state,dt,csr,split);
 return cudaGetLastError();
}
#endif
