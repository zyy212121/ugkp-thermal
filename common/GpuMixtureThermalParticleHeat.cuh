#pragma once
#include "gasTransport/MixtureParticleDirectory.cuh"
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
struct Response {
    const State& state;int cell;double dt,volume;bool csr,split;
    __device__ ugkwp::ThermoStatus operator()(double T,double& q,double& derivative)const noexcept {
        q=0.;derivative=0.;
        const int n=mixtureParticles::count(state,cell,csr,split);
        for(int pos=0;pos<n;++pos) {
            const int i=mixtureParticles::parcel(state,cell,pos,csr,split);
            if(!member(state,i,cell))continue;
            double h,dh;response(state,cell,i,dt)(T,h,dh);
            q+=state.pm[i]*h/volume;derivative+=state.pm[i]*dh/volume;
        }
        return ugkwp::gasFinite(q)&&ugkwp::gasFinite(derivative)?
            ugkwp::ThermoStatus::Success:ugkwp::ThermoStatus::NonFiniteInput;
    }
};
}
template<class State>
__global__ void mixtureThermalParticleHeatKernel(State* sp,double dt,bool csr=true,bool split=false) {
    State& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells || !s.solveParticleTemperature || !s.particleGasHeatTransferModelId || dt==0.)return;
    constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
    double species[Ns];for(int k=0;k<Ns;++k)species[k]=s.gasSpecies.rho[k*s.nCells+c];
    const double volume=s.epsGPrev[c]*s.V[c];
    const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
    const double oldInternal=s.rhoE[c]-kinetic;
    const mixtureThermalParticles::Response<State> response{s,c,dt,volume,csr,split};
    double target=s.Tgas[c];
    const auto status=ugkwp::invertMixtureParticleHeatExchange(species,oldInternal,response,
        s.gasSpecies.thermo,s.gasSpecies.thermoControls,target);
    if(status!=ugkwp::ThermoStatus::Success) {
        s.gasSpecies.cellStatus[c]=static_cast<int>(status);
        using namespace ugkp::diagnostics;
        auto failure=makeRecord(Code::InvalidPhysicalState,30700,0,0,
            EntityKind::Cell,c,-1,static_cast<int>(status));
        failure.metadata=static_cast<unsigned>(status);recordFatal(s.diagnostics,failure);return;
    }
    double heat=0.;
    const int n=mixtureParticles::count(s,c,csr,split);
    for(int pos=0;pos<n;++pos) {
        const int i=mixtureParticles::parcel(s,c,pos,csr,split);
        if(!mixtureThermalParticles::member(s,i,c))continue;
        const auto response=mixtureThermalParticles::response(s,c,i,dt);
        double q,derivative;response(target,q,derivative);
        ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i],s.pThermalEnthalpyResidual[i]};
        const auto publication=ugkwp::publishParticleThermalHeat(state,q,s.TpMin,s.TpMax,response.law);
        if(!publication.valid) {
            // Preserve the reverse settlement for any already published
            // parcels and report this step's existing invalid-state diagnostic.
            s.rhoE[c]-=heat/volume;recoverGasPrimitiveCell(s,c);
            s.gasSpecies.cellStatus[c]=static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput);
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::InvalidPhysicalState,30700,0,0,
                EntityKind::Cell,c,-1,static_cast<int>(ugkwp::ThermoStatus::NonFiniteInput)));
            return;
        }
        s.pT[i]=state.temperature;s.pThermalEnthalpyResidual[i]=state.specificEnthalpyResidual;
        heat+=s.pm[i]*publication.acceptedSpecificHeat;
    }
    s.rhoE[c]-=heat/volume;
    recoverGasPrimitiveCell(s,c);
}
