#pragma once
#include "ParticleResidentInterval.H"
#include <type_traits>
#include <utility>
#include "MixtureHeatExchange.H"
#include "MixtureParticleDirectory.cuh"
#include "MixtureCouplingTeam.cuh"
#include "MixtureParticleResponse.cuh"

// Physical-time source transaction shared by gasUGKP and gasUGKPSteady. One cell worker
// owns all parcel writes and the reverse gas publication for its cell. CSR and
// split direct-base + indexed-injection ranges consume the same post-collision
// epoch. A full scan remains available only for the serial reference oracle.

template<class DragModel,bool ConstantHeat,class State,class Team>
__device__ void mixtureParticleCouplingCell(State&s,int c,double dt,bool dragActive,bool useCsr,bool useSplit,const Team&team)
{
    const int begin=0,end=mixtureParticles::count(s,c,useCsr,useSplit);
    const double epsG=s.epsGPrev[c],volume=epsG*s.V[c];
    const double gasMass=volume*s.rho[c];
    double weight=0,x=0,y=0,z=0;
    mixtureParticles::DragCellProperties dragProperties{};
    bool dragPropertiesReady=false;
    // No parcel or gas state is changed while forming the finite-bath target.
    for(int pos=begin+team.rank();pos<end;pos+=team.width())
    {
        const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
        if(!mixtureParticles::member(s,i,c))continue;
        s.puxOld[i]=s.pux[i];s.puyOld[i]=s.puy[i];s.puzOld[i]=s.puz[i];
        if(dragActive)
        {
            if(!dragPropertiesReady) {
                dragProperties=mixtureParticles::dragCellProperties(s,c);
                dragPropertiesReady=true;
            }
            const auto b=mixtureParticles::beta<DragModel>(s,i,epsG,dragProperties);
            const double a=-expm1(-b*dt);
            // Every eligible parcel is overwritten before the second pass.
            // No sorting, migration or physical publication occurs in between.
            s.couplingDragBeta[i]=b;s.couplingResponseFraction[i]=a;
            const double w=s.pm[i]*a;
            weight+=w;x+=w*s.pux[i];y+=w*s.puy[i];z+=w*s.puz[i];
        }
    }
    weight=team.sum(weight);x=team.sum(x);y=team.sum(y);z=team.sum(z);
    if(dragActive)
    {
        const double ux=(gasMass*s.Ux[c]+x)/(gasMass+weight);
        const double uy=(gasMass*s.Uy[c]+y)/(gasMass+weight);
        const double uz=(gasMass*s.Uz[c]+z)/(gasMass+weight);
        double dpx=0,dpy=0,dpz=0,dEnergy=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!mixtureParticles::member(s,i,c))continue;
            const auto b=s.couplingDragBeta[i];
            const double a=s.couplingResponseFraction[i],oldX=s.pux[i],oldY=s.puy[i],oldZ=s.puz[i],oldTheta=s.pTheta[i];
            s.pux[i]=oldX+a*(ux-oldX);s.puy[i]=oldY+a*(uy-oldY);s.puz[i]=oldZ+a*(uz-oldZ);
            s.pTheta[i]=oldTheta*exp(-2*b*dt);
            const double dx=s.pux[i]-oldX,dy=s.puy[i]-oldY,dz=s.puz[i]-oldZ;
            dpx+=s.pm[i]*dx;dpy+=s.pm[i]*dy;dpz+=s.pm[i]*dz;
            // Actual represented differences, not independently recomputed work
            // or subtraction of two large mechanical energies.
            dEnergy+=s.pm[i]*(.5*(dx*(s.pux[i]+oldX)+dy*(s.puy[i]+oldY)+dz*(s.puz[i]+oldZ))
                              +1.5*(s.pTheta[i]-oldTheta));
        }
        dpx=team.sum(dpx);dpy=team.sum(dpy);dpz=team.sum(dpz);dEnergy=team.sum(dEnergy);
        if(team.rank()==0){
        s.rhoUx[c]-=dpx/volume;s.rhoUy[c]-=dpy/volume;s.rhoUz[c]-=dpz/volume;
        s.rhoE[c]-=dEnergy/volume;
        recoverGasPrimitiveCell(s,c);
        }team.sync();
    }
    if constexpr(ConstantHeat) {
    const double heatFactor=s.particleThermalRho*s.particleCp/s.rhoSolid;
    if(s.solveParticleTemperature!=0 && s.particleGasHeatTransferModelId!=0 && heatFactor>0.)
    {
        constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
        double species[Ns];for(int k=0;k<Ns;++k)species[k]=s.gasSpecies.rho[k*s.nCells+c];
        const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
        const double oldInternal=s.rhoE[c]-kinetic;
        const double referenceTemperature=s.Tgas[c];
        // Drag has published and recovered the gas state. Its viscosity
        // cache is no longer valid for this heat source epoch.
        mixtureParticles::HeatCellProperties heatProperties{};
        bool heatPropertiesReady=false;
        double A=0,B=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!mixtureParticles::member(s,i,c))continue;
            if(!heatPropertiesReady) {
                heatProperties=mixtureParticles::heatCellProperties(s,c);
                heatPropertiesReady=true;
            }
            const double a=mixtureParticles::heatFraction(s,i,dt,heatProperties);
            s.couplingResponseFraction[i]=a;
            const double w=s.pm[i]*heatFactor*a;
            A+=w;B+=w*(s.pT[i]-referenceTemperature);
        }
        A=team.sum(A);B=team.sum(B);
        if(A==0.)return;
        double target=referenceTemperature;
        auto status=ugkwp::ThermoStatus::Success;
        if(team.rank()==0)status=ugkwp::invertMixtureHeatExchange
        (species,oldInternal,A/volume,B/volume,s.gasSpecies.thermo,
         s.gasSpecies.thermoControls,target);
        target=team.broadcast(target);status=static_cast<ugkwp::ThermoStatus>(team.broadcast(int(status)));
        if(status!=ugkwp::ThermoStatus::Success)
        {
            if(team.rank()==0){s.gasSpecies.cellStatus[c]=static_cast<int>(status);
            // A failed physical exchange must reach this step's completion,
            // including the final sample, without waiting for another flux.
            using namespace ugkp::diagnostics;
            auto failure=makeRecord(Code::InvalidPhysicalState,30700,0,0,
                EntityKind::Cell,c,-1,static_cast<int>(status));
            failure.metadata=static_cast<unsigned>(status);
            recordFatal(s.diagnostics,failure);}
            return;
        }
        double heat=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!mixtureParticles::member(s,i,c))continue;
            const double oldT=s.pT[i],a=s.couplingResponseFraction[i];
            s.pT[i]=fmin(s.TpMax,fmax(s.TpMin,oldT+a*(target-oldT)));
            heat+=s.pm[i]*heatFactor*(s.pT[i]-oldT);
        }
        heat=team.sum(heat);
        if(team.rank()==0){s.rhoE[c]-=heat/volume;recoverGasPrimitiveCell(s,c);}
    }
    }
}

// Separate scalar and cooperative launches retain dense scalar work for light
// cells. Giving every light cell a full warp regresses sparse populations.
template<class DragModel,bool ConstantHeat=true,bool Cooperative=false,class State>
__global__ void mixtureParticleCouplingKernel
(State*sp,double requestedDt,bool dragActive,bool useCsr=true,bool useSplit=false,bool skipHeavy=false){
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
 State&s=*sp;const int thread=blockIdx.x*blockDim.x+threadIdx.x;
#if defined(__CUDACC__)
 if constexpr(Cooperative){
  for(int c=thread/32;c<s.nCells;c+=gridDim.x*(blockDim.x/32)){
   if(mixtureParticles::count(s,c,useCsr,useSplit)>=32)
    mixtureParticleCouplingCell<DragModel,ConstantHeat>(s,c,dt,dragActive,useCsr,useSplit,mixtureParticles::WarpCouplingTeam{});
  }
 }else
#endif
 {const int c=thread;if(c<s.nCells&&(!skipHeavy||mixtureParticles::count(s,c,useCsr,useSplit)<32))
   mixtureParticleCouplingCell<DragModel,ConstantHeat>(s,c,dt,dragActive,useCsr,useSplit,mixtureParticles::SerialCouplingTeam{});}
}
#if defined(__CUDACC__)
template<class DragModel,bool ConstantHeat=true,class State>
inline cudaError_t launchMixtureParticleCouplingCells(State*state,int cells,double dt,bool drag,bool csr,bool split,cudaStream_t stream=nullptr){
 if(cells<=0)return cudaSuccess;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat><<<(cells+127)/128,128,0,stream>>>(state,dt,drag,csr,split,true);
 auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 // One warp per cell; the device scheduler, not an SM-independent cap,
 // distributes the available cell work.
 const int groups=(cells+3)/4;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat,true><<<groups,128,0,stream>>>(state,dt,drag,csr,split);
 return cudaGetLastError();
}
#endif

// External pressure/gravity work is not internal drag exchange. Launch after
// the paired transaction; gas gravity has already been applied by its owner.
template<class State>
__global__ void mixtureParticleExternalKickKernel(State*sp,double requestedDt)
{
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
    State&s=*sp;const int count=mixtureParticles::activeCount(s);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    {
        if(s.pStatus[i]!=1 || !mixtureParticles::mobile(s,i))continue;
        const int c=s.pCellId[i];if(c<0 || c>=s.nCells)continue;
        s.pux[i]+=dt*(s.gravityX-s.gradPx[c]/s.rhoSolid);
        s.puy[i]+=dt*(s.gravityY-s.gradPy[c]/s.rhoSolid);
        s.puz[i]+=dt*(s.gravityZ-s.gradPz[c]/s.rhoSolid);
    }
}
