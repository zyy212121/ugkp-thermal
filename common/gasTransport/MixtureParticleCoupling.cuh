#pragma once
#include <type_traits>
#include <utility>
#include "MixtureHeatExchange.H"
#include "MixtureParticleDirectory.cuh"
#include "MixtureCouplingTeam.cuh"
#include "../gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "../gasNumerics/GpuDragModels.cuh"

// Physical-time source transaction shared by gasUGKP and gasUGKPSteady. One cell worker
// owns all parcel writes and the reverse gas publication for its cell. CSR and
// split direct-base + indexed-injection ranges consume the same post-collision
// epoch. A full scan remains available only for the serial reference oracle.
namespace mixtureParticles
{
template<class DragModel,class State>
__device__ double beta(const State&s,int c,int i,double epsG)
{
    const double dx=s.Ux[c]-s.pux[i],dy=s.Uy[c]-s.puy[i],dz=s.Uz[c]-s.puz[i];
    const ugkwpGpuDrag::DragInput input={s.rho[c],
        ugkwp::gasCellMolecularViscosityAt(s,c,s.Tgas[c]),epsG,s.rhoSolid,
        s.pd[i],sqrt(dx*dx+dy*dy+dz*dz),1e-300,
        s.dragParameter0,s.dragParameter1,s.dragParameter2,s.dragParameter3};
    return fmax(0.,DragModel::inverseResponseTime(input));
}

template<class State>
__device__ double heatFraction(const State&s,int c,int i,double dt)
{
    const double dx=s.Ux[c]-s.pux[i],dy=s.Uy[c]-s.puy[i],dz=s.Uz[c]-s.puz[i];
    const double mu=ugkwp::gasCellMolecularViscosityAt(s,c,s.Tgas[c]);
    const double re=s.rho[c]*s.pd[i]*sqrt(dx*dx+dy*dy+dz*dz)/fmax(mu,1e-30);
    const double nu=ugkwp::ranzMarshallNuFromPrOneThird(fmax(0.,re),ugkwp::gasCellPrandtlOneThird(s,c));
    const double rate=6*nu*molecularGasConductivity(s,c,s.Tgas[c])
        /(s.particleThermalCapacity*s.pd[i]*s.pd[i]+1e-300);
    return -expm1(-fmax(0.,rate)*dt);
}
}

template<class DragModel,bool ConstantHeat,class State,class Team>
__device__ void mixtureParticleCouplingCell(State&s,int c,double dt,bool dragActive,bool useCsr,bool useSplit,const Team&team)
{
    const int begin=0,end=mixtureParticles::count(s,c,useCsr,useSplit);
    const double epsG=s.epsGPrev[c],volume=epsG*s.V[c];
    const double gasMass=volume*s.rho[c];
    double weight=0,x=0,y=0,z=0;
    // No parcel or gas state is changed while forming the finite-bath target.
    for(int pos=begin+team.rank();pos<end;pos+=team.width())
    {
        const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
        if(!mixtureParticles::member(s,i,c))continue;
        s.puxOld[i]=s.pux[i];s.puyOld[i]=s.puy[i];s.puzOld[i]=s.puz[i];
        if(dragActive)
        {
            const double a=-expm1(-mixtureParticles::beta<DragModel>(s,c,i,epsG)*dt);
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
            const double b=mixtureParticles::beta<DragModel>(s,c,i,epsG);
            const double a=-expm1(-b*dt),oldX=s.pux[i],oldY=s.puy[i],oldZ=s.puz[i],oldTheta=s.pTheta[i];
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
        double A=0,B=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!mixtureParticles::member(s,i,c))continue;
            const double w=s.pm[i]*heatFactor*mixtureParticles::heatFraction(s,c,i,dt);
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
            const double oldT=s.pT[i],a=mixtureParticles::heatFraction(s,c,i,dt);
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
(State*sp,double dt,bool dragActive,bool useCsr=true,bool useSplit=false,bool skipHeavy=false){
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
inline cudaError_t launchMixtureParticleCouplingCells(State*state,int cells,double dt,bool drag,bool csr,bool split){
 if(cells<=0)return cudaSuccess;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat><<<(cells+127)/128,128>>>(state,dt,drag,csr,split,true);
 auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 const int groups=(cells+3)/4;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat,true><<<groups<256?groups:256,128>>>(state,dt,drag,csr,split);
 return cudaGetLastError();
}
#endif

// External pressure/gravity work is not internal drag exchange. Launch after
// the paired transaction; gas gravity has already been applied by its owner.
template<class State>
__global__ void mixtureParticleExternalKickKernel(State*sp,double dt)
{
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
