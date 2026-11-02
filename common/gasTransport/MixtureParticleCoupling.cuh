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

#include "ParticleCouplingReservoirViews.cuh"

template<class DragModel,bool ConstantHeat,class State,class Team,class Reservoir>
__device__ void mixtureParticleCouplingReservoir(State&s,int c,double dt,bool dragActive,bool useCsr,bool useSplit,const Team&team,Reservoir& reservoir)
{
    const int begin=0,end=mixtureParticles::count(s,c,useCsr,useSplit);
    const double epsG=s.epsGPrev[c],volume=reservoir.volume(s,c);
    const double gasMass=reservoir.mass(s,c);
    double weight=0,x=0,y=0,z=0;
    // No parcel or gas state is changed while forming the finite-bath target.
    for(int pos=begin+team.rank();pos<end;pos+=team.width())
    {
        const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
        if(!reservoir.member(s,i,c))continue;
        s.puxOld[i]=s.pux[i];s.puyOld[i]=s.puy[i];s.puzOld[i]=s.puz[i];
        if(dragActive)
        {
            const double a=-expm1(-reservoir.template beta<DragModel>(s,c,i,epsG)*dt);
            const double w=s.pm[i]*a;
            weight+=w;x+=w*s.pux[i];y+=w*s.puy[i];z+=w*s.puz[i];
        }
    }
    weight=team.sum(weight);x=team.sum(x);y=team.sum(y);z=team.sum(z);
    if(dragActive)
    {
        const double ux=(gasMass*reservoir.velocity(s,c,0)+x)/(gasMass+weight);
        const double uy=(gasMass*reservoir.velocity(s,c,1)+y)/(gasMass+weight);
        const double uz=(gasMass*reservoir.velocity(s,c,2)+z)/(gasMass+weight);
        double dpx=0,dpy=0,dpz=0,dEnergy=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!reservoir.member(s,i,c))continue;
            const double b=reservoir.template beta<DragModel>(s,c,i,epsG);
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
        const double momentumChange[3]={dpx,dpy,dpz};
        reservoir.apply(s,c,momentumChange,dEnergy);
        }team.sync();
    }
    if constexpr(ConstantHeat) {
    const double heatFactor=s.particleThermalRho*s.particleCp/s.rhoSolid;
    if(s.solveParticleTemperature!=0 && s.particleGasHeatTransferModelId!=0 && heatFactor>0.)
    {
        constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
        double species[Ns];for(int k=0;k<Ns;++k)species[k]=reservoir.species(s,c,k);
        const double kinetic=.5*(reservoir.momentum(s,c,0)*reservoir.momentum(s,c,0)+reservoir.momentum(s,c,1)*reservoir.momentum(s,c,1)+reservoir.momentum(s,c,2)*reservoir.momentum(s,c,2))/reservoir.density(s,c);
        const double oldInternal=reservoir.energy(s,c)-kinetic;
        const double referenceTemperature=reservoir.temperature(s,c);
        double A=0,B=0;
        for(int pos=begin+team.rank();pos<end;pos+=team.width())
        {
            const int i=mixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!reservoir.member(s,i,c))continue;
            const double w=s.pm[i]*heatFactor*reservoir.heatFraction(s,c,i,dt);
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
            if(!reservoir.member(s,i,c))continue;
            const double oldT=s.pT[i],a=reservoir.heatFraction(s,c,i,dt);
            s.pT[i]=fmin(s.TpMax,fmax(s.TpMin,oldT+a*(target-oldT)));
            heat+=s.pm[i]*heatFactor*(s.pT[i]-oldT);
        }
        heat=team.sum(heat);
        if(team.rank()==0){const double momentumChange[3]={};reservoir.apply(s,c,momentumChange,heat,true);}
    }
    }
}

#include "WallParticleReservoirCoupling.cuh"

template<class DragModel,bool ConstantHeat,class State,class Team>
__device__ void mixtureParticleCouplingCell(State&s,int c,double dt,bool dragActive,bool useCsr,bool useSplit,const Team&team){
 if constexpr(mixtureParticles::HasWallReservoir<State>::value){
  const int slot=mixtureParticles::wallReservoirSlot(s,c);
  if(slot>=0){mixtureParticleWallReservoirs<DragModel,ConstantHeat>(s,c,slot,dt,dragActive,useCsr,useSplit);return;}
 }
 mixtureParticles::MacroReservoir<State> reservoir;
 mixtureParticleCouplingReservoir<DragModel,ConstantHeat>(s,c,dt,dragActive,useCsr,useSplit,team,reservoir);
}

// Separate scalar and cooperative launches retain dense scalar work for light
// cells. Giving every light cell a full warp regresses sparse populations.
template<class DragModel,bool ConstantHeat,class State,class Team>
__device__ void mixtureParticleMacroReservoir(State&s,int c,double dt,bool drag,bool csr,bool split,const Team&team){
 mixtureParticles::MacroReservoir<State> reservoir;
 mixtureParticleCouplingReservoir<DragModel,ConstantHeat>(s,c,dt,drag,csr,split,team,reservoir);
}
template<class DragModel,bool ConstantHeat,class State>
__global__ void mixtureParticleWallCouplingKernel(State*sp,double dt,bool drag,bool csr,bool split){
 if constexpr(mixtureParticles::HasWallReservoir<State>::value){
  State&s=*sp;const int slot=blockIdx.x*blockDim.x+threadIdx.x;
  if(slot>=s.gasBoundaryLayer.count)return;
  const int c=s.faceOwner[s.gasBoundaryLayerModel.faces[slot]];
  // ownerSlot defines the existing wall reconstruction for this cell. Avoid
  // assigning the same macro inventory twice where two wall faces share it.
  if(mixtureParticles::wallReservoirSlot(s,c)!=slot)return;
  mixtureParticleWallReservoirs<DragModel,ConstantHeat>(s,c,slot,dt,drag,csr,split);
 }
}
template<class DragModel,bool ConstantHeat=true,bool Cooperative=false,class State>
__global__ void mixtureParticleCouplingKernel
(State*sp,double dt,bool dragActive,bool useCsr=true,bool useSplit=false,bool skipHeavy=false,bool spatialWalls=false){
 State&s=*sp;const int thread=blockIdx.x*blockDim.x+threadIdx.x;
#if defined(__CUDACC__)
 if constexpr(Cooperative){
  for(int c=thread/32;c<s.nCells;c+=gridDim.x*(blockDim.x/32)){
   if((!spatialWalls||mixtureParticles::wallReservoirSlot(s,c)<0)&&mixtureParticles::count(s,c,useCsr,useSplit)>=32)
    mixtureParticleMacroReservoir<DragModel,ConstantHeat>(s,c,dt,dragActive,useCsr,useSplit,mixtureParticles::WarpCouplingTeam{});
  }
 }else
#endif
 {const int c=thread;if(c<s.nCells&&(!spatialWalls||mixtureParticles::wallReservoirSlot(s,c)<0)&&(!skipHeavy||mixtureParticles::count(s,c,useCsr,useSplit)<32))
   mixtureParticleMacroReservoir<DragModel,ConstantHeat>(s,c,dt,dragActive,useCsr,useSplit,mixtureParticles::SerialCouplingTeam{});}
}
#if defined(__CUDACC__)
template<class DragModel,bool ConstantHeat=true,class State>
inline cudaError_t launchMixtureParticleCouplingCells(State*state,int cells,double dt,bool drag,bool csr,bool split,int walls=0){
 if(cells<=0)return cudaSuccess;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat><<<(cells+127)/128,128>>>(state,dt,drag,csr,split,true,walls>0);
 auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 const int groups=(cells+3)/4;
 mixtureParticleCouplingKernel<DragModel,ConstantHeat,true><<<groups<256?groups:256,128>>>(state,dt,drag,csr,split,false,walls>0);
 error=cudaGetLastError();if(error!=cudaSuccess)return error;
 if(walls>0)mixtureParticleWallCouplingKernel<DragModel,ConstantHeat><<<(walls+63)/64,64>>>(state,dt,drag,csr,split);
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
