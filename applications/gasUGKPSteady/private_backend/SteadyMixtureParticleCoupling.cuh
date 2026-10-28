#pragma once
#include <type_traits>
#include <utility>
#include "../../../common/gasTransport/MixtureHeatExchange.H"
#include "../../../common/gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "../../gasUGKP/private_backend/GpuDragModels.cuh"

// New steady application's physical-time source transaction. One cell worker
// owns all parcel writes and the reverse gas publication for its cell. CSR and
// split direct-base + indexed-injection ranges consume the same post-collision
// epoch. A full scan remains available only for the serial reference oracle.
namespace steadyMixtureParticles
{
template<class State>
__device__ int activeCount(const State& s)
{return *s.particleCountDevice<0?0:(*s.particleCountDevice>s.particleCapacity?s.particleCapacity:*s.particleCountDevice);}
template<class State>
__device__ bool member(const State& s,int i,int c)
{return i>=0 && i<activeCount(s) && s.pStatus[i]==1 && s.pCellId[i]==c;}
template<class State,class=void>struct HasSplitDirectory:std::false_type{};
template<class State>struct HasSplitDirectory<State,std::void_t<
    decltype(std::declval<const State&>().preBaseCellOffset)>>:std::true_type{};

template<class State>
__device__ int count(const State&s,int c,bool useCsr,bool useSplit)
{
    if constexpr(HasSplitDirectory<State>::value)
        if(useSplit)return s.preBaseCellOffset[c+1]-s.preBaseCellOffset[c]
            +s.cellParticleOffset[c+1]-s.cellParticleOffset[c];
    return useCsr?s.cellParticleOffset[c+1]-s.cellParticleOffset[c]:activeCount(s);
}
template<class State>
__device__ int parcel(const State&s,int c,int position,bool useCsr,bool useSplit)
{
    if constexpr(HasSplitDirectory<State>::value)
        if(useSplit)
        {
            const int baseCount=s.preBaseCellOffset[c+1]-s.preBaseCellOffset[c];
            return position<baseCount?s.preBaseCellOffset[c]+position
                :s.sortedParticleIndex[s.cellParticleOffset[c]+position-baseCount];
        }
    return useCsr?s.sortedParticleIndex[s.cellParticleOffset[c]+position]:position;
}

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
    const double nu=ugkwp::ranzMarshallNuFromPrOneThird(fmax(0.,re),s.gasPrOneThird);
    const double rate=6*nu*molecularGasConductivity(s,c,s.Tgas[c])
        /(s.particleThermalCapacity*s.pd[i]*s.pd[i]+1e-300);
    return -expm1(-fmax(0.,rate)*dt);
}
}

template<class DragModel,class State>
__global__ void steadyMixtureParticleCouplingKernel
(State*sp,double dt,bool dragActive,bool useCsr=true,bool useSplit=false)
{
    State&s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    const int begin=0,end=steadyMixtureParticles::count(s,c,useCsr,useSplit);
    const double epsG=s.epsGPrev[c],volume=epsG*s.V[c];
    const double gasMass=volume*s.rho[c];
    const double heatFactor=s.particleThermalRho*s.particleCp/s.rhoSolid;
    double weight=0,x=0,y=0,z=0;
    // No parcel or gas state is changed while forming the finite-bath target.
    for(int pos=begin;pos<end;++pos)
    {
        const int i=steadyMixtureParticles::parcel(s,c,pos,useCsr,useSplit);
        if(!steadyMixtureParticles::member(s,i,c))continue;
        s.puxOld[i]=s.pux[i];s.puyOld[i]=s.puy[i];s.puzOld[i]=s.puz[i];
        if(dragActive)
        {
            const double a=-expm1(-steadyMixtureParticles::beta<DragModel>(s,c,i,epsG)*dt);
            const double w=s.pm[i]*a;
            weight+=w;x+=w*s.pux[i];y+=w*s.puy[i];z+=w*s.puz[i];
        }
    }
    if(dragActive)
    {
        const double ux=(gasMass*s.Ux[c]+x)/(gasMass+weight);
        const double uy=(gasMass*s.Uy[c]+y)/(gasMass+weight);
        const double uz=(gasMass*s.Uz[c]+z)/(gasMass+weight);
        double dpx=0,dpy=0,dpz=0,dEnergy=0;
        for(int pos=begin;pos<end;++pos)
        {
            const int i=steadyMixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!steadyMixtureParticles::member(s,i,c))continue;
            const double b=steadyMixtureParticles::beta<DragModel>(s,c,i,epsG);
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
        s.rhoUx[c]-=dpx/volume;s.rhoUy[c]-=dpy/volume;s.rhoUz[c]-=dpz/volume;
        s.rhoE[c]-=dEnergy/volume;
        recoverGasPrimitiveCell(s,c);
    }
    if(s.solveParticleTemperature!=0 && s.particleGasHeatTransferModelId!=0 && heatFactor>0.)
    {
        constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
        double species[Ns];for(int k=0;k<Ns;++k)species[k]=s.gasSpecies.rho[k*s.nCells+c];
        const double kinetic=.5*(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/s.rho[c];
        const double oldInternal=s.rhoE[c]-kinetic;
        const double referenceTemperature=s.Tgas[c];
        double A=0,B=0;
        for(int pos=begin;pos<end;++pos)
        {
            const int i=steadyMixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!steadyMixtureParticles::member(s,i,c))continue;
            const double w=s.pm[i]*heatFactor*steadyMixtureParticles::heatFraction(s,c,i,dt);
            A+=w;B+=w*(s.pT[i]-referenceTemperature);
        }
        if(A==0.)return;
        double target=referenceTemperature;
        const auto status=ugkwp::invertMixtureHeatExchange
        (species,oldInternal,A/volume,B/volume,s.gasSpecies.thermo,
         s.gasSpecies.thermoControls,target);
        if(status!=ugkwp::ThermoStatus::Success)
        {
            s.gasSpecies.cellStatus[c]=static_cast<int>(status);
            return;
        }
        double heat=0;
        for(int pos=begin;pos<end;++pos)
        {
            const int i=steadyMixtureParticles::parcel(s,c,pos,useCsr,useSplit);
            if(!steadyMixtureParticles::member(s,i,c))continue;
            const double oldT=s.pT[i],a=steadyMixtureParticles::heatFraction(s,c,i,dt);
            s.pT[i]=fmin(s.TpMax,fmax(s.TpMin,oldT+a*(target-oldT)));
            heat+=s.pm[i]*heatFactor*(s.pT[i]-oldT);
        }
        s.rhoE[c]-=heat/volume;
        recoverGasPrimitiveCell(s,c);
    }
}

// External pressure/gravity work is not internal drag exchange. Launch after
// the paired transaction; gas gravity has already been applied by its owner.
template<class State>
__global__ void steadyParticleExternalKickKernel(State*sp,double dt)
{
    State&s=*sp;const int count=steadyMixtureParticles::activeCount(s);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    {
        if(s.pStatus[i]!=1)continue;
        const int c=s.pCellId[i];if(c<0 || c>=s.nCells)continue;
        s.pux[i]+=dt*(s.gravityX-s.gradPx[c]/s.rhoSolid);
        s.puy[i]+=dt*(s.gravityY-s.gradPy[c]/s.rhoSolid);
        s.puz[i]+=dt*(s.gravityZ-s.gradPz[c]/s.rhoSolid);
    }
}
