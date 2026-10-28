// Included after the gas reference adapter. CPU-only serial launch/atomic adapters.
#include "GpuParticleBoundaryModel.H"
struct ParticleState : DeviceState {
 int particleCapacity=0,particleCount=0,*particleCountDevice=&particleCount,nBoundarySources=1,maxFaceWalkHops=1000;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1,dragModel=0;
 double rhoSolid=1000,invRhoSolid=.001,epsSMin=1e-12,particleThermalRho=1000,particleCp=1000,particleThermalCapacity=1e6;
 double particleDiameterFallback=.001,particleDiameterMin=.001,particleDiameterMax=.001,particleDiameterSigma=0;
 double TpMin=1,TpMax=10000,gasPrOneThird=0,dragParameter0=0,dragParameter1=0,dragParameter2=0,dragParameter3=0,injectionParcelMass=1e-5;
 double referenceMaxCollisionProbability=0,thetaMin=1e-12;
 int*poolThermalCount=nullptr,*poissonPoolSampleTargetCount=nullptr;
 double *poolThermalSumUx=nullptr,*poolThermalSumUy=nullptr,*poolThermalSumUz=nullptr,*poolThermalSumU2=nullptr,*poissonPoolMass=nullptr,*poissonPoolMomX=nullptr,*poissonPoolMomY=nullptr,*poissonPoolMomZ=nullptr,*poissonPoolEnergy=nullptr,*poissonPoolDiameter=nullptr,*poissonPoolDiameter2=nullptr;
 unsigned long long rngSeed=12345;
 double *px=nullptr,*py=nullptr,*pz=nullptr,*pux=nullptr,*puy=nullptr,*puz=nullptr,*puxOld=nullptr,*puyOld=nullptr,*puzOld=nullptr,*pT=nullptr,*pTheta=nullptr,*pd=nullptr,*pm=nullptr;
 unsigned long long *pRng=nullptr,*pOrigId=nullptr;
 // Particle transport sees the full unit-area box; gas base retains its axial FV graph.
 int *cellPlaneStart=nullptr,*cellPlaneCount=nullptr,*cellFaceId=nullptr;
 int *pStatus=nullptr,*pCellId=nullptr,*cellParticleCount=nullptr,*cellFaceKind=nullptr,*cellFaceNeighbor=nullptr;
 double *planeNx=nullptr,*planeNy=nullptr,*planeNz=nullptr,*planeD=nullptr,*cellFaceRestitution=nullptr;
 int *cellParticleOffset=nullptr,*sortedParticleIndex=nullptr;
 double *epsG=nullptr,*epsS=nullptr,*rhoUsx=nullptr,*rhoUsy=nullptr,*rhoUsz=nullptr,*rhoEs=nullptr,*rhoHp=nullptr,*rhoDs=nullptr,*Usx=nullptr,*Usy=nullptr,*Usz=nullptr,*Tp=nullptr,*dMeanCell=nullptr;
 double *epsGPrev=nullptr;
 double *momRhoP=nullptr,*momRhoUPx=nullptr,*momRhoUPy=nullptr,*momRhoUPz=nullptr,*momRhoEP=nullptr,*momRhoPD=nullptr,*momRhoHpP=nullptr,*thetaDragAlpha=nullptr;
 double *couplingRhoOld=nullptr,*couplingUxOld=nullptr,*couplingUyOld=nullptr,*couplingUzOld=nullptr,*couplingTgasOld=nullptr,*theta=nullptr;
 double* volumeFractionScheduleTimes=nullptr;double* volumeFractionScheduleValues=nullptr;int nVolumeFractionScheduleRows=0;
 int sourceFace[1]={},sourceCell[1]={};
 double sourceUx[1]={10},sourceUy[1]={},sourceUz[1]={},sourceTheta[1]={},sourceT[1]={300},sourceD[1]={.001},sourceMassRate[1]={.01},sourcePx[1]={1e-10},sourcePy[1]={},sourcePz[1]={},sourceResidualMass[1]={};
 // Diagnostic residence age is not a physical state input to production operators.
 double *referenceAge=nullptr; unsigned long long injected=0,outflow=0;
 double injectedMass=0,injectedMomentumX=0,injectedMechanicalEnergy=0,injectedMaterialHeat=0;
 double outflowMass=0,outflowMomentumX=0,outflowMechanicalEnergy=0,outflowMaterialHeat=0;
 double gasDragMomentumX=0,gasDragEnergy=0,gasMaterialHeat=0,gasPairedMomentumX=0,gasPairedEnergy=0;
};
#define DeviceState ParticleState
#define GPU_OPERATOR_THERMAL 0
#define GPU_POOL_INITIALIZATION_WITH_PROBABILITY 0
#define GPU_POOL_MASS_FALLBACK GPU_OPERATOR_R(0.0)
#define GPU_POOL_PARTICLE_THETA(s,i) clampMin(finiteOr((s).pTheta[i],0.),0.)
#define GPU_RESET_CONTACT_AGE(s,i)
#define GPU_PERIODIC_FACE_KIND 5
#define GPU_PERIODIC_COORDINATE(hit,shift,eps,normal) ((hit)+(shift)-(eps)*(normal))
#define GPU_WALL_COORDINATE(hit,eps,normal) ((hit)-(eps)*(normal))
ThreadIndex gridDim{1};
template<class T> T atomicAdd(T*ptr,T value){T old=*ptr;*ptr+=value;return old;}
template<class T> T atomicSub(T*ptr,T value){T old=*ptr;*ptr-=value;return old;}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
bool gasDragModelActive(int model){return model!=2;}
// Required only by uninstantiated thermal-moment templates (gas path is nonthermal).
double particleSpecificEnthalpyDevice(double t){return 1000*t;}
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "gpu/GpuCouplingMath.H"
#include "GpuDragModels.cuh"
/* PRODUCTION_HEAT_FACTOR */
#include "operators/solidEpsFromMomentDevice.cuh"
#include "operators/snapshotParticleGasCouplingStateKernel.cuh"
#include "operators/uniform01Device.cuh"
#include "operators/normalDevice.cuh"
#include "operators/scheduledSolidVolumeFractionDevice.cuh"
constexpr double OfPi=3.14159265358979323846;
#include "operators/granularCollisionTauFromCellDevice.cuh"
#include "GpuCollisionProbability.cuh"
#include "GpuContactAgeStorage.cuh"
#include "operators/injectBoundaryParticlesKernel.cuh"
#include "operators/pointInsideCell.cuh"
#include "GpuParticleTransport.cuh"
#include "operators/trackParticlesLocalFaceWalkKernel.cuh"
#include "operators/computePressureGradientKernel.cuh"
#include "applications/gasUGKPSteady/private_backend/SteadyMixturePorosity.cuh"
#define GPU_GAS_POROSITY_SPECIES(s,c,scale) steadyScalePorositySpecies(s,c,scale)
#include "operators/applyGasVolumeFractionSourceKernel.cuh"
#undef GPU_GAS_POROSITY_SPECIES
#include "GpuCellDragRate.cuh"
template<class DragModel> void referenceEulerianDrag(ParticleState*sp,double dt){const DragModel dragModel{};
#include "GpuCellDragUpdate.inl"
}
#include "operators/applyEulerianParticleMaterialHeatKernel.cuh"
/* PRODUCTION_RELAXATION */
#include "applications/gasUGKPSteady/private_backend/SteadyMixtureParticleCoupling.cuh"
#include "operators/clearPoissonThermalPoolKernel.cuh"
#include "GpuCollisionPoolAtomic.cuh"
#include "operators/preparePoissonPoolSamplingKernel.cuh"
#include "GpuCollisionPoolSampling.cuh"
#include "operators/samplePoissonPoolParticlesKernel.cuh"
#include "GpuCollisionPoolCorrection.cuh"
#include "operators/clearParticleMomentsAndCountsAtomicKernel.cuh"
#include "operators/accumulateParticleMomentsAtomicKernel.cuh"
#include "operators/normalizeParticleMomentsAtomicKernel.cuh"
#include "GpuParticleFields.cuh"
#include "operators/clearSolidCell.cuh"
#define GPU_RECOVERY_INLINE inline
#include "GpuMechanicalRecovery.cuh"
#undef GPU_RECOVERY_INLINE
#undef DeviceState
void initialiseParticles(ParticleState&s,int capacity){
 s.particleCapacity=capacity;s.gasPrOneThird=std::cbrt(s.gasPrClamped);
#define PA(name) s.name=new double[capacity]{};
 PA(px) PA(py) PA(pz) PA(pux) PA(puy) PA(puz) PA(puxOld) PA(puyOld) PA(puzOld) PA(pT) PA(pTheta) PA(pd) PA(pm) PA(referenceAge)
#undef PA
 s.pRng=new unsigned long long[capacity]{};s.pOrigId=new unsigned long long[capacity]{};s.pStatus=new int[capacity]{};s.pCellId=new int[capacity]{};
#define CA(name) s.name=new double[s.nCells+1]{};
 CA(poolThermalSumUx) CA(poolThermalSumUy) CA(poolThermalSumUz) CA(poolThermalSumU2) CA(poissonPoolMass) CA(poissonPoolMomX) CA(poissonPoolMomY) CA(poissonPoolMomZ) CA(poissonPoolEnergy) CA(poissonPoolDiameter) CA(poissonPoolDiameter2)
 CA(epsG) CA(epsS) CA(rhoUsx) CA(rhoUsy) CA(rhoUsz) CA(rhoEs) CA(rhoHp) CA(rhoDs) CA(Usx) CA(Usy) CA(Usz) CA(Tp) CA(dMeanCell)
 CA(epsGPrev) CA(momRhoP) CA(momRhoUPx) CA(momRhoUPy) CA(momRhoUPz) CA(momRhoEP) CA(momRhoPD) CA(momRhoHpP) CA(thetaDragAlpha) CA(couplingRhoOld) CA(couplingUxOld) CA(couplingUyOld) CA(couplingUzOld) CA(couplingTgasOld) CA(theta)
#undef CA
 s.poolThermalCount=new int[s.nCells]{};s.poissonPoolSampleTargetCount=new int[s.nCells]{};
 s.cellParticleCount=new int[s.nCells+1]{};for(int c=0;c<s.nCells;++c){s.epsGPrev[c]=s.epsG[c]=1;clearSolidCell(s,c);}
 s.cellPlaneStart=new int[s.nCells]{};s.cellPlaneCount=new int[s.nCells]{};s.cellFaceId=new int[6*s.nCells]{};
 s.planeNx=new double[6*s.nCells]{};s.planeNy=new double[6*s.nCells]{};s.planeNz=new double[6*s.nCells]{};s.planeD=new double[6*s.nCells]{};
 s.cellFaceKind=new int[6*s.nCells]{};s.cellFaceNeighbor=new int[6*s.nCells]{};s.cellFaceRestitution=new double[6*s.nCells]{};
 // Match GpuResidentStrict.H: escape=4, periodic=5. Pressure-gradient
 // boundary selection consumes these same kinds, not only particle tracking.
 // Unit-area box: axial open boundaries, empty side patches use geometric reflection.
 const DeviceState& gas=s;
 for(int c=0;c<s.nCells;++c){
  s.cellPlaneStart[c]=6*c;s.cellPlaneCount[c]=6;
  for(int j=0;j<6;++j){const int p=6*c+j;s.cellFaceId[p]=-1;s.cellFaceNeighbor[p]=-1;s.cellFaceRestitution[p]=Foam::gpuParticleBoundary::geometricReflectionRestitution;
   if(j<2){const int f=gas.cellFaceId[gas.cellPlaneStart[c]+j];s.cellFaceId[p]=f;const bool left=s.faceCx[f]<s.Cx[c];s.planeNx[p]=left?-1:1;s.planeD[p]=s.planeNx[p]*s.faceCx[f];const int next=left?c-1:c+1;s.cellFaceNeighbor[p]=next;s.cellFaceKind[p]=(next<0||next>=s.nCells)?4:0;}
   else {s.cellFaceKind[p]=1;const bool positive=j%2!=0;s.planeD[p]=positive?1:0;if(j<4)s.planeNy[p]=positive?1:-1;else s.planeNz[p]=positive?1:-1;}
  }
 }
 for(int f=s.nInternalFaces;f<s.nFaces;++f)if(s.Sfx[f]<0){s.sourceFace[0]=f;s.sourceCell[0]=s.faceOwner[f];s.sourcePx[0]=s.faceCx[f]+1e-6*std::cbrt(s.V[s.sourceCell[0]]);}
 s.sourcePy[0]=s.sourcePz[0]=.5;
}
void particleMoments(ParticleState&s){
 for(int c=0;c<=s.nCells;++c){threadIdx.x=c;clearParticleMomentsAndCountsAtomicKernel(&s);}
 threadIdx.x=0;accumulateParticleMomentsAtomicKernel(&s);
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;normalizeParticleMomentsAtomicKernel(&s);solidRecoveryFromParticleMomentsCell(s,c);s.epsG[c]=1-s.epsS[c];}
}
// Called AFTER the production gas flux step; production atomic collision path.
// Collisional-pressure kicks and mobile-packing projection are configured off.
void particleStep(ParticleState&s,double dt,double frozenBoundaryTime){
 const bool mixture=ugkwp::mixtureGasActive(s);
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;applyGasVolumeFractionSourceKernel(&s,dt);recoverGasPrimitivesKernel(&s);}
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;computePressureGradientKernel(&s);}
 if(!mixture)for(int c=0;c<s.nCells;++c){threadIdx.x=c;
 const double weight=s.V[c]*(1-solidEpsFromMomentDevice(s,c)),mx=s.rhoUx[c],en=s.rhoE[c];
 referenceEulerianDrag<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,dt);
 s.gasDragMomentumX+=weight*(s.rhoUx[c]-mx);s.gasDragEnergy+=weight*(s.rhoE[c]-en);
 recoverGasPrimitivesKernel(&s);const double heatBefore=s.rhoE[c];
 applyEulerianParticleMaterialHeatKernel(&s,dt);s.gasMaterialHeat+=weight*(s.rhoE[c]-heatBefore);recoverGasPrimitivesKernel(&s);
 }
 if(mixture)for(int c=0;c<s.nCells;++c){threadIdx.x=c;snapshotParticleGasCouplingStateKernel(&s);s.thetaDragAlpha[c]=1;}
 const int before=s.particleCount;threadIdx.x=0;injectBoundaryParticlesKernel(&s,dt,frozenBoundaryTime);s.injected+=s.particleCount-before;for(int i=before;i<s.particleCount;++i){s.referenceAge[i]=0;s.injectedMass+=s.pm[i];s.injectedMomentumX+=s.pm[i]*s.pux[i];s.injectedMechanicalEnergy+=s.pm[i]*(.5*sqr3(s.pux[i],s.puy[i],s.puz[i])+1.5*s.pTheta[i]);s.injectedMaterialHeat+=s.pm[i]*particleHeatFactorDevice(s)*s.pT[i];}
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;clearPoissonThermalPoolKernel(&s);s.referenceMaxCollisionProbability=std::max(s.referenceMaxCollisionProbability,poissonCollisionProbabilityForCell(s,c,dt));}
 threadIdx.x=0;accumulateParticlePoolAtomicKernel<true>(&s,dt);
 for(int c=0;c<s.nCells;++c){threadIdx.x=c;preparePoissonPoolSamplingKernel(&s);}
 threadIdx.x=0;samplePoissonPoolParticlesKernel(&s,0);
 for(int i=0;i<s.particleCount;++i)correctOnePoissonThermalizedParticle(s,i,false);
 if(mixture){
  for(int c=0;c<s.nCells;++c){threadIdx.x=c;const double mx=s.rhoUx[c],energy=s.rhoE[c],volume=s.V[c]*s.epsGPrev[c];
   steadyMixtureParticleCouplingKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,dt,true,false);
   s.gasPairedMomentumX+=volume*(s.rhoUx[c]-mx);s.gasPairedEnergy+=volume*(s.rhoE[c]-energy);
  }
  threadIdx.x=0;steadyParticleExternalKickKernel(&s,dt);
 }else{threadIdx.x=0;relaxParticlesToResidentGasKernel<ugkwpGpuDrag::SchillerNaumannDeviceDrag>(&s,dt);}
 for(int i=0;i<s.particleCount;++i){const bool active=s.pStatus[i]!=0;trackOneParticleLocalFaceWalk(s,i,dt);if(active){s.referenceAge[i]+=dt;if(!s.pStatus[i]){++s.outflow;s.outflowMass+=s.pm[i];s.outflowMomentumX+=s.pm[i]*s.pux[i];s.outflowMechanicalEnergy+=s.pm[i]*(.5*sqr3(s.pux[i],s.puy[i],s.puz[i])+1.5*s.pTheta[i]);s.outflowMaterialHeat+=s.pm[i]*particleHeatFactorDevice(s)*s.pT[i];}}}
 particleMoments(s);
 int live=0;
 for(int i=0;i<s.particleCount;++i)if(s.pStatus[i]==1){
 const int c=s.pCellId[i];
#define COPY_PRIMARY(src,dst,value) s.src[live]=value;
 GPU_PARTICLE_FIELDS_PRIMARY_BEFORE_CONTACT(COPY_PRIMARY)
 GPU_PARTICLE_FIELDS_PRIMARY_AFTER_CONTACT(COPY_PRIMARY)
 GPU_PARTICLE_FIELDS_IDENTITY(COPY_PRIMARY)
#undef COPY_PRIMARY
 s.puxOld[live]=s.puxOld[i];s.puyOld[live]=s.puyOld[i];s.puzOld[live]=s.puzOld[i];s.referenceAge[live]=s.referenceAge[i];
 ++live;
 }
 for(int i=live;i<s.particleCount;++i)s.pStatus[i]=0;
 s.particleCount=live;
}
