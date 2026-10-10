// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuLesAlgebra.cuh"
using Real=GpuReal;
using Time=double;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME Time
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_R(1e-30)
#define __host__
#define __global__
#define __device__
#define __forceinline__ inline
// PTX traps are represented by process failure in this serial host emulator.
#define asm(...) std::abort()
struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;
constexpr Real OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr Real OfVSmall=GPU_R(1e-30);
constexpr Real OfGreat=GPU_R(1.0)/OfSmall;
struct GasPrimDevice { Real rho,ux,uy,uz,p,T; };
#include "legacy_gas_fixture.hpp"
#ifdef LEGACY_REFERENCE
using DeviceState=LegacyGasFixture;
#else
// No application DeviceState definition: the actual operators must be generic.
#include "gasTransport/GasStateView.H"
#endif
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
#include "operators/computeSstGradientsKernel.cuh"
#include "operators/sstVelocityInvariants.cuh"
#include "operators/computeGasHllcAdcSensorKernel.cuh"
#include "operators/updateBarthLimiter.cuh"
#include "operators/computeGasGradientLimiterKernel.cuh"
#include "operators/computeGasEddyViscosityKernel.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "operators/gasFaceSubgridTransportProperties.cuh"
#include "operators/computeRiemannGasFaceFluxDevice.cuh"
#include "operators/computeGasInternalFaceFluxKernel.cuh"
#include "operators/computeGasFluxPositivityScaleKernel.cuh"
#include "operators/computeSstFaceFluxKernel.cuh"

#ifdef LEGACY_OPTIONAL_SPECIES
struct TestLegacyGasFixture:LegacyGasFixture{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
#else
using TestLegacyGasFixture=LegacyGasFixture;
#endif
void hashBytes(std::uint64_t&h,const void*p,std::size_t n){auto b=static_cast<const unsigned char*>(p);for(std::size_t j=0;j<n;++j){h^=b[j];h*=1099511628211ull;}}
template<class State>void run(State&s,int scheme,int reconstruction,int limiter,int turbulence){
 s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gammaGas=Real(1.4);s.Rgas=287;s.gasCp=1004.5;s.gasMu=1.8e-5;s.gasPrClamped=.71;s.rhoMin=1e-10;s.TgasMin=1e-6;
 s.gasFluxScheme=scheme;s.gasReconstruction=reconstruction;s.gasLimiter=limiter;s.turbulenceModel=turbulence;s.sstConfigured=turbulence==3;
 s.sstCoefficients=ugkwp::defaultSstCoefficients();s.sstKMin=1e-12;s.sstOmegaMin=1e-6;s.sstMaxSourceNumber=.25;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;s.maxDiffusionNumber=.25;s.turbulentPrandtl=.9;s.lesDeltaCoeff=1;s.waleCw=.325;s.smagorinskyCs=.17;
 for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.cellLength[c]=1;s.rho[c]=1+.125*c;s.Ux[c]=.25-.125*c;s.Uy[c]=.02*c;s.Uz[c]=.03;s.Tgas[c]=300+2*c;s.p[c]=s.rho[c]*s.Rgas*s.Tgas[c];s.rhoUx[c]=s.rho[c]*s.Ux[c];s.rhoUy[c]=s.rho[c]*s.Uy[c];s.rhoUz[c]=s.rho[c]*s.Uz[c];s.rhoE[c]=s.p[c]/(s.gammaGas-1)+.5*s.rho[c]*(s.Ux[c]*s.Ux[c]+s.Uy[c]*s.Uy[c]+s.Uz[c]*s.Uz[c]);s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;s.sstWallDistance[c]=.5;s.k[c]=.1+.01*c;s.omega[c]=3+.1*c;}
 s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
 for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.riemannBoundaryKind[f]=s.gasBoundaryKind[f]=f==0?0:1;}
 blockDim.x=1;blockIdx.x=0;
 for(int c=0;c<2;++c){threadIdx.x=c;initialiseSstConservativeStateKernel(&s);saveGasConservativeStateKernel(&s);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);}
 for(int f=0;f<3;++f){threadIdx.x=f;updateLegacyGasBoundaryMirrorKernel(&s,0);updateRiemannBoundaryMirrorKernel(&s);updateWaveTransmissivePressureBoundaryKernel(&s,1e-6);}
 for(int c=0;c<2;++c){threadIdx.x=c;applySstWallFunctionStateKernel(&s);computeGasHllcAdcSensorKernel(&s);computeGasPrimitiveGradientsKernel(&s);computeSstGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}
 for(int f=0;f<3;++f){threadIdx.x=f;if(turbulence)computeGasInternalFaceFluxKernel<true>(&s,1e-6);else computeGasInternalFaceFluxKernel<false>(&s,1e-6);enforcePeriodicGasFluxAntisymmetryKernel(&s);}
 for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
 for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);computeSstFaceFluxKernel(&s);enforcePeriodicSstFluxAntisymmetryKernel(&s);}
 for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,1e-6);applyGasFluxDivergenceByCellKernel(&s,1e-6);blendGasConservativeStateKernel(&s,Real(.75),Real(.25));recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);computeGasDiffusionNumberKernel(&s,1e-6,.5);computeSstStabilityNumberKernel(&s,1e-6,.5);}
 for(int f=0;f<3;++f){threadIdx.x=f;computeGasCourantFieldKernel(&s,1e-6);}
 for(int c=0;c<2;++c){threadIdx.x=c;computeGasConvectiveCourantByCellKernel(&s,1e-6);if(!(s.rho[c]>0&&s.Tgas[c]>0&&std::isfinite(s.rhoE[c])))std::abort();}
}
int main(){
 for(int scheme=1;scheme<=9;++scheme)for(int reconstruction=0;reconstruction<3;++reconstruction)for(int limiter=0;limiter<3;++limiter)for(int turbulence=0;turbulence<4;++turbulence){
 TestLegacyGasFixture s{};
    s.Cx=new Real[8]{};
    s.Cy=new Real[8]{};
    s.Cz=new Real[8]{};
    s.Sfx=new Real[8]{};
    s.Sfy=new Real[8]{};
    s.Sfz=new Real[8]{};
    s.Tgas=new Real[8]{};
    s.Ux=new Real[8]{};
    s.Uy=new Real[8]{};
    s.Uz=new Real[8]{};
    s.V=new Real[8]{};
    s.cellFaceId=new int[8]{};
    s.cellLength=new Real[8]{};
    s.cellPlaneCount=new int[8]{};
    s.cellPlaneStart=new int[8]{};
    s.deltaCoeffs=new Real[8]{};
    s.faceCx=new Real[8]{};
    s.faceCy=new Real[8]{};
    s.faceCz=new Real[8]{};
    s.faceNeighbour=new int[8]{};
    s.faceOwner=new int[8]{};
    s.facePeriodicDx=new Real[8]{};
    s.facePeriodicDy=new Real[8]{};
    s.facePeriodicDz=new Real[8]{};
    s.facePeriodicPair=new int[8]{};
    s.faceWeight=new Real[8]{};
    s.gasBoundaryKind=new int[8]{};
    s.gasBoundaryP=new Real[8]{};
    s.gasBoundaryPFix=new int[8]{};
    s.gasBoundaryPWave=new int[8]{};
    s.gasBoundaryPWaveFieldInf=new Real[8]{};
    s.gasBoundaryPWaveGamma=new Real[8]{};
    s.gasBoundaryPWaveLInf=new Real[8]{};
    s.gasBoundaryRho=new Real[8]{};
    s.gasBoundaryRhoFix=new int[8]{};
    s.gasBoundaryT=new Real[8]{};
    s.gasBoundaryTFix=new int[8]{};
    s.gasBoundaryUFix=new int[8]{};
    s.gasBoundaryUx=new Real[8]{};
    s.gasBoundaryUy=new Real[8]{};
    s.gasBoundaryUz=new Real[8]{};
    s.gasDiffusionNumber=new Real[8]{};
    s.gasFluxPositivityScale=new Real[8]{};
    s.gasGradientLimiterP=new Real[8]{};
    s.gasGradientLimiterRho=new Real[8]{};
    s.gasGradientLimiterT=new Real[8]{};
    s.gasGradientLimiterUx=new Real[8]{};
    s.gasGradientLimiterUy=new Real[8]{};
    s.gasGradientLimiterUz=new Real[8]{};
    s.gasHllcAdcSensor=new Real[8]{};
    s.gasPhiRho=new Real[8]{};
    s.gasPhiRhoE=new Real[8]{};
    s.gasPhiRhoUx=new Real[8]{};
    s.gasPhiRhoUy=new Real[8]{};
    s.gasPhiRhoUz=new Real[8]{};
    s.gradKX=new Real[8]{};
    s.gradKY=new Real[8]{};
    s.gradKZ=new Real[8]{};
    s.gradOmegaX=new Real[8]{};
    s.gradOmegaY=new Real[8]{};
    s.gradOmegaZ=new Real[8]{};
    s.gradPx=new Real[8]{};
    s.gradPy=new Real[8]{};
    s.gradPz=new Real[8]{};
    s.gradRhoX=new Real[8]{};
    s.gradRhoY=new Real[8]{};
    s.gradRhoZ=new Real[8]{};
    s.gradTX=new Real[8]{};
    s.gradTY=new Real[8]{};
    s.gradTZ=new Real[8]{};
    s.gradUxX=new Real[8]{};
    s.gradUxY=new Real[8]{};
    s.gradUxZ=new Real[8]{};
    s.gradUyX=new Real[8]{};
    s.gradUyY=new Real[8]{};
    s.gradUyZ=new Real[8]{};
    s.gradUzX=new Real[8]{};
    s.gradUzY=new Real[8]{};
    s.gradUzZ=new Real[8]{};
    s.k=new Real[8]{};
    s.magSf=new Real[8]{};
    s.nut=new Real[8]{};
    s.omega=new Real[8]{};
    s.p=new Real[8]{};
    s.pressureScheduleTimes=new Time[8]{};
    s.pressureScheduleValues=new Real[8]{};
    s.rho=new Real[8]{};
    s.rhoE=new Real[8]{};
    s.rhoENext=new Real[8]{};
    s.rhoK=new Real[8]{};
    s.rhoKInitial=new Real[8]{};
    s.rhoNext=new Real[8]{};
    s.rhoOmega=new Real[8]{};
    s.rhoOmegaInitial=new Real[8]{};
    s.rhoUx=new Real[8]{};
    s.rhoUxNext=new Real[8]{};
    s.rhoUy=new Real[8]{};
    s.rhoUyNext=new Real[8]{};
    s.rhoUz=new Real[8]{};
    s.rhoUzNext=new Real[8]{};
    s.riemannBoundaryKind=new int[8]{};
    s.riemannBoundaryP=new Real[8]{};
    s.riemannBoundaryPFix=new int[8]{};
    s.riemannBoundaryPWave=new int[8]{};
    s.riemannBoundaryRho=new Real[8]{};
    s.riemannBoundaryRhoFix=new int[8]{};
    s.riemannBoundaryT=new Real[8]{};
    s.riemannBoundaryTFix=new int[8]{};
    s.riemannBoundaryUFix=new int[8]{};
    s.riemannBoundaryUx=new Real[8]{};
    s.riemannBoundaryUy=new Real[8]{};
    s.riemannBoundaryUz=new Real[8]{};
    s.scheduledInletFaceMask=new int[8]{};
    s.sstBoundaryK=new Real[8]{};
    s.sstBoundaryKMode=new int[8]{};
    s.sstBoundaryOmega=new Real[8]{};
    s.sstBoundaryOmegaMode=new int[8]{};
    s.sstF1=new Real[8]{};
    s.sstF2=new Real[8]{};
    s.sstPhiRhoK=new Real[8]{};
    s.sstPhiRhoOmega=new Real[8]{};
    s.sstSourceNumber=new Real[8]{};
    s.sstWallDistance=new Real[8]{};
#ifdef LEGACY_REFERENCE
 s.Rgas = Real(1.25);
 s.TgasMin = Real(1.25);
 s.gammaGas = Real(1.25);
 s.gasCp = Real(1.25);
 s.gasFluxScheme = 23;
 s.gasLimiter = 23;
 s.gasMu = Real(1.25);
 s.gasPrClamped = Real(1.25);
 s.gasReconstruction = 23;
 s.lesDeltaCoeff = Real(1.25);
 s.maxDiffusionNumber = Real(1.25);
 s.nCells = 23;
 s.nFaces = 23;
 s.nInternalFaces = 23;
 s.nPressureScheduleRows = 0;
 s.nScheduledInletFaces = 0;
 s.rhoMin = Real(1.25);
 s.scheduledInletTemperature = Real(1.25);
 s.smagorinskyCs = Real(1.25);
 s.sstCoefficients = ugkwp::defaultSstCoefficients();
 s.sstConfigured = 23;
 s.sstJayatillekeP = Real(1.25);
 s.sstKMin = Real(1.25);
 s.sstMaxSourceNumber = Real(1.25);
 s.sstOmegaMin = Real(1.25);
 s.sstThermalYPlus = Real(1.25);
 s.sstWallCmu = Real(1.25);
 s.sstWallE = Real(1.25);
 s.sstWallKappa = Real(1.25);
 s.sstWallTreatment = 0;
 s.turbulenceModel = 23;
 s.turbulentPrandtl = Real(1.25);
 s.waleCw = Real(1.25);
 run(s,scheme,reconstruction,limiter,turbulence);
#else
 s.Rgas = Real(1.25);
 s.TgasMin = Real(1.25);
 s.gammaGas = Real(1.25);
 s.gasCp = Real(1.25);
 s.gasFluxScheme = 23;
 s.gasLimiter = 23;
 s.gasMu = Real(1.25);
 s.gasPrClamped = Real(1.25);
 s.gasReconstruction = 23;
 s.lesDeltaCoeff = Real(1.25);
 s.maxDiffusionNumber = Real(1.25);
 s.nCells = 23;
 s.nFaces = 23;
 s.nInternalFaces = 23;
 s.nPressureScheduleRows = 0;
 s.nScheduledInletFaces = 0;
 s.rhoMin = Real(1.25);
 s.scheduledInletTemperature = Real(1.25);
 s.smagorinskyCs = Real(1.25);
 s.sstCoefficients = ugkwp::defaultSstCoefficients();
 s.sstConfigured = 23;
 s.sstJayatillekeP = Real(1.25);
 s.sstKMin = Real(1.25);
 s.sstMaxSourceNumber = Real(1.25);
 s.sstOmegaMin = Real(1.25);
 s.sstThermalYPlus = Real(1.25);
 s.sstWallCmu = Real(1.25);
 s.sstWallE = Real(1.25);
 s.sstWallKappa = Real(1.25);
 s.sstWallTreatment = 0;
 s.turbulenceModel = 23;
 s.turbulentPrandtl = Real(1.25);
 s.waleCw = Real(1.25);
 auto gas=ugkwp::makeGasStateView(s);
 if(std::memcmp(&gas.Cx,&s.Cx,sizeof(s.Cx)))std::abort();
 if(std::memcmp(&gas.Cy,&s.Cy,sizeof(s.Cy)))std::abort();
 if(std::memcmp(&gas.Cz,&s.Cz,sizeof(s.Cz)))std::abort();
 if(std::memcmp(&gas.Rgas,&s.Rgas,sizeof(s.Rgas)))std::abort();
 if(std::memcmp(&gas.Sfx,&s.Sfx,sizeof(s.Sfx)))std::abort();
 if(std::memcmp(&gas.Sfy,&s.Sfy,sizeof(s.Sfy)))std::abort();
 if(std::memcmp(&gas.Sfz,&s.Sfz,sizeof(s.Sfz)))std::abort();
 if(std::memcmp(&gas.Tgas,&s.Tgas,sizeof(s.Tgas)))std::abort();
 if(std::memcmp(&gas.TgasMin,&s.TgasMin,sizeof(s.TgasMin)))std::abort();
 if(std::memcmp(&gas.Ux,&s.Ux,sizeof(s.Ux)))std::abort();
 if(std::memcmp(&gas.Uy,&s.Uy,sizeof(s.Uy)))std::abort();
 if(std::memcmp(&gas.Uz,&s.Uz,sizeof(s.Uz)))std::abort();
 if(std::memcmp(&gas.V,&s.V,sizeof(s.V)))std::abort();
 if(std::memcmp(&gas.cellFaceId,&s.cellFaceId,sizeof(s.cellFaceId)))std::abort();
 if(std::memcmp(&gas.cellLength,&s.cellLength,sizeof(s.cellLength)))std::abort();
 if(std::memcmp(&gas.cellPlaneCount,&s.cellPlaneCount,sizeof(s.cellPlaneCount)))std::abort();
 if(std::memcmp(&gas.cellPlaneStart,&s.cellPlaneStart,sizeof(s.cellPlaneStart)))std::abort();
 if(std::memcmp(&gas.deltaCoeffs,&s.deltaCoeffs,sizeof(s.deltaCoeffs)))std::abort();
 if(std::memcmp(&gas.faceCx,&s.faceCx,sizeof(s.faceCx)))std::abort();
 if(std::memcmp(&gas.faceCy,&s.faceCy,sizeof(s.faceCy)))std::abort();
 if(std::memcmp(&gas.faceCz,&s.faceCz,sizeof(s.faceCz)))std::abort();
 if(std::memcmp(&gas.faceNeighbour,&s.faceNeighbour,sizeof(s.faceNeighbour)))std::abort();
 if(std::memcmp(&gas.faceOwner,&s.faceOwner,sizeof(s.faceOwner)))std::abort();
 if(std::memcmp(&gas.facePeriodicDx,&s.facePeriodicDx,sizeof(s.facePeriodicDx)))std::abort();
 if(std::memcmp(&gas.facePeriodicDy,&s.facePeriodicDy,sizeof(s.facePeriodicDy)))std::abort();
 if(std::memcmp(&gas.facePeriodicDz,&s.facePeriodicDz,sizeof(s.facePeriodicDz)))std::abort();
 if(std::memcmp(&gas.facePeriodicPair,&s.facePeriodicPair,sizeof(s.facePeriodicPair)))std::abort();
 if(std::memcmp(&gas.faceWeight,&s.faceWeight,sizeof(s.faceWeight)))std::abort();
 if(std::memcmp(&gas.gammaGas,&s.gammaGas,sizeof(s.gammaGas)))std::abort();
 if(std::memcmp(&gas.gasBoundaryKind,&s.gasBoundaryKind,sizeof(s.gasBoundaryKind)))std::abort();
 if(std::memcmp(&gas.gasBoundaryP,&s.gasBoundaryP,sizeof(s.gasBoundaryP)))std::abort();
 if(std::memcmp(&gas.gasBoundaryPFix,&s.gasBoundaryPFix,sizeof(s.gasBoundaryPFix)))std::abort();
 if(std::memcmp(&gas.gasBoundaryPWave,&s.gasBoundaryPWave,sizeof(s.gasBoundaryPWave)))std::abort();
 if(std::memcmp(&gas.gasBoundaryPWaveFieldInf,&s.gasBoundaryPWaveFieldInf,sizeof(s.gasBoundaryPWaveFieldInf)))std::abort();
 if(std::memcmp(&gas.gasBoundaryPWaveGamma,&s.gasBoundaryPWaveGamma,sizeof(s.gasBoundaryPWaveGamma)))std::abort();
 if(std::memcmp(&gas.gasBoundaryPWaveLInf,&s.gasBoundaryPWaveLInf,sizeof(s.gasBoundaryPWaveLInf)))std::abort();
 if(std::memcmp(&gas.gasBoundaryRho,&s.gasBoundaryRho,sizeof(s.gasBoundaryRho)))std::abort();
 if(std::memcmp(&gas.gasBoundaryRhoFix,&s.gasBoundaryRhoFix,sizeof(s.gasBoundaryRhoFix)))std::abort();
 if(std::memcmp(&gas.gasBoundaryT,&s.gasBoundaryT,sizeof(s.gasBoundaryT)))std::abort();
 if(std::memcmp(&gas.gasBoundaryTFix,&s.gasBoundaryTFix,sizeof(s.gasBoundaryTFix)))std::abort();
 if(std::memcmp(&gas.gasBoundaryUFix,&s.gasBoundaryUFix,sizeof(s.gasBoundaryUFix)))std::abort();
 if(std::memcmp(&gas.gasBoundaryUx,&s.gasBoundaryUx,sizeof(s.gasBoundaryUx)))std::abort();
 if(std::memcmp(&gas.gasBoundaryUy,&s.gasBoundaryUy,sizeof(s.gasBoundaryUy)))std::abort();
 if(std::memcmp(&gas.gasBoundaryUz,&s.gasBoundaryUz,sizeof(s.gasBoundaryUz)))std::abort();
 if(std::memcmp(&gas.gasCp,&s.gasCp,sizeof(s.gasCp)))std::abort();
 if(std::memcmp(&gas.gasDiffusionNumber,&s.gasDiffusionNumber,sizeof(s.gasDiffusionNumber)))std::abort();
 if(std::memcmp(&gas.gasFluxPositivityScale,&s.gasFluxPositivityScale,sizeof(s.gasFluxPositivityScale)))std::abort();
 if(std::memcmp(&gas.gasFluxScheme,&s.gasFluxScheme,sizeof(s.gasFluxScheme)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterP,&s.gasGradientLimiterP,sizeof(s.gasGradientLimiterP)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterRho,&s.gasGradientLimiterRho,sizeof(s.gasGradientLimiterRho)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterT,&s.gasGradientLimiterT,sizeof(s.gasGradientLimiterT)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterUx,&s.gasGradientLimiterUx,sizeof(s.gasGradientLimiterUx)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterUy,&s.gasGradientLimiterUy,sizeof(s.gasGradientLimiterUy)))std::abort();
 if(std::memcmp(&gas.gasGradientLimiterUz,&s.gasGradientLimiterUz,sizeof(s.gasGradientLimiterUz)))std::abort();
 if(std::memcmp(&gas.gasHllcAdcSensor,&s.gasHllcAdcSensor,sizeof(s.gasHllcAdcSensor)))std::abort();
 if(std::memcmp(&gas.gasLimiter,&s.gasLimiter,sizeof(s.gasLimiter)))std::abort();
 if(std::memcmp(&gas.gasMu,&s.gasMu,sizeof(s.gasMu)))std::abort();
 if(std::memcmp(&gas.gasPhiRho,&s.gasPhiRho,sizeof(s.gasPhiRho)))std::abort();
 if(std::memcmp(&gas.gasPhiRhoE,&s.gasPhiRhoE,sizeof(s.gasPhiRhoE)))std::abort();
 if(std::memcmp(&gas.gasPhiRhoUx,&s.gasPhiRhoUx,sizeof(s.gasPhiRhoUx)))std::abort();
 if(std::memcmp(&gas.gasPhiRhoUy,&s.gasPhiRhoUy,sizeof(s.gasPhiRhoUy)))std::abort();
 if(std::memcmp(&gas.gasPhiRhoUz,&s.gasPhiRhoUz,sizeof(s.gasPhiRhoUz)))std::abort();
 if(std::memcmp(&gas.gasPrClamped,&s.gasPrClamped,sizeof(s.gasPrClamped)))std::abort();
 if(std::memcmp(&gas.gasReconstruction,&s.gasReconstruction,sizeof(s.gasReconstruction)))std::abort();
 if(std::memcmp(&gas.gradKX,&s.gradKX,sizeof(s.gradKX)))std::abort();
 if(std::memcmp(&gas.gradKY,&s.gradKY,sizeof(s.gradKY)))std::abort();
 if(std::memcmp(&gas.gradKZ,&s.gradKZ,sizeof(s.gradKZ)))std::abort();
 if(std::memcmp(&gas.gradOmegaX,&s.gradOmegaX,sizeof(s.gradOmegaX)))std::abort();
 if(std::memcmp(&gas.gradOmegaY,&s.gradOmegaY,sizeof(s.gradOmegaY)))std::abort();
 if(std::memcmp(&gas.gradOmegaZ,&s.gradOmegaZ,sizeof(s.gradOmegaZ)))std::abort();
 if(std::memcmp(&gas.gradPx,&s.gradPx,sizeof(s.gradPx)))std::abort();
 if(std::memcmp(&gas.gradPy,&s.gradPy,sizeof(s.gradPy)))std::abort();
 if(std::memcmp(&gas.gradPz,&s.gradPz,sizeof(s.gradPz)))std::abort();
 if(std::memcmp(&gas.gradRhoX,&s.gradRhoX,sizeof(s.gradRhoX)))std::abort();
 if(std::memcmp(&gas.gradRhoY,&s.gradRhoY,sizeof(s.gradRhoY)))std::abort();
 if(std::memcmp(&gas.gradRhoZ,&s.gradRhoZ,sizeof(s.gradRhoZ)))std::abort();
 if(std::memcmp(&gas.gradTX,&s.gradTX,sizeof(s.gradTX)))std::abort();
 if(std::memcmp(&gas.gradTY,&s.gradTY,sizeof(s.gradTY)))std::abort();
 if(std::memcmp(&gas.gradTZ,&s.gradTZ,sizeof(s.gradTZ)))std::abort();
 if(std::memcmp(&gas.gradUxX,&s.gradUxX,sizeof(s.gradUxX)))std::abort();
 if(std::memcmp(&gas.gradUxY,&s.gradUxY,sizeof(s.gradUxY)))std::abort();
 if(std::memcmp(&gas.gradUxZ,&s.gradUxZ,sizeof(s.gradUxZ)))std::abort();
 if(std::memcmp(&gas.gradUyX,&s.gradUyX,sizeof(s.gradUyX)))std::abort();
 if(std::memcmp(&gas.gradUyY,&s.gradUyY,sizeof(s.gradUyY)))std::abort();
 if(std::memcmp(&gas.gradUyZ,&s.gradUyZ,sizeof(s.gradUyZ)))std::abort();
 if(std::memcmp(&gas.gradUzX,&s.gradUzX,sizeof(s.gradUzX)))std::abort();
 if(std::memcmp(&gas.gradUzY,&s.gradUzY,sizeof(s.gradUzY)))std::abort();
 if(std::memcmp(&gas.gradUzZ,&s.gradUzZ,sizeof(s.gradUzZ)))std::abort();
 if(std::memcmp(&gas.k,&s.k,sizeof(s.k)))std::abort();
 if(std::memcmp(&gas.lesDeltaCoeff,&s.lesDeltaCoeff,sizeof(s.lesDeltaCoeff)))std::abort();
 if(std::memcmp(&gas.magSf,&s.magSf,sizeof(s.magSf)))std::abort();
 if(std::memcmp(&gas.maxDiffusionNumber,&s.maxDiffusionNumber,sizeof(s.maxDiffusionNumber)))std::abort();
 if(std::memcmp(&gas.nCells,&s.nCells,sizeof(s.nCells)))std::abort();
 if(std::memcmp(&gas.nFaces,&s.nFaces,sizeof(s.nFaces)))std::abort();
 if(std::memcmp(&gas.nInternalFaces,&s.nInternalFaces,sizeof(s.nInternalFaces)))std::abort();
 if(std::memcmp(&gas.nPressureScheduleRows,&s.nPressureScheduleRows,sizeof(s.nPressureScheduleRows)))std::abort();
 if(std::memcmp(&gas.nScheduledInletFaces,&s.nScheduledInletFaces,sizeof(s.nScheduledInletFaces)))std::abort();
 if(std::memcmp(&gas.nut,&s.nut,sizeof(s.nut)))std::abort();
 if(std::memcmp(&gas.omega,&s.omega,sizeof(s.omega)))std::abort();
 if(std::memcmp(&gas.p,&s.p,sizeof(s.p)))std::abort();
 if(std::memcmp(&gas.pressureScheduleTimes,&s.pressureScheduleTimes,sizeof(s.pressureScheduleTimes)))std::abort();
 if(std::memcmp(&gas.pressureScheduleValues,&s.pressureScheduleValues,sizeof(s.pressureScheduleValues)))std::abort();
 if(std::memcmp(&gas.rho,&s.rho,sizeof(s.rho)))std::abort();
 if(std::memcmp(&gas.rhoE,&s.rhoE,sizeof(s.rhoE)))std::abort();
 if(std::memcmp(&gas.rhoENext,&s.rhoENext,sizeof(s.rhoENext)))std::abort();
 if(std::memcmp(&gas.rhoK,&s.rhoK,sizeof(s.rhoK)))std::abort();
 if(std::memcmp(&gas.rhoKInitial,&s.rhoKInitial,sizeof(s.rhoKInitial)))std::abort();
 if(std::memcmp(&gas.rhoMin,&s.rhoMin,sizeof(s.rhoMin)))std::abort();
 if(std::memcmp(&gas.rhoNext,&s.rhoNext,sizeof(s.rhoNext)))std::abort();
 if(std::memcmp(&gas.rhoOmega,&s.rhoOmega,sizeof(s.rhoOmega)))std::abort();
 if(std::memcmp(&gas.rhoOmegaInitial,&s.rhoOmegaInitial,sizeof(s.rhoOmegaInitial)))std::abort();
 if(std::memcmp(&gas.rhoUx,&s.rhoUx,sizeof(s.rhoUx)))std::abort();
 if(std::memcmp(&gas.rhoUxNext,&s.rhoUxNext,sizeof(s.rhoUxNext)))std::abort();
 if(std::memcmp(&gas.rhoUy,&s.rhoUy,sizeof(s.rhoUy)))std::abort();
 if(std::memcmp(&gas.rhoUyNext,&s.rhoUyNext,sizeof(s.rhoUyNext)))std::abort();
 if(std::memcmp(&gas.rhoUz,&s.rhoUz,sizeof(s.rhoUz)))std::abort();
 if(std::memcmp(&gas.rhoUzNext,&s.rhoUzNext,sizeof(s.rhoUzNext)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryKind,&s.riemannBoundaryKind,sizeof(s.riemannBoundaryKind)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryP,&s.riemannBoundaryP,sizeof(s.riemannBoundaryP)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryPFix,&s.riemannBoundaryPFix,sizeof(s.riemannBoundaryPFix)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryPWave,&s.riemannBoundaryPWave,sizeof(s.riemannBoundaryPWave)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryRho,&s.riemannBoundaryRho,sizeof(s.riemannBoundaryRho)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryRhoFix,&s.riemannBoundaryRhoFix,sizeof(s.riemannBoundaryRhoFix)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryT,&s.riemannBoundaryT,sizeof(s.riemannBoundaryT)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryTFix,&s.riemannBoundaryTFix,sizeof(s.riemannBoundaryTFix)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryUFix,&s.riemannBoundaryUFix,sizeof(s.riemannBoundaryUFix)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryUx,&s.riemannBoundaryUx,sizeof(s.riemannBoundaryUx)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryUy,&s.riemannBoundaryUy,sizeof(s.riemannBoundaryUy)))std::abort();
 if(std::memcmp(&gas.riemannBoundaryUz,&s.riemannBoundaryUz,sizeof(s.riemannBoundaryUz)))std::abort();
 if(std::memcmp(&gas.scheduledInletFaceMask,&s.scheduledInletFaceMask,sizeof(s.scheduledInletFaceMask)))std::abort();
 if(std::memcmp(&gas.scheduledInletTemperature,&s.scheduledInletTemperature,sizeof(s.scheduledInletTemperature)))std::abort();
 if(std::memcmp(&gas.smagorinskyCs,&s.smagorinskyCs,sizeof(s.smagorinskyCs)))std::abort();
 if(std::memcmp(&gas.sstBoundaryK,&s.sstBoundaryK,sizeof(s.sstBoundaryK)))std::abort();
 if(std::memcmp(&gas.sstBoundaryKMode,&s.sstBoundaryKMode,sizeof(s.sstBoundaryKMode)))std::abort();
 if(std::memcmp(&gas.sstBoundaryOmega,&s.sstBoundaryOmega,sizeof(s.sstBoundaryOmega)))std::abort();
 if(std::memcmp(&gas.sstBoundaryOmegaMode,&s.sstBoundaryOmegaMode,sizeof(s.sstBoundaryOmegaMode)))std::abort();
 if(std::memcmp(&gas.sstCoefficients,&s.sstCoefficients,sizeof(s.sstCoefficients)))std::abort();
 if(std::memcmp(&gas.sstConfigured,&s.sstConfigured,sizeof(s.sstConfigured)))std::abort();
 if(std::memcmp(&gas.sstF1,&s.sstF1,sizeof(s.sstF1)))std::abort();
 if(std::memcmp(&gas.sstF2,&s.sstF2,sizeof(s.sstF2)))std::abort();
 if(std::memcmp(&gas.sstJayatillekeP,&s.sstJayatillekeP,sizeof(s.sstJayatillekeP)))std::abort();
 if(std::memcmp(&gas.sstKMin,&s.sstKMin,sizeof(s.sstKMin)))std::abort();
 if(std::memcmp(&gas.sstMaxSourceNumber,&s.sstMaxSourceNumber,sizeof(s.sstMaxSourceNumber)))std::abort();
 if(std::memcmp(&gas.sstOmegaMin,&s.sstOmegaMin,sizeof(s.sstOmegaMin)))std::abort();
 if(std::memcmp(&gas.sstPhiRhoK,&s.sstPhiRhoK,sizeof(s.sstPhiRhoK)))std::abort();
 if(std::memcmp(&gas.sstPhiRhoOmega,&s.sstPhiRhoOmega,sizeof(s.sstPhiRhoOmega)))std::abort();
 if(std::memcmp(&gas.sstSourceNumber,&s.sstSourceNumber,sizeof(s.sstSourceNumber)))std::abort();
 if(std::memcmp(&gas.sstThermalYPlus,&s.sstThermalYPlus,sizeof(s.sstThermalYPlus)))std::abort();
 if(std::memcmp(&gas.sstWallCmu,&s.sstWallCmu,sizeof(s.sstWallCmu)))std::abort();
 if(std::memcmp(&gas.sstWallDistance,&s.sstWallDistance,sizeof(s.sstWallDistance)))std::abort();
 if(std::memcmp(&gas.sstWallE,&s.sstWallE,sizeof(s.sstWallE)))std::abort();
 if(std::memcmp(&gas.sstWallKappa,&s.sstWallKappa,sizeof(s.sstWallKappa)))std::abort();
 if(std::memcmp(&gas.sstWallTreatment,&s.sstWallTreatment,sizeof(s.sstWallTreatment)))std::abort();
 if(std::memcmp(&gas.turbulenceModel,&s.turbulenceModel,sizeof(s.turbulenceModel)))std::abort();
 if(std::memcmp(&gas.turbulentPrandtl,&s.turbulentPrandtl,sizeof(s.turbulentPrandtl)))std::abort();
 if(std::memcmp(&gas.waleCw,&s.waleCw,sizeof(s.waleCw)))std::abort();
 static_assert(std::is_trivially_copyable<decltype(gas)>::value,"gas view must be CUDA-copyable");
 static_assert(decltype(gas)::speciesCount==0,"legacy mode has no species allocation");
 if(gas.rho!=s.rho||gas.pressureScheduleTimes!=s.pressureScheduleTimes)std::abort();
 run(gas,scheme,reconstruction,limiter,turbulence);
#endif
 std::uint64_t h=14695981039346656037ull;
    hashBytes(h,s.Cx,8*sizeof(*s.Cx));
    hashBytes(h,s.Cy,8*sizeof(*s.Cy));
    hashBytes(h,s.Cz,8*sizeof(*s.Cz));
    hashBytes(h,s.Sfx,8*sizeof(*s.Sfx));
    hashBytes(h,s.Sfy,8*sizeof(*s.Sfy));
    hashBytes(h,s.Sfz,8*sizeof(*s.Sfz));
    hashBytes(h,s.Tgas,8*sizeof(*s.Tgas));
    hashBytes(h,s.Ux,8*sizeof(*s.Ux));
    hashBytes(h,s.Uy,8*sizeof(*s.Uy));
    hashBytes(h,s.Uz,8*sizeof(*s.Uz));
    hashBytes(h,s.V,8*sizeof(*s.V));
    hashBytes(h,s.cellFaceId,8*sizeof(*s.cellFaceId));
    hashBytes(h,s.cellLength,8*sizeof(*s.cellLength));
    hashBytes(h,s.cellPlaneCount,8*sizeof(*s.cellPlaneCount));
    hashBytes(h,s.cellPlaneStart,8*sizeof(*s.cellPlaneStart));
    hashBytes(h,s.deltaCoeffs,8*sizeof(*s.deltaCoeffs));
    hashBytes(h,s.faceCx,8*sizeof(*s.faceCx));
    hashBytes(h,s.faceCy,8*sizeof(*s.faceCy));
    hashBytes(h,s.faceCz,8*sizeof(*s.faceCz));
    hashBytes(h,s.faceNeighbour,8*sizeof(*s.faceNeighbour));
    hashBytes(h,s.faceOwner,8*sizeof(*s.faceOwner));
    hashBytes(h,s.facePeriodicDx,8*sizeof(*s.facePeriodicDx));
    hashBytes(h,s.facePeriodicDy,8*sizeof(*s.facePeriodicDy));
    hashBytes(h,s.facePeriodicDz,8*sizeof(*s.facePeriodicDz));
    hashBytes(h,s.facePeriodicPair,8*sizeof(*s.facePeriodicPair));
    hashBytes(h,s.faceWeight,8*sizeof(*s.faceWeight));
    hashBytes(h,s.gasBoundaryKind,8*sizeof(*s.gasBoundaryKind));
    hashBytes(h,s.gasBoundaryP,8*sizeof(*s.gasBoundaryP));
    hashBytes(h,s.gasBoundaryPFix,8*sizeof(*s.gasBoundaryPFix));
    hashBytes(h,s.gasBoundaryPWave,8*sizeof(*s.gasBoundaryPWave));
    hashBytes(h,s.gasBoundaryPWaveFieldInf,8*sizeof(*s.gasBoundaryPWaveFieldInf));
    hashBytes(h,s.gasBoundaryPWaveGamma,8*sizeof(*s.gasBoundaryPWaveGamma));
    hashBytes(h,s.gasBoundaryPWaveLInf,8*sizeof(*s.gasBoundaryPWaveLInf));
    hashBytes(h,s.gasBoundaryRho,8*sizeof(*s.gasBoundaryRho));
    hashBytes(h,s.gasBoundaryRhoFix,8*sizeof(*s.gasBoundaryRhoFix));
    hashBytes(h,s.gasBoundaryT,8*sizeof(*s.gasBoundaryT));
    hashBytes(h,s.gasBoundaryTFix,8*sizeof(*s.gasBoundaryTFix));
    hashBytes(h,s.gasBoundaryUFix,8*sizeof(*s.gasBoundaryUFix));
    hashBytes(h,s.gasBoundaryUx,8*sizeof(*s.gasBoundaryUx));
    hashBytes(h,s.gasBoundaryUy,8*sizeof(*s.gasBoundaryUy));
    hashBytes(h,s.gasBoundaryUz,8*sizeof(*s.gasBoundaryUz));
    hashBytes(h,s.gasDiffusionNumber,8*sizeof(*s.gasDiffusionNumber));
    hashBytes(h,s.gasFluxPositivityScale,8*sizeof(*s.gasFluxPositivityScale));
    hashBytes(h,s.gasGradientLimiterP,8*sizeof(*s.gasGradientLimiterP));
    hashBytes(h,s.gasGradientLimiterRho,8*sizeof(*s.gasGradientLimiterRho));
    hashBytes(h,s.gasGradientLimiterT,8*sizeof(*s.gasGradientLimiterT));
    hashBytes(h,s.gasGradientLimiterUx,8*sizeof(*s.gasGradientLimiterUx));
    hashBytes(h,s.gasGradientLimiterUy,8*sizeof(*s.gasGradientLimiterUy));
    hashBytes(h,s.gasGradientLimiterUz,8*sizeof(*s.gasGradientLimiterUz));
    hashBytes(h,s.gasHllcAdcSensor,8*sizeof(*s.gasHllcAdcSensor));
    hashBytes(h,s.gasPhiRho,8*sizeof(*s.gasPhiRho));
    hashBytes(h,s.gasPhiRhoE,8*sizeof(*s.gasPhiRhoE));
    hashBytes(h,s.gasPhiRhoUx,8*sizeof(*s.gasPhiRhoUx));
    hashBytes(h,s.gasPhiRhoUy,8*sizeof(*s.gasPhiRhoUy));
    hashBytes(h,s.gasPhiRhoUz,8*sizeof(*s.gasPhiRhoUz));
    hashBytes(h,s.gradKX,8*sizeof(*s.gradKX));
    hashBytes(h,s.gradKY,8*sizeof(*s.gradKY));
    hashBytes(h,s.gradKZ,8*sizeof(*s.gradKZ));
    hashBytes(h,s.gradOmegaX,8*sizeof(*s.gradOmegaX));
    hashBytes(h,s.gradOmegaY,8*sizeof(*s.gradOmegaY));
    hashBytes(h,s.gradOmegaZ,8*sizeof(*s.gradOmegaZ));
    hashBytes(h,s.gradPx,8*sizeof(*s.gradPx));
    hashBytes(h,s.gradPy,8*sizeof(*s.gradPy));
    hashBytes(h,s.gradPz,8*sizeof(*s.gradPz));
    hashBytes(h,s.gradRhoX,8*sizeof(*s.gradRhoX));
    hashBytes(h,s.gradRhoY,8*sizeof(*s.gradRhoY));
    hashBytes(h,s.gradRhoZ,8*sizeof(*s.gradRhoZ));
    hashBytes(h,s.gradTX,8*sizeof(*s.gradTX));
    hashBytes(h,s.gradTY,8*sizeof(*s.gradTY));
    hashBytes(h,s.gradTZ,8*sizeof(*s.gradTZ));
    hashBytes(h,s.gradUxX,8*sizeof(*s.gradUxX));
    hashBytes(h,s.gradUxY,8*sizeof(*s.gradUxY));
    hashBytes(h,s.gradUxZ,8*sizeof(*s.gradUxZ));
    hashBytes(h,s.gradUyX,8*sizeof(*s.gradUyX));
    hashBytes(h,s.gradUyY,8*sizeof(*s.gradUyY));
    hashBytes(h,s.gradUyZ,8*sizeof(*s.gradUyZ));
    hashBytes(h,s.gradUzX,8*sizeof(*s.gradUzX));
    hashBytes(h,s.gradUzY,8*sizeof(*s.gradUzY));
    hashBytes(h,s.gradUzZ,8*sizeof(*s.gradUzZ));
    hashBytes(h,s.k,8*sizeof(*s.k));
    hashBytes(h,s.magSf,8*sizeof(*s.magSf));
    hashBytes(h,s.nut,8*sizeof(*s.nut));
    hashBytes(h,s.omega,8*sizeof(*s.omega));
    hashBytes(h,s.p,8*sizeof(*s.p));
    hashBytes(h,s.pressureScheduleTimes,8*sizeof(*s.pressureScheduleTimes));
    hashBytes(h,s.pressureScheduleValues,8*sizeof(*s.pressureScheduleValues));
    hashBytes(h,s.rho,8*sizeof(*s.rho));
    hashBytes(h,s.rhoE,8*sizeof(*s.rhoE));
    hashBytes(h,s.rhoENext,8*sizeof(*s.rhoENext));
    hashBytes(h,s.rhoK,8*sizeof(*s.rhoK));
    hashBytes(h,s.rhoKInitial,8*sizeof(*s.rhoKInitial));
    hashBytes(h,s.rhoNext,8*sizeof(*s.rhoNext));
    hashBytes(h,s.rhoOmega,8*sizeof(*s.rhoOmega));
    hashBytes(h,s.rhoOmegaInitial,8*sizeof(*s.rhoOmegaInitial));
    hashBytes(h,s.rhoUx,8*sizeof(*s.rhoUx));
    hashBytes(h,s.rhoUxNext,8*sizeof(*s.rhoUxNext));
    hashBytes(h,s.rhoUy,8*sizeof(*s.rhoUy));
    hashBytes(h,s.rhoUyNext,8*sizeof(*s.rhoUyNext));
    hashBytes(h,s.rhoUz,8*sizeof(*s.rhoUz));
    hashBytes(h,s.rhoUzNext,8*sizeof(*s.rhoUzNext));
    hashBytes(h,s.riemannBoundaryKind,8*sizeof(*s.riemannBoundaryKind));
    hashBytes(h,s.riemannBoundaryP,8*sizeof(*s.riemannBoundaryP));
    hashBytes(h,s.riemannBoundaryPFix,8*sizeof(*s.riemannBoundaryPFix));
    hashBytes(h,s.riemannBoundaryPWave,8*sizeof(*s.riemannBoundaryPWave));
    hashBytes(h,s.riemannBoundaryRho,8*sizeof(*s.riemannBoundaryRho));
    hashBytes(h,s.riemannBoundaryRhoFix,8*sizeof(*s.riemannBoundaryRhoFix));
    hashBytes(h,s.riemannBoundaryT,8*sizeof(*s.riemannBoundaryT));
    hashBytes(h,s.riemannBoundaryTFix,8*sizeof(*s.riemannBoundaryTFix));
    hashBytes(h,s.riemannBoundaryUFix,8*sizeof(*s.riemannBoundaryUFix));
    hashBytes(h,s.riemannBoundaryUx,8*sizeof(*s.riemannBoundaryUx));
    hashBytes(h,s.riemannBoundaryUy,8*sizeof(*s.riemannBoundaryUy));
    hashBytes(h,s.riemannBoundaryUz,8*sizeof(*s.riemannBoundaryUz));
    hashBytes(h,s.scheduledInletFaceMask,8*sizeof(*s.scheduledInletFaceMask));
    hashBytes(h,s.sstBoundaryK,8*sizeof(*s.sstBoundaryK));
    hashBytes(h,s.sstBoundaryKMode,8*sizeof(*s.sstBoundaryKMode));
    hashBytes(h,s.sstBoundaryOmega,8*sizeof(*s.sstBoundaryOmega));
    hashBytes(h,s.sstBoundaryOmegaMode,8*sizeof(*s.sstBoundaryOmegaMode));
    hashBytes(h,s.sstF1,8*sizeof(*s.sstF1));
    hashBytes(h,s.sstF2,8*sizeof(*s.sstF2));
    hashBytes(h,s.sstPhiRhoK,8*sizeof(*s.sstPhiRhoK));
    hashBytes(h,s.sstPhiRhoOmega,8*sizeof(*s.sstPhiRhoOmega));
    hashBytes(h,s.sstSourceNumber,8*sizeof(*s.sstSourceNumber));
    hashBytes(h,s.sstWallDistance,8*sizeof(*s.sstWallDistance));
 std::printf("%d %d %d %d %llu\n",scheme,reconstruction,limiter,turbulence,static_cast<unsigned long long>(h));
    delete[] s.Cx;
    delete[] s.Cy;
    delete[] s.Cz;
    delete[] s.Sfx;
    delete[] s.Sfy;
    delete[] s.Sfz;
    delete[] s.Tgas;
    delete[] s.Ux;
    delete[] s.Uy;
    delete[] s.Uz;
    delete[] s.V;
    delete[] s.cellFaceId;
    delete[] s.cellLength;
    delete[] s.cellPlaneCount;
    delete[] s.cellPlaneStart;
    delete[] s.deltaCoeffs;
    delete[] s.faceCx;
    delete[] s.faceCy;
    delete[] s.faceCz;
    delete[] s.faceNeighbour;
    delete[] s.faceOwner;
    delete[] s.facePeriodicDx;
    delete[] s.facePeriodicDy;
    delete[] s.facePeriodicDz;
    delete[] s.facePeriodicPair;
    delete[] s.faceWeight;
    delete[] s.gasBoundaryKind;
    delete[] s.gasBoundaryP;
    delete[] s.gasBoundaryPFix;
    delete[] s.gasBoundaryPWave;
    delete[] s.gasBoundaryPWaveFieldInf;
    delete[] s.gasBoundaryPWaveGamma;
    delete[] s.gasBoundaryPWaveLInf;
    delete[] s.gasBoundaryRho;
    delete[] s.gasBoundaryRhoFix;
    delete[] s.gasBoundaryT;
    delete[] s.gasBoundaryTFix;
    delete[] s.gasBoundaryUFix;
    delete[] s.gasBoundaryUx;
    delete[] s.gasBoundaryUy;
    delete[] s.gasBoundaryUz;
    delete[] s.gasDiffusionNumber;
    delete[] s.gasFluxPositivityScale;
    delete[] s.gasGradientLimiterP;
    delete[] s.gasGradientLimiterRho;
    delete[] s.gasGradientLimiterT;
    delete[] s.gasGradientLimiterUx;
    delete[] s.gasGradientLimiterUy;
    delete[] s.gasGradientLimiterUz;
    delete[] s.gasHllcAdcSensor;
    delete[] s.gasPhiRho;
    delete[] s.gasPhiRhoE;
    delete[] s.gasPhiRhoUx;
    delete[] s.gasPhiRhoUy;
    delete[] s.gasPhiRhoUz;
    delete[] s.gradKX;
    delete[] s.gradKY;
    delete[] s.gradKZ;
    delete[] s.gradOmegaX;
    delete[] s.gradOmegaY;
    delete[] s.gradOmegaZ;
    delete[] s.gradPx;
    delete[] s.gradPy;
    delete[] s.gradPz;
    delete[] s.gradRhoX;
    delete[] s.gradRhoY;
    delete[] s.gradRhoZ;
    delete[] s.gradTX;
    delete[] s.gradTY;
    delete[] s.gradTZ;
    delete[] s.gradUxX;
    delete[] s.gradUxY;
    delete[] s.gradUxZ;
    delete[] s.gradUyX;
    delete[] s.gradUyY;
    delete[] s.gradUyZ;
    delete[] s.gradUzX;
    delete[] s.gradUzY;
    delete[] s.gradUzZ;
    delete[] s.k;
    delete[] s.magSf;
    delete[] s.nut;
    delete[] s.omega;
    delete[] s.p;
    delete[] s.pressureScheduleTimes;
    delete[] s.pressureScheduleValues;
    delete[] s.rho;
    delete[] s.rhoE;
    delete[] s.rhoENext;
    delete[] s.rhoK;
    delete[] s.rhoKInitial;
    delete[] s.rhoNext;
    delete[] s.rhoOmega;
    delete[] s.rhoOmegaInitial;
    delete[] s.rhoUx;
    delete[] s.rhoUxNext;
    delete[] s.rhoUy;
    delete[] s.rhoUyNext;
    delete[] s.rhoUz;
    delete[] s.rhoUzNext;
    delete[] s.riemannBoundaryKind;
    delete[] s.riemannBoundaryP;
    delete[] s.riemannBoundaryPFix;
    delete[] s.riemannBoundaryPWave;
    delete[] s.riemannBoundaryRho;
    delete[] s.riemannBoundaryRhoFix;
    delete[] s.riemannBoundaryT;
    delete[] s.riemannBoundaryTFix;
    delete[] s.riemannBoundaryUFix;
    delete[] s.riemannBoundaryUx;
    delete[] s.riemannBoundaryUy;
    delete[] s.riemannBoundaryUz;
    delete[] s.scheduledInletFaceMask;
    delete[] s.sstBoundaryK;
    delete[] s.sstBoundaryKMode;
    delete[] s.sstBoundaryOmega;
    delete[] s.sstBoundaryOmegaMode;
    delete[] s.sstF1;
    delete[] s.sstF2;
    delete[] s.sstPhiRhoK;
    delete[] s.sstPhiRhoOmega;
    delete[] s.sstSourceNumber;
    delete[] s.sstWallDistance;
 }
}
