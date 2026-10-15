// CPU-only eight-lane execution of real production headers, not a GPU emulator.
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <string>
#include "GpuColdWall2DSolidification.H"
#if !ROUTE_FSH
#define GPU_PAIRED_PARTICLE_HEAT 1
#endif
#include "GpuColdWallSolidification.H"
#include "GpuFiniteWallContact.H"
#include "GpuCapillaryDetachment.H"
#include "GpuParticleWallContactHeat.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
#if ROUTE_FSH
#define GPU_CONTACT_AGE(s,i) s.pTheta[i]
#define GPU_CONTACT_TIME_ZERO 0.0
#define GPU_RESET_CONTACT_AGE(s,i)
#define GPU_CONTACT_SEPARATE_AGE 0
#define GPU_THERMAL_RELAX_NATIVE_ORDER 1
#else
#define GPU_CONTACT_AGE(s,i) s.pContactAge[i]
#define GPU_CONTACT_TIME_ZERO GpuTime(0)
#define GPU_RESET_CONTACT_AGE(s,i) s.pContactAge[i] = GpuTime(0);
#define GPU_CONTACT_SEPARATE_AGE 1
#define GPU_THERMAL_RELAX_NATIVE_ORDER 0
#endif
using std::max;
using std::min;
using namespace Foam::gpuThermal;
struct Index { int x; };
thread_local Index threadIdx{0};
Index blockDim{32}, blockIdx{0}, gridDim{1};
std::barrier<> lanes(8);
long double laneValues[8];
struct DeviceTrap {};
void __syncwarp(unsigned){lanes.arrive_and_wait();}
#define asm(...) throw DeviceTrap{}
template<class T> T shuffle(T value,int target) {
    laneValues[threadIdx.x] = value;
    lanes.arrive_and_wait();
    const T result = static_cast<T>(laneValues[target]);
    lanes.arrive_and_wait();
    return result;
}
template<class T> T __shfl_sync(unsigned,T value,int target,int) {return shuffle(value,target);}
template<class T> T __shfl_up_sync(unsigned,T value,int offset,int) {
    return shuffle(value,threadIdx.x >= offset ? threadIdx.x-offset : threadIdx.x);
}
template<class T> T __shfl_down_sync(unsigned,T value,int offset,int) {
    return shuffle(value,threadIdx.x+offset < 8 ? threadIdx.x+offset : threadIdx.x);
}
unsigned __ballot_sync(unsigned,bool value) {
    laneValues[threadIdx.x] = value;
    lanes.arrive_and_wait();
    unsigned result=0;
    for(int i=0;i<8;++i) if(laneValues[i]) result |= 1u << i;
    lanes.arrive_and_wait();
    return result;
}
bool __all_sync(unsigned mask,bool value) {return __ballot_sync(mask,value)==mask;}
int __ffs(unsigned value) {return __builtin_ffs(value);}
template<class T> T atomicAdd(T* target,T value) {if(!target)throw DeviceTrap{};T old=*target;*target+=value;return old;}
template<class T,class U> T finiteOr(T value,U fallback) {return std::isfinite(value)?value:T(fallback);}
template<class T,class U> T clampMin(T value,U lo) {return value < lo ? T(lo):value;}
template<class T,class U,class V> T clampRange(T value,U lo,V hi) {return value<lo?T(lo):(value>hi?T(hi):value);}
template<class T> T sqr3(T x,T y,T z) {return x*x+y*y+z*z;}
template<class T> bool finiteDevice(T x) {return std::isfinite(x);}
template<class T> bool nonFiniteDevice(T x) {return !std::isfinite(x);}
struct DeviceState {
 GpuReal epsGPrev[3]{GPU_R(.9)},V[3]{GPU_R(1)},rho[3]{GPU_R(1)},rhoUx[3]{},rhoUy[3]{},rhoUz[3]{};
 GpuReal Tgas[3]{GPU_R(4200)},p[3]{},Ux[3]{},Uy[3]{},Uz[3]{};
 GpuReal Rgas=GPU_R(287),gammaGas=GPU_R(1.4);
 double rhoE[3]{3.e6},rhoENext[3]{};
 unsigned char* compactPStuck=pStuck;int* compactPStuckFaceId=pStuckFaceId;GpuReal* compactPT=pT;
 float* compactPColdNodeSpecificEnthalpy=coldH;float* compactPCold2DNodeSpecificEnthalpy=cold2DH;
 double* compactPThermalEnthalpyResidual=pThermalEnthalpyResidual;

 double pThermalEnthalpyResidual[3]{}; double heatBudget[3]{}; double heatCoefficient[3]{1.}; double* particleHeatBudget=heatBudget; double* particleHeatCoefficient=heatCoefficient;
 int coldWall2DEnabled=1;
 float cold2DH[192]{}; GpuTime cold2DAge[24]{};
 float* pCold2DNodeSpecificEnthalpy=cold2DH;
 GpuTime* pCold2DRingContactAge=cold2DAge;
    int coldWallSolidificationEnabled=1, nCells=1, nFaces=1, nInternalFaces=0;
    int particleCapacity=1, solveParticleTemperature=1, particleGasHeatTransferModelId=1;
    int particleWallHeatTransferEnabled=1, dragModelId=0;
    int particleCount=1, wallBoundCount=1;
    int* particleCountDevice=&particleCount;
    int* wallBoundParticleCountDevice=&wallBoundCount;
    int wallBoundParticleIndex[3]{0}, pCellId[3]{0}, pStuckFaceId[3]{0};
    unsigned char pStatus[3]{1}, pStuck[3]{particleWallTransientRebound};
    unsigned char particleStuckCandidateMask[3]{particleWallSolidifyingDeposition};
    GpuReal pd[3]{GPU_R(1e-4)}, pm[3]{}, pT[3]{GPU_R(3200)};
    GpuReal couplingTgasOld[3]{GPU_R(4200)}, couplingRhoOld[3]{GPU_R(1)};
    GpuReal couplingUxOld[3]{}, couplingUyOld[3]{}, couplingUzOld[3]{};
    GpuReal pux[3]{}, puy[3]{}, puz[3]{}, puxOld[3]{}, puyOld[3]{}, puzOld[3]{};
    GpuReal pTheta[3]{}, gradPx[3]{}, gradPy[3]{}, gradPz[3]{}, thetaDragAlpha[3]{GPU_R(1)};
    GpuReal gasBoundaryT[3]{GPU_R(3200)}, gasBoundaryUx[3]{}, gasBoundaryUy[3]{}, gasBoundaryUz[3]{};
    GpuReal Sfx[3]{GPU_R(1)}, Sfy[3]{}, Sfz[3]{}, magSf[3]{GPU_R(1)};
    GpuReal particleWallContactAreaScale[3]{GPU_R(1)};
    GpuReal particleWallRepresentedContactArea[3]{}, particleWallMaximumCoverage=GPU_R(1);
    GpuReal *particleWallEffusivityByFace=nullptr;
    GpuReal particleDiameterFallback=GPU_R(1e-4), TpMin=GPU_R(1), TpMax=GPU_R(1e5), TgasMin=GPU_R(1);
    GpuReal rhoMin=GPU_R(1e-8), gasMu=GPU_R(1e-4), gasCp=GPU_R(1000), gasPrClamped=GPU_R(.7);
    GpuReal gasPrOneThird=GPU_R(.8879040017), rhoSolid=GPU_R(3200), invRhoSolid=GPU_R(1./3200.);
    GpuReal particleWallReflectionHeatTransferEfficiency=GPU_R(1e-5);
    GpuReal particleWallDepositionHeatTransferEfficiency=GPU_R(1e-5);
    GpuReal particleWallDensityKgM3=GPU_R(1800), particleWallSpecificHeatJkgK=GPU_R(710);
    GpuReal particleWallConductivityWmK=GPU_R(100), particleWallAdhesionEnergyScale=GPU_R(1);
    GpuReal particleWallContactAngleCosine=GPU_R(.5), gravityX=0,gravityY=0,gravityZ=0;
    float pContactMaximumArea[3]{2.2e-8f}, pDepositionArea[3]{}, pContactPeakFraction[3]{.4f};
    float pContactDuration[3]{.01f};
#if !ROUTE_FSH
    GpuTime pContactAge[3]{.008};
#endif
    GpuTime coldAge[3]{.008};
    float coldH[24]{}, coldRing[24]{}, coldFrozen[3]{}, pCold2DFrozenArea[3]{};
    float* pColdNodeSpecificEnthalpy=coldH;
    float* pColdRingSolidMass=coldRing;
    float* pColdFrozenArea=coldFrozen;
    GpuTime* pColdContactAge=coldAge;
    GpuWallEnergy reflected[3]{}, deposited[3]{};
    GpuWallEnergy* particleWallReflectedEnergy=reflected;
    GpuWallEnergy* particleWallDepositedEnergy=deposited;
    ColdWallSolidificationParameters coldWallSolidificationParameters{
        GPU_R(2327),GPU_R(20),GPU_R(1.16e6),GPU_R(3990),GPU_R(1273),GPU_R(5.9),
        GPU_R(.25),GPU_R(0),0,12};
};
#include "operators/particleSpecificHeatDevice.cuh"
// Exact production conductivity function extracted by the fixture generator;
// its owning header also contains unrelated large gas-boundary implementations.
#include "route_conductivity.cuh"
// Drag is explicitly disabled in these thermal tests. Fail if accidentally used.
bool gasDragModelActive(int model) {return model!=0;}
GpuReal solidEpsFromMomentDevice(const DeviceState&,int) {throw DeviceTrap{};}
template<class T> GpuReal dragInverseTimeDevice(const DeviceState&,GpuReal,GpuReal,GpuReal,GpuReal,T) {throw DeviceTrap{};}
void clearColdWall2DParticleState(DeviceState&,int);
#include "GpuWallContactDirectory.cuh"
#include "operators/atomicAddParticleWallEnergyByFace.cuh"
#include "operators/clearColdWallParticleState.cuh"
#if ROUTE_FSH
// Exact alias used around this header by FSH's real backend.
#define pContactAge pTheta
#endif
#include "GpuParticleThermalCarry.H"
#include "GpuPairedParticleHeat.H"
#include "GpuColdWallThermalCarry.cuh"
#include "GpuParticleCaloricDevice.cuh"
constexpr GpuReal OfSmall=GPU_R(2.22044604925031308085e-16);
#include "actual_gas_recovery.cuh"
#define publishPairedParticleGasHeat actualPublishPairedParticleGasHeat
#include "GpuPairedParticleHeat.cuh"
#include "paired_heat_reference.cuh"
#undef publishPairedParticleGasHeat
std::atomic<int> pairedCallbackCalls{0};
void publishPairedParticleGasHeat(DeviceState& s,int i,GpuTime dt) {
 ++pairedCallbackCalls;actualPublishPairedParticleGasHeat(s,i,dt);
}
#include "GpuColdWall1DDevice.cuh"
#if ROUTE_FSH
#undef pContactAge
#endif

#include <vector>
std::vector<double> observedStepTimes;
std::vector<double> observedTrace[8];

#if ROUTE_FSH
#define pContactAge pTheta
#endif
#include "actual_2d_group.cuh"
#if ROUTE_FSH
#undef pContactAge
#endif
#include "GpuThermalParticleFinalization.cuh"
#include "GpuThermalParticleRelaxation.cuh"
#include "GpuContactAgeStorage.cuh"
#include "operators/accumulateParticleWallRepresentedContactAreaKernel.cuh"
#include "GpuParticleThermalConsumerDevice.cuh"
void require(bool ok,const char* message,double got=0,double expected=0) {
    if(!ok) {std::fprintf(stderr,"FAIL %s: got %.17g expected %.17g\n",message,got,expected);std::exit(1);}
}
void near(double got,double expected,double abs,const char* message) {
    require(std::isfinite(got)&&std::abs(got-expected)<=abs,message,got,expected);
}
template<class F> int launch8(F f) {
    std::atomic<int> traps{0};
    std::thread workers[8];
    for(int lane=0;lane<8;++lane) workers[lane]=std::thread([&,lane]{
        threadIdx.x=lane;
        try {f();} catch(const DeviceTrap&) {++traps;lanes.arrive_and_drop();}
    });
    for(auto& worker:workers) worker.join();
    return traps;
}
GpuTime& mechanicalAge(DeviceState& s) {
#if ROUTE_FSH
    return s.pTheta[0];
#else
    return s.pContactAge[0];
#endif
}
const GpuTime& mechanicalAge(const DeviceState& s) {
#if ROUTE_FSH
    return s.pTheta[0];
#else
    return s.pContactAge[0];
#endif
}
void seed(DeviceState& s) {
    mechanicalAge(s)=.008;
    initialiseColdWallParticleState(s,0,s.pT[0]);
    s.pColdContactAge[0]=mechanicalAge(s);
    const GpuReal volume=finiteContactPi/GPU_R(6)*s.pd[0]*s.pd[0]*s.pd[0];
    s.pm[0]=GPU_R(7)*liquidAluminaProperties(s.pT[0]).densityKgM3*volume;
}
GpuReal mass(const DeviceState& s) {
    return liquidAluminaProperties(s.pT[0]).densityKgM3
      *(finiteContactPi/GPU_R(6))*s.pd[0]*s.pd[0]*s.pd[0];
}
GpuReal conductance(const DeviceState& s) {
    const GpuReal cp=particleSpecificHeatDevice(s.pT[0]);
    const GpuReal re=s.couplingRhoOld[0]*s.pd[0]*std::sqrt(sqr3(s.couplingUxOld[0],s.couplingUyOld[0],s.couplingUzOld[0]))/s.gasMu;
    const GpuReal nu=GPU_R(2)+GPU_R(.6)*std::sqrt(re)*s.gasPrOneThird;
    const GpuReal rate=GPU_R(6)*nu*molecularGasConductivity(s)/(s.rhoSolid*cp*s.pd[0]*s.pd[0]+GPU_TINY(1e-300));
    return rate*mass(s)*cp;
}

#include "GpuColdWallThermalCarry.cuh"
#include <random>
ugkwp::ParticleCaloric law(const DeviceState& s) {
 return ugkwp::coldWallCarryCaloric(s.coldWallSolidificationParameters);
}
double inventory(const float* h,int n,double r) {double a=r;for(int j=0;j<n;++j)a+=double(h[j])/n;return a;}
