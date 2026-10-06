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
template<class T> T atomicAdd(T* target,T value) {T old=*target;*target+=value;return old;}
template<class T,class U> T finiteOr(T value,U fallback) {return std::isfinite(value)?value:T(fallback);}
template<class T,class U> T clampMin(T value,U lo) {return value < lo ? T(lo):value;}
template<class T,class U,class V> T clampRange(T value,U lo,V hi) {return value<lo?T(lo):(value>hi?T(hi):value);}
template<class T> T sqr3(T x,T y,T z) {return x*x+y*y+z*z;}
template<class T> bool finiteDevice(T x) {return std::isfinite(x);}
template<class T> bool nonFiniteDevice(T x) {return !std::isfinite(x);}
struct DeviceState {
    int coldWallSolidificationEnabled=1, nCells=1, nFaces=1, nInternalFaces=0;
    int particleCapacity=1, solveParticleTemperature=1, particleGasHeatTransferModelId=1;
    int particleWallHeatTransferEnabled=1, dragModelId=0;
    int particleCount=1, wallBoundCount=1;
    int* particleCountDevice=&particleCount;
    int* wallBoundParticleCountDevice=&wallBoundCount;
    int wallBoundParticleIndex[1]{0}, pCellId[1]{0}, pStuckFaceId[1]{0};
    unsigned char pStatus[1]{1}, pStuck[1]{particleWallTransientRebound};
    unsigned char particleStuckCandidateMask[1]{particleWallSolidifyingDeposition};
    GpuReal pd[1]{GPU_R(1e-4)}, pm[1]{}, pT[1]{GPU_R(3200)};
    GpuReal couplingTgasOld[1]{GPU_R(4200)}, couplingRhoOld[1]{GPU_R(1)};
    GpuReal couplingUxOld[1]{}, couplingUyOld[1]{}, couplingUzOld[1]{};
    GpuReal pux[1]{}, puy[1]{}, puz[1]{}, puxOld[1]{}, puyOld[1]{}, puzOld[1]{};
    GpuReal pTheta[1]{}, gradPx[1]{}, gradPy[1]{}, gradPz[1]{}, thetaDragAlpha[1]{GPU_R(1)};
    GpuReal gasBoundaryT[1]{GPU_R(3200)}, gasBoundaryUx[1]{}, gasBoundaryUy[1]{}, gasBoundaryUz[1]{};
    GpuReal Sfx[1]{GPU_R(1)}, Sfy[1]{}, Sfz[1]{}, magSf[1]{GPU_R(1)};
    GpuReal particleWallContactAreaScale[1]{GPU_R(1)};
    GpuReal particleWallRepresentedContactArea[1]{}, particleWallMaximumCoverage=GPU_R(1);
    GpuReal *particleWallEffusivityByFace=nullptr;
    GpuReal particleDiameterFallback=GPU_R(1e-4), TpMin=GPU_R(1), TpMax=GPU_R(1e5), TgasMin=GPU_R(1);
    GpuReal rhoMin=GPU_R(1e-8), gasMu=GPU_R(1e-4), gasCp=GPU_R(1000), gasPrClamped=GPU_R(.7);
    GpuReal gasPrOneThird=GPU_R(.8879040017), rhoSolid=GPU_R(3200), invRhoSolid=GPU_R(1./3200.);
    GpuReal particleWallReflectionHeatTransferEfficiency=GPU_R(1e-5);
    GpuReal particleWallDepositionHeatTransferEfficiency=GPU_R(1e-5);
    GpuReal particleWallDensityKgM3=GPU_R(1800), particleWallSpecificHeatJkgK=GPU_R(710);
    GpuReal particleWallConductivityWmK=GPU_R(100), particleWallAdhesionEnergyScale=GPU_R(1);
    GpuReal particleWallContactAngleCosine=GPU_R(.5), gravityX=0,gravityY=0,gravityZ=0;
    float pContactMaximumArea[1]{2.2e-8f}, pDepositionArea[1]{}, pContactPeakFraction[1]{.4f};
    float pContactDuration[1]{.01f};
    GpuTime pContactAge[1]{.008}, coldAge[1]{.008};
    float coldH[8]{}, coldRing[8]{}, coldFrozen[1]{}, pCold2DFrozenArea[1]{};
    float* pColdNodeSpecificEnthalpy=coldH;
    float* pColdRingSolidMass=coldRing;
    float* pColdFrozenArea=coldFrozen;
    GpuTime* pColdContactAge=coldAge;
    GpuWallEnergy reflected[1]{}, deposited[1]{};
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
void clearColdWall2DParticleState(DeviceState&,int) {}
#include "GpuWallContactDirectory.cuh"
#include "operators/atomicAddParticleWallEnergyByFace.cuh"
#include "operators/clearColdWallParticleState.cuh"
#if ROUTE_FSH
// Exact alias used around this header by FSH's real backend.
#define pContactAge pTheta
#endif
#include "GpuColdWall1DDevice.cuh"
#if ROUTE_FSH
#undef pContactAge
#endif
#include "GpuThermalParticleRelaxation.cuh"
#include "operators/relaxMobileParticlesToResidentGasKernelStatic.cuh"
#include "GpuThermalParticleFinalization.cuh"
#include "operators/rebuildWallBoundParticleDirectoryKernel.cuh"

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
        try {f();} catch(const DeviceTrap&) {++traps;}
    });
    for(auto& worker:workers) worker.join();
    return traps;
}
GpuTime& mechanicalAge(DeviceState& s) {
#if ROUTE_FSH
    static_assert(UGKWP_GPU_REAL_BITS==64);
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
double meanH(const DeviceState& s) {double h=0;for(auto x:s.coldH) h+=x/8.;return h;}
GpuReal expectedGasIncrement(const DeviceState& s,GpuTime dt) {
    GpuReal dh=0;
    require(coldWallGasSpecificEnthalpyIncrement(GpuReal(meanH(s)),mass(s),s.couplingTgasOld[0],
        s.solveParticleTemperature&&s.particleGasHeatTransferModelId?conductance(s):GPU_R(0),
        dt,s.coldWallSolidificationParameters,dh),"valid independent bulk gas oracle");
    return dh;
}
GpuReal areaAt(const DeviceState& s,GpuTime age) {
    return GpuReal(s.pContactMaximumArea[0])*normalizedKinematicArea(age/GpuTime(s.pContactDuration[0]),GpuReal(s.pContactPeakFraction[0]));
}
void collisionDamageToZeroMidpoint(DeviceState& s,GpuTime dt) {
    const GpuReal start=areaAt(s,mechanicalAge(s));
    const GpuReal midpoint=areaAt(s,mechanicalAge(s)+dt/2);
    const GpuReal wantedDamage=(start+midpoint)/GPU_R(2);
    const auto capillary=evaluateCapillaryDetachmentState(s.pT[0],s.pd[0],s.particleWallAdhesionEnergyScale,s.particleWallContactAngleCosine);
    const GpuReal energy=wantedDamage*capillary.adhesionSpecificEnergyJkg/capillary.equilibriumContactAreaM2;
    finalizeOneThermalizedStuckParticle(s,0,std::sqrt(GPU_R(2)*energy),0,0,0,0,0);
    require(s.pStuck[0]==particleWallTransientRebound,"partial collision damage retains finite contact");
    require(start>s.pDepositionArea[0],"reachable zero-area case has positive area at age0");
    require(midpoint<=s.pDepositionArea[0],"reachable zero-area case has zero midpoint area");
}
void coldStep(DeviceState& s,GpuTime dt) {
    require(launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&s,dt);})==0,"production cold kernel accepted valid state");
}
void genericStep(DeviceState& s,GpuTime dt) {
    threadIdx.x=0;
    // Real production launch order: mobile selection precedes wall finalization.
    relaxMobileParticlesToResidentGasKernelStatic(&s,dt,0);
    relaxWallBoundParticlesToResidentGasKernelStatic(&s,dt,0);
}
void profileCleared(const DeviceState& s) {
    for(int j=0;j<8;++j) {require(s.coldH[j]==0,"detach clears all node enthalpies");require(s.coldRing[j]==0,"detach clears all ring masses");}
    require(s.coldFrozen[0]==0&&s.coldAge[0]==0,"detach clears frozen area and thermal age");
}
void run(const char* mode) {
    DeviceState s;seed(s);
    const GpuTime dt=.001;
    if(!std::strcmp(mode,"zero_midpoint_area") || !std::strcmp(mode,"detach_then_mobile") || !std::strcmp(mode,"gas_off")) {
        collisionDamageToZeroMidpoint(s,dt);
        const bool off=!std::strcmp(mode,"gas_off");
        if(off) s.particleGasHeatTransferModelId=0;
        const double oldH=meanH(s), dh=expectedGasIncrement(s,dt);
        const GpuTime oldThermalAge=s.coldAge[0], oldMechanicalAge=mechanicalAge(s);
        coldStep(s,dt);
        near(meanH(s),oldH+dh,1.1,"zero midpoint area receives full-step uniform gas enthalpy");
        for(auto h:s.coldH) near(h,meanH(s),.6,"gas-only increment is uniform across eight nodes");
        near(s.reflected[0]+s.deposited[0],0,0,"zero-area branch adds no wall heat");
        near(s.coldAge[0],oldThermalAge+dt,1e-14,"thermal age advances active contact interval");
        near(mechanicalAge(s),oldMechanicalAge,0,"thermal kernel does not advance mechanical age");
        const GpuReal afterCold=s.pT[0];
        genericStep(s,dt);
        near(s.pT[0],afterCold,0,"generic finalizer does not add gas again on detachment");
        require(s.pStuck[0]==particleWallMobile,"shrink-damaged contact detaches in generic finalizer");
        profileCleared(s);
        if(!std::strcmp(mode,"detach_then_mobile")) {
            const GpuReal expected=particleTemperatureAfterGasRelaxation(s,s.pT[0],s.couplingTgasOld[0],0,s.pd[0],s.rhoSolid*particleSpecificHeatDevice(s.pT[0]),dt);
            coldStep(s,dt);genericStep(s,dt);
            near(s.pT[0],expected,.002,"next step applies ordinary mobile gas relaxation exactly once");
            profileCleared(s);
        }
    } else if(!std::strcmp(mode,"short_contact_interval") || !std::strcmp(mode,"deposited_positive_area")) {
        const bool deposited=!std::strcmp(mode,"deposited_positive_area");
        if(deposited) {s.pStuck[0]=particleWallDeposited;s.pDepositionArea[0]=s.pContactMaximumArea[0]*.6f;}
        else {mechanicalAge(s)=GpuTime(s.pContactDuration[0])-dt/4;s.coldAge[0]=mechanicalAge(s);}
        const GpuTime active=deposited?dt:GpuTime(s.pContactDuration[0])-mechanicalAge(s);
        const double oldH=meanH(s), dh=expectedGasIncrement(s,dt), oldAge=s.coldAge[0];
        const double physicalMass=mass(s), multiplicity=s.pm[0]/physicalMass;
        std::array<GpuReal,8> referenceH{},referenceRing{};
        for(int j=0;j<8;++j) {referenceH[j]=s.coldH[j];referenceRing[j]=s.coldRing[j];}
        GpuTime referenceAge=s.coldAge[0];GpuReal referenceFrozen=s.coldFrozen[0];
        const GpuReal referenceArea=deposited ? GpuReal(s.pDepositionArea[0]) : areaAt(s,mechanicalAge(s)+active/2);
        const GpuReal referenceVolume=finiteContactPi/GPU_R(6)*s.pd[0]*s.pd[0]*s.pd[0];
        const auto reference=advanceColdWallProfile(referenceH.data(),referenceRing.data(),referenceAge,referenceFrozen,
            s.coldWallSolidificationParameters,referenceVolume,GpuReal(physicalMass),GpuReal(s.pContactMaximumArea[0]),
            referenceArea,GpuTime(s.pContactDuration[0]),GpuReal(s.pContactPeakFraction[0]),active,
            s.gasBoundaryT[0],std::sqrt(s.particleWallDensityKgM3*s.particleWallSpecificHeatJkgK*s.particleWallConductivityWmK),
            deposited?s.particleWallDepositionHeatTransferEfficiency:s.particleWallReflectionHeatTransferEfficiency,
            s.couplingTgasOld[0],conductance(s),dt);
        require(reference.valid,"independent scalar full-gas-duration/active-wall-duration oracle valid");
        coldStep(s,dt);
        const double wall=s.reflected[0]+s.deposited[0];
        near(wall,multiplicity*reference.wallEnergyJ,std::abs(multiplicity*reference.wallEnergyJ)*.003+1e-22,
            "production wall heat equals activeDt scalar reference with full-step gas source");
        near(meanH(s)-oldH+wall/(physicalMass*multiplicity),dh,1.5,"enthalpy plus wall heat equals full physical-step gas input");
        near(s.coldAge[0],oldAge+active,1e-14,"wall thermal age advances only activeDt");
        if(deposited) {require(s.reflected[0]==0,"deposition uses deposited ledger only");require(s.deposited[0]>0,"positive-area deposition transfers wall heat");}
        else {require(s.deposited[0]==0,"transient uses reflected ledger only");require(s.reflected[0]>0,"short active contact retains wall heat");}
    } else if(!std::strcmp(mode,"zero_active_time")) {
        mechanicalAge(s)=s.pContactDuration[0];s.coldAge[0]=mechanicalAge(s);
        const double oldH=meanH(s),dh=expectedGasIncrement(s,dt),oldAge=s.coldAge[0];
        coldStep(s,dt);
        near(meanH(s),oldH+dh,1.1,"zero active contact time retains full-step gas source");
        near(s.coldAge[0],oldAge,0,"zero active time does not advance thermal age");
        near(s.reflected[0],0,0,"zero active time has no wall heat");
    } else if(!std::strcmp(mode,"deposited_zero_area_invalid")) {
        s.pStuck[0]=particleWallDeposited;s.pDepositionArea[0]=0;
        const double oldH=meanH(s),oldTemperature=s.pT[0];
        const int traps=launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&s,dt);});
        require(traps==8,"permanently deposited zero area is rejected before thermal writes",traps,8);
        near(meanH(s),oldH,0,"invalid deposited state preserves enthalpy");
        near(s.pT[0],oldTemperature,0,"invalid deposited state preserves bulk temperature");
        near(s.reflected[0]+s.deposited[0],0,0,"invalid deposited state preserves ledgers");
    } else if(!std::strncmp(mode,"invalid_",8)) {
        collisionDamageToZeroMidpoint(s,dt);
        if(!std::strcmp(mode,"invalid_duration_zero")) s.pContactDuration[0]=0;
        else if(!std::strcmp(mode,"invalid_duration_nan")) s.pContactDuration[0]=std::numeric_limits<float>::quiet_NaN();
        else if(!std::strcmp(mode,"invalid_duration_inf")) s.pContactDuration[0]=std::numeric_limits<float>::infinity();
        else if(!std::strcmp(mode,"invalid_peak_zero")) s.pContactPeakFraction[0]=0;
        else if(!std::strcmp(mode,"invalid_peak_one")) s.pContactPeakFraction[0]=1;
        else if(!std::strcmp(mode,"invalid_peak_nan")) s.pContactPeakFraction[0]=std::numeric_limits<float>::quiet_NaN();
        else if(!std::strcmp(mode,"invalid_damage_negative")) s.pDepositionArea[0]=-1;
        else if(!std::strcmp(mode,"invalid_damage_nan")) s.pDepositionArea[0]=std::numeric_limits<float>::quiet_NaN();
        else if(!std::strcmp(mode,"invalid_damage_inf")) s.pDepositionArea[0]=std::numeric_limits<float>::infinity();
        else require(false,"unknown invalid metadata case");
        std::array<float,8> oldH{},oldRing{};
        std::copy_n(s.coldH,8,oldH.begin());std::copy_n(s.coldRing,8,oldRing.begin());
        const auto temperature=s.pT[0];const auto frozen=s.coldFrozen[0];const auto age=s.coldAge[0];
        const auto contactAge=mechanicalAge(s);const auto wallState=s.pStuck[0];
        const int traps=launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&s,dt);});
        require(traps==8,"invalid contact metadata must reject all eight lanes before thermal publication",traps,8);
        require(!std::memcmp(s.coldH,oldH.data(),sizeof(s.coldH)),"invalid metadata preserves all enthalpy bits");
        require(!std::memcmp(s.coldRing,oldRing.data(),sizeof(s.coldRing)),"invalid metadata preserves all ring bits");
        near(s.pT[0],temperature,0,"invalid metadata preserves bulk temperature");
        near(s.coldAge[0],age,0,"invalid metadata preserves thermal age");
        near(mechanicalAge(s),contactAge,0,"invalid metadata preserves mechanical age");
        near(s.pStuck[0],wallState,0,"invalid metadata preserves mechanical wall state");
        near(s.coldFrozen[0],frozen,0,"invalid metadata preserves frozen footprint");
        near(s.reflected[0]+s.deposited[0],0,0,"invalid metadata preserves wall ledgers");
    } else if(!std::strcmp(mode,"gas_only_invalid_profile")) {
        collisionDamageToZeroMidpoint(s,dt);
        // One invalid non-leader lane must veto publication on every lane.
        s.coldRing[5]=-1;
        std::array<float,8> oldH{},oldRing{};
        std::copy_n(s.coldH,8,oldH.begin());std::copy_n(s.coldRing,8,oldRing.begin());
        const auto temperature=s.pT[0];const auto frozen=s.coldFrozen[0];
        const auto age=s.coldAge[0];
        const int traps=launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&s,dt);});
        require(traps==8,"one invalid gas-only lane rejects the whole eight-lane group",traps,8);
        require(!std::memcmp(s.coldH,oldH.data(),sizeof(s.coldH)),"invalid gas-only profile preserves all enthalpy bits");
        require(!std::memcmp(s.coldRing,oldRing.data(),sizeof(s.coldRing)),"invalid gas-only profile preserves all ring bits");
        near(s.pT[0],temperature,0,"invalid gas-only profile preserves bulk temperature");
        near(s.coldAge[0],age,0,"invalid gas-only profile preserves thermal age");
        near(s.coldFrozen[0],frozen,0,"invalid gas-only profile preserves frozen footprint");
        near(s.reflected[0]+s.deposited[0],0,0,"invalid gas-only profile preserves wall ledgers");
    } else if(!std::strcmp(mode,"completed_frozen_heating") || !std::strcmp(mode,"completed_frozen_cooling")) {
        const bool cooling=!std::strcmp(mode,"completed_frozen_cooling");
        s.pT[0]=GPU_R(2327);seed(s);
        mechanicalAge(s)=s.pContactDuration[0];s.coldAge[0]=mechanicalAge(s);
        s.coldFrozen[0]=s.pContactMaximumArea[0]*.5f;
        const double physicalMass=mass(s);
        if(cooling) s.couplingTgasOld[0]=GPU_R(1000);
        for(int j=0;j<8;++j) s.coldRing[j]=float(physicalMass*(cooling?.1:.5)/8);
        const double oldH=meanH(s),dh=expectedGasIncrement(s,dt),oldAge=s.coldAge[0];
        const float frozen=s.coldFrozen[0],oldRing=s.coldRing[0];
        coldStep(s,dt);
        near(meanH(s),oldH+dh,1.1,"completed frozen contact receives full-step gas enthalpy");
        near(s.coldAge[0],oldAge,0,"completed frozen contact has no further wall exposure age");
        near(s.coldFrozen[0],frozen,0,"gas-only remelting retains frozen footprint");
        near(s.reflected[0]+s.deposited[0],0,0,"completed frozen contact has no wall heat");
        double assigned=0,connected=0,prefix=1;
        for(int j=0;j<8;++j) {
            const double fraction=coldWallSolidFraction(GpuReal(s.coldH[j]),s.coldWallSolidificationParameters);
            prefix=std::min(prefix,fraction);connected+=physicalMass*prefix/8;
            if(cooling) near(s.coldRing[j],oldRing,0,"gas-only cooling does not assign new ring solid mass");
            else require(s.coldRing[j]>=0&&s.coldRing[j]<oldRing,"gas heating only reduces existing ring solid mass");
            assigned+=s.coldRing[j];
        }
        require(assigned<=connected+physicalMass*2e-6,"assigned ring solid mass stays below connected axial solid mass",assigned,connected);
        const GpuReal afterCold=s.pT[0];
        genericStep(s,dt);
        require(s.pStuck[0]==particleWallDeposited,"retained frozen footprint enters permanent deposition");
        near(s.pDepositionArea[0],frozen,0,"deposition area comes from retained frozen footprint");
        near(s.pT[0],afterCold,0,"deposit transition does not duplicate gas heating");
    } else if(!std::strcmp(mode,"gas_off_profile_preserved")) {
        s.pT[0]=GPU_R(2200);seed(s);
        s.particleGasHeatTransferModelId=0;
        s.coldFrozen[0]=s.pContactMaximumArea[0]*.001f;
        std::array<float,8> oldH{},oldRing{};
        for(int j=0;j<8;++j) {
            s.coldH[j]=float(coldWallSpecificEnthalpyJkg(GPU_R(2100)+GPU_R(j)*GPU_R(10),s.coldWallSolidificationParameters));
            s.coldRing[j]=float(mass(s)*.01/8);
            oldH[j]=s.coldH[j];oldRing[j]=s.coldRing[j];
        }
        collisionDamageToZeroMidpoint(s,dt);
        const auto frozen=s.coldFrozen[0];const auto age=s.coldAge[0];
        coldStep(s,dt);
        for(int j=0;j<8;++j) {
            near(s.coldH[j],oldH[j],0,"gas-off fallback preserves each nonuniform node exactly");
            near(s.coldRing[j],oldRing[j],0,"gas-off fallback preserves each assigned ring exactly");
        }
        near(s.coldFrozen[0],frozen,0,"gas-off fallback preserves frozen footprint exactly");
        near(s.coldAge[0],age+dt,1e-14,"gas-off fallback still advances active thermal age");
        near(s.reflected[0]+s.deposited[0],0,0,"gas-off zero-area fallback produces no wall heat");
    } else if(!std::strcmp(mode,"collision_released_before_thermal")) {
        finalizeOneThermalizedStuckParticle(s,0,GPU_R(100),0,0,0,0,0);
        require(s.pStuck[0]==particleWallMobile,"collision finalizer releases particle before thermal dispatch");
        profileCleared(s);
        const GpuReal old=s.pT[0];
        const GpuReal re=s.couplingRhoOld[0]*s.pd[0]*std::sqrt(sqr3(s.pux[0],s.puy[0],s.puz[0]))/s.gasMu;
        const GpuReal expected=particleTemperatureAfterGasRelaxation(s,old,s.couplingTgasOld[0],re,s.pd[0],s.rhoSolid*particleSpecificHeatDevice(old),dt);
        coldStep(s,dt);near(s.pT[0],old,0,"cold kernel skips collision-released mobile particle");
        genericStep(s,dt);near(s.pT[0],expected,.002,"collision-released particle receives normal mobile gas exactly once");
        profileCleared(s);near(s.reflected[0]+s.deposited[0],0,0,"collision-released particle has no wall heat");
    } else require(false,"unknown test mode");
}
int main(int argc,char** argv) {
    require(argc==2,"one test mode required");
    try {run(argv[1]);} catch(const DeviceTrap&) {require(false,"unexpected scalar production trap");}
    std::printf("PASS %s (CPU production-header execution, %d-bit)\n",argv[1],UGKWP_GPU_REAL_BITS);
}
