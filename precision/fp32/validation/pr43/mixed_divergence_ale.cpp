// Checkpoint 2A fixture: real mixed storage and production divergence/ALE operator.
// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "GpuOperatorContract.cuh"
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
#include "gasTransport/GasStateView.H"
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


#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <string>

struct State : ugkwp::GasStateView<GpuReal,GpuTime,ugkwp::SstCoefficients,
                                 GpuEnergy,GpuGeometry,GpuWallEnergy> {
    ugkwp::GasSpeciesState<GpuReal,2,double> gasSpecies;
};
static_assert(std::is_same<decltype(State{}.rho),float*>::value,"bulk must be float");
static_assert(std::is_same<decltype(State{}.gasPhiRhoE),float*>::value,"instantaneous flux must be float");
static_assert(std::is_same<decltype(State{}.rhoE),double*>::value,"energy must be double");
static_assert(std::is_same<decltype(State{}.V),double*>::value,"geometry must be double");
static_assert(std::is_same<decltype(State{}.gasGeometry.interval),double>::value,"time must be double");

void require(bool valid,const char* message) {
    if(!valid){std::fprintf(stderr,"mixed divergence/ALE: %s\n",message);std::abort();}
}
template<class T> void allocateField(T&) {}
template<class T> void allocateField(T*& field){field=new T[8]{};}
template<class T> void releaseField(T&) {}
template<class T> void releaseField(T*& field){delete[] field;}
struct Fixture {
    State s;
    ugkp::diagnostics::DeviceContext diagnostics{};
    int cellStatus[1]{},faceStatus[1]{};
    double oldVolume[1]{},newVolume[1]{},sweep[1]{};
    Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) allocateField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        s.nCells=1;s.nFaces=1;s.nInternalFaces=0;s.diagnostics=&diagnostics;
        s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
        s.gasSpecies.cellStatus=cellStatus;s.gasSpecies.faceStatus=faceStatus;
        s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=1;s.cellFaceId[0]=0;
        s.faceOwner[0]=0;s.faceNeighbour[0]=-1;s.facePeriodicPair[0]=-1;
        s.magSf[0]=s.Sfx[0]=1.;
        s.gammaGas=GpuReal(1.4);s.Rgas=GpuReal(287);
        s.rhoMin=GpuReal(1e-12);s.TgasMin=GpuReal(1e-6);
        s.V[0]=1.;s.rho[0]=GpuReal(2);s.rhoE[0]=1e9;
        blockDim.x=1;threadIdx.x=blockIdx.x=0;
    }
    void moving(double oldV,double newV,double dt) {
        oldVolume[0]=oldV;newVolume[0]=newV;sweep[0]=newV-oldV;
        s.gasGeometry.enabled=true;
        s.gasGeometry.oldVolume=oldVolume;s.gasGeometry.newVolume=newVolume;
        s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=dt;
    }
    ~Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) releaseField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    }
};

// Instantiate the other changed geometry consumers during compilation only.
// Their numerical audits remain in the parent checkpoint's existing SST/CFL set.
void instantiateMixedSstAndCourant(State* s,GpuTime dt) {
    applySstFluxAndSourceKernel(s,dt);
    computeGasConvectiveCourantByCellKernel(s,dt);
}

// Independent integrated-balance oracle. Production performs its density update;
// this oracle forms inventories first in long double, without copying its scale
// or densityRatio intermediate. Assertions protect the deliberately FP64 channels,
// not exact equality of ordinary FP32 state across different solver executions.
void runCase(const std::string& name,bool strict) {
    Fixture f;auto& s=f.s;
    double dt=.123456789012345,oldV=1.,newV=1.;
    bool moving=false;
    if(name=="static_zero") {
        oldV=newV=1.00000003125;
        s.rhoE[0]=std::nextafter(1e16,2e16);
    } else if(name=="moving_zero") {
        moving=true;oldV=3.00000009375;newV=7.00000015625;
        s.rhoE[0]=std::nextafter(1e16,2e16);
    } else if(name=="small_motion") {
        moving=true;oldV=1.;newV=1.+std::ldexp(1.,-40);
        s.rhoE[0]=std::nextafter(1e16,2e16);
    } else if(name=="static_dt_volume") {
        oldV=newV=3.00000003125;
        s.rhoE[0]=1e9+.125;s.gasPhiRhoE[0]=GpuReal(9876543.);
    } else if(name=="moving_dt_volume") {
        moving=true;oldV=3.00000003125;newV=3.00000012345;
        s.rhoE[0]=1e9+.125;s.gasPhiRhoE[0]=GpuReal(9876543.);
    } else if(name=="subfloat_dt_volume") {
        oldV=newV=1.;dt=std::ldexp(1.,-150);
        s.rhoE[0]=1.;s.gasPhiRhoE[0]=-std::ldexp(GpuReal(1),120);
    } else {
        require(false,"unknown fixture case");
    }
    s.V[0]=oldV;
    if(moving)f.moving(oldV,newV,dt);
    const double before=s.rhoE[0],outward=s.gasPhiRhoE[0];
    const long double inventory=static_cast<long double>(before)*oldV
        -static_cast<long double>(dt)*outward;
    const long double oracle=inventory/newV;
    applyGasFluxDivergenceByCellKernel(&s,dt,strict);
    require(!f.diagnostics.fatalClaim&&f.cellStatus[0]==0&&f.faceStatus[0]==0,
        "finite mixed input reported a fatal or invalid status");
    require(std::isfinite(s.rhoE[0])&&std::isfinite(s.rho[0]),
        "divergence produced nonfinite state");
    const long double error=std::abs(static_cast<long double>(s.rhoE[0])-oracle);
    const long double scale=std::max(1.L,std::abs(oracle));
    const long double tolerance=32*std::numeric_limits<double>::epsilon()*scale;
    require(error<=tolerance,"geometry/time was narrowed before the double energy update");
    const long double inventoryResidual=static_cast<long double>(s.rhoE[0])*newV-inventory;
    require(std::abs(inventoryResidual)<=
        32*std::numeric_limits<double>::epsilon()*std::max(1.L,std::abs(inventory)),
        "double integrated energy balance is lost");
    if(name=="static_zero")require(s.rhoE[0]==before,"zero flux changed authoritative energy");
    if(name=="small_motion")require(s.rhoE[0]<before,"sub-float volume change was discarded");
    if(name=="subfloat_dt_volume")require(s.rhoE[0]>before,"sub-float time/volume scale was discarded");
    const GpuReal expectedDensity=GpuReal(2.*(oldV/newV));
    require(s.rho[0]==expectedDensity,"ordinary density did not publish the rounded geometric update");
    std::cout<<name<<" strict="<<strict<<" energy_error="<<double(error)
             <<" relative_error="<<double(error/scale)
             <<" inventory_residual="<<double(inventoryResidual)<<'\n';
}
int main(int argc,char** argv) {
    require(argc==2,"one named case is required");
    std::cout<<std::setprecision(17);
    runCase(argv[1],false);runCase(argv[1],true);
}
