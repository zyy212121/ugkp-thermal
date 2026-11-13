// Checkpoint 2 host fixture. Executes real production operators serially, not CUDA.
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

struct State : ugkwp::GasStateView<GpuReal,GpuTime,ugkwp::SstCoefficients,
                                 GpuEnergy,GpuGeometry,GpuWallEnergy> {
    ugkwp::GasSpeciesState<GpuReal,2,double> gasSpecies;
};
static_assert(std::is_same<decltype(State{}.rho),float*>::value,"bulk storage must be float");
static_assert(std::is_same<decltype(State{}.p),float*>::value,"primitive storage must be float");
static_assert(std::is_same<decltype(State{}.rhoE),double*>::value,"authoritative energy stays double");
static_assert(std::is_same<decltype(State{}.rhoENext),double*>::value,"RK energy stays double");
static_assert(std::is_same<decltype(State{}.V),double*>::value,"geometry stays double");
static_assert(std::is_same<decltype(State{}.pressureScheduleTimes),double*>::value,"time stays double");
static_assert(std::is_same<decltype(State{}.gasSpecies.thermo.coefficients),const double*>::value,"immutable thermo tables stay double");

void require(bool value,const char* text) {
    if(!value){std::fprintf(stderr,"mixed EOS/RK: %s\n",text);std::abort();}
}
template<class T> void allocateField(T&) {}
template<class T> void allocateField(T*& field) {field=new T[8]{};}
template<class T> void releaseField(T&) {}
template<class T> void releaseField(T*& field) {delete[] field;}
struct Fixture {
    State s;
    ugkp::diagnostics::DeviceContext diagnostics{};
    int cellStatus[1]{};
    Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) allocateField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        s.nCells=1;s.nFaces=s.nInternalFaces=0;s.diagnostics=&diagnostics;
        s.gasSpecies.cellStatus=cellStatus;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
        s.gammaGas=GpuReal(1.4);s.Rgas=GpuReal(287);s.gasCp=GpuReal(1004.5);
        s.rhoMin=GpuReal(1e-12);s.TgasMin=GpuReal(1e-6);s.V[0]=1.;s.cellLength[0]=1.;
        s.rho[0]=GpuReal(2);s.rhoE[0]=1e9;s.Tgas[0]=300;
        blockDim.x=1;threadIdx.x=blockIdx.x=0;
    }
    ~Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) releaseField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    }
};
bool close(double a,double b,double tolerance) {
    return std::abs(a-b)<=tolerance*std::max({1.,std::abs(a),std::abs(b)});
}
void authorityAndRk() {
    Fixture f;auto& s=f.s;
    s.rhoUx[0]=GpuReal(.00048828125);
    s.rhoE[0]=std::nextafter(1e9,2e9);
    const double authority=s.rhoE[0];
    recoverGasPrimitivesKernel(&s);
    require(s.rhoE[0]==authority,"valid EOS inventory was narrowed or rewritten");
    require(std::isfinite(s.p[0])&&std::isfinite(s.Tgas[0]),"valid EOS primitives are not finite");
    saveGasConservativeStateKernel(&s);
    for(int stage=0;stage<1000;++stage)blendGasConservativeStateKernel(&s,1./3.,2./3.);
    require(s.rhoE[0]==authority,"identity RK blend rewrote authoritative energy");
    s.rhoE[0]=authority+32.;
    const double expected=.1*authority+.9*s.rhoE[0];
    blendGasConservativeStateKernel(&s,.1,.9);
    require(close(s.rhoE[0],expected,8*std::numeric_limits<double>::epsilon()),"RK energy weights were narrowed to float");
    require(!f.diagnostics.fatalClaim,"valid RK/EOS path reported fatal");
    std::cout<<"authority="<<authority<<" RK energy error="<<s.rhoE[0]-expected<<'\n';
}
void highKineticRecovery(bool moving) {
    double lastTemperature=-1;
    for(double thermal:{1e6,2e6,4e6}) {
        Fixture f;auto& s=f.s;
        s.rho[0]=GpuReal(1.1);s.rhoUx[0]=GpuReal(19999000.875);
        const double density=s.rho[0],momentum=s.rhoUx[0],velocity=momentum/density;
        const double kinetic=ugkwp::gasLegacyKineticEnergy(density,velocity,0.,0.);
        s.rhoE[0]=kinetic+thermal;
        if(moving) {
            s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=s.V;s.gasGeometry.newVolume=s.V;
            s.gasGeometry.interval=1e-6;
        }
        const double authority=s.rhoE[0];
        const double expected=(authority-kinetic)*(double(s.gammaGas)-1)/(density*s.Rgas);
        for(int pass=0;pass<10;++pass)recoverGasPrimitivesKernel(&s);
        require(!f.diagnostics.fatalClaim,"finite high-kinetic state reported fatal");
        require(std::isfinite(s.Tgas[0])&&s.Tgas[0]>0,"high-kinetic temperature is invalid");
        require(s.rhoE[0]==authority,"high-kinetic recovery rewrote valid energy");
        require(close(s.Tgas[0],expected,2e-5),"high-kinetic temperature lost internal energy");
        require(s.Tgas[0]>lastTemperature,"temperature did not increase with internal energy");
        lastTemperature=s.Tgas[0];
        std::cout<<"moving="<<moving<<" thermal="<<thermal<<" T="<<s.Tgas[0]
                 <<" relative T error="<<std::abs(s.Tgas[0]-expected)/expected<<'\n';
    }
}
void existingRepairsAndInvalidInput() {
    Fixture f;auto& s=f.s;
    s.rhoMin=GpuReal(.5);s.rho[0]=GpuReal(.25);s.rhoUx[0]=GpuReal(3);
    s.rhoE[0]=-1;s.TgasMin=GpuReal(300);
    recoverGasPrimitivesKernel(&s);
    require(!f.diagnostics.fatalClaim,"existing finite density/energy repair was rejected");
    require(s.rho[0]==GpuReal(.5)&&s.rhoUx[0]==GpuReal(3),"existing density repair changed momentum");
    require(close(s.Tgas[0],300.,2e-5)&&std::isfinite(s.rhoE[0]),"existing energy floor repair failed");
    const double thermal=ugkwp::gasLegacyMinimumInternalEnergy(
        double(s.rhoMin),double(s.rho[0]),double(s.Rgas),double(s.TgasMin),double(s.gammaGas),double(OfSmall));
    require(close(s.rhoE[0],9.+thermal,2e-6),"repaired energy is inconsistent with the existing floor");
    Fixture bad;bad.s.rhoE[0]=std::numeric_limits<double>::quiet_NaN();
    recoverGasPrimitivesKernel(&bad.s);
    require(bad.diagnostics.fatalClaim&&std::isnan(bad.s.rhoE[0]),"existing nonfinite diagnostic was hidden");
    require(bad.diagnostics.firstFatal.code==ugkp::diagnostics::Code::NonFinite,"nonfinite classification changed");
}
int main() {
    std::cout<<std::setprecision(17);
    authorityAndRk();highKineticRecovery(false);highKineticRecovery(true);existingRepairsAndInvalidInput();
    std::cout<<"PASS mixed bulk/energy/geometry/time EOS and RK\n";
}
