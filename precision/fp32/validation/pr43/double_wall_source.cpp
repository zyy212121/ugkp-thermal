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


#include "gasTransport/GasBoundaryLayerModelState.H"
struct SourceState : ugkwp::GasStateView<GpuReal,GpuTime,ugkwp::SstCoefficients,
                                      GpuEnergy,GpuGeometry,GpuWallEnergy> {
    ugkwp::GasSpeciesState<GpuReal,2,double> gasSpecies;
    ugkwp::GasBoundaryLayerModelState<double,2> gasBoundaryLayerModel;
};
int main() {
    static_assert(std::is_same<decltype(SourceState{}.rho),float*>::value,"float bulk");
    SourceState s;
    int slot=0,active=1,status=0;
    ugkwp::GasBoundaryLayerExchange<double> exchange;exchange.ready=true;
    ugkwp::GasBoundaryLayerSstClosure<double> closure;
    s.nCells=1;s.gasBoundaryLayer.enabled=true;s.gasBoundaryLayer.count=1;
    s.gasBoundaryLayer.exchange=&exchange;s.gasBoundaryLayer.sst=&closure;s.gasBoundaryLayer.status=&status;
    auto& d=s.gasBoundaryLayerModel.conservative;
    d.enabled=true;d.cellSlot=&slot;d.active=&active;
    double k=std::nextafter(1e12,2e12),omega=std::nextafter(1e-9,2e-9);
    d.sourceK=&k;d.sourceOmega=&omega;
    double source=0,omegaSource=0;
    const bool ok=sstSourcesForCell(s,0,GpuReal(0),source,omegaSource);
    if(!ok||source!=k||omegaSource!=omega) {
        std::fprintf(stderr,"active double wall sources were narrowed by the bulk interface\n");
        return 1;
    }
    // The normal float consumer remains the same interface and rounding.
    GpuReal bulkSource=0,bulkOmega=0;
    if(!sstSourcesForCell(s,0,GpuReal(0),bulkSource,bulkOmega)
        ||bulkSource!=GpuReal(k)||bulkOmega!=GpuReal(omega))return 2;
    std::cout<<std::setprecision(17)<<"PASS double wall source="<<source
             <<" omega="<<omegaSource<<" retained beyond bulk float precision\n";
}
