#ifndef CHMT_GPU_GASWINDOWSTATE_H
#define CHMT_GPU_GASWINDOWSTATE_H
#include "gpu/Backend.H"
#include "gpu/BackendResources.H"
#include <memory>
namespace chmt {
struct PendingGasWindow {
    WallProgram program;
    GasWindowLimits limits;
    bool staticGeometry=false;
    std::size_t historyBytes=0;
    Real lastGasDt=0;
    IntervalHistory history;
    MaterialDonorReserve reserve;
    std::unique_ptr<StateBuffers> current{new StateBuffers};
    HostMesh gasMesh,solidMesh;
    SurfaceMesh surface;
    std::array<HostStageGeometry,2> gasStages,solidStages;
    Budget budget;
    Real time=0,nextGasDt=0;
    std::uint64_t microSequence=0,rejectedTrials=0;
    Buffer<WallFaceSample> wall;
    Buffer<GasPrimitive> drive;
    Buffer<GasGradient> driveGradient;
    Buffer<Vec3> traction;
};
}
#endif
