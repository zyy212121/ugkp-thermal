#ifndef CHMT_GPU_MATERIALSURFACEKERNELS_CUH
#define CHMT_GPU_MATERIALSURFACEKERNELS_CUH
#include "ablation/DrySurfaceInterfaceMath.H"
namespace chmt {
struct MaterialSurfaceStatus {
    bool success=false,numericalFailure=true;
    DeviceStatus device{};
    ugkwp::gaswall::WallStatus wall{};
};
// Borrowed device views. Context chemistry and quadrature in inputs must refer
// to device-owned storage. Each worker gets a separate aligned wall workspace;
// its strided faces are serial, while statuses and outputs remain face-sized.
struct MaterialSurfaceView {
    int count=0;
    const SurfacePhysicsInput* inputs=nullptr;
    const int* solidCells=nullptr;
    const SolidQ* solid=nullptr;
    const Real* volumes=nullptr;
    const SolidQ* materialRate=nullptr;
    const unsigned char* active=nullptr; // null means every face is active
    SurfacePhysicsResult* physicsOut=nullptr;
    WallFaceSample* wallOut=nullptr;
    MaterialSurfaceStatus* status=nullptr;
    void* wallWorkspaces=nullptr;
    std::size_t workspaceStride=0;
    int workspaceCapacity=0,workers=0;
};
__global__ void evaluateMaterialSurfaceKernel(MaterialSurfaceView view,const PhysicsConfig* physics) {
    const int worker=blockIdx.x*blockDim.x+threadIdx.x;
    if(worker>=view.workers)return;
    for(int face=worker;face<view.count;face+=view.workers){
        auto& status=view.status[face];status=MaterialSurfaceStatus{};
        if(view.active&&!view.active[face]){
            view.physicsOut[face]=SurfacePhysicsResult{};view.wallOut[face]=WallFaceSample{};
            status.success=true;continue;
        }
        SurfacePhysicsInput input=view.inputs[face];
        const int cell=view.solidCells[face];input.hasSolid=cell>=0;
        if(input.hasSolid){
            input.solid=input.solidBase=view.solid[cell];input.solidVolume=view.volumes[cell];
            input.solidLocalRate=view.materialRate?view.materialRate[cell]:SolidQ{};
        }
        input.wallContext.workspace=view.wallWorkspaces
            ?static_cast<void*>(static_cast<unsigned char*>(view.wallWorkspaces)+worker*view.workspaceStride):nullptr;
        input.wallContext.workspaceCapacity=view.workspaceCapacity;
        input.wallContext.failureStatus=&status.wall;
        SurfacePhysicsResult result;
        if(!solveDrySurfaceInterface(input,*physics,result,&status.device,&status.numericalFailure))continue;
        view.physicsOut[face]=result;view.wallOut[face]=dryWallFaceSample(input,result,*physics);status.success=true;
    }
}
}
#endif
