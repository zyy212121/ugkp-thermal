#ifndef CHMT_GPU_MATERIALSURFACEKERNELS_CUH
#define CHMT_GPU_MATERIALSURFACEKERNELS_CUH
#include "gpu/OrderedBisectionCuda.cuh"
#include "ablation/DrySurfaceInterfaceMath.H"
#include "../../../common/gasWall/WallCudaTeam.cuh"
#include "../../../common/gpu/DeviceValue.cuh"
namespace chmt {
template<int Depth> using MaterialSurfaceRootTeam=ugkp::gpu::OrderedBisectionTeam<Depth,ugkwp::gaswall::detail::CudaWallWarpGroup>;
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
    ugkwp::gaswall::WallOutput<Real,Ns>* wallTeamOutput=nullptr;
    const PhysicsConfig* quietPhysics=nullptr;
    std::size_t workspaceStride=0;
    int workspaceCapacity=0,workers=0;
};
template<class View>
__global__ void evaluateMaterialSurfaceKernel(View binding,const PhysicsConfig* physics) {
    const MaterialSurfaceView view=binding;
    const std::size_t worker=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(worker>=std::size_t(view.workers))return;
    for(std::size_t face=worker;face<std::size_t(view.count);face+=view.workers){
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
// A face retains its scalar reaction/root traversal. All team members follow
// that same traversal; independent wall nodes/Jacobian blocks use the common
// cooperative ODE implementation and the owner alone publishes face state.
// A worker owns either one warp or a block of independent probe warps.
// Each probe has a disjoint workspace; one worker still publishes each face.
template<class View,class Team=ugkwp::gaswall::detail::CudaWallTeam>
__global__ __launch_bounds__(Team::width)
void evaluateMaterialSurfaceTeamKernel(View binding,const PhysicsConfig* physics) {
    const MaterialSurfaceView view=binding;
    const int worker=blockIdx.x;
    if(worker>=view.workers)return;
    const Team team{};
    if constexpr(SurfaceProbeExecution<Team>::count>1)team.template initialize<SurfacePhysicsResult>(view.workspaceStride);
    for(std::size_t face=worker;face<std::size_t(view.count);face+=view.workers){
        MaterialSurfaceStatus status;
        if(view.active&&!view.active[face]){
            if(team.leader()){
                view.physicsOut[face]=SurfacePhysicsResult{};view.wallOut[face]=WallFaceSample{};
                status.success=true;view.status[face]=status;
            }
            continue;
        }
        SurfacePhysicsInput input=view.inputs[face];
        const int cell=view.solidCells[face];input.hasSolid=cell>=0;
        if(input.hasSolid){
            input.solid=input.solidBase=view.solid[cell];input.solidVolume=view.volumes[cell];
            input.solidLocalRate=view.materialRate?view.materialRate[cell]:SolidQ{};
        }
        input.wallContext.workspace=static_cast<unsigned char*>(view.wallWorkspaces)+std::size_t(worker)*view.workspaceStride*SurfaceProbeExecution<Team>::count;
        input.wallContext.workspaceCapacity=view.workspaceCapacity;
        input.wallContext.failureStatus=&status.wall;
        input.wallContext.teamOutput=view.wallTeamOutput+std::size_t(worker)*SurfaceProbeExecution<Team>::count;
        input.wallContext.teamDiagnostics=physics->diagnostics;
        input.wallContext.quietPhysics=view.quietPhysics;
        SurfacePhysicsResult result;
        status.success=solveDrySurfaceInterface(input,*physics,result,&status.device,&status.numericalFailure,team);
        if(team.leader()){
            if(status.success){
                view.physicsOut[face]=result;
                view.wallOut[face]=dryWallFaceSample(input,result,*physics);
            }
            view.status[face]=status;
        }
        team.sync();
    }
}
}
#endif
