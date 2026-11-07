#include "mesh_trajectory_probe.H"
#include "gpu/ResidentMeshTrajectory.cuh"
#include <stdexcept>
namespace trajectoryProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void advanceClock(chmt::ResidentWallClock* clock,double dt){if(threadIdx.x||blockIdx.x)return;clock->elapsed.add(dt);++clock->sequence;}
template<class T>struct Device{T* p=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;
    check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}~Device(){if(p)cudaFree(p);}};
std::vector<Step> evaluate(const chmt::HostMesh& mesh,const chmt::WallProgram& program,bool gas,int steps,double dt){
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    chmt::SharedGasDeviceStorage transfers;transfers.bindStream(stream);chmt::ResidentMeshTrajectory owner;std::string error;
    if(!owner.initialize(mesh,chmt::Tolerances{},transfers,error)||!owner.installProgram(program,gas,transfers,error)||!transfers.observeDiagnostics())throw std::runtime_error(error);
    std::vector<chmt::ResidentWallKnotTime> times;for(const auto& k:program.knots)times.push_back({chmt::residentWindowTime(chmt::knotOffset(program.interval,k)),k.time});
    Device<chmt::ResidentWallKnotTime> time(times);chmt::ResidentWallClock initial;initial.origin=program.interval.begin;
    Device<chmt::ResidentWallClock> clock(std::vector<chmt::ResidentWallClock>{initial});
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    if(!owner.prepare(dt,time.p,times.size(),clock.p,nullptr,stream,error)||!owner.accept(nullptr,stream,error))throw std::runtime_error(error);
    advanceClock<<<1,1,0,stream>>>(clock.p,dt);check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));
    std::size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeHost)throw std::runtime_error("mesh trajectory graph contains host exchange");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));std::vector<Step> result;
    for(int i=0;i<steps;++i){check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));Step s;
        check(cudaMemcpy(&s.status,owner.view().status,sizeof(s.status),cudaMemcpyDeviceToHost));
        const auto copy=[&](const auto* src,auto& dst){dst.resize(mesh.points.size());if(!dst.empty())check(cudaMemcpy(dst.data(),src,dst.size()*sizeof(*src),cudaMemcpyDeviceToHost));};
        copy(owner.previous(),s.previous);copy(owner.current(),s.current);copy(owner.endpoint(),s.endpoint);
        if(!owner.queuePublication(transfers,error)||!transfers.observeDiagnostics())throw std::runtime_error(error);
        s.mesh=mesh;owner.publish(s.mesh,s.stage,mesh.geometryVersion+i+1);result.push_back(std::move(s));
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamSynchronize(stream));
    transfers.bindStream(nullptr);check(cudaStreamDestroy(stream));return result;
}
}
