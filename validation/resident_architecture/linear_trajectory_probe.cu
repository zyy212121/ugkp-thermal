#include "linear_trajectory_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace trajectoryProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<Report> evaluate(const Topology& topology,const std::vector<Point>& old,const std::vector<Point>& end){
    ResidentLinearTrajectory trajectory(topology,old.size());const auto view=trajectory.view();
    Point* points=nullptr;Point* endpoint=nullptr;cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    if(!old.empty()){
        check(cudaMalloc(&points,old.size()*sizeof(Point)));check(cudaMalloc(&endpoint,end.size()*sizeof(Point)));
        check(cudaMemcpy(points,old.data(),old.size()*sizeof(Point),cudaMemcpyHostToDevice));
        check(cudaMemcpy(endpoint,end.data(),end.size()*sizeof(Point),cudaMemcpyHostToDevice));
    }
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));trajectory.evaluate(points,endpoint,stream);
    check(cudaStreamEndCapture(stream,&graph));
    std::size_t nodeCount=0;check(cudaGraphGetNodes(graph,nullptr,&nodeCount));std::vector<cudaGraphNode_t> nodes(nodeCount);
    check(cudaGraphGetNodes(graph,nodes.data(),&nodeCount));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("non-kernel trajectory graph node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    // Repeated execution must clear previous failure and warning state, including
    // when a moving invalid trajectory becomes globally stationary.
    for(int replay=0;replay<3;++replay){
        const auto& target=replay==1?old:end;
        if(!target.empty())check(cudaMemcpyAsync(endpoint,target.data(),target.size()*sizeof(Point),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        if(replay==1){TrajectoryStatus status;check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
            if(status.moving||status.failure!=TrajectoryFailure::None||status.firstFailure!=view.cells)throw std::runtime_error("stale trajectory status");}
    }
    TrajectoryStatus status;check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    std::vector<TrajectoryFailure> failures(view.cells);std::vector<TrajectoryCellDiagnostic> diagnostics(view.cells);
    if(view.cells){check(cudaMemcpy(failures.data(),view.failure,view.cells*sizeof(TrajectoryFailure),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(diagnostics.data(),view.diagnostics,view.cells*sizeof(TrajectoryCellDiagnostic),cudaMemcpyDeviceToHost));}
    std::vector<Report> result(view.cells);int first=view.cells;
    for(int c=0;c<view.cells;++c){result[c].failure=failures[c];result[c].warnings=diagnostics[c].count;
        if(first==view.cells&&failures[c]!=TrajectoryFailure::None)first=c;
        if(diagnostics[c].count){result[c].observed=diagnostics[c].first.observed;result[c].tolerance=diagnostics[c].first.limit;}
        if(c>status.firstFailure&&diagnostics[c].count)throw std::runtime_error("warnings published after first failed cell");}
    if(first!=status.firstFailure||(first<view.cells&&failures[first]!=status.failure))throw std::runtime_error("trajectory first failure mismatch");
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    check(cudaFree(points));check(cudaFree(endpoint));return result;
}
}
