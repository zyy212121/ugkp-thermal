#ifndef UGKP_GPU_EXECUTION_GRAPH_CUH
#define UGKP_GPU_EXECUTION_GRAPH_CUH
#include "ExecutionGraph.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu { namespace executionGraphDetail {
inline void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA execution graph: ")+cudaGetErrorString(code));}
// Capture ordinary kernels as a child. Conditional nodes must be attached to
// their owning graph directly because CUDA cannot clone them as child graphs.
template<class Capture> inline cudaGraphNode_t appendCaptured(cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& dependencies,cudaStream_t stream,Capture capture){
    cudaGraph_t child=nullptr;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    try{capture();checked(cudaStreamEndCapture(stream,&child));}
    catch(...){cudaStreamCaptureStatus status;
        if(cudaStreamIsCapturing(stream,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone)
            cudaStreamEndCapture(stream,&child);
        if(child)cudaGraphDestroy(child);throw;}
    cudaGraphNode_t node=nullptr;
    const auto code=cudaGraphAddChildGraphNode(&node,graph,dependencies.data(),dependencies.size(),child);
    cudaGraphDestroy(child);checked(code);return node;
}
inline cudaGraphNode_t addConditional(cudaGraph_t graph,cudaGraphConditionalHandle handle,
    const std::vector<cudaGraphNode_t>& dependencies,cudaGraph_t& body,cudaGraphConditionalNodeType type){
    cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
    parameters.conditional.handle=handle;parameters.conditional.type=type;parameters.conditional.size=1;
    cudaGraphNode_t node=nullptr;
#if CUDART_VERSION >= 13000
    checked(cudaGraphAddNode(&node,graph,dependencies.data(),nullptr,dependencies.size(),&parameters));
#else
    checked(cudaGraphAddNode(&node,graph,dependencies.data(),dependencies.size(),&parameters));
#endif
    body=parameters.conditional.phGraph_out[0];return node;
}
inline cudaGraphNode_t addWhile(cudaGraph_t graph,cudaGraphConditionalHandle handle,
    const std::vector<cudaGraphNode_t>& dependencies,cudaGraph_t& body){
    return addConditional(graph,handle,dependencies,body,cudaGraphCondTypeWhile);
}
inline cudaGraphNode_t addIf(cudaGraph_t graph,cudaGraphConditionalHandle handle,
    const std::vector<cudaGraphNode_t>& dependencies,cudaGraph_t& body){
    return addConditional(graph,handle,dependencies,body,cudaGraphCondTypeIf);
}
}}}
#endif
