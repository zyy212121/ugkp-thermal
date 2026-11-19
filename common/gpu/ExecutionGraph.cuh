#ifndef UGKP_GPU_EXECUTION_GRAPH_CUH
#define UGKP_GPU_EXECUTION_GRAPH_CUH
#include "ExecutionGraph.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
namespace ugkp { namespace gpu { namespace executionGraphDetail {
inline void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA execution graph: ")+cudaGetErrorString(code));}
inline std::vector<cudaGraphNode_t> capturedDependencies(cudaGraphNode_t node){
    size_t count=0;
#if CUDART_VERSION >= 13000
    checked(cudaGraphNodeGetDependencies(node,nullptr,nullptr,&count));
    std::vector<cudaGraphNode_t> result(count);checked(cudaGraphNodeGetDependencies(node,result.data(),nullptr,&count));
#else
    checked(cudaGraphNodeGetDependencies(node,nullptr,&count));
    std::vector<cudaGraphNode_t> result(count);checked(cudaGraphNodeGetDependencies(node,result.data(),&count));
#endif
    return result;
}
inline cudaGraphNode_t joinGraphTails(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& tails){
    if(tails.size()==1)return tails.front();
    cudaGraphNode_t node=nullptr;
    checked(cudaGraphAddEmptyNode(&node,graph,tails.data(),tails.size()));
    return node;
}
inline cudaGraphNode_t appendCapturedGraph(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,cudaGraph_t captured){
    size_t count=0;checked(cudaGraphGetNodes(captured,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);checked(cudaGraphGetNodes(captured,nodes.data(),&count));
    std::map<cudaGraphNode_t,cudaGraphNodeType> types;
    for(auto node:nodes){cudaGraphNodeType type;checked(cudaGraphNodeGetType(node,&type));types[node]=type;
        bool preserveChild=type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemcpy&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty&&type!=cudaGraphNodeTypeGraph;
#if CUDART_VERSION >= 13000
        size_t edgeCount=0;checked(cudaGraphNodeGetDependencies(node,nullptr,nullptr,&edgeCount));
        std::vector<cudaGraphNode_t> upstream(edgeCount);std::vector<cudaGraphEdgeData> edges(edgeCount);
        checked(cudaGraphNodeGetDependencies(node,upstream.data(),edges.data(),&edgeCount));
        for(const auto& edge:edges)preserveChild=preserveChild||edge.from_port||edge.to_port||edge.type;
#endif
        if(preserveChild){
            cudaGraphNode_t child=nullptr;checked(cudaGraphAddChildGraphNode(&child,graph,dependencies.data(),dependencies.size(),captured));return child;
        }
    }
    std::map<cudaGraphNode_t,cudaGraphNode_t> copied;
    std::set<cudaGraphNode_t> terminals(nodes.begin(),nodes.end());
    std::function<cudaGraphNode_t(cudaGraphNode_t)> append=[&](cudaGraphNode_t source){
        auto found=copied.find(source);if(found!=copied.end())return found->second;
        const auto original=capturedDependencies(source);std::vector<cudaGraphNode_t> before;
        for(auto node:original){before.push_back(append(node));terminals.erase(node);}
        if(original.empty())before=dependencies;
        cudaGraphNode_t node=nullptr;
        switch(types[source]){
        case cudaGraphNodeTypeKernel:{cudaKernelNodeParams parameters{};checked(cudaGraphKernelNodeGetParams(source,&parameters));checked(cudaGraphAddKernelNode(&node,graph,before.data(),before.size(),&parameters));checked(cudaGraphKernelNodeCopyAttributes(node,source));break;}
        case cudaGraphNodeTypeMemcpy:{cudaMemcpy3DParms parameters{};checked(cudaGraphMemcpyNodeGetParams(source,&parameters));checked(cudaGraphAddMemcpyNode(&node,graph,before.data(),before.size(),&parameters));break;}
        case cudaGraphNodeTypeMemset:{cudaMemsetParams parameters{};checked(cudaGraphMemsetNodeGetParams(source,&parameters));checked(cudaGraphAddMemsetNode(&node,graph,before.data(),before.size(),&parameters));break;}
        case cudaGraphNodeTypeGraph:{cudaGraph_t child=nullptr;checked(cudaGraphChildGraphNodeGetGraph(source,&child));node=appendCapturedGraph(graph,before,child);break;}
        default:checked(cudaGraphAddEmptyNode(&node,graph,before.data(),before.size()));break;
        }
        copied[source]=node;return node;
    };
    for(auto node:nodes)append(node);
    std::vector<cudaGraphNode_t> tails;for(auto node:terminals)tails.push_back(copied[node]);
    return joinGraphTails(graph,nodes.empty()?dependencies:tails);
}
template<class Capture> inline cudaGraphNode_t appendCaptured(cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& dependencies,cudaStream_t stream,Capture capture){
    cudaGraph_t captured=nullptr;checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    try{capture();checked(cudaStreamEndCapture(stream,&captured));const auto node=appendCapturedGraph(graph,dependencies,captured);checked(cudaGraphDestroy(captured));return node;}
    catch(...){cudaStreamCaptureStatus state;if(cudaStreamIsCapturing(stream,&state)==cudaSuccess&&state!=cudaStreamCaptureStatusNone)cudaStreamEndCapture(stream,&captured);if(captured)cudaGraphDestroy(captured);throw;}
}



template<class Capture> inline cudaGraphNode_t appendRecorded(cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& dependencies,cudaStream_t stream,Capture capture){
    checked(cudaStreamBeginCaptureToGraph(stream,graph,dependencies.data(),nullptr,
        dependencies.size(),cudaStreamCaptureModeThreadLocal));
    try{
        capture();cudaStreamCaptureStatus status;cudaGraph_t target=nullptr;unsigned long long id=0;
        const cudaGraphNode_t* tail=nullptr;std::size_t count=0;
#if CUDART_VERSION >= 13000
        checked(cudaStreamGetCaptureInfo(stream,&status,&id,&target,&tail,nullptr,&count));
#else
        checked(cudaStreamGetCaptureInfo_v2(stream,&status,&id,&target,&tail,&count));
#endif
        std::vector<cudaGraphNode_t> tails;
        if(count)tails.assign(tail,tail+count);else tails=dependencies;
        checked(cudaStreamEndCapture(stream,&target));
        return joinGraphTails(graph,tails);
    }catch(...){
        cudaStreamCaptureStatus status;cudaGraph_t target=nullptr;
        if(cudaStreamIsCapturing(stream,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone)
            cudaStreamEndCapture(stream,&target);
        throw;
    }
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
