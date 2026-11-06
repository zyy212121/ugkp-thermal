#ifndef UGKP_GPU_EXECUTION_GRAPH_H
#define UGKP_GPU_EXECUTION_GRAPH_H
struct CUgraph_st;
struct CUgraphNode_st;
namespace ugkp { namespace gpu {
// CUDA handle declarations without a toolkit dependency for host adapters.
using ExecutionGraph = CUgraph_st*;
using ExecutionGraphNode = CUgraphNode_st*;
}}
#endif
