#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace ugkp { namespace gpu {
struct CsrAutomaticRuntime {
    int active=1;double timeStep=0;
    unsigned long long advanceCount=0;int inspect=0;
};
struct CsrChoiceKey {
    const void *state=nullptr,*temporary=nullptr,*counts=nullptr,*offsets=nullptr,*queue=nullptr,*runtime=nullptr,*heavyTasks=nullptr,*heavyCells=nullptr;
    std::size_t temporaryBytes=0;
    int cells=0,block=0,workers=0,multiprocessors=0,directory=0,variant=0;bool inspectionOnly=false;
    bool operator==(const CsrChoiceKey& b)const {
        return state==b.state&&temporary==b.temporary&&counts==b.counts&&offsets==b.offsets&&queue==b.queue&&runtime==b.runtime&&heavyTasks==b.heavyTasks&&heavyCells==b.heavyCells
            &&temporaryBytes==b.temporaryBytes&&cells==b.cells&&block==b.block&&workers==b.workers
            &&multiprocessors==b.multiprocessors&&directory==b.directory&&variant==b.variant&&inspectionOnly==b.inspectionOnly;
    }
};
struct CsrChoiceGraph {CsrChoiceKey key{};cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;};
struct CsrChoiceGraphs {CsrChoiceGraph entry[11]{};};
inline cudaError_t releaseCsrChoiceGraph(CsrChoiceGraph& graph) {
    cudaError_t error=cudaSuccess;
    if(graph.executable)error=cudaGraphExecDestroy(graph.executable);
    if(graph.graph){const auto next=cudaGraphDestroy(graph.graph);if(error==cudaSuccess)error=next;}
    graph={};return error;
}
inline void releaseCsrChoiceGraphs(CsrChoiceGraphs& graphs){for(auto& graph:graphs.entry)releaseCsrChoiceGraph(graph);}
}}
