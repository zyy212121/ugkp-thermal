#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>
#include <utility>
namespace ugkp { namespace gpu {
// Only launch-time bindings belong to this key. Particle and cell inventories
// remain in DeviceState and are loaded by the same kernels on every replay.
struct ParticleDirectoryGraphKey {
    const void* state=nullptr;
    const void* temporary=nullptr;
    const void* counts=nullptr;
    const void* offsets=nullptr;
    std::size_t temporaryBytes=0;
    int cells=0,cellBlock=0,particleGrid=0,particleBlock=0;
    bool warpBinning=false,hasInjection=false;
    bool operator==(const ParticleDirectoryGraphKey& b)const {
        return state==b.state&&temporary==b.temporary&&counts==b.counts&&offsets==b.offsets
            &&temporaryBytes==b.temporaryBytes&&cells==b.cells&&cellBlock==b.cellBlock
            &&particleGrid==b.particleGrid&&particleBlock==b.particleBlock
            &&warpBinning==b.warpBinning&&hasInjection==b.hasInjection;
    }
};
struct ParticleDirectoryGraph {
    ParticleDirectoryGraphKey key{};
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
};
struct ParticleDirectoryGraphs {
    // Full pre-transport, full survivor, and split injection directories.
    ParticleDirectoryGraph entry[3]{};
};
inline cudaError_t releaseParticleDirectoryGraph(ParticleDirectoryGraph& entry) {
    cudaError_t error=cudaSuccess;
    if(entry.executable)error=cudaGraphExecDestroy(entry.executable);
    if(entry.graph){const auto next=cudaGraphDestroy(entry.graph);if(error==cudaSuccess)error=next;}
    entry={};return error;
}
inline void releaseParticleDirectoryGraphs(ParticleDirectoryGraphs& graphs) {
    for(auto& entry:graphs.entry)releaseParticleDirectoryGraph(entry);
}
template<class State,class=void>struct HasParticleDirectoryGraphs:std::false_type{};
template<class State>struct HasParticleDirectoryGraphs<State,std::void_t<
    decltype(std::declval<State>().particleDirectoryGraphs)>>:std::true_type{};
}}
