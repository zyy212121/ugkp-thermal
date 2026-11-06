#pragma once
#include "GpuParticleDirectoryGraph.H"
// Captured stages contain no host decisions, floating-point update, random
// sampling, allocation or diagnostic readback. Task readiness and automatic
// scheduling remain outside this graph and execute in their original order.
template<class State,class Submit>
int submitParticleDirectoryGraph(State* s,int kind,int block,Submit&& submit)
{
#ifdef UGKP_DEVELOPMENT_PROBES
    return submit(nullptr);
#else
    if constexpr(!ugkp::gpu::HasParticleDirectoryGraphs<State>::value)
        return submit(nullptr);
    else {
        ugkp::gpu::ParticleDirectoryGraphKey key;
        key.state=s->deviceState;key.temporary=s->cellScanTempStorage;
        key.counts=s->cellParticleCount;key.offsets=s->cellParticleOffset;
        key.temporaryBytes=s->cellScanTempBytes;key.cells=s->nCells;key.cellBlock=block;
        key.particleGrid=s->particleWorkGrid;key.particleBlock=s->particleBlockThreads;
        key.warpBinning=s->csrWarpAggregatedBinning!=0;key.hasInjection=s->nBoundarySources>0;
        auto& entry=s->particleDirectoryGraphs.entry[kind];
        if(!entry.executable||!(entry.key==key)){
            auto error=ugkp::gpu::releaseParticleDirectoryGraph(entry);
            if(error!=cudaSuccess){setLastError("release particle directory graph",error);return 1;}
            cudaStream_t stream=nullptr;
            error=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
            if(error!=cudaSuccess){setLastError("create particle directory capture stream",error);return 1;}
            error=cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
            if(error!=cudaSuccess){cudaStreamDestroy(stream);setLastError("begin particle directory capture",error);return 1;}
            const int status=submit(stream);
            error=cudaStreamEndCapture(stream,&entry.graph);
            const auto destroyError=cudaStreamDestroy(stream);
            if(status){ugkp::gpu::releaseParticleDirectoryGraph(entry);return status;}
            if(error!=cudaSuccess){ugkp::gpu::releaseParticleDirectoryGraph(entry);setLastError("end particle directory capture",error);return 1;}
            if(destroyError!=cudaSuccess){ugkp::gpu::releaseParticleDirectoryGraph(entry);setLastError("release particle directory capture stream",destroyError);return 1;}
            error=cudaGraphInstantiateWithFlags(&entry.executable,entry.graph,0);
            if(error!=cudaSuccess){ugkp::gpu::releaseParticleDirectoryGraph(entry);setLastError("instantiate particle directory graph",error);return 1;}
            entry.key=key;
        }
        // The original particle owner uses the default stream. Replay stays in
        // that stream, after injection/tracking and before pressure/moments.
        const auto error=cudaGraphLaunch(entry.executable,nullptr);
        if(error!=cudaSuccess){setLastError("replay particle directory graph",error);return 1;}
        return 0;
    }
#endif
}

