#pragma once

// Compile-time directory capabilities and field/kernel adapters. All scan,
// scatter, invalidation, task publication and automatic scheduling live in
// GpuParticleDirectoryHost.cuh. Thermal applications expose full/split only.
inline GPU_DIRECTORY_PARAMETER_TYPE currentPreTransportDirectoryKind(const DeviceState* s)
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    if (s->useSplitPreDirectory == 0) return HeavyDirectoryKind::full;
    return s->preInjectionSegmentActive == 0
        ? HeavyDirectoryKind::baseOnly : HeavyDirectoryKind::splitBaseAndInjection;
#else
    return s->splitPreDirectoryActive != 0;
#endif
}

inline GPU_DIRECTORY_PARAMETER_TYPE fullParticleDirectoryKind()
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    return HeavyDirectoryKind::full;
#else
    return false;
#endif
}

inline GPU_DIRECTORY_PARAMETER_TYPE splitParticleDirectoryKind()
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    return HeavyDirectoryKind::splitBaseAndInjection;
#else
    return true;
#endif
}

inline bool splitParticleDirectoryEnabled(const DeviceState* s)
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    return s->csrSplitPreDirectoryEnabled != 0;
#else
    return s->csrCellLocalPathEnabled != 0;
#endif
}

inline void selectParticleDirectory(DeviceState* s, const bool split)
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    s->useSplitPreDirectory = split ? 1 : 0;
    s->preInjectionSegmentActive = split ? 1 : 0;
#else
    s->splitPreDirectoryActive = split ? 1 : 0;
#endif
}

inline int clearParticleInjectionBins(DeviceState* s, const int grid, const int block)
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    clearParticleCellBinsKernel<<<grid, block>>>(s->deviceState);
#else
    clearSplitPreInjectionBinsKernel<<<grid, block>>>(s->deviceState);
#endif
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("clear split-Dpre injection bins launch", err); return 1; }
    return 0;
}

inline int initialiseParticleInjectionWrites(DeviceState* s, const int grid, const int block)
{
#if GPU_DIRECTORY_HAS_BASE_ONLY
    initialiseParticleCellWritesKernel<<<grid, block>>>(s->deviceState);
#else
    initialiseSplitPreInjectionWritesKernel<<<grid, block>>>(s->deviceState);
#endif
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("initialise split-Dpre injection writes launch", err); return 1; }
    return 0;
}

inline int prepareParticleDirectoryTasks(DeviceState* s, const int block,
    const GPU_DIRECTORY_PARAMETER_TYPE kind)
{
#if !GPU_DIRECTORY_OWNS_TILE_POLICY
    if (updateDynamicHeavyPolicy(s) != 0) return 1;
#endif
    return prepareCsrSegmentedReductionTasks(s, block, kind);
}
