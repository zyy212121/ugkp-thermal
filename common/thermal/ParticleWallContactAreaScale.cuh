int prepareParticleWallContactAreaScale(DeviceState* s, const int block,cudaStream_t stream)
{
    if (s->particleStuckModelConfigured == 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemsetAsync
    (
        s->particleWallRepresentedContactArea,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(*s->particleWallRepresentedContactArea),stream
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset particle-wall represented contact area", err);
        return 1;
    }

    accumulateParticleWallRepresentedContactAreaKernel
        <<<s->particleWorkGrid, s->particleBlockThreads,0,stream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "accumulateParticleWallRepresentedContactAreaKernel launch",
            err
        );
        return 1;
    }

    const int faceGrid = (s->nFaces + block - 1)/block;
    finalizeParticleWallContactAreaScaleKernel
        <<<faceGrid, block,0,stream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "finalizeParticleWallContactAreaScaleKernel launch",
            err
        );
        return 1;
    }
    return 0;
}
