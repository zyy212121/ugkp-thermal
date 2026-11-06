#pragma once
// The sole post-transport moment/gather/recovery dispatch for all applications.
// The compile-time payload schedule is identical for S1/S2 in an application.
// It controls copy placement only; particle moments and field copy bodies are shared.
int submitPostTransportMomentPipeline(DeviceState* s,const int particleGrid,const int block,cudaStream_t stream,bool heavy)
{
    const int grid = (s->nCells + block - 1)/block;
    const int warpCount = (block + 31)/32;
    cudaError_t err = cudaSuccess;
    if (particleGrid > 0)
    {
        if (s->csrCellLocalPathEnabled != 0)
        {
            clearParticleMomentsKernel<<<grid, block,0,stream>>>(s->deviceState);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("clearParticleMomentsKernel launch", err);
                return 1;
            }

            const size_t momentSharedBytes =
                8u*static_cast<size_t>(warpCount)*sizeof(GPU_PIPELINE_REAL);
            if (heavy)
            {
                if (launchCommonSegmentedMomentReduction(s, block,
                    {postTransportFusePayload ? MomentPayload::gatherSurvivors : MomentPayload::momentsOnly,
                     MomentRecovery::deferToAdvance},stream) != 0)
                {
                    return 1;
                }
            }
            else
            {
                accumulateParticleMomentsSegmentedKernel<false, postTransportFusePayload>
                    <<<s->nCells, block, momentSharedBytes,stream>>>
                    (s->deviceState);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    setLastError
                    (
                        "accumulateParticleMomentsSegmentedKernel launch",
                        err
                    );
                    return 1;
                }
            }
        }
        else
        {
            const int countGrid = (s->nCells + 1 + block - 1)/block;
            clearParticleMomentsAndCountsAtomicKernel<<<countGrid, block,0,stream>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "clearParticleMomentsAndCountsAtomicKernel launch",
                    err
                );
                return 1;
            }

            accumulateParticleMomentsAtomicKernel<<<particleGrid, s->particleBlockThreads,0,stream>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("accumulateParticleMomentsAtomicKernel launch", err);
                return 1;
            }

            normalizeParticleMomentsAtomicKernel<<<grid, block,0,stream>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("normalizeParticleMomentsAtomicKernel launch", err);
                return 1;
            }
        }
        if (heavy)
        {
            const int heavyGrid = s->multiprocessorCount < s->nCells
                ? s->multiprocessorCount : s->nCells;
            const size_t sharedBytes = 8u*static_cast<size_t>((block + 31)/32)*sizeof(GPU_PIPELINE_REAL);
            err = resetCsrPersistentQueue(s,stream);
            if (err != cudaSuccess)
            {
                setLastError("reset CSR persistent queue", err);
                return 1;
            }
            finalizeCsrSegmentedMomentsAndRecoverKernel
                <<<heavyGrid + grid, block, sharedBytes,stream>>>(s->deviceState);
        }
        else
        {
            solidRecoveryFromParticleMomentsKernel<<<grid, block,0,stream>>>(s->deviceState);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("solidRecoveryFromParticleMomentsKernel launch", err);
            return 1;
        }
    }
    return 0;
}
int launchPostTransportMomentPipeline(DeviceState* s,const int particleGrid,const int block)
{
#ifdef UGKP_RESIDENT_CSR_CHOICE
    if(s->csrHeavyReductionMode==2&&particleGrid>0&&s->csrCellLocalPathEnabled)
        return launchCsrConditional(s,4,block,0,postTransportFusePayload?1:0,
            [&](cudaStream_t stream,bool heavy){return submitPostTransportMomentPipeline(s,particleGrid,block,stream,heavy);});
#endif
    return submitPostTransportMomentPipeline(s,particleGrid,block,nullptr,s->csrHeavyReductionEnabled!=0);
}
#undef GPU_PIPELINE_REAL
