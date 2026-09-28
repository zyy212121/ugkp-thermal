#pragma once
// The sole post-transport moment/gather/recovery dispatch for all applications.
// The compile-time payload schedule is identical for S1/S2 in an application.
// It controls copy placement only; particle moments and field copy bodies are shared.
int launchPostTransportMomentPipeline(DeviceState* s, const int particleGrid, const int block)
{
    const int grid = (s->nCells + block - 1)/block;
    const int warpCount = (block + 31)/32;
    cudaError_t err = cudaSuccess;
    if (particleGrid > 0)
    {
        if (s->csrCellLocalPathEnabled != 0)
        {
            clearParticleMomentsKernel<<<grid, block>>>(s->deviceState);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("clearParticleMomentsKernel launch", err);
                return 1;
            }

            const size_t momentSharedBytes =
                8u*static_cast<size_t>(warpCount)*sizeof(GPU_PIPELINE_REAL);
            if (s->csrHeavyReductionEnabled != 0)
            {
                if (launchCommonSegmentedMomentReduction(s, block, postTransportFusePayload, true) != 0)
                {
                    return 1;
                }
            }
            else
            {
                accumulateParticleMomentsSegmentedKernel<false, postTransportFusePayload>
                    <<<s->nCells, block, momentSharedBytes>>>
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
            clearParticleMomentsAndCountsAtomicKernel<<<countGrid, block>>>
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

            accumulateParticleMomentsAtomicKernel<<<particleGrid, s->particleBlockThreads>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("accumulateParticleMomentsAtomicKernel launch", err);
                return 1;
            }

            normalizeParticleMomentsAtomicKernel<<<grid, block>>>
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
        if (s->csrHeavyReductionEnabled != 0)
        {
            const int heavyGrid = s->multiprocessorCount < s->nCells
                ? s->multiprocessorCount : s->nCells;
            const size_t sharedBytes = 8u*static_cast<size_t>((block + 31)/32)*sizeof(GPU_PIPELINE_REAL);
            err = resetCsrPersistentQueue(s);
            if (err != cudaSuccess)
            {
                setLastError("reset CSR persistent queue", err);
                return 1;
            }
            finalizeCsrSegmentedMomentsAndRecoverKernel
                <<<heavyGrid + grid, block, sharedBytes>>>(s->deviceState);
        }
        else
        {
            solidRecoveryFromParticleMomentsKernel<<<grid, block>>>(s->deviceState);
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
#undef GPU_PIPELINE_REAL
