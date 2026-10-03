#pragma once
// Thermal sample schema; retain each entry scalar cast before publishing double.
int collectDevelopmentProbeSample
(
    DeviceState* s,
    const bool particlePath,
    DevelopmentProbeSample& sample
)
{
    cudaError_t err = cudaEventSynchronize(developmentProbe.totalStopEvent);
    if (err != cudaSuccess)
    {
        setLastError("UGKP development probe final event synchronization", err);
        return 1;
    }

    err = cudaEventElapsedTime
    (
        &sample.totalMs,
        developmentProbe.totalStartEvent,
        developmentProbe.totalStopEvent
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaEventElapsedTime UGKP development probe total", err);
        return 1;
    }
    for (int stage = 0; stage < ProbeStageCount; ++stage)
    {
        sample.stageMs[stage] = 0.0f;
        for
        (
            int occurrence = 0;
            occurrence < developmentProbe.stageOccurrenceCount[stage];
            ++occurrence
        )
        {
            if
            (
                !developmentProbe.stageOccurrenceExecuted[stage][occurrence]
            )
            {
                continue;
            }
            float elapsedMs = 0.0f;
            err = cudaEventElapsedTime
            (
                &elapsedMs,
                developmentProbe.stageStartEvents[stage][occurrence],
                developmentProbe.stageStopEvents[stage][occurrence]
            );
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "cudaEventElapsedTime UGKP development probe stage",
                    err
                );
                return 1;
            }
            sample.stageMs[stage] += elapsedMs;
        }
    }
    sample.timingValid = 1;

    int rawParticleCount = 0;
    err = cudaMemcpy
    (
        &rawParticleCount,
        s->particleCountDevice,
        sizeof(rawParticleCount),
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemcpy UGKP development probe particle count", err);
        return 1;
    }
    sample.particleCount = rawParticleCount;
    sample.particleUtilisation = s->particleCapacity > 0
      ? static_cast<GPU_OPERATOR_REAL>(rawParticleCount)/static_cast<GPU_OPERATOR_REAL>(s->particleCapacity)
      : GPU_OPERATOR_R(0.0);
    if (rawParticleCount < 0 || rawParticleCount > s->particleCapacity)
    {
        sample.badFieldMask |= ProbeBadParticleCount;
        ++sample.badParticles;
    }

    if (particlePath)
    {
        err = cudaMemcpy
        (
            developmentProbe.occupancy.data(),
            s->cellParticleCount,
            static_cast<size_t>(s->nCells)*sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemcpy UGKP development probe cell occupancy", err);
            return 1;
        }
    }
    else
    {
        std::fill
        (
            developmentProbe.occupancy.begin(),
            developmentProbe.occupancy.end(),
            0
        );
    }

    long double sumSquares = 0.0L;
    int occupancyMin = INT_MAX;
    int occupancyMax = INT_MIN;
    for (const int count : developmentProbe.occupancy)
    {
        sample.occupancySum += static_cast<long long>(count);
        sumSquares += static_cast<long double>(count)*count;
        occupancyMin = std::min(occupancyMin, count);
        occupancyMax = std::max(occupancyMax, count);
        sample.occupancyNonEmpty += count > 0 ? 1 : 0;
        if (count < 0)
        {
            sample.badFieldMask |= ProbeBadOccupancy;
        }
    }
    if (s->nCells > 0)
    {
        sample.occupancyMin = occupancyMin;
        sample.occupancyMax = occupancyMax;
        sample.occupancyMean =
            static_cast<GPU_OPERATOR_REAL>(sample.occupancySum)/static_cast<GPU_OPERATOR_REAL>(s->nCells);
        const long double mean = static_cast<long double>(sample.occupancyMean);
        long double variance = sumSquares/static_cast<long double>(s->nCells)
                             - mean*mean;
        variance = variance > 0.0L ? variance : 0.0L;
        sample.occupancyStddev = std::sqrt(static_cast<GPU_OPERATOR_REAL>(variance));
        sample.occupancyCv = sample.occupancyMean > GPU_OPERATOR_R(0.0)
          ? sample.occupancyStddev/sample.occupancyMean
          : GPU_OPERATOR_R(0.0);

        std::sort
        (
            developmentProbe.occupancy.begin(),
            developmentProbe.occupancy.end()
        );
        const size_t last = developmentProbe.occupancy.size() - 1;
        sample.occupancyP50 = developmentProbe.occupancy[(50u*last)/100u];
        sample.occupancyP95 = developmentProbe.occupancy[(95u*last)/100u];
        sample.occupancyP99 = developmentProbe.occupancy[(99u*last)/100u];
    }

    sample.occupancyMatchesCount =
        rawParticleCount >= 0
     && rawParticleCount <= s->particleCapacity
     && sample.occupancySum == static_cast<long long>(rawParticleCount);
    if (!sample.occupancyMatchesCount)
    {
        sample.badFieldMask |= ProbeBadOccupancy;
    }

    if (developmentProbeFullValidation())
    {
        err = cudaMemset
        (
            developmentProbe.deviceSummary,
            0,
            sizeof(DevelopmentProbeDeviceSummary)
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemset UGKP development probe summary", err);
            return 1;
        }

        const int block = s->reductionBlockThreads;
        const int cellGrid = (s->nCells + block - 1)/block;
        validateDevelopmentProbeCellsKernel<<<cellGrid, block>>>
        (
            s->deviceState,
            developmentProbe.deviceSummary,
            particlePath ? 1 : 0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("validateDevelopmentProbeCellsKernel launch", err);
            return 1;
        }

        if (particlePath && s->particleWorkGrid > 0)
        {
            validateDevelopmentProbeParticlesKernel<<<s->particleWorkGrid, s->particleBlockThreads>>>
            (
                s->deviceState,
                developmentProbe.deviceSummary
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("validateDevelopmentProbeParticlesKernel launch", err);
                return 1;
            }
        }

        DevelopmentProbeDeviceSummary summary{};
        err = cudaMemcpy
        (
            &summary,
            developmentProbe.deviceSummary,
            sizeof(summary),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemcpy UGKP development probe summary", err);
            return 1;
        }
        sample.badCells += summary.badCells;
        sample.badParticles += summary.badParticles;
        sample.badFieldMask |= summary.badFieldMask;
        sample.firstBadCell = summary.firstBadCellPlusOne == 0
          ? -1
          : summary.firstBadCellPlusOne - 1;
        sample.firstBadParticle = summary.firstBadParticlePlusOne == 0
          ? -1
          : summary.firstBadParticlePlusOne - 1;
    }

    if (sample.badFieldMask != 0)
    {
        sample.status = "state_invalid";
    }
    return 0;
}
