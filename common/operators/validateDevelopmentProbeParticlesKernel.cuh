#pragma once
__global__ void validateDevelopmentProbeParticlesKernel
(
    DeviceState* sp,
    DevelopmentProbeDeviceSummary* summary
)
{
    DeviceState& s = *sp;
    int nParticles = *s.particleCountDevice;
    nParticles = nParticles < 0 ? 0 : nParticles;
    nParticles = nParticles > s.particleCapacity ? s.particleCapacity : nParticles;

    const int stride = blockDim.x*gridDim.x;
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += stride
    )
    {
        unsigned long long mask = 0;
        if
        (
            !finiteDevice(s.px[i])
         || !finiteDevice(s.py[i])
         || !finiteDevice(s.pz[i])
        )
        {
            mask |= ProbeBadParticlePosition;
        }
        if
        (
            !finiteDevice(s.pux[i])
         || !finiteDevice(s.puy[i])
         || !finiteDevice(s.puz[i])
        )
        {
            mask |= ProbeBadParticleVelocity;
        }
        if
        (
            !finiteDevice(s.pT[i])
         || !finiteDevice(s.pTheta[i])
         || !finiteDevice(s.pd[i])
         || !finiteDevice(s.pm[i])
         || s.pTheta[i] < 0.0
         || !(s.pd[i] > 0.0)
#if GPU_PROBE_REQUIRE_POSITIVE_ACTIVE_MASS
         || (s.pStatus[i] != 0 && !(s.pm[i] > 0.0))
#else
         || s.pm[i] < 0.0
#endif
         || (s.solveParticleTemperature != 0 && !(s.pT[i] > 0.0))
        )
        {
            mask |= ProbeBadParticleThermal;
        }
        if
        (
            s.pCellId[i] < 0
         || s.pCellId[i] >= s.nCells
         || s.pStatus[i] != 1
#if GPU_OPERATOR_THERMAL
         || s.pStuck[i] > Foam::gpuThermal::particleWallTransientDeposit
         || (
                s.pStuck[i] == Foam::gpuThermal::particleWallDeposited
             && (
                    !(s.pDepositionArea[i] > 0.0f)
                 || !
                    (
                        (
                            s.pContactDuration[i] == 0.0f
                         && s.pContactMaximumArea[i] == 0.0f
                         && s.pContactPeakFraction[i] == 0.0f
                        )
                     || (
                            s.pContactDuration[i] > 0.0f
                         && s.pContactMaximumArea[i] > 0.0f
                         && s.pContactPeakFraction[i] > 0.0f
                         && s.pContactPeakFraction[i] < 1.0f
                        )
                    )
                )
            )
         || (
                (
                    s.pStuck[i]
                 == Foam::gpuThermal::particleWallTransientRebound
                 || s.pStuck[i]
                 == Foam::gpuThermal::particleWallTransientDeposit
                )
             && (
                    !(s.pContactDuration[i] > 0.0f)
                 || !(s.pContactMaximumArea[i] > 0.0f)
                 || !(s.pContactPeakFraction[i] > 0.0f)
                 || !(s.pContactPeakFraction[i] < 1.0f)
                 || particleContactAgeStorage(s, i, 0) > s.pContactDuration[i]
                 || s.pDepositionArea[i] < 0.0f
                )
            )
#endif
        )
        {
            mask |= ProbeBadParticleMetadata;
        }

        recordDevelopmentProbeParticleFailure(summary, i, mask);
    }
}
