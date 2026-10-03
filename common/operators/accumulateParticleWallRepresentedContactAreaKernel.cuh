#pragma once
__global__ void accumulateParticleWallRepresentedContactAreaKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*blockDim.x + threadIdx.x;
        entry < nWallBound;
        entry += blockDim.x*gridDim.x
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        if (i < 0 || i >= s.particleCapacity)
        {
            asm("trap;");
        }
        if
        (
            s.pStatus[i] == 0
         || s.pStuck[i] == Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const unsigned char wallState = s.pStuck[i];
        const int faceI = s.pStuckFaceId[i];
        if
        (
            faceI < 0
         || faceI >= s.nFaces
         || s.particleStuckCandidateMask[faceI] == 0
        )
        {
            asm("trap;");
        }

        GPU_OPERATOR_REAL physicalContactArea = GPU_OPERATOR_R(0.0);
        if (wallState == Foam::gpuThermal::particleWallDeposited)
        {
            physicalContactArea =
                static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
            if (!(physicalContactArea > GPU_OPERATOR_R(0.0)))
            {
                asm("trap;");
            }
        }
        else if
        (
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit
        )
        {
            const GPU_OPERATOR_TIME duration =
                static_cast<GPU_OPERATOR_TIME>(s.pContactDuration[i]);
            const GPU_OPERATOR_REAL maximumArea =
                static_cast<GPU_OPERATOR_REAL>(s.pContactMaximumArea[i]);
            const GPU_OPERATOR_REAL peakTimeFraction =
                static_cast<GPU_OPERATOR_REAL>(s.pContactPeakFraction[i]);
            const GPU_OPERATOR_REAL damageArea =
                static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
            const GPU_OPERATOR_TIME age =
                clampRange(finiteOr(particleContactAgeStorage(s, i, 0), GPU_OPERATOR_TIME(0)), GPU_OPERATOR_TIME(0), duration);
            if
            (
                !(duration > GPU_OPERATOR_R(0.0)) || !(maximumArea > GPU_OPERATOR_R(0.0))
             || !(peakTimeFraction > GPU_OPERATOR_R(0.0)) || !(peakTimeFraction < GPU_OPERATOR_R(1.0))
             || damageArea < GPU_OPERATOR_R(0.0)
            )
            {
                asm("trap;");
            }
            const GPU_OPERATOR_REAL kinematicArea =
                maximumArea*Foam::gpuThermal::normalizedKinematicArea
                (
                    age/duration, peakTimeFraction
                );
            const unsigned char interactionType =
                s.particleStuckCandidateMask[faceI];
            const GPU_OPERATOR_REAL frozenArea =
                interactionType
             == Foam::gpuThermal::particleWallSolidifyingDeposition
              ? static_cast<GPU_OPERATOR_REAL>(s.pColdFrozenArea[i])
              : interactionType == Foam::gpuThermal::particleWallColdWall2D
              ? static_cast<GPU_OPERATOR_REAL>(s.pCold2DFrozenArea[i])
              : GPU_OPERATOR_R(0.0);
            physicalContactArea = clampMin
            (
                fmax(kinematicArea, frozenArea) - damageArea,
                GPU_OPERATOR_R(0.0)
            );


            if (!(physicalContactArea > GPU_OPERATOR_R(0.0)))
            {
                continue;
            }
        }
        else
        {
            asm("trap;");
        }

        const GPU_OPERATOR_REAL representedArea =
            Foam::gpuThermal::representedDepositionContactArea
            (
                s.rhoSolid,
                s.pd[i],
                physicalContactArea,
                s.pm[i]
            );
        if (!(representedArea > GPU_OPERATOR_R(0.0)))
        {
            asm("trap;");
        }
        atomicAdd
        (
            &s.particleWallRepresentedContactArea[faceI],
            representedArea
        );
    }
}
