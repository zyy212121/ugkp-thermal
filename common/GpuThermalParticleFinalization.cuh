#pragma once
// Common finite-contact/capillary finalization with native contact-age storage.
__device__ void finalizeOneThermalizedStuckParticle
(
    DeviceState& s,
    const int i,
    const GPU_OPERATOR_REAL candidateUx,
    const GPU_OPERATOR_REAL candidateUy,
    const GPU_OPERATOR_REAL candidateUz,
    const GPU_OPERATOR_REAL poolMeanUx,
    const GPU_OPERATOR_REAL poolMeanUy,
    const GPU_OPERATOR_REAL poolMeanUz
)
{
    if (s.pStuck[i] == 0)
    {
        asm("trap;");
    }

    const int faceI = s.pStuckFaceId[i];
    if
    (
        faceI < s.nInternalFaces
     || faceI >= s.nFaces
     || s.particleStuckCandidateMask[faceI] == 0
    )
    {
        asm("trap;");
    }
    const Foam::gpuThermal::CapillaryDetachmentState capillary =
        Foam::gpuThermal::evaluateCapillaryDetachmentState
        (
            s.pT[i],
            s.pd[i],
            s.particleWallAdhesionEnergyScale,
            s.particleWallContactAngleCosine
        );
    const GPU_OPERATOR_REAL wallUx = s.gasBoundaryUx[faceI];
    const GPU_OPERATOR_REAL wallUy = s.gasBoundaryUy[faceI];
    const GPU_OPERATOR_REAL wallUz = s.gasBoundaryUz[faceI];
                                                                        
                                                                            
    const GPU_OPERATOR_REAL fluctuationUx = candidateUx - poolMeanUx;
    const GPU_OPERATOR_REAL fluctuationUy = candidateUy - poolMeanUy;
    const GPU_OPERATOR_REAL fluctuationUz = candidateUz - poolMeanUz;
    const GPU_OPERATOR_REAL sampledSpecificEnergy =
        GPU_OPERATOR_R(0.5)*sqr3(fluctuationUx, fluctuationUy, fluctuationUz);
    const GPU_OPERATOR_REAL contactAreaScale =
        s.particleWallContactAreaScale[faceI];
    if (!(contactAreaScale > GPU_OPERATOR_R(0.0)) || contactAreaScale > GPU_OPERATOR_R(1.0))
    {
        asm("trap;");
    }
    const bool finiteContact =
        s.pStuck[i] == Foam::gpuThermal::particleWallTransientRebound
     || s.pStuck[i] == Foam::gpuThermal::particleWallTransientDeposit;
    const GPU_OPERATOR_TIME contactAge = finiteContact ? GPU_CONTACT_AGE(s, i) : GPU_CONTACT_TIME_ZERO;
    Foam::gpuThermal::CapillaryContactDamageResult damage;
    GPU_OPERATOR_REAL accumulatedDamageArea = GPU_OPERATOR_R(0.0);
    if (finiteContact)
    {
        const GPU_OPERATOR_TIME duration = static_cast<GPU_OPERATOR_TIME>(s.pContactDuration[i]);
        const GPU_OPERATOR_REAL maximumArea = static_cast<GPU_OPERATOR_REAL>(s.pContactMaximumArea[i]);
        const GPU_OPERATOR_REAL peakTimeFraction =
            static_cast<GPU_OPERATOR_REAL>(s.pContactPeakFraction[i]);
        const GPU_OPERATOR_REAL oldDamage = static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
        const GPU_OPERATOR_REAL kinematicArea =
            maximumArea*Foam::gpuThermal::normalizedKinematicArea
            (
                contactAge/duration, peakTimeFraction
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
        const GPU_OPERATOR_REAL remainingArea =
            fmax(kinematicArea, frozenArea) - oldDamage;
        const GPU_OPERATOR_REAL adhesionSpecificEnergyPerArea =
            capillary.adhesionSpecificEnergyJkg
           /capillary.equilibriumContactAreaM2;
        const GPU_OPERATOR_REAL requiredEnergy =
            contactAreaScale*adhesionSpecificEnergyPerArea
           *clampMin(remainingArea, GPU_OPERATOR_R(0.0));
        if
        (
            !(duration > GPU_OPERATOR_R(0.0)) || !(maximumArea > GPU_OPERATOR_R(0.0))
         || !(peakTimeFraction > GPU_OPERATOR_R(0.0)) || !(peakTimeFraction < GPU_OPERATOR_R(1.0))
         || oldDamage < GPU_OPERATOR_R(0.0) || !(adhesionSpecificEnergyPerArea > GPU_OPERATOR_R(0.0))
        )
        {
            asm("trap;");
        }
        if (sampledSpecificEnergy >= requiredEnergy)
        {
            const GPU_OPERATOR_REAL residual = sampledSpecificEnergy - requiredEnergy;
            damage = {finiteDevice(residual), true, GPU_OPERATOR_R(0.0), residual};
        }
        else
        {
            const GPU_OPERATOR_REAL consumedArea =
                sampledSpecificEnergy
               /(contactAreaScale*adhesionSpecificEnergyPerArea);
            accumulatedDamageArea = oldDamage + consumedArea;
            damage =
            {
                finiteDevice(accumulatedDamageArea),
                false,
                remainingArea - consumedArea,
                GPU_OPERATOR_R(0.0)
            };
        }
    }
    else
    {
        const GPU_OPERATOR_REAL currentContactAreaM2 =
            static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
        damage = Foam::gpuThermal::applyCapillaryContactDamage
        (
            capillary,
            currentContactAreaM2,
            contactAreaScale,
            sampledSpecificEnergy
        );
    }
    if
    (
        !damage.valid
     || nonFiniteDevice(sampledSpecificEnergy)
     || sampledSpecificEnergy < GPU_OPERATOR_R(0.0)
    )
    {
        asm("trap;");
    }

    GPU_CONTACT_AGE(s, i) = finiteContact ? contactAge : GPU_CONTACT_TIME_ZERO;
#if GPU_CONTACT_SEPARATE_AGE
    s.pTheta[i] = GPU_OPERATOR_R(0.0);
#endif
    if (damage.detached)
    {
        const GPU_OPERATOR_REAL area = s.magSf[faceI];
        if (!(area > GPU_OPERATOR_R(0.0)) || nonFiniteDevice(area))
        {
            asm("trap;");
        }
        const GPU_OPERATOR_REAL fluctuationScale =
            sampledSpecificEnergy > GPU_OPERATOR_R(0.0)
          ? sqrt
            (
                clampRange
                (
                    damage.residualSpecificEnergyJkg/sampledSpecificEnergy,
                    GPU_OPERATOR_R(0.0),
                    GPU_OPERATOR_R(1.0)
                )
            )
          : GPU_OPERATOR_R(0.0);
        const GPU_OPERATOR_REAL outwardNx = s.Sfx[faceI]/area;
        const GPU_OPERATOR_REAL outwardNy = s.Sfy[faceI]/area;
        const GPU_OPERATOR_REAL outwardNz = s.Sfz[faceI]/area;
        GPU_OPERATOR_REAL releaseUx = poolMeanUx + fluctuationScale*fluctuationUx;
        GPU_OPERATOR_REAL releaseUy = poolMeanUy + fluctuationScale*fluctuationUy;
        GPU_OPERATOR_REAL releaseUz = poolMeanUz + fluctuationScale*fluctuationUz;
        const GPU_OPERATOR_REAL relativeNormal =
            (releaseUx - wallUx)*outwardNx
          + (releaseUy - wallUy)*outwardNy
          + (releaseUz - wallUz)*outwardNz;
        if (relativeNormal > GPU_OPERATOR_R(0.0))
        {
            releaseUx -= GPU_OPERATOR_R(2.0)*relativeNormal*outwardNx;
            releaseUy -= GPU_OPERATOR_R(2.0)*relativeNormal*outwardNy;
            releaseUz -= GPU_OPERATOR_R(2.0)*relativeNormal*outwardNz;
        }
        s.pux[i] = releaseUx;
        s.puy[i] = releaseUy;
        s.puz[i] = releaseUz;
        s.puxOld[i] = releaseUx;
        s.puyOld[i] = releaseUy;
        s.puzOld[i] = releaseUz;
        s.pStuck[i] = 0;
        s.pStuckFaceId[i] = -1;
        s.pDepositionArea[i] = 0.0f;
        s.pContactDuration[i] = 0.0f;
        s.pContactMaximumArea[i] = 0.0f;
        s.pContactPeakFraction[i] = 0.0f;
        clearColdWallParticleState(s, i);
        clearColdWall2DParticleState(s, i);
    }
    else
    {
        if
        (
            !(damage.remainingContactAreaM2 > GPU_OPERATOR_R(0.0))
         || damage.remainingContactAreaM2 > static_cast<GPU_OPERATOR_REAL>(FLT_MAX)
        )
        {
            asm("trap;");
        }
        s.pux[i] = GPU_OPERATOR_R(0.0);
        s.puy[i] = GPU_OPERATOR_R(0.0);
        s.puz[i] = GPU_OPERATOR_R(0.0);
        if (finiteContact)
        {
            s.pDepositionArea[i] =
                static_cast<float>(accumulatedDamageArea);
        }
        else
        {
            s.puxOld[i] = GPU_OPERATOR_R(0.0);
            s.puyOld[i] = GPU_OPERATOR_R(0.0);
            s.puzOld[i] = GPU_OPERATOR_R(0.0);
            s.pDepositionArea[i] =
                static_cast<float>(damage.remainingContactAreaM2);
        }
    }
    s.pStatus[i] = 1;
}
