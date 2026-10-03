#pragma once
#include "GpuParticleGasHeatRelaxation.cuh"
// Shared thermal relaxation; preserve native ordering of independent loads and stuck-state writes.
template<class DragModel>
__device__ void relaxOneParticleToResidentGas
(
    DeviceState& s,
    const int i,
    const GPU_OPERATOR_TIME dt,
    const DragModel dragModel
)
{
    if (i >= s.particleCapacity || s.pStatus[i] == 0)
    {
        return;
    }

    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    const GPU_OPERATOR_REAL rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL ugx = finiteOr(s.couplingUxOld[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ugy = finiteOr(s.couplingUyOld[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ugz = finiteOr(s.couplingUzOld[c], GPU_OPERATOR_R(0.0));
    const unsigned char wallStateAtStepStart = s.pStuck[i];
    const bool stuck = wallStateAtStepStart != Foam::gpuThermal::particleWallMobile;
    const bool finiteContact =
        wallStateAtStepStart == Foam::gpuThermal::particleWallTransientRebound
     || wallStateAtStepStart == Foam::gpuThermal::particleWallTransientDeposit;
    const int wallFaceAtStepStart = stuck ? s.pStuckFaceId[i] : -1;
    const unsigned char wallInteractionType =
        wallFaceAtStepStart >= 0 && wallFaceAtStepStart < s.nFaces
      ? s.particleStuckCandidateMask[wallFaceAtStepStart]
      : Foam::gpuThermal::particleWallInteractionNone;
    if
    (
        stuck
     && wallInteractionType == Foam::gpuThermal::particleWallColdWall2D
    )
    {
        return;
    }
    const bool coldWallContact =
        wallInteractionType
     == Foam::gpuThermal::particleWallSolidifyingDeposition;
    const GPU_OPERATOR_REAL ux = stuck ? GPU_OPERATOR_R(0.0) : s.pux[i];
    const GPU_OPERATOR_REAL uy = stuck ? GPU_OPERATOR_R(0.0) : s.puy[i];
    const GPU_OPERATOR_REAL uz = stuck ? GPU_OPERATOR_R(0.0) : s.puz[i];
    if (!finiteContact)
    {
        s.puxOld[i] = ux;
        s.puyOld[i] = uy;
        s.puzOld[i] = uz;
    }

#if GPU_THERMAL_RELAX_NATIVE_ORDER
    const GPU_OPERATOR_REAL dPart =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );
#endif
    const GPU_OPERATOR_REAL relX = ugx - ux;
    const GPU_OPERATOR_REAL relY = ugy - uy;
    const GPU_OPERATOR_REAL relZ = ugz - uz;
    const GPU_OPERATOR_REAL relMag = sqrt(sqr3(relX, relY, relZ));
#if !GPU_THERMAL_RELAX_NATIVE_ORDER
    const GPU_OPERATOR_REAL dPart =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );
#endif
    const GPU_OPERATOR_REAL mu = clampMin(s.gasMu, GPU_OPERATOR_R(1.0e-30));
    const GPU_OPERATOR_REAL re = rhoG*dPart*relMag/mu;
    if (gasDragModelActive(s.dragModelId))
    {
    const GPU_OPERATOR_REAL invTauDrag = dragInverseTimeDevice
    (
        s,
        rhoG,
        clampRange(GPU_OPERATOR_R(1.0) - solidEpsFromMomentDevice(s, c), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0)),
        dPart,
        relMag,
        dragModel
    );

    if (!stuck)
    {
        
#include "operators/ParticleDragMomentum.inl"

    }
    else
    {
        s.pux[i] = GPU_OPERATOR_R(0.0);
        s.puy[i] = GPU_OPERATOR_R(0.0);
        s.puz[i] = GPU_OPERATOR_R(0.0);
    }

                                                                         
                                                                             
                                                                      
                                                                  
    if (!finiteContact)
    {
        decayParticleUnresolvedTheta(s, i, c);
    }
    }
#if !GPU_THERMAL_RELAX_NATIVE_ORDER
    else if (stuck)
    {
        s.pux[i] = GPU_OPERATOR_R(0.0);
        s.puy[i] = GPU_OPERATOR_R(0.0);
        s.puz[i] = GPU_OPERATOR_R(0.0);
    }

#endif

    if
    (
        s.solveParticleTemperature != 0
     && s.particleGasHeatTransferModelId != 0
     && !coldWallContact
    )
    {
        const GPU_OPERATOR_REAL tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const GPU_OPERATOR_REAL particleCp = particleSpecificHeatDevice(tpOld);
        const GPU_OPERATOR_REAL gasTemperatureK = clampRange
            (finiteOr(s.couplingTgasOld[c], s.TgasMin), s.TgasMin, GPU_OPERATOR_R(1.0e30));
        s.pT[i] = particleTemperatureAfterGasRelaxation
            (s, tpOld, gasTemperatureK, re, dPart, s.rhoSolid*particleCp, dt);
    }

#if GPU_THERMAL_RELAX_NATIVE_ORDER
    if (s.dragModelId == 0 && stuck)
    {
        s.pux[i] = GPU_OPERATOR_R(0.0);
        s.puy[i] = GPU_OPERATOR_R(0.0);
        s.puz[i] = GPU_OPERATOR_R(0.0);
    }

#endif
    if (finiteContact)
    {
#include "operators/FiniteContactRelaxation.inl"
    }
    else if
    (
        wallStateAtStepStart == Foam::gpuThermal::particleWallDeposited
     && s.particleWallHeatTransferEnabled != 0
    )
    {
        const int stuckFaceId = s.pStuckFaceId[i];
        const GPU_OPERATOR_REAL storedDepositionArea =
            static_cast<GPU_OPERATOR_REAL>(s.pDepositionArea[i]);
        if
        (
            stuckFaceId < 0
         || stuckFaceId >= s.nFaces
         || s.particleStuckCandidateMask[stuckFaceId] == 0
         || !(storedDepositionArea > GPU_OPERATOR_R(0.0))
        )
        {
            asm("trap;");
        }
        const GPU_OPERATOR_REAL contactAreaScale =
            s.particleWallContactAreaScale[stuckFaceId];
        if (!(contactAreaScale > GPU_OPERATOR_R(0.0)) || contactAreaScale > GPU_OPERATOR_R(1.0))
        {
            asm("trap;");
        }
        if (!coldWallContact)
        {
            asm("trap;");
        }
    }
}
