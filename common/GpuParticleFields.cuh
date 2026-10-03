#pragma once
// Generated from ParticleFieldManifest.json by tools/particle_field_contract.py.
// Keep field order: copy/swap instruction order is part of the operator contract.

#define GPU_PARTICLE_FIELDS_PRIMARY_BEFORE_CONTACT(X) \
    X(px, compactPx, s.px[i]) \
    X(py, compactPy, s.py[i]) \
    X(pz, compactPz, s.pz[i]) \
    X(pux, compactPux, s.pux[i]) \
    X(puy, compactPuy, s.puy[i]) \
    X(puz, compactPuz, s.puz[i]) \
    X(pT, compactPT, s.pT[i]) \
    X(pTheta, compactPTheta, s.pTheta[i])

#define GPU_PARTICLE_FIELDS_PRIMARY_AFTER_CONTACT(X) \
    X(pd, compactPd, s.pd[i]) \
    X(pm, compactPm, s.pm[i]) \
    X(pCellId, compactPCellId, c) \
    X(pStatus, compactPStatus, 1)

#define GPU_PARTICLE_FIELDS_IDENTITY(X) \
    X(pRng, compactPRng, s.pRng[i]) \
    X(pOrigId, compactPOrigId, s.pOrigId[i])

#define GPU_PARTICLE_FIELDS_CONTACT_SCALAR(X) \
    X(puxOld, compactPuxOld, s.puxOld[i]) \
    X(puyOld, compactPuyOld, s.puyOld[i]) \
    X(puzOld, compactPuzOld, s.puzOld[i]) \
    X(pStuck, compactPStuck, s.pStuck[i]) \
    X(pStuckFaceId, compactPStuckFaceId, s.pStuckFaceId[i]) \
    X(pDepositionArea, compactPDepositionArea, s.pDepositionArea[i]) \
    X(pContactDuration, compactPContactDuration, s.pContactDuration[i]) \
    X(pContactMaximumArea, compactPContactMaximumArea, s.pContactMaximumArea[i]) \
    X(pContactPeakFraction, compactPContactPeakFraction, s.pContactPeakFraction[i])

#define GPU_PARTICLE_FIELDS_COLD1D_SCALAR(X) \
    X(pColdFrozenArea, compactPColdFrozenArea, s.pColdFrozenArea[i]) \
    X(pColdContactAge, compactPColdContactAge, s.pColdContactAge[i])

#define GPU_PARTICLE_FIELDS_COLD2D_SCALAR(X) \
    X(pCold2DFrozenArea, compactPCold2DFrozenArea, s.pCold2DFrozenArea[i])

#define GPU_PARTICLE_FIELDS_COLD1D_ARRAY(X) \
    X(pColdNodeSpecificEnthalpy, compactPColdNodeSpecificEnthalpy, Foam::gpuThermal::coldWallAxialNodeCount) \
    X(pColdRingSolidMass, compactPColdRingSolidMass, Foam::gpuThermal::coldWallRadialRingCount)

#define GPU_PARTICLE_FIELDS_COLD2D_ARRAY(X) \
    X(pCold2DNodeSpecificEnthalpy, compactPCold2DNodeSpecificEnthalpy, Foam::gpuThermal::coldWall2DNodeCount) \
    X(pCold2DRingContactAge, compactPCold2DRingContactAge, Foam::gpuThermal::coldWall2DRadialNodeCount)

#define GPU_PARTICLE_FIELDS_SEPARATE_AGE(X) \
    X(pContactAge, compactPContactAge, s.pContactAge[i])
