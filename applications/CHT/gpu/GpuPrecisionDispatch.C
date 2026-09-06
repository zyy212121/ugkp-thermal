#include "GpuParticleRadiationMath.H"
namespace { int selectedPrecision = 32; int liveHandles = 0; }
extern "C" int ugkwpGpuSelectPrecision(int bits)
{
    if ((bits != 32 && bits != 64) || (liveHandles && bits != selectedPrecision)) return 1;
    selectedPrecision = bits;
    return 0;
}
extern "C" int ugkwpGpuSelectedPrecision() { return selectedPrecision; }
extern "C" const char* ugkwpGpuResidentStrictLastError_32();
extern "C" const char* ugkwpGpuResidentStrictLastError_64();
extern "C" int ugkwpGpuResidentStrictCreate_32(
    int nCells,
    int nFaces,
    int nInternalFaces,
    int nCellPlanes,
    int particleCapacity,
    int maxFaceWalkHops,
    double injectionParcelMass,
    unsigned long long rngSeed,
    double gammaGas,
    double Rgas,
    double rhoSolid,
    int solveParticleTemperature,
    double gasMu,
    double gasPr,
    int dragModelId,
    int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX,
    double gravityY,
    double gravityZ,
    double particleDiameterFallback,
    double particleDiameterMin,
    double particleDiameterMax,
    double particleDiameterSigma,
    double injectionTheta,
    double rhoMin,
    double TgasMin,
    double epsSMin,
    double thetaMin,
    double TpMin,
    double TpMax,
    int collisionalPressureEnabled,
    double collisionalRestitution,
    double pressureKickFraction,
    int jammingPressureEnabled,
    double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme,
    int gasReconstruction,
    int gasLimiter,
    int gasTimeIntegrator,
    int gasRobustFallback,
    int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl,
    double waleCw,
    double smagorinskyCs,
    double maxDiffusionNumber,
    int csrCellLocalPathEnabled,
    int csrHeavyReductionMode,
    int csrHeavyAutoInterval,
    int particleBlockThreads,
    int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    void** handle
);
extern "C" int ugkwpGpuResidentStrictCreate_64(
    int nCells,
    int nFaces,
    int nInternalFaces,
    int nCellPlanes,
    int particleCapacity,
    int maxFaceWalkHops,
    double injectionParcelMass,
    unsigned long long rngSeed,
    double gammaGas,
    double Rgas,
    double rhoSolid,
    int solveParticleTemperature,
    double gasMu,
    double gasPr,
    int dragModelId,
    int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX,
    double gravityY,
    double gravityZ,
    double particleDiameterFallback,
    double particleDiameterMin,
    double particleDiameterMax,
    double particleDiameterSigma,
    double injectionTheta,
    double rhoMin,
    double TgasMin,
    double epsSMin,
    double thetaMin,
    double TpMin,
    double TpMax,
    int collisionalPressureEnabled,
    double collisionalRestitution,
    double pressureKickFraction,
    int jammingPressureEnabled,
    double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme,
    int gasReconstruction,
    int gasLimiter,
    int gasTimeIntegrator,
    int gasRobustFallback,
    int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl,
    double waleCw,
    double smagorinskyCs,
    double maxDiffusionNumber,
    int csrCellLocalPathEnabled,
    int csrHeavyReductionMode,
    int csrHeavyAutoInterval,
    int particleBlockThreads,
    int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    void** handle
);
extern "C" int ugkwpGpuResidentStrictUploadMesh_32(
    void* handle,
    const int* faceOwner,
    const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx,
    const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* faceCx,
    const double* faceCy,
    const double* faceCz,
    const double* Sfx,
    const double* Sfy,
    const double* Sfz,
    const double* magSf,
    const double* deltaCoeffs,
    const double* faceWeight,
    const double* cellLength,
    const int* cellPlaneStart,
    const int* cellPlaneCount,
    const int* cellFaceId,
    const int* cellFaceNeighbor,
    const int* cellFaceKind,
    const double* cellFaceRestitution,
    const double* cellFaceTangential,
    const double* planeNx,
    const double* planeNy,
    const double* planeNz,
    const double* planeD
);
extern "C" int ugkwpGpuResidentStrictUploadMesh_64(
    void* handle,
    const int* faceOwner,
    const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx,
    const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* faceCx,
    const double* faceCy,
    const double* faceCz,
    const double* Sfx,
    const double* Sfy,
    const double* Sfz,
    const double* magSf,
    const double* deltaCoeffs,
    const double* faceWeight,
    const double* cellLength,
    const int* cellPlaneStart,
    const int* cellPlaneCount,
    const int* cellFaceId,
    const int* cellFaceNeighbor,
    const int* cellFaceKind,
    const double* cellFaceRestitution,
    const double* cellFaceTangential,
    const double* planeNx,
    const double* planeNy,
    const double* planeNz,
    const double* planeD
);
extern "C" int ugkwpGpuResidentStrictUploadBoundarySources_32(
    void* handle,
    int nBoundarySources,
    const int* sourceCell,
    const int* sourceFace,
    const double* sourcePx,
    const double* sourcePy,
    const double* sourcePz,
    const double* sourceUx,
    const double* sourceUy,
    const double* sourceUz,
    const double* sourceT,
    const double* sourceTheta,
    const double* sourceD,
    const double* sourceMassRate
);
extern "C" int ugkwpGpuResidentStrictUploadBoundarySources_64(
    void* handle,
    int nBoundarySources,
    const int* sourceCell,
    const int* sourceFace,
    const double* sourcePx,
    const double* sourcePy,
    const double* sourcePz,
    const double* sourceUx,
    const double* sourceUy,
    const double* sourceUz,
    const double* sourceT,
    const double* sourceTheta,
    const double* sourceD,
    const double* sourceMassRate
);
extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet_32(
    void* handle,
    int nFaces,
    const int* faceIds,
    double inletTemperature,
    int nPressureRows,
    const double* pressureTimes,
    const double* pressureValues,
    int nVolumeFractionRows,
    const double* volumeFractionTimes,
    const double* volumeFractionValues
);
extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet_64(
    void* handle,
    int nFaces,
    const int* faceIds,
    double inletTemperature,
    int nPressureRows,
    const double* pressureTimes,
    const double* pressureValues,
    int nVolumeFractionRows,
    const double* volumeFractionTimes,
    const double* volumeFractionValues
);
extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass_32(
    void* handle,
    int* nSources,
    int maxSources,
    int* sourceFace,
    double* residualMass
);
extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass_64(
    void* handle,
    int* nSources,
    int maxSources,
    int* sourceFace,
    double* residualMass
);
extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass_32(
    void* handle,
    int nSources,
    const int* sourceFace,
    const double* residualMass
);
extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass_64(
    void* handle,
    int nSources,
    const int* sourceFace,
    const double* residualMass
);
extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields_32(
    void* handle,
    const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix,
    const int* gasBoundaryUFix,
    const int* gasBoundaryPFix,
    const int* gasBoundaryTFix,
    const int* gasBoundaryPWave,
    const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf,
    const double* gasBoundaryRho,
    const double* gasBoundaryUx,
    const double* gasBoundaryUy,
    const double* gasBoundaryUz,
    const double* gasBoundaryP,
    const double* gasBoundaryT
);
extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields_64(
    void* handle,
    const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix,
    const int* gasBoundaryUFix,
    const int* gasBoundaryPFix,
    const int* gasBoundaryTFix,
    const int* gasBoundaryPWave,
    const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf,
    const double* gasBoundaryRho,
    const double* gasBoundaryUx,
    const double* gasBoundaryUy,
    const double* gasBoundaryUz,
    const double* gasBoundaryP,
    const double* gasBoundaryT
);
extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch_32(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* temperatures
);
extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch_64(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* temperatures
);
extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch_32(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
);
extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch_64(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
);
extern "C" int ugkwpGpuResidentStrictUploadFields_32(
    void* handle,
    const double* rho,
    const double* rhoUx,
    const double* rhoUy,
    const double* rhoUz,
    const double* rhoE,
    const double* Ux,
    const double* Uy,
    const double* Uz,
    const double* p,
    const double* Tgas
);
extern "C" int ugkwpGpuResidentStrictUploadFields_64(
    void* handle,
    const double* rho,
    const double* rhoUx,
    const double* rhoUy,
    const double* rhoUz,
    const double* rhoE,
    const double* Ux,
    const double* Uy,
    const double* Uz,
    const double* p,
    const double* Tgas
);
extern "C" int ugkwpGpuResidentStrictConfigureSst_32(
    void* handle,
    double alphaK1,
    double alphaK2,
    double alphaOmega1,
    double alphaOmega2,
    double beta1,
    double beta2,
    double betaStar,
    double gamma1,
    double gamma2,
    double a1,
    double b1,
    double c1,
    double kMin,
    double omegaMin,
    double maxSourceNumber,
    int wallTreatment,
    double wallKappa,
    double wallE,
    double wallCmu,
    const double* k,
    const double* omega,
    const double* wallDistance,
    const int* boundaryKMode,
    const int* boundaryOmegaMode,
    const double* boundaryK,
    const double* boundaryOmega
);
extern "C" int ugkwpGpuResidentStrictConfigureSst_64(
    void* handle,
    double alphaK1,
    double alphaK2,
    double alphaOmega1,
    double alphaOmega2,
    double beta1,
    double beta2,
    double betaStar,
    double gamma1,
    double gamma2,
    double a1,
    double b1,
    double c1,
    double kMin,
    double omegaMin,
    double maxSourceNumber,
    int wallTreatment,
    double wallKappa,
    double wallE,
    double wallCmu,
    const double* k,
    const double* omega,
    const double* wallDistance,
    const int* boundaryKMode,
    const int* boundaryOmegaMode,
    const double* boundaryK,
    const double* boundaryOmega
);
extern "C" int ugkwpGpuResidentStrictComputeGasCourant_32(
    void* handle,
    double dt,
    double targetMaxCo,
    double scheduleTime,
    double* maxCo
);
extern "C" int ugkwpGpuResidentStrictComputeGasCourant_64(
    void* handle,
    double dt,
    double targetMaxCo,
    double scheduleTime,
    double* maxCo
);
extern "C" int ugkwpGpuResidentStrictAdvance_32(
    void* handle,
    double dt,
    double simulationTime
);
extern "C" int ugkwpGpuResidentStrictAdvance_64(
    void* handle,
    double dt,
    double simulationTime
);
extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly_32(
    void* handle,
    double dt,
    double simulationTime
);
extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly_64(
    void* handle,
    double dt,
    double simulationTime
);
extern "C" int ugkwpGpuResidentStrictDownloadFields_32(
    void* handle,
    double* rho,
    double* rhoUx,
    double* rhoUy,
    double* rhoUz,
    double* rhoE,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas,
    double* epsS,
    double* rhoUsx,
    double* rhoUsy,
    double* rhoUsz,
    double* rhoEs,
    double* rhoDs,
    double* rhoHp,
    double* Usx,
    double* Usy,
    double* Usz,
    double* theta,
    double* Tp,
    double* dMeanCell
);
extern "C" int ugkwpGpuResidentStrictDownloadFields_64(
    void* handle,
    double* rho,
    double* rhoUx,
    double* rhoUy,
    double* rhoUz,
    double* rhoE,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas,
    double* epsS,
    double* rhoUsx,
    double* rhoUsy,
    double* rhoUsz,
    double* rhoEs,
    double* rhoDs,
    double* rhoHp,
    double* Usx,
    double* Usy,
    double* Usz,
    double* theta,
    double* Tp,
    double* dMeanCell
);
extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature_32(
    void* handle,
    int nCells,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
);
extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature_64(
    void* handle,
    int nCells,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
);
extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums_32(
    void* handle,
    int nCells,
    double* particleMassKg,
    double* particleTemperatureMassKgK,
    double* particleDiameterMassKgM
);
extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums_64(
    void* handle,
    int nCells,
    double* particleMassKg,
    double* particleTemperatureMassKgK,
    double* particleDiameterMassKgM
);
extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea_32(
    void* handle,
    int nFaces,
    double* occupiedAreaM2
);
extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea_64(
    void* handle,
    int nFaces,
    double* occupiedAreaM2
);
extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked_32(
    void* handle
);
extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked_64(
    void* handle
);
extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev_32(
    void* handle,
    double* epsGPrev
);
extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev_64(
    void* handle,
    double* epsGPrev
);
extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev_32(
    void* handle,
    const double* epsGPrev
);
extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev_64(
    void* handle,
    const double* epsGPrev
);
extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields_32(
    void* handle,
    double* rho,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas
);
extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields_64(
    void* handle,
    double* rho,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas
);
extern "C" int ugkwpGpuResidentStrictDownloadNut_32(
    void* handle,
    double* nut
);
extern "C" int ugkwpGpuResidentStrictDownloadNut_64(
    void* handle,
    double* nut
);
extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger_32(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
);
extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger_64(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
);
extern "C" int ugkwpGpuResidentStrictPeekGasWallEnergy_32(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);
extern "C" int ugkwpGpuResidentStrictPeekGasWallEnergy_64(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);
extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange_32(
    void* handle,
    int firstFace,
    int nFaces,
    double* gasWallEnergyJ,
    double* particleDepositedWallEnergyJ,
    double* particleReflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange_64(
    void* handle,
    int firstFace,
    int nFaces,
    double* gasWallEnergyJ,
    double* particleDepositedWallEnergyJ,
    double* particleReflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange_32(
    void* handle,
    int firstFace,
    int nFaces,
    const double* gasWallEnergyJ,
    const double* particleDepositedWallEnergyJ,
    const double* particleReflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange_64(
    void* handle,
    int firstFace,
    int nFaces,
    const double* gasWallEnergyJ,
    const double* particleDepositedWallEnergyJ,
    const double* particleReflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy_32(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);
extern "C" int ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy_64(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);
extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel_32(
    void* handle,
    int nFaces,
    const unsigned char* candidateFaceMask,
    double sommerfeldThreshold,
    int heatTransferEnabled,
    double maximumCoverage,
    double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale,
    double contactAngleDegree,
    int wallTransientResistance,
    int nonlinearIterations,
    double meltingTemperatureK,
    double mushyRangeK,
    double latentHeatJkg,
    double solidDensityKgM3,
    double solidSpecificHeatJkgK,
    double solidThermalConductivityWmK,
    double pinningThicknessFraction,
    double interfaceResistanceM2KW
);
extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel_64(
    void* handle,
    int nFaces,
    const unsigned char* candidateFaceMask,
    double sommerfeldThreshold,
    int heatTransferEnabled,
    double maximumCoverage,
    double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale,
    double contactAngleDegree,
    int wallTransientResistance,
    int nonlinearIterations,
    double meltingTemperatureK,
    double mushyRangeK,
    double latentHeatJkg,
    double solidDensityKgM3,
    double solidSpecificHeatJkgK,
    double solidThermalConductivityWmK,
    double pinningThicknessFraction,
    double interfaceResistanceM2KW
);
extern "C" int ugkwpGpuResidentStrictPeekParticleWallHeatLedgers_32(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictPeekParticleWallHeatLedgers_64(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers_32(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers_64(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);
extern "C" int ugkwpGpuResidentStrictDownloadSst_32(
    void* handle,
    double* k,
    double* omega,
    double* nut
);
extern "C" int ugkwpGpuResidentStrictDownloadSst_64(
    void* handle,
    double* k,
    double* omega,
    double* nut
);
extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror_32(
    void* handle,
    int nParticles,
    const double* px,
    const double* py,
    const double* pz,
    const double* pux,
    const double* puy,
    const double* puz,
    const double* pT,
    const double* pTheta,
    const double* pd,
    const double* pm,
    const int* pCellId,
    const int* pStatus,
    const unsigned char* pStuck,
    const int* pStuckFaceId,
    const float* pDepositionArea,
    const double* pContactDuration,
    const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const double* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const double* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng,
    const unsigned long long* pOrigId
);
extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror_64(
    void* handle,
    int nParticles,
    const double* px,
    const double* py,
    const double* pz,
    const double* pux,
    const double* puy,
    const double* puz,
    const double* pT,
    const double* pTheta,
    const double* pd,
    const double* pm,
    const int* pCellId,
    const int* pStatus,
    const unsigned char* pStuck,
    const int* pStuckFaceId,
    const float* pDepositionArea,
    const double* pContactDuration,
    const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const double* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const double* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng,
    const unsigned long long* pOrigId
);
extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror_32(
    void* handle,
    int* nParticles,
    int maxParticles,
    double* px,
    double* py,
    double* pz,
    double* pux,
    double* puy,
    double* puz,
    double* pT,
    double* pTheta,
    double* pd,
    double* pm,
    int* pCellId,
    int* pStatus,
    unsigned char* pStuck,
    int* pStuckFaceId,
    float* pDepositionArea,
    double* pContactDuration,
    float* pContactMaximumArea,
    float* pContactPeakFraction,
    float* pColdNodeSpecificEnthalpy,
    float* pColdRingSolidMass,
    float* pColdFrozenArea,
    double* pColdContactAge,
    float* pCold2DNodeSpecificEnthalpy,
    double* pCold2DRingContactAge,
    float* pCold2DFrozenArea,
    unsigned long long* pRng,
    unsigned long long* pOrigId
);
extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror_64(
    void* handle,
    int* nParticles,
    int maxParticles,
    double* px,
    double* py,
    double* pz,
    double* pux,
    double* puy,
    double* puz,
    double* pT,
    double* pTheta,
    double* pd,
    double* pm,
    int* pCellId,
    int* pStatus,
    unsigned char* pStuck,
    int* pStuckFaceId,
    float* pDepositionArea,
    double* pContactDuration,
    float* pContactMaximumArea,
    float* pContactPeakFraction,
    float* pColdNodeSpecificEnthalpy,
    float* pColdRingSolidMass,
    float* pColdFrozenArea,
    double* pColdContactAge,
    float* pCold2DNodeSpecificEnthalpy,
    double* pCold2DRingContactAge,
    float* pCold2DFrozenArea,
    unsigned long long* pRng,
    unsigned long long* pOrigId
);
extern "C" void ugkwpGpuResidentStrictRelease_32(void* handle);
extern "C" void ugkwpGpuResidentStrictRelease_64(void* handle);
extern "C" const char* ugkwpGpuResidentStrictLastError()
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictLastError_32() : ugkwpGpuResidentStrictLastError_64();
}

extern "C" int ugkwpGpuResidentStrictCreate(
    int nCells,
    int nFaces,
    int nInternalFaces,
    int nCellPlanes,
    int particleCapacity,
    int maxFaceWalkHops,
    double injectionParcelMass,
    unsigned long long rngSeed,
    double gammaGas,
    double Rgas,
    double rhoSolid,
    int solveParticleTemperature,
    double gasMu,
    double gasPr,
    int dragModelId,
    int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX,
    double gravityY,
    double gravityZ,
    double particleDiameterFallback,
    double particleDiameterMin,
    double particleDiameterMax,
    double particleDiameterSigma,
    double injectionTheta,
    double rhoMin,
    double TgasMin,
    double epsSMin,
    double thetaMin,
    double TpMin,
    double TpMax,
    int collisionalPressureEnabled,
    double collisionalRestitution,
    double pressureKickFraction,
    int jammingPressureEnabled,
    double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme,
    int gasReconstruction,
    int gasLimiter,
    int gasTimeIntegrator,
    int gasRobustFallback,
    int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl,
    double waleCw,
    double smagorinskyCs,
    double maxDiffusionNumber,
    int csrCellLocalPathEnabled,
    int csrHeavyReductionMode,
    int csrHeavyAutoInterval,
    int particleBlockThreads,
    int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    void** handle
)
{
    int rc = selectedPrecision == 32 ? ugkwpGpuResidentStrictCreate_32(nCells, nFaces, nInternalFaces, nCellPlanes, particleCapacity, maxFaceWalkHops, injectionParcelMass, rngSeed, gammaGas, Rgas, rhoSolid, solveParticleTemperature, gasMu, gasPr, dragModelId, particleGasHeatTransferModelId, dragResidualRe, gravityX, gravityY, gravityZ, particleDiameterFallback, particleDiameterMin, particleDiameterMax, particleDiameterSigma, injectionTheta, rhoMin, TgasMin, epsSMin, thetaMin, TpMin, TpMax, collisionalPressureEnabled, collisionalRestitution, pressureKickFraction, jammingPressureEnabled, packingFraction, packingProjectionIterations, gasFluxScheme, gasReconstruction, gasLimiter, gasTimeIntegrator, gasRobustFallback, turbulenceModel, lesDeltaCoeff, turbulentPrandtl, waleCw, smagorinskyCs, maxDiffusionNumber, csrCellLocalPathEnabled, csrHeavyReductionMode, csrHeavyAutoInterval, particleBlockThreads, reductionBlockThreads, csrWarpAggregatedBinning, handle) : ugkwpGpuResidentStrictCreate_64(nCells, nFaces, nInternalFaces, nCellPlanes, particleCapacity, maxFaceWalkHops, injectionParcelMass, rngSeed, gammaGas, Rgas, rhoSolid, solveParticleTemperature, gasMu, gasPr, dragModelId, particleGasHeatTransferModelId, dragResidualRe, gravityX, gravityY, gravityZ, particleDiameterFallback, particleDiameterMin, particleDiameterMax, particleDiameterSigma, injectionTheta, rhoMin, TgasMin, epsSMin, thetaMin, TpMin, TpMax, collisionalPressureEnabled, collisionalRestitution, pressureKickFraction, jammingPressureEnabled, packingFraction, packingProjectionIterations, gasFluxScheme, gasReconstruction, gasLimiter, gasTimeIntegrator, gasRobustFallback, turbulenceModel, lesDeltaCoeff, turbulentPrandtl, waleCw, smagorinskyCs, maxDiffusionNumber, csrCellLocalPathEnabled, csrHeavyReductionMode, csrHeavyAutoInterval, particleBlockThreads, reductionBlockThreads, csrWarpAggregatedBinning, handle);
    if (rc == 0) ++liveHandles;
    return rc;
}

extern "C" int ugkwpGpuResidentStrictUploadMesh(
    void* handle,
    const int* faceOwner,
    const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx,
    const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* faceCx,
    const double* faceCy,
    const double* faceCz,
    const double* Sfx,
    const double* Sfy,
    const double* Sfz,
    const double* magSf,
    const double* deltaCoeffs,
    const double* faceWeight,
    const double* cellLength,
    const int* cellPlaneStart,
    const int* cellPlaneCount,
    const int* cellFaceId,
    const int* cellFaceNeighbor,
    const int* cellFaceKind,
    const double* cellFaceRestitution,
    const double* cellFaceTangential,
    const double* planeNx,
    const double* planeNy,
    const double* planeNz,
    const double* planeD
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadMesh_32(handle, faceOwner, faceNeighbour, facePeriodicPair, facePeriodicDx, facePeriodicDy, facePeriodicDz, V, Cx, Cy, Cz, faceCx, faceCy, faceCz, Sfx, Sfy, Sfz, magSf, deltaCoeffs, faceWeight, cellLength, cellPlaneStart, cellPlaneCount, cellFaceId, cellFaceNeighbor, cellFaceKind, cellFaceRestitution, cellFaceTangential, planeNx, planeNy, planeNz, planeD) : ugkwpGpuResidentStrictUploadMesh_64(handle, faceOwner, faceNeighbour, facePeriodicPair, facePeriodicDx, facePeriodicDy, facePeriodicDz, V, Cx, Cy, Cz, faceCx, faceCy, faceCz, Sfx, Sfy, Sfz, magSf, deltaCoeffs, faceWeight, cellLength, cellPlaneStart, cellPlaneCount, cellFaceId, cellFaceNeighbor, cellFaceKind, cellFaceRestitution, cellFaceTangential, planeNx, planeNy, planeNz, planeD);
}

extern "C" int ugkwpGpuResidentStrictUploadBoundarySources(
    void* handle,
    int nBoundarySources,
    const int* sourceCell,
    const int* sourceFace,
    const double* sourcePx,
    const double* sourcePy,
    const double* sourcePz,
    const double* sourceUx,
    const double* sourceUy,
    const double* sourceUz,
    const double* sourceT,
    const double* sourceTheta,
    const double* sourceD,
    const double* sourceMassRate
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadBoundarySources_32(handle, nBoundarySources, sourceCell, sourceFace, sourcePx, sourcePy, sourcePz, sourceUx, sourceUy, sourceUz, sourceT, sourceTheta, sourceD, sourceMassRate) : ugkwpGpuResidentStrictUploadBoundarySources_64(handle, nBoundarySources, sourceCell, sourceFace, sourcePx, sourcePy, sourcePz, sourceUx, sourceUy, sourceUz, sourceT, sourceTheta, sourceD, sourceMassRate);
}

extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet(
    void* handle,
    int nFaces,
    const int* faceIds,
    double inletTemperature,
    int nPressureRows,
    const double* pressureTimes,
    const double* pressureValues,
    int nVolumeFractionRows,
    const double* volumeFractionTimes,
    const double* volumeFractionValues
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictConfigureScheduledInlet_32(handle, nFaces, faceIds, inletTemperature, nPressureRows, pressureTimes, pressureValues, nVolumeFractionRows, volumeFractionTimes, volumeFractionValues) : ugkwpGpuResidentStrictConfigureScheduledInlet_64(handle, nFaces, faceIds, inletTemperature, nPressureRows, pressureTimes, pressureValues, nVolumeFractionRows, volumeFractionTimes, volumeFractionValues);
}

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass(
    void* handle,
    int* nSources,
    int maxSources,
    int* sourceFace,
    double* residualMass
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadSourceResidualMass_32(handle, nSources, maxSources, sourceFace, residualMass) : ugkwpGpuResidentStrictDownloadSourceResidualMass_64(handle, nSources, maxSources, sourceFace, residualMass);
}

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass(
    void* handle,
    int nSources,
    const int* sourceFace,
    const double* residualMass
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadSourceResidualMass_32(handle, nSources, sourceFace, residualMass) : ugkwpGpuResidentStrictUploadSourceResidualMass_64(handle, nSources, sourceFace, residualMass);
}

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields(
    void* handle,
    const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix,
    const int* gasBoundaryUFix,
    const int* gasBoundaryPFix,
    const int* gasBoundaryTFix,
    const int* gasBoundaryPWave,
    const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf,
    const double* gasBoundaryRho,
    const double* gasBoundaryUx,
    const double* gasBoundaryUy,
    const double* gasBoundaryUz,
    const double* gasBoundaryP,
    const double* gasBoundaryT
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadGasBoundaryFields_32(handle, gasBoundaryKind, gasBoundaryRhoFix, gasBoundaryUFix, gasBoundaryPFix, gasBoundaryTFix, gasBoundaryPWave, gasBoundaryPWaveGamma, gasBoundaryPWaveFieldInf, gasBoundaryPWaveLInf, gasBoundaryRho, gasBoundaryUx, gasBoundaryUy, gasBoundaryUz, gasBoundaryP, gasBoundaryT) : ugkwpGpuResidentStrictUploadGasBoundaryFields_64(handle, gasBoundaryKind, gasBoundaryRhoFix, gasBoundaryUFix, gasBoundaryPFix, gasBoundaryTFix, gasBoundaryPWave, gasBoundaryPWaveGamma, gasBoundaryPWaveFieldInf, gasBoundaryPWaveLInf, gasBoundaryRho, gasBoundaryUx, gasBoundaryUy, gasBoundaryUz, gasBoundaryP, gasBoundaryT);
}

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* temperatures
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch_32(handle, patchStartFace, patchFaceCount, temperatures) : ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch_64(handle, patchStartFace, patchFaceCount, temperatures);
}

extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch_32(handle, patchStartFace, patchFaceCount, wallEffusivity) : ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch_64(handle, patchStartFace, patchFaceCount, wallEffusivity);
}

extern "C" int ugkwpGpuResidentStrictUploadFields(
    void* handle,
    const double* rho,
    const double* rhoUx,
    const double* rhoUy,
    const double* rhoUz,
    const double* rhoE,
    const double* Ux,
    const double* Uy,
    const double* Uz,
    const double* p,
    const double* Tgas
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadFields_32(handle, rho, rhoUx, rhoUy, rhoUz, rhoE, Ux, Uy, Uz, p, Tgas) : ugkwpGpuResidentStrictUploadFields_64(handle, rho, rhoUx, rhoUy, rhoUz, rhoE, Ux, Uy, Uz, p, Tgas);
}

extern "C" int ugkwpGpuResidentStrictConfigureSst(
    void* handle,
    double alphaK1,
    double alphaK2,
    double alphaOmega1,
    double alphaOmega2,
    double beta1,
    double beta2,
    double betaStar,
    double gamma1,
    double gamma2,
    double a1,
    double b1,
    double c1,
    double kMin,
    double omegaMin,
    double maxSourceNumber,
    int wallTreatment,
    double wallKappa,
    double wallE,
    double wallCmu,
    const double* k,
    const double* omega,
    const double* wallDistance,
    const int* boundaryKMode,
    const int* boundaryOmegaMode,
    const double* boundaryK,
    const double* boundaryOmega
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictConfigureSst_32(handle, alphaK1, alphaK2, alphaOmega1, alphaOmega2, beta1, beta2, betaStar, gamma1, gamma2, a1, b1, c1, kMin, omegaMin, maxSourceNumber, wallTreatment, wallKappa, wallE, wallCmu, k, omega, wallDistance, boundaryKMode, boundaryOmegaMode, boundaryK, boundaryOmega) : ugkwpGpuResidentStrictConfigureSst_64(handle, alphaK1, alphaK2, alphaOmega1, alphaOmega2, beta1, beta2, betaStar, gamma1, gamma2, a1, b1, c1, kMin, omegaMin, maxSourceNumber, wallTreatment, wallKappa, wallE, wallCmu, k, omega, wallDistance, boundaryKMode, boundaryOmegaMode, boundaryK, boundaryOmega);
}

extern "C" int ugkwpGpuResidentStrictComputeGasCourant(
    void* handle,
    double dt,
    double targetMaxCo,
    double scheduleTime,
    double* maxCo
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictComputeGasCourant_32(handle, dt, targetMaxCo, scheduleTime, maxCo) : ugkwpGpuResidentStrictComputeGasCourant_64(handle, dt, targetMaxCo, scheduleTime, maxCo);
}

extern "C" int ugkwpGpuResidentStrictAdvance(
    void* handle,
    double dt,
    double simulationTime
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictAdvance_32(handle, dt, simulationTime) : ugkwpGpuResidentStrictAdvance_64(handle, dt, simulationTime);
}

extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly(
    void* handle,
    double dt,
    double simulationTime
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictAdvanceGasOnly_32(handle, dt, simulationTime) : ugkwpGpuResidentStrictAdvanceGasOnly_64(handle, dt, simulationTime);
}

extern "C" int ugkwpGpuResidentStrictDownloadFields(
    void* handle,
    double* rho,
    double* rhoUx,
    double* rhoUy,
    double* rhoUz,
    double* rhoE,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas,
    double* epsS,
    double* rhoUsx,
    double* rhoUsy,
    double* rhoUsz,
    double* rhoEs,
    double* rhoDs,
    double* rhoHp,
    double* Usx,
    double* Usy,
    double* Usz,
    double* theta,
    double* Tp,
    double* dMeanCell
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadFields_32(handle, rho, rhoUx, rhoUy, rhoUz, rhoE, Ux, Uy, Uz, p, Tgas, epsS, rhoUsx, rhoUsy, rhoUsz, rhoEs, rhoDs, rhoHp, Usx, Usy, Usz, theta, Tp, dMeanCell) : ugkwpGpuResidentStrictDownloadFields_64(handle, rho, rhoUx, rhoUy, rhoUz, rhoE, Ux, Uy, Uz, p, Tgas, epsS, rhoUsx, rhoUsy, rhoUsz, rhoEs, rhoDs, rhoHp, Usx, Usy, Usz, theta, Tp, dMeanCell);
}

extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(
    void* handle,
    int nCells,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature_32(handle, nCells, updateByCell) : ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature_64(handle, nCells, updateByCell);
}

extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(
    void* handle,
    int nCells,
    double* particleMassKg,
    double* particleTemperatureMassKgK,
    double* particleDiameterMassKgM
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums_32(handle, nCells, particleMassKg, particleTemperatureMassKgK, particleDiameterMassKgM) : ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums_64(handle, nCells, particleMassKg, particleTemperatureMassKgK, particleDiameterMassKgM);
}

extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(
    void* handle,
    int nFaces,
    double* occupiedAreaM2
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea_32(handle, nFaces, occupiedAreaM2) : ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea_64(handle, nFaces, occupiedAreaM2);
}

extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(
    void* handle
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked_32(handle) : ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked_64(handle);
}

extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev(
    void* handle,
    double* epsGPrev
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadEpsGPrev_32(handle, epsGPrev) : ugkwpGpuResidentStrictDownloadEpsGPrev_64(handle, epsGPrev);
}

extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev(
    void* handle,
    const double* epsGPrev
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadEpsGPrev_32(handle, epsGPrev) : ugkwpGpuResidentStrictUploadEpsGPrev_64(handle, epsGPrev);
}

extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields(
    void* handle,
    double* rho,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadGasBoundaryFields_32(handle, rho, Ux, Uy, Uz, p, Tgas) : ugkwpGpuResidentStrictDownloadGasBoundaryFields_64(handle, rho, Ux, Uy, Uz, p, Tgas);
}

extern "C" int ugkwpGpuResidentStrictDownloadNut(
    void* handle,
    double* nut
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadNut_32(handle, nut) : ugkwpGpuResidentStrictDownloadNut_64(handle, nut);
}

extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictConfigureGasWallEnergyLedger_32(handle, nEnabledFaces, enabledFaceIds) : ugkwpGpuResidentStrictConfigureGasWallEnergyLedger_64(handle, nEnabledFaces, enabledFaceIds);
}

extern "C" int ugkwpGpuResidentStrictPeekGasWallEnergy(
    void* handle,
    int nFaces,
    double* gasWallEnergy
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictPeekGasWallEnergy_32(handle, nFaces, gasWallEnergy) : ugkwpGpuResidentStrictPeekGasWallEnergy_64(handle, nFaces, gasWallEnergy);
}

extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(
    void* handle,
    int firstFace,
    int nFaces,
    double* gasWallEnergyJ,
    double* particleDepositedWallEnergyJ,
    double* particleReflectedWallEnergyJ
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictPeekWallEnergyLedgerRange_32(handle, firstFace, nFaces, gasWallEnergyJ, particleDepositedWallEnergyJ, particleReflectedWallEnergyJ) : ugkwpGpuResidentStrictPeekWallEnergyLedgerRange_64(handle, firstFace, nFaces, gasWallEnergyJ, particleDepositedWallEnergyJ, particleReflectedWallEnergyJ);
}

extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(
    void* handle,
    int firstFace,
    int nFaces,
    const double* gasWallEnergyJ,
    const double* particleDepositedWallEnergyJ,
    const double* particleReflectedWallEnergyJ
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadWallEnergyLedgerRange_32(handle, firstFace, nFaces, gasWallEnergyJ, particleDepositedWallEnergyJ, particleReflectedWallEnergyJ) : ugkwpGpuResidentStrictUploadWallEnergyLedgerRange_64(handle, firstFace, nFaces, gasWallEnergyJ, particleDepositedWallEnergyJ, particleReflectedWallEnergyJ);
}

extern "C" int ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy(
    void* handle,
    int nFaces,
    double* gasWallEnergy
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy_32(handle, nFaces, gasWallEnergy) : ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy_64(handle, nFaces, gasWallEnergy);
}

extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel(
    void* handle,
    int nFaces,
    const unsigned char* candidateFaceMask,
    double sommerfeldThreshold,
    int heatTransferEnabled,
    double maximumCoverage,
    double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale,
    double contactAngleDegree,
    int wallTransientResistance,
    int nonlinearIterations,
    double meltingTemperatureK,
    double mushyRangeK,
    double latentHeatJkg,
    double solidDensityKgM3,
    double solidSpecificHeatJkgK,
    double solidThermalConductivityWmK,
    double pinningThicknessFraction,
    double interfaceResistanceM2KW
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictConfigureParticleStuckModel_32(handle, nFaces, candidateFaceMask, sommerfeldThreshold, heatTransferEnabled, maximumCoverage, depositionHeatTransferEfficiency, reflectionHeatTransferEfficiency, adhesionEnergyScale, contactAngleDegree, wallTransientResistance, nonlinearIterations, meltingTemperatureK, mushyRangeK, latentHeatJkg, solidDensityKgM3, solidSpecificHeatJkgK, solidThermalConductivityWmK, pinningThicknessFraction, interfaceResistanceM2KW) : ugkwpGpuResidentStrictConfigureParticleStuckModel_64(handle, nFaces, candidateFaceMask, sommerfeldThreshold, heatTransferEnabled, maximumCoverage, depositionHeatTransferEfficiency, reflectionHeatTransferEfficiency, adhesionEnergyScale, contactAngleDegree, wallTransientResistance, nonlinearIterations, meltingTemperatureK, mushyRangeK, latentHeatJkg, solidDensityKgM3, solidSpecificHeatJkgK, solidThermalConductivityWmK, pinningThicknessFraction, interfaceResistanceM2KW);
}

extern "C" int ugkwpGpuResidentStrictPeekParticleWallHeatLedgers(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictPeekParticleWallHeatLedgers_32(handle, nFaces, depositedWallEnergyJ, reflectedWallEnergyJ) : ugkwpGpuResidentStrictPeekParticleWallHeatLedgers_64(handle, nFaces, depositedWallEnergyJ, reflectedWallEnergyJ);
}

extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers_32(handle, nFaces, depositedWallEnergyJ, reflectedWallEnergyJ) : ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers_64(handle, nFaces, depositedWallEnergyJ, reflectedWallEnergyJ);
}

extern "C" int ugkwpGpuResidentStrictDownloadSst(
    void* handle,
    double* k,
    double* omega,
    double* nut
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadSst_32(handle, k, omega, nut) : ugkwpGpuResidentStrictDownloadSst_64(handle, k, omega, nut);
}

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror(
    void* handle,
    int nParticles,
    const double* px,
    const double* py,
    const double* pz,
    const double* pux,
    const double* puy,
    const double* puz,
    const double* pT,
    const double* pTheta,
    const double* pd,
    const double* pm,
    const int* pCellId,
    const int* pStatus,
    const unsigned char* pStuck,
    const int* pStuckFaceId,
    const float* pDepositionArea,
    const double* pContactDuration,
    const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const double* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const double* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng,
    const unsigned long long* pOrigId
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictUploadParticleRestartMirror_32(handle, nParticles, px, py, pz, pux, puy, puz, pT, pTheta, pd, pm, pCellId, pStatus, pStuck, pStuckFaceId, pDepositionArea, pContactDuration, pContactMaximumArea, pContactPeakFraction, pColdNodeSpecificEnthalpy, pColdRingSolidMass, pColdFrozenArea, pColdContactAge, pCold2DNodeSpecificEnthalpy, pCold2DRingContactAge, pCold2DFrozenArea, pRng, pOrigId) : ugkwpGpuResidentStrictUploadParticleRestartMirror_64(handle, nParticles, px, py, pz, pux, puy, puz, pT, pTheta, pd, pm, pCellId, pStatus, pStuck, pStuckFaceId, pDepositionArea, pContactDuration, pContactMaximumArea, pContactPeakFraction, pColdNodeSpecificEnthalpy, pColdRingSolidMass, pColdFrozenArea, pColdContactAge, pCold2DNodeSpecificEnthalpy, pCold2DRingContactAge, pCold2DFrozenArea, pRng, pOrigId);
}

extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror(
    void* handle,
    int* nParticles,
    int maxParticles,
    double* px,
    double* py,
    double* pz,
    double* pux,
    double* puy,
    double* puz,
    double* pT,
    double* pTheta,
    double* pd,
    double* pm,
    int* pCellId,
    int* pStatus,
    unsigned char* pStuck,
    int* pStuckFaceId,
    float* pDepositionArea,
    double* pContactDuration,
    float* pContactMaximumArea,
    float* pContactPeakFraction,
    float* pColdNodeSpecificEnthalpy,
    float* pColdRingSolidMass,
    float* pColdFrozenArea,
    double* pColdContactAge,
    float* pCold2DNodeSpecificEnthalpy,
    double* pCold2DRingContactAge,
    float* pCold2DFrozenArea,
    unsigned long long* pRng,
    unsigned long long* pOrigId
)
{
    return selectedPrecision == 32 ? ugkwpGpuResidentStrictDownloadParticleRestartMirror_32(handle, nParticles, maxParticles, px, py, pz, pux, puy, puz, pT, pTheta, pd, pm, pCellId, pStatus, pStuck, pStuckFaceId, pDepositionArea, pContactDuration, pContactMaximumArea, pContactPeakFraction, pColdNodeSpecificEnthalpy, pColdRingSolidMass, pColdFrozenArea, pColdContactAge, pCold2DNodeSpecificEnthalpy, pCold2DRingContactAge, pCold2DFrozenArea, pRng, pOrigId) : ugkwpGpuResidentStrictDownloadParticleRestartMirror_64(handle, nParticles, maxParticles, px, py, pz, pux, puy, puz, pT, pTheta, pd, pm, pCellId, pStatus, pStuck, pStuckFaceId, pDepositionArea, pContactDuration, pContactMaximumArea, pContactPeakFraction, pColdNodeSpecificEnthalpy, pColdRingSolidMass, pColdFrozenArea, pColdContactAge, pCold2DNodeSpecificEnthalpy, pCold2DRingContactAge, pCold2DFrozenArea, pRng, pOrigId);
}

extern "C" void ugkwpGpuResidentStrictRelease(void* handle)
{
    if (selectedPrecision == 32) ugkwpGpuResidentStrictRelease_32(handle); else ugkwpGpuResidentStrictRelease_64(handle);
    if (handle && liveHandles > 0) --liveHandles;
}
