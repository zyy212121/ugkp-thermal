#ifndef UGKWP_GPU_BACKEND_API_H
#define UGKWP_GPU_BACKEND_API_H

                                                                           
                                                                         

extern "C" int ugkwpGpuResidentStrictCreate
(
    int nCells, int nFaces, int nInternalFaces, int nCellPlanes,
    int particleCapacity, int maxFaceWalkHops, double injectionParcelMass,
    unsigned long long rngSeed, double gammaGas, double Rgas,
    double rhoSolid, int solveParticleTemperature, double gasMu,
    double gasPr, int dragModelId, int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX, double gravityY, double gravityZ,
    double particleDiameterFallback,
    double particleDiameterMin, double particleDiameterMax,
    double particleDiameterSigma, double injectionTheta, double rhoMin,
    double TgasMin, double epsSMin, double thetaMin, double TpMin,
    double TpMax, int collisionalPressureEnabled,
    double collisionalRestitution, double pressureKickFraction,
    int jammingPressureEnabled, double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme, int gasReconstruction, int gasLimiter,
    int gasTimeIntegrator, int gasRobustFallback, int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl, double waleCw, double smagorinskyCs,
    double maxDiffusionNumber, int csrCellLocalPathEnabled,
    int csrHeavyReductionMode, int csrHeavyAutoInterval,
    int particleBlockThreads, int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    void** handle
);

extern "C" int ugkwpGpuResidentStrictUploadMesh
(
    void* handle, const int* faceOwner, const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx, const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V, const double* Cx, const double* Cy, const double* Cz,
    const double* faceCx, const double* faceCy, const double* faceCz,
    const double* Sfx, const double* Sfy, const double* Sfz,
    const double* magSf, const double* deltaCoeffs, const double* faceWeight,
    const double* cellLength, const int* cellPlaneStart,
    const int* cellPlaneCount, const int* cellFaceId,
    const int* cellFaceNeighbor, const int* cellFaceKind,
    const double* cellFaceRestitution, const double* cellFaceTangential,
    const double* planeNx, const double* planeNy, const double* planeNz,
    const double* planeD
);

extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel
(
    void* handle, int nFaces, const unsigned char* candidateFaceMask,
    double sommerfeldThreshold, int heatTransferEnabled,
    double maximumCoverage, double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale, double wallDensityKgM3,
    double wallSpecificHeatJkgK, double wallConductivityWmK,
    double contactAngleDegree, int wallTransientResistance,
    int nonlinearIterations, double meltingTemperatureK,
    double mushyRangeK, double latentHeatJkg, double solidDensityKgM3,
    double solidSpecificHeatJkgK, double solidThermalConductivityWmK,
    double pinningThicknessFraction, double interfaceResistanceM2KW
);

extern "C" int ugkwpGpuResidentStrictUploadBoundarySources
(
    void* handle, int nBoundarySources, const int* sourceCell,
    const int* sourceFace, const double* sourcePx, const double* sourcePy,
    const double* sourcePz, const double* sourceUx, const double* sourceUy,
    const double* sourceUz, const double* sourceT, const double* sourceTheta,
    const double* sourceD, const double* sourceMassRate
);

extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet
(
    void* handle, int nFaces, const int* faceIds, double inletTemperature,
    int nPressureRows, const double* pressureTimes,
    const double* pressureValues, int nVolumeFractionRows,
    const double* volumeFractionTimes, const double* volumeFractionValues
);

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass
(
    void* handle, int* nSources, int maxSources, int* sourceFace,
    double* residualMass
);

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass
(
    void* handle, int nSources, const int* sourceFace,
    const double* residualMass
);

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields
(
    void* handle, const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix, const int* gasBoundaryUFix,
    const int* gasBoundaryPFix, const int* gasBoundaryTFix,
    const int* gasBoundaryPWave, const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf, const double* gasBoundaryRho,
    const double* gasBoundaryUx, const double* gasBoundaryUy,
    const double* gasBoundaryUz, const double* gasBoundaryP,
    const double* gasBoundaryT
);

extern "C" int ugkwpGpuResidentStrictUploadFields
(
    void* handle, const double* rho, const double* rhoUx,
    const double* rhoUy, const double* rhoUz, const double* rhoE,
    const double* Ux, const double* Uy, const double* Uz, const double* p,
    const double* Tgas
);

extern "C" int ugkwpGpuResidentStrictAdvance
(
    void* handle, double dt, double simulationTime
);

extern "C" int ugkwpGpuResidentStrictConfigureSst
(
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

extern "C" int ugkwpGpuResidentStrictDownloadSst
(
    void* handle,
    double* k,
    double* omega,
    double* nut
);

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(
    void* handle, double dt, double targetMaxCo, double scheduleTime,
    double* maxCo
);

extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly
(
    void* handle, double dt, double simulationTime
);

extern "C" int ugkwpGpuResidentStrictDownloadFields
(
    void* handle, double* rho, double* rhoUx, double* rhoUy, double* rhoUz,
    double* rhoE, double* Ux, double* Uy, double* Uz, double* p,
    double* Tgas, double* epsS, double* rhoUsx, double* rhoUsy,
    double* rhoUsz, double* rhoEs, double* rhoDs, double* rhoHp,
    double* Usx, double* Usy, double* Usz, double* theta, double* Tp,
    double* dMeanCell
);

extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields
(
    void* handle, double* rho, double* Ux, double* Uy, double* Uz,
    double* p, double* Tgas
);

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror
(
    void* handle, int nParticles, const double* px, const double* py,
    const double* pz, const double* pux, const double* puy,
    const double* puz, const double* pT, const double* pTheta,
    const double* pd, const double* pm, const int* pCellId,
    const int* pStatus, const unsigned char* pStuck,
    const int* pStuckFaceId, const float* pDepositionArea,
    const float* pContactDuration, const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const float* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const float* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng, const unsigned long long* pOrigId
);

extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror
(
    void* handle, int* nParticles, int maxParticles, double* px, double* py,
    double* pz, double* pux, double* puy, double* puz, double* pT,
    double* pTheta, double* pd, double* pm, int* pCellId, int* pStatus,
    unsigned char* pStuck, int* pStuckFaceId, float* pDepositionArea,
    float* pContactDuration, float* pContactMaximumArea,
    float* pContactPeakFraction,
    float* pColdNodeSpecificEnthalpy,
    float* pColdRingSolidMass,
    float* pColdFrozenArea,
    float* pColdContactAge,
    float* pCold2DNodeSpecificEnthalpy,
    float* pCold2DRingContactAge,
    float* pCold2DFrozenArea,
    unsigned long long* pRng,
    unsigned long long* pOrigId
);

extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev
(
    void* handle, double* epsGPrev
);

extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev
(
    void* handle, const double* epsGPrev
);

extern "C" int ugkwpGpuResidentStrictDownloadNut
(
    void* handle, double* nut
);

extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers
(
    void* handle, int nFaces, double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);

extern "C" void ugkwpGpuResidentStrictRelease(void* handle);
extern "C" const char* ugkwpGpuResidentStrictLastError();

#endif
