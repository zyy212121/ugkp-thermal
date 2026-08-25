#ifndef GpuResidentStrict_H
#define GpuResidentStrict_H

#include "fvCFD.H"
#include "GpuSchedulingConfiguration.H"
#include "../../FSH/gpu/GpuFshParticleRestart.H"
#include "GpuGasBoundarySpec.H"
#include "GpuParticleRadiationMath.H"
#include "SHA1.H"
#include "cyclicPolyPatch.H"
#include "emptyPolyPatch.H"
#include "inletOutletFvPatchFields.H"
#include "processorPolyPatch.H"
#include "symmetryPlanePolyPatch.H"
#include "symmetryPolyPatch.H"
#include "wallPolyPatch.H"
#include "wedgePolyPatch.H"
#include "waveTransmissiveFvPatchField.H"

#ifdef UGKWP_USE_CUDA
#include "GpuBackendApi.H"
#endif

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

namespace Foam
{

#ifdef UGKWP_USE_CUDA

extern "C" int ugkwpGpuResidentStrictCreate
(
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

extern "C" int ugkwpGpuResidentStrictUploadMesh
(
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

extern "C" int ugkwpGpuResidentStrictUploadBoundarySources
(
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

extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel
(
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

extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
);

extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet
(
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

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass
(
    void* handle,
    int* nSources,
    int maxSources,
    int* sourceFace,
    double* residualMass
);

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass
(
    void* handle,
    int nSources,
    const int* sourceFace,
    const double* residualMass
);

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields
(
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

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch
(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* temperatures
);

extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger
(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
);

extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange
(
    void* handle,
    int firstFace,
    int nFaces,
    double* gasWallEnergyJ,
    double* particleDepositedWallEnergyJ,
    double* particleReflectedWallEnergyJ
);

extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange
(
    void* handle,
    int firstFace,
    int nFaces,
    const double* gasWallEnergyJ,
    const double* particleDepositedWallEnergyJ,
    const double* particleReflectedWallEnergyJ
);

extern "C" int ugkwpGpuResidentStrictPeekGasWallEnergy
(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);

extern "C" int ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy
(
    void* handle,
    int nFaces,
    double* gasWallEnergy
);

extern "C" int ugkwpGpuResidentStrictUploadFields
(
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

extern "C" int ugkwpGpuResidentStrictAdvance
(
    void* handle,
    double dt,
    double simulationTime
);

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(
    void* handle,
    double dt,
    double targetMaxCo,
    double scheduleTime,
    double* maxCo
);

extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly
(
    void* handle,
    double dt,
    double simulationTime
);

extern "C" int ugkwpGpuResidentStrictDownloadFields
(
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

extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields
(
    void* handle,
    double* rho,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas
);

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror
(
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
    const float* pContactDuration,
    const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const float* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const float* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng,
    const unsigned long long* pOrigId
);

extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror
(
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
    float* pContactDuration,
    float* pContactMaximumArea,
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
    void* handle,
    double* epsGPrev
);

extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev
(
    void* handle,
    const double* epsGPrev
);

extern "C" int ugkwpGpuResidentStrictDownloadNut
(
    void* handle,
    double* nut
);

extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers
(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);

extern "C" int ugkwpGpuResidentStrictPeekParticleWallHeatLedgers
(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
);

extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature
(
    void* handle,
    int nCells,
    const gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
);

extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums
(
    void* handle,
    int nCells,
    double* particleMassKg,
    double* particleTemperatureMassKgK,
    double* particleDiameterMassKgM
);

extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea
(
    void* handle,
    int nFaces,
    double* occupiedAreaM2
);

extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked
(
    void* handle
);

extern "C" void ugkwpGpuResidentStrictRelease(void* handle);
extern "C" const char* ugkwpGpuResidentStrictLastError();

struct GpuThermalRestartMirrorMetadata
{
    fileName particleFile;
    fileName epsGPrevFile;
    fileName sourceResidualFile;
    label particleCount{0};
    label epsGPrevCount{0};
    label sourceResidualCount{0};
    word particleSha1;
    word epsGPrevSha1;
    word sourceResidualSha1;
};

class GpuResidentStrictBase
{
protected:
    void* handle_;
    label hostMirrorCells_;
    label hostMirrorFaces_;
    const fvMesh* residentMesh_;
    scalar injectionParcelMass_;
    scalar legacyRestartParcelMass_;
    bool particleWallHeatTransferEnabled_;
    word particleStuckWallHeatFluxFieldName_;
    word particleReflectedWallHeatFluxFieldName_;
    List<unsigned char> particleStuckCandidateFaceMask_;
    scalar residentParticleHeatFactorJkgK_;
    scalar residentRhoSolid_;
    scalar residentTpMin_;
    scalar residentTpMax_;
    label residentParticleCapacity_;
    label gasWallEnergyMeshFaces_;
    labelList gasWallEnergyFaceIds_;
    labelList gasWallEnergyPatchSizes_;
    bool gasWallEnergyConfigured_;
    bool particleStateCellLocalPacked_;

public:
    GpuResidentStrictBase()
    :
        handle_(nullptr),
        hostMirrorCells_(0),
        hostMirrorFaces_(0),
        residentMesh_(nullptr),
        injectionParcelMass_(0),
        legacyRestartParcelMass_(0),
        particleWallHeatTransferEnabled_(false),
        particleStuckWallHeatFluxFieldName_("particleStuckWallHeatFlux"),
        particleReflectedWallHeatFluxFieldName_
        (
            "particleReflectedWallHeatFlux"
        ),
        particleStuckCandidateFaceMask_(),
        residentParticleHeatFactorJkgK_(0),
        residentRhoSolid_(0),
        residentTpMin_(0),
        residentTpMax_(0),
        residentParticleCapacity_(0),
        gasWallEnergyMeshFaces_(0),
        gasWallEnergyFaceIds_(),
        gasWallEnergyPatchSizes_(),
        gasWallEnergyConfigured_(false),
        particleStateCellLocalPacked_(false)
    {}

    ~GpuResidentStrictBase()
    {
        if (handle_ != nullptr)
        {
            ugkwpGpuResidentStrictRelease(handle_);
            handle_ = nullptr;
        }
    }

    GpuResidentStrictBase(const GpuResidentStrictBase&) = delete;
    void operator=(const GpuResidentStrictBase&) = delete;

    bool particleWallHeatTransferEnabled() const
    {
        return particleWallHeatTransferEnabled_;
    }

    bool particleWallContactEnabledOnPatch
    (
        const fvMesh& mesh,
        const label patchI
    ) const
    {
        requireInitialised("particleWallContactEnabledOnPatch");
        if
        (
            residentMesh_ != &mesh
         || patchI < 0
         || patchI >= mesh.boundaryMesh().size()
        )
        {
            FatalErrorInFunction
                << "Particle-wall contact patch query requires a valid patch "
                << "of the resident mesh." << exit(FatalError);
        }
        if (!particleWallHeatTransferEnabled_)
        {
            return false;
        }
        const polyPatch& patch = mesh.boundaryMesh()[patchI];
        bool anyCandidate = false;
        bool anyNonCandidate = false;
        forAll(patch, faceI)
        {
            const bool candidate =
                particleStuckCandidateFaceMask_[patch.start() + faceI] != 0;
            anyCandidate = anyCandidate || candidate;
            anyNonCandidate = anyNonCandidate || !candidate;
        }
        if (anyCandidate && anyNonCandidate)
        {
            FatalErrorInFunction
                << "A wall patch cannot be partially configured for the "
                << "stuck contact model: " << patch.name()
                << exit(FatalError);
        }
        return anyCandidate;
    }

    const word& particleStuckWallHeatFluxFieldName() const
    {
        return particleStuckWallHeatFluxFieldName_;
    }

    const word& particleReflectedWallHeatFluxFieldName() const
    {
        return particleReflectedWallHeatFluxFieldName_;
    }

    void downloadAndResetParticleWallHeatLedgers
    (
        scalarField& depositedEnergy,
        scalarField& reflectedEnergy
    )
    {
        requireInitialised("downloadAndResetParticleWallHeatLedgers");
        if (!particleWallHeatTransferEnabled_)
        {
            depositedEnergy.setSize(0);
            reflectedEnergy.setSize(0);
            return;
        }
        depositedEnergy.setSize(hostMirrorFaces_);
        reflectedEnergy.setSize(hostMirrorFaces_);
        check
        (
            ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers
            (
                handle_,
                int(depositedEnergy.size()),
                depositedEnergy.begin(),
                reflectedEnergy.begin()
            ),
            "download and reset GPU particle wall heat ledgers"
        );
    }

    void configureGasWallEnergyLedger
    (
        const fvMesh& mesh,
        const labelList& coupledWallPatchIds
    )
    {
        requireInitialised("configureGasWallEnergyLedger");
        if (gasWallEnergyConfigured_ || residentMesh_ != &mesh)
        {
            FatalErrorInFunction
                << "Gas-wall ledger is already configured or the mesh is "
                << "not the resident mesh." << exit(FatalError);
        }
        label nConfigured = 0;
        forAll(coupledWallPatchIds, patchListI)
        {
            const label patchI = coupledWallPatchIds[patchListI];
            if
            (
                patchI < 0
             || patchI >= mesh.boundaryMesh().size()
             || !isA<wallPolyPatch>(mesh.boundaryMesh()[patchI])
            )
            {
                FatalErrorInFunction
                    << "Invalid coupled gas-wall patch " << patchI
                    << exit(FatalError);
            }
            nConfigured += mesh.boundaryMesh()[patchI].size();
        }
        labelList faceIds(nConfigured);
        labelList patchSizes(coupledWallPatchIds.size());
        label faceListI = 0;
        forAll(coupledWallPatchIds, patchListI)
        {
            const polyPatch& patch =
                mesh.boundaryMesh()[coupledWallPatchIds[patchListI]];
            patchSizes[patchListI] = patch.size();
            forAll(patch, faceI)
            {
                faceIds[faceListI++] = patch.start() + faceI;
            }
        }
        check
        (
            ugkwpGpuResidentStrictConfigureGasWallEnergyLedger
            (
                handle_,
                int(faceIds.size()),
                faceIds.empty()
              ? nullptr
              : reinterpret_cast<const int*>(faceIds.begin())
            ),
            "configure GPU gas-wall energy ledger"
        );
        gasWallEnergyMeshFaces_ = mesh.nFaces();
        gasWallEnergyFaceIds_.transfer(faceIds);
        gasWallEnergyPatchSizes_.transfer(patchSizes);
        gasWallEnergyConfigured_ = !gasWallEnergyFaceIds_.empty();
    }

    const labelList& configuredGasWallEnergyFaceIds() const
    {
        return gasWallEnergyFaceIds_;
    }

    void peekPendingWallEnergyLedgers
    (
        scalarField& gasEnergy,
        scalarField& depositedEnergy,
        scalarField& reflectedEnergy
    ) const
    {
        requireInitialised("peekPendingWallEnergyLedgers");
        const label nConfigured = gasWallEnergyFaceIds_.size();
        gasEnergy.setSize(nConfigured, scalar(0));
        depositedEnergy.setSize(nConfigured, scalar(0));
        reflectedEnergy.setSize(nConfigured, scalar(0));
        label offset = 0;
        forAll(gasWallEnergyPatchSizes_, patchListI)
        {
            const label patchSize = gasWallEnergyPatchSizes_[patchListI];
            if (patchSize == 0)
            {
                continue;
            }
            const label firstFace = gasWallEnergyFaceIds_[offset];
            for (label faceI = 1; faceI < patchSize; ++faceI)
            {
                if (gasWallEnergyFaceIds_[offset + faceI] != firstFace + faceI)
                {
                    FatalErrorInFunction
                        << "Configured coupled-patch faces are not contiguous."
                        << exit(FatalError);
                }
            }
            check
            (
                ugkwpGpuResidentStrictPeekWallEnergyLedgerRange
                (
                    handle_,
                    int(firstFace),
                    int(patchSize),
                    gasEnergy.begin() + offset,
                    depositedEnergy.begin() + offset,
                    reflectedEnergy.begin() + offset
                ),
                "peek compact GPU pending wall-energy ledgers"
            );
            offset += patchSize;
        }
        if (offset != nConfigured)
        {
            FatalErrorInFunction
                << "Configured gas-wall ledger patch sizes are inconsistent."
                << exit(FatalError);
        }
    }

    void restorePendingWallEnergyLedgers
    (
        const scalarField& gasEnergy,
        const scalarField& depositedEnergy,
        const scalarField& reflectedEnergy
    )
    {
        requireInitialised("restorePendingWallEnergyLedgers");
        const label nConfigured = gasWallEnergyFaceIds_.size();
        if
        (
            gasEnergy.size() != nConfigured
         || depositedEnergy.size() != nConfigured
         || reflectedEnergy.size() != nConfigured
        )
        {
            FatalErrorInFunction
                << "Pending wall-energy ledger size does not match the "
                << "configured coupled faces." << exit(FatalError);
        }
        label offset = 0;
        forAll(gasWallEnergyPatchSizes_, patchListI)
        {
            const label patchSize = gasWallEnergyPatchSizes_[patchListI];
            if (patchSize == 0)
            {
                continue;
            }
            check
            (
                ugkwpGpuResidentStrictUploadWallEnergyLedgerRange
                (
                    handle_,
                    int(gasWallEnergyFaceIds_[offset]),
                    int(patchSize),
                    gasEnergy.begin() + offset,
                    depositedEnergy.begin() + offset,
                    reflectedEnergy.begin() + offset
                ),
                "restore compact GPU pending wall-energy ledgers"
            );
            offset += patchSize;
        }
        if (offset != nConfigured)
        {
            FatalErrorInFunction
                << "Configured gas-wall ledger patch sizes are inconsistent."
                << exit(FatalError);
        }
    }

    List<scalarField> peekGasWallEnergy() const
    {
        requireInitialised("peekGasWallEnergy");
        List<scalarField> grouped(gasWallEnergyPatchSizes_.size());
        forAll(grouped, patchI)
        {
            grouped[patchI].setSize(gasWallEnergyPatchSizes_[patchI]);
        }
        if (gasWallEnergyFaceIds_.empty())
        {
            return grouped;
        }
        scalarField full(gasWallEnergyMeshFaces_, scalar(0));
        check
        (
            ugkwpGpuResidentStrictPeekGasWallEnergy
            (
                handle_, int(full.size()), full.begin()
            ),
            "peek GPU gas-wall energy ledger"
        );
        label configuredI = 0;
        forAll(grouped, patchI)
        {
            forAll(grouped[patchI], localFaceI)
            {
                grouped[patchI][localFaceI] =
                    full[gasWallEnergyFaceIds_[configuredI++]];
            }
        }
        return grouped;
    }

    void downloadAndResetGasWallEnergy(scalarField& configuredEnergy)
    {
        requireInitialised("downloadAndResetGasWallEnergy");
        if (gasWallEnergyFaceIds_.empty())
        {
            configuredEnergy.setSize(0);
            return;
        }
        scalarField full(gasWallEnergyMeshFaces_, scalar(0));
        check
        (
            ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy
            (
                handle_, int(full.size()), full.begin()
            ),
            "download and reset GPU gas-wall energy ledger"
        );
        configuredEnergy.setSize(gasWallEnergyFaceIds_.size());
        forAll(gasWallEnergyFaceIds_, faceListI)
        {
            configuredEnergy[faceListI] =
                full[gasWallEnergyFaceIds_[faceListI]];
        }
    }

    scalarField peekParticleWallEnergy() const
    {
        scalarField deposited;
        scalarField reflected;
        peekParticleWallHeatLedgers(deposited, reflected);
        deposited += reflected;
        return deposited;
    }

    void peekParticleWallHeatLedgers
    (
        scalarField& depositedEnergy,
        scalarField& reflectedEnergy
    ) const
    {
        requireInitialised("peekParticleWallHeatLedgers");
        if (!particleWallHeatTransferEnabled_)
        {
            depositedEnergy.setSize(0);
            reflectedEnergy.setSize(0);
            return;
        }
        depositedEnergy.setSize(hostMirrorFaces_, scalar(0));
        reflectedEnergy.setSize(hostMirrorFaces_, scalar(0));
        check
        (
            ugkwpGpuResidentStrictPeekParticleWallHeatLedgers
            (
                handle_,
                int(hostMirrorFaces_),
                depositedEnergy.begin(),
                reflectedEnergy.begin()
            ),
            "peek GPU particle-wall heat ledgers"
        );
    }

    void downloadAndResetParticleWallEnergy(scalarField& totalEnergy)
    {
        scalarField deposited;
        scalarField reflected;
        downloadAndResetParticleWallHeatLedgers(deposited, reflected);
        if (deposited.empty())
        {
            totalEnergy.setSize(0);
            return;
        }
        deposited += reflected;
        totalEnergy.transfer(deposited);
    }

    void updateSolidWallTemperatures
    (
        const fvMesh& mesh,
        const volScalarField& Tgas,
        const labelList& coupledWallPatchIds
    )
    {
        requireInitialised("updateSolidWallTemperatures");
        if (residentMesh_ != &mesh)
        {
            FatalErrorInFunction
                << "Wall temperatures require the resident mesh."
                << exit(FatalError);
        }
        forAll(coupledWallPatchIds, patchListI)
        {
            const label patchI = coupledWallPatchIds[patchListI];
            const polyPatch& patch = mesh.boundaryMesh()[patchI];
            const scalarField values(Tgas.boundaryField()[patchI]);
            check
            (
                ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch
                (
                    handle_,
                    int(patch.start()),
                    int(patch.size()),
                    values.begin()
                ),
                "upload coupled gas-wall temperature patch"
            );
        }
    }

    void updateParticleWallEffusivities
    (
        const fvMesh& mesh,
        const List<scalarField>& wallEffusivity,
        const labelList& coupledWallPatchIds
    )
    {
        requireInitialised("updateParticleWallEffusivities");
        if (residentMesh_ != &mesh || wallEffusivity.size() != mesh.boundary().size())
        {
            FatalErrorInFunction
                << "Particle-wall effusivity requires the resident mesh and "
                << "one field slot per boundary patch."
                << exit(FatalError);
        }
        forAll(coupledWallPatchIds, patchListI)
        {
            const label patchI = coupledWallPatchIds[patchListI];
            const polyPatch& patch = mesh.boundaryMesh()[patchI];
            if (!particleWallContactEnabledOnPatch(mesh, patchI))
            {
                FatalErrorInFunction
                    << "Particle-wall effusivity may only be uploaded for a "
                    << "configured stuck contact patch: " << patch.name()
                    << exit(FatalError);
            }
            const scalarField& values = wallEffusivity[patchI];
            if (values.size() != patch.size())
            {
                FatalErrorInFunction
                    << "Mapped wall effusivity size mismatch on patch "
                    << patch.name() << exit(FatalError);
            }
            forAll(values, faceI)
            {
                if (!finiteMirrorScalar(values[faceI]) || values[faceI] <= 0)
                {
                    FatalErrorInFunction
                        << "Mapped wall effusivity must be finite and positive "
                        << "on patch " << patch.name() << exit(FatalError);
                }
            }
            check
            (
                ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
                (
                    handle_, int(patch.start()), int(patch.size()), values.begin()
                ),
                "upload CHT particle-wall effusivity patch"
            );
        }
    }

    scalar particleHeatFactorJkgK() const
    {
        return residentParticleHeatFactorJkgK_;
    }

    scalar rhoSolid() const
    {
        return residentRhoSolid_;
    }

    scalar particleTemperatureMin() const
    {
        return residentTpMin_;
    }

    scalar particleTemperatureMax() const
    {
        return residentTpMax_;
    }

    label particleCapacity() const
    {
        requireInitialised("particleCapacity");
        return residentParticleCapacity_;
    }

    label particleCount() const
    {
        requireInitialised("particleCount");
        int count = 0;
        check
        (
            ugkwpGpuResidentStrictDownloadParticleRestartMirror
            (
                handle_, &count, 0,
                nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                nullptr, nullptr
            ),
            "query GPU resident particle count"
        );
        return label(count);
    }

    void applyParticleRadiationEnergy
    (
        const fvMesh& mesh,
        const scalarField& deltaParticleRadiationEnergyJ,
        const scalarField& frozenParticleMassInCellKg,
        const scalarField& frozenParticleMeanTemperatureK,
        const scalar particleHeatFactorJkgK
    )
    {
        requireInitialised("applyParticleRadiationEnergy");
        if
        (
            residentMesh_ != &mesh
         || deltaParticleRadiationEnergyJ.size() != mesh.nCells()
         || frozenParticleMassInCellKg.size() != mesh.nCells()
         || frozenParticleMeanTemperatureK.size() != mesh.nCells()
         || particleHeatFactorJkgK != residentParticleHeatFactorJkgK_
        )
        {
            FatalErrorInFunction
                << "Particle-radiation input is inconsistent with the "
                << "resident state." << exit(FatalError);
        }
        List<gpuThermal::RadiationAffineTemperatureUpdate> updates
        (
            mesh.nCells()
        );
        forAll(updates, cellI)
        {
            updates[cellI] = gpuThermal::radiationAffineTemperatureUpdate
            (
                deltaParticleRadiationEnergyJ[cellI],
                frozenParticleMassInCellKg[cellI],
                particleHeatFactorJkgK,
                frozenParticleMeanTemperatureK[cellI],
                residentTpMin_,
                residentTpMax_
            );
        }
        check
        (
            ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature
            (
                handle_, int(updates.size()), updates.begin()
            ),
            "apply conservative particle-radiation temperature update"
        );
    }

    void downloadMobileParticleRadiationMaterial
    (
        const fvMesh& mesh,
        scalarField& particleMassKg,
        scalarField& particleMeanTemperatureK,
        scalarField& representativeDiameterM
    ) const
    {
        requireInitialised("downloadMobileParticleRadiationMaterial");
        if (residentMesh_ != &mesh || !particleStateCellLocalPacked_)
        {
            FatalErrorInFunction
                << "Mobile particle radiation material requires the packed "
                << "post-step resident state." << exit(FatalError);
        }
        particleMassKg.setSize(mesh.nCells(), scalar(0));
        scalarField temperatureMassKgK(mesh.nCells(), scalar(0));
        scalarField diameterMassKgM(mesh.nCells(), scalar(0));
        check
        (
            ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums
            (
                handle_,
                int(mesh.nCells()),
                particleMassKg.begin(),
                temperatureMassKgK.begin(),
                diameterMassKgM.begin()
            ),
            "download mobile-only particle radiation material"
        );
        particleMeanTemperatureK.setSize(mesh.nCells(), residentTpMin_);
        representativeDiameterM.setSize(mesh.nCells(), scalar(0));
        forAll(particleMassKg, cellI)
        {
            const scalar mass = particleMassKg[cellI];
            if (!std::isfinite(mass) || mass < scalar(0))
            {
                FatalErrorInFunction
                    << "Invalid mobile particle radiation mass in cell "
                    << cellI << exit(FatalError);
            }
            if (mass > scalar(0))
            {
                particleMeanTemperatureK[cellI] =
                    temperatureMassKgK[cellI]/mass;
                representativeDiameterM[cellI] =
                    diameterMassKgM[cellI]/mass;
                if
                (
                    !std::isfinite(particleMeanTemperatureK[cellI])
                 || !std::isfinite(representativeDiameterM[cellI])
                 || representativeDiameterM[cellI] <= scalar(0)
                )
                {
                    FatalErrorInFunction
                        << "Invalid mobile particle radiation material in "
                        << "cell " << cellI << exit(FatalError);
                }
            }
        }
    }

    scalarField particleWallOccupiedContactArea() const
    {
        requireInitialised("particleWallOccupiedContactArea");
        scalarField occupiedArea(hostMirrorFaces_, scalar(0));
        check
        (
            ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea
            (
                handle_, int(occupiedArea.size()), occupiedArea.begin()
            ),
            "download instantaneous particle-wall occupied contact area"
        );
        return occupiedArea;
    }

    void refreshParticleEnthalpyAfterRadiation()
    {
        requireInitialised("refreshParticleEnthalpyAfterRadiation");
        if (!particleStateCellLocalPacked_)
        {
            FatalErrorInFunction
                << "Particle radiation requires the packed post-step state."
                << exit(FatalError);
        }
        check
        (
            ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(handle_),
            "refresh packed particle enthalpy after radiation"
        );
    }

protected:
    void initialiseBase
    (
        const Time& runTime,
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const volScalarField& epsS,
        const volVectorField& rhoUs,
        const volScalarField& rhoEs,
        const volScalarField& rhoDs,
        const volScalarField& rhoHp,
        const volVectorField& Us,
        const volScalarField& theta,
        const volScalarField& Tp,
        const volScalarField& dMeanCell,
        const scalar injectionParcelMass,
        const scalar legacyRestartParcelMass,
        const label particleCapacity,
        const label maxFaceWalkHops,
        const unsigned long long rngSeed,
        const scalar gammaGas,
        const scalar RgasValue,
        const bool strictOpenFoamGasBoundaries,
        const scalar rhoSolid,
        const bool solveParticleTemperature,
        const label particleGasHeatTransferModelId,
        const scalar gasMu,
        const scalar gasPr,
        const label dragModelId,
        const scalar dragResidualRe,
        const scalar gravityX,
        const scalar gravityY,
        const scalar gravityZ,
        const label gasFluxScheme,
        const label gasReconstruction,
        const label gasLimiter,
        const label gasTimeIntegrator,
        const bool gasRobustFallback,
        const label gasTurbulenceModel,
        const scalar lesDeltaCoeff,
        const scalar turbulentPrandtl,
        const scalar waleCw,
        const scalar smagorinskyCs,
        const scalar maxDiffusionNumber,
        const scalar particleDiameterFallback,
        const scalar particleDiameterMin,
        const scalar particleDiameterMax,
        const scalar particleDiameterSigma,
        const scalar injectionTheta,
        const scalar injectionTp,
        const scalar rhoMin,
        const scalar TgasMin,
        const scalar epsSMin,
        const scalar thetaMin,
        const scalar TpMin,
        const scalar TpMax,
        const bool enableParticleModule,
        const GpuSchedulingConfiguration& scheduling,
        const dictionary* particleWallProps
    )
    {
        ensureStorage();
        ensureFieldSizes(mesh, rho, rhoU, rhoE, U, p, Tgas, epsS, rhoUs, rhoEs, rhoDs, rhoHp, Us, theta, Tp, dMeanCell);

        const label nCells = mesh.nCells();
        const label nFaces = mesh.nFaces();
        const label nInternalFaces = mesh.nInternalFaces();
        label nCellPlanes = 0;
        forAll(mesh.cells(), cellI)
        {
            nCellPlanes += mesh.cells()[cellI].size();
        }
        hostMirrorCells_ = nCells;
        hostMirrorFaces_ = nFaces;
        residentMesh_ = &mesh;
        injectionParcelMass_ = injectionParcelMass;
        legacyRestartParcelMass_ = legacyRestartParcelMass;

        if (particleCapacity < 0)
        {
            FatalErrorInFunction
                << "gpuResidentParticleCapacity must be non-negative. Current value = "
                << particleCapacity << exit(FatalError);
        }

        if (maxFaceWalkHops <= 0)
        {
            FatalErrorInFunction
                << "gpuResidentMaxFaceWalkHops must be positive. Current value = "
                << maxFaceWalkHops << exit(FatalError);
        }

        if (!enableParticleModule && particleCapacity != 0)
        {
            FatalErrorInFunction
                << "GpuGasResidentSolver must not allocate particle storage. "
                << "particleCapacity = " << particleCapacity
                << exit(FatalError);
        }

        void* raw = nullptr;
        const bool collisionalPressureEnabled =
            enableParticleModule
         && particleWallProps != nullptr
         && particleWallProps->lookupOrDefault<bool>
            (
                "gpuResidentCollisionalPressure",
                false
            );
        const scalar collisionalRestitution =
            particleWallProps == nullptr
          ? scalar(0.9)
          : particleWallProps->lookupOrDefault<scalar>
            (
                "gpuResidentCollisionalRestitution",
                scalar(0.9)
            );
        const scalar pressureKickFraction =
            particleWallProps == nullptr
          ? scalar(0.25)
          : particleWallProps->lookupOrDefault<scalar>
            (
                "gpuResidentPressureKickFraction",
                scalar(0.25)
            );
        const bool jammingPressureEnabled =
            enableParticleModule
         && particleWallProps != nullptr
         && particleWallProps->lookupOrDefault<bool>
            (
                "gpuResidentJammingPressure",
                false
            );
        const scalar packingFraction =
            particleWallProps == nullptr
          ? scalar(0.63)
          : particleWallProps->lookupOrDefault<scalar>
            (
                "gpuResidentPackingFraction",
                scalar(0.63)
            );
        const label packingProjectionIterations =
            particleWallProps == nullptr
          ? label(20)
          : particleWallProps->lookupOrDefault<label>
            (
                "gpuResidentPackingProjectionIterations",
                label(20)
            );
                                                                      
                                                                         
        if
        (
            !finiteMirrorScalar(collisionalRestitution)
         || collisionalRestitution < scalar(0)
         || collisionalRestitution > scalar(1)
        )
        {
            FatalErrorInFunction
                << "gpuResidentCollisionalRestitution must be finite and in "
                << "[0, 1]. Current value = " << collisionalRestitution
                << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(pressureKickFraction)
         || pressureKickFraction <= scalar(0)
         || pressureKickFraction > scalar(1)
        )
        {
            FatalErrorInFunction
                << "gpuResidentPressureKickFraction must be finite and in "
                << "(0, 1]. Current value = " << pressureKickFraction
                << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(packingFraction)
         || packingFraction <= scalar(0)
         || packingFraction >= scalar(1)
        )
        {
            FatalErrorInFunction
                << "Invalid GPU resident mobile packing-projection parameter: "
                << "packingFraction=" << packingFraction
                << exit(FatalError);
        }
        if (packingProjectionIterations < 1)
        {
            FatalErrorInFunction
                << "gpuResidentPackingProjectionIterations must be a "
                << "positive integer. Current value = "
                << packingProjectionIterations
                << exit(FatalError);
        }

        const bool csrCellLocalPathEnabled = scheduling.csrCellLocalPath;
        const int csrHeavyReductionMode = int(scheduling.heavyReductionMode);
        const bool csrWarpAggregatedBinning =
            scheduling.warpAggregatedBinning;

        if (enableParticleModule)
        {
            Info<< "Particle cell path: csrCellLocalPath="
                << csrCellLocalPathEnabled
                << " heavyReductionMode=" << csrHeavyReductionMode
                << " warpAggregatedBinning=" << csrWarpAggregatedBinning
                << " B2=" << scheduling.particleBlockThreads
                << " B3=" << scheduling.reductionBlockThreads
                << " heavyAutoInterval="
                << scheduling.heavyReductionAutoInterval
                << endl;
        }

        const int createRc = ugkwpGpuResidentStrictCreate
        (
            int(nCells),
            int(nFaces),
            int(nInternalFaces),
            int(nCellPlanes),
            int(particleCapacity),
            int(maxFaceWalkHops),
            injectionParcelMass,
            rngSeed,
            gammaGas,
            RgasValue,
            rhoSolid,
            solveParticleTemperature ? 1 : 0,
            gasMu,
            gasPr,
            int(dragModelId),
            int(particleGasHeatTransferModelId),
            dragResidualRe,
            gravityX,
            gravityY,
            gravityZ,
            particleDiameterFallback,
            particleDiameterMin,
            particleDiameterMax,
            particleDiameterSigma,
            injectionTheta,
            rhoMin,
            TgasMin,
            epsSMin,
            thetaMin,
            TpMin,
            TpMax,
            collisionalPressureEnabled ? 1 : 0,
            collisionalRestitution,
            pressureKickFraction,
            jammingPressureEnabled ? 1 : 0,
            packingFraction,
            int(packingProjectionIterations),
            int(gasFluxScheme),
            int(gasReconstruction),
            int(gasLimiter),
            int(gasTimeIntegrator),
            gasRobustFallback ? 1 : 0,
            int(gasTurbulenceModel),
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            maxDiffusionNumber,
            csrCellLocalPathEnabled ? 1 : 0,
            csrHeavyReductionMode,
            int(scheduling.heavyReductionAutoInterval),
            int(scheduling.particleBlockThreads),
            int(scheduling.reductionBlockThreads),
            csrWarpAggregatedBinning ? 1 : 0,
            &raw
        );

        check(createRc, "create GPU resident strict state");
        handle_ = raw;

        uploadMesh(mesh, nCellPlanes, particleWallProps);
        if (enableParticleModule)
        {
            configureParticleStuckModel
            (
                mesh,
                *particleWallProps,
                solveParticleTemperature
            );
        }
        uploadGasBoundaryFields
        (
            mesh,
            rho,
            U,
            p,
            Tgas,
            RgasValue,
            strictOpenFoamGasBoundaries
        );
        if (enableParticleModule)
        {
            uploadBoundarySources
            (
                mesh,
                epsS,
                U,
                Tp,
                theta,
                dMeanCell,
                rhoSolid,
                particleDiameterFallback,
                thetaMin,
                injectionTheta,
                injectionTp,
                TpMin
            );
        }
        if (enableParticleModule)
        {
            const fileName restartPath = particleRestartFile(runTime);
            const bool hasParticleRestart = isFile(restartPath);
            label restartParticleCount = -1;
            if (hasParticleRestart)
            {
                std::ifstream restartHeaderStream(restartPath.c_str());
                string restartHeader;
                if (!(restartHeaderStream >> restartHeader))
                {
                    FatalErrorInFunction
                        << "Invalid GPU resident particle restart mirror: "
                        << restartPath << exit(FatalError);
                }
                if
                (
                    restartHeader == gpuFshRestart::formatName
                 || restartHeader == gpuFshRestart::schema4FormatName
                 || restartHeader == gpuFshRestart::schema3FormatName
                 || restartHeader == gpuFshRestart::schema2FormatName
                 || restartHeader == gpuFshRestart::legacyFormatName
                 || restartHeader == "UGKP_PARTICLES_SCHEMA2"
                 || restartHeader == "UGKP_PARTICLES_SCHEMA3"
                )
                {
                    if (!(restartHeaderStream >> restartParticleCount))
                    {
                        FatalErrorInFunction
                            << "Invalid versioned GPU resident particle restart count: "
                            << restartPath << exit(FatalError);
                    }
                }
                else
                {
                    std::istringstream countStream(restartHeader);
                    if
                    (
                        !(countStream >> restartParticleCount)
                     || !countStream.eof()
                    )
                    {
                        FatalErrorInFunction
                            << "Invalid legacy GPU resident particle restart "
                            << "count: " << restartPath << exit(FatalError);
                    }
                }
            }

            scalar initialSolidVolume = 0;
            const scalarField& epsI = epsS.primitiveField();
            const scalarField& V = mesh.V();
            forAll(epsI, cellI)
            {
                initialSolidVolume += max(epsI[cellI], scalar(0))*V[cellI];
            }
            reduce(initialSolidVolume, sumOp<scalar>());

            if
            (
                initialSolidVolume > SMALL
             && (!hasParticleRestart || restartParticleCount == 0)
            )
            {
                FatalErrorInFunction
                    << "GPU resident particle mode uses particles as the solid "
                    << "primary state. Non-zero initial epsilonS was found, "
                    << "but the GPU particle restart is absent or empty. "
                    << "Provide a non-empty gpuResidentStrictParticles.dat; "
                    << "macroscopic solid fields are never converted back into "
                    << "particles."
                    << exit(FatalError);
            }
        }

        uploadFields(rho, rhoU, rhoE, U, p, Tgas);

        if (enableParticleModule)
        {
            readParticleRestartMirrorIfPresent(runTime);
                                                                             
                                                                          
                                                                             
            readEpsGPrevRestartMirrorIfPresent(runTime);
            readSourceResidualRestartMirrorIfPresent(runTime);
        }

    }

    void initialiseGasBase
    (
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const label maxFaceWalkHops,
        const unsigned long long rngSeed,
        const scalar gammaGas,
        const scalar RgasValue,
        const bool strictOpenFoamGasBoundaries,
        const scalar gasMu,
        const scalar gasPr,
        const label dragModelId,
        const scalar dragResidualRe,
        const scalar gravityX,
        const scalar gravityY,
        const scalar gravityZ,
        const label gasFluxScheme,
        const label gasReconstruction,
        const label gasLimiter,
        const label gasTimeIntegrator,
        const bool gasRobustFallback,
        const label gasTurbulenceModel,
        const scalar lesDeltaCoeff,
        const scalar turbulentPrandtl,
        const scalar waleCw,
        const scalar smagorinskyCs,
        const scalar maxDiffusionNumber,
        const scalar rhoMin,
        const scalar TgasMin,
        const GpuSchedulingConfiguration& scheduling
    )
    {
        ensureStorage();
        ensureGasFieldSizes(mesh, rho, rhoU, rhoE, U, p, Tgas);

        const label nCells = mesh.nCells();
        const label nFaces = mesh.nFaces();
        const label nInternalFaces = mesh.nInternalFaces();
        label nCellPlanes = 0;
        forAll(mesh.cells(), cellI)
        {
            nCellPlanes += mesh.cells()[cellI].size();
        }
        hostMirrorCells_ = nCells;
        hostMirrorFaces_ = nFaces;
        residentMesh_ = &mesh;

        if (maxFaceWalkHops <= 0)
        {
            FatalErrorInFunction
                << "gpuResidentMaxFaceWalkHops must be positive. Current value = "
                << maxFaceWalkHops << exit(FatalError);
        }

        void* raw = nullptr;
        const int createRc = ugkwpGpuResidentStrictCreate
        (
            int(nCells),
            int(nFaces),
            int(nInternalFaces),
            int(nCellPlanes),
            0,
            int(maxFaceWalkHops),
            scalar(1),
            rngSeed,
            gammaGas,
            RgasValue,
            scalar(1),
            0,
            gasMu,
            gasPr,
            int(dragModelId),
            0,
            dragResidualRe,
            gravityX,
            gravityY,
            gravityZ,
            scalar(1),
            scalar(1),
            scalar(1),
            scalar(0),
            scalar(0),
            rhoMin,
            TgasMin,
            scalar(0),
            scalar(0),
            scalar(300),
            scalar(300),
            0,
            scalar(0.9),
            scalar(0.25),
            0,
            scalar(0.63),
            20,
            int(gasFluxScheme),
            int(gasReconstruction),
            int(gasLimiter),
            int(gasTimeIntegrator),
            gasRobustFallback ? 1 : 0,
            int(gasTurbulenceModel),
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            maxDiffusionNumber,
            scheduling.csrCellLocalPath ? 1 : 0,
            int(scheduling.heavyReductionMode),
            int(scheduling.heavyReductionAutoInterval),
            int(scheduling.particleBlockThreads),
            int(scheduling.reductionBlockThreads),
            scheduling.warpAggregatedBinning ? 1 : 0,
            &raw
        );

        check(createRc, "create GPU resident pure-gas state");
        handle_ = raw;

        uploadMesh(mesh, nCellPlanes, nullptr);
        uploadGasBoundaryFields
        (
            mesh,
            rho,
            U,
            p,
            Tgas,
            RgasValue,
            strictOpenFoamGasBoundaries
        );
        uploadGasOnlyFields(rho, rhoU, rhoE, U, p, Tgas);
    }

public:
    void configureSst
    (
        const fvMesh& mesh,
        const volScalarField& k,
        const volScalarField& omega,
        const volScalarField& wallDistance,
        const scalar alphaK1,
        const scalar alphaK2,
        const scalar alphaOmega1,
        const scalar alphaOmega2,
        const scalar beta1,
        const scalar beta2,
        const scalar betaStar,
        const scalar gamma1,
        const scalar gamma2,
        const scalar a1,
        const scalar b1,
        const scalar c1,
        const scalar kMin,
        const scalar omegaMin,
        const scalar maxSourceNumber,
        const label wallTreatment,
        const scalar wallKappa,
        const scalar wallE,
        const scalar wallCmu
    )
    {
        requireInitialised("configure SST");
        if
        (
            residentMesh_ != &mesh
         || k.size() != mesh.nCells()
         || omega.size() != mesh.nCells()
         || wallDistance.size() != mesh.nCells()
        )
        {
            FatalErrorInFunction
                << "SST fields and wall distance must use the resident mesh."
                << exit(FatalError);
        }

        labelList boundaryKMode(mesh.nFaces(), 0);
        labelList boundaryOmegaMode(mesh.nFaces(), 0);
        scalarField boundaryK(mesh.nFaces(), kMin);
        scalarField boundaryOmega(mesh.nFaces(), omegaMin);
        forAll(mesh.boundary(), patchI)
        {
            const polyPatch& patch = mesh.boundaryMesh()[patchI];
            const fvPatchScalarField& kPatch = k.boundaryField()[patchI];
            const fvPatchScalarField& omegaPatch =
                omega.boundaryField()[patchI];

            const bool kInletOutlet =
                isA<inletOutletFvPatchScalarField>(kPatch);
            const bool omegaInletOutlet =
                isA<inletOutletFvPatchScalarField>(omegaPatch);
            const inletOutletFvPatchScalarField* kIo = kInletOutlet
              ? &refCast<const inletOutletFvPatchScalarField>(kPatch)
              : nullptr;
            const inletOutletFvPatchScalarField* omegaIo = omegaInletOutlet
              ? &refCast<const inletOutletFvPatchScalarField>(omegaPatch)
              : nullptr;

            forAll(patch, patchFaceI)
            {
                const label faceI = patch.start() + patchFaceI;
                if (isA<wallPolyPatch>(patch))
                {
                    boundaryKMode[faceI] = wallTreatment == 0 ? 1 : 0;
                    boundaryOmegaMode[faceI] = 1;
                    boundaryK[faceI] =
                        wallTreatment == 0 ? scalar(0) : kMin;
                    boundaryOmega[faceI] = wallTreatment == 0
                      ? max(omegaPatch[patchFaceI], omegaMin)
                      : omegaMin;
                    continue;
                }

                if (kInletOutlet)
                {
                    boundaryKMode[faceI] = 2;
                    boundaryK[faceI] = max(kIo->refValue()[patchFaceI], kMin);
                }
                else if (kPatch.fixesValue())
                {
                    boundaryKMode[faceI] = 1;
                    boundaryK[faceI] = max(kPatch[patchFaceI], kMin);
                }

                if (omegaInletOutlet)
                {
                    boundaryOmegaMode[faceI] = 2;
                    boundaryOmega[faceI] = max
                    (
                        omegaIo->refValue()[patchFaceI],
                        omegaMin
                    );
                }
                else if (omegaPatch.fixesValue())
                {
                    boundaryOmegaMode[faceI] = 1;
                    boundaryOmega[faceI] = max
                    (
                        omegaPatch[patchFaceI],
                        omegaMin
                    );
                }
            }
        }

        check
        (
            ugkwpGpuResidentStrictConfigureSst
            (
                handle_,
                alphaK1,
                alphaK2,
                alphaOmega1,
                alphaOmega2,
                beta1,
                beta2,
                betaStar,
                gamma1,
                gamma2,
                a1,
                b1,
                c1,
                kMin,
                omegaMin,
                maxSourceNumber,
                int(wallTreatment),
                wallKappa,
                wallE,
                wallCmu,
                k.primitiveField().begin(),
                omega.primitiveField().begin(),
                wallDistance.primitiveField().begin(),
                boundaryKMode.begin(),
                boundaryOmegaMode.begin(),
                boundaryK.begin(),
                boundaryOmega.begin()
            ),
            "configure GPU resident explicit SST"
        );
    }

    void downloadSstToHostMirror
    (
        const Time& runTime,
        volScalarField& k,
        volScalarField& omega,
        volScalarField& nut
    )
    {
        requireInitialised("download SST fields");
        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "SST fields may only be downloaded at writeTime()."
                << exit(FatalError);
        }
        scalarField kValues(hostMirrorCells_);
        scalarField omegaValues(hostMirrorCells_);
        scalarField nutValues(hostMirrorCells_);
        check
        (
            ugkwpGpuResidentStrictDownloadSst
            (
                handle_,
                kValues.begin(),
                omegaValues.begin(),
                nutValues.begin()
            ),
            "download GPU resident explicit SST"
        );
        validateMirrorScalar("k", kValues);
        validateMirrorScalar("omega", omegaValues);
        validateMirrorScalar("nut", nutValues);
        assignScalar(k, kValues);
        assignScalar(omega, omegaValues);
        assignScalar(nut, nutValues);
    }

    scalar computeGasCourant
    (
        const scalar dt,
        const scalar targetMaxCo,
        const scalar scheduleTime
    ) const
    {
        requireInitialised("computeGasCourant");
        if
        (
            !finiteMirrorScalar(dt)
         || dt <= SMALL
         || !finiteMirrorScalar(targetMaxCo)
         || targetMaxCo <= SMALL
         || !finiteMirrorScalar(scheduleTime)
         || scheduleTime < scalar(0)
        )
        {
            FatalErrorInFunction
                << "computeGasCourant requires positive finite dt and "
                << "targetMaxCo. dt=" << dt
                << ", targetMaxCo=" << targetMaxCo
                << exit(FatalError);
        }
        double maxCo = 0.0;
        check
        (
            ugkwpGpuResidentStrictComputeGasCourant
            (
                handle_,
                dt,
                targetMaxCo,
                scheduleTime,
                &maxCo
            ),
            "compute GPU resident gas Courant number"
        );
        return scalar(maxCo);
    }

    void downloadNutToHostMirror
    (
        const Time& runTime,
        volScalarField& nut
    )
    {
        requireInitialised("downloadNutToHostMirror");
        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "downloadNutToHostMirror is only allowed at writeTime(). "
                << "LES diagnostics remain GPU resident between write times."
                << exit(FatalError);
        }

        scalarField nutF(hostMirrorCells_);
        check
        (
            ugkwpGpuResidentStrictDownloadNut(handle_, nutF.begin()),
            "download GPU resident nut field to host mirror"
        );
        validateMirrorScalar("nut", nutF);
        forAll(nutF, cellI)
        {
            if (nutF[cellI] < scalar(0))
            {
                FatalErrorInFunction
                    << "GPU resident nut is negative at cell " << cellI
                    << ": " << nutF[cellI] << exit(FatalError);
            }
        }
        assignScalar(nut, nutF);
        nut.correctBoundaryConditions();
    }

    void configureScheduledInlet
    (
        const labelList& faceIds,
        const scalar inletTemperature,
        const scalarField& pressureTimes,
        const scalarField& pressureValues,
        const scalarField& volumeFractionTimes,
        const scalarField& volumeFractionValues
    )
    {
        requireInitialised("configureScheduledInlet");
        if
        (
            faceIds.empty()
         || !finiteMirrorScalar(inletTemperature)
         || inletTemperature <= SMALL
         || pressureTimes.empty()
         || pressureTimes.size() != pressureValues.size()
         || volumeFractionTimes.empty()
         || volumeFractionTimes.size() != volumeFractionValues.size()
        )
        {
            FatalErrorInFunction
                << "Scheduled inlet configuration requires non-empty faces, "
                << "positive finite inlet temperature, and non-empty "
                << "time/value tables with matching sizes"
                << exit(FatalError);
        }

        check
        (
            ugkwpGpuResidentStrictConfigureScheduledInlet
            (
                handle_,
                int(faceIds.size()),
                reinterpret_cast<const int*>(faceIds.begin()),
                inletTemperature,
                int(pressureTimes.size()),
                pressureTimes.begin(),
                pressureValues.begin(),
                int(volumeFractionTimes.size()),
                volumeFractionTimes.begin(),
                volumeFractionValues.begin()
            ),
            "configure GPU resident scheduled inlet"
        );
    }

    void advanceOneStep(const scalar dt, const scalar simulationTime)
    {
        requireInitialised("advanceOneStep");
        particleStateCellLocalPacked_ = false;
        check
        (
            ugkwpGpuResidentStrictAdvance(handle_, dt, simulationTime),
            "advance GPU resident strict state"
        );
        particleStateCellLocalPacked_ = true;
    }

    void downloadToHostMirror
    (
        const Time& runTime,
        volScalarField& rho,
        volVectorField& rhoU,
        volScalarField& rhoE,
        volVectorField& U,
        volScalarField& p,
        volScalarField& Tgas,
        volScalarField& epsS,
        volVectorField& rhoUs,
        volScalarField& rhoEs,
        volScalarField& rhoDs,
        volScalarField& rhoHp,
        volVectorField& Us,
        volScalarField& theta,
        volScalarField& Tp,
        volScalarField& dMeanCell,
        const bool allowThermalExchange = false
    )
    {
        requireInitialised("downloadToHostMirror");
        if (!runTime.writeTime() && !allowThermalExchange)
        {
            FatalErrorInFunction
                << "downloadToHostMirror is only allowed at writeTime(). "
                << "Active GPU resident state must not be mirrored to CPU "
                << "between write/restart boundaries."
                << exit(FatalError);
        }

        scalarField rhoF(hostMirrorCells_);
        scalarField rhoUxF(hostMirrorCells_);
        scalarField rhoUyF(hostMirrorCells_);
        scalarField rhoUzF(hostMirrorCells_);
        scalarField rhoEF(hostMirrorCells_);
        scalarField UxF(hostMirrorCells_);
        scalarField UyF(hostMirrorCells_);
        scalarField UzF(hostMirrorCells_);
        scalarField pF(hostMirrorCells_);
        scalarField TgasF(hostMirrorCells_);
        scalarField epsSF(hostMirrorCells_);
        scalarField rhoUsxF(hostMirrorCells_);
        scalarField rhoUsyF(hostMirrorCells_);
        scalarField rhoUszF(hostMirrorCells_);
        scalarField rhoEsF(hostMirrorCells_);
        scalarField rhoDsF(hostMirrorCells_);
        scalarField rhoHpF(hostMirrorCells_);
        scalarField UsxF(hostMirrorCells_);
        scalarField UsyF(hostMirrorCells_);
        scalarField UszF(hostMirrorCells_);
        scalarField thetaF(hostMirrorCells_);
        scalarField TpF(hostMirrorCells_);
        scalarField dMeanF(hostMirrorCells_);

        check
        (
            ugkwpGpuResidentStrictDownloadFields
            (
                handle_,
                rhoF.begin(),
                rhoUxF.begin(),
                rhoUyF.begin(),
                rhoUzF.begin(),
                rhoEF.begin(),
                UxF.begin(),
                UyF.begin(),
                UzF.begin(),
                pF.begin(),
                TgasF.begin(),
                epsSF.begin(),
                rhoUsxF.begin(),
                rhoUsyF.begin(),
                rhoUszF.begin(),
                rhoEsF.begin(),
                rhoDsF.begin(),
                rhoHpF.begin(),
                UsxF.begin(),
                UsyF.begin(),
                UszF.begin(),
                thetaF.begin(),
                TpF.begin(),
                dMeanF.begin()
            ),
            "download GPU resident fields to host mirror"
        );

        validateMirrorScalar("rho", rhoF);
        validateMirrorVector("rhoU", rhoUxF, rhoUyF, rhoUzF);
        validateMirrorScalar("rhoE", rhoEF);
        validateMirrorVector("U", UxF, UyF, UzF);
        validateMirrorScalar("p", pF);
        validateMirrorScalar("T", TgasF);
        validateMirrorScalar("epsilonS", epsSF);
        validateMirrorVector("rhoUs", rhoUsxF, rhoUsyF, rhoUszF);
        validateMirrorScalar("rhoEs", rhoEsF);
        validateMirrorScalar("rhoDs", rhoDsF);
        validateMirrorScalar("rhoHp", rhoHpF);
        validateMirrorVector("Us", UsxF, UsyF, UszF);
        validateMirrorScalar("theta", thetaF);
        validateMirrorScalar("Tp", TpF);
        validateMirrorScalar("dMeanCell", dMeanF);

        assignScalar(rho, rhoF);
        assignVector(rhoU, rhoUxF, rhoUyF, rhoUzF);
        assignScalar(rhoE, rhoEF);
        assignVector(U, UxF, UyF, UzF);
        assignScalar(p, pF);
        assignScalar(Tgas, TgasF);
        assignScalar(epsS, epsSF);
        assignVector(rhoUs, rhoUsxF, rhoUsyF, rhoUszF);
        assignScalar(rhoEs, rhoEsF);
        assignScalar(rhoDs, rhoDsF);
        assignScalar(rhoHp, rhoHpF);
        assignVector(Us, UsxF, UsyF, UszF);
        assignScalar(theta, thetaF);
        assignScalar(Tp, TpF);
        assignScalar(dMeanCell, dMeanF);

        downloadGasBoundaryToHostMirror(rho, U, p, Tgas);
    }

    GpuThermalRestartMirrorMetadata stageThermalRestartMirrors
    (
        const Time& runTime
    ) const
    {
        requireInitialised("stageThermalRestartMirrors");
        const label nParticles = writeParticleRestartMirror(runTime);
        int nSources = 0;
        check
        (
            ugkwpGpuResidentStrictDownloadSourceResidualMass
            (
                handle_, &nSources, 0, nullptr, nullptr
            ),
            "query source residual count for thermal restart manifest"
        );
        const fileName particlePath = particleRestartFile(runTime);
        const fileName epsPath = epsGPrevRestartFile(runTime);
        const fileName sourcePath = sourceResidualRestartFile(runTime);
        GpuThermalRestartMirrorMetadata result;
        result.particleFile = particlePath.name();
        result.epsGPrevFile = epsPath.name();
        result.sourceResidualFile = sourcePath.name();
        result.particleCount = nParticles;
        result.epsGPrevCount = hostMirrorCells_;
        result.sourceResidualCount = nSources;
        result.particleSha1 = sha1File(particlePath);
        result.epsGPrevSha1 = sha1File(epsPath);
        result.sourceResidualSha1 = sha1File(sourcePath);
        return result;
    }

    label writeParticleRestartMirror(const Time& runTime) const
    {
        requireInitialised("writeParticleRestartMirror");
        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "writeParticleRestartMirror is only allowed at writeTime(). "
                << "Particle SoA must remain GPU resident between restart "
                << "boundaries."
                << exit(FatalError);
        }

        mkDir(runTime.timePath());

        int nParticles = 0;
        check
        (
            ugkwpGpuResidentStrictDownloadParticleRestartMirror
            (
                handle_,
                &nParticles,
                0,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            ),
            "query GPU resident particle restart mirror count"
        );

        scalarField px(nParticles);
        scalarField py(nParticles);
        scalarField pz(nParticles);
        scalarField pux(nParticles);
        scalarField puy(nParticles);
        scalarField puz(nParticles);
        scalarField pT(nParticles);
        scalarField pTheta(nParticles);
        scalarField pd(nParticles);
        scalarField pm(nParticles);
        labelList pCellId(nParticles);
        labelList pStatus(nParticles);
        List<unsigned char> pStuck(nParticles);
        pStuck = static_cast<unsigned char>(0);
        labelList pStuckFaceId(nParticles, -1);
        List<float> pDepositionArea(nParticles, 0.0f);
        List<float> pContactDuration(nParticles, 0.0f);
        List<float> pContactMaximumArea(nParticles, 0.0f);
        List<float> pContactPeakFraction(nParticles, 0.0f);
        List<float> pColdNodeSpecificEnthalpy
        (
            nParticles*gpuFshRestart::coldWallAxialNodeCount,
            0.0f
        );
        List<float> pColdRingSolidMass
        (
            nParticles*gpuFshRestart::coldWallRadialRingCount,
            0.0f
        );
        List<float> pColdFrozenArea(nParticles, 0.0f);
        List<float> pColdContactAge(nParticles, 0.0f);
        List<float> pCold2DNodeSpecificEnthalpy
        (
            nParticles*gpuFshRestart::coldWall2DNodeCount,
            0.0f
        );
        List<float> pCold2DRingContactAge
        (
            nParticles*gpuFshRestart::coldWall2DRadialNodeCount,
            0.0f
        );
        List<float> pCold2DFrozenArea(nParticles, 0.0f);
        List<unsigned long long> pRng(nParticles);
        List<unsigned long long> pOrigId(nParticles);

        if (nParticles > 0)
        {
            check
            (
                ugkwpGpuResidentStrictDownloadParticleRestartMirror
                (
                    handle_,
                    &nParticles,
                    nParticles,
                    px.begin(),
                    py.begin(),
                    pz.begin(),
                    pux.begin(),
                    puy.begin(),
                    puz.begin(),
                    pT.begin(),
                    pTheta.begin(),
                    pd.begin(),
                    pm.begin(),
                    reinterpret_cast<int*>(pCellId.begin()),
                    reinterpret_cast<int*>(pStatus.begin()),
                    pStuck.begin(),
                    reinterpret_cast<int*>(pStuckFaceId.begin()),
                    pDepositionArea.begin(),
                    pContactDuration.begin(),
                    pContactMaximumArea.begin(),
                    pContactPeakFraction.begin(),
                    pColdNodeSpecificEnthalpy.begin(),
                    pColdRingSolidMass.begin(),
                    pColdFrozenArea.begin(),
                    pColdContactAge.begin(),
                    pCold2DNodeSpecificEnthalpy.begin(),
                    pCold2DRingContactAge.begin(),
                    pCold2DFrozenArea.begin(),
                    pRng.begin(),
                    pOrigId.begin()
                ),
                "download GPU resident particle restart mirror"
            );
        }

        const fileName restartFile = particleRestartFile(runTime);
        std::ofstream os(restartFile.c_str(), std::ios::binary);
        if (!os.good())
        {
            FatalErrorInFunction
                << "Cannot write GPU resident particle restart mirror: "
                << restartFile << exit(FatalError);
        }

        const gpuFshRestart::ConstView restartView
        {
            static_cast<std::size_t>(nParticles),
            px.cdata(), py.cdata(), pz.cdata(),
            pux.cdata(), puy.cdata(), puz.cdata(),
            pT.cdata(), pTheta.cdata(), pd.cdata(), pm.cdata(),
            reinterpret_cast<const std::int32_t*>(pCellId.cdata()),
            reinterpret_cast<const std::int32_t*>(pStatus.cdata()),
            reinterpret_cast<const std::uint64_t*>(pRng.cdata()),
            reinterpret_cast<const std::uint64_t*>(pOrigId.cdata()),
            reinterpret_cast<const std::uint8_t*>(pStuck.cdata()),
            reinterpret_cast<const std::int32_t*>(pStuckFaceId.cdata()),
            pDepositionArea.cdata()
            , pContactDuration.cdata()
            , pContactMaximumArea.cdata()
            , pContactPeakFraction.cdata()
            , pColdNodeSpecificEnthalpy.cdata()
            , pColdRingSolidMass.cdata()
            , pColdFrozenArea.cdata()
            , pColdContactAge.cdata()
            , pCold2DNodeSpecificEnthalpy.cdata()
            , pCold2DRingContactAge.cdata()
            , pCold2DFrozenArea.cdata()
        };
        try
        {
            gpuFshRestart::writeV5(os, restartView);
        }
        catch (const std::exception& error)
        {
            FatalErrorInFunction
                << "Failed writing UGKP FSH particle restart: "
                << error.what() << exit(FatalError);
        }

        writeEpsGPrevRestartMirror(runTime);
        writeSourceResidualRestartMirror(runTime);
        return label(nParticles);
    }

    void writeEpsGPrevRestartMirror(const Time& runTime) const
    {
        requireInitialised("writeEpsGPrevRestartMirror");

        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "writeEpsGPrevRestartMirror is only allowed at writeTime()."
                << exit(FatalError);
        }

        scalarField epsGPrev(hostMirrorCells_, scalar(1));

        check
        (
            ugkwpGpuResidentStrictDownloadEpsGPrev
            (
                handle_,
                epsGPrev.begin()
            ),
            "download GPU resident epsGPrev restart mirror"
        );

        const fileName restartFile = epsGPrevRestartFile(runTime);
        std::ofstream os(restartFile.c_str());
        if (!os.good())
        {
            FatalErrorInFunction
                << "Cannot write GPU resident epsGPrev restart mirror: "
                << restartFile << exit(FatalError);
        }

        os.setf(std::ios::scientific);
        os.precision(std::numeric_limits<double>::max_digits10);

        os << hostMirrorCells_ << '\n';
        forAll(epsGPrev, cellI)
        {
            os << epsGPrev[cellI] << '\n';
        }
    }

    void writeSourceResidualRestartMirror(const Time& runTime) const
    {
        requireInitialised("writeSourceResidualRestartMirror");
        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "writeSourceResidualRestartMirror is only allowed at writeTime()."
                << exit(FatalError);
        }

        int nSources = 0;
        check
        (
            ugkwpGpuResidentStrictDownloadSourceResidualMass
            (
                handle_, &nSources, 0, nullptr, nullptr
            ),
            "query GPU resident source residual count"
        );

        labelList sourceFace(nSources);
        scalarField residualMass(nSources);
        if (nSources > 0)
        {
            check
            (
                ugkwpGpuResidentStrictDownloadSourceResidualMass
                (
                    handle_,
                    &nSources,
                    nSources,
                    reinterpret_cast<int*>(sourceFace.begin()),
                    residualMass.begin()
                ),
                "download GPU resident source residual mass"
            );
        }

        const fileName restartFile = sourceResidualRestartFile(runTime);
        std::ofstream os(restartFile.c_str());
        if (!os.good())
        {
            FatalErrorInFunction
                << "Cannot write GPU resident source residual mirror: "
                << restartFile << exit(FatalError);
        }
        os.setf(std::ios::scientific);
        os.precision(std::numeric_limits<double>::max_digits10);
        os << "UGKP_SOURCE_RESIDUAL_SCHEMA1 " << nSources << '\n';
        for (label i = 0; i < nSources; ++i)
        {
            os << sourceFace[i] << ' ' << residualMass[i] << '\n';
        }
    }

protected:
    static fileName particleRestartFile(const Time& runTime)
    {
        return runTime.timePath()/"gpuResidentStrictParticles.dat";
    }

    static fileName epsGPrevRestartFile(const Time& runTime)
    {
        return runTime.timePath()/"gpuResidentStrictEpsGPrev.dat";
    }

    static fileName sourceResidualRestartFile(const Time& runTime)
    {
        return runTime.timePath()/"gpuResidentStrictSourceResidual.dat";
    }

    static word sha1File(const fileName& path)
    {
        std::ifstream stream(path.c_str(), std::ios::binary);
        if (!stream)
        {
            FatalErrorInFunction
                << "Cannot open restart mirror for SHA-1: " << path
                << exit(FatalError);
        }
        SHA1 digest;
        char buffer[65536];
        while (stream.good())
        {
            stream.read(buffer, sizeof(buffer));
            const std::streamsize count = stream.gcount();
            if (count > 0)
            {
                digest.append(buffer, static_cast<std::size_t>(count));
            }
        }
        if (!stream.eof())
        {
            FatalErrorInFunction
                << "Failed reading restart mirror for SHA-1: " << path
                << exit(FatalError);
        }
        return word(digest.digest().str());
    }

    static void ensureStorage()
    {
        if (sizeof(scalar) != sizeof(double) || sizeof(label) != sizeof(int))
        {
            FatalErrorInFunction
                << "GPU resident strict mode currently requires DP scalar "
                << "storage and 32-bit labels."
                << exit(FatalError);
        }
    }

    void requireInitialised(const char* action) const
    {
        if (handle_ == nullptr)
        {
            FatalErrorInFunction
                << "Cannot " << action
                << ": GPU resident strict state is not initialised."
                << exit(FatalError);
        }
    }

    static void check(const int rc, const char* action)
    {
        if (rc != 0)
        {
            FatalErrorInFunction
                << "Failed to " << action << ": "
                << ugkwpGpuResidentStrictLastError()
                << exit(FatalError);
        }
    }

    static void ensureInternalSize
    (
        const fvMesh& mesh,
        const GeometricField<scalar, fvPatchField, volMesh>& field,
        const char* name
    )
    {
        if (field.primitiveField().size() != mesh.nCells())
        {
            FatalErrorInFunction
                << "GPU resident strict field size mismatch for " << name
                << ": expected " << mesh.nCells()
                << ", got " << field.primitiveField().size()
                << exit(FatalError);
        }
    }

    static void ensureInternalSize
    (
        const fvMesh& mesh,
        const GeometricField<vector, fvPatchField, volMesh>& field,
        const char* name
    )
    {
        if (field.primitiveField().size() != mesh.nCells())
        {
            FatalErrorInFunction
                << "GPU resident strict field size mismatch for " << name
                << ": expected " << mesh.nCells()
                << ", got " << field.primitiveField().size()
                << exit(FatalError);
        }
    }

    static void ensureFieldSizes
    (
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const volScalarField& epsS,
        const volVectorField& rhoUs,
        const volScalarField& rhoEs,
        const volScalarField& rhoDs,
        const volScalarField& rhoHp,
        const volVectorField& Us,
        const volScalarField& theta,
        const volScalarField& Tp,
        const volScalarField& dMeanCell
    )
    {
        ensureInternalSize(mesh, rho, "rho");
        ensureInternalSize(mesh, rhoU, "rhoU");
        ensureInternalSize(mesh, rhoE, "rhoE");
        ensureInternalSize(mesh, U, "U");
        ensureInternalSize(mesh, p, "p");
        ensureInternalSize(mesh, Tgas, "T");
        ensureInternalSize(mesh, epsS, "epsilonS");
        ensureInternalSize(mesh, rhoUs, "rhoUs");
        ensureInternalSize(mesh, rhoEs, "rhoEs");
        ensureInternalSize(mesh, rhoDs, "rhoDs");
        ensureInternalSize(mesh, rhoHp, "rhoHp");
        ensureInternalSize(mesh, Us, "Us");
        ensureInternalSize(mesh, theta, "theta");
        ensureInternalSize(mesh, Tp, "Tp");
        ensureInternalSize(mesh, dMeanCell, "dMeanCell");
    }

    static void ensureGasFieldSizes
    (
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas
    )
    {
        ensureInternalSize(mesh, rho, "rho");
        ensureInternalSize(mesh, rhoU, "rhoU");
        ensureInternalSize(mesh, rhoE, "rhoE");
        ensureInternalSize(mesh, U, "U");
        ensureInternalSize(mesh, p, "p");
        ensureInternalSize(mesh, Tgas, "T");
    }

    static void packScalar
    (
        scalarField& out,
        const volScalarField& field
    )
    {
        out = field.primitiveField();
    }

    static void packVector
    (
        scalarField& x,
        scalarField& y,
        scalarField& z,
        const volVectorField& field
    )
    {
        const vectorField& values = field.primitiveField();
        x.setSize(values.size());
        y.setSize(values.size());
        z.setSize(values.size());

        forAll(values, i)
        {
            x[i] = values[i].x();
            y[i] = values[i].y();
            z[i] = values[i].z();
        }
    }

    static void assignScalar
    (
        volScalarField& field,
        const scalarField& values
    )
    {
        field.primitiveFieldRef() = values;
    }

    static void assignVector
    (
        volVectorField& field,
        const scalarField& x,
        const scalarField& y,
        const scalarField& z
    )
    {
        vectorField& values = field.primitiveFieldRef();
        forAll(values, i)
        {
            values[i] = vector(x[i], y[i], z[i]);
        }
    }

    static bool finiteMirrorScalar(const scalar value)
    {
        return (value == value) && (mag(value) < GREAT);
    }

    static void validateMirrorScalar
    (
        const word& name,
        const scalarField& values
    )
    {
        forAll(values, i)
        {
            if (!finiteMirrorScalar(values[i]))
            {
                FatalErrorInFunction
                    << "GPU resident strict downloaded a non-finite scalar "
                    << "mirror value before write: field=" << name
                    << " cell=" << i
                    << " value=" << values[i]
                    << exit(FatalError);
            }
        }
    }

    static void validateMirrorVector
    (
        const word& name,
        const scalarField& x,
        const scalarField& y,
        const scalarField& z
    )
    {
        forAll(x, i)
        {
            if
            (
                !finiteMirrorScalar(x[i])
             || !finiteMirrorScalar(y[i])
             || !finiteMirrorScalar(z[i])
            )
            {
                FatalErrorInFunction
                    << "GPU resident strict downloaded a non-finite vector "
                    << "mirror value before write: field=" << name
                    << " cell=" << i
                    << " value=(" << x[i] << ' ' << y[i] << ' ' << z[i] << ')'
                    << exit(FatalError);
            }
        }
    }

    void downloadGasBoundaryToHostMirror
    (
        volScalarField& rho,
        volVectorField& U,
        volScalarField& p,
        volScalarField& Tgas
    ) const
    {
        const fvMesh& mesh = rho.mesh();
        const label nFaces = mesh.nFaces();
        scalarField rhoB(nFaces, scalar(0));
        scalarField UxB(nFaces, scalar(0));
        scalarField UyB(nFaces, scalar(0));
        scalarField UzB(nFaces, scalar(0));
        scalarField pB(nFaces, scalar(0));
        scalarField TB(nFaces, scalar(0));

        check
        (
            ugkwpGpuResidentStrictDownloadGasBoundaryFields
            (
                handle_,
                rhoB.begin(),
                UxB.begin(),
                UyB.begin(),
                UzB.begin(),
                pB.begin(),
                TB.begin()
            ),
            "download GPU resident gas boundary fields to host mirror"
        );

        forAll(mesh.boundaryMesh(), patchI)
        {
            const polyPatch& pp = mesh.boundaryMesh()[patchI];
            const label start = pp.start();
            const label size = pp.size();

            fvPatchScalarField& rhoPatch = rho.boundaryFieldRef()[patchI];
            fvPatchVectorField& UPatch = U.boundaryFieldRef()[patchI];
            fvPatchScalarField& pPatch = p.boundaryFieldRef()[patchI];
            fvPatchScalarField& TPatch = Tgas.boundaryFieldRef()[patchI];
            const bool UIsInletOutlet =
                isA<inletOutletFvPatchField<vector>>(UPatch);

            if
            (
                rhoPatch.size() != size
             || UPatch.size() != size
             || pPatch.size() != size
             || TPatch.size() != size
            )
            {
                continue;
            }

            for (label i = 0; i < size; ++i)
            {
                const label faceI = start + i;
                rhoPatch[i] = rhoB[faceI];
                vector boundaryU(UxB[faceI], UyB[faceI], UzB[faceI]);
                if (UIsInletOutlet)
                {
                    const label owner = mesh.faceOwner()[faceI];
                    if ((U[owner] & mesh.faceAreas()[faceI]) >= scalar(0))
                    {
                        boundaryU = U[owner];
                    }
                }
                UPatch[i] = boundaryU;
                pPatch[i] = pB[faceI];
                TPatch[i] = TB[faceI];
            }
        }
    }

    void readParticleRestartMirrorIfPresent(const Time& runTime)
    {
        const fileName restartFile = particleRestartFile(runTime);
        if (!isFile(restartFile))
        {
            return;
        }

        std::ifstream is(restartFile.c_str(), std::ios::binary);
        string restartHeader;
        label nParticles = 0;
        label maximumChunk = 0;
        bool weightedFshV1 = false;
        bool weightedFshV2 = false;
        bool weightedFshV3 = false;
        bool weightedFshV4 = false;
        bool weightedFshV5 = false;
        bool version3 = false;
        bool version2 = false;
        if (!(is >> restartHeader))
        {
            FatalErrorInFunction
                << "Invalid GPU resident particle restart mirror: "
                << restartFile << exit(FatalError);
        }
        if (restartHeader == gpuFshRestart::formatName)
        {
            weightedFshV5 = true;
            if (!(is >> nParticles >> maximumChunk))
            {
                FatalErrorInFunction
                    << "Invalid UGKP FSH weighted restart header: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == gpuFshRestart::schema4FormatName)
        {
            weightedFshV4 = true;
            if (!(is >> nParticles >> maximumChunk))
            {
                FatalErrorInFunction
                    << "Invalid UGKP FSH schema-4 weighted restart header: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == gpuFshRestart::schema3FormatName)
        {
            weightedFshV3 = true;
            if (!(is >> nParticles >> maximumChunk))
            {
                FatalErrorInFunction
                    << "Invalid UGKP FSH schema-3 weighted restart header: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == gpuFshRestart::schema2FormatName)
        {
            weightedFshV2 = true;
            if (!(is >> nParticles >> maximumChunk))
            {
                FatalErrorInFunction
                    << "Invalid UGKP FSH schema-2 weighted restart header: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == gpuFshRestart::legacyFormatName)
        {
            weightedFshV1 = true;
            if (!(is >> nParticles >> maximumChunk))
            {
                FatalErrorInFunction
                    << "Invalid legacy UGKP FSH weighted restart header: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == "UGKP_PARTICLES_SCHEMA3")
        {
            version3 = true;
            if (!(is >> nParticles))
            {
                FatalErrorInFunction
                    << "Invalid V3 GPU resident particle restart count: "
                    << restartFile << exit(FatalError);
            }
        }
        else if (restartHeader == "UGKP_PARTICLES_SCHEMA2")
        {
            version2 = true;
            if (!(is >> nParticles))
            {
                FatalErrorInFunction
                    << "Invalid V2 GPU resident particle restart count: "
                    << restartFile << exit(FatalError);
            }
        }
        else
        {
            std::istringstream countStream(restartHeader);
            if (!(countStream >> nParticles))
            {
                FatalErrorInFunction
                    << "Invalid legacy GPU resident particle restart count: "
                    << restartFile << exit(FatalError);
            }
        }
        if (nParticles < 0)
        {
            FatalErrorInFunction
                << "Negative GPU resident particle restart count: "
                << restartFile << exit(FatalError);
        }

        scalarField px(nParticles);
        scalarField py(nParticles);
        scalarField pz(nParticles);
        scalarField pux(nParticles);
        scalarField puy(nParticles);
        scalarField puz(nParticles);
        scalarField pT(nParticles);
        scalarField pTheta(nParticles, scalar(0));
        scalarField pd(nParticles);
        scalarField pm(nParticles);
        labelList pCellId(nParticles);
        labelList pStatus(nParticles);
        List<unsigned char> pStuck(nParticles);
        pStuck = static_cast<unsigned char>(0);
        labelList pStuckFaceId(nParticles, -1);
        List<float> pDepositionArea(nParticles, 0.0f);
        List<float> pContactDuration(nParticles, 0.0f);
        List<float> pContactMaximumArea(nParticles, 0.0f);
        List<float> pContactPeakFraction(nParticles, 0.0f);
        List<float> pColdNodeSpecificEnthalpy
        (
            nParticles*gpuFshRestart::coldWallAxialNodeCount,
            0.0f
        );
        List<float> pColdRingSolidMass
        (
            nParticles*gpuFshRestart::coldWallRadialRingCount,
            0.0f
        );
        List<float> pColdFrozenArea(nParticles, 0.0f);
        List<float> pColdContactAge(nParticles, 0.0f);
        List<float> pCold2DNodeSpecificEnthalpy
        (
            nParticles*gpuFshRestart::coldWall2DNodeCount,
            0.0f
        );
        List<float> pCold2DRingContactAge
        (
            nParticles*gpuFshRestart::coldWall2DRadialNodeCount,
            0.0f
        );
        List<float> pCold2DFrozenArea(nParticles, 0.0f);
        List<unsigned long long> pRng(nParticles);
        List<unsigned long long> pOrigId(nParticles);

        if
        (
            weightedFshV1 || weightedFshV2 || weightedFshV3
         || weightedFshV4 || weightedFshV5
        )
        {
            const gpuFshRestart::MutableView restartView
            {
                static_cast<std::size_t>(nParticles),
                px.begin(), py.begin(), pz.begin(),
                pux.begin(), puy.begin(), puz.begin(),
                pT.begin(), pTheta.begin(), pd.begin(), pm.begin(),
                reinterpret_cast<std::int32_t*>(pCellId.begin()),
                reinterpret_cast<std::int32_t*>(pStatus.begin()),
                reinterpret_cast<std::uint64_t*>(pRng.begin()),
                reinterpret_cast<std::uint64_t*>(pOrigId.begin()),
                reinterpret_cast<std::uint8_t*>(pStuck.begin()),
                reinterpret_cast<std::int32_t*>(pStuckFaceId.begin()),
                pDepositionArea.begin()
                , pContactDuration.begin()
                , pContactMaximumArea.begin()
                , pContactPeakFraction.begin()
                , pColdNodeSpecificEnthalpy.begin()
                , pColdRingSolidMass.begin()
                , pColdFrozenArea.begin()
                , pColdContactAge.begin()
                , pCold2DNodeSpecificEnthalpy.begin()
                , pCold2DRingContactAge.begin()
                , pCold2DFrozenArea.begin()
            };
            try
            {
                if (weightedFshV5)
                {
                    gpuFshRestart::readV5Payload
                    (
                        is,
                        static_cast<std::uint32_t>(maximumChunk),
                        restartView
                    );
                }
                else if (weightedFshV4)
                {
                    gpuFshRestart::readV4Payload
                    (
                        is,
                        static_cast<std::uint32_t>(maximumChunk),
                        restartView
                    );
                }
                else if (weightedFshV3)
                {
                    gpuFshRestart::readV3Payload
                    (
                        is,
                        static_cast<std::uint32_t>(maximumChunk),
                        restartView
                    );
                }
                else if (weightedFshV2)
                {
                    gpuFshRestart::readV2Payload
                    (
                        is,
                        static_cast<std::uint32_t>(maximumChunk),
                        restartView
                    );
                }
                else
                {
                    gpuFshRestart::readV1Payload
                    (
                        is,
                        static_cast<std::uint32_t>(maximumChunk),
                        restartView
                    );
                }
            }
            catch (const std::exception& error)
            {
                FatalErrorInFunction
                    << "Failed reading UGKP FSH weighted restart: "
                    << error.what() << exit(FatalError);
            }

            for (label i = 0; i < nParticles; ++i)
            {
                if
                (
                    !finiteMirrorScalar(pm[i]) || pm[i] <= scalar(0)
                 || pStuck[i] > 3
                 || pStuckFaceId[i] < -2
                 || pStuckFaceId[i] >= hostMirrorFaces_
                 || !std::isfinite(static_cast<double>(pDepositionArea[i]))
                 || pDepositionArea[i] < 0.0f
                 || !std::isfinite(static_cast<double>(pContactDuration[i]))
                 || !std::isfinite(static_cast<double>(pContactMaximumArea[i]))
                 || !std::isfinite(static_cast<double>(pContactPeakFraction[i]))
                 || pContactDuration[i] < 0.0f
                 || pContactMaximumArea[i] < 0.0f
                 || pContactPeakFraction[i] < 0.0f
                 || (pStuck[i] == 0 && pStuckFaceId[i] != -1)
                 || (pStuck[i] == 0 && pDepositionArea[i] != 0.0f)
                 || (pStuck[i] != 0 && pStuckFaceId[i] == -1)
                 || (pStuck[i] == 1 && pStuckFaceId[i] >= 0
                     && pDepositionArea[i] <= 0.0f)
                 || (pStuck[i] == 1 && pTheta[i] != 0.0)
                 || (pStuck[i] == 1 && !
                    (
                        (
                            pContactDuration[i] == 0.0f
                         && pContactMaximumArea[i] == 0.0f
                         && pContactPeakFraction[i] == 0.0f
                        )
                     || (
                            pContactDuration[i] > 0.0f
                         && pContactMaximumArea[i] > 0.0f
                         && pContactPeakFraction[i] > 0.0f
                         && pContactPeakFraction[i] < 1.0f
                        )
                    ))
                 || ((pStuck[i] == 2 || pStuck[i] == 3)
                     && !(pContactDuration[i] > 0.0f))
                 || ((pStuck[i] == 2 || pStuck[i] == 3)
                     && !(pContactMaximumArea[i] > 0.0f))
                 || ((pStuck[i] == 2 || pStuck[i] == 3)
                     && !(pContactPeakFraction[i] > 0.0f))
                 || ((pStuck[i] == 2 || pStuck[i] == 3)
                     && !(pContactPeakFraction[i] < 1.0f))
                 || ((pStuck[i] == 2 || pStuck[i] == 3)
                     && pTheta[i] > pContactDuration[i])
                 || (pStuckFaceId[i] == -2 && pDepositionArea[i] != 0.0f)
                 || (particleWallHeatTransferEnabled_ && pStuckFaceId[i] == -2)
                )
                {
                    FatalErrorInFunction
                        << "Invalid weighted FSH parcel state at particle "
                        << i << exit(FatalError);
                }
            }
        }

        string line;
        if
        (
            !weightedFshV1 && !weightedFshV2
         && !weightedFshV3 && !weightedFshV4 && !weightedFshV5
        )
        {
          std::getline(is, line);
          for (label i = 0; i < nParticles; ++i)
          {
            std::getline(is, line);
            while (line.empty() && is.good())
            {
                std::getline(is, line);
            }

            std::istringstream ls(line);
            scalar values[10] = {};
            label cellId = -1;
            label status = 0;
            unsigned long long rng = 0;
            unsigned long long origId = 0;
            label stuck = 0;
            label stuckFaceId = -1;
            scalar depositionArea = scalar(0);

            bool ok14 = false;
            if (version3)
            {
                ok14 = static_cast<bool>
                (
                    ls  >> values[0] >> values[1] >> values[2]
                        >> values[3] >> values[4] >> values[5]
                        >> values[6] >> values[7] >> values[8]
                        >> cellId >> status >> rng >> origId >> stuck
                        >> stuckFaceId >> depositionArea
                );
                values[9] = legacyRestartParcelMass_;
                if
                (
                    (stuck != 0 && stuck != 1)
                 || !finiteMirrorScalar(depositionArea)
                 || depositionArea < scalar(0)
                 || stuckFaceId < -2
                 || stuckFaceId >= hostMirrorFaces_
                 || (
                        stuckFaceId >= 0
                     && (
                            particleStuckCandidateFaceMask_.size()
                         != hostMirrorFaces_
                         || particleStuckCandidateFaceMask_[stuckFaceId] == 0
                        )
                    )
                 || (stuck == 0 && stuckFaceId != -1)
                 || (stuck == 0 && depositionArea != scalar(0))
                 || (stuckFaceId < 0 && depositionArea != scalar(0))
                 || (stuck != 0 && stuckFaceId == -1)
                 || (
                        particleWallHeatTransferEnabled_
                     && stuck != 0
                     && (stuckFaceId < 0 || depositionArea <= scalar(0))
                    )
                )
                {
                    ok14 = false;
                }
            }
            else if (version2)
            {
                ok14 = static_cast<bool>
                (
                    ls  >> values[0] >> values[1] >> values[2]
                        >> values[3] >> values[4] >> values[5]
                        >> values[6] >> values[7] >> values[8]
                        >> cellId >> status >> rng >> origId >> stuck
                );
                values[9] = legacyRestartParcelMass_;
                if (stuck != 0 && stuck != 1)
                {
                    ok14 = false;
                }
            }
            else
            {
                ok14 = static_cast<bool>
                (
                    ls  >> values[0] >> values[1] >> values[2]
                        >> values[3] >> values[4] >> values[5]
                        >> values[6] >> values[7] >> values[8]
                        >> values[9] >> cellId >> status >> rng >> origId
                );
            }

            if (!version3 && !version2 && !ok14)
            {
                std::istringstream noMass(line);
                ok14 = static_cast<bool>
                (
                    noMass >> values[0] >> values[1] >> values[2]
                           >> values[3] >> values[4] >> values[5]
                           >> values[6] >> values[7] >> values[8]
                           >> cellId >> status >> rng >> origId
                );
                values[9] = legacyRestartParcelMass_;
            }

            if (!version3 && !version2 && !ok14)
            {
        std::istringstream old(line);
                ok14 = static_cast<bool>
                (
                    old >> values[0] >> values[1] >> values[2]
                        >> values[3] >> values[4] >> values[5]
                        >> values[6] >> values[8] >> values[9]
                        >> cellId >> status >> rng >> origId
                );
                values[7] = scalar(0);
            }

            if (!ok14)
            {
                FatalErrorInFunction
                    << "Failed reading GPU resident particle restart mirror "
                    << restartFile << " at particle " << i
                    << exit(FatalError);
            }

            px[i] = values[0];
            py[i] = values[1];
            pz[i] = values[2];
            pux[i] = values[3];
            puy[i] = values[4];
            puz[i] = values[5];
            pT[i] = values[6];
            pTheta[i] = values[7];
            pd[i] = values[8];
            pm[i] = values[9];
            pCellId[i] = cellId;
            pStatus[i] = status;
            pStuck[i] = static_cast<unsigned char>(stuck);
            if (version3)
            {
                pStuckFaceId[i] = stuckFaceId;
                pDepositionArea[i] = static_cast<float>(depositionArea);
                if
                (
                    !std::isfinite
                    (
                        static_cast<double>(pDepositionArea[i])
                    )
                 || (
                        depositionArea > scalar(0)
                     && pDepositionArea[i] <= 0.0f
                    )
                )
                {
                    FatalErrorInFunction
                        << "V3 deposition area cannot be represented as "
                        << "single precision at particle " << i
                        << exit(FatalError);
                }
            }
            else if (version2 && stuck != 0)
            {
                if (particleWallHeatTransferEnabled_)
                {
                    FatalErrorInFunction
                        << "Heat-enabled restart cannot recover a stuck V2 "
                        << "particle because its wall face and deposition "
                        << "area were not stored." << exit(FatalError);
                }
                pStuckFaceId[i] = -2;
            }
            pRng[i] = rng;
            pOrigId[i] = origId;
          }
        }

        check
        (
            ugkwpGpuResidentStrictUploadParticleRestartMirror
            (
                handle_,
                int(nParticles),
                px.begin(),
                py.begin(),
                pz.begin(),
                pux.begin(),
                puy.begin(),
                puz.begin(),
                pT.begin(),
                pTheta.begin(),
                pd.begin(),
                pm.begin(),
                reinterpret_cast<const int*>(pCellId.begin()),
                reinterpret_cast<const int*>(pStatus.begin()),
                pStuck.begin(),
                reinterpret_cast<const int*>(pStuckFaceId.begin()),
                pDepositionArea.begin(),
                pContactDuration.begin(),
                pContactMaximumArea.begin(),
                pContactPeakFraction.begin(),
                pColdNodeSpecificEnthalpy.begin(),
                pColdRingSolidMass.begin(),
                pColdFrozenArea.begin(),
                pColdContactAge.begin(),
                pCold2DNodeSpecificEnthalpy.begin(),
                pCold2DRingContactAge.begin(),
                pCold2DFrozenArea.begin(),
                pRng.begin(),
                pOrigId.begin()
            ),
            "upload GPU resident particle restart mirror"
        );

    }

    bool readEpsGPrevRestartMirrorIfPresent(const Time& runTime)
    {
        const fileName epsGPrevFile = epsGPrevRestartFile(runTime);
        if (!isFile(epsGPrevFile))
        {
            return false;
        }

        std::ifstream epsGIs(epsGPrevFile.c_str());

        label nCells = 0;
        if (!(epsGIs >> nCells) || nCells < 0)
        {
            FatalErrorInFunction
                << "Invalid GPU resident epsGPrev restart mirror: "
                << epsGPrevFile << exit(FatalError);
        }

        if (nCells != hostMirrorCells_)
        {
            FatalErrorInFunction
                << "GPU resident epsGPrev restart mirror cell count mismatch: "
                << epsGPrevFile << ", expected " << hostMirrorCells_
                << ", got " << nCells << exit(FatalError);
        }

        scalarField epsGPrev(hostMirrorCells_);
        forAll(epsGPrev, cellI)
        {
            scalar value = scalar(1);
            if (!(epsGIs >> value))
            {
                FatalErrorInFunction
                    << "Failed reading GPU resident epsGPrev restart mirror "
                    << epsGPrevFile << " at cell " << cellI
                    << exit(FatalError);
            }

            if (!finiteMirrorScalar(value))
            {
                value = scalar(1);
            }

            epsGPrev[cellI] = min(max(value, scalar(0)), scalar(1));
        }

        check
        (
            ugkwpGpuResidentStrictUploadEpsGPrev
            (
                handle_,
                epsGPrev.begin()
            ),
            "upload GPU resident epsGPrev restart mirror"
        );

        return true;
    }

    bool readSourceResidualRestartMirrorIfPresent(const Time& runTime)
    {
        const fileName restartFile = sourceResidualRestartFile(runTime);
        if (!isFile(restartFile))
        {
            return false;
        }

        std::ifstream is(restartFile.c_str());
        string version;
        label nSources = 0;
        if
        (
            !(is >> version >> nSources)
         || version != "UGKP_SOURCE_RESIDUAL_SCHEMA1"
         || nSources < 0
        )
        {
            FatalErrorInFunction
                << "Invalid GPU resident source residual mirror: "
                << restartFile << exit(FatalError);
        }

        labelList sourceFace(nSources);
        scalarField residualMass(nSources);
        for (label i = 0; i < nSources; ++i)
        {
            if
            (
                !(is >> sourceFace[i] >> residualMass[i])
             || !finiteMirrorScalar(residualMass[i])
             || residualMass[i] < scalar(0)
            )
            {
                FatalErrorInFunction
                    << "Invalid source residual row " << i << " in "
                    << restartFile << exit(FatalError);
            }
        }

        check
        (
            ugkwpGpuResidentStrictUploadSourceResidualMass
            (
                handle_,
                int(nSources),
                reinterpret_cast<const int*>(sourceFace.begin()),
                residualMass.begin()
            ),
            "upload GPU resident source residual mass"
        );
        return true;
    }

    void configureParticleStuckModel
    (
        const fvMesh& mesh,
        const dictionary& ugkwpProps,
        const bool solveParticleTemperature
    )
    {
        const dictionary* model = nullptr;
        if (ugkwpProps.found("gpuResidentStuckModel"))
        {
            model = &ugkwpProps.subDict("gpuResidentStuckModel");
        }

        const scalar sommerfeldThreshold =
            model == nullptr
          ? scalar(20)
          : model->lookupOrDefault<scalar>
            (
                "sommerfeldThreshold", scalar(20)
            );
        const bool heatTransfer =
            model != nullptr
         && model->lookupOrDefault<bool>("heatTransfer", false);
        const scalar maximumCoverage =
            model == nullptr
          ? scalar(0.63)
          : model->lookupOrDefault<scalar>
            (
                "maximumCoverage", scalar(0.63)
            );
        if (model != nullptr && model->found("thermalContactAreaFraction"))
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel thermalContactAreaFraction was "
                << "replaced by depositionHeatTransferEfficiency and "
                << "reflectionHeatTransferEfficiency; configure both "
                << "channels explicitly."
                << exit(FatalError);
        }
        const scalar depositionHeatTransferEfficiency =
            model == nullptr
          ? scalar(1)
          : model->lookupOrDefault<scalar>
            (
                "depositionHeatTransferEfficiency", scalar(1)
            );
        const scalar reflectionHeatTransferEfficiency =
            model == nullptr
          ? scalar(1)
          : model->lookupOrDefault<scalar>
            (
                "reflectionHeatTransferEfficiency", scalar(1)
            );
        const scalar adhesionEnergyScale =
            model == nullptr
          ? scalar(1)
          : model->lookupOrDefault<scalar>
            (
                "adhesionEnergyScale", scalar(1)
            );
        const scalar contactAngleDegree =
            model == nullptr
          ? scalar(145)
          : model->lookupOrDefault<scalar>
            (
                "contactAngleDegree", scalar(145)
            );
        const word defaultWallInteractionModel =
            model == nullptr
          ? word("liquidDeposition")
          : model->lookupOrDefault<word>
            (
                "wallInteractionModel", word("liquidDeposition")
            );
        const scalar meltingTemperatureK =
            model == nullptr
          ? scalar(2327)
          : model->lookupOrDefault<scalar>
            (
                "meltingTemperature", scalar(2327)
            );
        const scalar mushyRangeK = model == nullptr
          ? scalar(20)
          : model->lookupOrDefault<scalar>("mushyRange", scalar(20));
        const scalar latentHeatJkg = model == nullptr
          ? scalar(1.16e6)
          : model->lookupOrDefault<scalar>("latentHeat", scalar(1.16e6));
        const scalar solidDensityKgM3 = model == nullptr
          ? scalar(3990)
          : model->lookupOrDefault<scalar>("solidDensity", scalar(3990));
        const scalar solidSpecificHeatJkgK = model == nullptr
          ? scalar(1273)
          : model->lookupOrDefault<scalar>
            (
                "solidSpecificHeat", scalar(1273)
            );
        const scalar solidThermalConductivityWmK = model == nullptr
          ? scalar(5.9)
          : model->lookupOrDefault<scalar>
            (
                "solidThermalConductivity", scalar(5.9)
            );
        const scalar pinningThicknessFraction = model == nullptr
          ? scalar(0.2)
          : model->lookupOrDefault<scalar>
            (
                "pinningThicknessFraction", scalar(0.2)
            );
        const scalar interfaceResistanceM2KW = model == nullptr
          ? scalar(0)
          : model->lookupOrDefault<scalar>
            (
                "interfaceThermalResistance", scalar(0)
            );
        const bool wallTransientResistance = model == nullptr
          ? false
          : model->lookupOrDefault<bool>("wallTransientResistance", false);
        const label nonlinearIterations = model == nullptr
          ? label(4)
          : model->lookupOrDefault<label>("solidificationIterations", 4);
        if
        (
            model != nullptr
         && (
                model->found("wallDensity")
             || model->found("wallSpecificHeat")
             || model->found("wallThermalConductivity")
            )
        )
        {
            FatalErrorInFunction
                << "CHT does not accept wallDensity, wallSpecificHeat or "
                << "wallThermalConductivity in gpuResidentStuckModel. "
                << "Particle-wall contact properties are evaluated from "
                << "solidRegionProperties rho/Cp/kappa at the local solid "
                << "wall temperature."
                << exit(FatalError);
        }
        const word heatFluxField =
            model == nullptr
          ? word("particleStuckWallHeatFlux")
          : model->lookupOrDefault<word>
            (
                "heatFluxField", word("particleStuckWallHeatFlux")
            );
        const word reflectionHeatFluxField =
            model == nullptr
          ? word("particleReflectedWallHeatFlux")
          : model->lookupOrDefault<word>
            (
                "reflectionHeatFluxField",
                word("particleReflectedWallHeatFlux")
            );

        if
        (
            !finiteMirrorScalar(sommerfeldThreshold)
         || sommerfeldThreshold <= scalar(0)
        )
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel sommerfeldThreshold must be "
                << "finite and positive." << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(maximumCoverage)
         || maximumCoverage <= scalar(0)
         || maximumCoverage > scalar(1)
        )
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel maximumCoverage must be finite and "
                << "in (0, 1]. Current value = " << maximumCoverage
                << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(depositionHeatTransferEfficiency)
         || depositionHeatTransferEfficiency <= scalar(0)
         || depositionHeatTransferEfficiency > scalar(1)
        )
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel depositionHeatTransferEfficiency must be "
                << "finite and in (0, 1]. Current value = "
                << depositionHeatTransferEfficiency << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(reflectionHeatTransferEfficiency)
         || reflectionHeatTransferEfficiency <= scalar(0)
         || reflectionHeatTransferEfficiency > scalar(1)
        )
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel reflectionHeatTransferEfficiency must be "
                << "finite and in (0, 1]. Current value = "
                << reflectionHeatTransferEfficiency << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(adhesionEnergyScale)
         || adhesionEnergyScale <= scalar(0)
        )
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel adhesionEnergyScale must be "
                << "finite and positive."
                << exit(FatalError);
        }
        if
        (
            !finiteMirrorScalar(contactAngleDegree)
         || contactAngleDegree <= scalar(0)
         || contactAngleDegree >= scalar(180)
        )
        {
            FatalIOErrorInFunction(ugkwpProps)
                << "gpuResidentStuckModel contactAngleDegree must be finite "
                << "and in (0, 180) degrees. Current value = "
                << contactAngleDegree << exit(FatalIOError);
        }
        if
        (
            !finiteMirrorScalar(meltingTemperatureK)
         || !finiteMirrorScalar(mushyRangeK)
         || !finiteMirrorScalar(latentHeatJkg)
         || !finiteMirrorScalar(solidDensityKgM3)
         || !finiteMirrorScalar(solidSpecificHeatJkgK)
         || !finiteMirrorScalar(solidThermalConductivityWmK)
         || !finiteMirrorScalar(pinningThicknessFraction)
         || !finiteMirrorScalar(interfaceResistanceM2KW)
         || meltingTemperatureK <= scalar(0)
         || mushyRangeK <= scalar(0)
         || latentHeatJkg <= scalar(0)
         || solidDensityKgM3 <= scalar(0)
         || solidSpecificHeatJkgK <= scalar(0)
         || solidThermalConductivityWmK <= scalar(0)
         || pinningThicknessFraction <= scalar(0)
         || pinningThicknessFraction > scalar(1)
         || interfaceResistanceM2KW < scalar(0)
         || nonlinearIterations < 1
         || nonlinearIterations > 12
        )
        {
            FatalErrorInFunction
                << "Invalid cold-wall solidification properties."
                << exit(FatalError);
        }
        if (heatTransfer && heatFluxField == reflectionHeatFluxField)
        {
            FatalErrorInFunction
                << "gpuResidentStuckModel heatFluxField and "
                << "reflectionHeatFluxField must be different."
                << exit(FatalError);
        }

        const wordList candidatePatches =
            ugkwpProps.lookupOrDefault<wordList>
            (
                "gpuResidentStuckWallPatches",
                wordList()
            );
        List<unsigned char> candidateFaceMask
        (
            mesh.nFaces(), static_cast<unsigned char>(0)
        );
        const auto wallInteractionCode =
        [&](const word& name) -> unsigned char
        {
            if (name == "reboundContact") return 1;
            if (name == "liquidDeposition") return 2;
            if (name == "solidifyingDeposition" || name == "coldWall1D") return 3;
            if (name == "coldWall2D") return 4;
            FatalErrorInFunction
                << "Unknown wall interaction model '" << name << "'. Valid "
                << "models are reboundContact, liquidDeposition, coldWall1D, "
                << "and coldWall2D. solidifyingDeposition is retained as an "
                << "alias of coldWall1D."
                << exit(FatalError);
            return 0;
        };
        const unsigned char defaultInteractionCode =
            wallInteractionCode(defaultWallInteractionModel);
        boolList seen(mesh.boundaryMesh().size(), false);
        bool coldWallInteractionPresent = false;
        forAll(candidatePatches, nameI)
        {
            const word& patchName = candidatePatches[nameI];
            const label patchI = mesh.boundaryMesh().findPatchID(patchName);
            if
            (
                patchI < 0
             || seen[patchI]
             || !isA<wallPolyPatch>(mesh.boundaryMesh()[patchI])
            )
            {
                FatalErrorInFunction
                    << "gpuResidentStuckWallPatches contains a missing, "
                    << "duplicate, or non-wall patch: " << patchName
                    << exit(FatalError);
            }
            seen[patchI] = true;
            const polyPatch& pp = mesh.boundaryMesh()[patchI];
            forAll(pp, localFaceI)
            {
                candidateFaceMask[pp.start() + localFaceI] =
                    defaultInteractionCode;
            }
            coldWallInteractionPresent =
                coldWallInteractionPresent
             || defaultInteractionCode == 3
             || defaultInteractionCode == 4;
        }
        if (model != nullptr && model->found("wallInteractionModels"))
        {
            const dictionary& interactionModels =
                model->subDict("wallInteractionModels");
            forAllConstIter(dictionary, interactionModels, iter)
            {
                const word patchName(iter().keyword());
                const label patchI = mesh.boundaryMesh().findPatchID(patchName);
                if (patchI < 0 || !seen[patchI])
                {
                    FatalErrorInFunction
                        << "wallInteractionModels entry '" << patchName
                        << "' is not listed in gpuResidentStuckWallPatches."
                        << exit(FatalError);
                }
                const unsigned char interactionCode = wallInteractionCode
                (
                    interactionModels.lookup<word>(patchName)
                );
                const polyPatch& pp = mesh.boundaryMesh()[patchI];
                forAll(pp, localFaceI)
                {
                    candidateFaceMask[pp.start() + localFaceI] = interactionCode;
                }
                coldWallInteractionPresent =
                    coldWallInteractionPresent
                 || interactionCode == 3
                 || interactionCode == 4;
            }
        }

        if (heatTransfer)
        {
            if
            (
                candidatePatches.empty()
             || !solveParticleTemperature
            )
            {
                FatalErrorInFunction
                    << "Enabled gpuResidentStuckModel heatTransfer requires "
                    << "candidate wall patches and "
                    << "particleTemperatureTransport true."
                    << exit(FatalError);
            }
        }
        if (coldWallInteractionPresent && !heatTransfer)
        {
            FatalErrorInFunction
                << "coldWall1D/coldWall2D requires heatTransfer true."
                << exit(FatalError);
        }

        check
        (
            ugkwpGpuResidentStrictConfigureParticleStuckModel
            (
                handle_,
                int(mesh.nFaces()),
                candidateFaceMask.begin(),
                sommerfeldThreshold,
                heatTransfer ? 1 : 0,
                maximumCoverage,
                depositionHeatTransferEfficiency,
                reflectionHeatTransferEfficiency,
                adhesionEnergyScale,
                contactAngleDegree,
                wallTransientResistance ? 1 : 0,
                int(nonlinearIterations),
                meltingTemperatureK,
                mushyRangeK,
                latentHeatJkg,
                solidDensityKgM3,
                solidSpecificHeatJkgK,
                solidThermalConductivityWmK,
                pinningThicknessFraction,
                interfaceResistanceM2KW
            ),
            "configure GPU resident Sommerfeld stuck model"
        );
        particleWallHeatTransferEnabled_ = heatTransfer;
        particleStuckWallHeatFluxFieldName_ = heatFluxField;
        particleReflectedWallHeatFluxFieldName_ = reflectionHeatFluxField;
        particleStuckCandidateFaceMask_.transfer(candidateFaceMask);

        Info<< "Particle stuck model: candidatePatches="
            << candidatePatches
            << " SommerfeldThreshold=" << sommerfeldThreshold
            << " heatTransfer=" << heatTransfer
            << " aluminaWallProperties=temperatureDependent"
            << " depositionHeatTransferEfficiency="
            << depositionHeatTransferEfficiency
            << " reflectionHeatTransferEfficiency="
            << reflectionHeatTransferEfficiency
            << " adhesionEnergyScale=" << adhesionEnergyScale
            << " contactAngleDegree=" << contactAngleDegree
            << " defaultWallInteractionModel="
            << defaultWallInteractionModel
            << " coldWall=" << coldWallInteractionPresent
            << " wallProperties=solidRegionProperties(rho,Cp,kappa)"
            << endl;
    }

    void uploadMesh
    (
        const fvMesh& mesh,
        const label nCellPlanes,
        const dictionary* particleWallProps
    )
    {
        const label nFaces = mesh.nFaces();
        const label nInternalFaces = mesh.nInternalFaces();
        const label nCells = mesh.nCells();

        labelList faceOwner(nFaces);
        labelList faceNeighbour(nFaces, -1);
        labelList facePeriodicPair(nFaces, -1);
        scalarField facePeriodicDx(nFaces, scalar(0));
        scalarField facePeriodicDy(nFaces, scalar(0));
        scalarField facePeriodicDz(nFaces, scalar(0));
        scalarField Cx(nCells);
        scalarField Cy(nCells);
        scalarField Cz(nCells);
        scalarField faceCx(nFaces);
        scalarField faceCy(nFaces);
        scalarField faceCz(nFaces);
        scalarField Sfx(nFaces);
        scalarField Sfy(nFaces);
        scalarField Sfz(nFaces);
        scalarField magSf(nFaces);
        scalarField deltaCoeffs(nFaces, scalar(0));
        scalarField faceWeight(nFaces, scalar(0.5));
        scalarField cellLength(nCells);
        labelList cellPlaneStart(nCells);
        labelList cellPlaneCount(nCells);
        labelList cellFaceId(nCellPlanes);
        labelList cellFaceNeighbor(nCellPlanes);
        labelList cellFaceKind(nCellPlanes);
        scalarField cellFaceRestitution(nCellPlanes);
        scalarField cellFaceTangential(nCellPlanes);
        scalarField planeNx(nCellPlanes);
        scalarField planeNy(nCellPlanes);
        scalarField planeNz(nCellPlanes);
        scalarField planeD(nCellPlanes);

        const labelList& owner = mesh.faceOwner();
        const labelList& neighbour = mesh.faceNeighbour();
        const scalarField& volumes = mesh.V();
        const vectorField& cellCentres = mesh.C().primitiveField();
        const vectorField& faceCentres = mesh.faceCentres();
        const vectorField& faceAreas = mesh.faceAreas();
        const scalarField& magFaceAreas = mesh.magFaceAreas();
        const scalarField& internalDeltaCoeffs =
            mesh.deltaCoeffs().primitiveField();
        const scalarField& internalWeights =
            mesh.surfaceInterpolation::weights().primitiveField();

        scalarField particleBoundaryCoeff(mesh.boundaryMesh().size(), scalar(1));
        labelList particleBoundaryStuck(mesh.boundaryMesh().size(), 0);
        if (particleWallProps != nullptr)
        {
            scalar defaultCoeff =
                particleWallProps->lookupOrDefault<scalar>("particleWallDefault", scalar(1));
            if
            (
                !finiteMirrorScalar(defaultCoeff)
             || defaultCoeff < scalar(0)
             || defaultCoeff > scalar(1)
            )
            {
                FatalErrorInFunction
                    << "ugkwpProperties: particleWallDefault must be finite "
                    << "and in [0, 1]. Current value = " << defaultCoeff
                    << exit(FatalError);
            }

            if (particleWallProps->found("particleWallCoeffs"))
            {
                const dictionary& coeffDict =
                    particleWallProps->subDict("particleWallCoeffs");
                const scalar dictDefault =
                    coeffDict.lookupOrDefault<scalar>("default", defaultCoeff);
                if
                (
                    !finiteMirrorScalar(dictDefault)
                 || dictDefault < scalar(0)
                 || dictDefault > scalar(1)
                )
                {
                    FatalErrorInFunction
                        << "ugkwpProperties/particleWallCoeffs: default must "
                        << "be finite and in [0, 1]. Current value = "
                        << dictDefault << exit(FatalError);
                }

                forAll(particleBoundaryCoeff, patchI)
                {
                    const word& patchName = mesh.boundaryMesh()[patchI].name();
                    const scalar coeff =
                        coeffDict.lookupOrDefault<scalar>(patchName, dictDefault);
                    if
                    (
                        !finiteMirrorScalar(coeff)
                     || coeff < scalar(0)
                     || coeff > scalar(1)
                    )
                    {
                        FatalErrorInFunction
                            << "ugkwpProperties/particleWallCoeffs: coefficient "
                            << "for patch " << patchName
                            << " must be finite and in [0, 1]. Current value = "
                            << coeff << exit(FatalError);
                    }
                    particleBoundaryCoeff[patchI] = coeff;
                }
            }
            else
            {
                particleBoundaryCoeff = defaultCoeff;
            }

            const wordList stuckWallPatches =
                particleWallProps->lookupOrDefault<wordList>
                (
                    "gpuResidentStuckWallPatches",
                    wordList()
                );
            forAll(stuckWallPatches, nameI)
            {
                const word& patchName = stuckWallPatches[nameI];
                const label patchI = mesh.boundaryMesh().findPatchID(patchName);
                if (patchI < 0)
                {
                    FatalErrorInFunction
                        << "gpuResidentStuckWallPatches cannot find patch "
                        << patchName << exit(FatalError);
                }
                if (!isA<wallPolyPatch>(mesh.boundaryMesh()[patchI]))
                {
                    FatalErrorInFunction
                        << "gpuResidentStuckWallPatches patch " << patchName
                        << " is not a wallPolyPatch" << exit(FatalError);
                }
                if (particleBoundaryStuck[patchI] != 0)
                {
                    FatalErrorInFunction
                        << "gpuResidentStuckWallPatches contains duplicate patch "
                        << patchName << exit(FatalError);
                }
                particleBoundaryStuck[patchI] = 1;
            }
        }

        for (label faceI = 0; faceI < nFaces; ++faceI)
        {
            faceOwner[faceI] = owner[faceI];
            if (faceI < nInternalFaces)
            {
                faceNeighbour[faceI] = neighbour[faceI];
                deltaCoeffs[faceI] = internalDeltaCoeffs[faceI];
                faceWeight[faceI] = internalWeights[faceI];
            }

            const vector& fc = faceCentres[faceI];
            const vector& sf = faceAreas[faceI];
            faceCx[faceI] = fc.x();
            faceCy[faceI] = fc.y();
            faceCz[faceI] = fc.z();
            Sfx[faceI] = sf.x();
            Sfy[faceI] = sf.y();
            Sfz[faceI] = sf.z();
            magSf[faceI] = magFaceAreas[faceI];
        }

        forAll(mesh.boundaryMesh(), patchI)
        {
            const polyPatch& pp = mesh.boundaryMesh()[patchI];
            const fvsPatchScalarField& patchDelta =
                mesh.deltaCoeffs().boundaryField()[patchI];
            const fvsPatchScalarField& patchWeights =
                mesh.surfaceInterpolation::weights().boundaryField()[patchI];
            forAll(patchDelta, i)
            {
                deltaCoeffs[pp.start() + i] = patchDelta[i];
            }

            if (!isA<cyclicPolyPatch>(pp))
            {
                continue;
            }

            const cyclicPolyPatch& cpp = refCast<const cyclicPolyPatch>(pp);
            if (cpp.transform().transforms())
            {
                FatalErrorInFunction
                    << "FSH serial cyclic support accepts conformal "
                    << "pure translations only. Patch '" << pp.name()
                    << "' applies a rotational or scaled transform."
                    << exit(FatalError);
            }
            if (cpp.size() != cpp.nbrPatch().size())
            {
                FatalErrorInFunction
                    << "FSH serial cyclic support requires one-to-one "
                    << "conformal face pairing. Patch '" << pp.name()
                    << "' has " << cpp.size() << " faces and neighbour patch '"
                    << cpp.nbrPatch().name() << "' has "
                    << cpp.nbrPatch().size() << '.' << exit(FatalError);
            }

            forAll(cpp, patchFaceI)
            {
                const label faceI = cpp.start() + patchFaceI;
                const label pairFaceI = cpp.transformGlobalFace(faceI);
                if
                (
                    pairFaceI < nInternalFaces
                 || pairFaceI >= nFaces
                 || owner[pairFaceI] < 0
                 || owner[pairFaceI] >= nCells
                )
                {
                    FatalErrorInFunction
                        << "Invalid serial cyclic pair for face " << faceI
                        << " on patch '" << pp.name() << "'. Paired face = "
                        << pairFaceI << exit(FatalError);
                }

                const vector mappedPairCentre =
                    cpp.transform().transformPosition(faceCentres[pairFaceI]);
                const scalar geometryTolerance =
                    scalar(1e-8)
                   *max
                    (
                        mag(faceCentres[faceI]) + mag(faceCentres[pairFaceI]),
                        scalar(1)
                    );
                if
                (
                    mag(mappedPairCentre - faceCentres[faceI])
                        > geometryTolerance
                 || mag(faceAreas[faceI] + faceAreas[pairFaceI])
                        > scalar(1e-8)*max(magFaceAreas[faceI], scalar(1))
                )
                {
                    FatalErrorInFunction
                        << "FSH cyclic face " << faceI << " on patch '"
                        << pp.name() << "' failed paired centre/area validation."
                        << exit(FatalError);
                }

                const vector neighbourToLocal =
                    faceCentres[faceI] - faceCentres[pairFaceI];
                facePeriodicPair[faceI] = pairFaceI;
                faceNeighbour[faceI] = owner[pairFaceI];
                facePeriodicDx[faceI] = neighbourToLocal.x();
                facePeriodicDy[faceI] = neighbourToLocal.y();
                facePeriodicDz[faceI] = neighbourToLocal.z();
                faceWeight[faceI] = patchWeights[patchFaceI];
            }
        }

        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            const vector& c = cellCentres[cellI];
            Cx[cellI] = c.x();
            Cy[cellI] = c.y();
            Cz[cellI] = c.z();
            cellLength[cellI] = Foam::pow(max(volumes[cellI], SMALL), scalar(1.0/3.0));
        }

        label planeI = 0;
        const cellList& cells = mesh.cells();
        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            const cell& cFaces = cells[cellI];
            cellPlaneStart[cellI] = planeI;
            cellPlaneCount[cellI] = cFaces.size();

            forAll(cFaces, i)
            {
                const label faceI = cFaces[i];
                vector n = faceAreas[faceI]/(magFaceAreas[faceI] + SMALL);
                if (owner[faceI] != cellI)
                {
                    n = -n;
                }

                label neighbor = -1;
                label kind = 0;
                scalar restitution = 1.0;
                scalar tangential = 1.0;

                if (faceI < nInternalFaces)
                {
                    neighbor = (owner[faceI] == cellI)
                      ? neighbour[faceI]
                      : owner[faceI];
                }
                else
                {
                    kind = 2;
                    restitution = 1.0;
                    const label patchI = mesh.boundaryMesh().whichPatch(faceI);
                    if (patchI >= 0)
                    {
                        const polyPatch& pp = mesh.boundaryMesh()[patchI];
                        if (isA<cyclicPolyPatch>(pp))
                        {
                                                                              
                                                                              
                            kind = 6;
                            neighbor = faceNeighbour[faceI];
                        }
                        else if (isA<processorPolyPatch>(pp))
                        {
                            kind = 3;
                        }
                        else if
                        (
                            isA<wedgePolyPatch>(pp)
                         || isA<emptyPolyPatch>(pp)
                        )
                        {
                            kind = 1;
                                                                         
                                                                          
                                                                        
                            restitution = 0.0;
                        }
                        else if
                        (
                            isA<symmetryPlanePolyPatch>(pp)
                         || isA<symmetryPolyPatch>(pp)
                        )
                        {
                            kind = 1;
                            restitution = 1.0;
                        }
                        else if (particleBoundaryStuck[patchI] != 0)
                        {
                            const scalar coeff =
                                particleBoundaryCoeff[patchI];
                            if (coeff <= SMALL)
                            {
                                FatalErrorInFunction
                                    << "Sommerfeld candidate wall "
                                    << pp.name()
                                    << " requires particleWallCoeffs > SMALL "
                                    << "so a sub-threshold impact can reflect. "
                                    << "Current value = " << coeff
                                    << exit(FatalError);
                            }
                            kind = 5;
                            restitution = coeff;
                        }
                        else
                        {
                            const scalar coeff = particleBoundaryCoeff[patchI];
                            if (coeff <= SMALL)
                            {
                                kind = 4;
                                restitution = 0.0;
                            }
                            else
                            {
                                kind = 2;
                                restitution = coeff;
                            }
                        }
                    }
                }

                cellFaceId[planeI] = faceI;
                cellFaceNeighbor[planeI] = neighbor;
                cellFaceKind[planeI] = kind;
                cellFaceRestitution[planeI] = restitution;
                cellFaceTangential[planeI] = tangential;
                planeNx[planeI] = n.x();
                planeNy[planeI] = n.y();
                planeNz[planeI] = n.z();
                planeD[planeI] = n & faceCentres[faceI];
                ++planeI;
            }
        }

        check
        (
            ugkwpGpuResidentStrictUploadMesh
            (
                handle_,
                reinterpret_cast<const int*>(faceOwner.begin()),
                reinterpret_cast<const int*>(faceNeighbour.begin()),
                reinterpret_cast<const int*>(facePeriodicPair.begin()),
                facePeriodicDx.begin(),
                facePeriodicDy.begin(),
                facePeriodicDz.begin(),
                volumes.begin(),
                Cx.begin(),
                Cy.begin(),
                Cz.begin(),
                faceCx.begin(),
                faceCy.begin(),
                faceCz.begin(),
                Sfx.begin(),
                Sfy.begin(),
                Sfz.begin(),
                magSf.begin(),
                deltaCoeffs.begin(),
                faceWeight.begin(),
                cellLength.begin(),
                reinterpret_cast<const int*>(cellPlaneStart.begin()),
                reinterpret_cast<const int*>(cellPlaneCount.begin()),
                reinterpret_cast<const int*>(cellFaceId.begin()),
                reinterpret_cast<const int*>(cellFaceNeighbor.begin()),
                reinterpret_cast<const int*>(cellFaceKind.begin()),
                cellFaceRestitution.begin(),
                cellFaceTangential.begin(),
                planeNx.begin(),
                planeNy.begin(),
                planeNz.begin(),
                planeD.begin()
            ),
            "upload mesh to GPU resident strict state"
        );
    }

    void uploadGasBoundaryFields
    (
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const scalar RgasValue,
        const bool strictOpenFoamGasBoundaries
    )
    {
        const label nFaces = mesh.nFaces();
        labelList kind(nFaces, 0);
        labelList rhoFix(nFaces, 0);
        labelList UFix(nFaces, 0);
        labelList pFix(nFaces, 0);
        labelList TFix(nFaces, 0);
        labelList pWave(nFaces, 0);
        scalarField pWaveGamma(nFaces, scalar(0));
        scalarField pWaveFieldInf(nFaces, scalar(0));
        scalarField pWaveLInf(nFaces, scalar(-GREAT));
        scalarField bRho(nFaces, scalar(0));
        scalarField bUx(nFaces, scalar(0));
        scalarField bUy(nFaces, scalar(0));
        scalarField bUz(nFaces, scalar(0));
        scalarField bP(nFaces, scalar(0));
        scalarField bT(nFaces, scalar(0));

        forAll(mesh.boundaryMesh(), patchI)
        {
            const polyPatch& pp = mesh.boundaryMesh()[patchI];
            const label start = pp.start();
            const label size = pp.size();
            label patchKind = static_cast<label>
            (
                ugkpboundary::PatchGeometry::standard
            );

            if (pp.coupled() && !isA<cyclicPolyPatch>(pp))
            {
                FatalErrorInFunction
                    << "UGKP is a serial resident solver and does not "
                    << "implement coupled boundary transport. Patch '"
                    << pp.name() << "' has type '" << pp.type() << "'."
                    << exit(FatalError);
            }

            if (isA<emptyPolyPatch>(pp))
            {
                patchKind = static_cast<label>
                (
                    ugkpboundary::PatchGeometry::empty
                );
            }
            else if
            (
                isA<wedgePolyPatch>(pp)
             || isA<symmetryPlanePolyPatch>(pp)
             || isA<symmetryPolyPatch>(pp)
            )
            {
                patchKind = static_cast<label>
                (
                    ugkpboundary::PatchGeometry::symmetry
                );
            }
            else if (isA<wallPolyPatch>(pp))
            {
                patchKind = static_cast<label>
                (
                    ugkpboundary::PatchGeometry::wall
                );
            }
            else if (isA<processorPolyPatch>(pp))
            {
                patchKind = static_cast<label>
                (
                    ugkpboundary::PatchGeometry::processor
                );
            }
            else if (isA<cyclicPolyPatch>(pp))
            {
                patchKind = static_cast<label>
                (
                    ugkpboundary::PatchGeometry::cyclic
                );
            }

            if
            (
                patchKind == 4
             || rho.boundaryField()[patchI].size() != size
             || U.boundaryField()[patchI].size() != size
             || p.boundaryField()[patchI].size() != size
             || Tgas.boundaryField()[patchI].size() != size
            )
            {
                for (label i = 0; i < size; ++i)
                {
                    kind[start + i] = 4;
                }
                continue;
            }

            const scalarField& rhoPatch = rho.boundaryField()[patchI];
            const vectorField& UPatch = U.boundaryField()[patchI];
            const scalarField& pPatch = p.boundaryField()[patchI];
            const scalarField& TPatch = Tgas.boundaryField()[patchI];
            const bool pIsWaveTransmissive =
                isA<waveTransmissiveFvPatchField<scalar>>
                (
                    p.boundaryField()[patchI]
                );
            const ugkpboundary::ScalarOperator rhoOperator =
                ugkpboundary::scalarOperator
                (
                    rho.boundaryField()[patchI].type(),
                    rho.boundaryField()[patchI].fixesValue(),
                    false
                );
            const bool UIsInletOutlet =
                isA<inletOutletFvPatchField<vector>>
                (
                    U.boundaryField()[patchI]
                );
            const ugkpboundary::VelocityOperator UOperator =
                ugkpboundary::velocityOperator
                (
                    U.boundaryField()[patchI].type(),
                    U.boundaryField()[patchI].fixesValue(),
                    UIsInletOutlet
                );
            const ugkpboundary::ScalarOperator pOperator =
                ugkpboundary::scalarOperator
                (
                    p.boundaryField()[patchI].type(),
                    p.boundaryField()[patchI].fixesValue(),
                    pIsWaveTransmissive
                );
            const ugkpboundary::ScalarOperator TOperator =
                ugkpboundary::scalarOperator
                (
                    Tgas.boundaryField()[patchI].type(),
                    Tgas.boundaryField()[patchI].fixesValue(),
                    false
                );

            if
            (
                strictOpenFoamGasBoundaries
             && patchKind
                == static_cast<label>
                (
                    ugkpboundary::PatchGeometry::standard
                )
             &&
                (
                    rhoOperator
                        == ugkpboundary::ScalarOperator::unsupported
                 || UOperator
                        == ugkpboundary::VelocityOperator::unsupported
                 || pOperator
                        == ugkpboundary::ScalarOperator::unsupported
                 || TOperator
                        == ugkpboundary::ScalarOperator::unsupported
                )
            )
            {
                FatalErrorInFunction
                    << "Unsupported OpenFOAM gas boundary field on patch '"
                    << pp.name() << "': rho="
                    << rho.boundaryField()[patchI].type()
                    << ", U=" << U.boundaryField()[patchI].type()
                    << ", p=" << p.boundaryField()[patchI].type()
                    << ", T=" << Tgas.boundaryField()[patchI].type()
                    << ". Supported standard-patch operators are "
                    << "fixedValue, zeroGradient, calculated(rho only), "
                    << "inletOutlet(U only), and "
                    << "waveTransmissive(p only)."
                    << exit(FatalError);
            }

            if
            (
                strictOpenFoamGasBoundaries
             && patchKind
                == static_cast<label>
                (
                    ugkpboundary::PatchGeometry::wall
                )
            )
            {
                if
                (
                    UOperator
                 != ugkpboundary::VelocityOperator::fixedValue
                )
                {
                    FatalErrorInFunction
                        << "OpenFOAM wall patch requires noSlip or fixedValue U"
                        << " in UGKP. Patch '" << pp.name()
                        << "' uses U type '"
                        << U.boundaryField()[patchI].type() << "'."
                        << exit(FatalError);
                }
                if
                (
                    pOperator
                        != ugkpboundary::ScalarOperator::zeroGradient
                 || (
                        rhoOperator
                            != ugkpboundary::ScalarOperator::zeroGradient
                     && rhoOperator
                            != ugkpboundary::ScalarOperator::calculated
                    )
                 || (
                        TOperator
                            != ugkpboundary::ScalarOperator::fixedValue
                     && TOperator
                            != ugkpboundary::ScalarOperator::zeroGradient
                    )
                )
                {
                    FatalErrorInFunction
                        << "Unsupported OpenFOAM gas boundary field on wall "
                        << "patch '" << pp.name() << "': rho="
                        << rho.boundaryField()[patchI].type()
                        << ", p=" << p.boundaryField()[patchI].type()
                        << ", T=" << Tgas.boundaryField()[patchI].type()
                        << ". UGKP wall support is p zeroGradient, rho "
                        << "zeroGradient/calculated, and T "
                        << "fixedValue/zeroGradient."
                        << exit(FatalError);
                }
            }

            const label rhoFixes =
                rhoOperator == ugkpboundary::ScalarOperator::fixedValue
              ? 1 : 0;
            const label UFixes =
                UOperator == ugkpboundary::VelocityOperator::inletOutlet
              ? 2
              : (
                    UOperator
                        == ugkpboundary::VelocityOperator::fixedValue
                  ? 1 : 0
                );
            const vectorField* UBoundaryValues = &UPatch;
            if (UIsInletOutlet)
            {
                UBoundaryValues =
                    &refCast<const inletOutletFvPatchField<vector>>
                    (
                        U.boundaryField()[patchI]
                    ).refValue();
            }
            const label pFixes =
                pOperator == ugkpboundary::ScalarOperator::fixedValue
              ? 1 : 0;
            const label TFixes =
                TOperator == ugkpboundary::ScalarOperator::fixedValue
              ? 1 : 0;
            scalar waveGamma = scalar(0);
            scalar waveFieldInf = scalar(0);
            scalar waveLInf = scalar(-GREAT);

            if (pIsWaveTransmissive)
            {
                const waveTransmissiveFvPatchField<scalar>& wavePatch =
                    refCast<const waveTransmissiveFvPatchField<scalar>>
                    (
                        p.boundaryField()[patchI]
                    );
                waveGamma = wavePatch.gamma();
                waveFieldInf = wavePatch.fieldInf();
                waveLInf = wavePatch.lInf();
            }

            if
            (
                patchKind != 2
             && rhoFixes != 0
             && pFixes != 0
             && TFixes != 0
            )
            {
                for (label i = 0; i < size; ++i)
                {
                    const scalar expectedT =
                        pPatch[i]
                       /max(rhoPatch[i]*RgasValue, scalar(VSMALL));
                    const scalar relativeEosError =
                        mag(TPatch[i] - expectedT)
                       /max(max(mag(TPatch[i]), mag(expectedT)), scalar(1));
                    if
                    (
                        !std::isfinite(expectedT)
                     || !std::isfinite(relativeEosError)
                     || relativeEosError > scalar(1e-8)
                    )
                    {
                        FatalErrorInFunction
                            << "Boundary patch '" << pp.name() << "' fixes "
                            << "rho, p and T simultaneously, but face " << i
                            << " violates p=rho*R*T. Uploaded values are rho="
                            << rhoPatch[i] << ", p=" << pPatch[i]
                            << ", T=" << TPatch[i] << ", R=" << RgasValue
                            << ". Fix two independent thermodynamic values "
                            << "or make all three EOS-consistent."
                            << exit(FatalError);
                    }
                }
            }

            for (label i = 0; i < size; ++i)
            {
                const label faceI = start + i;
                kind[faceI] = patchKind;
                rhoFix[faceI] = rhoFixes;
                UFix[faceI] = UFixes;
                pFix[faceI] = pFixes;
                TFix[faceI] = TFixes;
                pWave[faceI] = pIsWaveTransmissive ? 1 : 0;
                pWaveGamma[faceI] = waveGamma;
                pWaveFieldInf[faceI] = waveFieldInf;
                pWaveLInf[faceI] = waveLInf;
                bRho[faceI] = rhoPatch[i];
                bUx[faceI] = (*UBoundaryValues)[i].x();
                bUy[faceI] = (*UBoundaryValues)[i].y();
                bUz[faceI] = (*UBoundaryValues)[i].z();
                bP[faceI] = pPatch[i];
                bT[faceI] = TPatch[i];
            }
        }

        check
        (
            ugkwpGpuResidentStrictUploadGasBoundaryFields
            (
                handle_,
                reinterpret_cast<const int*>(kind.begin()),
                reinterpret_cast<const int*>(rhoFix.begin()),
                reinterpret_cast<const int*>(UFix.begin()),
                reinterpret_cast<const int*>(pFix.begin()),
                reinterpret_cast<const int*>(TFix.begin()),
                reinterpret_cast<const int*>(pWave.begin()),
                pWaveGamma.begin(),
                pWaveFieldInf.begin(),
                pWaveLInf.begin(),
                bRho.begin(),
                bUx.begin(),
                bUy.begin(),
                bUz.begin(),
                bP.begin(),
                bT.begin()
            ),
            "upload GPU resident gas boundary fields"
        );
    }

    void uploadBoundarySources
    (
        const fvMesh& mesh,
        const volScalarField& epsS,
        const volVectorField& U,
        const volScalarField& Tp,
        const volScalarField& theta,
        const volScalarField& dMeanCell,
        const scalar rhoSolid,
        const scalar particleDiameterFallback,
        const scalar thetaMin,
        const scalar injectionTheta,
        const scalar injectionTp,
        const scalar TpMin
    )
    {
        DynamicList<label> sourceCell;
        DynamicList<label> sourceFace;
        DynamicList<scalar> sourcePx;
        DynamicList<scalar> sourcePy;
        DynamicList<scalar> sourcePz;
        DynamicList<scalar> sourceUx;
        DynamicList<scalar> sourceUy;
        DynamicList<scalar> sourceUz;
        DynamicList<scalar> sourceT;
        DynamicList<scalar> sourceTheta;
        DynamicList<scalar> sourceD;
        DynamicList<scalar> sourceMassRate;

        const vectorField& faceCentres = mesh.faceCentres();
        const vectorField& faceAreas = mesh.faceAreas();
        const scalarField& magFaceAreas = mesh.magFaceAreas();
        const scalarField& volumes = mesh.V();
        const vectorField& cellCentres = mesh.C().primitiveField();

        scalar totalMassRate = 0;

        forAll(mesh.boundary(), patchI)
        {
            const fvPatchScalarField& epsPatch = epsS.boundaryField()[patchI];
            if (!epsPatch.fixesValue())
            {
                continue;
            }

            const fvPatch& patch = mesh.boundary()[patchI];
            const labelUList& faceCells = patch.faceCells();
            const label start = mesh.boundaryMesh()[patchI].start();
            const fvPatchVectorField& UPatch = U.boundaryField()[patchI];
            const fvPatchScalarField& TpPatch = Tp.boundaryField()[patchI];
            const fvPatchScalarField& thetaPatch = theta.boundaryField()[patchI];
            const fvPatchScalarField& dPatch = dMeanCell.boundaryField()[patchI];

            forAll(faceCells, i)
            {
                const label cellI = faceCells[i];
                if (cellI < 0 || cellI >= mesh.nCells())
                {
                    continue;
                }

                const scalar epsBc = max(epsPatch[i], scalar(0));
                const label faceI = start + i;
                const vector& Sf = faceAreas[faceI];
                const scalar massFlux = epsBc*rhoSolid*(UPatch[i] & Sf);
                const scalar incomingMassRate = max(-massFlux, scalar(0));

                vector n = Sf/(magFaceAreas[faceI] + SMALL);
                const scalar h =
                    Foam::pow(max(volumes[cellI], SMALL), scalar(1.0/3.0));
                vector x = faceCentres[faceI] - n*(scalar(1.0e-6)*h);

                const vector toCell = cellCentres[cellI] - faceCentres[faceI];
                if (((x - faceCentres[faceI]) & toCell) < scalar(0))
                {
                    x = faceCentres[faceI] + n*(scalar(1.0e-6)*h);
                }

                scalar dValue = dPatch[i];
                if (!finiteMirrorScalar(dValue) || dValue <= SMALL)
                {
                    dValue = dMeanCell.primitiveField()[cellI];
                }
                if (!finiteMirrorScalar(dValue) || dValue <= SMALL)
                {
                    dValue = particleDiameterFallback;
                }
                if (!finiteMirrorScalar(dValue) || dValue <= SMALL)
                {
                    dValue = scalar(1.0e-6);
                }

                scalar Tvalue = Tp.boundaryField()[patchI].fixesValue()
                    && TpPatch.size() == epsPatch.size()
                    ? TpPatch[i]
                    : injectionTp;
                if (!finiteMirrorScalar(Tvalue) || Tvalue <= SMALL)
                {
                    Tvalue = injectionTp;
                }
                if (!finiteMirrorScalar(Tvalue) || Tvalue <= SMALL)
                {
                    Tvalue = TpMin;
                }

                scalar thetaValue = theta.boundaryField()[patchI].fixesValue()
                    && thetaPatch.size() == epsPatch.size()
                    ? thetaPatch[i]
                    : theta.primitiveField()[cellI];
                if (!finiteMirrorScalar(thetaValue) || thetaValue < scalar(0))
                {
                    thetaValue = scalar(0);
                }

                sourceCell.append(cellI);
                sourceFace.append(faceI);
                sourcePx.append(x.x());
                sourcePy.append(x.y());
                sourcePz.append(x.z());
                sourceUx.append(UPatch[i].x());
                sourceUy.append(UPatch[i].y());
                sourceUz.append(UPatch[i].z());
                sourceT.append(max(Tvalue, max(TpMin, SMALL)));
                sourceTheta.append(thetaValue);
                sourceD.append(dValue);
                sourceMassRate.append(incomingMassRate);
                totalMassRate += incomingMassRate;
            }
        }

        labelList sourceCellList(sourceCell.size());
        labelList sourceFaceList(sourceFace.size());
        scalarField sourcePxField(sourcePx.size());
        scalarField sourcePyField(sourcePy.size());
        scalarField sourcePzField(sourcePz.size());
        scalarField sourceUxField(sourceUx.size());
        scalarField sourceUyField(sourceUy.size());
        scalarField sourceUzField(sourceUz.size());
        scalarField sourceTField(sourceT.size());
        scalarField sourceThetaField(sourceTheta.size());
        scalarField sourceDField(sourceD.size());
        scalarField sourceMassRateField(sourceMassRate.size());

        forAll(sourceCellList, i)
        {
            sourceCellList[i] = sourceCell[i];
            sourceFaceList[i] = sourceFace[i];
            sourcePxField[i] = sourcePx[i];
            sourcePyField[i] = sourcePy[i];
            sourcePzField[i] = sourcePz[i];
            sourceUxField[i] = sourceUx[i];
            sourceUyField[i] = sourceUy[i];
            sourceUzField[i] = sourceUz[i];
            sourceTField[i] = sourceT[i];
            sourceThetaField[i] = sourceTheta[i];
            sourceDField[i] = sourceD[i];
            sourceMassRateField[i] = sourceMassRate[i];
        }

        check
        (
            ugkwpGpuResidentStrictUploadBoundarySources
            (
                handle_,
                int(sourceCellList.size()),
                reinterpret_cast<const int*>(sourceCellList.begin()),
                reinterpret_cast<const int*>(sourceFaceList.begin()),
                sourcePxField.begin(),
                sourcePyField.begin(),
                sourcePzField.begin(),
                sourceUxField.begin(),
                sourceUyField.begin(),
                sourceUzField.begin(),
                sourceTField.begin(),
                sourceThetaField.begin(),
                sourceDField.begin(),
                sourceMassRateField.begin()
            ),
            "upload GPU resident boundary particle sources"
        );

    }

    void uploadFields
    (
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas
    )
    {
        scalarField rhoF;
        scalarField rhoUxF;
        scalarField rhoUyF;
        scalarField rhoUzF;
        scalarField rhoEF;
        scalarField UxF;
        scalarField UyF;
        scalarField UzF;
        scalarField pF;
        scalarField TgasF;

        packScalar(rhoF, rho);
        packVector(rhoUxF, rhoUyF, rhoUzF, rhoU);
        packScalar(rhoEF, rhoE);
        packVector(UxF, UyF, UzF, U);
        packScalar(pF, p);
        packScalar(TgasF, Tgas);

        check
        (
            ugkwpGpuResidentStrictUploadFields
            (
                handle_,
                rhoF.begin(),
                rhoUxF.begin(),
                rhoUyF.begin(),
                rhoUzF.begin(),
                rhoEF.begin(),
                UxF.begin(),
                UyF.begin(),
                UzF.begin(),
                pF.begin(),
                TgasF.begin()
            ),
            "upload fields to GPU resident strict state"
        );
    }

    void uploadGasOnlyFields
    (
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas
    )
    {
        scalarField rhoF;
        scalarField rhoUxF;
        scalarField rhoUyF;
        scalarField rhoUzF;
        scalarField rhoEF;
        scalarField UxF;
        scalarField UyF;
        scalarField UzF;
        scalarField pF;
        scalarField TgasF;

        packScalar(rhoF, rho);
        packVector(rhoUxF, rhoUyF, rhoUzF, rhoU);
        packScalar(rhoEF, rhoE);
        packVector(UxF, UyF, UzF, U);
        packScalar(pF, p);
        packScalar(TgasF, Tgas);

        check
        (
            ugkwpGpuResidentStrictUploadFields
            (
                handle_,
                rhoF.begin(),
                rhoUxF.begin(),
                rhoUyF.begin(),
                rhoUzF.begin(),
                rhoEF.begin(),
                UxF.begin(),
                UyF.begin(),
                UzF.begin(),
                pF.begin(),
                TgasF.begin()
            ),
            "upload gas-only fields to GPU resident strict state"
        );
    }
};

class GpuGasResidentSolver
:
    public GpuResidentStrictBase
{
public:
    void advanceOneStep
    (
        const scalar dt,
        const scalar simulationTime
    )
    {
        if (handle_ == nullptr)
        {
            FatalErrorInFunction
                << "Cannot advanceOneStep: GPU resident pure-gas state is "
                << "not initialised."
                << exit(FatalError);
        }

        if
        (
            ugkwpGpuResidentStrictAdvanceGasOnly
            (
                handle_,
                dt,
                simulationTime
            ) != 0
        )
        {
            FatalErrorInFunction
                << "Failed to advance GPU resident pure-gas state: "
                << ugkwpGpuResidentStrictLastError()
                << exit(FatalError);
        }
    }

    void initialise
    (
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const label maxFaceWalkHops,
        const unsigned long long rngSeed,
        const scalar gammaGas,
        const scalar RgasValue,
        const bool strictOpenFoamGasBoundaries,
        const scalar gasMu,
        const scalar gasPr,
        const label dragModelId,
        const scalar dragResidualRe,
        const scalar gravityX,
        const scalar gravityY,
        const scalar gravityZ,
        const label gasFluxScheme,
        const label gasReconstruction,
        const label gasLimiter,
        const label gasTimeIntegrator,
        const bool gasRobustFallback,
        const label gasTurbulenceModel,
        const scalar lesDeltaCoeff,
        const scalar turbulentPrandtl,
        const scalar waleCw,
        const scalar smagorinskyCs,
        const scalar maxDiffusionNumber,
        const scalar rhoMin,
        const scalar TgasMin,
        const GpuSchedulingConfiguration& scheduling
    )
    {
        initialiseGasBase
        (
            mesh,
            rho,
            rhoU,
            rhoE,
            U,
            p,
            Tgas,
            maxFaceWalkHops,
            rngSeed,
            gammaGas,
            RgasValue,
            strictOpenFoamGasBoundaries,
            gasMu,
            gasPr,
            dragModelId,
            dragResidualRe,
            gravityX,
            gravityY,
            gravityZ,
            gasFluxScheme,
            gasReconstruction,
            gasLimiter,
            gasTimeIntegrator,
            gasRobustFallback,
            gasTurbulenceModel,
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            maxDiffusionNumber,
            rhoMin,
            TgasMin,
            scheduling
        );
    }

    void downloadToHostMirror
    (
        const Time& runTime,
        volScalarField& rho,
        volVectorField& rhoU,
        volScalarField& rhoE,
        volVectorField& U,
        volScalarField& p,
        volScalarField& Tgas
    )
    {
        requireInitialised("downloadToHostMirror");
        if (!runTime.writeTime())
        {
            FatalErrorInFunction
                << "downloadToHostMirror is only allowed at writeTime(). "
                << "Active GPU resident state must not be mirrored to CPU "
                << "between write/restart boundaries."
                << exit(FatalError);
        }

        scalarField rhoF(hostMirrorCells_);
        scalarField rhoUxF(hostMirrorCells_);
        scalarField rhoUyF(hostMirrorCells_);
        scalarField rhoUzF(hostMirrorCells_);
        scalarField rhoEF(hostMirrorCells_);
        scalarField UxF(hostMirrorCells_);
        scalarField UyF(hostMirrorCells_);
        scalarField UzF(hostMirrorCells_);
        scalarField pF(hostMirrorCells_);
        scalarField TgasF(hostMirrorCells_);
        scalarField zero(hostMirrorCells_);

        check
        (
            ugkwpGpuResidentStrictDownloadFields
            (
                handle_,
                rhoF.begin(),
                rhoUxF.begin(),
                rhoUyF.begin(),
                rhoUzF.begin(),
                rhoEF.begin(),
                UxF.begin(),
                UyF.begin(),
                UzF.begin(),
                pF.begin(),
                TgasF.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin(),
                zero.begin()
            ),
            "download GPU resident pure-gas fields to host mirror"
        );

        validateMirrorScalar("rho", rhoF);
        validateMirrorVector("rhoU", rhoUxF, rhoUyF, rhoUzF);
        validateMirrorScalar("rhoE", rhoEF);
        validateMirrorVector("U", UxF, UyF, UzF);
        validateMirrorScalar("p", pF);
        validateMirrorScalar("T", TgasF);

        assignScalar(rho, rhoF);
        assignVector(rhoU, rhoUxF, rhoUyF, rhoUzF);
        assignScalar(rhoE, rhoEF);
        assignVector(U, UxF, UyF, UzF);
        assignScalar(p, pF);
        assignScalar(Tgas, TgasF);

        downloadGasBoundaryToHostMirror(rho, U, p, Tgas);
    }

private:
    using GpuResidentStrictBase::downloadToHostMirror;
    using GpuResidentStrictBase::writeParticleRestartMirror;
};

class GpuParticleResidentSolver
:
    public GpuResidentStrictBase
{
public:
    void initialise
    (
        const Time& runTime,
        const fvMesh& mesh,
        const volScalarField& rho,
        const volVectorField& rhoU,
        const volScalarField& rhoE,
        const volVectorField& U,
        const volScalarField& p,
        const volScalarField& Tgas,
        const volScalarField& epsS,
        const volVectorField& rhoUs,
        const volScalarField& rhoEs,
        const volScalarField& rhoDs,
        const volScalarField& rhoHp,
        const volVectorField& Us,
        const volScalarField& theta,
        const volScalarField& Tp,
        const volScalarField& dMeanCell,
        const scalar injectionParcelMass,
        const scalar legacyRestartParcelMass,
        const label particleCapacity,
        const label maxFaceWalkHops,
        const unsigned long long rngSeed,
        const scalar gammaGas,
        const scalar RgasValue,
        const bool strictOpenFoamGasBoundaries,
        const scalar rhoSolid,
        const bool solveParticleTemperature,
        const label particleGasHeatTransferModelId,
        const scalar particleThermalRho,
        const scalar particleCp,
        const scalar gasMu,
        const scalar gasPr,
        const label gasFluxScheme,
        const label gasReconstruction,
        const label gasLimiter,
        const label gasTimeIntegrator,
        const bool gasRobustFallback,
        const label gasTurbulenceModel,
        const scalar lesDeltaCoeff,
        const scalar turbulentPrandtl,
        const scalar waleCw,
        const scalar smagorinskyCs,
        const scalar maxDiffusionNumber,
        const scalar particleDiameterFallback,
        const scalar particleDiameterMin,
        const scalar particleDiameterMax,
        const scalar particleDiameterSigma,
        const scalar injectionTheta,
        const scalar injectionTp,
        const scalar rhoMin,
        const scalar TgasMin,
        const scalar epsSMin,
        const scalar thetaMin,
        const scalar TpMin,
        const scalar TpMax,
        const label dragModelId,
        const FixedList<scalar, 4>& dragParameters,
        const vector& gravity,
        const GpuSchedulingConfiguration& scheduling,
        const dictionary& ugkwpProps
    )
    {
        if
        (
            !std::isfinite(particleThermalRho)
         || !std::isfinite(particleCp)
         || particleThermalRho <= scalar(0)
         || particleCp <= scalar(0)
         || !std::isfinite(rhoSolid)
         || rhoSolid <= scalar(0)
        )
        {
            FatalErrorInFunction
                << "CHT particle thermal density, heat capacity and material "
                << "density must be finite and positive."
                << exit(FatalError);
        }
        residentParticleHeatFactorJkgK_ =
            particleThermalRho*particleCp/rhoSolid;
        residentRhoSolid_ = rhoSolid;
        residentTpMin_ = TpMin;
        residentTpMax_ = TpMax;
        residentParticleCapacity_ = particleCapacity;
        initialiseBase
        (
            runTime,
            mesh,
            rho,
            rhoU,
            rhoE,
            U,
            p,
            Tgas,
            epsS,
            rhoUs,
            rhoEs,
            rhoDs,
            rhoHp,
            Us,
            theta,
            Tp,
            dMeanCell,
            injectionParcelMass,
            legacyRestartParcelMass,
            particleCapacity,
            maxFaceWalkHops,
            rngSeed,
            gammaGas,
            RgasValue,
            strictOpenFoamGasBoundaries,
            rhoSolid,
            solveParticleTemperature,
            particleGasHeatTransferModelId,
            gasMu,
            gasPr,
            dragModelId,
            dragParameters[0] > scalar(0)
          ? dragParameters[0]
          : scalar(1.0e-3),
            gravity.x(),
            gravity.y(),
            gravity.z(),
            gasFluxScheme,
            gasReconstruction,
            gasLimiter,
            gasTimeIntegrator,
            gasRobustFallback,
            gasTurbulenceModel,
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            maxDiffusionNumber,
            particleDiameterFallback,
            particleDiameterMin,
            particleDiameterMax,
            particleDiameterSigma,
            injectionTheta,
            injectionTp,
            rhoMin,
            TgasMin,
            epsSMin,
            thetaMin,
            TpMin,
            TpMax,
            true,
            scheduling,
            &ugkwpProps
        );
        particleStateCellLocalPacked_ = true;
    }
};

#endif

}

#endif
