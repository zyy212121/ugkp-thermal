#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "zeroGradientFvPatchFields.H"
#include "wallDist.H"

#ifndef UGKWP_USE_CUDA
#error FSH must be built with UGKWP_USE_CUDA; use private_backend/build_private_backend.sh
#endif

#ifdef UGKWP_USE_CUDA
#include "gpu/GpuResidentStrict.H"
#include "gpu/GpuBoundarySchedule.H"
#include "gpu/GpuDragModel.H"
#include "autoPtr.H"
#endif

using namespace Foam;


namespace
{
#ifdef UGKWP_USE_CUDA
inline bool finiteScalar(const scalar x)
{
    return (x == x) && (mag(x) < GREAT);
}
#else
inline bool finiteScalar(const scalar x)
{
    return (x == x) && (mag(x) < GREAT);
}
#endif
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    Info<< "FSH: weighted-parcel separated-backend GPU-Riemann-GKP/FSH solver" << nl;
    #include "createMesh.H"
    #include "createFields.H"

    const GpuDragModelSelection gpuDragModel =
        selectGpuDragModel(ugkwpProps);
    const word particleGasHeatTransferModel
    (
        ugkwpProps.lookupOrDefault<word>
        (
            "particleGasHeatTransferModel",
            "RanzMarshall"
        )
    );
    label particleGasHeatTransferModelId = -1;
    if (particleGasHeatTransferModel == "none")
    {
        particleGasHeatTransferModelId = 0;
    }
    else if (particleGasHeatTransferModel == "RanzMarshall")
    {
        particleGasHeatTransferModelId = 1;
    }
    else
    {
        FatalErrorInFunction
            << "Unknown particleGasHeatTransferModel '"
            << particleGasHeatTransferModel
            << "'. Supported models are none and RanzMarshall."
            << exit(FatalError);
    }
    Info<< "Particle force configuration: dragModel="
        << gpuDragModel.name
        << " particleGasHeatTransferModel="
        << particleGasHeatTransferModel
        << " g=" << gravityValue << nl;

    const volScalarField* sstWallDistancePtr = nullptr;
    if (gasTurbulenceModel == 3)
    {
        sstWallDistancePtr = &wallDist::New(mesh).y();
        Info<< "Gas turbulence: RAS kOmegaSST, explicit resident CUDA; "
            << "wallTreatment=" << sstWallTreatmentName
            << " wallKappa=" << sstWallKappa
            << " wallE=" << sstWallE
            << " wallCmu=" << sstWallCmu
            << "; kMin=" << sstKMin
            << " omegaMin=" << sstOmegaMin
            << " maxSourceNumber=" << sstMaxSourceNumber << nl;
    }

    constexpr bool gpuResidentStrict = true;
    const bool gpuResidentPureGasOnly = gpuScheduling.pureGasOnly;
    const Switch gpuResidentDynamicInlet(gpuScheduling.dynamicInlet);
    if (gpuResidentStrict)
    {
#ifndef UGKWP_USE_CUDA
        FatalErrorInFunction
            << "FSH requires the CUDA GPU-resident solver build. "
            << "CPU/OpenFOAM is only allowed to read the case, "
            << "construct the mesh, read dictionaries/initial fields, write "
            << "host mirrors, restart, and log."
            << exit(FatalError);
#else
        const bool forbiddenPartialOffload =
            ugkwpProps.lookupOrDefault<bool>("useCudaGas", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaGasFused", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaSolid", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleMoments", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleMomentsBinned", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleAdvanceInterior", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaCollisionThermalizer", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaCoupling", false)
         || ugkwpProps.lookupOrDefault<bool>("useGpuRuntimeSkeleton", false)
         || ugkwpProps.lookupOrDefault<bool>("useGpuRuntimePersistent", false)
         || ugkwpProps.lookupOrDefault<bool>("cudaBenchmarkBatchMode", false);

        if (forbiddenPartialOffload)
        {
            FatalErrorInFunction
                << "GPU-resident execution rejects the legacy partial CUDA "
                << "offload switches. The strict path owns all active state "
                << "in one GPU resident driver; do not combine it with "
                << "useCudaGas/useCudaSolid/useCudaParticle*"
                << "/useCudaCoupling/useGpuRuntime*/cudaBenchmarkBatchMode."
                << exit(FatalError);
        }

        if (ugkwpProps.lookupOrDefault<bool>("useCudaRepresentativeMerge", false))
        {
            FatalErrorInFunction
                << "GPU-resident execution currently rejects "
                << "useCudaRepresentativeMerge. The active strict path uses "
                << "stopped-state refinement; in-step representative merge "
                << "remains outside the resident update."
                << exit(FatalError);
        }

        if (ugkwpProps.lookupOrDefault<bool>("cudaCollisionUseCpuRandomBridge", false))
        {
            FatalErrorInFunction
                << "GPU-resident execution rejects cudaCollisionUseCpuRandomBridge. "
                << "Random state must be device resident."
                << exit(FatalError);
        }

        const label gpuResidentParticleCapacity =
            gpuScheduling.particleCapacity;

        const scalar gpuResidentSeedInput =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentRandomSeed",
                12345.0
            );
        const unsigned long long gpuResidentSeed =
            static_cast<unsigned long long>(max(gpuResidentSeedInput, scalar(0)));

        const label gpuResidentMaxFaceWalkHops =
            gpuScheduling.maxFaceWalkHops;
        const label gpuResidentCourantUpdateInterval =
            gpuScheduling.courantUpdateInterval;
        const scalar gpuResidentMaxDeltaTGrowth =
            gpuScheduling.maxDeltaTGrowth;

        const scalar epsSMinStrict =
            ugkwpProps.lookupOrDefault<scalar>("epsSMin", 1.0e-12);
        const scalar thetaMinStrict =
            ugkwpProps.lookupOrDefault<scalar>("thetaMin", SMALL);
        const scalar rhoMinStrict = rhoMinG.value();
        const scalar TgasMinStrict = TgasMinG.value();
        const scalar TpMinStrict = max(TpMin.value(), SMALL);
        const scalar TpMaxStrict = max(TpMax.value(), TpMinStrict + SMALL);
        const scalar dMinStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dMin", dS.value()), SMALL);
        const scalar dMaxStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dMax", dS.value()), dMinStrict);
        const scalar dSigmaStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dSigma", scalar(0)), scalar(0));
        const scalar gpuResidentInjectionTheta =
            max
            (
                ugkwpProps.lookupOrDefault<scalar>
                (
                    "gpuResidentInjectionTheta",
                    scalar(15000)
                ),
                scalar(0)
            );
        const scalar gpuResidentInjectionTp =
            min
            (
                max
                (
                    ugkwpProps.lookupOrDefault<scalar>
                    (
                        "gpuResidentInjectionTp",
                        TpMinStrict
                    ),
                    TpMinStrict
                ),
                TpMaxStrict
            );

        word gpuResidentInletPatch;
        word gpuResidentPressureTable;
        word gpuResidentVolumeFractionTable;
        scalar gpuResidentInletTemperature = scalar(0);
        labelList scheduledInletFaceIds;
        autoPtr<GpuBoundaryScheduleTable> pressureSchedule;
        autoPtr<GpuBoundaryScheduleTable> volumeFractionSchedule;
        scalarField pressureTimes;
        scalarField pressureValues;
        scalarField volumeFractionTimes;
        scalarField volumeFractionValues;

        if (gpuResidentDynamicInlet)
        {
            gpuResidentInletPatch =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentInletPatch",
                    "inlet"
                );
            gpuResidentPressureTable =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentPressureTable",
                    "inletPressure.table"
                );
            gpuResidentVolumeFractionTable =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentVolumeFractionTable",
                    "inletVolumeFraction.table"
                );
            gpuResidentInletTemperature =
                ugkwpProps.lookupOrDefault<scalar>
                (
                    "gpuResidentInletTemperature",
                    scalar(3600)
                );
            if
            (
                !finiteScalar(gpuResidentInletTemperature)
             || gpuResidentInletTemperature <= SMALL
            )
            {
                FatalErrorInFunction
                    << "gpuResidentInletTemperature must be positive and finite"
                    << exit(FatalError);
            }

            const label scheduledInletPatchId =
                mesh.boundaryMesh().findPatchID(gpuResidentInletPatch);
            if (scheduledInletPatchId < 0)
            {
                FatalErrorInFunction
                    << "Cannot find scheduled inlet patch "
                    << gpuResidentInletPatch << exit(FatalError);
            }
            const polyPatch& scheduledInletPatch =
                mesh.boundaryMesh()[scheduledInletPatchId];
            if (scheduledInletPatch.empty())
            {
                FatalErrorInFunction
                    << "Scheduled inlet patch is empty: "
                    << gpuResidentInletPatch << exit(FatalError);
            }
            if
            (
                !p.boundaryField()[scheduledInletPatchId].fixesValue()
             || !Tgas.boundaryField()[scheduledInletPatchId].fixesValue()
             || !epsS.boundaryField()[scheduledInletPatchId].fixesValue()
            )
            {
                FatalErrorInFunction
                    << "Scheduled inlet requires fixed-value p, T, and "
                    << "epsilonS on patch " << gpuResidentInletPatch
                    << exit(FatalError);
            }

            scheduledInletFaceIds.setSize(scheduledInletPatch.size());
            forAll(scheduledInletFaceIds, faceI)
            {
                scheduledInletFaceIds[faceI] =
                    scheduledInletPatch.start() + faceI;
            }

            pressureSchedule.reset
            (
                new GpuBoundaryScheduleTable
                (
                    runTime.path()/runTime.constant()/gpuResidentPressureTable
                )
            );
            volumeFractionSchedule.reset
            (
                new GpuBoundaryScheduleTable
                (
                    runTime.path()
                   /runTime.constant()
                   /gpuResidentVolumeFractionTable
                )
            );
            pressureSchedule->copyColumns(pressureTimes, pressureValues);
            volumeFractionSchedule->copyColumns
            (
                volumeFractionTimes,
                volumeFractionValues
            );
        }

        const auto configureScheduledBoundary =
        [&]
        (
            auto& resident
        )
        {
            if (!gpuResidentDynamicInlet)
            {
                return;
            }
            resident.configureScheduledInlet
            (
                scheduledInletFaceIds,
                gpuResidentInletTemperature,
                pressureTimes,
                pressureValues,
                volumeFractionTimes,
                volumeFractionValues
            );
        };

        const auto configureResidentSst = [&](auto& resident)
        {
            if (gasTurbulenceModel != 3)
            {
                return;
            }
            if (sstWallDistancePtr == nullptr)
            {
                FatalErrorInFunction
                    << "SST wall-distance geometry was not constructed."
                    << exit(FatalError);
            }
            resident.configureSst
            (
                mesh,
                k,
                omega,
                *sstWallDistancePtr,
                sstAlphaK1,
                sstAlphaK2,
                sstAlphaOmega1,
                sstAlphaOmega2,
                sstBeta1,
                sstBeta2,
                sstBetaStar,
                sstGamma1,
                sstGamma2,
                sstA1,
                sstB1,
                sstC1,
                sstKMin,
                sstOmegaMin,
                sstMaxSourceNumber,
                sstWallTreatment,
                sstWallKappa,
                sstWallE,
                sstWallCmu
            );
        };

        bool adjustTimeStep = false;
        scalar maxCo = 0.5;
        scalar maxDeltaT = GREAT;

        if (gpuResidentPureGasOnly)
        {
            GpuGasResidentSolver resident;
            resident.initialise
            (
                mesh,
                rho,
                rhoU,
                rhoE,
                U,
                p,
                Tgas,
                gpuResidentMaxFaceWalkHops,
                gpuResidentSeed,
                gammaG.value(),
                Rgas.value(),
                strictOpenFoamGasBoundaries,
                muG.value(),
                PrG.value(),
                int(gpuDragModel.id),
                gpuDragModel.residualRe,
                gravityValue.x(),
                gravityValue.y(),
                gravityValue.z(),
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
                rhoMinStrict,
                TgasMinStrict,
                gpuScheduling
            );
            configureResidentSst(resident);

            configureScheduledBoundary(resident);

            scalar lastMeasuredCo = scalar(0);
            label stepsSinceCourant = gpuResidentCourantUpdateInterval;

            while (runTime.run())
            {
                #include "readTimeControls.H"

                const bool courantRefreshDue =
                    adjustTimeStep
                 && stepsSinceCourant >= gpuResidentCourantUpdateInterval;
                if (courantRefreshDue)
                {
                    const scalar oldDt = runTime.deltaTValue();
                    lastMeasuredCo = resident.computeGasCourant
                    (
                        oldDt,
                        maxCo,
                        runTime.value() + oldDt
                    );
                    if (lastMeasuredCo > SMALL)
                    {
                        const scalar requestedDt =
                            oldDt*maxCo/max(lastMeasuredCo, SMALL);
                        const scalar growthLimitedDt =
                            min(requestedDt, oldDt*gpuResidentMaxDeltaTGrowth);
                        runTime.setDeltaT(min(growthLimitedDt, maxDeltaT));
                    }
                    stepsSinceCourant = 1;
                }
                else
                {
                    stepsSinceCourant = min
                    (
                        stepsSinceCourant + 1,
                        gpuResidentCourantUpdateInterval
                    );
                }

                runTime++;

                resident.advanceOneStep
                (
                    runTime.deltaTValue(),
                    runTime.value()
                );

                if (runTime.writeTime())
                {
                    resident.downloadToHostMirror
                    (
                        runTime,
                        rho,
                        rhoU,
                        rhoE,
                        U,
                        p,
                        Tgas
                    );
                    if (gasTurbulenceModel == 3)
                    {
                        resident.downloadSstToHostMirror
                        (
                            runTime,
                            k,
                            omega,
                            nut
                        );
                    }
                    else
                    {
                        resident.downloadNutToHostMirror(runTime, nut);
                    }

                    runTime.write();
                    Info<< "runTime = " << runTime.elapsedClockTime()
                        << " simulationTime = " << runTime.timeName()
                        << " particleCount = " << 0
                        << " CoMax = " << lastMeasuredCo << endl;
                }
            }

            return 0;
        }

        GpuParticleResidentSolver resident;
        resident.initialise
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
            injectionParcelMass.value(),
            legacyRestartParcelMass.value(),
            gpuResidentParticleCapacity,
            gpuResidentMaxFaceWalkHops,
            gpuResidentSeed,
            gammaG.value(),
            Rgas.value(),
            strictOpenFoamGasBoundaries,
            rhoS.value(),
            solveParticleTemperature,
            particleGasHeatTransferModelId,
            muG.value(),
            PrG.value(),
            int(gpuDragModel.id),
            gpuDragModel.residualRe,
            gravityValue.x(),
            gravityValue.y(),
            gravityValue.z(),
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
            dS.value(),
            dMinStrict,
            dMaxStrict,
            dSigmaStrict,
            gpuResidentInjectionTheta,
            gpuResidentInjectionTp,
            rhoMinStrict,
            TgasMinStrict,
            epsSMinStrict,
            thetaMinStrict,
            TpMinStrict,
            TpMaxStrict,
            gpuScheduling,
            ugkwpProps
        );
        configureResidentSst(resident);
        configureScheduledBoundary(resident);

        if
        (
            resident.particleWallHeatTransferEnabled()
         != particleStuckWallHeatTransfer
        )
        {
            FatalErrorInFunction
                << "Particle wall heat-transfer configuration differs "
                << "between field creation and the resident solver."
                << exit(FatalError);
        }
        if
        (
            particleStuckWallHeatTransfer
         && resident.particleStuckWallHeatFluxFieldName()
            != particleStuckWallHeatFluxName
        )
        {
            FatalErrorInFunction
                << "Particle wall heat-flux field name differs between "
                << "field creation and the resident solver."
                << exit(FatalError);
        }
        if
        (
            particleStuckWallHeatTransfer
         && resident.particleReflectedWallHeatFluxFieldName()
            != particleReflectedWallHeatFluxName
        )
        {
            FatalErrorInFunction
                << "Reflected-particle wall heat-flux field name differs "
                << "between field creation and the resident solver."
                << exit(FatalError);
        }

        scalar particleWallEnergyAccumulationTime = scalar(0);
        scalar lastMeasuredCo = scalar(0);
        label stepsSinceCourant = gpuResidentCourantUpdateInterval;

        while (runTime.run())
        {
            #include "readTimeControls.H"

            const bool courantRefreshDue =
                adjustTimeStep
             && stepsSinceCourant >= gpuResidentCourantUpdateInterval;
            if (courantRefreshDue)
            {
                const scalar oldDt = runTime.deltaTValue();
                lastMeasuredCo = resident.computeGasCourant
                (
                    oldDt,
                    maxCo,
                    runTime.value() + oldDt
                );
                if (lastMeasuredCo > SMALL)
                {
                    const scalar requestedDt =
                        oldDt*maxCo/max(lastMeasuredCo, SMALL);
                    const scalar growthLimitedDt =
                        min(requestedDt, oldDt*gpuResidentMaxDeltaTGrowth);
                    runTime.setDeltaT(min(growthLimitedDt, maxDeltaT));
                }
                stepsSinceCourant = 1;
            }
            else
            {
                stepsSinceCourant = min
                (
                    stepsSinceCourant + 1,
                    gpuResidentCourantUpdateInterval
                );
            }

            runTime++;

            resident.advanceOneStep
            (
                runTime.deltaTValue(),
                runTime.value()
            );
            if (particleStuckWallHeatTransfer)
            {
                particleWallEnergyAccumulationTime +=
                    runTime.deltaTValue();
            }

            if (runTime.writeTime())
            {
                resident.downloadToHostMirror
                (
                    runTime,
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
                    dMeanCell
                );
                if (gasTurbulenceModel == 3)
                {
                    resident.downloadSstToHostMirror
                    (
                        runTime,
                        k,
                        omega,
                        nut
                    );
                }
                else
                {
                    resident.downloadNutToHostMirror(runTime, nut);
                }

                if (particleStuckWallHeatTransfer)
                {
                    if (particleWallEnergyAccumulationTime <= SMALL)
                    {
                        FatalErrorInFunction
                            << "Cannot construct particle wall heat flux "
                            << "without a positive accumulation interval."
                            << exit(FatalError);
                    }

                    scalarField depositedWallEnergyJ;
                    scalarField reflectedWallEnergyJ;
                    resident.downloadAndResetParticleWallHeatLedgers
                    (
                        depositedWallEnergyJ,
                        reflectedWallEnergyJ
                    );
                    if
                    (
                        depositedWallEnergyJ.size() != mesh.nFaces()
                     || reflectedWallEnergyJ.size() != mesh.nFaces()
                    )
                    {
                        FatalErrorInFunction
                            << "Particle wall heat ledgers have "
                            << depositedWallEnergyJ.size() << " and "
                            << reflectedWallEnergyJ.size()
                            << " faces, expected "
                            << mesh.nFaces() << exit(FatalError);
                    }

                    surfaceScalarField& depositedWallHeatFlux =
                        particleStuckWallHeatFluxPtr();
                    surfaceScalarField& reflectedWallHeatFlux =
                        particleReflectedWallHeatFluxPtr();
                    depositedWallHeatFlux.primitiveFieldRef() = scalar(0);
                    reflectedWallHeatFlux.primitiveFieldRef() = scalar(0);
                    forAll(depositedWallHeatFlux.boundaryField(), patchI)
                    {
                        fvsPatchScalarField& depositedQPatch =
                            depositedWallHeatFlux.boundaryFieldRef()[patchI];
                        fvsPatchScalarField& reflectedQPatch =
                            reflectedWallHeatFlux.boundaryFieldRef()[patchI];
                        depositedQPatch = scalar(0);
                        reflectedQPatch = scalar(0);
                        const polyPatch& pp = mesh.boundaryMesh()[patchI];
                        forAll(depositedQPatch, localFaceI)
                        {
                            const label globalFaceI =
                                pp.start() + localFaceI;
                            const scalar faceIntervalArea =
                                mesh.magFaceAreas()[globalFaceI]
                               *particleWallEnergyAccumulationTime;
                            depositedQPatch[localFaceI] =
                                depositedWallEnergyJ[globalFaceI]
                               /(
                                    faceIntervalArea
                                );
                            reflectedQPatch[localFaceI] =
                                reflectedWallEnergyJ[globalFaceI]
                               /(
                                    faceIntervalArea
                                );
                        }
                    }
                    particleWallEnergyAccumulationTime = scalar(0);
                }

                runTime.write();
                const label currentParticleCount =
                    resident.writeParticleRestartMirror(runTime);
                Info<< "runTime = " << runTime.elapsedClockTime()
                    << " simulationTime = " << runTime.timeName()
                    << " particleCount = " << currentParticleCount
                    << " CoMax = " << lastMeasuredCo << endl;
            }
        }

        return 0;
#endif
    }

#ifdef UGKWP_USE_CUDA
    FatalErrorInFunction
        << "Internal error: the mandatory CUDA GPU-resident path was not entered. "
        << "The CUDA solver build does not allow the legacy CPU active path "
        << "or partial CUDA offload path. CPU/OpenFOAM may only read the case, "
        << "construct the mesh, read dictionaries/initial fields, write host "
        << "mirrors, and restart data."
        << exit(FatalError);
#else
    FatalErrorInFunction
        << "This source tree no longer builds a CPU active UGKWP solver. "
        << "Rebuild with UGKWP_USE_CUDA. "
        << "CPU/OpenFOAM may only read the case, construct the mesh, read "
        << "dictionaries/initial fields, write host mirrors, and restart data."
        << exit(FatalError);
#endif

    return 1;
}
