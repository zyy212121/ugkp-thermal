#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "zeroGradientFvPatchFields.H"
#include "wallDist.H"
#include "uniformDimensionedFields.H"

#ifndef UGKWP_USE_CUDA
#error CHT must be built with UGKWP_USE_CUDA; use tools/build_cuda_solver.sh
#endif

#ifdef UGKWP_USE_CUDA
#include "gpu/GpuResidentStrict.H"
#include "gpu/GpuBoundarySchedule.H"
#include "gpu/GpuDragModel.H"
#include "thermal/GpuSolidThermalCoupler.H"
#include "thermal/GpuThermalExchangeState.H"
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
    const scalar configuredWriteInterval =
        runTime.controlDict().lookup<scalar>("writeInterval");
    const bool writeAtThermalCoupling =
        configuredWriteInterval == scalar(-1);
    if (configuredWriteInterval <= scalar(0) && !writeAtThermalCoupling)
    {
        FatalErrorInFunction
            << "CHT writeInterval must be positive or exactly -1 for "
            << "thermal-coupling writes."
            << exit(FatalError);
    }
    Info<< "CHT: weighted-parcel directly linked GPU-Riemann-GKP/CHT solver" << nl;
    IOdictionary particlePropertiesFile
    (
        IOobject
        (
            "particleProperties",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );
    const word fluidRegionName
    (
        particlePropertiesFile.lookupOrDefault<word>
        (
            "gpuResidentFluidRegion",
            fvMesh::defaultRegion
        )
    );
    autoPtr<fvMesh> fluidMeshPtr
    (
        new fvMesh
        (
            IOobject
            (
                fluidRegionName,
                runTime.timeName(),
                runTime,
                IOobject::MUST_READ
            )
        )
    );
    fvMesh& mesh = fluidMeshPtr();
    #include "createFields.H"

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
    const Switch gpuResidentSolidThermalCoupling =
        ugkwpProps.lookupOrDefault<Switch>
        (
            "gpuResidentSolidThermalCoupling",
            Switch(false)
        );
    if (writeAtThermalCoupling && !gpuResidentSolidThermalCoupling)
    {
        FatalErrorInFunction
            << "writeInterval -1 requires gpuResidentSolidThermalCoupling true."
            << exit(FatalError);
    }
    if (writeAtThermalCoupling)
    {
        runTime.setWriteInterval(GREAT);
    }
    if (gpuResidentSolidThermalCoupling)
    {
        rhoHp.writeOpt() = IOobject::AUTO_WRITE;
                                                                           
                                                                            
                                                                             
        Tp.writeOpt() = IOobject::AUTO_WRITE;
    }
    if (gpuResidentStrict)
    {
#ifndef UGKWP_USE_CUDA
        FatalErrorInFunction
            << "CHT requires the CUDA GPU-resident solver build. "
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
                << "useCudaRepresentativeMerge. gasUGKP supports explicit "
                << "restart-persistent parcel weights, but not in-step "
                << "representative merging."
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
                    fluidRegionName == fvMesh::defaultRegion
                  ? runTime.path()/runTime.constant()/gpuResidentPressureTable
                  : runTime.path()/runTime.constant()/fluidRegionName
                   /gpuResidentPressureTable
                )
            );
            volumeFractionSchedule.reset
            (
                new GpuBoundaryScheduleTable
                (
                    fluidRegionName == fvMesh::defaultRegion
                  ? runTime.path()/runTime.constant()
                   /gpuResidentVolumeFractionTable
                  : runTime.path()/runTime.constant()/fluidRegionName
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

        bool adjustTimeStep = false;
        scalar maxCo = 0.5;
        scalar maxDeltaT = GREAT;

        const bool gpuResidentPureGasChtBridge =
            gpuResidentPureGasOnly && gpuResidentSolidThermalCoupling;

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

        if (gpuResidentPureGasChtBridge)
        {
            Info<< "Pure-gas CHT mode: gas/solid coupling is active while "
                << "particle loading and particle physics remain disabled. "
                << "The zero-particle resident bridge is used because the "
                << "thermal coupler consumes the shared gas-wall energy "
                << "ledger." << endl;
        }

        if (gpuResidentPureGasOnly && !gpuResidentSolidThermalCoupling)
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
                0,
                scalar(1.0e-3),
                g.value().x(),
                g.value().y(),
                g.value().z(),
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
        autoPtr<gpuThermal::GpuSolidThermalCoupler> solidThermalCoupler;
        gpuThermal::ThermalExchangeRestartState validatedThermalRestartState;
        dictionary resolvedSolidThermalConfiguration;
        if (gpuResidentSolidThermalCoupling)
        {
            IOdictionary solidRegionProperties
            (
                IOobject
                (
                    "solidRegionProperties",
                    runTime.constant(),
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false
                )
            );
            IOdictionary radiationProperties
            (
                IOobject
                (
                    "radiationProperties",
                    runTime.constant(),
                    runTime,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false
                )
            );
            if
            (
                solidRegionProperties.lookupOrDefault<label>
                (
                    "schemaVersion", -1
                ) != 1
             || radiationProperties.lookupOrDefault<label>
                (
                    "schemaVersion", -1
                ) != 1
            )
            {
                FatalErrorInFunction
                    << "solidRegionProperties and radiationProperties require "
                    << "schemaVersion 1." << exit(FatalError);
            }
            resolvedSolidThermalConfiguration = solidRegionProperties;
            dictionary& resolvedCoupling =
                resolvedSolidThermalConfiguration.subDict
                (
                    "solidThermalCoupling"
                );
            resolvedCoupling.set
            (
                "radiationEnabled",
                radiationProperties.lookupOrDefault<bool>("enabled", false)
            );
            resolvedCoupling.set
            (
                "radiationCouplingFrequency",
                radiationProperties.lookupOrDefault<label>
                (
                    "couplingFrequency", 1
                )
            );
            resolvedCoupling.set
            (
                "radiation",
                radiationProperties.subDict("model")
            );
            dictionary resolvedParticleContract;
            resolvedParticleContract.add("rhoSolid", rhoS.value());
            resolvedParticleContract.add
            (
                "particleHeatFactorJkgK",
                particleThermalRho.value()*particleCp.value()/rhoS.value()
            );
            resolvedParticleContract.add("TpMin", TpMinStrict);
            resolvedParticleContract.add("TpMax", TpMaxStrict);
            resolvedParticleContract.add
            (
                "particleCapacity",
                gpuResidentParticleCapacity
            );
            resolvedSolidThermalConfiguration.add
            (
                "resolvedParticleContract",
                resolvedParticleContract
            );
            validatedThermalRestartState =
                gpuThermal::ThermalRestartPreflight::
                    validateBeforeResidentInitialise
                (
                    runTime,
                    mesh,
                    resolvedSolidThermalConfiguration
                );
        }
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
            particleThermalRho.value(),
            particleCp.value(),
            muG.value(),
            PrG.value(),
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
            gpuDragModel->modelId(),
            gpuDragModel->parameters(),
            g.value(),
            gpuScheduling,
            ugkwpProps
        );
        configureResidentSst(resident);
        if (gpuResidentPureGasChtBridge && resident.particleCount() != 0)
        {
            FatalErrorInFunction
                << "Pure-gas CHT requires an empty particle state at "
                << "initialisation, but the resident bridge restored "
                << resident.particleCount() << " particles. Remove the "
                << "particle restart and keep every inlet particle volume "
                << "fraction equal to zero."
                << exit(FatalError);
        }
        configureScheduledBoundary(resident);
        if (gpuResidentSolidThermalCoupling)
        {
            solidThermalCoupler.reset
            (
                new gpuThermal::GpuSolidThermalCoupler
                (
                    runTime,
                    mesh,
                    resident,
                    resolvedSolidThermalConfiguration,
                    validatedThermalRestartState
                )
            );
            solidThermalCoupler->initialiseWallTemperatureBeforeFirstAdvance
            (
                Tgas
            );
        }

        scalar lastMeasuredCo = scalar(0);
        label stepsSinceCourant = gpuResidentCourantUpdateInterval;

        while (runTime.run())
        {
            if (writeAtThermalCoupling)
            {
                runTime.setWriteInterval(GREAT);
            }
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

            bool thermalCoupledThisStep = false;
            if (solidThermalCoupler.valid())
            {
                const gpuThermal::GpuThermalCouplingResult coupling =
                    solidThermalCoupler->exchangeIfDue
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
                if (coupling.coupled)
                {
                    thermalCoupledThisStep = true;
                    Info<< "runTime = " << runTime.elapsedClockTime()
                        << " simulationTime = " << runTime.timeName()
                        << " particleCount = " << coupling.particleCount
                        << " CoMax = " << lastMeasuredCo
                        << " CHTdeltaT = " << coupling.elapsedCouplingTimeS
                        << " radiationCoupled = "
                        << coupling.radiationCoupled
                        << endl;
                }
            }

            if
            (
                runTime.writeTime()
             || (writeAtThermalCoupling && thermalCoupledThisStep)
            )
            {
                const bool thermalCouplingWrite =
                    writeAtThermalCoupling && thermalCoupledThisStep;
                if (gasTurbulenceModel == 3)
                {
                    resident.downloadSstToHostMirror
                    (
                        runTime,
                        k,
                        omega,
                        nut,
                        thermalCouplingWrite
                    );
                }
                else
                {
                    resident.downloadNutToHostMirror
                    (
                        runTime,
                        nut,
                        thermalCouplingWrite
                    );
                }
                if (solidThermalCoupler.valid())
                {
                    (void)
                        solidThermalCoupler->persistAtWriteTime
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
                            dMeanCell,
                            thermalCouplingWrite
                        );
                }
                else
                {
                    label currentParticleCount = 0;
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

                    runTime.write();
                    currentParticleCount =
                        resident.writeParticleRestartMirror(runTime);
                    Info<< "runTime = " << runTime.elapsedClockTime()
                        << " simulationTime = " << runTime.timeName()
                        << " particleCount = " << currentParticleCount
                        << " CoMax = " << lastMeasuredCo << endl;
                }
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
