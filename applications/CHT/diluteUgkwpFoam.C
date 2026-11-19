#include "../../common/FoamPhysicalAdvanceTime.H"
#include "FoamGasNumericsControls.H"
#include "FoamSolverScheme.H"
#include "FoamGasTimeStepControls.H"
#include "FoamMomentumTransportInput.H"
#include "FoamParticleSolverControls.H"
#include "gasNumerics/FoamGasScalarSolverControls.H"
#include <type_traits>
#include "../../common/gasNumerics/GasWindowStep.H"
#include "FoamParticleDiameterDistribution.H"
#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "zeroGradientFvPatchFields.H"
#include "wallDist.H"
#include "uniformDimensionedFields.H"
#include "ProgressLogging.H"
#include "FoamPrecisionIO.H"
#include <limits>
#include <exception>

#ifndef UGKWP_USE_CUDA
#error CHT must be built with UGKWP_USE_CUDA; use tool/CHT/build_cuda_solver.sh from the source root
#endif

#ifdef UGKWP_USE_CUDA
#include "gpu/GpuResidentStrict.H"
#include "gpu/GpuBoundarySchedule.H"
#include "gpu/GpuDragModel.H"
#include "thermal/GpuSolidThermalCoupler.H"
#include "thermal/FrozenFoamFields.H"
#include "thermal/GpuThermalExchangeState.H"
#include "autoPtr.H"
#endif

using namespace Foam;

namespace
{

inline bool finiteScalar(const scalar x)
{
    return (x == x) && (mag(x) < GREAT);
}
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
#ifdef UGKWP_USE_CUDA
    gpuThermal::SnapshotFileOperation::install();
#endif
    Info<< "Create time\n" << endl;
    ugkp::gpu::PhysicalAdvanceTime runTime(Time::controlDictName,args);
    (void)ugkp::solverControls::startupScheme(runTime);

    std::string precisionError;
    if (!ugkp::precision::checkRestartPrecision(runTime.timePath().c_str(), precisionError))
    {
        SeriousError<< precisionError << nl;
        return 2;
    }

    try
    {
    const label progressLogInterval =
        ugkwp::logging::progressLogInterval(runTime.controlDict());

    const scalar configuredWriteInterval =
        runTime.controlDict().lookup<scalar>("writeInterval");
    const bool writeAtThermalCoupling =
        configuredWriteInterval == scalar(-1);

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
            << "; kMin=" << sstCoefficients.minimumK
            << " omegaMin=" << sstCoefficients.minimumOmega << nl;
    }

#ifdef UGKWP_USE_CUDA
    Info<< "GPU precision=" << ugkp::precision::compiledBits
        << " (fixed at startup), time=64 wallEnergyAccumulation="
        << ugkp::precision::compiledBits << nl;
#endif
    constexpr bool gpuResidentStrict = true;
    const bool gpuResidentPureGasOnly = gpuScheduling.pureGasOnly;
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
        const auto gasTimeStepControls=ugkp::solverControls::readGasTimeStepControls(runTime,mesh.name());

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
        const scalar dnStrict =
            readParticleDiameterShape(ugkwpProps);
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




        const bool gpuResidentPureGasChtBridge =
            gpuResidentPureGasOnly && gpuResidentSolidThermalCoupling;

        using GasSolverReal=typename std::conditional<ugkp::precision::compiledBits==32,float,double>::type;
        const auto gasScalarSolvers=ugkp::gasNumerics::readGasScalarSolverControls<GasSolverReal>
            (ugkp::solverControls::startupScheme(runTime),mesh.name());
        const auto configureResidentSst = [&](auto& resident)
        {
            resident.configureGasScalarSolvers(gasScalarSolvers);
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
                sstCoefficients.sigmaK1,
                sstCoefficients.sigmaK2,
                sstCoefficients.sigmaOmega1,
                sstCoefficients.sigmaOmega2,
                sstCoefficients.beta1,
                sstCoefficients.beta2,
                sstCoefficients.betaStar,
                sstCoefficients.gamma1,
                sstCoefficients.gamma2,
                sstCoefficients.a1,
                sstB1,
                sstCoefficients.productionLimit,
                sstCoefficients.minimumK,
                sstCoefficients.minimumOmega,
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
            resident.useSharedGasModel(sharedGasModel);
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
                gasTurbulenceModel,
                lesDeltaCoeff,
                turbulentPrandtl,
                waleCw,
                smagorinskyCs,
                rhoMinStrict,
                TgasMinStrict,
                gpuScheduling
            );
            configureResidentSst(resident);


            scalar lastMeasuredCo = scalar(0);
            resident.configureMolecularTransport(molecularTransport.config());
            resident.configureBoundaryLayer(boundaryLayerInput,runTime,mesh,Tgas);
            const auto advanceControls=runTime.advanceControls(gasTimeStepControls);
            resident.configureAdvance(runTime.initialAdvance(advanceControls),true);
            while(runTime.run()){
                const auto request=runTime.advanceBoundary(advanceControls,progressLogInterval);
                resident.enqueueAdvance(request);
                const auto completed=resident.collectAdvance(&runTime);
                runTime.publishAdvance(completed);lastMeasuredCo=completed.lastCourant;

                if
                (
                    ugkwp::logging::progressLogDue
                    (
                        runTime.timeIndex(),
                        progressLogInterval
                    )
                )
                {
                    ugkwp::logging::writeProgressLog(runTime);
                }

                if (runTime.writeTime())
                {
                    ugkp::precision::ScopedRestartWritePrecision writePrecision;
                    ugkp::precision::beginPrecisionWrite(runTime.timePath().c_str());

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

                    resident.downloadSharedGasSpecies(rho);
                    resident.writeBoundaryLayerDiagnostics(boundaryLayerInput, runTime);
                    if (!runTime.write())
                    {
                        FatalIOErrorInFunction(runTime.controlDict())
                            << "OpenFOAM field write failed; time=" << runTime.timeName()
                            << exit(FatalIOError);
                    }
                    ugkp::precision::writeRestartPrecision(runTime.timePath().c_str());
                }
            }

            return 0;
        }

        GpuParticleResidentSolver resident;
        resident.useSharedGasModel(sharedGasModel);
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
            gasTurbulenceModel,
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            dS.value(),
            dMinStrict,
            dMaxStrict,
            dnStrict,
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
        resident.configureMolecularTransport(molecularTransport.config());
        resident.configureBoundaryLayer(boundaryLayerInput,runTime,mesh,Tgas);

        const label maxThermalEvents=runTime.controlDict().lookupOrDefault<label>("maxThermalCouplingEvents",0);
        if(maxThermalEvents<0||(maxThermalEvents>0&&!solidThermalCoupler.valid()))
        {
            FatalIOErrorInFunction(runTime.controlDict())
                << "maxThermalCouplingEvents requires an active thermal coupler and a nonnegative limit"
                << exit(FatalIOError);
        }
        const auto advanceControls=runTime.advanceControls(gasTimeStepControls);
        resident.configureAdvance(runTime.initialAdvance(advanceControls),false);
        bool deviceThermal=false,requestedDeviceThermal=false;
        label completedThermalEvents=0,completedRadiationEvents=0;
        bool advancePending=false;
        auto enqueueBoundary=[&]{
            auto request=runTime.advanceBoundary(advanceControls,progressLogInterval);
            const bool useDevice=solidThermalCoupler.valid()&&!runTime.observesExternalWrite();
            if(useDevice!=requestedDeviceThermal){
                if(useDevice)deviceThermal=solidThermalCoupler->configureDeviceAdvance(Tgas);
                else if(deviceThermal){resident.bindAdvanceExtension({});deviceThermal=false;}
                requestedDeviceThermal=useDevice;
            }
            if(solidThermalCoupler.valid())request.nextHeat=solidThermalCoupler->nextCouplingTime();
            if(maxThermalEvents)request.thermalOutputEvent=maxThermalEvents-completedThermalEvents;
            if(solidThermalCoupler.valid()&&!deviceThermal){

                if(request.nextHeat>runTime.value())
                    request.boundary=ugkp::gpu::eventDeadline(runTime.value(),request.nextHeat,request.boundary);
                else request.maximumSteps=1;
            }
            if(deviceThermal){
                if(writeAtThermalCoupling||request.progressInterval)request.maximumThermalEvents=1;
                else if(maxThermalEvents)request.maximumThermalEvents=maxThermalEvents-completedThermalEvents;
            }
            resident.enqueueAdvance(request);
        };
        while (runTime.run()&&(!maxThermalEvents||completedThermalEvents<maxThermalEvents))
        {
            if (writeAtThermalCoupling)
            {
                runTime.setWriteInterval(GREAT);
            }

            if(!advancePending)enqueueBoundary();
            advancePending=false;
            const auto completed=resident.collectAdvance(&runTime);
            runTime.publishAdvance(completed);lastMeasuredCo=completed.lastCourant;

            bool thermalCoupledThisStep = false;
            if (solidThermalCoupler.valid())
            {
                const gpuThermal::GpuThermalCouplingResult coupling =
                    deviceThermal?solidThermalCoupler->collectDeviceEvents(completed):
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
                        dMeanCell,
                        maxThermalEvents && completedThermalEvents+1==maxThermalEvents
                );
                if (coupling.coupled)
                {
                    thermalCoupledThisStep = true;
                    completedThermalEvents+=coupling.heatEvents;
                    completedRadiationEvents+=coupling.radiationEvents;
                    Info<< "THERMAL_EVENT accepted=" << completedThermalEvents
                        << " radiation=" << completedRadiationEvents
                        << " time=" << Time::timeName(runTime.timeToUserTime(coupling.completedCouplingTimeS))
                        << " interval=" << coupling.elapsedCouplingTimeS << nl;
                    if
                    (
                        ugkwp::logging::progressLogDue
                        (
                            runTime.timeIndex(),
                            progressLogInterval
                        )
                    )
                    {
                        Info<< "runTime = " << runTime.elapsedClockTime()
                            << " simulationTime = " << runTime.timeName()
                            << " particleCount = " << resident.particleCount()
                            << " CoMax = " << lastMeasuredCo
                            << " CHTdeltaT = " << coupling.elapsedCouplingTimeS
                            << " radiationCoupled = "
                            << coupling.radiationCoupled
                            << endl;
                    }
                }
            }

            if
            (
                ugkwp::logging::progressLogDue
                (
                    runTime.timeIndex(),
                    progressLogInterval
                )
            )
            {
                ugkwp::logging::writeProgressLog(runTime);
            }

            if
            (
                runTime.writeTime()
             || (writeAtThermalCoupling && thermalCoupledThisStep)
             || (maxThermalEvents&&completedThermalEvents==maxThermalEvents)
            )
            {
                ugkp::precision::ScopedRestartWritePrecision writePrecision;
                ugkp::precision::beginPrecisionWrite(runTime.timePath().c_str());
                const bool thermalCouplingWrite =
                    (writeAtThermalCoupling && thermalCoupledThisStep)
                 || (maxThermalEvents&&completedThermalEvents==maxThermalEvents);

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
                resident.writeBoundaryLayerDiagnostics(boundaryLayerInput, runTime);
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
                            thermalCouplingWrite,[&]{
                                if(deviceThermal&&(!maxThermalEvents||completedThermalEvents<maxThermalEvents)&&
                                   ugkp::gpu::foamAdvancePrefetchAllowed(runTime)){
                                    enqueueBoundary();advancePending=true;
                                }
                            }
                        );
                }
                else
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

                    resident.downloadSharedGasSpecies(rho);
                    if (!runTime.write())
                    {
                        FatalIOErrorInFunction(runTime.controlDict())
                            << "OpenFOAM field write failed; time=" << runTime.timeName()
                            << exit(FatalIOError);
                    }
                    (void)resident.writeParticleRestartMirror(runTime);
                }
                if(!solidThermalCoupler.valid())ugkp::precision::writeRestartPrecision(runTime.timePath().c_str());
            }
        }

        if(solidThermalCoupler.valid())solidThermalCoupler->finishWrites();
        Info<< "THERMAL_RUN_COMPLETE heatEvents=" << completedThermalEvents
            << " radiationEvents=" << completedRadiationEvents
            << " time=" << runTime.timeName() << endl;
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
    catch (const std::exception& error)
    {
        FatalErrorInFunction
            << "CHT host operation failed; time=" << runTime.timeName()
            << "; cause=" << error.what() << exit(FatalError);
    }
    return 1;
}
