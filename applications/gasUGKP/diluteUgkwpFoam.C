#include "FoamGasNumericsControls.H"
#include "FoamSolverScheme.H"
#include "FoamGasTimeStepControls.H"
#include "FoamMomentumTransportInput.H"
#include "FoamParticleSolverControls.H"
#include "FoamGasSteadyControls.H"
#include "gasNumerics/FoamGasScalarSolverControls.H"
#include <type_traits>
#include "FoamParticleDiameterDistribution.H"
#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "zeroGradientFvPatchFields.H"
#include "wallDist.H"
#include "uniformDimensionedFields.H"
#include "gasTransport/GasNumericsIO.H"
#include "ProgressLogging.H"
#include "FoamGasOutput.H"
#include "FoamPhysicalAdvanceTime.H"

#ifndef UGKWP_USE_CUDA
#error gasUGKP must be built with UGKWP_USE_CUDA; use private_backend/build_private_backend.sh
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
inline bool finiteScalar(const scalar x)
{
    return std::isfinite(x);
}
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
#ifdef UGKWP_STEADY_APPLICATION
    #include "createTime.H"
#else
    ugkp::gpu::PhysicalAdvanceTime runTime(Time::controlDictName,args);
#endif
    (void)ugkp::solverControls::startupScheme(runTime);

    Info<< "GPU precision=" << ugkp::precision::compiledBits << " (fixed at startup)" << nl;
    std::string precisionError;
    if (!ugkp::precision::checkRestartPrecision(runTime.timePath().c_str(), precisionError))
    {
        SeriousError<< precisionError << nl;
        return 2;
    }

    const label progressLogInterval =
        ugkwp::logging::progressLogInterval(runTime.controlDict());
#ifdef UGKWP_STEADY_APPLICATION
    Info<< "gasUGKPSteady: gas LTS / physical-step particle statistical steady solver" << nl;
#else
    Info<< "gasUGKP: weighted-parcel configurable-block dynamic-heavy resident CUDA solver" << nl;
#endif
    #include "createMesh.H"
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
            << " omegaMin=" << sstCoefficients.minimumOmega
            << nl;
    }

    constexpr bool gpuResidentStrict = true;
    const bool gpuResidentPureGasOnly = gpuScheduling.pureGasOnly;
    if (gpuResidentStrict)
    {
#ifndef UGKWP_USE_CUDA
        FatalErrorInFunction
            << "gasUGKP requires the CUDA GPU-resident solver build. "
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
#ifndef UGKWP_STEADY_APPLICATION
        const auto gasTimeStepControls=ugkp::solverControls::readGasTimeStepControls(runTime,mesh.name());
#endif

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
                true,
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
                g.value(),
                rhoMinStrict,
                TgasMinStrict,
                gpuScheduling,
                &sharedGasModel
            );
            configureResidentSst(resident);
            resident.configureMolecularTransport(molecularTransport.config());
            resident.configureBoundaryLayer(boundaryLayerInput, runTime, mesh, Tgas);



#ifdef UGKWP_STEADY_APPLICATION
            #include "../gasUGKPSteady/steadyGasLoop.H"
#else
            const auto advanceControls=runTime.advanceControls(gasTimeStepControls);
            resident.configureAdvance(runTime.initialAdvance(advanceControls),true);
            bool advancePending=false;
            while (runTime.run())
            {
                const auto request=runTime.advanceBoundary(advanceControls,progressLogInterval);
                if(!advancePending)resident.enqueueAdvance(request);
                const auto completed=resident.collectAdvance(&runTime);advancePending=false;
                runTime.publishAdvance(completed);

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
                    ugkp::gpu::writeGasOutput(runTime,false,[&]{
                        resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas);
                        resident.downloadSharedGasSpecies(sharedGasModel,rho);
                        resident.writeBoundaryLayerDiagnostics(boundaryLayerInput,runTime);
                        ugkp::gpu::mirrorGasTurbulence(resident,runTime,gasTurbulenceModel,k,omega,nut);
                        return []{};
                    },[&]{
                        resident.enqueueAdvance(runTime.advanceBoundary(advanceControls,progressLogInterval));
                        advancePending=true;
                    },[]{return true;});
                }
            }

#endif

            if (sharedGasModel.active()) Info<< "End" << nl;
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
            true,
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
            , &sharedGasModel
        );
        configureResidentSst(resident);

        resident.configureMolecularTransport(molecularTransport.config());

#ifdef UGKWP_STEADY_APPLICATION
        #include "../gasUGKPSteady/steadyParticleLoop.H"
#else
        const auto advanceControls=runTime.advanceControls(gasTimeStepControls);
        resident.configureAdvance(runTime.initialAdvance(advanceControls),false);
        bool advancePending=false;
        while (runTime.run())
        {
            const auto request=runTime.advanceBoundary(advanceControls,progressLogInterval);
            if(!advancePending)resident.enqueueAdvance(request);
            const auto completed=resident.collectAdvance(&runTime);advancePending=false;
            runTime.publishAdvance(completed);

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
                ugkp::gpu::writeGasOutput(runTime,false,[&]{
                    resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas,epsS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);
                    resident.downloadSharedGasSpecies(sharedGasModel,rho);
                    ugkp::gpu::mirrorGasTurbulence(resident,runTime,gasTurbulenceModel,k,omega,nut);
                    return resident.prepareParticleRestartMirror(runTime);
                },[&]{
                    resident.enqueueAdvance(runTime.advanceBoundary(advanceControls,progressLogInterval));
                    advancePending=true;
                },[]{return true;});
            }
        }

#endif

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
