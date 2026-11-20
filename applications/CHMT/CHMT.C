#include "Switch.H"
#include "FoamSolverScheme.H"
#include "../../common/HostMemory.H"
#include "DiagnosticsFoam.H"
#include "configuration/CouplingInput.H"
#include "ProgressLogging.H"
#include "PrecisionBuild.H"
#include "io/CoupledOutput.H"
#include "io/MultirateEvolution.H"
#ifndef CHMT_FRONTEND_COMPILE_ONLY
#include "io/CoupledRadiation.H"
#include "io/AcceptedOutputWriter.H"
#endif
#include "gpu/Backend.H"
#include "restart/Checkpoint.H"
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    Foam::argList::noParallel();
    Foam::argList::addBoolOption("build-info","Print actual source/compiler/build identity.");
    Foam::argList::addBoolOption("check-input","Validate coupled input only; does not advance time.");
    Foam::FatalError.throwExceptions();Foam::FatalIOError.throwExceptions();
    try {
        Foam::argList args(argc,argv);
        if(args.optionFound("build-info")){std::cout<<CHMT_BUILD_MANIFEST_JSON<<'\n';return 0;}
        if(!args.checkRootCase())return 2;
        Foam::Time time(Foam::Time::controlDictName,args);
        (void)ugkp::solverControls::startupScheme(time);
        std::cout << "GPU precision=" << ugkp::precision::compiledBits
                  << " (fixed at startup)" << std::endl;
        const Foam::label progressLogInterval =
            ugkwp::logging::progressLogInterval(time.controlDict());
        Foam::IOdictionary properties(Foam::IOobject("chmtProperties",time.constant(),time,
            Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        chmt::io::CoupledCaseInput input;chmt::io::readCoupledCase(time,properties,input);
        std::cout<<"CHMT gas wall treatment="
            <<(input.model.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer?"boundaryLayer":"lowRe")
            <<" layer model="<<(input.model.physics.wallModel.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport?"constantTransport":
                input.model.physics.wallModel.model==ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde?"equilibriumOde":"reactingSst")
            <<" equilibrium stress="<<(input.model.physics.wallModel.localStressCorrection?"localStress":"wallStress")<<'\n';
        if(args.optionFound("check-input")) {
            std::cout<<"CHMT coupled input valid; shared GPU evolution was not executed.\n";return 0;
        }
#ifdef CHMT_FRONTEND_COMPILE_ONLY
        std::cerr<<"CHMT input-only build has no CUDA backend; use the native build to evolve.\n";return 2;
#else
        std::string error;
        Foam::fileName restart;
        if(time.controlDict().found("restartDirectory")){
            restart=chmt::io::text(time.controlDict(),"restartDirectory");
            if(!restart.isAbsolute())restart=time.path()/restart;
            chmt::io::require(chmt::readCheckpoint(restart,input.model,input.state,error),error);

            for(auto& particle:input.state.particles)
                ugkp::thermal::normalizeParticleContactMechanicalEnergy(particle);
        }
        chmt::GasExecutionOptions options;
        options.adjustTimeStep=input.gasTime.adjustTimeStep;
        options.fixedDeltaT=time.controlDict().lookup<Foam::scalar>("deltaT");
        options.fluxScheme=input.gasNumerics.gasFluxScheme;options.reconstruction=input.gasNumerics.gasReconstruction;
        options.limiter=input.gasNumerics.gasLimiter;options.timeIntegrator=input.gasNumerics.gasTimeIntegrator;
        options.turbulenceModel=input.model.physics.enableSst?3:0;
        options.courantUpdateInterval=input.gasTime.courantUpdateInterval;
        options.maximumTimeStepGrowth=input.gasTime.maximumTimeStepGrowth;
        chmt::io::require(options.courantUpdateInterval>0&&std::isfinite(options.maximumTimeStepGrowth)
            &&options.maximumTimeStepGrowth>=1,"invalid gas Courant scheduling");
        std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> backend(
            chmt::createBackend(input.model,input.gasModel,input.gasMechanism,options,input.state,error),chmt::destroyBackend);
        chmt::io::require(bool(backend),error);
        chmt::setDiagnosticsSink(*backend,ugkp::diagnostics::makeFoamSink());


        std::cout<<"CHMT_TIME_CONTROL adjustTimeStep="<<options.adjustTimeStep
            <<" deltaT="<<options.fixedDeltaT<<" source=system/controlDict readOnce=1"<<std::endl;
        std::cout<<"CHMT_SOLVER_CONTROLS gasMinDt="<<input.model.physics.minDt
            <<" gasMaxDt="<<input.model.physics.maxDt<<" gasMaxCo="<<input.model.physics.cfl
            <<" materialMaxSubstep="<<input.material.maxSubstep
            <<" materialMaxSubsteps="<<input.material.maxSubsteps
            <<" materialNonlinearMaxIterations="<<input.material.nonlinearMaxIterations
            <<" materialNonlinearRelativeTolerance="<<input.material.nonlinearRelativeTolerance
            <<" materialEnergyAbsoluteTolerance="<<input.material.energyAbsoluteTolerance
            <<" materialEnergyRelativeTolerance="<<input.material.energyRelativeTolerance
            <<" materialLinearMaxIterations="<<input.material.linearMaxIterations
            <<" materialLinearEnergyBudgetFraction="<<input.material.linearEnergyBudgetFraction
            <<" thermoMaximumIterations="<<input.model.physics.numerics.thermoMaximumIterations
            <<" reconstructionMaximumIterations="<<input.model.physics.numerics.reconstructionMaximumIterations
            <<" surfaceCompositionMaximumIterations="<<input.model.physics.numerics.surface.compositionMaximumIterations
            <<" surfaceThermalMaximumIterations="<<input.model.physics.numerics.surface.thermalMaximumIterations
            <<" meshMaximumIterations="<<input.model.physics.numerics.meshMotion.maximumIterations
            <<" particlesActive="<<input.model.physics.enableParticles
            <<" finiteContactMaximumIterations="<<input.model.particleNumerics.finiteContactMaximumIterations
            <<" packingIterations="<<input.model.particleNumerics.packingIterations
            <<" maxFaceWalkHops="<<input.model.particlePhase.maxFaceWalkHops
            <<" contactNonlinearIterations="<<input.model.particleContact.nonlinearIterations
            <<std::endl;
        chmt::CpuMaterialDriver material(input.model,input.solidMesh.get(),&input.gasMechanism);
        std::unique_ptr<chmt::CoupledRadiation> radiation;
        if(input.thermalSchedule.radiationEnabled)
            radiation.reset(new chmt::CoupledRadiation(time,*input.gasMesh,input.model,input.state,input.thermalSchedule.radiationFrequency));
        if(radiation&&!restart.empty()){
            radiation->readWarmStart((restart/"domWarmStart.bin").c_str());
        }
        std::cout<<"CHMT couplingInterval="<<input.coupling.interval<<" radiationFrequency="
            <<(radiation?input.thermalSchedule.radiationFrequency:0)<<std::endl;
        const chmt::Real end=time.controlDict().lookup<Foam::scalar>("endTime");
        const Foam::fileName directory=time.path()/chmt::io::text(time.controlDict(),"outputDirectory","chmtOutput");
        namespace fs=std::filesystem;
        const fs::path outputPath(directory.c_str());
        std::error_code ec;
        fs::create_directories(outputPath,ec);
        chmt::io::require(!ec,"cannot open coupled output directory "+outputPath.string()+": "+ec.message());
        std::optional<std::uint64_t> resumedStep;
        if(!restart.empty()&&fs::equivalent(fs::path(restart.c_str()),
            outputPath/("checkpoint-"+std::to_string(input.state.acceptedSteps)),ec))
            resumedStep=input.state.acceptedSteps;
        const auto& outputControls=time.controlDict();
        const auto diagnosticInterval=outputControls.lookupOrDefault<Foam::label>("diagnosticInterval",1);
        const auto fieldInterval=outputControls.lookupOrDefault<Foam::label>("fieldInterval",1);
        const auto checkpointInterval=outputControls.lookupOrDefault<Foam::label>("checkpointInterval",1);
        chmt::io::require(diagnosticInterval>=0&&fieldInterval>=0&&checkpointInterval>=0,"output intervals must be nonnegative accepted-step counts");
        const bool writeInitial=chmt::io::flag(outputControls,"writeInitial",true);
        const bool writeFinal=chmt::io::flag(outputControls,"writeFinal",true);
        chmt::io::AcceptedOutputWriter output(directory.c_str(),resumedStep,input.model,ugkp::diagnostics::makeFoamSink());
        const auto selection=[&](std::uint64_t step,bool force){
            const auto due=[&](Foam::label interval){return force||(interval>0&&step%interval==0);};
            return chmt::io::OutputSelection{due(diagnosticInterval),due(fieldInterval),due(checkpointInterval)};
        };
        const auto publish=[&](chmt::io::OutputSelection choice,double radiationInterval=0){
            output.submit(choice,*backend,input.state,radiation.get(),radiationInterval);
        };
        try {
        if(writeInitial||(writeFinal&&input.state.time>=end))publish(selection(input.state.acceptedSteps,true));
        input.state.ledger.clear();
        while(input.state.time<end){
            output.collectReady();
            auto choice=selection(input.state.acceptedSteps+1,false);
            chmt::WindowEvolutionReport report;
            chmt::io::require(chmt::advanceCoupledWindow(*backend,material,input.model,input.coupling,input.material,end,false,input.state,report,error,radiation.get(),false),error);
            publish(writeFinal&&input.state.time>=end?selection(input.state.acceptedSteps,true):choice,report.radiationInterval);
            input.state.ledger.clear();ugkp::releaseUnusedHostPages();
            if
            (
                ugkwp::logging::progressLogDue
                (
                    input.state.acceptedSteps,
                    progressLogInterval
                )
            )
            {
                std::cout << "CHMT_REPRO step=" << input.state.acceptedSteps
                    << " time=" << input.state.time
                    << " radiationTime=" << input.state.radiationTime
                    << " gasMicrosteps=" << report.acceptedGasMicrosteps
                    << " materialSolves=" << report.materialSolves << std::endl;
            }
        }
        } catch(...) {

            try {
                auto secondary=output.collect();
                if(secondary.hasFatal){
                    const auto* source=ugkp::diagnostics::findSource(secondary,secondary.fatal.sourceSiteId);
                    Foam::SeriousError<<ugkp::diagnostics::formatRecord(secondary.fatal,source).c_str()<<Foam::endl;
                    secondary.hasFatal=false;
                }
                ugkp::diagnostics::dispatch(secondary,ugkp::diagnostics::makeFoamSink());
            } catch(const std::exception& secondary) {
                Foam::SeriousError<<secondary.what()<<Foam::endl;
            }
            throw;
        }
        output.finish();
        return 0;
#endif
    } catch(const Foam::error& error) {
        Foam::SeriousError<<"CHMT OpenFOAM input failure: "<<error<<Foam::endl;return 2;
    } catch(const std::exception& error) {
        std::cerr<<"CHMT failed: "<<error.what()<<'\n';return 2;
    }
}
