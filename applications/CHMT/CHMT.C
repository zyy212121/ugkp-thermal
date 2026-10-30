#include "DiagnosticsFoam.H"
#include "configuration/CouplingInput.H"
#include "ProgressLogging.H"
#include "io/CoupledOutput.H"
#include "io/MultirateEvolution.H"
#include "gpu/Backend.H"
#include "restart/Checkpoint.H"
#include <fstream>
#include <iostream>
// Coupling-only application. Gas physics and stage ordering have common owners.
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
        if(properties.found("restartDirectory")){
            Foam::fileName restart(chmt::io::text(properties,"restartDirectory"));
            if(!restart.isAbsolute())restart=time.path()/restart;
            chmt::io::require(chmt::readCheckpoint(restart,input.model,input.state,error),error);
            // Older checkpoints may contain an unnormalized initial diagnostic.
            for(auto& particle:input.state.particles)
                ugkp::thermal::normalizeParticleContactMechanicalEnergy(particle);
        }
        chmt::GasExecutionOptions options;
        options.fluxScheme=input.gasNumerics.gasFluxScheme;options.reconstruction=input.gasNumerics.gasReconstruction;
        options.limiter=input.gasNumerics.gasLimiter;options.timeIntegrator=input.gasNumerics.gasTimeIntegrator;
        options.turbulenceModel=input.model.physics.enableSst?3:0;
        std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> backend(
            chmt::createBackend(input.model,input.gasModel,input.gasMechanism,options,input.state,error),chmt::destroyBackend);
        chmt::io::require(bool(backend),error);
        chmt::setDiagnosticsSink(*backend,ugkp::diagnostics::makeFoamSink());
        chmt::CpuMaterialDriver material(input.model,input.solidMesh.get(),&input.gasMechanism);
        const chmt::Real end=time.controlDict().lookup<Foam::scalar>("endTime");
        const Foam::fileName directory=time.path()/chmt::io::text(properties,"outputDirectory","chmtOutput");
        chmt::io::require(!Foam::isDir(directory)&&!Foam::isFile(directory),"output directory already exists");
        chmt::io::require(Foam::mkDir(directory),"cannot create coupled output directory");
        const auto& outputControls=properties.found("outputControls")?properties.subDict("outputControls"):properties;
        const auto diagnosticInterval=outputControls.lookupOrDefault<Foam::label>("diagnosticInterval",1);
        const auto fieldInterval=outputControls.lookupOrDefault<Foam::label>("fieldInterval",1);
        const auto checkpointInterval=outputControls.lookupOrDefault<Foam::label>("checkpointInterval",1);
        chmt::io::require(diagnosticInterval>=0&&fieldInterval>=0&&checkpointInterval>=0,"output intervals must be nonnegative accepted-step counts");
        const bool writeInitial=chmt::io::flag(outputControls,"writeInitial",true);
        const bool writeFinal=chmt::io::flag(outputControls,"writeFinal",true);
        auto publish=[&](bool force){
            const auto step=input.state.acceptedSteps;
            const bool diagnostics=force||(diagnosticInterval>0&&step%diagnosticInterval==0);
            const bool full=force||(fieldInterval>0&&step%fieldInterval==0);
            const bool checkpoint=force||(checkpointInterval>0&&step%checkpointInterval==0);
            if(!diagnostics&&!full&&!checkpoint)return;
            const std::string suffix=std::to_string(input.state.acceptedSteps);
            if(diagnostics){
            std::ofstream summary((directory/Foam::fileName("accepted-"+suffix+".csv")).c_str());
            chmt::io::require(chmt::io::writeAcceptedCoupledSummary(summary,input.state,input.model,error),error);
            }
            if(full){
            std::ofstream fields((directory/Foam::fileName("gas-"+suffix+".csv")).c_str());
            chmt::io::require(chmt::io::writeAcceptedGasFields(fields,input.state,input.model,error),error);
            if(input.model.physics.enableParticles){
                std::ofstream particles((directory/Foam::fileName("particles-"+suffix+".csv")).c_str());
                chmt::io::require(chmt::io::writeAcceptedParticleFields(particles,input.state,error),error);
            }
            std::ofstream solid((directory/Foam::fileName("material-"+suffix+".csv")).c_str());
            chmt::io::require(chmt::io::writeAcceptedMaterialFields(solid,input.state,input.model,error),error);
            std::ofstream exchange((directory/Foam::fileName("exchange-"+suffix+".csv")).c_str());
            chmt::io::require(chmt::io::writeAcceptedExchangeFields(exchange,input.state,input.model,error),error);
            if(input.model.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
                std::ofstream wall((directory/Foam::fileName("wall-layer-"+suffix+".csv")).c_str());
                chmt::io::require(chmt::io::writeAcceptedWallLayerFields(wall,input.state,input.model,error),error);
            }
            }
            if(checkpoint)chmt::io::require(chmt::writeCheckpoint(directory/Foam::fileName("checkpoint-"+suffix),input.model,input.state,error),error);
        };
        if(writeInitial||(writeFinal&&input.state.time>=end))publish(true);
        while(input.state.time<end){
            chmt::WindowEvolutionReport report;
            chmt::io::require(chmt::advanceCoupledWindow(*backend,material,input.model,input.coupling,input.material,end,false,input.state,report,error),error);
            publish(writeFinal&&input.state.time>=end);
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
                    << " gasMicrosteps=" << report.acceptedGasMicrosteps
                    << " materialSolves=" << report.materialSolves << std::endl;
            }
        }
        return 0;
#endif
    } catch(const Foam::error& error) {
        Foam::SeriousError<<"CHMT OpenFOAM input failure: "<<error<<Foam::endl;return 2;
    } catch(const std::exception& error) {
        std::cerr<<"CHMT failed: "<<error.what()<<'\n';return 2;
    }
}
