#include "materials/ImplicitMaterialMass.H"
#include "materials/CpuMaterialDriver.H"
#include "gpu/MaterialBackend.H"
#include "gpu/ResidentMaterialDrive.H"
#include "gpu/MaterialThermalBackend.H"
#include "../../../common/gpu/FoamSparseSolve.H"
#include "../../../common/gpu/FoamImplicitDiffusion.H"
#include "core/ThermalBoundary.H"
#include "core/AuditArithmetic.H"
#include "core/HostAuditReduction.H"
#include "materials/MaterialTransport.H"
#include "materials/MaterialDrive.H"
#include "materials/MaterialSourceAudit.H"
#include "materials/MaterialRecessionMath.H"
#include "materials/SurfacePacketAddressing.H"
#include "materials/MaterialStepReuse.H"
#include "ablation/CpuSurfaceInterface.H"
#include "ablation/WallClosureHost.H"
#include "mesh/Geometry.H"
#include "mesh/MeshConnectivity.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "coupling/IntervalAudit.H"
#include "fvCFD.H"
#include "OStringStream.H"
#include "DiagnosticsFoam.H"
#include "DiagnosticsCatalogue.H"
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include <algorithm>
#include <stdexcept>
namespace chmt { namespace {
void materialWarning(const HostState& state,ugkp::diagnostics::Code code,const char* message,Real observed,Real reference=0) {
    using namespace ugkp::diagnostics;
    auto record=makeRecord(code,4101,4,0,EntityKind::None,-1,-1,observed,reference,observed,
        code==Code::ConservationResidual?Quantity::Conservation:Quantity::Accuracy);
    record.step=state.acceptedSteps;record.simulationTime=state.time;record.operationSequence=state.commitSequence+1;
    Snapshot snapshot;snapshot.warnings.push_back({record,1,false});
    snapshot.sources.push_back({4101,"applications/CHMT/materials/CpuMaterialDriver.C","materialWarning",__LINE__,message});
    dispatch(snapshot,makeFoamSink());
}
Foam::pointField foamPoints(const std::vector<Vec3>& points){Foam::pointField result(points.size());for(std::size_t i=0;i<points.size();++i)result[i]=Foam::point(points[i].x,points[i].y,points[i].z);return result;}
bool controlsValid(const CpuMaterialControls& c){return (finite(c.maxSubstep)&&c.maxSubstep>=0&&((finite(c.transportCfl)&&c.transportCfl>0))
    &&((finite(c.reactionCfl)&&c.reactionCfl>0&&finite(c.reactionMaxSubstep)&&c.reactionMaxSubstep>=0))
    &&c.driveSourceDiagnosticSamples>=0&&finite(c.driveRelativeTolerance)&&c.driveRelativeTolerance>=0&&finite(c.driveSourceFraction)&&c.driveSourceFraction>0
    &&finite(c.driveTractionScale)&&c.driveTractionScale>0&&(c.maxSubsteps>0)&&((c.nonlinearMaxIterations>0&&finite(c.nonlinearRelativeTolerance)&&c.nonlinearRelativeTolerance>=0
    &&finite(c.energyAbsoluteTolerance)&&c.energyAbsoluteTolerance>=0&&finite(c.energyRelativeTolerance)&&c.energyRelativeTolerance>=0)));}
struct MaterialGeometryInventory {
    std::vector<int> cells,row;
    std::vector<SolidQ> values;
    MaterialGeometryInventory(const HostState& base,const IntervalHistory& history,bool inspectPackets=true):row(base.solid.size(),-1){
        const auto include=[&](int cell){
            if(cell>=0&&std::size_t(cell)<row.size()&&row[cell]<0){
                row[cell]=cells.size();cells.push_back(cell);values.push_back(base.solid[cell]);
            }
        };
        for(int cell:base.surface.solidCell)include(cell);
        if(inspectPackets)for(std::size_t i=0;i<history.records().size();++i)for(const auto& packet:history.readRecord(i).packets)
            if(packet.kind==ExchangeKind::GasSolid)include(packet.solidCell);
    }
    const SolidQ& at(int cell)const{return values.at(row.at(cell));}
};
#include "MaterialConductionWorkspace.inl"
bool nativeConduction(Foam::fvMesh& acceptedMesh,const HostState& old,HostState& next,const PhysicsConfig& p,
    Real dt,const CpuMaterialControls& controls,CpuMaterialReport& report,std::string& error,ugkp::gpu::SparseConjugateGradient& sparse,MaterialBackend& thermal,
    MaterialConductionWorkspace& workspace,bool* preparedParent=nullptr){
    if(preparedParent)*preparedParent=false;
    auto& residentDiffusion=workspace.diffusion;
    const std::size_t n=next.solidMesh.volumes.size();if(n==0)return true;
    next.solid.resize(n);
    if(!prepareConductionWorkspace(acceptedMesh,next,p,workspace,error)){
        report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
    }
    auto& mesh=*workspace.mesh;auto& temperature=*workspace.temperature;
    std::vector<MaterialCaloric> caloric(n);std::vector<Real> anchor(n),target(n);
    std::vector<MaterialThermalInput> thermalInput;bool numerical=false;
    const std::vector<Real> nativeVolumes(mesh.V().begin(),mesh.V().end());
    Foam::dictionary solver;solver.add("solver",Foam::word("PCG"));solver.add("preconditioner",Foam::word("jacobi"));solver.add("tolerance",Foam::scalar(1e-12));solver.add("relTol",Foam::scalar(0));solver.add("maxIter",Foam::label(controls.linearMaxIterations));
    bool residentConvergence=false;std::vector<int> externalBoundary;
    forAll(mesh.boundary(),patch)forAll(mesh.boundary()[patch],f)externalBoundary.push_back(mesh.boundary()[patch].coupled()?0:1);
    auto& capacity=*workspace.capacity;auto& conductivity=*workspace.conductivity;auto& source=*workspace.source;
    bool deviceStartup=ugkp::gpu::residentMaterialStartupSupported(conductivity,temperature);
    if(preparedParent&&!deviceStartup){error.clear();return true;}
    if(!prepareResidentMaterialConduction(thermal,next.solidMesh,nativeVolumes,thermalInput,error,numerical,!deviceStartup)){
        if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
    }
    if(residentMaterialConductionFields(thermal).cells!=n){error="resident material conduction inventory/mesh extent mismatch";return false;}
    if(!deviceStartup)for(std::size_t c=0;c<n;++c){const auto& input=thermalInput[c];
        caloric[c]=input.caloric;anchor[c]=input.anchor;target[c]=input.target;temperature[c]=input.anchor;}
    const auto initialDeviceFields=residentMaterialConductionFields(thermal);
    residentDiffusion.copySolution(initialDeviceFields.initialTemperature,n,initialDeviceFields.readyStream);
    bool startupDiagnosticsEmitted=false;
    for(int iteration=0;iteration<(controls.nonlinearMaxIterations);++iteration){Foam::scalarField previous(temperature.primitiveField());


        capacity.primitiveFieldRef()=Foam::scalar(0);conductivity.primitiveFieldRef()=Foam::scalar(0);source.primitiveFieldRef()=Foam::scalar(0);
        forAll(mesh.boundary(),patch){
            capacity.boundaryFieldRef()[patch]==Foam::scalar(0);
            conductivity.boundaryFieldRef()[patch]==Foam::scalar(0);
            source.boundaryFieldRef()[patch]==Foam::scalar(0);
        }
        std::vector<Real> trial(temperature.primitiveField().begin(),temperature.primitiveField().end());
        std::vector<MaterialThermalCoefficients> coefficients;
        if(!evaluateResidentMaterialConduction(thermal,trial,dt,coefficients,error,
            residentDiffusion.deviceSolution(),residentDiffusion.deviceValues().readyStream,!residentConvergence&&!deviceStartup,deviceStartup&&iteration==0&&!preparedParent)){
            report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}

        if(deviceStartup&&iteration==0)startupDiagnosticsEmitted=true;
        if(!residentConvergence&&!deviceStartup)for(std::size_t c=0;c<n;++c){
            const auto& value=coefficients[c];
            if(value.error){error=value.error==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            capacity[c]=value.capacity;source[c]=value.source;conductivity[c]=value.conductivity;


            if(!(startupDiagnosticsEmitted&&iteration==0)&&value.rawPorosity!=value.porosity){Real phi=value.rawPorosity;clipPhysicsMassFraction(p,phi,3824,static_cast<int>(c));}
            next.solid[c].porosity=value.porosity;
        }
        if(!residentConvergence&&!deviceStartup){capacity.correctBoundaryConditions();conductivity.correctBoundaryConditions();temperature.correctBoundaryConditions();}
        Foam::surfaceScalarField faceK(ugkp::gpu::thermalHarmonicBoundaryValues(conductivity));
        const auto deviceFields=residentMaterialConductionFields(thermal);
        auto equation=ugkp::gpu::residentImplicitThermalEquation(residentDiffusion,sparse,faceK,temperature,capacity,source,dt,false,
            deviceFields.capacity,deviceFields.source,deviceFields.readyStream,deviceFields.conductivity,true,&conductivity,&faceK);
        residentConvergence=residentDiffusion.residentIteration();
        if(residentConvergence&&equation.configureResidentLocalSolve(solver)){



            equation.residentBoundaryFlux();
            if(preparedParent){
                if(!prepareMaterialLocalConduction(thermal,residentDiffusion,sparse,externalBoundary,dt,controls,
                    Foam::solverPerformance::small_,Foam::solverPerformance::vsmall_,error))return false;
                *preparedParent=true;return true;
            }
            if(!solveResidentMaterialConduction(thermal,residentDiffusion,sparse,externalBoundary,dt,controls,
                Foam::solverPerformance::small_,Foam::solverPerformance::vsmall_,error)){
                report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
            }
            const std::vector<Real> unusedTemperature;
            if(!finishResidentMaterialConduction(thermal,unusedTemperature,next.solid,error,numerical,
                residentDiffusion.deviceTemperature(),residentDiffusion.deviceValues().readyStream)){
                if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
            }
            return true;
        }
        if(preparedParent){error.clear();return true;}
        if(deviceStartup){



            if(!publishResidentMaterialConductionInput(thermal,thermalInput,error))return false;
            for(std::size_t c=0;c<n;++c){const auto& input=thermalInput[c];
                caloric[c]=input.caloric;anchor[c]=input.anchor;target[c]=input.target;temperature[c]=input.anchor;}
            deviceStartup=false;residentConvergence=false;--iteration;continue;
        }




        long double initialEnergy=0,inventoryRoundoff=0;
        const Real epsilonScale=64*std::numeric_limits<Real>::epsilon();
        if(residentConvergence){
            const auto residual=equation.residentResidual();MaterialConductionAudit audit;
            if(!auditResidentMaterialConduction(thermal,residentDiffusion.deviceSolution(),nullptr,residual,nullptr,externalBoundary,dt,
                residentDiffusion.deviceValues().readyStream,audit,error))return false;
            if(audit.coefficientError){error=audit.coefficientError==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve";
                report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            initialEnergy=hostAuditGlobalSum(audit.residualEnergy.extended(),mesh.comm());
            inventoryRoundoff=hostAuditGlobalSum(static_cast<long double>(epsilonScale)*audit.inventory.extended(),mesh.comm());
        }else{
            const Foam::tmp<Foam::scalarField> powerResidual=equation.residual();Real powerScale=0;
            for(const Real value:powerResidual())powerScale=maxValue(powerScale,absValue(value));
            Foam::reduce(powerScale,Foam::maxOp<Real>(),Foam::Pstream::msgType(),mesh.comm());Real scaledPower=0;
            if(powerScale>0)for(const Real value:powerResidual())scaledPower+=absValue(value)/powerScale;
            Foam::reduce(scaledPower,Foam::sumOp<Real>(),Foam::Pstream::msgType(),mesh.comm());
            for(std::size_t c=0;c<n;++c)inventoryRoundoff=hostAuditAdd(inventoryRoundoff,
                static_cast<long double>(epsilonScale)*absValue(target[c])+static_cast<long double>(epsilonScale)*absValue(caloric[c].energy<false>(temperature[c])));
            inventoryRoundoff=hostAuditGlobalSum(inventoryRoundoff,mesh.comm());initialEnergy=static_cast<long double>(dt)*powerScale*scaledPower;
        }
        const long double linearEnergyTolerance=controls.energyAbsoluteTolerance
            +controls.energyRelativeTolerance*std::max(1.L,initialEnergy)+inventoryRoundoff;


        const Real relativeLinearTarget=initialEnergy>0
            ?static_cast<Real>(std::min(1.L,(linearEnergyTolerance/initialEnergy)*controls.linearEnergyBudgetFraction)):1;
        solver.set("tolerance",std::numeric_limits<Foam::scalar>::min());
        solver.set("relTol",minValue(1,relativeLinearTarget));
        const Foam::solverPerformance linear=equation.solve(solver,!residentConvergence);
        if(!finite(linear.finalResidual())){error="nonfinite OpenFOAM implicit material sparse residual";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
        if(!linear.converged()||linear.singular())materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,"implicit material sparse residual",linear.finalResidual(),linear.initialResidual()*minValue(1,relativeLinearTarget));
        Real change=0;long double boundary=0,stableChange=0,inventoryScale=0;
        if(residentConvergence){
            residentDiffusion.refreshBoundaryTemperature();const auto flux=equation.residentBoundaryFlux();MaterialConductionAudit audit;
            if(!auditResidentMaterialConduction(thermal,residentDiffusion.deviceSolution(),residentDiffusion.devicePreviousSolution(),nullptr,flux,
                externalBoundary,dt,residentDiffusion.deviceValues().readyStream,audit,error))return false;
            if(audit.invalidCell>=0){using namespace ugkp::diagnostics;
                const Real T=audit.invalidTemperature;recordFatal(p.diagnostics,makeRecord(!finite(T)?Code::NonFinite:Code::NegativeTemperature,
                    4103,4,0,EntityKind::Cell,audit.invalidCell,-1,T,0,0,Quantity::Temperature));
                error="nonfinite or nonpositive implicit material temperature";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            boundary=audit.boundaryEnergy.extended();stableChange=audit.stableChange.extended();inventoryScale=audit.inventory.extended();change=audit.change;
        }else{
            temperature.correctBoundaryConditions();const auto flux=equation.boundaryFlux();
        forAll(flux,patch)if(!mesh.boundary()[patch].coupled())forAll(flux[patch],f)boundary=hostAuditAdd(boundary,static_cast<long double>(dt)*flux[patch][f]);
        for(std::size_t c=0;c<n;++c){const Real T=temperature[c];if(!finite(T)||T<=0){
                using namespace ugkp::diagnostics;
                recordFatal(p.diagnostics,makeRecord(!finite(T)?Code::NonFinite:Code::NegativeTemperature,4103,4,0,EntityKind::Cell,c,-1,T,0,0,Quantity::Temperature));
                error="nonfinite or nonpositive implicit material temperature";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            stableChange=hostAuditAdd(stableChange,static_cast<long double>(caloric[c].secant<false>(T,anchor[c]))*(T-anchor[c])
                -(static_cast<long double>(target[c])-caloric[c].energy<false>(anchor[c])));
            inventoryScale=hostAuditAdd(inventoryScale,static_cast<long double>(absValue(target[c]))+absValue(caloric[c].energy<false>(T)));change=maxValue(change,absValue(T-previous[c])/maxValue(1,maxValue(absValue(T),absValue(previous[c]))));}
        }

        const long double globalChangeEnergy=hostAuditGlobalSum(stableChange,mesh.comm());
        boundary=hostAuditGlobalSum(boundary,mesh.comm());
        Foam::reduce(change,Foam::maxOp<Real>(),Foam::Pstream::msgType(),mesh.comm());
        const Real residual=hostAuditNarrow(hostAuditAdd(globalChangeEnergy,boundary));
        const long double scale=std::max(1.L,std::max(std::abs(globalChangeEnergy),std::abs(boundary)));
        const long double roundoff=64*std::numeric_limits<Real>::epsilon()*inventoryScale;
        const long double globalRoundoff=hostAuditGlobalSum(roundoff,mesh.comm());
        if(!finite(change)){
            using namespace ugkp::diagnostics;
            recordFatal(p.diagnostics,makeRecord(Code::NonFinite,4104,4,0,EntityKind::None,-1,-1,change));
            error="nonfinite implicit material residual";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
        }
        const Real energyTolerance=hostAuditNarrow(hostAuditAdd(controls.energyAbsoluteTolerance+controls.energyRelativeTolerance*scale,globalRoundoff));
        const bool energyConverged=finite(residual)&&absValue(residual)<=energyTolerance;
        const bool stalled=change==0&&linear.nIterations()==0;
        if((change<=controls.nonlinearRelativeTolerance&&energyConverged)
            ||stalled||iteration+1==controls.nonlinearMaxIterations){
            if(stalled&&!energyConverged)materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,
                "implicit material iteration stagnated above energy goal",residual,energyTolerance);
            if(change>controls.nonlinearRelativeTolerance)materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,"implicit material nonlinear residual",change,controls.nonlinearRelativeTolerance);
            if(!finite(residual)||absValue(residual)>energyTolerance)materialWarning(next,ugkp::diagnostics::Code::ConservationResidual,"implicit material energy balance",residual,energyTolerance);
            if(residentConvergence)equation.publishSolution();
            const std::vector<Real> endpoint(temperature.primitiveField().begin(),temperature.primitiveField().end());
            if(!finishResidentMaterialConduction(thermal,endpoint,next.solid,error,numerical,
                residentDiffusion.deviceSolution(),residentDiffusion.deviceValues().readyStream)){
                if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            return true;}
    }
    error="OpenFOAM material caloric nonlinear iteration failed";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
}


std::vector<FilmAux> dryGeometryAux(const HostState& state){
    auto aux=state.filmAux;const auto oldSize=aux.size();aux.resize(state.surface.area.size());
    for(std::size_t f=0;f<aux.size();++f){
        aux[f].area=state.surface.area[f];aux[f].normal=state.surface.normal[f];
        aux[f].baseVelocity=state.surface.baseVelocity.empty()?Vec3{}:state.surface.baseVelocity[f];
        aux[f].thickness=0;
        if(f>=oldSize){aux[f].meanVelocity=aux[f].baseVelocity;aux[f].topVelocity=aux[f].baseVelocity;}
    }
    return aux;
}
bool baseTrace(const HostState& base,const PhysicsConfig& p,std::vector<GasPrimitive>& bulk,std::string& error,bool* numerical=nullptr){
    if(numerical)*numerical=false;
    bulk.resize(base.surface.area.size());
    if(base.surface.gasFace.size()!=bulk.size()){error="wall predictor face map size mismatch";return false;}
    for(std::size_t f=0;f<bulk.size();++f){const int face=base.surface.gasFace[f];if(face<0)continue;
        if(static_cast<std::size_t>(face)>=base.gasMesh.owner.size()){if(numerical)*numerical=true;error="wall predictor face outside gas mesh";return false;}

        const int c=base.gasMesh.owner[face];
        if((c<0||static_cast<std::size_t>(c)>=base.gas.size()||static_cast<std::size_t>(c)>=base.gasMesh.volumes.size())){if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
        if(!recoverGas(base.gas[c],gasPhaseVolume(base,c),p,bulk[f],nullptr,c)){
            {if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
        }
    }
    return true;
}
}
struct CpuMaterialDriver::Implementation {
    std::unique_ptr<ugkp::gpu::SparseConjugateGradient> conduction;
    MaterialConductionWorkspace conductionWorkspace;
    bool parentPrepared=false,parentConduction=false;
    ModelConfig model;Foam::fvMesh* mesh;
    ugkwp::GasMechanismConfiguration mechanism;
    std::unique_ptr<WallClosureHost> wall;bool terminalWallFailure=false,surfaceNumericalFailure=false;
    std::unique_ptr<MaterialBackend,decltype(&destroyMaterialBackend)> gpu{nullptr,destroyMaterialBackend};
    Implementation(const ModelConfig& m,Foam::fvMesh* region,const ugkwp::GasMechanismConfiguration* canonical):model(m),mesh(region){
        if(m.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){wall.reset(new WallClosureHost);if(canonical)mechanism=*canonical;}
    }
    bool begin(const HostState& state,const PhysicsConfig& p,std::string& error){
        if(!gpu){
            gpu.reset(createMaterialBackend(error));
            if(!gpu)return false;
            setMaterialDiagnosticsSink(*gpu,ugkp::diagnostics::makeFoamSink());
        }
        return beginMaterialWindow(*gpu,state,p,error);
    }
    bool evaluate(const HostState& state,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,
        const std::vector<GasGradient>& gradients,bool useTransportRates,bool candidateState,
        Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
        const std::vector<Real>* sideVolume=nullptr,
        const std::vector<GasWallMatchingSample>* recorded=nullptr,
        const std::vector<WallFaceSample>* acceptedRates=nullptr,int localAudit=-1){
        std::vector<GasWallClosureContext> contexts;terminalWallFailure=false;surfaceNumericalFailure=false;
        if(wall){
            terminalWallFailure=true;
            auto config=p.wallModel;config.enableSst=p.enableSst;
            if(!wall->prepareGeometry(state.gasMesh,state.surface.gasFace,error,config,p.diagnostics))return false;
            std::vector<GasWallMatchingSample> fresh;
            if(!recorded){if(!wall->sample(state,p,fresh,error))return false;recorded=&fresh;}
            ugkwp::GasMechanismView<Real,Ns> view;
            if(p.gasMode==ugkwp::GasMode::MixtureChemistry){
                view=mechanism.mechanismView<Ns>();}
            if(!wall->contexts(*recorded,view,p.wallModel,contexts,error,&surfaceNumericalFailure)){
                {
                    terminalWallFailure=!surfaceNumericalFailure;return false;
                }
            }
        }
        terminalWallFailure=false;
        return evaluateResidentMaterialSurface(*gpu,state,bulk,gradients,dt,
            useTransportRates,candidateState,output,error,surfaceNumericalFailure,
            sideVolume,wall?&contexts:nullptr,&terminalWallFailure,acceptedRates,localAudit);
    }
#include "MaterialHistoryEvaluation.inl"

};
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig& model,Foam::fvMesh* mesh):CpuMaterialDriver(model,mesh,nullptr){}
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig& model,Foam::fvMesh* mesh,const ugkwp::GasMechanismConfiguration* mechanism):data_(new Implementation(model,mesh,mechanism)){}
CpuMaterialDriver::~CpuMaterialDriver()=default;
#include "MaterialWallPrediction.inl"
bool CpuMaterialDriver::predictWallImpl(const HostState& base,const CouplingInterval& interval,WallProgram& output,std::string& error,CpuMaterialReport* diagnostics,const PhysicsConfig& p,bool holdRecession)const{
    if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=false;}
    const auto fail=[&](MaterialFailureOrigin origin){if(diagnostics)diagnostics->failureOrigin=origin;return false;};
    if(!(validateMaterialSweepRemainder(base,error,p.diagnostics)))return fail(MaterialFailureOrigin::GeometryNumerical);
    if((!validCouplingInterval(interval)||interval.begin!=base.time)){error="wall predictor interval/base time mismatch";return fail(MaterialFailureOrigin::GeometryNumerical);}
    if(!data_->begin(base,p,error))return fail(MaterialFailureOrigin::None);
    std::vector<GasPrimitive> bulk;bool numerical=false;
    if(!(baseTrace(base,p,bulk,error,&numerical)))return fail(numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);CpuSurfaceResult surface;
    if(!data_->evaluate(base,p,bulk,{},false,false,interval.end-interval.begin,interval.identity.sequence,surface,error,nullptr,nullptr)){if(diagnostics)diagnostics->recoverable=false;return fail(data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);}
    WallProgram program;program.interval=interval;program.surface=base.surface;WallKnot first;first.time=interval.begin;first.faces=surface.wall;first.gasPoints=base.gasMesh.points;first.solidPoints=base.solidMesh.points;
    if(holdRecession&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession)
        for(auto& face:first.faces){face.normalVelocity=0;face.solidNormalVelocity=0;}
    WallKnot last=first;last.time=interval.end;
    if(p.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){
        if(!base.surface.area.empty()){error="prescribed sinusoidal motion with material interface requires a compatible imposed transfer model";return fail(MaterialFailureOrigin::GeometryNumerical);}
        const auto& motion=p.meshMotion;const Real temporal=::sin(motion.angularFrequency*(interval.end-motion.timeOrigin));
        if((base.gasMesh.referencePoints.size()!=last.gasPoints.size())){error="prescribed gas motion reference topology missing";return fail(base.gasMesh.referencePoints.size()<last.gasPoints.size()?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical);}
        for(std::size_t i=0;i<last.gasPoints.size();++i){const Vec3 X=base.gasMesh.referencePoints[i];last.gasPoints[i]={X.x+motion.amplitude.x*::sin(motion.spatialWaveNumber.x*(X.x-motion.spatialOrigin.x))*temporal,
            X.y+motion.amplitude.y*::sin(motion.spatialWaveNumber.y*(X.y-motion.spatialOrigin.y))*temporal,X.z+motion.amplitude.z*::sin(motion.spatialWaveNumber.z*(X.z-motion.spatialOrigin.z))*temporal};}
    }
    if(!holdRecession&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.area.empty()){HostState geometryBase=base;geometryBase.filmAux=dryGeometryAux(base);auto aux=geometryBase.filmAux;std::vector<Real> targets(aux.size());const Real dt=interval.end-interval.begin;
        for(std::size_t f=0;f<aux.size();++f){aux[f].solidNormalVelocity=surface.physics[f].solidSpeed;aux[f].normalVelocity=surface.physics[f].topSpeed;aux[f].solidFront+=dt*aux[f].solidNormalVelocity;aux[f].gasFront+=dt*aux[f].normalVelocity;
            const int sf=base.surface.solidFace[f];targets[f]=sf>=0?dt*aux[f].solidNormalVelocity*mag(base.solidMesh.areaVectors[sf]):0;}
        HostMesh gas,solid;SurfaceMesh moved;SweepConstraintReport report;
        if(!moveCoupledMeshesConstrained(geometryBase,aux,targets,dt,gas,solid,moved,report,error,materialSweepControls(p.numerics.meshMotion),p.tolerances,&numerical)){
            if(diagnostics)diagnostics->recoverable=false;
            return fail(numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None);
        }
        last.gasPoints=gas.points;last.solidPoints=solid.points;}
    program.knots={first,last};output=std::move(program);error.clear();return true;
}
bool CpuMaterialDriver::prepareParentWindow(const HostState& base,const WallProgram& program,
    const ResidentMaterialDriveHistory& history,const IntervalRadiationDeviceView& radiation,
    const CpuMaterialControls& controls,MaterialBackend*& owner,std::string& error){
    owner=nullptr;
    if(data_->parentPrepared){error="material parent window has not been collected";return false;}
    const auto& physics=data_->model.physics;
    if(physics.meshMotion.policy!=MeshMotionPolicy::Static||controls.retainDonorHistory||base.surface.area.empty())return true;
    if(!controlsValid(controls)||(!base.solid.empty()&&!data_->mesh)){
        error="material parent controls or native mesh missing";return false;
    }
    if(!data_->begin(base,physics,error))return false;
    HostState prepared=base;prepared.filmAux=dryGeometryAux(base);
    if(!preparePendingResidentMaterialLocalWindow(*data_->gpu,prepared,physics,history,radiation,program,controls,error))return false;
    data_->parentConduction=false;
    if(!prepared.solidMesh.volumes.empty()){
        if(!data_->conduction)data_->conduction.reset(new ugkp::gpu::SparseConjugateGradient);
        CpuMaterialReport report;
        if(!nativeConduction(*data_->mesh,base,prepared,physics,program.interval.end-program.interval.begin,
            controls,report,error,*data_->conduction,*data_->gpu,data_->conductionWorkspace,&data_->parentConduction))return false;
    }
    if(!prepareResidentMaterialEndpointSurface(*data_->gpu,base,error))return false;
    data_->parentPrepared=true;owner=data_->gpu.get();error.clear();return true;
}
#include "ImplicitMaterialWindow.inl"
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram,const ResidentMaterialDriveHistory* residentHistory,const std::shared_ptr<CoupledHistoryArchive>& historyArchive){
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,base.commitSequence+1,reinterpret_cast<std::uintptr_t>(this),base.acceptedSteps,base.time);
    PhysicsConfig physics=data_->model.physics;physics.diagnostics=&context;physics.tolerances.diagnostics=&context;
    const bool result=advanceCandidateImpl(base,geometryEndpoint,history,controls,output,corrected,report,error,executedProgram,physics,residentHistory,historyArchive);
    auto snapshot=ugkp::diagnostics::takeSnapshot(context,ugkp::diagnostics::sourceCatalogue());
    if(!result&&!snapshot.hasFatal){
        snapshot.hasFatal=true;
        snapshot.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,10053,4);
        snapshot.fatal.step=context.step;snapshot.fatal.simulationTime=context.simulationTime;snapshot.fatal.instanceId=context.instanceId;
        snapshot.fatal.operationSequence=context.operationSequence;
        snapshot.sources.push_back({10053,"applications/CHMT/materials/CpuMaterialDriver.C",__func__,__LINE__,error});
    }
    snapshot.sources.push_back({4103,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material temperature"});
    snapshot.sources.push_back({4104,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material residual"});
    ugkp::diagnostics::dispatch(snapshot,ugkp::diagnostics::makeFoamSink());
    return result;
}
bool CpuMaterialDriver::advanceCandidateImpl(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram,const PhysicsConfig& p,const ResidentMaterialDriveHistory* residentHistory,const std::shared_ptr<CoupledHistoryArchive>& historyArchive){
    report=CpuMaterialReport{};error.clear();
    if(!(validateMaterialSweepRemainder(base,error,p.diagnostics))){report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;report.recoverable=false;return false;}
    const auto& window=history.interval();
    if(!base.solid.empty()&&!data_->mesh){report.recoverable=false;error="CPU material native mesh unavailable";return false;}
    if((!controlsValid(controls)||!history.complete()||window.begin!=base.time||!sameMeshConnectivity(base.solidMesh,geometryEndpoint.solidMesh)||!sameMeshConnectivity(base.gasMesh,geometryEndpoint.gasMesh))){
        report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;
        if(!controlsValid(controls)){
            report.failureOrigin=(!finite(controls.transportCfl)||controls.transportCfl<=0||controls.transportCfl>1)?MaterialFailureOrigin::TransportNumerical:controls.maxSubsteps<=0?MaterialFailureOrigin::LocalSourceNumerical:
                (!finite(controls.reactionCfl)||controls.reactionCfl<=0||controls.reactionCfl>1||!finite(controls.reactionMaxSubstep)||controls.reactionMaxSubstep<0)
                    ?MaterialFailureOrigin::ReactionNumerical:
                    (controls.nonlinearMaxIterations<=0||!finite(controls.nonlinearRelativeTolerance)||controls.nonlinearRelativeTolerance<0
                     ||!finite(controls.energyAbsoluteTolerance)||controls.energyAbsoluteTolerance<0||!finite(controls.energyRelativeTolerance)||controls.energyRelativeTolerance<0)
                        ?MaterialFailureOrigin::ConductionNumerical:MaterialFailureOrigin::OwnerNumerical;
        }
        report.recoverable=false;error="unsupported or inconsistent CPU material interval/configuration/topology";return false;}
    const bool completeResident=residentHistory&&residentHistory->interval&&residentHistory->surface
        &&residentHistory->frames==history.records().size()&&residentHistory->faces==base.surface.area.size();


    const bool parentComputed=data_->parentPrepared;
    try {

        if(!parentComputed){
            if(!data_->begin(base,p,error)){report.failureOrigin=MaterialFailureOrigin::None;return false;}
            if(!beginResidentMaterialHistory(*data_->gpu,residentHistory,history,error))return false;
        }
        SurfacePacketAddressing packetAddressing;
        if(!packetAddressing.build(base,geometryEndpoint,error)){report.recoverable=false;return false;}
        auto stageGeometry=[&](const HostMesh& mesh,const std::vector<Vec3>& points,Real dt,HostMesh& out,
            std::vector<Real>& sweeps,MaterialFailureOrigin& origin){
            bool numerical=false;
            if(makeStageGeometry(mesh,points,dt,out,sweeps,error,p.tolerances,&numerical))return true;
            origin=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
            return false;
        };
        auto trajectorySurface=[&](const HostState& accepted,const HostMesh& gas,const HostMesh& solid,Real dt,
            SurfaceMesh& out,MaterialFailureOrigin& origin){
            bool numerical=false;
            if(rebuildTrajectorySurface(accepted,gas,solid,dt,out,error,p.tolerances,&numerical))return true;
            origin=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
            return false;
        };
        const bool localDeviceWindow=p.meshMotion.policy==MeshMotionPolicy::Static&&!controls.retainDonorHistory
            &&executedProgram&&(completeResident||historyArchive)&&!base.surface.area.empty();
        MaterialGeometryInventory geometryInventory(base,history,!localDeviceWindow);
        HostState state=base;state.filmAux=dryGeometryAux(base);state.ledger.clear();state.filmStorage.clear();



        state.solid.clear();
        const bool staticGeometry=p.meshMotion.policy==MeshMotionPolicy::Static;
        HostState trial;
        MaterialStepReuse reuse;
        if(staticGeometry){trial.gasMesh=state.gasMesh;trial.solidMesh=state.solidMesh;
            trial.gasStages=state.gasStages;trial.solidStages=state.solidStages;}
        WallProgram program;program.interval=window;program.surface=base.surface;
        const bool streamWallProgram=!controls.retainWallProgram&&executedProgram&&!localDeviceWindow;
        WallKnot initialProgramKnot;
        CpuMaterialReport accumulated;accumulated.donorPlan.interval=window;
        if(controls.retainDonorHistory){
            MaterialReserveKnot initial;initial.time=base.time;
            initial.cumulativeSolid.resize(base.solid.size());initial.cumulativeFilm.resize(base.film.size());
            accumulated.donorPlan.knots.push_back(std::move(initial));
        }
        std::vector<IntervalDonorSum> physicalSweepSum(base.surface.area.size()),actualSweepSum(base.surface.area.size());

        auto cursor=history.cursor();const Real interval=window.end-window.begin;
        MaterialDriveCursor driveCursor(history),endpointCursor(history);
        MaterialSourceAudit sourceAudit(history,controls.driveSourceDiagnosticSamples);
        accumulated.sourceAuditPopulation=sourceAudit.population();
        WindowTime elapsed{};const WindowTime windowEnd=intervalDuration(window);



        bool conductionInParent=false,endpointInParent=false;CpuSurfaceResult deviceEndpoint;
        if(localDeviceWindow){
            HostState prepared=base;prepared.filmAux=state.filmAux;
            MaterialLocalReport local;
            if(parentComputed){conductionInParent=data_->parentConduction;}
            else{
            const bool preparedLocal=completeResident
                ?prepareResidentMaterialLocalWindow(*data_->gpu,prepared,p,history,*residentHistory,*executedProgram,controls,error)
                :prepareStreamedMaterialLocalWindow(*data_->gpu,prepared,p,history,historyArchive,*executedProgram,controls,error);
            if(!preparedLocal){report=accumulated;report.failureOrigin=MaterialFailureOrigin::LocalSourceNumerical;return false;}
            if(!prepared.solidMesh.volumes.empty()){
                if(!data_->conduction)data_->conduction.reset(new ugkp::gpu::SparseConjugateGradient);
                if(!nativeConduction(*data_->mesh,base,prepared,p,interval,controls,accumulated,error,
                    *data_->conduction,*data_->gpu,data_->conductionWorkspace,&conductionInParent)){
                    report=accumulated;return false;
                }
            }
            if(!prepareResidentMaterialEndpointSurface(*data_->gpu,geometryEndpoint,error))return false;
            if(!enqueueResidentMaterialLocalWindow(*data_->gpu,error)){
                report=accumulated;report.failureOrigin=MaterialFailureOrigin::LocalSourceNumerical;return false;
            }
            }
            if(!collectResidentMaterialLocalWindow(*data_->gpu,local,state.filmAux,error)){
                report=accumulated;report.failureOrigin=MaterialFailureOrigin::LocalSourceNumerical;return false;
            }
            data_->parentPrepared=false;
            if(!collectResidentMaterialEndpointSurface(*data_->gpu,deviceEndpoint,endpointInParent,error))return false;
            if(conductionInParent&&!collectResidentMaterialInventory(*data_->gpu,state.solid,error))return false;
            accumulated.materialSteps=local.acceptedSteps;accumulated.deviceCommit=local.commit;
            std::vector<MaterialLocalAudit> observations;
            if(!collectResidentMaterialLocalAudit(*data_->gpu,observations,error))return false;


            if(!data_->appendAuditKnots(state,p,history,observations,program,error)){
                report=accumulated;report.failureOrigin=MaterialFailureOrigin::SurfaceNumerical;return false;
            }
            accumulated.sourceAuditSamples=observations.empty()?0:observations.size()-1;
            if(!program.knots.empty()){
                WallKnot endpoint=program.knots.back();endpoint.time=window.end;endpoint.localTime=windowEnd;
                program.knots.push_back(std::move(endpoint));
            }
            state.time=window.end;elapsed=windowEnd;
            Foam::OStringStream progress;progress.precision(17);
            progress<<"CHMT_MATERIAL window="<<window.identity.sequence<<" step="<<local.acceptedSteps
                <<" time="<<state.time<<" execution=device_local_window";
            Foam::Info<<progress.str()<<Foam::nl;
        } else while(elapsed<windowEnd){
            if(p.diagnostics)ugkp::diagnostics::beginOperation(*p.diagnostics,window.identity.sequence,base.acceptedSteps,state.time);
            if(accumulated.materialSteps==static_cast<std::uint64_t>(controls.maxSubsteps))materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"CPU material substep warning budget exceeded; advancing calculation retained",accumulated.materialSteps,controls.maxSubsteps);
            if(accumulated.materialSteps==std::numeric_limits<std::uint64_t>::max()){error="CPU material substep counter overflow";return false;}
            Real dt=std::min(boundedWindowDuration(windowEnd,elapsed),controls.maxSubstep>0?controls.maxSubstep:interval);


            const GasIntervalRecord* driveRecord=driveCursor.at(elapsed);
            if(!driveRecord){error="CPU material driving history exhausted";return false;}
            WindowTime boundedEnd=elapsed+dt;
            if(!driveCursor.slabEndLocal(elapsed,boundedEnd,controls.driveRelativeTolerance,controls.driveTractionScale,boundedEnd,error)){
                {report=accumulated;report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;}

            }
            dt=boundedWindowDuration(boundedEnd,elapsed);
            if(executedProgram)for(const auto& knot:executedProgram->knots)if(knotOffset(window,knot)>elapsed&&knotOffset(window,knot)<elapsed+dt)dt=boundedWindowDuration(knotOffset(window,knot),elapsed);
            if((driveRecord->gasTrace.size()!=state.surface.area.size()||driveRecord->gasTraction.size()!=state.surface.area.size())&&!state.surface.area.empty()){
                report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;report.recoverable=false;error="CPU material requires actual face-ordered gas drive history";return false;}
            const std::vector<GasPrimitive> emptyBulk;
            const auto& bulk=state.surface.area.empty()?emptyBulk:driveRecord->gasTrace;

            WallKnot beginning;CpuSurfaceResult surface;MaterialTransportResult transport;
            std::vector<ExchangePacket> packets;std::vector<Real> radiation;IntervalSliceMetadata selectedSlice;
            std::vector<SolidQ> cumulativeDonor;
            std::vector<Real> sideVolume;
            std::vector<Real> solidSweep,gasSweep;IntervalCursor trialCursor=cursor;
            bool accepted=false;WindowTime trialElapsed;
            MaterialFailureOrigin lastFailure=MaterialFailureOrigin::None;


            bool transportNumerical=false;
            if(!evaluateResidentMaterialTransport(*data_->gpu,state.solidMesh,{},dt,controls.transportCfl,transport.dtLimit,error,transportNumerical)) {
                report=accumulated;report.failureOrigin=transportNumerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;return false;
            }
            dt=std::min(dt,transport.dtLimit);
            do {
                if(!finite(dt)||!(dt>0)||elapsed+dt==elapsed){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU material substep does not advance";break;}
                error.clear();
                if(staticGeometry)static_cast<HostDynamicState&>(trial)=state;
                else trial=state;
                trialElapsed=elapsed+dt;trial.time=windowAbsolute(window.begin,trialElapsed);trialCursor=cursor;
                if(!trialCursor.takeThroughLocal(trialElapsed,packets,radiation,error,residentHistory?&selectedSlice:nullptr))break;

                std::vector<GasPrimitive> terminalBulk;
                const std::vector<GasPrimitive>* endpointBulk=&terminalBulk;
                const GasIntervalRecord* endpointRecord=nullptr;
                if(trialElapsed==windowEnd){bool numerical=false;
                    if(!(baseTrace(geometryEndpoint,p,terminalBulk,error,&numerical))){lastFailure=numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;break;}}
                else{
                    endpointRecord=endpointCursor.at(trialElapsed);
                    if(!endpointRecord){error="gas driving history has a gap";break;}
                    endpointBulk=&endpointRecord->gasTrace;
                }
                if(radiation.size()!=state.surface.area.size()&&!radiation.empty()){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU radiation face layout mismatch";break;}

                auto prepareImposedGeometry=[&]()->bool{
                    if(base.gasMesh.points.size()!=geometryEndpoint.gasMesh.points.size()
                        ||base.solidMesh.points.size()!=geometryEndpoint.solidMesh.points.size()){
                        lastFailure=MaterialFailureOrigin::OwnerNumerical;
                        error="material trajectory topology mismatch";return false;
                    }
                    std::vector<Vec3> gp,sp;
                    if(executedProgram){
                        WallKnot executed;
                        if(!intervalSampleWallLocal(*executedProgram,trialElapsed,executed,error))return false;
                        gp=std::move(executed.gasPoints);
                        sp=std::move(executed.solidPoints);
                    }else{
                        const Real fraction=windowFraction(trialElapsed,WindowTime{},windowEnd,trial.time,window.begin,window.end);
                        gp=base.gasMesh.points;
                        sp=base.solidMesh.points;
                        for(std::size_t i=0;i<gp.size();++i)gp[i]+=(geometryEndpoint.gasMesh.points[i]-gp[i])*fraction;
                        for(std::size_t i=0;i<sp.size();++i)sp[i]+=(geometryEndpoint.solidMesh.points[i]-sp[i])*fraction;
                    }
                    if(reuse.useStationaryGeometry(state,gp,sp,gasSweep,solidSweep))return true;
                    if(!stageGeometry(state.gasMesh,gp,dt,trial.gasMesh,gasSweep,lastFailure)||!stageGeometry(state.solidMesh,sp,dt,trial.solidMesh,solidSweep,lastFailure))return false;
                    if(!state.surface.area.empty()&&!trajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,lastFailure))return false;
                    return true;
                };
                if(staticGeometry){
                    solidSweep.assign(state.solidMesh.owner.size(),0);
                    gasSweep.assign(state.gasMesh.owner.size(),0);
                }else if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession){
                    if(!prepareImposedGeometry())break;
                }
                bool stageSweepsReady=p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession;



                const HostState& evaluation=state;
                std::vector<FilmAux> aux;std::vector<Real> target;
                sideVolume.assign(state.surface.area.size(),0);
                if(!reuse.takeSurface(elapsed,dt,sideVolume,surface)
                    &&!data_->evaluate(evaluation,p,bulk,driveRecord->gasGradient,true,false,dt,window.identity.sequence,surface,error,&sideVolume,&driveRecord->gasWallMatching,&driveRecord->gasAcceptedWall.values())){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}


                target.assign(state.surface.area.size(),0);
                for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasSolid){
                    std::size_t match=0;
                    if(!packetAddressing.gasSolidIndex(state,packet,match,error))break;
                    const int donor=packet.solidCell;
                    const auto recession=addMaterialRecessionTarget(packet.condensed,geometryInventory.at(donor),state.solidMesh.volumes[donor],p,target[match]);
                    if(recession!=MaterialRecessionError::None){
                        lastFailure=MaterialFailureOrigin::GeometryNumerical;
                        error=recession==MaterialRecessionError::DonorEos?"surface donor EOS failed":"invalid surface recession constituent";break;
                    }
                }
                if(!error.empty())break;
                aux=evaluation.filmAux;
                for(std::size_t f=0;f<aux.size();++f)if(state.surface.gasFace[f]>=0){const int sf=state.surface.solidFace[f];const Real as=sf>=0?mag(state.solidMesh.areaVectors[sf]):0;
                    aux[f]=materialRecessionFront(state.filmAux[f],target[f],as,dt);}
                if(!error.empty())break;




                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!target.empty()
                    &&(!executedProgram||trialElapsed==windowEnd)){SweepConstraintReport motion;
                    bool numerical=false;
                    if(!moveCoupledMeshesConstrained(state,aux,target,geometryInventory.row.size(),dt,trial.gasMesh,trial.solidMesh,trial.surface,motion,error,materialSweepControls(p.numerics.meshMotion),p.tolerances,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
                        {
                            if(!numerical)lastFailure=MaterialFailureOrigin::None;
                            if(motion.status==SweepConstraintStatus::Incompatible||motion.status==SweepConstraintStatus::SizeLimit||motion.status==SweepConstraintStatus::InvalidInput){report.failureOrigin=lastFailure;report.recoverable=false;return false;}break;
                        }
                        lastFailure=MaterialFailureOrigin::None;
                    }
                }else{
                    if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){
                        if(!prepareImposedGeometry())break;
                        stageSweepsReady=true;
                    }
                }
                if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!stageSweepsReady){WallKnot executed;
                    if(!intervalSampleWallLocal(*executedProgram,trialElapsed,executed,error))break;
                    const bool close=materialPointsClose(trial.gasMesh,executed.gasPoints,p.tolerances)&&materialPointsClose(trial.solidMesh,executed.solidPoints,p.tolerances);
                    if(close){if(!stageGeometry(state.gasMesh,executed.gasPoints,dt,trial.gasMesh,gasSweep,lastFailure)||!stageGeometry(state.solidMesh,executed.solidPoints,dt,trial.solidMesh,solidSweep,lastFailure)||!trajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,lastFailure))break;
                        stageSweepsReady=true;
                        std::vector<Real> solveTarget;if(!(compensateMaterialSweepTargets(state,target,solveTarget,error,p.diagnostics))){lastFailure=MaterialFailureOrigin::GeometryNumerical;break;}
                        const bool compatible=materialSweepsCompatible(state.solidMesh,state.surface,solveTarget,solidSweep,p.tolerances);
                        if(!compatible)materialWarning(state,ugkp::diagnostics::Code::ConservationResidual,"executed material sweep discrepancy",1);
                    }
                }
                if(!stageSweepsReady&&(!stageGeometry(state.solidMesh,trial.solidMesh.points,dt,trial.solidMesh,solidSweep,lastFailure)
                    ||!stageGeometry(state.gasMesh,trial.gasMesh.points,dt,trial.gasMesh,gasSweep,lastFailure)))break;

                std::vector<std::size_t> variationRecords;
                sourceAudit.samplesInside(elapsed,trialElapsed,variationRecords);


                const std::size_t recordsPerBatch=data_->historyBatchSize(state.surface.area.size());
                for(std::size_t begin=0;begin<variationRecords.size();begin+=recordsPerBatch){
                    const std::size_t end=std::min(variationRecords.size(),begin+recordsPerBatch);
                    std::vector<GasIntervalRecord> payloads;payloads.reserve(end-begin);
                    for(std::size_t i=begin;i<end;++i)payloads.push_back(history.readRecord(variationRecords[i]));
                    std::vector<CpuSurfaceResult> responses;
                    if(!data_->evaluateHistory(evaluation,p,payloads,dt,&sideVolume,responses,error)){
                        lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;
                        if(data_->terminalWallFailure){report.recoverable=false;return false;}break;
                    }
                    for(const auto& response:responses){
                        ++accumulated.sourceAuditSamples;
                        Real fraction=0;
                        for(std::size_t f=0;f<surface.physics.size();++f){if(state.surface.gasFace[f]<0)continue;
                            const auto& a=surface.physics[f];const auto& b=response.physics[f];const Real scale=state.surface.area[f]*dt;
                            const int cell=state.surface.solidCell[f];Real solidDonor=0;
                            if(cell>=0)for(int c=0;c<Nc;++c)solidDonor+=geometryInventory.at(cell).condensed[c];
                            for(int species=0;species<Ns;++species){const Real donor=solidDonor;
                                fraction=maxValue(fraction,scale*absValue(a.gasSpecies[species]-b.gasSpecies[species])/maxValue(p.tolerances.absoluteMass,donor));
                                if(cell>=0)fraction=maxValue(fraction,scale*absValue(a.poreSpecies[species]-b.poreSpecies[species])/maxValue(p.tolerances.absoluteMass,geometryInventory.at(cell).pore[species]));}
                        }
                        if(fraction>controls.driveSourceFraction)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"material source variation",fraction,controls.driveSourceFraction);
                    }
                }
                if(!error.empty())break;



                if(std::any_of(solidSweep.begin(),solidSweep.end(),[](Real swept){return swept!=0;})){
                bool numerical=false;if(!evaluateResidentMaterialTransport(*data_->gpu,state.solidMesh,solidSweep,dt,controls.transportCfl,transport.dtLimit,error,numerical)){
                    lastFailure=numerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;
                    break;
                }}
                if(dt>transport.dtLimit)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"ALE material transport step bound",dt,transport.dtLimit);
                bool sourceNumerical=false;std::vector<SolidQ> nextGeometryInventory;
                if(!advanceResidentMaterialSources(*data_->gpu,trial.solidMesh,state.surface,packets,
                    radiation,driveRecord->radiationReceiver,dt,controls,trialElapsed==windowEnd,
                    nextGeometryInventory,cumulativeDonor,error,sourceNumerical,&geometryInventory.cells,
                    residentHistory?&selectedSlice:nullptr)){
                    lastFailure=sourceNumerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;
                    break;
                }
                trial.filmAux=aux;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!(updateMaterialSweepRemainder(state,target,solidSweep,trial.solidMesh,trial.solidSweepRemainder,error,p.diagnostics))){lastFailure=MaterialFailureOrigin::GeometryNumerical;break;}
                for(std::size_t f=0;f<target.size();++f){
                    physicalSweepSum[f].add(target[f]);
                    const int face=state.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(solidSweep[face]);
                }
                beginning.time=state.time;beginning.localTime=elapsed;beginning.faces=surface.wall;beginning.gasPoints=state.gasMesh.points;beginning.solidPoints=state.solidMesh.points;
                if(program.knots.empty()){
                    program.knots.push_back(beginning);
                    if(streamWallProgram)initialProgramKnot=beginning;
                }
                WallKnot ending=beginning;ending.time=trial.time;ending.localTime=trialElapsed;ending.gasPoints=trial.gasMesh.points;ending.solidPoints=trial.solidMesh.points;
                if(trialElapsed<windowEnd){CpuSurfaceResult endpointProposal;
                    if(!endpointRecord){error="missing endpoint wall matching record";break;}if(!data_->evaluate(trial,p,*endpointBulk,endpointRecord->gasGradient,false,true,dt,window.identity.sequence,endpointProposal,error,nullptr,&endpointRecord->gasWallMatching,&endpointRecord->gasAcceptedWall.values())){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                    ending.faces=endpointProposal.wall;
                    reuse.rememberSurface(trialElapsed,dt,*endpointRecord,std::move(endpointProposal));
                    for(std::size_t f=0;f<ending.faces.size();++f)ending.faces[f].primaryKind=beginning.faces[f].primaryKind;}
                for(std::size_t f=0;f<ending.faces.size();++f){ending.faces[f].solidNormalVelocity=aux[f].solidNormalVelocity;ending.faces[f].normalVelocity=aux[f].normalVelocity;}
                program.knots.push_back(std::move(ending));
                if(streamWallProgram&&program.knots.size()==3){
                    if(!accumulateWallProgramSegment(*executedProgram,program,p.tolerances,accumulated.wallComparison,error)){
                        report=accumulated;report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;
                    }
                    program.knots.erase(program.knots.begin());
                }
                if(controls.retainDonorHistory){
                MaterialReserveKnot knot;knot.time=trial.time;knot.localTime=trialElapsed;
                knot.cumulativeSolid=std::move(cumulativeDonor);knot.cumulativeFilm.resize(trial.film.size());
                accumulated.donorPlan.knots.push_back(std::move(knot));
                }
                accumulated.materialSteps++;
                if(!commitResidentMaterialStep(*data_->gpu,trial.time,error))return false;
                geometryInventory.values=std::move(nextGeometryInventory);
                if(staticGeometry)std::swap(static_cast<HostDynamicState&>(state),static_cast<HostDynamicState&>(trial));
                else state=std::move(trial);
                elapsed=trialElapsed;cursor=trialCursor;accepted=true;
                if(accumulated.materialSteps==1||accumulated.materialSteps%4096==0||elapsed==windowEnd){
                    Foam::OStringStream progress;progress.precision(17);
                    progress<<"CHMT_MATERIAL window="<<window.identity.sequence
                        <<" step="<<accumulated.materialSteps<<" time="<<state.time<<" dt="<<dt;
                    Foam::Info<<progress.str()<<Foam::nl;
                }

            } while(false);
            if(!accepted){if(error.empty())error="CPU material single-pass operation failed";report=accumulated;report.failureOrigin=lastFailure;return false;}
        }



        if(!conductionInParent&&!state.solidMesh.volumes.empty()){
            if(p.diagnostics)ugkp::diagnostics::beginOperation(*p.diagnostics,window.identity.sequence,base.acceptedSteps,state.time);
            if(!data_->conduction)data_->conduction.reset(new ugkp::gpu::SparseConjugateGradient);
            const bool solved=nativeConduction(*data_->mesh,base,state,p,interval,controls,accumulated,error,*data_->conduction,*data_->gpu,data_->conductionWorkspace);
            if(!solved){report=accumulated;return false;}
        }
        const auto samePoints=[](const HostMesh& a,const HostMesh& b){if(a.points.size()!=b.points.size())return false;for(std::size_t i=0;i<a.points.size();++i)if(a.points[i].x!=b.points[i].x||a.points[i].y!=b.points[i].y||a.points[i].z!=b.points[i].z)return false;return true;};
        if(samePoints(state.gasMesh,geometryEndpoint.gasMesh)&&samePoints(state.solidMesh,geometryEndpoint.solidMesh)){
            state.gasMesh=geometryEndpoint.gasMesh;state.solidMesh=geometryEndpoint.solidMesh;state.gasStages=geometryEndpoint.gasStages;state.solidStages=geometryEndpoint.solidStages;
            if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){



                if((executedProgram->knots.size()<2||executedProgram->knots.front().time!=window.begin||executedProgram->knots.back().time!=window.end)){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder trajectory interval mismatch";return false;}
                {
                    const auto& firstPoints=executedProgram->knots.front().solidPoints;
                    if(firstPoints.size()!=base.solidMesh.points.size()){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder base topology mismatch";return false;}
                    for(std::size_t i=0;i<firstPoints.size();++i)if(firstPoints[i].x!=base.solidMesh.points[i].x||firstPoints[i].y!=base.solidMesh.points[i].y||firstPoints[i].z!=base.solidMesh.points[i].z){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder base geometry mismatch";return false;}
                }
                HostMesh path=base.solidMesh;actualSweepSum.assign(base.surface.area.size(),IntervalDonorSum{});
                for(std::size_t k=1;k<executedProgram->knots.size();++k){HostMesh endpoint;std::vector<Real> measured;
                    const Real duration=windowDuration(knotOffset(window,executedProgram->knots[k]),knotOffset(window,executedProgram->knots[k-1]));
                    if(!stageGeometry(path,executedProgram->knots[k].solidPoints,duration,endpoint,measured,report.failureOrigin))return false;
                    for(std::size_t f=0;f<actualSweepSum.size();++f){const int face=base.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(measured[face]);}
                    path=std::move(endpoint);
                }
                if(!samePoints(path,state.solidMesh)){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder endpoint mismatch";return false;}
            }
        }
        if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
            state.solidSweepRemainder.resize(base.surface.solidFace.size());
            for(std::size_t f=0;f<state.solidSweepRemainder.size();++f)state.solidSweepRemainder[f]=static_cast<Real>(
                (base.solidSweepRemainder.empty()?0:static_cast<long double>(base.solidSweepRemainder[f]))
                +physicalSweepSum[f].value()-actualSweepSum[f].value());
            if(!(validateMaterialSweepRemainder(state,error,p.diagnostics))){report=accumulated;report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;return false;}
        }
        state.gasWallDiagnostics=geometryEndpoint.gasWallDiagnostics;state.gasWallDiagnosticTime=geometryEndpoint.gasWallDiagnosticTime;
        state.particleSourceFaces=geometryEndpoint.particleSourceFaces;
        state.particleSourceResidualMass=geometryEndpoint.particleSourceResidualMass;
        state.gas=geometryEndpoint.gas;state.sst=geometryEndpoint.sst;state.particles=geometryEndpoint.particles;state.gasTemperature=geometryEndpoint.gasTemperature;
        state.rejectedSteps=geometryEndpoint.rejectedSteps;state.gasVoidFraction=geometryEndpoint.gasVoidFraction;state.gasStages=geometryEndpoint.gasStages;
        state.gasMesh.boundaryPrimitive=geometryEndpoint.gasMesh.boundaryPrimitive;state.gasMesh.thermalBoundary=geometryEndpoint.gasMesh.thermalBoundary;state.gasMesh.boundarySst=geometryEndpoint.gasMesh.boundarySst;
        state.time=window.end;program.donorPlan=accumulated.donorPlan;
            if(!program.knots.empty()){const auto last=history.readRecord(history.records().size()-1);CpuSurfaceResult endpointSurface;


            if(endpointInParent)endpointSurface=std::move(deviceEndpoint);
            else if(!data_->evaluate(state,p,last.gasTrace,last.gasGradient,false,false,interval,window.identity.sequence,endpointSurface,error,nullptr,&last.gasWallMatching,&last.gasAcceptedWall.values())){report=accumulated;report.failureOrigin=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;report.recoverable=false;return false;}
            program.knots.back().faces=endpointSurface.wall;


            for(std::size_t f=0;f<program.knots.back().faces.size();++f)program.knots.back().faces[f].primaryKind=(streamWallProgram?initialProgramKnot:program.knots.front()).faces[f].primaryKind;
        }
        if(streamWallProgram){


            if(!accumulateWallProgramSegment(*executedProgram,program,p.tolerances,accumulated.wallComparison,error)){
                report=accumulated;report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;
            }
            accumulated.streamedWallComparison=true;
            program.knots.front()=std::move(initialProgramKnot);
        }
        for(std::size_t i=0;i<state.gasMesh.points.size();++i)if(mag(state.gasMesh.points[i]-geometryEndpoint.gasMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        for(std::size_t i=0;i<state.solidMesh.points.size();++i)if(mag(state.solidMesh.points[i]-geometryEndpoint.solidMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        Foam::OStringStream auditReport;
        auditReport<<"CHMT_SOURCE_AUDIT sequence="<<window.identity.sequence<<" materialSteps="<<accumulated.materialSteps
            <<" samples="<<accumulated.sourceAuditSamples<<" population="<<accumulated.sourceAuditPopulation
            <<" budget="<<controls.driveSourceDiagnosticSamples<<" coverage="
            <<(localDeviceWindow?"sampled_waveform_snapshot":"sampled_source_variation");
        Foam::Info<<auditReport.str()<<Foam::nl;
        output=std::move(state);corrected=std::move(program);report=std::move(accumulated);error.clear();return true;
    }catch(const Foam::error& e){report.failureOrigin=MaterialFailureOrigin::None;error="OpenFOAM CPU material failure: "+std::string(e.what());return false;}
    catch(const std::exception& e){report.failureOrigin=MaterialFailureOrigin::None;error="CPU material failure: "+std::string(e.what());return false;}
}
}
