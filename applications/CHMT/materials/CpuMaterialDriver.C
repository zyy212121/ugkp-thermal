#include "materials/CpuMaterialDriver.H"
#include "core/ThermalBoundary.H"
#include "core/AuditArithmetic.H"
#include "core/HostAuditReduction.H"
#include "materials/MaterialTransport.H"
#include "materials/MaterialDrive.H"
#include "materials/SurfacePacketAddressing.H"
#include "ablation/CpuSurfaceInterface.H"
#include "ablation/WallClosureHost.H"
#include "film/CpuFilmDriver.H"
#include "mesh/Geometry.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "coupling/IntervalAudit.H"
#include "fvCFD.H"
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
    &&finite(c.driveRelativeTolerance)&&c.driveRelativeTolerance>=0&&finite(c.driveSourceFraction)&&c.driveSourceFraction>0
    &&finite(c.driveTractionScale)&&c.driveTractionScale>0&&(c.maxSubsteps>0)&&((c.nonlinearMaxIterations>0&&finite(c.nonlinearRelativeTolerance)&&c.nonlinearRelativeTolerance>=0
    &&finite(c.energyAbsoluteTolerance)&&c.energyAbsoluteTolerance>=0&&finite(c.energyRelativeTolerance)&&c.energyRelativeTolerance>=0)));}
bool nativeConduction(Foam::fvMesh& acceptedMesh,const HostState& old,HostState& next,const PhysicsConfig& p,
    Real dt,const CpuMaterialControls& controls,CpuMaterialReport& report,std::string& error){
    const std::size_t n=next.solid.size();if(n==0)return true;
    if(!validateThermalBoundaries(next.solidMesh,error)){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
    if((acceptedMesh.nCells()!=static_cast<Foam::label>(n)||next.solidMesh.points.size()!=static_cast<std::size_t>(acceptedMesh.nPoints()))){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;error="CPU material fvMesh topology mismatch";return false;}
    // A fresh unregistered full native region avoids movePoints side effects
    // (old volumes, meshPhi, moving flag and Time callbacks) on accepted state.
    Foam::fvMesh mesh(Foam::IOobject(acceptedMesh.name(),acceptedMesh.time().timeName(),acceptedMesh.time(),
        Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),foamPoints(next.solidMesh.points),
        Foam::faceList(acceptedMesh.faces()),Foam::labelList(acceptedMesh.faceOwner()),Foam::labelList(acceptedMesh.faceNeighbour()));
    Foam::List<Foam::polyPatch*> patches(acceptedMesh.boundaryMesh().size());
    forAll(patches,patch)patches[patch]=acceptedMesh.boundaryMesh()[patch].clone(mesh.boundaryMesh()).ptr();
    mesh.addFvPatches(patches);
    for(std::size_t c=0;c<n;++c)if(!closeEnough(mesh.V()[c],next.solidMesh.volumes[c],p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)){
        materialWarning(next,ugkp::diagnostics::Code::ConservationResidual,"OpenFOAM / conservative material volume discrepancy",mesh.V()[c]-next.solidMesh.volumes[c],p.tolerances.absoluteGeometry+p.tolerances.relativeGeometry*absValue(next.solidMesh.volumes[c]));}
    Foam::wordList patchTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch){if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")patchTypes[patch]=mesh.boundary()[patch].type();
        else{bool hasFixed=false,hasFree=false;forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;

            const bool fixed=next.solidMesh.boundaryKind[face]!=BoundaryKind::Interface&&fixedTemperature(next.solidMesh.thermalBoundary.empty()?nullptr:next.solidMesh.thermalBoundary.data(),face);
            hasFixed=hasFixed||fixed;hasFree=hasFree||!fixed;}if(hasFixed&&hasFree){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;error="mixed per-face material thermal policy within one native patch";return false;}if(hasFixed)patchTypes[patch]=Foam::fixedValueFvPatchScalarField::typeName;}}
    Foam::volScalarField temperature(Foam::IOobject("CHMTMaterialTemperature",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,
        Foam::dimensionedScalar("initial",Foam::dimTemperature,300),patchTypes);
    std::vector<MaterialCaloric> caloric(n);std::vector<Real> anchor(n),target(n);
    for(std::size_t c=0;c<n;++c){bool numerical=false;if(!materialCaloric<false>(next.solid[c],p,caloric[c],error,&numerical)){if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}MaterialPrimitive primitive;
        if(!recoverMaterial<false>(old.solid[c],old.solidMesh.volumes[c],p,primitive,nullptr,-1,&numerical)){error="CPU conduction old EOS recovery failed";if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
        anchor[c]=primitive.temperature;temperature[c]=anchor[c];
        target[c]=next.solid[c].energy;}
    forAll(mesh.boundary(),patch)if(patchTypes[patch]==Foam::fixedValueFvPatchScalarField::typeName)forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;temperature.boundaryFieldRef()[patch][f]=next.solidMesh.boundaryPrimitive[face].temperature;}
    Foam::wordList coefficientTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")coefficientTypes[patch]=mesh.boundary()[patch].type();
    const Foam::dimensionSet capacityDimensions=Foam::dimEnergy/Foam::dimVolume/Foam::dimTemperature;
    const Foam::dimensionSet conductivityDimensions=Foam::dimPower/Foam::dimLength/Foam::dimTemperature;
    const Foam::dimensionedScalar deltaT("materialDt",Foam::dimTime,dt);
    Foam::dictionary solver;solver.add("solver",Foam::word("PCG"));solver.add("preconditioner",Foam::word("DIC"));solver.add("tolerance",Foam::scalar(1e-12));solver.add("relTol",Foam::scalar(0));solver.add("maxIter",Foam::label(2000));
    mesh.schemes().setFluxRequired(temperature.name());
    for(int iteration=0;iteration<(controls.nonlinearMaxIterations);++iteration){Foam::scalarField previous(temperature.primitiveField());
        Foam::volScalarField capacity(Foam::IOobject("CHMTMaterialSecant",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",capacityDimensions,0),coefficientTypes);
        Foam::volScalarField conductivity(Foam::IOobject("CHMTMaterialKappa",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",conductivityDimensions,0),coefficientTypes);
        Foam::volScalarField source(Foam::IOobject("CHMTMaterialEnergySource",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",Foam::dimPower/Foam::dimVolume,0));
        for(std::size_t c=0;c<n;++c){const Real cs=caloric[c].secant<false>(temperature[c],anchor[c]);if((!(cs>0)||!finite(cs))){error="nonpositive implicit caloric secant";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            capacity[c]=cs/mesh.V()[c];source[c]=(target[c]-caloric[c].energy<false>(anchor[c])+cs*anchor[c])/(mesh.V()[c]*dt);
            // Inventories and material EOS belong to the conservative HostMesh
            // volume, already checked against the independently rounded native
            // metric above. Keep native V only in the matrix integration weights
            // so the cell-integral energy equation is unchanged.
            const Real materialVolume=next.solidMesh.volumes[c];
            Real occupied=0,k=0,poreMass=0;for(int j=0;j<Nc;++j){if(next.solid[c].condensed[j]==0)continue;const Real v=next.solid[c].condensed[j]/p.condensed[j].rho;occupied+=v;k+=v*p.condensed[j].conductivity/materialVolume;}
            for(int s=0;s<Ns;++s)poreMass+=next.solid[c].pore[s];Real phi=1-occupied/materialVolume;
            if((!clipPhysicsMassFraction(p,phi,3824,static_cast<int>(c)))){error="material mass/geometry incompatible during implicit solve";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            next.solid[c].porosity=phi;conductivity[c]=k+(poreMass>0?phi*p.gasConductivity:0);}
        capacity.correctBoundaryConditions();conductivity.correctBoundaryConditions();temperature.correctBoundaryConditions();
        Foam::surfaceScalarField faceK(Foam::fvc::interpolate(conductivity));
        forAll(mesh.neighbour(),f){const Real a=conductivity[mesh.owner()[f]],b=conductivity[mesh.neighbour()[f]],w=mesh.weights()[f];faceK[f]=a>0&&b>0?1/((1-w)/a+w/b):0;}
        forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()){const Foam::scalarField neighbourK(conductivity.boundaryField()[patch].patchNeighbourField());const Foam::scalarField ownK(conductivity.boundaryField()[patch].patchInternalField());
            forAll(neighbourK,f){const Real a=ownK[f],b=neighbourK[f],w=mesh.weights().boundaryField()[patch][f];faceK.boundaryFieldRef()[patch][f]=a>0&&b>0?1/((1-w)/a+w/b):0;}}
        Foam::fvScalarMatrix equation(Foam::fvm::Sp(capacity/deltaT,temperature)-Foam::fvm::laplacian(faceK,temperature)==source);
        // The FV residual is an integral power, whereas PCG normalises by
        // a matrix-dependent normFactor. Use a relative residual ratio: the
        // same normFactor cancels, so this is an energy budget in physical
        // units without copying OpenFOAM's boundary/parallel normalisation.
        const Foam::tmp<Foam::scalarField> powerResidual=equation.residual();
        Real powerScale=0;
        for(const Real value:powerResidual())powerScale=maxValue(powerScale,absValue(value));
        Foam::reduce(powerScale,Foam::maxOp<Real>(),Foam::Pstream::msgType(),mesh.comm());
        Real scaledPower=0;
        if(powerScale>0)for(const Real value:powerResidual())scaledPower+=absValue(value)/powerScale;
        Foam::reduce(scaledPower,Foam::sumOp<Real>(),Foam::Pstream::msgType(),mesh.comm());
        long double inventoryRoundoff=0;
        const Real epsilonScale=64*std::numeric_limits<Real>::epsilon();
        for(std::size_t c=0;c<n;++c)inventoryRoundoff=hostAuditAdd(inventoryRoundoff,
            static_cast<long double>(epsilonScale)*absValue(target[c])
            +static_cast<long double>(epsilonScale)*absValue(caloric[c].energy<false>(temperature[c])));
        inventoryRoundoff=hostAuditGlobalSum(inventoryRoundoff,mesh.comm());
        // Keep the audit product wider than Real: finite cell residuals may
        // have an L1 sum, or dt product, that cannot be represented in Real.
        const long double initialEnergy=static_cast<long double>(dt)*powerScale*scaledPower;
        const long double linearEnergyTolerance=controls.energyAbsoluteTolerance
            +controls.energyRelativeTolerance*std::max(1.L,initialEnergy)+inventoryRoundoff;
        // Split the existing error budget between the linear equation and
        // nonlinear caloric closure. This is not a new user/case tolerance.
        const Real relativeLinearTarget=initialEnergy>0
            ?static_cast<Real>(std::min(1.L,(linearEnergyTolerance/initialEnergy)/2)):1;
        solver.set("tolerance",std::numeric_limits<Foam::scalar>::min());
        solver.set("relTol",minValue(1,relativeLinearTarget));
        const Foam::solverPerformance linear=equation.solve(solver);
        if(!finite(linear.finalResidual())){error="nonfinite OpenFOAM implicit material sparse residual";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
        if(!linear.converged()||linear.singular())materialWarning(next,ugkp::diagnostics::Code::AccuracyResidual,"implicit material sparse residual",linear.finalResidual(),linear.initialResidual()*minValue(1,relativeLinearTarget));
        temperature.correctBoundaryConditions();const Foam::tmp<Foam::surfaceScalarField> flux=equation.flux();Real change=0;long double boundary=0,stableChange=0,inventoryScale=0;
        forAll(flux().boundaryField(),patch)if(!mesh.boundary()[patch].coupled())forAll(flux().boundaryField()[patch],f)boundary=hostAuditAdd(boundary,static_cast<long double>(dt)*flux().boundaryField()[patch][f]);
        for(std::size_t c=0;c<n;++c){const Real T=temperature[c];if(!finite(T)||T<=0){
                using namespace ugkp::diagnostics;
                recordFatal(p.diagnostics,makeRecord(!finite(T)?Code::NonFinite:Code::NegativeTemperature,4103,4,0,EntityKind::Cell,c,-1,T,0,0,Quantity::Temperature));
                error="nonfinite or nonpositive implicit material temperature";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            stableChange=hostAuditAdd(stableChange,static_cast<long double>(caloric[c].secant<false>(T,anchor[c]))*(T-anchor[c])
                -(static_cast<long double>(target[c])-caloric[c].energy<false>(anchor[c])));
            inventoryScale=hostAuditAdd(inventoryScale,static_cast<long double>(absValue(target[c]))+absValue(caloric[c].energy<false>(T)));change=maxValue(change,absValue(T-previous[c])/maxValue(1,maxValue(absValue(T),absValue(previous[c]))));}
        // All ranks must make the same nonlinear stopping decision.
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
            for(std::size_t c=0;c<n;++c){next.solid[c].energy=caloric[c].energy<false>(temperature[c]);MaterialPrimitive recovered;bool numerical=false;if(!recoverMaterial<false>(next.solid[c],next.solidMesh.volumes[c],p,recovered,nullptr,-1,&numerical)){error="implicit material endpoint EOS invalid";if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}next.solid[c].porosity=recovered.porosity;}
            return true;}
    }
    error="OpenFOAM material caloric nonlinear iteration failed";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;
}
void filmForcing(const ExchangePacket& packet,std::vector<CpuFilmForcing>& forcing){if(!(requiredConsumers(packet.kind)&ConsumeFilm))return;const auto d=packetDelta(packet);auto& f=forcing.at(packet.filmFace);
    if(packet.kind==ExchangeKind::SolidFilm){f.phaseMass+=d.film.mass;for(int s=0;s<Ns;++s)f.phaseSpecies[s]+=d.film.species[s];}
    else{f.mass+=d.film.mass;for(int s=0;s<Ns;++s)f.species[s]+=d.film.species[s];}
    f.withdrawnMass+=maxValue(0,-d.film.mass);for(int s=0;s<Ns;++s)f.withdrawnSpecies[s]+=maxValue(0,-d.film.species[s]);
    if(packet.kind==ExchangeKind::SolidFilm){f.phaseEnergy+=packet.energy;f.phaseKineticOutflow+=packet.liquidKineticAdvection;f.bottomWorkIncluded=true;}else{f.energy-=packet.energy;f.interfaceKineticOutflow+=packet.liquidKineticAdvection;f.topWorkIncluded=true;}
}
void gasDrive(const HostState& state,const std::vector<GasPrimitive>& bulk,const std::vector<Vec3>& traction,std::vector<CpuFilmDrive>& drive){drive.resize(state.film.size());for(std::size_t f=0;f<drive.size();++f){const auto& a=state.filmAux[f];auto& d=drive[f];const Vec3 n=state.surface.normal[f];const bool coupled=state.surface.gasFace[f]>=0;
    d.pressure=coupled?bulk[f].pressure:a.pressure;d.endpointPressure=d.pressure;d.shear=coupled?(traction[f]-n*dot(traction[f],n))*(mag(state.gasMesh.areaVectors[state.surface.gasFace[f]])/state.surface.area[f]):(state.surface.prescribedTopTraction.empty()?Vec3{}:state.surface.prescribedTopTraction[f]);
    d.hasCoupledNormalTrace=coupled;d.bottomNormalVelocity=2*dot(a.meanVelocity,n)-dot(a.topVelocity,n);d.topNormalVelocity=dot(a.topVelocity,n);d.interfaceNormalVelocity=a.normalVelocity;d.solidNormalVelocity=a.solidNormalVelocity;}}
bool baseTrace(const HostState& base,const PhysicsConfig& p,std::vector<GasPrimitive>& bulk,std::string& error,bool* numerical=nullptr){
    if(numerical)*numerical=false;
    bulk.resize(base.surface.area.size());
    if(base.surface.gasFace.size()!=bulk.size()){error="wall predictor face map size mismatch";return false;}
    for(std::size_t f=0;f<bulk.size();++f){const int face=base.surface.gasFace[f];if(face<0)continue;
        if(static_cast<std::size_t>(face)>=base.gasMesh.owner.size()){if(numerical)*numerical=true;error="wall predictor face outside gas mesh";return false;}

        const int c=base.gasMesh.owner[face];
        if((c<0||static_cast<std::size_t>(c)>=base.gas.size()||static_cast<std::size_t>(c)>=base.gasMesh.volumes.size())){if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
        if(!recoverGas(base.gas[c],base.gasMesh.volumes[c],p,bulk[f],nullptr,c)){
            {if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
        }
    }
    return true;
}
} // namespace
struct CpuMaterialDriver::Implementation {
    ModelConfig model;Foam::fvMesh* mesh;
    ugkwp::GasMechanismConfiguration mechanism;
    std::unique_ptr<WallClosureHost> wall;bool terminalWallFailure=false,surfaceNumericalFailure=false;
    Implementation(const ModelConfig& m,Foam::fvMesh* region,const ugkwp::GasMechanismConfiguration* canonical):model(m),mesh(region){
        if(m.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){wall.reset(new WallClosureHost);if(canonical)mechanism=*canonical;}
    }
    bool evaluate(const HostState& state,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,
        const std::vector<GasGradient>& gradients,const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
        Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
        const std::vector<Real>* sideVolume=nullptr,
        const std::vector<GasWallMatchingSample>* recorded=nullptr){
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
        bool success=evaluateCpuSurface(state,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,sideVolume,wall?&contexts:nullptr,&surfaceNumericalFailure);

        if(!success&&wall&&wall->failureStatus().code!=ugkwp::gaswall::WallCode::Success){
            terminalWallFailure=true;const auto& status=wall->failureStatus();
            error+="; stationary wall profile code "+std::to_string(int(status.code))+" node "+std::to_string(status.node)+" iteration "+std::to_string(status.iteration)+" residual "+(std::isfinite(status.residual)?std::to_string(status.residual):"NOT_AVAILABLE");
        }
        return success;
    }
};
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig& model,Foam::fvMesh* mesh):CpuMaterialDriver(model,mesh,nullptr){}
CpuMaterialDriver::CpuMaterialDriver(const ModelConfig& model,Foam::fvMesh* mesh,const ugkwp::GasMechanismConfiguration* mechanism):data_(new Implementation(model,mesh,mechanism)){}
CpuMaterialDriver::~CpuMaterialDriver()=default;
bool CpuMaterialDriver::predictWall(const HostState& base,const CouplingInterval& interval,WallProgram& output,std::string& error,CpuMaterialReport* diagnostics)const{
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,base.commitSequence+1,reinterpret_cast<std::uintptr_t>(this),base.acceptedSteps,base.time);
    PhysicsConfig physics=data_->model.physics;physics.diagnostics=&context;physics.tolerances.diagnostics=&context;
    const bool result=predictWallImpl(base,interval,output,error,diagnostics,physics);
    auto snapshot=ugkp::diagnostics::takeSnapshot(context,ugkp::diagnostics::sourceCatalogue());
    if(!result&&!snapshot.hasFatal){
        snapshot.hasFatal=true;
        snapshot.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,4102,4);
        snapshot.fatal.step=context.step;snapshot.fatal.simulationTime=context.simulationTime;snapshot.fatal.instanceId=context.instanceId;
        snapshot.fatal.operationSequence=context.operationSequence;
        snapshot.sources.push_back({4102,"applications/CHMT/materials/CpuMaterialDriver.C",__func__,__LINE__,error});
    }
    snapshot.sources.push_back({4103,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material temperature"});
    snapshot.sources.push_back({4104,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material residual"});
    ugkp::diagnostics::dispatch(snapshot,ugkp::diagnostics::makeFoamSink());
    return result;
}
bool CpuMaterialDriver::predictWallImpl(const HostState& base,const CouplingInterval& interval,WallProgram& output,std::string& error,CpuMaterialReport* diagnostics,const PhysicsConfig& p)const{
    if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
    if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=false;}
    const auto fail=[&](MaterialFailureOrigin origin){if(diagnostics)diagnostics->failureOrigin=origin;return false;};
    if(!(validateMaterialSweepRemainder(base,error,p.diagnostics)))return fail(MaterialFailureOrigin::GeometryNumerical);
    if((!validCouplingInterval(interval)||interval.begin!=base.time)){error="wall predictor interval/base time mismatch";return fail(MaterialFailureOrigin::GeometryNumerical);}
    if((p.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty())){error="CPU material owner requires actual volume/surface state, not normal columns";return fail(MaterialFailureOrigin::SurfaceNumerical);}
    std::vector<GasPrimitive> bulk;bool numerical=false;
    if(!(baseTrace(base,p,bulk,error,&numerical)))return fail(numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);CpuSurfaceResult surface;
    if(!data_->evaluate(base,p,bulk,{},{},{},interval.end-interval.begin,interval.identity.sequence,surface,error,nullptr,nullptr)){if(diagnostics)diagnostics->recoverable=false;return fail(data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);}
    WallProgram program;program.interval=interval;program.surface=base.surface;WallKnot first;first.time=interval.begin;first.faces=surface.wall;first.gasPoints=base.gasMesh.points;first.solidPoints=base.solidMesh.points;
    WallKnot last=first;last.time=interval.end;
    if(p.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){
        if(!base.surface.area.empty()){error="prescribed sinusoidal motion with material interface requires a compatible imposed transfer model";return fail(MaterialFailureOrigin::GeometryNumerical);}
        const auto& motion=p.meshMotion;const Real temporal=::sin(motion.angularFrequency*(interval.end-motion.timeOrigin));
        if((base.gasMesh.referencePoints.size()!=last.gasPoints.size())){error="prescribed gas motion reference topology missing";return fail(base.gasMesh.referencePoints.size()<last.gasPoints.size()?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical);}
        for(std::size_t i=0;i<last.gasPoints.size();++i){const Vec3 X=base.gasMesh.referencePoints[i];last.gasPoints[i]={X.x+motion.amplitude.x*::sin(motion.spatialWaveNumber.x*(X.x-motion.spatialOrigin.x))*temporal,
            X.y+motion.amplitude.y*::sin(motion.spatialWaveNumber.y*(X.y-motion.spatialOrigin.y))*temporal,X.z+motion.amplitude.z*::sin(motion.spatialWaveNumber.z*(X.z-motion.spatialOrigin.z))*temporal};}
    }
    if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.area.empty()){auto aux=base.filmAux;std::vector<Real> targets(aux.size());const Real dt=interval.end-interval.begin;
        for(std::size_t f=0;f<aux.size();++f){aux[f].solidNormalVelocity=surface.physics[f].solidSpeed;aux[f].normalVelocity=surface.physics[f].topSpeed;aux[f].solidFront+=dt*aux[f].solidNormalVelocity;aux[f].gasFront+=dt*aux[f].normalVelocity;
            const int sf=base.surface.solidFace[f];targets[f]=sf>=0?dt*aux[f].solidNormalVelocity*mag(base.solidMesh.areaVectors[sf]):0;}
        HostMesh gas,solid;SurfaceMesh moved;SweepConstraintReport report;
        if(!moveCoupledMeshesConstrained(base,aux,targets,dt,gas,solid,moved,report,error,SweepConstraintControls{},p.tolerances,&numerical)){
            if(diagnostics)diagnostics->recoverable=false;
            return fail(numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None);
        }
        last.gasPoints=gas.points;last.solidPoints=solid.points;}
    program.knots={first,last};output=std::move(program);error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram){
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,base.commitSequence+1,reinterpret_cast<std::uintptr_t>(this),base.acceptedSteps,base.time);
    PhysicsConfig physics=data_->model.physics;physics.diagnostics=&context;physics.tolerances.diagnostics=&context;
    const bool result=advanceCandidateImpl(base,geometryEndpoint,history,controls,output,corrected,report,error,executedProgram,physics);
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
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram,const PhysicsConfig& p){
    report=CpuMaterialReport{};error.clear();
    if(!(validateMaterialSweepRemainder(base,error,p.diagnostics))){report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;report.recoverable=false;return false;}
    const auto& window=history.interval();
    auto wallGeometryConfig=p.wallModel;wallGeometryConfig.enableSst=p.enableSst;
    if(!base.solid.empty()&&!data_->mesh){report.recoverable=false;error="CPU material native mesh unavailable";return false;}
    if((!controlsValid(controls)||!history.complete()||window.begin!=base.time||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty()||base.solidMesh.topologyHash!=geometryEndpoint.solidMesh.topologyHash||base.gasMesh.topologyHash!=geometryEndpoint.gasMesh.topologyHash)){
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
    try {

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
        HostState state=base;state.ledger.clear();state.filmStorage.clear();
        WallProgram program;program.interval=window;program.surface=base.surface;
        CpuMaterialReport accumulated;accumulated.donorPlan.interval=window;
        MaterialReserveKnot initial;initial.time=base.time;initial.cumulativeSolid.resize(base.solid.size());initial.cumulativeFilm.resize(base.film.size());accumulated.donorPlan.knots.push_back(initial);
        std::vector<IntervalDonorSum> physicalSweepSum(base.surface.area.size()),actualSweepSum(base.surface.area.size());
        std::vector<SolidQ> gasSolidChange(base.solid.size());std::vector<FilmQ> gasFilmChange(base.film.size());
        auto cursor=history.cursor();const Real interval=window.end-window.begin;
        WindowTime elapsed{};const WindowTime windowEnd=intervalDuration(window);
        // The macro endpoint has an actual gas state. Do not substitute the
        // representative last RK stage when converting film endpoint pV.
        std::vector<GasWallMatchingSample> endpointMatching;bool endpointMatchingReady=false;
        while(elapsed<windowEnd){
            if(p.diagnostics)ugkp::diagnostics::beginOperation(*p.diagnostics,window.identity.sequence,base.acceptedSteps,state.time);
            if(accumulated.materialSteps==static_cast<std::uint64_t>(controls.maxSubsteps))materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"CPU material substep warning budget exceeded; advancing calculation retained",accumulated.materialSteps,controls.maxSubsteps);
            if(accumulated.materialSteps==std::numeric_limits<std::uint64_t>::max()){error="CPU material substep counter overflow";return false;}
            Real dt=std::min(boundedWindowDuration(windowEnd,elapsed),controls.maxSubstep>0?controls.maxSubstep:interval);
            // Only a real regime change forces alignment with a gas-history knot.
            // Ordinary gas micro boundaries do not determine the material clock.
            const GasIntervalRecord* driveRecord=nullptr;
            for(const auto& r:history.records())if(elapsed<r.localEnd){driveRecord=&r;break;}
            if(!driveRecord){error="CPU material driving history exhausted";return false;}
            WindowTime boundedEnd=elapsed+dt;
            if(!materialDriveSlabEndLocal(history,elapsed,boundedEnd,controls.driveRelativeTolerance,controls.driveTractionScale,boundedEnd,error)){
                {report=accumulated;report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;}

            }
            dt=boundedWindowDuration(boundedEnd,elapsed);
            if(executedProgram)for(const auto& knot:executedProgram->knots)if(knotOffset(window,knot)>elapsed&&knotOffset(window,knot)<elapsed+dt)dt=boundedWindowDuration(knotOffset(window,knot),elapsed);
            std::vector<GasPrimitive> bulk=driveRecord->gasTrace;std::vector<Vec3> traction=driveRecord->gasTraction;
            if((bulk.size()!=state.surface.area.size()||traction.size()!=state.surface.area.size())){
                if(state.surface.area.empty()){bulk.clear();traction.clear();}else{report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;report.recoverable=false;error="CPU material requires actual face-ordered gas drive history";return false;}}

            HostState trial;WallKnot beginning;CpuSurfaceResult surface;MaterialTransportResult transport;
            std::vector<ExchangePacket> packets;std::vector<Real> radiation;std::vector<CpuFilmDrive> drive;
            std::vector<CpuFilmForcing> forcing;std::vector<FilmQ> filmRate;
            std::vector<Real> sideVolume;
            std::vector<Real> solidSweep,gasSweep;IntervalCursor trialCursor=cursor;
            SurfaceMesh filmEvaluationSurface=state.surface;filmEvaluationSurface.sweptEdgeArea.assign(state.surface.edgeOwner.size(),0);
            bool accepted=false;WindowTime trialElapsed;
            MaterialFailureOrigin lastFailure=MaterialFailureOrigin::None;
            // Select the physical transport CFL step once, before any sources
            // or candidate state are evaluated.
            bool transportNumerical=false;
            if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,{},dt,controls.transportCfl,transport,error,&transportNumerical)) {
                report=accumulated;report.failureOrigin=transportNumerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;return false;
            }
            dt=std::min(dt,transport.dtLimit);
            do {
                if(!finite(dt)||!(dt>0)||elapsed+dt==elapsed){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU material substep does not advance";break;}
                error.clear();trial=state;trialElapsed=elapsed+dt;trial.time=windowAbsolute(window.begin,trialElapsed);trialCursor=cursor;
                if(!trialCursor.takeThroughLocal(trialElapsed,packets,radiation,error))break;

                gasDrive(state,bulk,traction,drive);
                std::vector<GasPrimitive> endpointBulk;std::vector<Vec3> endpointTraction;const GasIntervalRecord* endpointRecord=nullptr;
                if(trialElapsed==windowEnd){bool numerical=false;
                    if(!(baseTrace(geometryEndpoint,p,endpointBulk,error,&numerical))){lastFailure=numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;break;}}
                else{
                    if(!history.sampleGasDriveLocal(trialElapsed,endpointBulk,endpointTraction,error))break;
                    for(const auto& r:history.records())if(trialElapsed<r.localEnd){endpointRecord=&r;break;}
                    if(!endpointRecord){error="missing endpoint gas driving record";break;}
                }
                for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){

                    drive[f].endpointPressure=endpointBulk[f].pressure;}
                if(!error.empty())break;
                if(data_->wall&&p.enableFilm){
                    const auto& matching=driveRecord->gasWallMatching;
                    if(trialElapsed==windowEnd&&!endpointMatchingReady){
                        if(!data_->wall->prepareGeometry(geometryEndpoint.gasMesh,geometryEndpoint.surface.gasFace,error,wallGeometryConfig,p.diagnostics)||!data_->wall->sample(geometryEndpoint,p,endpointMatching,error)){report.recoverable=false;return false;}
                        endpointMatchingReady=true;
                    }
                    const auto& ending=trialElapsed==windowEnd?endpointMatching:endpointRecord->gasWallMatching;
                    if((matching.size()!=state.surface.area.size()||ending.size()!=matching.size())){
                        report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;report.recoverable=false;error="film wall matching pressure history layout mismatch";return false;}
                    for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){

                        drive[f].pressure=matching[f].pressure;drive[f].endpointPressure=ending[f].pressure;
                    }
                }
                forcing.assign(state.film.size(),CpuFilmForcing{});
                for(const auto& packet:packets){if((requiredConsumers(packet.kind)&ConsumeFilm)&&(packet.filmFace<0||static_cast<std::size_t>(packet.filmFace)>=forcing.size())){lastFailure=MaterialFailureOrigin::FilmNumerical;error="CPU gas history film receiver outside mesh";break;}filmForcing(packet,forcing);}
                if(!error.empty())break;
                if(radiation.size()!=state.surface.area.size()&&!radiation.empty()){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU radiation face layout mismatch";break;}
                for(std::size_t f=0;f<radiation.size();++f)if(driveRecord->radiationReceiver[f]==ConsumeFilm){if(f>=forcing.size()){lastFailure=MaterialFailureOrigin::FilmNumerical;error="film radiation has no film owner";break;}forcing[f].radiationEnergy+=radiation[f];}
                if(!error.empty())break;
                // An imposed trajectory is independent of film transport. Build
                // its endpoint/ALE geometry once before the interface RHS.
                auto prepareImposedGeometry=[&]()->bool{
                    const Real fraction=windowFraction(trialElapsed,WindowTime{},windowEnd,trial.time,window.begin,window.end);std::vector<Vec3> gp=base.gasMesh.points,sp=base.solidMesh.points;
                    if((gp.size()!=geometryEndpoint.gasMesh.points.size()||sp.size()!=geometryEndpoint.solidMesh.points.size())){lastFailure=MaterialFailureOrigin::OwnerNumerical;error="material trajectory topology mismatch";return false;}

                    for(std::size_t i=0;i<gp.size();++i)gp[i]+=(geometryEndpoint.gasMesh.points[i]-gp[i])*fraction;
                    for(std::size_t i=0;i<sp.size();++i)sp[i]+=(geometryEndpoint.solidMesh.points[i]-sp[i])*fraction;
                    if(executedProgram){WallKnot executed;if(!intervalSampleWallLocal(*executedProgram,trialElapsed,executed,error))return false;gp=executed.gasPoints;sp=executed.solidPoints;}
                    if(!stageGeometry(state.gasMesh,gp,dt,trial.gasMesh,gasSweep,lastFailure)||!stageGeometry(state.solidMesh,sp,dt,trial.solidMesh,solidSweep,lastFailure))return false;
                    if(!state.surface.area.empty()&&!trajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,lastFailure))return false;
                    return true;
                };
                if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession){
                    if(!prepareImposedGeometry())break;
                    if(p.enableFilm)filmEvaluationSurface=trial.surface;
                }
                const auto primaryForcing=forcing;
                const int filmCouplingBudget=p.enableFilm?std::max(1,p.tolerances.maxCouplingIterations):1;
                HostState evaluation;std::vector<FilmAux> aux;std::vector<Real> target;
                bool filmCouplingReady=false;
                for(int localIteration=0;localIteration<filmCouplingBudget;++localIteration){
                forcing=primaryForcing;const auto previousDrive=drive;
                bool stageSweepsReady=p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession;
                evaluation=state;evaluation.surface=filmEvaluationSurface;
                if(p.enableFilm){std::vector<FilmAux> profiles;
                    bool numerical=false;
                    if(!evaluateCpuFilmTransport(evaluation,p,dt,drive,filmRate,error,&profiles,nullptr,&sideVolume,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        break;
                    }
                    evaluation.filmAux=profiles;
                    for(std::size_t f=0;f<profiles.size();++f)evaluation.film[f].enthalpy+=pressureVolumeProduct(state.filmAux[f].pressure,state.film[f].mass/p.liquid.rho,profiles[f].pressure,state.film[f].mass/p.liquid.rho);
                }else{filmRate.clear();sideVolume.assign(state.surface.area.size(),0);}
                if(!data_->evaluate(evaluation,p,bulk,driveRecord->gasGradient,transport.rates,filmRate,dt,window.identity.sequence,surface,error,&sideVolume,&driveRecord->gasWallMatching)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}

                // Preserve actual gas-facing source energy. Only the CPU-owned
                // internal phase packet is taken from the constitutive proposal.
                for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){const auto& physics=surface.physics[f];
                    drive[f].hasCoupledNormalTrace=true;drive[f].bottomNormalVelocity=physics.liquidBottomNormal;drive[f].topNormalVelocity=physics.liquidTopNormal;
                    drive[f].solidNormalVelocity=physics.solidSpeed;drive[f].interfaceNormalVelocity=physics.topSpeed;}
                for(auto& phase:surface.phasePackets){const std::size_t f=phase.filmFace;
                    const Vec3 velocity=state.surface.baseVelocity.empty()?Vec3{}:state.surface.baseVelocity[f];
                    const Vec3 n=state.surface.normal[f];const Vec3 bottom=velocity-n*dot(velocity,n)+n*surface.physics[f].liquidBottomNormal;
                    bool exactEvent=false;
                    bool numerical=false;
                    if(!finalizeCpuFilmPhasePacket(state.film[f],state.filmAux[f].pressure,filmRate[f],forcing[f],dt,bottom,p,phase,exactEvent,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        break;
                    }
                    forcing[f].exactPhaseEvent=forcing[f].exactPhaseEvent||exactEvent;
                    if(!error.empty())break;filmForcing(phase,forcing);}
                if(!error.empty())break;
                if(p.enableFilm)for(std::size_t f=0;f<forcing.size();++f){const Real remaining=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                    const Real scale=absValue(state.film[f].mass)+absValue(auditProduct(dt,filmRate[f].mass))+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass);
                    if((forcing[f].mass<0||forcing[f].phaseMass<0)&&absValue(remaining)<=16*std::numeric_limits<Real>::epsilon()*scale)forcing[f].exactPhaseEvent=true;
                }
                if(p.enableFilm){Real filmDt=dt;bool numerical=false;
                    if(!estimateCpuFilmStep(evaluation,p,forcing,drive,dt,dt,filmDt,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        break;
                    }
                    if(filmDt<dt*(1-32*std::numeric_limits<Real>::epsilon()))materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"film estimated step exceeded",dt,filmDt);}
                target.assign(state.surface.area.size(),0);
                for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasSolid){
                    std::size_t match=0;
                    if(!packetAddressing.gasSolidIndex(state,packet,match,error))break;
                    const int donor=packet.solidCell;
                    MaterialPrimitive material;
                    if(!(recoverMaterial(state.solid[donor],state.solidMesh.volumes[donor],p,material))){
                        lastFailure=MaterialFailureOrigin::GeometryNumerical;error="surface donor EOS failed";break;}
                    for(int c=0;c<Nc;++c)if(packet.condensed[c]!=0){
                        if((!finite(packet.condensed[c])||!finite(p.condensed[c].rho)||!(p.condensed[c].rho>0))){
                            lastFailure=MaterialFailureOrigin::GeometryNumerical;error="invalid surface recession constituent";break;}
                        target[match]-=packet.condensed[c]/p.condensed[c].rho/(1-material.porosity);
                    }
                    if(!error.empty())break;
                }
                for(const auto& phase:surface.phasePackets){MaterialPrimitive material;const int c=phase.solidCell;if(!(recoverMaterial(state.solid[c],state.solidMesh.volumes[c],p,material))){lastFailure=MaterialFailureOrigin::GeometryNumerical;error="phase donor EOS failed";break;}

                    target[phase.filmFace]-=phase.mass==0 ? Real(0)
                        : phase.mass/p.condensed[p.material.phaseCondensed].rho/(1-material.porosity);}
                if(!error.empty())break;
                aux=evaluation.filmAux;
                for(std::size_t f=0;f<aux.size();++f)if(state.surface.gasFace[f]>=0){const int sf=state.surface.solidFace[f],gf=state.surface.gasFace[f];const Real as=sf>=0?mag(state.solidMesh.areaVectors[sf]):0,ag=mag(state.gasMesh.areaVectors[gf]);
                    aux[f].solidNormalVelocity=as>0?target[f]/(as*dt):0;
                    const Real mass=p.enableFilm?dt*filmRate[f].mass+forcing[f].mass+forcing[f].phaseMass:0;
                    const Real side=p.enableFilm?sideVolume[f]:0;
                    if(p.enableFilm){Real endpointMass=state.film[f].mass+mass;const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(mass));if(absValue(endpointMass)<=roundoff&&forcing[f].exactPhaseEvent)endpointMass=0;if((endpointMass<0||!finite(endpointMass))){lastFailure=MaterialFailureOrigin::GeometryNumerical;error="negative film mass in geometric proposal";break;}
                        if(!(filmThicknessFromMass(endpointMass,evaluation.surface.area[f],p.liquid.rho,aux[f].thickness))){lastFailure=MaterialFailureOrigin::GeometryNumerical;error="invalid endpoint film thickness";break;}aux[f].area=evaluation.surface.area[f];}
                    aux[f].normalVelocity=p.enableFilm&&surface.physics[f].wet?(target[f]/dt+mass/(dt*p.liquid.rho)-side)/ag:aux[f].solidNormalVelocity;
                    aux[f].solidFront=state.filmAux[f].solidFront+dt*aux[f].solidNormalVelocity;aux[f].gasFront=state.filmAux[f].gasFront+dt*aux[f].normalVelocity;}
                if(!error.empty())break;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!target.empty()){SweepConstraintReport motion;
                    bool numerical=false;
                    if(!moveCoupledMeshesConstrained(state,aux,target,dt,trial.gasMesh,trial.solidMesh,trial.surface,motion,error,SweepConstraintControls{},p.tolerances,&numerical)){
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
                if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){WallKnot executed;
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
                if(!p.enableFilm){filmCouplingReady=true;break;}
                HostState consistentFilm=state;consistentFilm.surface=trial.surface;
                std::vector<FilmQ> actualRates;std::vector<Real> actualSideVolume;
                bool rateNumerical=false;
                if(!evaluateCpuFilmTransport(consistentFilm,p,dt,drive,actualRates,error,nullptr,nullptr,&actualSideVolume,&rateNumerical)){
                    lastFailure=rateNumerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;break;
                }
                if(actualRates.size()!=filmRate.size()||actualSideVolume.size()!=sideVolume.size()){
                    lastFailure=MaterialFailureOrigin::FilmNumerical;error="film local coupling layout changed";break;
                }
                Real couplingResidual=0;
                // Host-only audit arithmetic: a finite tolerance or a small
                // residual must not overflow before their dimensionless ratio.
                const auto scaledDifference=[](long double difference,long double tolerance){
                    if(difference==0)return Real(0);
                    const long double magnitude=std::abs(difference);
                    if(!std::isfinite(magnitude)||std::isnan(tolerance)||tolerance<=0)
                        return std::numeric_limits<Real>::infinity();
                    return hostAuditNarrow(magnitude/tolerance);
                };
                const long double auditDt=dt;
                for(std::size_t f=0;f<filmRate.size();++f){
                    const long double massScale=std::max(std::abs(static_cast<long double>(state.film[f].mass)),
                        std::max(std::abs(auditDt*filmRate[f].mass),std::abs(auditDt*actualRates[f].mass)));
                    const long double massTolerance=p.tolerances.absoluteMass+static_cast<long double>(p.tolerances.relativeMass)*massScale;
                    couplingResidual=maxValue(couplingResidual,scaledDifference(auditDt*(static_cast<long double>(actualRates[f].mass)-filmRate[f].mass),massTolerance));
                    for(int species=0;species<Ns;++species)couplingResidual=maxValue(couplingResidual,
                        scaledDifference(auditDt*(static_cast<long double>(actualRates[f].species[species])-filmRate[f].species[species]),massTolerance));
                    const long double energyScale=std::max(std::abs(static_cast<long double>(state.film[f].enthalpy)),
                        std::max(std::abs(auditDt*filmRate[f].enthalpy),std::abs(auditDt*actualRates[f].enthalpy)));
                    const long double energyTolerance=p.tolerances.absoluteEnergy+static_cast<long double>(p.tolerances.relativeEnergy)*energyScale;
                    couplingResidual=maxValue(couplingResidual,scaledDifference(auditDt*(static_cast<long double>(actualRates[f].enthalpy)-filmRate[f].enthalpy),energyTolerance));
                    couplingResidual=maxValue(couplingResidual,scaledDifference(p.liquid.rho*auditDt*(static_cast<long double>(actualSideVolume[f])-sideVolume[f]),massTolerance));
                    const long double normalChange=std::max(std::abs(static_cast<long double>(drive[f].bottomNormalVelocity)-previousDrive[f].bottomNormalVelocity),
                        std::max(std::abs(static_cast<long double>(drive[f].topNormalVelocity)-previousDrive[f].topNormalVelocity),
                        std::max(std::abs(static_cast<long double>(drive[f].solidNormalVelocity)-previousDrive[f].solidNormalVelocity),
                        std::abs(static_cast<long double>(drive[f].interfaceNormalVelocity)-previousDrive[f].interfaceNormalVelocity))));
                    couplingResidual=maxValue(couplingResidual,scaledDifference(p.liquid.rho*auditDt*trial.surface.area[f]*normalChange,massTolerance));
                }
                if(couplingResidual<=1||localIteration+1==filmCouplingBudget){
                    if(couplingResidual>1)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,
                        "same-dt film geometry/transport coupling residual",couplingResidual,1);
                    filmCouplingReady=true;break;
                }
                // Same accepted inventory, physical interval and external packets.
                // Only the mathematical geometry/transport/interface iterate changes.
                filmEvaluationSurface=trial.surface;
                }
                if(!filmCouplingReady)break;
                // Audit the final local RHS once; this does not drive the fixed point.
                for(const auto& record:history.records())if(record.localBegin>elapsed&&record.localBegin<trialElapsed){
                    CpuSurfaceResult response;
                    if(!data_->evaluate(evaluation,p,record.gasTrace,record.gasGradient,transport.rates,filmRate,dt,window.identity.sequence,response,error,&sideVolume,&record.gasWallMatching)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                    Real fraction=0;
                    for(std::size_t f=0;f<surface.physics.size();++f){if(state.surface.gasFace[f]<0)continue;
                        const auto& a=surface.physics[f];const auto& b=response.physics[f];const Real scale=state.surface.area[f]*dt;
                        const int cell=state.surface.solidCell[f];Real solidThermalScale=0,solidDonor=0,phaseDonor=0,solidSpecific=0,filmThermalScale=0,filmSpecific=0;
                        if(cell>=0){MaterialCaloric caloric;MaterialPrimitive primitive;if(!materialCaloric(state.solid[cell],p,caloric,error)||!recoverMaterial(state.solid[cell],state.solidMesh.volumes[cell],p,primitive)){report=accumulated;report.failureOrigin=MaterialFailureOrigin::SurfaceNumerical;return false;}
                            solidThermalScale=caloric.capacity(primitive.temperature)*maxValue(1,primitive.temperature);
                            for(int c=0;c<Nc;++c)solidDonor+=state.solid[cell].condensed[c];
                            if(p.material.phaseCondensed>=0){phaseDonor=state.solid[cell].condensed[p.material.phaseCondensed];solidSpecific=condensedE(p.condensed[p.material.phaseCondensed],primitive.temperature);}}
                        Real thermalScale=solidThermalScale,primaryDonor=solidDonor;
                        if(p.enableFilm&&state.film[f].mass>0){const Real T=state.filmAux[f].temperature;
                            filmThermalScale=state.film[f].mass*(p.liquid.cp0+p.liquid.cp1*T)*maxValue(1,T);filmSpecific=condensedE(p.liquid,T)-p.liquidReferencePressure/p.liquid.rho;
                            thermalScale=thermalScale>0?minValue(thermalScale,filmThermalScale):filmThermalScale;primaryDonor=state.film[f].mass;
                            if(a.phaseMass<0||b.phaseMass<0)phaseDonor=state.film[f].mass;}
                        fraction=maxValue(fraction,scale*absValue(a.phaseMass-b.phaseMass)/maxValue(p.tolerances.absoluteMass,phaseDonor));
                        for(int species=0;species<Ns;++species){const Real donor=p.enableFilm&&state.film[f].mass>0?state.film[f].species[species]:primaryDonor;
                            fraction=maxValue(fraction,scale*absValue(a.gasSpecies[species]-b.gasSpecies[species])/maxValue(p.tolerances.absoluteMass,donor));
                            if(cell>=0)fraction=maxValue(fraction,scale*absValue(a.poreSpecies[species]-b.poreSpecies[species])/maxValue(p.tolerances.absoluteMass,state.solid[cell].pore[species]));}
                        if(thermalScale>0)fraction=maxValue(fraction,scale*absValue(a.phaseConductive-b.phaseConductive)/thermalScale);
                        const long double massChange=static_cast<long double>(b.phaseMass)-a.phaseMass;
                        const long double energyChange=static_cast<long double>(b.phaseEnergy)-a.phaseEnergy;
                        if(solidThermalScale>0)fraction=maxValue(fraction,scale*static_cast<Real>(std::fabs(energyChange-massChange*solidSpecific))/solidThermalScale);
                        if(filmThermalScale>0)fraction=maxValue(fraction,scale*static_cast<Real>(std::fabs(energyChange+static_cast<long double>(b.bottomKineticAdvection)-a.bottomKineticAdvection-massChange*filmSpecific))/filmThermalScale);
                    }
                    if(fraction>controls.driveSourceFraction)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"material source variation",fraction,controls.driveSourceFraction);
                }
                if(!error.empty())break;

                if(p.enableFilm&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){
                    for(std::size_t f=0;f<state.film.size();++f)if(state.surface.gasFace[f]>=0){
                        const int gf=state.surface.gasFace[f],sf=state.surface.solidFace[f];
                        Real updatedMass=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                        const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(auditProduct(dt,filmRate[f].mass))+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass));
                        if(forcing[f].exactPhaseEvent&&absValue(updatedMass)<=roundoff)updatedMass=0;
                        if(!filmSweepsCompatible(state.film[f].mass,updatedMass,p.liquid.rho,gasSweep[gf],sf>=0?solidSweep[sf]:0,dt*sideVolume[f],p.tolerances)){
                            materialWarning(state,ugkp::diagnostics::Code::ConservationResidual,"film swept volume / mass mismatch",gasSweep[gf]-(sf>=0?solidSweep[sf]:0)-dt*sideVolume[f],(updatedMass-state.film[f].mass)/p.liquid.rho);}
                    }
                    if(!error.empty())break;
                }
                {bool numerical=false;if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,solidSweep,dt,controls.transportCfl,transport,error,&numerical)){
                    lastFailure=numerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;
                    break;
                }}
                if(dt>transport.dtLimit)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"ALE material transport step bound",dt,transport.dtLimit);
                bool reactionOk=true;for(std::size_t c=0;c<state.solid.size();++c){std::uint64_t count=0;SolidQ reacted;
                    bool numerical=false;
                    if(!advanceMaterialReactions(state.solid[c],state.solidMesh.volumes[c],p,dt,controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps,reacted,count,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;
                        reactionOk=false;break;
                    }
                    trial.solid[c]=addSolid(reacted,transport.rates[c],dt);}
                if(!reactionOk){break;}
                std::vector<ExchangePacket> all=packets;all.insert(all.end(),surface.phasePackets.begin(),surface.phasePackets.end());
                std::vector<SolidQ> applied;
                bool packetNumerical=false;
                if(!applyMaterialPacketBatch(trial.solid,all,applied,error,&packetNumerical)){
                    lastFailure=packetNumerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;
                    break;
                }
                trial.solid=std::move(applied);
                if(!error.empty())break;
                for(std::size_t f=0;f<radiation.size();++f){if(driveRecord->radiationReceiver[f]==ConsumeSolid){const int c=state.surface.solidCell[f];if(c<0){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="radiation has no material owner";break;}trial.solid[c].energy+=radiation[f];}}
                if(!error.empty())break;
                bool materialAdmissible=true;
                for(std::size_t c=0;c<trial.solid.size();++c){MaterialPrimitive material;
                    bool numerical=true;
                    if((!validSolidInventory(trial.solid[c]))||(trialElapsed<windowEnd&&!(recoverMaterial(trial.solid[c],trial.solidMesh.volumes[c],p,material,nullptr,-1,&numerical)))){
                        lastFailure=numerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;materialAdmissible=false;break;
                    }
                    if(trialElapsed<windowEnd)trial.solid[c].porosity=material.porosity;
                }
                if(!materialAdmissible){break;}
                if(p.enableFilm){HostState filmState=state;filmState.surface=trial.surface;CpuFilmCandidate film;
                    bool numerical=false;
                    if(!advanceCpuFilmCandidate(filmState,p,dt,forcing,drive,film,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        break;
                    }
                    trial.film=std::move(film.film);trial.filmAux=std::move(film.filmAux);
                    for(std::size_t f=0;f<trial.film.size();++f){trial.filmAux[f].solidFront=aux[f].solidFront;trial.filmAux[f].gasFront=aux[f].gasFront;trial.filmAux[f].solidNormalVelocity=aux[f].solidNormalVelocity;trial.filmAux[f].normalVelocity=aux[f].normalVelocity;
                        if((state.film[f].mass==0)!=(trial.film[f].mass==0)&&trialElapsed<windowEnd)materialWarning(state,ugkp::diagnostics::Code::AccuracyResidual,"film phase transition inside coupling window",trial.time);}}
                else trial.filmAux=aux;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!(updateMaterialSweepRemainder(state,target,solidSweep,trial.solidMesh,trial.solidSweepRemainder,error,p.diagnostics))){lastFailure=MaterialFailureOrigin::GeometryNumerical;break;}
                for(std::size_t f=0;f<target.size();++f){
                    physicalSweepSum[f].add(target[f]);
                    const int face=state.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(solidSweep[face]);
                }
                beginning.time=state.time;beginning.localTime=elapsed;beginning.faces=surface.wall;beginning.gasPoints=state.gasMesh.points;beginning.solidPoints=state.solidMesh.points;
                if(program.knots.empty())program.knots.push_back(beginning);
                WallKnot ending=beginning;ending.time=trial.time;ending.localTime=trialElapsed;ending.gasPoints=trial.gasMesh.points;ending.solidPoints=trial.solidMesh.points;
                if(trialElapsed<windowEnd){CpuSurfaceResult endpointProposal;
                    if(!endpointRecord){error="missing endpoint wall matching record";break;}if(!data_->evaluate(trial,p,endpointBulk,endpointRecord->gasGradient,{},{},dt,window.identity.sequence,endpointProposal,error,nullptr,&endpointRecord->gasWallMatching)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                    ending.faces=endpointProposal.wall;for(std::size_t f=0;f<ending.faces.size();++f)ending.faces[f].primaryKind=beginning.faces[f].primaryKind;}
                for(std::size_t f=0;f<ending.faces.size();++f){ending.faces[f].solidNormalVelocity=aux[f].solidNormalVelocity;ending.faces[f].normalVelocity=aux[f].normalVelocity;}
                program.knots.push_back(ending);
                for(const auto& packet:packets){const auto d=packetDelta(packet);if(requiredConsumers(packet.kind)&ConsumeSolid)gasSolidChange[packet.solidCell]=addSolid(gasSolidChange[packet.solidCell],d.solid);
                    if(requiredConsumers(packet.kind)&ConsumeFilm){auto& q=gasFilmChange[packet.filmFace];q.mass+=d.film.mass;for(int s=0;s<Ns;++s)q.species[s]+=d.film.species[s];}}
                MaterialReserveKnot knot;knot.time=trial.time;knot.localTime=trialElapsed;knot.cumulativeSolid.resize(trial.solid.size());knot.cumulativeFilm.resize(trial.film.size());
                for(std::size_t c=0;c<trial.solid.size();++c)knot.cumulativeSolid[c]=addSolid(addSolid(trial.solid[c],base.solid[c],-1),gasSolidChange[c],-1);
                for(std::size_t f=0;f<trial.film.size();++f){auto& q=knot.cumulativeFilm[f];q.mass=trial.film[f].mass-base.film[f].mass-gasFilmChange[f].mass;for(int s=0;s<Ns;++s)q.species[s]=trial.film[f].species[s]-base.film[f].species[s]-gasFilmChange[f].species[s];}
                accumulated.donorPlan.knots.push_back(std::move(knot));
                for(auto phase:surface.phasePackets){phase.consumerMask=requiredConsumers(phase.kind);trial.ledger.push_back(phase);}
                accumulated.materialSteps++;
                state=std::move(trial);elapsed=trialElapsed;cursor=trialCursor;accepted=true;
            } while(false);
            if(!accepted){if(error.empty())error="CPU material single-pass operation failed";report=accumulated;report.failureOrigin=lastFailure;return false;}
        }
        // Lie/IMEX splitting: all conservative local sources are now in Utarget.
        // This is one true 3-D backward-Euler conduction candidate per window,
        // regardless of gas microstep or local reaction/film step count.
        if(!state.solid.empty()){
            if(p.diagnostics)ugkp::diagnostics::beginOperation(*p.diagnostics,window.identity.sequence,base.acceptedSteps,state.time);
            const bool solved=nativeConduction(*data_->mesh,base,state,p,interval,controls,accumulated,error);
            if(!solved){report=accumulated;return false;}
        }
        state.filmStorage.resize(state.film.size());for(std::size_t f=0;f<state.film.size();++f){auto& storage=state.filmStorage[f];storage.step=window.identity.sequence;storage.stage=1;storage.geometry=state.gasMesh.geometryVersion;storage.face=state.surface.persistentId[f];storage.filmFace=f;storage.oldPV=base.filmAux[f].pressure*base.film[f].mass/p.liquid.rho;storage.newPV=state.filmAux[f].pressure*state.film[f].mass/p.liquid.rho;storage.consumed=true;}
        const auto samePoints=[](const HostMesh& a,const HostMesh& b){if(a.points.size()!=b.points.size())return false;for(std::size_t i=0;i<a.points.size();++i)if(a.points[i].x!=b.points[i].x||a.points[i].y!=b.points[i].y||a.points[i].z!=b.points[i].z)return false;return true;};
        if(samePoints(state.gasMesh,geometryEndpoint.gasMesh)&&samePoints(state.solidMesh,geometryEndpoint.solidMesh)){
            state.gasMesh=geometryEndpoint.gasMesh;state.solidMesh=geometryEndpoint.solidMesh;state.gasStages=geometryEndpoint.gasStages;state.solidStages=geometryEndpoint.solidStages;
            if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
                // Re-certify the executed piecewise trajectory, not an invented
                // straight macro chord. Do not increment a provisional carry a
                // second time when adopting the gas owner's endpoint metadata.
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
        state.gas=geometryEndpoint.gas;state.sst=geometryEndpoint.sst;state.particles=geometryEndpoint.particles;state.gasTemperature=geometryEndpoint.gasTemperature;
        state.rejectedSteps=geometryEndpoint.rejectedSteps;state.gasVoidFraction=geometryEndpoint.gasVoidFraction;state.gasStages=geometryEndpoint.gasStages;
        state.gasMesh.boundaryPrimitive=geometryEndpoint.gasMesh.boundaryPrimitive;state.gasMesh.thermalBoundary=geometryEndpoint.gasMesh.thermalBoundary;state.gasMesh.boundarySst=geometryEndpoint.gasMesh.boundarySst;
        state.time=window.end;program.donorPlan=accumulated.donorPlan;
        if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
        if(!program.knots.empty()){const auto& last=history.records().back();CpuSurfaceResult endpointSurface;
            std::vector<GasPrimitive> finalBulk;bool numerical=false;if(!(baseTrace(state,p,finalBulk,error,&numerical))){report=accumulated;report.failureOrigin=numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;return false;}
            if(!data_->evaluate(state,p,finalBulk,last.gasGradient,{},{},interval,window.identity.sequence,endpointSurface,error,nullptr,nullptr)){report=accumulated;report.failureOrigin=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;report.recoverable=false;return false;}
            program.knots.back().faces=endpointSurface.wall;
            // The executed regime owns the interval including its terminal flux.
            // An endpoint phase event changes the next window's predictor only.
            for(std::size_t f=0;f<program.knots.back().faces.size();++f)program.knots.back().faces[f].primaryKind=program.knots.front().faces[f].primaryKind;
        }
        for(std::size_t i=0;i<state.gasMesh.points.size();++i)if(mag(state.gasMesh.points[i]-geometryEndpoint.gasMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        for(std::size_t i=0;i<state.solidMesh.points.size();++i)if(mag(state.solidMesh.points[i]-geometryEndpoint.solidMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        output=std::move(state);corrected=std::move(program);report=std::move(accumulated);error.clear();return true;
    }catch(const Foam::error& e){report.failureOrigin=MaterialFailureOrigin::None;error="OpenFOAM CPU material failure: "+std::string(e.what());return false;}
    catch(const std::exception& e){report.failureOrigin=MaterialFailureOrigin::None;error="CPU material failure: "+std::string(e.what());return false;}
}
} // namespace chmt
