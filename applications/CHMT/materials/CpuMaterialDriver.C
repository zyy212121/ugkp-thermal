#include "materials/CpuMaterialDriver.H"
#include "core/ThermalBoundary.H"
#include "materials/MaterialTransport.H"
#include "materials/MaterialDrive.H"
#include "ablation/CpuSurfaceInterface.H"
#include "ablation/WallClosureHost.H"
#include "film/CpuFilmDriver.H"
#include "mesh/Geometry.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "coupling/IntervalAudit.H"
#include "fvCFD.H"
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include <algorithm>
#include <stdexcept>
namespace chmt { namespace {
Foam::pointField foamPoints(const std::vector<Vec3>& points){Foam::pointField result(points.size());for(std::size_t i=0;i<points.size();++i)result[i]=Foam::point(points[i].x,points[i].y,points[i].z);return result;}
bool controlsValid(const CpuMaterialControls& c){return finite(c.maxSubstep)&&c.maxSubstep>=0&&finite(c.transportCfl)&&c.transportCfl>0&&c.transportCfl<=1
    &&finite(c.reactionCfl)&&c.reactionCfl>0&&c.reactionCfl<=1&&finite(c.reactionMaxSubstep)&&c.reactionMaxSubstep>=0
    &&finite(c.driveRelativeTolerance)&&c.driveRelativeTolerance>=0&&finite(c.driveSourceFraction)&&c.driveSourceFraction>0&&c.driveSourceFraction<=1
    &&finite(c.driveTractionScale)&&c.driveTractionScale>0&&c.maxSubsteps>0&&c.nonlinearMaxIterations>0&&finite(c.nonlinearRelativeTolerance)&&c.nonlinearRelativeTolerance>=0
    &&finite(c.energyAbsoluteTolerance)&&c.energyAbsoluteTolerance>=0&&finite(c.energyRelativeTolerance)&&c.energyRelativeTolerance>=0;}
void addBudget(Budget& a,const Budget& b){
    a.boundaryMass+=b.boundaryMass;a.boundaryEnergy+=b.boundaryEnergy;a.boundaryMomentum+=b.boundaryMomentum;a.supportImpulse+=b.supportImpulse;
    a.supportWork+=b.supportWork;a.bodyWork+=b.bodyWork;a.radiation+=b.radiation;a.gclResidual+=b.gclResidual;
    for(int s=0;s<Ns;++s)a.boundarySpecies[s]+=b.boundarySpecies[s];for(int e=0;e<Ne;++e)a.boundaryElements[e]+=b.boundaryElements[e];
    for(int k=0;k<NParticipants;++k){a.exchangeMass[k]+=b.exchangeMass[k];a.exchangeEnergy[k]+=b.exchangeEnergy[k];a.exchangeMomentum[k]+=b.exchangeMomentum[k];for(int s=0;s<Ns;++s)a.exchangeSpecies[k][s]+=b.exchangeSpecies[k][s];}
    a.filmPressureVolume+=b.filmPressureVolume;a.filmKineticAdvection+=b.filmKineticAdvection;a.filmKineticStorage+=b.filmKineticStorage;
    a.filmReducedResidual+=b.filmReducedResidual;a.filmKineticDefect+=b.filmKineticDefect;a.numericalEnergyResidual+=b.numericalEnergyResidual;a.consumedPackets+=b.consumedPackets;
}
void accountCpuPacket(const ExchangePacket& packet,Budget& budget){const unsigned mask=requiredConsumers(packet.kind);const auto d=packetDelta(packet);
    if(mask&ConsumeSolid){const Real sign=packet.kind==ExchangeKind::ParticleWall?1:-1;budget.exchangeMass[SolidParticipant]+=sign*packet.mass;budget.exchangeEnergy[SolidParticipant]+=d.solid.energy;
        budget.exchangeMomentum[SolidParticipant]+=packet.momentum*sign;budget.supportImpulse+=packet.momentum*sign;for(int s=0;s<Ns;++s)budget.exchangeSpecies[SolidParticipant][s]+=sign*packet.species[s];}
    if(mask&ConsumeFilm){const Real sign=packet.kind==ExchangeKind::SolidFilm?1:-1;budget.exchangeMass[FilmParticipant]+=d.film.mass;budget.exchangeEnergy[FilmParticipant]+=sign*packet.energy;
        budget.exchangeMomentum[FilmParticipant]+=packet.momentum*sign;for(int s=0;s<Ns;++s)budget.exchangeSpecies[FilmParticipant][s]+=d.film.species[s];budget.filmKineticAdvection+=packet.liquidKineticAdvection;}
    if(packet.kind==ExchangeKind::SolidFilm)++budget.consumedPackets;
}
bool nativeConduction(Foam::fvMesh& acceptedMesh,const HostState& old,HostState& next,const PhysicsConfig& p,
    Real dt,const CpuMaterialControls& controls,CpuMaterialReport& report,std::string& error){
    const std::size_t n=next.solid.size();if(n==0)return true;
    if(!validateThermalBoundaries(next.solidMesh,error))return false;
    if(acceptedMesh.nCells()!=static_cast<Foam::label>(n)||next.solidMesh.points.size()!=static_cast<std::size_t>(acceptedMesh.nPoints())){error="CPU material fvMesh topology mismatch";return false;}
    // A fresh unregistered full native region avoids movePoints side effects
    // (old volumes, meshPhi, moving flag and Time callbacks) on accepted state.
    Foam::fvMesh mesh(Foam::IOobject(acceptedMesh.name(),acceptedMesh.time().timeName(),acceptedMesh.time(),
        Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),foamPoints(next.solidMesh.points),
        Foam::faceList(acceptedMesh.faces()),Foam::labelList(acceptedMesh.faceOwner()),Foam::labelList(acceptedMesh.faceNeighbour()));
    Foam::List<Foam::polyPatch*> patches(acceptedMesh.boundaryMesh().size());
    forAll(patches,patch)patches[patch]=acceptedMesh.boundaryMesh()[patch].clone(mesh.boundaryMesh()).ptr();
    mesh.addFvPatches(patches);
    for(std::size_t c=0;c<n;++c)if(!closeEnough(mesh.V()[c],next.solidMesh.volumes[c],p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)){
        error="OpenFOAM and conservative material volume disagree";return false;}
    Foam::wordList patchTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch){if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")patchTypes[patch]=mesh.boundary()[patch].type();
        else{bool hasFixed=false,hasFree=false;forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;
            const bool fixed=next.solidMesh.boundaryKind[face]!=BoundaryKind::Interface&&fixedTemperature(next.solidMesh.thermalBoundary.empty()?nullptr:next.solidMesh.thermalBoundary.data(),face);
            hasFixed=hasFixed||fixed;hasFree=hasFree||!fixed;}if(hasFixed&&hasFree){error="mixed per-face material thermal policy within one native patch";return false;}if(hasFixed)patchTypes[patch]=Foam::fixedValueFvPatchScalarField::typeName;}}
    Foam::volScalarField temperature(Foam::IOobject("CHMTMaterialTemperature",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,
        Foam::dimensionedScalar("initial",Foam::dimTemperature,300),patchTypes);
    std::vector<MaterialCaloric> caloric(n);std::vector<Real> anchor(n),target(n);Real targetSum=0;
    for(std::size_t c=0;c<n;++c){if(!materialCaloric(next.solid[c],p,caloric[c],error))return false;MaterialPrimitive primitive;
        if(!recoverMaterial(old.solid[c],old.solidMesh.volumes[c],p,primitive)){error="CPU conduction old EOS recovery failed";return false;}
        anchor[c]=std::max(caloric[c].minimumTemperature,std::min(caloric[c].maximumTemperature,primitive.temperature));temperature[c]=anchor[c];
        target[c]=next.solid[c].energy;targetSum+=target[c];}
    forAll(mesh.boundary(),patch)if(patchTypes[patch]==Foam::fixedValueFvPatchScalarField::typeName)forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;temperature.boundaryFieldRef()[patch][f]=next.solidMesh.boundaryPrimitive[face].temperature;}
    Foam::wordList coefficientTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")coefficientTypes[patch]=mesh.boundary()[patch].type();
    const Foam::dimensionSet capacityDimensions=Foam::dimEnergy/Foam::dimVolume/Foam::dimTemperature;
    const Foam::dimensionSet conductivityDimensions=Foam::dimPower/Foam::dimLength/Foam::dimTemperature;
    const Foam::dimensionedScalar deltaT("materialDt",Foam::dimTime,dt);
    Foam::dictionary solver;solver.add("solver",Foam::word("PCG"));solver.add("preconditioner",Foam::word("DIC"));solver.add("tolerance",Foam::scalar(1e-12));solver.add("relTol",Foam::scalar(0));solver.add("maxIter",Foam::label(2000));
    mesh.schemes().setFluxRequired(temperature.name());
    for(int iteration=0;iteration<controls.nonlinearMaxIterations;++iteration){Foam::scalarField previous(temperature.primitiveField());
        Foam::volScalarField capacity(Foam::IOobject("CHMTMaterialSecant",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",capacityDimensions,0),coefficientTypes);
        Foam::volScalarField conductivity(Foam::IOobject("CHMTMaterialKappa",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",conductivityDimensions,0),coefficientTypes);
        Foam::volScalarField source(Foam::IOobject("CHMTMaterialEnergySource",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",Foam::dimPower/Foam::dimVolume,0));
        for(std::size_t c=0;c<n;++c){const Real cs=caloric[c].secant(temperature[c],anchor[c]);if(!(cs>0)||!finite(cs)){error="nonpositive implicit caloric secant";return false;}
            capacity[c]=cs/mesh.V()[c];source[c]=(target[c]-caloric[c].energy(anchor[c])+cs*anchor[c])/(mesh.V()[c]*dt);
            // Inventories and material EOS belong to the conservative HostMesh
            // volume, already checked against the independently rounded native
            // metric above. Keep native V only in the matrix integration weights
            // so the cell-integral energy equation is unchanged.
            const Real materialVolume=next.solidMesh.volumes[c];
            Real occupied=0,k=0,poreMass=0;for(int j=0;j<Nc;++j){if(next.solid[c].condensed[j]==0)continue;const Real v=next.solid[c].condensed[j]/p.condensed[j].rho;occupied+=v;k+=v*p.condensed[j].conductivity/materialVolume;}
            for(int s=0;s<Ns;++s)poreMass+=next.solid[c].pore[s];Real phi=1-occupied/materialVolume;if(absValue(occupied-materialVolume)<=32*std::numeric_limits<Real>::epsilon()*maxValue(occupied,materialVolume))phi=0;
            if(!finite(phi)||phi<p.material.minimumPorosity||phi>=p.material.maximumPorosity){error="material mass/geometry incompatible during implicit solve";return false;}
            next.solid[c].porosity=phi;conductivity[c]=k+(poreMass>0?phi*p.gasConductivity:0);}
        capacity.correctBoundaryConditions();conductivity.correctBoundaryConditions();temperature.correctBoundaryConditions();
        Foam::surfaceScalarField faceK(Foam::fvc::interpolate(conductivity));
        forAll(mesh.neighbour(),f){const Real a=conductivity[mesh.owner()[f]],b=conductivity[mesh.neighbour()[f]],w=mesh.weights()[f];faceK[f]=a>0&&b>0?1/((1-w)/a+w/b):0;}
        forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()){const Foam::scalarField neighbourK(conductivity.boundaryField()[patch].patchNeighbourField());const Foam::scalarField ownK(conductivity.boundaryField()[patch].patchInternalField());
            forAll(neighbourK,f){const Real a=ownK[f],b=neighbourK[f],w=mesh.weights().boundaryField()[patch][f];faceK.boundaryFieldRef()[patch][f]=a>0&&b>0?1/((1-w)/a+w/b):0;}}
        Foam::fvScalarMatrix equation(Foam::fvm::Sp(capacity/deltaT,temperature)-Foam::fvm::laplacian(faceK,temperature)==source);
        const Foam::solverPerformance linear=equation.solve(solver);++report.linearSolves;++report.nonlinearIterations;
        if(!linear.converged()||linear.singular()||!finite(linear.finalResidual())){error="OpenFOAM implicit material sparse solve failed";return false;}
        temperature.correctBoundaryConditions();const Foam::tmp<Foam::surfaceScalarField> flux=equation.flux();Real boundary=0,sum=0,change=0;long double stableChange=0,inventoryScale=0;
        forAll(flux().boundaryField(),patch)if(!mesh.boundary()[patch].coupled())forAll(flux().boundaryField()[patch],f)boundary+=dt*flux().boundaryField()[patch][f];
        bool admissible=true;for(std::size_t c=0;c<n;++c){const Real T=temperature[c];if(!finite(T)||T<caloric[c].minimumTemperature||T>caloric[c].maximumTemperature)admissible=false;
            sum+=caloric[c].energy(T);stableChange+=static_cast<long double>(caloric[c].secant(T,anchor[c]))*(T-anchor[c])-(target[c]-caloric[c].energy(anchor[c]));
            inventoryScale+=absValue(target[c])+absValue(caloric[c].energy(T));change=maxValue(change,absValue(T-previous[c])/maxValue(1,maxValue(absValue(T),absValue(previous[c]))));}
        const Real residual=static_cast<Real>(stableChange+boundary);const Real storedResidual=sum-targetSum+boundary;const Real scale=maxValue(1,maxValue(absValue(static_cast<Real>(stableChange)),absValue(boundary)));
        const Real roundoff=static_cast<Real>(64*std::numeric_limits<Real>::epsilon()*inventoryScale);
        if(admissible&&change<=controls.nonlinearRelativeTolerance&&absValue(residual)<=controls.energyAbsoluteTolerance+controls.energyRelativeTolerance*scale+roundoff){
            for(std::size_t c=0;c<n;++c){next.solid[c].energy=caloric[c].energy(temperature[c]);MaterialPrimitive recovered;if(!recoverMaterial(next.solid[c],next.solidMesh.volumes[c],p,recovered)){error="implicit material endpoint EOS invalid";return false;}next.solid[c].porosity=recovered.porosity;report.minimumTemperature=minValue(report.minimumTemperature,recovered.temperature);}
            report.energyResidual+=residual;report.budgetDelta.numericalEnergyResidual+=storedResidual;report.budgetDelta.boundaryEnergy+=boundary;return true;}
    }
    error="OpenFOAM material caloric nonlinear iteration failed";return false;
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
bool baseTrace(const HostState& base,const PhysicsConfig& p,std::vector<GasPrimitive>& bulk,std::string& error){bulk.resize(base.surface.area.size());for(std::size_t f=0;f<bulk.size();++f){const int face=base.surface.gasFace[f];if(face<0)continue;if(static_cast<std::size_t>(face)>=base.gasMesh.owner.size()){error="wall predictor face outside gas mesh";return false;}const int c=base.gasMesh.owner[face];if(c<0||static_cast<std::size_t>(c)>=base.gas.size()||!recoverGas(base.gas[c],base.gasMesh.volumes[c],p,bulk[f])){error="wall predictor gas EOS failed";return false;}}return true;}
} // namespace
struct CpuMaterialDriver::Implementation {
    ModelConfig model;Foam::fvMesh* mesh;
    ugkwp::GasMechanismConfiguration mechanism;
    std::unique_ptr<WallClosureHost> wall;bool terminalWallFailure=false;
    Implementation(const ModelConfig& m,Foam::fvMesh* region,const ugkwp::GasMechanismConfiguration* canonical):model(m),mesh(region){
        if(m.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){wall.reset(new WallClosureHost);if(canonical)mechanism=*canonical;}
    }
    bool evaluate(const HostState& state,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,
        const std::vector<GasGradient>& gradients,const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
        Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
        const std::vector<Vec3>* pressureGradient=nullptr,const std::vector<Real>* sideVolume=nullptr,
        const std::vector<GasWallMatchingSample>* recorded=nullptr){
        std::vector<GasWallClosureContext> contexts;terminalWallFailure=false;
        if(wall){
            terminalWallFailure=true;
            auto config=p.wallModel;config.enableSst=p.enableSst;
            if(!wall->prepareGeometry(state.gasMesh,state.surface.gasFace,error,config))return false;
            std::vector<GasWallMatchingSample> fresh;
            if(!recorded){if(!wall->sample(state,p,fresh,error))return false;recorded=&fresh;}
            ugkwp::GasMechanismView<Real,Ns> view;
            if(p.gasMode==ugkwp::GasMode::MixtureChemistry){
                if(mechanism.speciesNames.size()!=Ns||mechanism.mechanismHash!=p.gasMechanismFingerprint){error="CPU wall canonical mechanism identity missing";return false;}
                view=mechanism.mechanismView<Ns>();}
            if(!wall->contexts(*recorded,view,p.wallModel,contexts,error))return false;
        }
        terminalWallFailure=false;
        const bool success=evaluateCpuSurface(state,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,pressureGradient,sideVolume,wall?&contexts:nullptr);
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
    if(data_->wall&&data_->model.physics.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
    if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=false;}
    if(!validateMaterialSweepRemainder(base,error))return false;
    if(!validCouplingInterval(interval)||interval.begin!=base.time){error="wall predictor interval/base time mismatch";return false;}
    if(data_->model.physics.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty()){error="CPU material owner requires actual volume/surface state, not normal columns";return false;}
    std::vector<GasPrimitive> bulk;if(!baseTrace(base,data_->model.physics,bulk,error))return false;CpuSurfaceResult surface;
    if(!data_->evaluate(base,data_->model.physics,bulk,{},{},{},interval.end-interval.begin,interval.identity.sequence,surface,error)){if(diagnostics)diagnostics->recoverable=!data_->terminalWallFailure;return false;}
    WallProgram program;program.interval=interval;program.surface=base.surface;WallKnot first;first.time=interval.begin;first.faces=surface.wall;first.gasPoints=base.gasMesh.points;first.solidPoints=base.solidMesh.points;
    WallKnot last=first;last.time=interval.end;
    if(data_->model.physics.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){
        if(!base.surface.area.empty()){error="prescribed sinusoidal motion with material interface requires a compatible imposed transfer model";return false;}
        const auto& motion=data_->model.physics.meshMotion;const Real temporal=::sin(motion.angularFrequency*(interval.end-motion.timeOrigin));
        if(base.gasMesh.referencePoints.size()!=last.gasPoints.size()){error="prescribed gas motion reference topology missing";return false;}
        for(std::size_t i=0;i<last.gasPoints.size();++i){const Vec3 X=base.gasMesh.referencePoints[i];last.gasPoints[i]={X.x+motion.amplitude.x*::sin(motion.spatialWaveNumber.x*(X.x-motion.spatialOrigin.x))*temporal,
            X.y+motion.amplitude.y*::sin(motion.spatialWaveNumber.y*(X.y-motion.spatialOrigin.y))*temporal,X.z+motion.amplitude.z*::sin(motion.spatialWaveNumber.z*(X.z-motion.spatialOrigin.z))*temporal};}
    }
    if(data_->model.physics.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.area.empty()){auto aux=base.filmAux;std::vector<Real> targets(aux.size());const Real dt=interval.end-interval.begin;
        for(std::size_t f=0;f<aux.size();++f){aux[f].solidNormalVelocity=surface.physics[f].solidSpeed;aux[f].normalVelocity=surface.physics[f].topSpeed;aux[f].solidFront+=dt*aux[f].solidNormalVelocity;aux[f].gasFront+=dt*aux[f].normalVelocity;
            const int sf=base.surface.solidFace[f];targets[f]=sf>=0?dt*aux[f].solidNormalVelocity*mag(base.solidMesh.areaVectors[sf]):0;}
        HostMesh gas,solid;SurfaceMesh moved;SweepConstraintReport report;if(!moveCoupledMeshesConstrained(base,aux,targets,dt,gas,solid,moved,report,error,SweepConstraintControls{},data_->model.physics.tolerances)){if(diagnostics)diagnostics->recoverable=report.status==SweepConstraintStatus::InvalidTrajectory||report.status==SweepConstraintStatus::Nonconverged;return false;}
        last.gasPoints=gas.points;last.solidPoints=solid.points;}
    program.knots={first,last};output=std::move(program);error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram){
    report=CpuMaterialReport{};error.clear();
    if(!validateMaterialSweepRemainder(base,error)){report.recoverable=false;return false;}
    const auto& p=data_->model.physics;const auto& window=history.interval();
    auto wallGeometryConfig=p.wallModel;wallGeometryConfig.enableSst=p.enableSst;
    if(!controlsValid(controls)||!history.complete()||window.begin!=base.time
        ||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty()
        ||(!base.solid.empty()&&!data_->mesh)||base.solidMesh.topologyHash!=geometryEndpoint.solidMesh.topologyHash
        ||base.gasMesh.topologyHash!=geometryEndpoint.gasMesh.topologyHash){
        report.recoverable=false;error="unsupported or inconsistent CPU material interval/configuration/topology";return false;}
    try {
        HostState state=base;state.ledger.clear();state.filmStorage.clear();
        WallProgram program;program.interval=window;program.surface=base.surface;
        CpuMaterialReport accumulated;accumulated.donorPlan.interval=window;
        MaterialReserveKnot initial;initial.time=base.time;initial.cumulativeSolid.resize(base.solid.size());initial.cumulativeFilm.resize(base.film.size());accumulated.donorPlan.knots.push_back(initial);
        accumulated.solidTargetSweeps.assign(base.surface.area.size(),0);
        std::vector<IntervalDonorSum> physicalSweepSum(base.surface.area.size()),actualSweepSum(base.surface.area.size());
        std::vector<SolidQ> gasSolidChange(base.solid.size());std::vector<FilmQ> gasFilmChange(base.film.size());
        auto cursor=history.cursor();const Real interval=window.end-window.begin;
        // The macro endpoint has an actual gas state. Do not substitute the
        // representative last RK stage when converting film endpoint pV.
        std::vector<GasWallMatchingSample> endpointMatching;bool endpointMatchingReady=false;
        while(state.time<window.end){
            if(accumulated.materialSteps>=static_cast<std::uint64_t>(controls.maxSubsteps)){error="CPU material substep limit reached";report=accumulated;return false;}
            Real dt=std::min(window.end-state.time,controls.maxSubstep>0?controls.maxSubstep:interval);
            // Only a real regime change forces alignment with a gas-history knot.
            // Ordinary gas micro boundaries do not determine the material clock.
            const GasIntervalRecord* driveRecord=nullptr;
            for(const auto& r:history.records())if(state.time<r.end){driveRecord=&r;break;}
            if(!driveRecord){error="CPU material driving history exhausted";return false;}
            Real boundedEnd=state.time+dt;
            if(!materialDriveSlabEnd(history,state.time,boundedEnd,controls.driveRelativeTolerance,controls.driveTractionScale,boundedEnd,accumulated.boundaryDriveSamples,error))return false;
            dt=boundedEnd-state.time;
            if(executedProgram)for(const auto& knot:executedProgram->knots)if(knot.time>state.time&&knot.time<state.time+dt)dt=knot.time-state.time;
            std::vector<GasPrimitive> bulk=driveRecord->gasTrace;std::vector<Vec3> traction=driveRecord->gasTraction;
            if(bulk.size()!=state.surface.area.size()||traction.size()!=state.surface.area.size()){
                if(state.surface.area.empty()){bulk.clear();traction.clear();}else{report.recoverable=false;error="CPU material requires actual face-ordered gas drive history";return false;}}
            HostState trial;WallKnot beginning;CpuSurfaceResult surface;MaterialTransportResult transport;
            std::vector<ExchangePacket> packets;std::vector<Real> radiation;std::vector<CpuFilmDrive> drive;
            std::vector<CpuFilmForcing> forcing;std::vector<FilmQ> filmRate;std::vector<FilmRateBudget> filmRateBudget;
            std::vector<Vec3> pressureGradient;std::vector<Real> sideVolume;
            std::vector<Real> solidSweep,gasSweep;IntervalCursor trialCursor=cursor;
            SurfaceMesh filmEvaluationSurface=state.surface;filmEvaluationSurface.sweptEdgeArea.assign(state.surface.edgeOwner.size(),0);
            std::vector<CpuFilmDrive> refinementDrive;Real filmGuessDt=dt;
            bool accepted=false;
            for(int retry=0;retry<=std::max(p.tolerances.maxRetries,p.tolerances.maxCouplingIterations);++retry){
                if(!(dt>0)||state.time+dt==state.time){error="CPU material substep does not advance";break;}
                if(dt!=filmGuessDt){filmEvaluationSurface=state.surface;filmEvaluationSurface.sweptEdgeArea.assign(state.surface.edgeOwner.size(),0);refinementDrive.clear();filmGuessDt=dt;}
                if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static&&retry>0)data_->wall->clearGeometry();
                error.clear();trial=state;trial.time=state.time+dt;trialCursor=cursor;
                if(!trialCursor.takeThrough(trial.time,packets,radiation,error))break;
                ++accumulated.materialRhsEvaluations;
                if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,{},dt,controls.transportCfl,transport,error))break;
                if(dt>transport.dtLimit){dt=transport.dtLimit;continue;}
                gasDrive(state,bulk,traction,drive);if(!refinementDrive.empty())drive=refinementDrive;
                std::vector<GasPrimitive> endpointBulk;std::vector<Vec3> endpointTraction;const GasIntervalRecord* endpointRecord=nullptr;
                if(trial.time==window.end){if(!baseTrace(geometryEndpoint,p,endpointBulk,error))break;}
                else{
                    if(!history.sampleGasDrive(trial.time,endpointBulk,endpointTraction,error))break;
                    for(const auto& r:history.records())if(trial.time<r.end){endpointRecord=&r;break;}
                    if(!endpointRecord){error="missing endpoint gas driving record";break;}
                }
                for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0)drive[f].endpointPressure=endpointBulk[f].pressure;
                if(data_->wall&&p.enableFilm){
                    const auto& matching=driveRecord->gasWallMatching;
                    if(trial.time==window.end&&!endpointMatchingReady){
                        if(!data_->wall->prepareGeometry(geometryEndpoint.gasMesh,geometryEndpoint.surface.gasFace,error,wallGeometryConfig)
                            ||!data_->wall->sample(geometryEndpoint,p,endpointMatching,error)){report.recoverable=false;return false;}
                        endpointMatchingReady=true;
                    }
                    const auto& ending=trial.time==window.end?endpointMatching:endpointRecord->gasWallMatching;
                    if(matching.size()!=state.surface.area.size()||ending.size()!=matching.size()){
                        report.recoverable=false;error="film wall matching pressure history layout mismatch";return false;}
                    for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){
                        drive[f].pressure=matching[f].pressure;drive[f].endpointPressure=ending[f].pressure;
                    }
                }
                forcing.assign(state.film.size(),CpuFilmForcing{});
                for(const auto& packet:packets){if((requiredConsumers(packet.kind)&ConsumeFilm)&&(packet.filmFace<0||static_cast<std::size_t>(packet.filmFace)>=forcing.size())){error="CPU gas history film receiver outside mesh";break;}filmForcing(packet,forcing);}
                if(!error.empty())break;
                if(radiation.size()!=state.surface.area.size()&&!radiation.empty()){error="CPU radiation face layout mismatch";break;}
                for(std::size_t f=0;f<radiation.size();++f)if(driveRecord->radiationReceiver[f]==ConsumeFilm){if(f>=forcing.size()){error="film radiation has no film owner";break;}forcing[f].radiationEnergy+=radiation[f];}
                if(!error.empty())break;
                HostState evaluation=state;evaluation.surface=filmEvaluationSurface;
                if(p.enableFilm){std::vector<FilmAux> profiles;
                    if(!evaluateCpuFilmTransport(evaluation,p,dt,drive,filmRate,filmRateBudget,error,&profiles,&pressureGradient,&sideVolume))break;
                    evaluation.filmAux=profiles;
                    for(std::size_t f=0;f<profiles.size();++f)evaluation.film[f].enthalpy+=pressureVolumeProduct(state.filmAux[f].pressure,state.film[f].mass/p.liquid.rho,profiles[f].pressure,state.film[f].mass/p.liquid.rho);
                }else{filmRate.clear();pressureGradient.assign(state.surface.area.size(),Vec3{});sideVolume.assign(state.surface.area.size(),0);}
                if(!data_->evaluate(evaluation,p,bulk,driveRecord->gasGradient,transport.rates,filmRate,dt,window.identity.sequence,surface,error,&pressureGradient,&sideVolume,&driveRecord->gasWallMatching)){if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                bool shortenedForSource=false;
                for(const auto& record:history.records())if(record.begin>state.time&&record.begin<trial.time){
                    CpuSurfaceResult response;++accumulated.boundaryDriveSamples;
                    if(!data_->evaluate(evaluation,p,record.gasTrace,record.gasGradient,transport.rates,filmRate,dt,window.identity.sequence,response,error,&pressureGradient,&sideVolume,&record.gasWallMatching)){if(data_->terminalWallFailure){report.recoverable=false;return false;}dt=record.begin-state.time;error.clear();shortenedForSource=true;break;}
                    Real fraction=0;
                    for(std::size_t f=0;f<surface.physics.size();++f){if(state.surface.gasFace[f]<0)continue;
                        const auto& a=surface.physics[f];const auto& b=response.physics[f];const Real scale=state.surface.area[f]*dt;
                        const int cell=state.surface.solidCell[f];Real solidThermalScale=0,solidDonor=0,phaseDonor=0,solidSpecific=0,filmThermalScale=0,filmSpecific=0;
                        if(cell>=0){MaterialCaloric caloric;MaterialPrimitive primitive;if(!materialCaloric(state.solid[cell],p,caloric,error)||!recoverMaterial(state.solid[cell],state.solidMesh.volumes[cell],p,primitive))return false;
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
                    if(fraction>controls.driveSourceFraction){dt=record.begin-state.time;shortenedForSource=true;break;}
                }
                if(shortenedForSource)continue;
                // Preserve actual gas-facing source energy. Only the CPU-owned
                // internal phase packet is taken from the constitutive proposal.
                for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){const auto& physics=surface.physics[f];
                    drive[f].hasCoupledNormalTrace=true;drive[f].bottomNormalVelocity=physics.liquidBottomNormal;drive[f].topNormalVelocity=physics.liquidTopNormal;
                    drive[f].solidNormalVelocity=physics.solidSpeed;drive[f].interfaceNormalVelocity=physics.topSpeed;}
                for(auto& phase:surface.phasePackets){const std::size_t f=phase.filmFace;
                    const Vec3 velocity=state.surface.baseVelocity.empty()?Vec3{}:state.surface.baseVelocity[f];
                    const Vec3 n=state.surface.normal[f];const Vec3 bottom=velocity-n*dot(velocity,n)+n*surface.physics[f].liquidBottomNormal;
                    bool exactEvent=false;
                    if(!finalizeCpuFilmPhasePacket(state.film[f],state.filmAux[f].pressure,filmRate[f],forcing[f],dt,bottom,p,phase,exactEvent,error))break;
                    forcing[f].exactPhaseEvent=forcing[f].exactPhaseEvent||exactEvent;
                    if(!error.empty())break;filmForcing(phase,forcing);}
                if(!error.empty())break;
                if(p.enableFilm)for(std::size_t f=0;f<forcing.size();++f){const Real remaining=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                    const Real scale=absValue(state.film[f].mass)+absValue(dt*filmRate[f].mass)+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass);
                    if((forcing[f].mass<0||forcing[f].phaseMass<0)&&absValue(remaining)<=16*std::numeric_limits<Real>::epsilon()*scale)forcing[f].exactPhaseEvent=true;
                }
                if(p.enableFilm){Real filmDt=dt;if(!estimateCpuFilmStep(evaluation,p,forcing,drive,dt,dt,filmDt,error))break;
                    if(filmDt<dt*(1-32*std::numeric_limits<Real>::epsilon())){dt=filmDt;continue;}}
                std::vector<Real> target(state.surface.area.size(),0);
                for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasSolid){for(std::size_t f=0;f<target.size();++f)if(state.surface.gasFace[f]>=0&&state.gasMesh.faceIds[state.surface.gasFace[f]]==packet.face){
                    MaterialPrimitive material;if(!recoverMaterial(state.solid[packet.solidCell],state.solidMesh.volumes[packet.solidCell],p,material)){error="surface donor EOS failed";break;}
                    for(int c=0;c<Nc;++c)if(packet.condensed[c]!=0)target[f]-=packet.condensed[c]/p.condensed[c].rho/(1-material.porosity);}}
                for(const auto& phase:surface.phasePackets){MaterialPrimitive material;const int c=phase.solidCell;if(!recoverMaterial(state.solid[c],state.solidMesh.volumes[c],p,material)){error="phase donor EOS failed";break;}
                    target[phase.filmFace]-=phase.mass/p.condensed[p.material.phaseCondensed].rho/(1-material.porosity);}
                if(!error.empty())break;
                std::vector<FilmAux> aux=evaluation.filmAux;
                for(std::size_t f=0;f<aux.size();++f)if(state.surface.gasFace[f]>=0){const int sf=state.surface.solidFace[f],gf=state.surface.gasFace[f];const Real as=sf>=0?mag(state.solidMesh.areaVectors[sf]):0,ag=mag(state.gasMesh.areaVectors[gf]);
                    aux[f].solidNormalVelocity=as>0?target[f]/(as*dt):0;
                    const Real mass=p.enableFilm?dt*filmRate[f].mass+forcing[f].mass+forcing[f].phaseMass:0;
                    const Real side=p.enableFilm?sideVolume[f]:0;
                    if(p.enableFilm){Real endpointMass=state.film[f].mass+mass;const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(mass));if(absValue(endpointMass)<=roundoff&&forcing[f].exactPhaseEvent)endpointMass=0;if(endpointMass<0||!finite(endpointMass)){error="negative film mass in geometric proposal";break;}
                        if(!filmThicknessFromMass(endpointMass,evaluation.surface.area[f],p.liquid.rho,aux[f].thickness)){error="invalid endpoint film thickness";break;}aux[f].area=evaluation.surface.area[f];}
                    aux[f].normalVelocity=p.enableFilm&&surface.physics[f].wet?(target[f]/dt+mass/(dt*p.liquid.rho)-side)/ag:aux[f].solidNormalVelocity;
                    aux[f].solidFront=state.filmAux[f].solidFront+dt*aux[f].solidNormalVelocity;aux[f].gasFront=state.filmAux[f].gasFront+dt*aux[f].normalVelocity;}
                if(!error.empty())break;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!target.empty()){SweepConstraintReport motion;
                    if(!moveCoupledMeshesConstrained(state,aux,target,dt,trial.gasMesh,trial.solidMesh,trial.surface,motion,error,SweepConstraintControls{},p.tolerances)){
                        if(motion.status==SweepConstraintStatus::Incompatible||motion.status==SweepConstraintStatus::SizeLimit||motion.status==SweepConstraintStatus::InvalidInput){report.recoverable=false;return false;}break;}
                }else{
                    const Real fraction=(trial.time-window.begin)/interval;std::vector<Vec3> gp=base.gasMesh.points,sp=base.solidMesh.points;
                    if(gp.size()!=geometryEndpoint.gasMesh.points.size()||sp.size()!=geometryEndpoint.solidMesh.points.size()){error="material trajectory topology mismatch";break;}
                    for(std::size_t i=0;i<gp.size();++i)gp[i]+=(geometryEndpoint.gasMesh.points[i]-gp[i])*fraction;
                    for(std::size_t i=0;i<sp.size();++i)sp[i]+=(geometryEndpoint.solidMesh.points[i]-sp[i])*fraction;
                    if(executedProgram){WallKnot executed;if(!intervalSampleWall(*executedProgram,trial.time,executed,error))break;gp=executed.gasPoints;sp=executed.solidPoints;}
                    if(!makeStageGeometry(state.gasMesh,gp,dt,trial.gasMesh,gasSweep,error,p.tolerances)||!makeStageGeometry(state.solidMesh,sp,dt,trial.solidMesh,solidSweep,error,p.tolerances))break;
                    if(!state.surface.area.empty()&&!rebuildTrajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,error,p.tolerances))break;
                }
                if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){WallKnot executed;
                    if(!intervalSampleWall(*executedProgram,trial.time,executed,error))break;
                    HostMesh constrainedGas=trial.gasMesh,constrainedSolid=trial.solidMesh;SurfaceMesh constrainedSurface=trial.surface;
                    const bool close=materialPointsClose(trial.gasMesh,executed.gasPoints,p.tolerances)&&materialPointsClose(trial.solidMesh,executed.solidPoints,p.tolerances);
                    if(close){if(!makeStageGeometry(state.gasMesh,executed.gasPoints,dt,trial.gasMesh,gasSweep,error,p.tolerances)||!makeStageGeometry(state.solidMesh,executed.solidPoints,dt,trial.solidMesh,solidSweep,error,p.tolerances)
                        ||!rebuildTrajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,error,p.tolerances))break;
                        std::vector<Real> solveTarget;if(!compensateMaterialSweepTargets(state,target,solveTarget,error))break;
                        const bool compatible=materialSweepsCompatible(state.solidMesh,state.surface,solveTarget,solidSweep,p.tolerances);
                        if(!compatible){trial.gasMesh=std::move(constrainedGas);trial.solidMesh=std::move(constrainedSolid);trial.surface=std::move(constrainedSurface);}
                    }
                }
                if(!makeStageGeometry(state.solidMesh,trial.solidMesh.points,dt,trial.solidMesh,solidSweep,error,p.tolerances)
                    ||!makeStageGeometry(state.gasMesh,trial.gasMesh.points,dt,trial.gasMesh,gasSweep,error,p.tolerances))break;
                if(p.enableFilm){HostState consistentFilm=state;consistentFilm.surface=trial.surface;std::vector<FilmQ> actualRates;std::vector<FilmRateBudget> actualBudgets;
                    if(!evaluateCpuFilmTransport(consistentFilm,p,dt,drive,actualRates,actualBudgets,error))break;
                    bool same=actualRates.size()==filmRate.size();for(std::size_t f=0;same&&f<filmRate.size();++f){same=actualRates[f].mass==filmRate[f].mass&&actualRates[f].enthalpy==filmRate[f].enthalpy;
                        for(int species=0;species<Ns;++species)same=same&&actualRates[f].species[species]==filmRate[f].species[species];}
                    if(!same){filmEvaluationSurface=trial.surface;refinementDrive=drive;continue;}
                }
                if(p.enableFilm&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){
                    for(std::size_t f=0;f<state.film.size();++f)if(state.surface.gasFace[f]>=0){
                        const int gf=state.surface.gasFace[f],sf=state.surface.solidFace[f];
                        Real updatedMass=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                        const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(dt*filmRate[f].mass)+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass));
                        if(forcing[f].exactPhaseEvent&&absValue(updatedMass)<=roundoff)updatedMass=0;
                        if(!filmSweepsCompatible(state.film[f].mass,updatedMass,p.liquid.rho,gasSweep[gf],sf>=0?solidSweep[sf]:0,dt*sideVolume[f],p.tolerances)){
                            error="actual gas/solid/side swept volume does not match updated film mass";break;}
                    }
                    if(!error.empty())break;
                }
                ++accumulated.materialRhsEvaluations;
                if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,solidSweep,dt,controls.transportCfl,transport,error))break;
                if(dt>transport.dtLimit){dt=transport.dtLimit;continue;}
                CpuMaterialReport stepReport;
                bool reactionOk=true;for(std::size_t c=0;c<state.solid.size();++c){std::uint64_t count=0;SolidQ reacted;
                    if(!advanceMaterialReactions(state.solid[c],state.solidMesh.volumes[c],p,dt,controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps,reacted,count,error)){reactionOk=false;break;}
                    trial.solid[c]=addSolid(reacted,transport.rates[c],dt);stepReport.reactionSteps+=count;}
                if(!reactionOk){dt*=.5;error.clear();continue;}
                std::vector<ExchangePacket> all=packets;all.insert(all.end(),surface.phasePackets.begin(),surface.phasePackets.end());
                Real packetRoundoff=0;std::vector<SolidQ> applied;
                if(!applyMaterialPacketBatch(trial.solid,all,applied,packetRoundoff,error)){dt*=.5;error.clear();continue;}
                trial.solid=std::move(applied);stepReport.numericalMassRoundoff+=packetRoundoff;
                for(const auto& packet:all)accountCpuPacket(packet,stepReport.budgetDelta);
                if(!error.empty())break;
                for(std::size_t f=0;f<radiation.size();++f){stepReport.budgetDelta.radiation+=radiation[f];if(driveRecord->radiationReceiver[f]==ConsumeSolid){const int c=state.surface.solidCell[f];if(c<0){error="radiation has no material owner";break;}trial.solid[c].energy+=radiation[f];}}
                if(!error.empty())break;
                bool materialAdmissible=true;
                for(std::size_t c=0;c<trial.solid.size();++c){MaterialPrimitive material;
                    if(!validSolidInventory(trial.solid[c])||(trial.time<window.end&&!recoverMaterial(trial.solid[c],trial.solidMesh.volumes[c],p,material))){materialAdmissible=false;break;}
                    if(trial.time<window.end)trial.solid[c].porosity=material.porosity;
                }
                if(!materialAdmissible){dt*=.5;error.clear();continue;}
                if(p.enableFilm){HostState filmState=state;filmState.surface=trial.surface;CpuFilmCandidate film;
                    if(!advanceCpuFilmCandidate(filmState,p,dt,forcing,drive,film,error)){dt*=.5;error.clear();continue;}
                    trial.film=std::move(film.film);trial.filmAux=std::move(film.filmAux);Budget filmBudget=film.report.budgetDelta;
                    for(int k=0;k<NParticipants;++k){filmBudget.exchangeMass[k]=0;filmBudget.exchangeEnergy[k]=0;filmBudget.exchangeMomentum[k]=Vec3{};for(int s=0;s<Ns;++s)filmBudget.exchangeSpecies[k][s]=0;}
                    filmBudget.filmKineticAdvection=0;stepReport.numericalMassRoundoff+=film.report.numericalMassRoundoff;addBudget(stepReport.budgetDelta,filmBudget);stepReport.filmSteps=1;
                    // Surface radiation is accounted by the interval owner above.
                    Real filmRadiation=0;for(const auto& f:forcing)filmRadiation+=f.radiationEnergy;stepReport.budgetDelta.radiation-=filmRadiation;
                    for(std::size_t f=0;f<trial.film.size();++f){trial.filmAux[f].solidFront=aux[f].solidFront;trial.filmAux[f].gasFront=aux[f].gasFront;trial.filmAux[f].solidNormalVelocity=aux[f].solidNormalVelocity;trial.filmAux[f].normalVelocity=aux[f].normalVelocity;
                        if((state.film[f].mass==0)!=(trial.film[f].mass==0)&&trial.time<window.end){report=accumulated;report.eventTime=trial.time;error="film phase event requires an exact macro boundary";return false;}}}
                else trial.filmAux=aux;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession
                    &&!updateMaterialSweepRemainder(state,target,solidSweep,trial.solidMesh,trial.solidSweepRemainder,error))break;
                for(std::size_t f=0;f<target.size();++f){
                    physicalSweepSum[f].add(target[f]);accumulated.solidTargetSweeps[f]=static_cast<Real>(physicalSweepSum[f].value());
                    const int face=state.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(solidSweep[face]);
                }
                stepReport.budgetDelta.bodyWork+=dt*transport.bodyPower;
                stepReport.budgetDelta.gclResidual+=maxValue(maximumGclResidual(state.gasMesh,trial.gasMesh,gasSweep),maximumGclResidual(state.solidMesh,trial.solidMesh,solidSweep));
                for(std::size_t f=0;f<state.solidMesh.owner.size();++f){const auto kind=state.solidMesh.boundaryKind[f];if(kind==BoundaryKind::Internal||kind==BoundaryKind::Periodic||kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;const auto& flux=transport.faceFlux[f];
                    stepReport.budgetDelta.boundaryEnergy+=dt*flux.energy;for(int c=0;c<Nc;++c){stepReport.budgetDelta.boundaryMass+=dt*flux.condensed[c];for(int e=0;e<p.nElements;++e)stepReport.budgetDelta.boundaryElements[e]+=dt*flux.condensed[c]*p.condensed[c].element[e];}
                    for(int s=0;s<Ns;++s){stepReport.budgetDelta.boundaryMass+=dt*flux.pore[s];stepReport.budgetDelta.boundarySpecies[s]+=dt*flux.pore[s];for(int e=0;e<p.nElements;++e)stepReport.budgetDelta.boundaryElements[e]+=dt*flux.pore[s]*p.species[s].element[e];}}
                beginning.time=state.time;beginning.faces=surface.wall;beginning.gasPoints=state.gasMesh.points;beginning.solidPoints=state.solidMesh.points;
                if(program.knots.empty())program.knots.push_back(beginning);
                WallKnot ending=beginning;ending.time=trial.time;ending.gasPoints=trial.gasMesh.points;ending.solidPoints=trial.solidMesh.points;
                if(trial.time<window.end){CpuSurfaceResult endpointProposal;
                    if(!endpointRecord){error="missing endpoint wall matching record";break;}if(!data_->evaluate(trial,p,endpointBulk,endpointRecord->gasGradient,{},{},dt,window.identity.sequence,endpointProposal,error,nullptr,nullptr,&endpointRecord->gasWallMatching)){if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                    ending.faces=endpointProposal.wall;for(std::size_t f=0;f<ending.faces.size();++f)ending.faces[f].primaryKind=beginning.faces[f].primaryKind;}
                for(std::size_t f=0;f<ending.faces.size();++f){ending.faces[f].solidNormalVelocity=aux[f].solidNormalVelocity;ending.faces[f].normalVelocity=aux[f].normalVelocity;}
                program.knots.push_back(ending);
                for(const auto& packet:packets){const auto d=packetDelta(packet);if(requiredConsumers(packet.kind)&ConsumeSolid)gasSolidChange[packet.solidCell]=addSolid(gasSolidChange[packet.solidCell],d.solid);
                    if(requiredConsumers(packet.kind)&ConsumeFilm){auto& q=gasFilmChange[packet.filmFace];q.mass+=d.film.mass;for(int s=0;s<Ns;++s)q.species[s]+=d.film.species[s];}}
                MaterialReserveKnot knot;knot.time=trial.time;knot.cumulativeSolid.resize(trial.solid.size());knot.cumulativeFilm.resize(trial.film.size());
                for(std::size_t c=0;c<trial.solid.size();++c)knot.cumulativeSolid[c]=addSolid(addSolid(trial.solid[c],base.solid[c],-1),gasSolidChange[c],-1);
                for(std::size_t f=0;f<trial.film.size();++f){auto& q=knot.cumulativeFilm[f];q.mass=trial.film[f].mass-base.film[f].mass-gasFilmChange[f].mass;for(int s=0;s<Ns;++s)q.species[s]=trial.film[f].species[s]-base.film[f].species[s]-gasFilmChange[f].species[s];}
                accumulated.donorPlan.knots.push_back(std::move(knot));
                for(auto phase:surface.phasePackets){phase.consumerMask=requiredConsumers(phase.kind);trial.ledger.push_back(phase);}
                accumulated.numericalMassRoundoff+=stepReport.numericalMassRoundoff;accumulated.materialSteps++;accumulated.reactionSteps+=stepReport.reactionSteps;accumulated.filmSteps+=stepReport.filmSteps;accumulated.linearSolves+=stepReport.linearSolves;
                accumulated.nonlinearIterations+=stepReport.nonlinearIterations;accumulated.energyResidual+=stepReport.energyResidual;accumulated.minimumTemperature=minValue(accumulated.minimumTemperature,stepReport.minimumTemperature);addBudget(accumulated.budgetDelta,stepReport.budgetDelta);
                state=std::move(trial);cursor=trialCursor;accepted=true;break;
            }
            if(!accepted){if(error.empty())error="CPU material local retries exhausted";report=accumulated;return false;}
        }
        // Lie/IMEX splitting: all conservative local sources are now in Utarget.
        // This is one true 3-D backward-Euler conduction candidate per window,
        // regardless of gas microstep or local reaction/film step count.
        if(!state.solid.empty()&&!nativeConduction(*data_->mesh,base,state,p,interval,controls,accumulated,error)){report=accumulated;return false;}
        for(const auto& r:history.records())for(const auto& packet:r.packets)if(requiredConsumers(packet.kind)&(ConsumeSolid|ConsumeFilm))++accumulated.budgetDelta.consumedPackets;
        state.filmStorage.resize(state.film.size());for(std::size_t f=0;f<state.film.size();++f){auto& storage=state.filmStorage[f];storage.step=window.identity.sequence;storage.stage=1;storage.geometry=state.gasMesh.geometryVersion;storage.face=state.surface.persistentId[f];storage.filmFace=f;storage.oldPV=base.filmAux[f].pressure*base.film[f].mass/p.liquid.rho;storage.newPV=state.filmAux[f].pressure*state.film[f].mass/p.liquid.rho;storage.consumed=true;}
        const auto samePoints=[](const HostMesh& a,const HostMesh& b){if(a.points.size()!=b.points.size())return false;for(std::size_t i=0;i<a.points.size();++i)if(a.points[i].x!=b.points[i].x||a.points[i].y!=b.points[i].y||a.points[i].z!=b.points[i].z)return false;return true;};
        if(samePoints(state.gasMesh,geometryEndpoint.gasMesh)&&samePoints(state.solidMesh,geometryEndpoint.solidMesh)){
            state.gasMesh=geometryEndpoint.gasMesh;state.solidMesh=geometryEndpoint.solidMesh;state.gasStages=geometryEndpoint.gasStages;state.solidStages=geometryEndpoint.solidStages;
            if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
                // Re-certify the executed piecewise trajectory, not an invented
                // straight macro chord. Do not increment a provisional carry a
                // second time when adopting the gas owner's endpoint metadata.
                if(executedProgram->knots.size()<2||executedProgram->knots.front().time!=window.begin
                    ||executedProgram->knots.back().time!=window.end){error="executed material remainder trajectory interval mismatch";return false;}
                const auto& firstPoints=executedProgram->knots.front().solidPoints;
                if(firstPoints.size()!=base.solidMesh.points.size()){error="executed material remainder base topology mismatch";return false;}
                for(std::size_t i=0;i<firstPoints.size();++i)if(firstPoints[i].x!=base.solidMesh.points[i].x
                    ||firstPoints[i].y!=base.solidMesh.points[i].y||firstPoints[i].z!=base.solidMesh.points[i].z){error="executed material remainder base geometry mismatch";return false;}
                HostMesh path=base.solidMesh;actualSweepSum.assign(base.surface.area.size(),IntervalDonorSum{});
                for(std::size_t k=1;k<executedProgram->knots.size();++k){HostMesh endpoint;std::vector<Real> measured;
                    const Real duration=executedProgram->knots[k].time-executedProgram->knots[k-1].time;
                    if(!makeStageGeometry(path,executedProgram->knots[k].solidPoints,duration,endpoint,measured,error,p.tolerances))return false;
                    for(std::size_t f=0;f<actualSweepSum.size();++f){const int face=base.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(measured[face]);}
                    path=std::move(endpoint);
                }
                if(!samePoints(path,state.solidMesh)){error="executed material remainder endpoint mismatch";return false;}
            }
        }
        if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
            state.solidSweepRemainder.resize(base.surface.solidFace.size());
            for(std::size_t f=0;f<state.solidSweepRemainder.size();++f)state.solidSweepRemainder[f]=static_cast<Real>(
                (base.solidSweepRemainder.empty()?0:static_cast<long double>(base.solidSweepRemainder[f]))
                +physicalSweepSum[f].value()-actualSweepSum[f].value());
            if(!validateMaterialSweepRemainder(state,error)){report=accumulated;return false;}
        }
        state.gasChemistryAudit=geometryEndpoint.gasChemistryAudit;state.gasSstAudit=geometryEndpoint.gasSstAudit;
        state.gasWallDiagnostics=geometryEndpoint.gasWallDiagnostics;state.gasWallDiagnosticTime=geometryEndpoint.gasWallDiagnosticTime;
        state.gas=geometryEndpoint.gas;state.sst=geometryEndpoint.sst;state.particles=geometryEndpoint.particles;
        state.rejectedSteps=geometryEndpoint.rejectedSteps;state.gasVoidFraction=geometryEndpoint.gasVoidFraction;state.gasStages=geometryEndpoint.gasStages;
        state.gasMesh.boundaryPrimitive=geometryEndpoint.gasMesh.boundaryPrimitive;state.gasMesh.thermalBoundary=geometryEndpoint.gasMesh.thermalBoundary;state.gasMesh.boundarySst=geometryEndpoint.gasMesh.boundarySst;
        state.budget=geometryEndpoint.budget;addBudget(state.budget,accumulated.budgetDelta);state.time=window.end;program.donorPlan=accumulated.donorPlan;
        if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
        if(!program.knots.empty()){const auto& last=history.records().back();CpuSurfaceResult endpointSurface;
            std::vector<GasPrimitive> finalBulk;if(!baseTrace(state,p,finalBulk,error)){report=accumulated;return false;}
            if(!data_->evaluate(state,p,finalBulk,last.gasGradient,{},{},interval,window.identity.sequence,endpointSurface,error)){report=accumulated;report.recoverable=!data_->terminalWallFailure;return false;}
            program.knots.back().faces=endpointSurface.wall;
            // The executed regime owns the interval including its terminal flux.
            // An endpoint phase event changes the next window's predictor only.
            for(std::size_t f=0;f<program.knots.back().faces.size();++f)program.knots.back().faces[f].primaryKind=program.knots.front().faces[f].primaryKind;
        }
        for(std::size_t i=0;i<state.gasMesh.points.size();++i)if(mag(state.gasMesh.points[i]-geometryEndpoint.gasMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        for(std::size_t i=0;i<state.solidMesh.points.size();++i)if(mag(state.solidMesh.points[i]-geometryEndpoint.solidMesh.points[i])>p.tolerances.absoluteGeometry)accumulated.geometryCorrection=true;
        output=std::move(state);corrected=std::move(program);report=std::move(accumulated);error.clear();return true;
    }catch(const Foam::error& e){error="OpenFOAM CPU material failure: "+std::string(e.what());return false;}
    catch(const std::exception& e){error="CPU material failure: "+std::string(e.what());return false;}
}
} // namespace chmt
