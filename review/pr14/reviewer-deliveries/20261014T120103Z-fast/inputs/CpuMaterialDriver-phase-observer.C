#include <cstdio>
#include "materials/CpuMaterialDriver.H"
#include "core/ThermalBoundary.H"
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
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include <algorithm>
#include <stdexcept>
namespace chmt { namespace {
Foam::pointField foamPoints(const std::vector<Vec3>& points){Foam::pointField result(points.size());for(std::size_t i=0;i<points.size();++i)result[i]=Foam::point(points[i].x,points[i].y,points[i].z);return result;}
bool controlsValid(const CpuMaterialControls& c,bool terminalMaterial=false){return terminalMaterial||(finite(c.maxSubstep)&&c.maxSubstep>=0&&(terminalMaterial||(finite(c.transportCfl)&&c.transportCfl>0&&c.transportCfl<=1))
    &&(terminalMaterial||(finite(c.reactionCfl)&&c.reactionCfl>0&&c.reactionCfl<=1&&finite(c.reactionMaxSubstep)&&c.reactionMaxSubstep>=0))
    &&finite(c.driveRelativeTolerance)&&c.driveRelativeTolerance>=0&&finite(c.driveSourceFraction)&&c.driveSourceFraction>0&&c.driveSourceFraction<=1
    &&finite(c.driveTractionScale)&&c.driveTractionScale>0&&(terminalMaterial||c.maxSubsteps>0)&&(terminalMaterial||(c.nonlinearMaxIterations>0&&finite(c.nonlinearRelativeTolerance)&&c.nonlinearRelativeTolerance>=0
    &&finite(c.energyAbsoluteTolerance)&&c.energyAbsoluteTolerance>=0&&finite(c.energyRelativeTolerance)&&c.energyRelativeTolerance>=0)));}
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
template<bool Terminal=false>
bool nativeConduction(Foam::fvMesh& acceptedMesh,const HostState& old,HostState& next,const PhysicsConfig& p,
    Real dt,const CpuMaterialControls& controls,CpuMaterialReport& report,std::string& error){
    const std::size_t n=next.solid.size();if(n==0)return true;
    if(!Terminal&&!validateThermalBoundaries(next.solidMesh,error)){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
    if(!Terminal&&(acceptedMesh.nCells()!=static_cast<Foam::label>(n)||next.solidMesh.points.size()!=static_cast<std::size_t>(acceptedMesh.nPoints()))){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;error="CPU material fvMesh topology mismatch";return false;}
    // A fresh unregistered full native region avoids movePoints side effects
    // (old volumes, meshPhi, moving flag and Time callbacks) on accepted state.
    Foam::fvMesh mesh(Foam::IOobject(acceptedMesh.name(),acceptedMesh.time().timeName(),acceptedMesh.time(),
        Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),foamPoints(next.solidMesh.points),
        Foam::faceList(acceptedMesh.faces()),Foam::labelList(acceptedMesh.faceOwner()),Foam::labelList(acceptedMesh.faceNeighbour()));
    Foam::List<Foam::polyPatch*> patches(acceptedMesh.boundaryMesh().size());
    forAll(patches,patch)patches[patch]=acceptedMesh.boundaryMesh()[patch].clone(mesh.boundaryMesh()).ptr();
    mesh.addFvPatches(patches);
    if(!Terminal)for(std::size_t c=0;c<n;++c)if(!closeEnough(mesh.V()[c],next.solidMesh.volumes[c],p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)){
        report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;
        error="OpenFOAM and conservative material volume disagree";return false;}
    Foam::wordList patchTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch){if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")patchTypes[patch]=mesh.boundary()[patch].type();
        else{bool hasFixed=false,hasFree=false;forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;
            if(Terminal&&(static_cast<std::size_t>(face)>=next.solidMesh.boundaryKind.size()||(next.solidMesh.boundaryKind[face]!=BoundaryKind::Interface&&!next.solidMesh.thermalBoundary.empty()&&static_cast<std::size_t>(face)>=next.solidMesh.thermalBoundary.size()))){error="CPU conduction consumed boundary policy missing";return false;}
            const bool fixed=next.solidMesh.boundaryKind[face]!=BoundaryKind::Interface&&fixedTemperature(next.solidMesh.thermalBoundary.empty()?nullptr:next.solidMesh.thermalBoundary.data(),face);
            hasFixed=hasFixed||fixed;hasFree=hasFree||!fixed;}if(!Terminal&&hasFixed&&hasFree){report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;error="mixed per-face material thermal policy within one native patch";return false;}if(hasFixed)patchTypes[patch]=Foam::fixedValueFvPatchScalarField::typeName;}}
    Foam::volScalarField temperature(Foam::IOobject("CHMTMaterialTemperature",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,
        Foam::dimensionedScalar("initial",Foam::dimTemperature,300),patchTypes);
    std::vector<MaterialCaloric> caloric(n);std::vector<Real> anchor(n),target(n);Real targetSum=0;
    for(std::size_t c=0;c<n;++c){bool numerical=false;if(!materialCaloric<Terminal>(next.solid[c],p,caloric[c],error,&numerical)){if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}MaterialPrimitive primitive;
        if(!recoverMaterial<Terminal>(old.solid[c],old.solidMesh.volumes[c],p,primitive,nullptr,-1,&numerical)){error="CPU conduction old EOS recovery failed";if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
        anchor[c]=std::max(caloric[c].minimumTemperature,std::min(caloric[c].maximumTemperature,primitive.temperature));temperature[c]=anchor[c];
        target[c]=next.solid[c].energy;targetSum+=target[c];}
    forAll(mesh.boundary(),patch)if(patchTypes[patch]==Foam::fixedValueFvPatchScalarField::typeName)forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;if(Terminal&&static_cast<std::size_t>(face)>=next.solidMesh.boundaryPrimitive.size()){error="CPU conduction consumed boundary temperature missing";return false;}temperature.boundaryFieldRef()[patch][f]=next.solidMesh.boundaryPrimitive[face].temperature;}
    Foam::wordList coefficientTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")coefficientTypes[patch]=mesh.boundary()[patch].type();
    const Foam::dimensionSet capacityDimensions=Foam::dimEnergy/Foam::dimVolume/Foam::dimTemperature;
    const Foam::dimensionSet conductivityDimensions=Foam::dimPower/Foam::dimLength/Foam::dimTemperature;
    const Foam::dimensionedScalar deltaT("materialDt",Foam::dimTime,dt);
    Foam::dictionary solver;solver.add("solver",Foam::word("PCG"));solver.add("preconditioner",Foam::word("DIC"));solver.add("tolerance",Foam::scalar(1e-12));solver.add("relTol",Foam::scalar(0));solver.add("maxIter",Foam::label(2000));
    mesh.schemes().setFluxRequired(temperature.name());
    for(int iteration=0;iteration<(Terminal?std::max(1,controls.nonlinearMaxIterations):controls.nonlinearMaxIterations);++iteration){Foam::scalarField previous(temperature.primitiveField());
        Foam::volScalarField capacity(Foam::IOobject("CHMTMaterialSecant",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",capacityDimensions,0),coefficientTypes);
        Foam::volScalarField conductivity(Foam::IOobject("CHMTMaterialKappa",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",conductivityDimensions,0),coefficientTypes);
        Foam::volScalarField source(Foam::IOobject("CHMTMaterialEnergySource",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",Foam::dimPower/Foam::dimVolume,0));
        for(std::size_t c=0;c<n;++c){const Real cs=caloric[c].secant<Terminal>(temperature[c],anchor[c]);if(!Terminal&&(!(cs>0)||!finite(cs))){error="nonpositive implicit caloric secant";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            capacity[c]=cs/mesh.V()[c];source[c]=(target[c]-caloric[c].energy<Terminal>(anchor[c])+cs*anchor[c])/(mesh.V()[c]*dt);
            // Inventories and material EOS belong to the conservative HostMesh
            // volume, already checked against the independently rounded native
            // metric above. Keep native V only in the matrix integration weights
            // so the cell-integral energy equation is unchanged.
            const Real materialVolume=next.solidMesh.volumes[c];
            Real occupied=0,k=0,poreMass=0;for(int j=0;j<Nc;++j){if(next.solid[c].condensed[j]==0)continue;const Real v=next.solid[c].condensed[j]/p.condensed[j].rho;occupied+=v;k+=v*p.condensed[j].conductivity/materialVolume;}
            for(int s=0;s<Ns;++s)poreMass+=next.solid[c].pore[s];Real phi=1-occupied/materialVolume;if(absValue(occupied-materialVolume)<=32*std::numeric_limits<Real>::epsilon()*maxValue(occupied,materialVolume))phi=0;
            if(!Terminal&&(!finite(phi)||phi<p.material.minimumPorosity||phi>=p.material.maximumPorosity)){error="material mass/geometry incompatible during implicit solve";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
            next.solid[c].porosity=phi;conductivity[c]=k+(poreMass>0?phi*p.gasConductivity:0);}
        capacity.correctBoundaryConditions();conductivity.correctBoundaryConditions();temperature.correctBoundaryConditions();
        Foam::surfaceScalarField faceK(Foam::fvc::interpolate(conductivity));
        forAll(mesh.neighbour(),f){const Real a=conductivity[mesh.owner()[f]],b=conductivity[mesh.neighbour()[f]],w=mesh.weights()[f];faceK[f]=a>0&&b>0?1/((1-w)/a+w/b):0;}
        forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()){const Foam::scalarField neighbourK(conductivity.boundaryField()[patch].patchNeighbourField());const Foam::scalarField ownK(conductivity.boundaryField()[patch].patchInternalField());
            forAll(neighbourK,f){const Real a=ownK[f],b=neighbourK[f],w=mesh.weights().boundaryField()[patch][f];faceK.boundaryFieldRef()[patch][f]=a>0&&b>0?1/((1-w)/a+w/b):0;}}
        Foam::fvScalarMatrix equation(Foam::fvm::Sp(capacity/deltaT,temperature)-Foam::fvm::laplacian(faceK,temperature)==source);
        const Foam::solverPerformance linear=equation.solve(solver);++report.linearSolves;++report.nonlinearIterations;
        if(!Terminal&&(!linear.converged()||linear.singular()||!finite(linear.finalResidual()))){error="OpenFOAM implicit material sparse solve failed";report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}
        temperature.correctBoundaryConditions();const Foam::tmp<Foam::surfaceScalarField> flux=equation.flux();Real boundary=0,sum=0,change=0;long double stableChange=0,inventoryScale=0;
        forAll(flux().boundaryField(),patch)if(!mesh.boundary()[patch].coupled())forAll(flux().boundaryField()[patch],f)boundary+=dt*flux().boundaryField()[patch][f];
        bool admissible=true;for(std::size_t c=0;c<n;++c){const Real T=temperature[c];if(!finite(T)||T<caloric[c].minimumTemperature||T>caloric[c].maximumTemperature)admissible=false;
            sum+=caloric[c].energy<Terminal>(T);stableChange+=static_cast<long double>(caloric[c].secant<Terminal>(T,anchor[c]))*(T-anchor[c])-(target[c]-caloric[c].energy<Terminal>(anchor[c]));
            inventoryScale+=absValue(target[c])+absValue(caloric[c].energy<Terminal>(T));change=maxValue(change,absValue(T-previous[c])/maxValue(1,maxValue(absValue(T),absValue(previous[c]))));}
        const Real residual=static_cast<Real>(stableChange+boundary);const Real storedResidual=sum-targetSum+boundary;const Real scale=maxValue(1,maxValue(absValue(static_cast<Real>(stableChange)),absValue(boundary)));
        const Real roundoff=static_cast<Real>(64*std::numeric_limits<Real>::epsilon()*inventoryScale);
        if((admissible&&change<=controls.nonlinearRelativeTolerance&&absValue(residual)<=controls.energyAbsoluteTolerance+controls.energyRelativeTolerance*scale+roundoff)
            ||(Terminal&&iteration+1>=std::max(1,controls.nonlinearMaxIterations))){
            for(std::size_t c=0;c<n;++c){next.solid[c].energy=caloric[c].energy<Terminal>(temperature[c]);MaterialPrimitive recovered;bool numerical=false;if(!recoverMaterial<Terminal>(next.solid[c],next.solidMesh.volumes[c],p,recovered,nullptr,-1,&numerical)){error="implicit material endpoint EOS invalid";if(numerical)report.failureOrigin=MaterialFailureOrigin::ConductionNumerical;return false;}next.solid[c].porosity=recovered.porosity;report.minimumTemperature=minValue(report.minimumTemperature,recovered.temperature);}
            report.energyResidual+=residual;report.budgetDelta.numericalEnergyResidual+=storedResidual;report.budgetDelta.boundaryEnergy+=boundary;return true;}
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
template<bool Terminal=false>
bool baseTrace(const HostState& base,const PhysicsConfig& p,std::vector<GasPrimitive>& bulk,std::string& error,bool* numerical=nullptr){
    if(numerical)*numerical=false;
    bulk.resize(base.surface.area.size());
    if(Terminal&&base.surface.gasFace.size()<bulk.size()){error="wall predictor consumed face map missing";return false;}
    for(std::size_t f=0;f<bulk.size();++f){const int face=base.surface.gasFace[f];if(face<0)continue;
        if(!Terminal&&static_cast<std::size_t>(face)>=base.gasMesh.owner.size()){if(numerical)*numerical=true;error="wall predictor face outside gas mesh";return false;}
        if(Terminal&&(base.gasMesh.owner.empty()||base.gas.empty()||base.gasMesh.volumes.empty())){error="wall predictor consumed gas storage missing";return false;}
        const int c=base.gasMesh.owner[face];
        if(!Terminal&&(c<0||static_cast<std::size_t>(c)>=base.gas.size())){if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
        if(!recoverGas(base.gas[c],base.gasMesh.volumes[c],p,bulk[f])){
            if(!Terminal){if(numerical)*numerical=true;error="wall predictor gas EOS failed";return false;}
            if(!completePredictorGas(base.gas[c],base.gasMesh.volumes[c],p,bulk[f])){error="wall predictor selected gas caloric expression unavailable";return false;}
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
        const std::vector<Vec3>* pressureGradient=nullptr,const std::vector<Real>* sideVolume=nullptr,
        const std::vector<GasWallMatchingSample>* recorded=nullptr,bool terminalSurface=false){
        std::vector<GasWallClosureContext> contexts;terminalWallFailure=false;surfaceNumericalFailure=false;
        if(wall){
            terminalWallFailure=true;
            auto config=p.wallModel;config.enableSst=p.enableSst;
            if(!wall->prepareGeometry(state.gasMesh,state.surface.gasFace,error,config))return false;
            std::vector<GasWallMatchingSample> fresh;
            if(!recorded){if(!wall->sample(state,p,fresh,error))return false;recorded=&fresh;}
            ugkwp::GasMechanismView<Real,Ns> view;
            if(p.gasMode==ugkwp::GasMode::MixtureChemistry){
                view=mechanism.mechanismView<Ns>();}
            if(!wall->contexts(*recorded,view,p.wallModel,contexts,error,&surfaceNumericalFailure)){
                if(!terminalSurface||!surfaceNumericalFailure||!wall->contexts<true>(*recorded,view,p.wallModel,contexts,error,&surfaceNumericalFailure)){
                    terminalWallFailure=!surfaceNumericalFailure;return false;
                }
            }
        }
        terminalWallFailure=false;
        bool success=evaluateCpuSurface(state,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,pressureGradient,sideVolume,wall?&contexts:nullptr,&surfaceNumericalFailure);
        if(!success&&terminalSurface&&surfaceNumericalFailure)
            success=detail::completeCpuSurface(state,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,pressureGradient,sideVolume,wall?&contexts:nullptr,&surfaceNumericalFailure);
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
    return predictWallImpl(base,interval,output,error,diagnostics,false);
}
bool CpuMaterialDriver::predictWallImpl(const HostState& base,const CouplingInterval& interval,WallProgram& output,std::string& error,CpuMaterialReport* diagnostics,bool terminalPredictor)const{
    if(data_->wall&&data_->model.physics.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
    if(diagnostics){*diagnostics=CpuMaterialReport{};diagnostics->recoverable=false;}
    const auto fail=[&](MaterialFailureOrigin origin){if(diagnostics)diagnostics->failureOrigin=origin;return false;};
    if(!(terminalPredictor?validateMaterialSweepRemainder<true>(base,error):validateMaterialSweepRemainder(base,error)))return fail(MaterialFailureOrigin::GeometryNumerical);
    if(!terminalPredictor&&(!validCouplingInterval(interval)||interval.begin!=base.time)){error="wall predictor interval/base time mismatch";return fail(MaterialFailureOrigin::GeometryNumerical);}
    if(!terminalPredictor&&(data_->model.physics.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty())){error="CPU material owner requires actual volume/surface state, not normal columns";return fail(MaterialFailureOrigin::SurfaceNumerical);}
    std::vector<GasPrimitive> bulk;bool numerical=false;
    if(!(terminalPredictor?baseTrace<true>(base,data_->model.physics,bulk,error,&numerical):baseTrace(base,data_->model.physics,bulk,error,&numerical)))return fail(numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);CpuSurfaceResult surface;
    if(!data_->evaluate(base,data_->model.physics,bulk,{},{},{},interval.end-interval.begin,interval.identity.sequence,surface,error,nullptr,nullptr,nullptr,terminalPredictor)){if(diagnostics)diagnostics->recoverable=!data_->terminalWallFailure;return fail(data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None);}
    WallProgram program;program.interval=interval;program.surface=base.surface;WallKnot first;first.time=interval.begin;first.faces=surface.wall;first.gasPoints=base.gasMesh.points;first.solidPoints=base.solidMesh.points;
    WallKnot last=first;last.time=interval.end;
    if(data_->model.physics.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){
        if(!terminalPredictor&&!base.surface.area.empty()){error="prescribed sinusoidal motion with material interface requires a compatible imposed transfer model";return fail(MaterialFailureOrigin::GeometryNumerical);}
        const auto& motion=data_->model.physics.meshMotion;const Real temporal=::sin(motion.angularFrequency*(interval.end-motion.timeOrigin));
        if((!terminalPredictor&&base.gasMesh.referencePoints.size()!=last.gasPoints.size())||(terminalPredictor&&base.gasMesh.referencePoints.size()<last.gasPoints.size())){error="prescribed gas motion reference topology missing";return fail(base.gasMesh.referencePoints.size()<last.gasPoints.size()?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical);}
        for(std::size_t i=0;i<last.gasPoints.size();++i){const Vec3 X=base.gasMesh.referencePoints[i];last.gasPoints[i]={X.x+motion.amplitude.x*::sin(motion.spatialWaveNumber.x*(X.x-motion.spatialOrigin.x))*temporal,
            X.y+motion.amplitude.y*::sin(motion.spatialWaveNumber.y*(X.y-motion.spatialOrigin.y))*temporal,X.z+motion.amplitude.z*::sin(motion.spatialWaveNumber.z*(X.z-motion.spatialOrigin.z))*temporal};}
    }
    if(data_->model.physics.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.area.empty()){auto aux=base.filmAux;std::vector<Real> targets(aux.size());const Real dt=interval.end-interval.begin;
        for(std::size_t f=0;f<aux.size();++f){aux[f].solidNormalVelocity=surface.physics[f].solidSpeed;aux[f].normalVelocity=surface.physics[f].topSpeed;aux[f].solidFront+=dt*aux[f].solidNormalVelocity;aux[f].gasFront+=dt*aux[f].normalVelocity;
            const int sf=base.surface.solidFace[f];targets[f]=sf>=0?dt*aux[f].solidNormalVelocity*mag(base.solidMesh.areaVectors[sf]):0;}
        HostMesh gas,solid;SurfaceMesh moved;SweepConstraintReport report;
        if(!moveCoupledMeshesConstrained(base,aux,targets,dt,gas,solid,moved,report,error,SweepConstraintControls{},data_->model.physics.tolerances,&numerical)){
            if(diagnostics)diagnostics->recoverable=report.status==SweepConstraintStatus::InvalidTrajectory||report.status==SweepConstraintStatus::Nonconverged;
            if(!terminalPredictor||!numerical||!detail::completeCoupledMeshes(base,aux,targets,dt,gas,solid,moved,report,error,SweepConstraintControls{},data_->model.physics.tolerances,&numerical))return fail(numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None);
        }
        last.gasPoints=gas.points;last.solidPoints=solid.points;}
    program.knots={first,last};output=std::move(program);error.clear();return true;
}
bool CpuMaterialDriver::advanceCandidate(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram){
    return advanceCandidateImpl(base,geometryEndpoint,history,controls,output,corrected,report,error,executedProgram,false);
}
bool CpuMaterialDriver::advanceCandidateImpl(const HostState& base,const HostState& geometryEndpoint,
    const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
    WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executedProgram,bool terminalMaterial){
    report=CpuMaterialReport{};error.clear();
    if(!(terminalMaterial?validateMaterialSweepRemainder<true>(base,error):validateMaterialSweepRemainder(base,error))){report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;report.recoverable=false;return false;}
    const auto& p=data_->model.physics;const auto& window=history.interval();
    auto wallGeometryConfig=p.wallModel;wallGeometryConfig.enableSst=p.enableSst;
    if(!base.solid.empty()&&!data_->mesh){report.recoverable=false;error="CPU material native mesh unavailable";return false;}
    if(!terminalMaterial&&(!controlsValid(controls)||!history.complete()||window.begin!=base.time
        ||p.filmThermalMode!=FilmThermalMode::ThicknessAveraged||!base.normalEnthalpy.empty()
        ||base.solidMesh.topologyHash!=geometryEndpoint.solidMesh.topologyHash
        ||base.gasMesh.topologyHash!=geometryEndpoint.gasMesh.topologyHash)){
        report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;
        if(!controlsValid(controls)&&controlsValid(controls,true)){
            report.failureOrigin=(!finite(controls.transportCfl)||controls.transportCfl<=0||controls.transportCfl>1)?MaterialFailureOrigin::TransportNumerical:controls.maxSubsteps<=0?MaterialFailureOrigin::LocalSourceNumerical:
                (!finite(controls.reactionCfl)||controls.reactionCfl<=0||controls.reactionCfl>1||!finite(controls.reactionMaxSubstep)||controls.reactionMaxSubstep<0)
                    ?MaterialFailureOrigin::ReactionNumerical:
                    (controls.nonlinearMaxIterations<=0||!finite(controls.nonlinearRelativeTolerance)||controls.nonlinearRelativeTolerance<0
                     ||!finite(controls.energyAbsoluteTolerance)||controls.energyAbsoluteTolerance<0||!finite(controls.energyRelativeTolerance)||controls.energyRelativeTolerance<0)
                        ?MaterialFailureOrigin::ConductionNumerical:MaterialFailureOrigin::OwnerNumerical;
        }
        report.recoverable=false;error="unsupported or inconsistent CPU material interval/configuration/topology";return false;}
    try {
        if(terminalMaterial&&base.surface.persistentId.size()<base.surface.area.size()){error="CPU material consumed surface identity map missing";report.recoverable=false;return false;}
        SurfacePacketAddressing packetAddressing;
        if(!packetAddressing.build(base,geometryEndpoint,error)){report.recoverable=false;return false;}
        auto stageGeometry=[&](const HostMesh& mesh,const std::vector<Vec3>& points,Real dt,HostMesh& out,
            std::vector<Real>& sweeps,MaterialFailureOrigin& origin){
            bool numerical=false;
            if(makeStageGeometry(mesh,points,dt,out,sweeps,error,p.tolerances,&numerical))return true;
            origin=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
            if(!terminalMaterial || !numerical)return false;
            if(!detail::completeStageGeometry(mesh,points,dt,out,sweeps,error,p.tolerances,&numerical)){
                if(!numerical)origin=MaterialFailureOrigin::None;
                return false;
            }
            origin=MaterialFailureOrigin::None;return true;
        };
        auto trajectorySurface=[&](const HostState& accepted,const HostMesh& gas,const HostMesh& solid,Real dt,
            SurfaceMesh& out,MaterialFailureOrigin& origin){
            bool numerical=false;
            if(rebuildTrajectorySurface(accepted,gas,solid,dt,out,error,p.tolerances,&numerical))return true;
            origin=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
            if(!terminalMaterial || !numerical)return false;
            if(!detail::completeTrajectorySurface(accepted.gasMesh,accepted.solidMesh,accepted.surface,gas,solid,dt,out,error,p.tolerances,&numerical)){
                if(!numerical)origin=MaterialFailureOrigin::None;
                return false;
            }
            origin=MaterialFailureOrigin::None;return true;
        };
        HostState state=base;state.ledger.clear();state.filmStorage.clear();
        WallProgram program;program.interval=window;program.surface=base.surface;
        CpuMaterialReport accumulated;accumulated.donorPlan.interval=window;
        MaterialReserveKnot initial;initial.time=base.time;initial.cumulativeSolid.resize(base.solid.size());initial.cumulativeFilm.resize(base.film.size());accumulated.donorPlan.knots.push_back(initial);
        accumulated.solidTargetSweeps.assign(base.surface.area.size(),0);
        std::vector<IntervalDonorSum> physicalSweepSum(base.surface.area.size()),actualSweepSum(base.surface.area.size());
        std::vector<SolidQ> gasSolidChange(base.solid.size());std::vector<FilmQ> gasFilmChange(base.film.size());
        auto cursor=history.cursor();const Real interval=window.end-window.begin;
        WindowTime elapsed{};const WindowTime windowEnd=intervalDuration(window);
        // The macro endpoint has an actual gas state. Do not substitute the
        // representative last RK stage when converting film endpoint pV.
        std::vector<GasWallMatchingSample> endpointMatching;bool endpointMatchingReady=false;
        while(elapsed<windowEnd){
            if(!terminalMaterial&&accumulated.materialSteps>=static_cast<std::uint64_t>(controls.maxSubsteps)){error="CPU material substep limit reached";report=accumulated;report.failureOrigin=MaterialFailureOrigin::LocalSourceNumerical;return false;}
            Real dt=std::min(boundedWindowDuration(windowEnd,elapsed),controls.maxSubstep>0?controls.maxSubstep:interval);
            // Only a real regime change forces alignment with a gas-history knot.
            // Ordinary gas micro boundaries do not determine the material clock.
            const GasIntervalRecord* driveRecord=nullptr;
            for(const auto& r:history.records())if(elapsed<r.localEnd){driveRecord=&r;break;}
            if(!driveRecord){error="CPU material driving history exhausted";return false;}
            WindowTime boundedEnd=elapsed+dt;
            if(!materialDriveSlabEndLocal(history,elapsed,boundedEnd,controls.driveRelativeTolerance,controls.driveTractionScale,boundedEnd,accumulated.boundaryDriveSamples,error)){
                if(!terminalMaterial){report=accumulated;report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;return false;}
                // The selected actual driveRecord already exists. This helper
                // only proposes a shorter bound and leaves boundedEnd unchanged
                // on failure; consume that previously evaluated requested bound.
                error.clear();
            }
            dt=boundedWindowDuration(boundedEnd,elapsed);
            if(executedProgram)for(const auto& knot:executedProgram->knots)if(knotOffset(window,knot)>elapsed&&knotOffset(window,knot)<elapsed+dt)dt=boundedWindowDuration(knotOffset(window,knot),elapsed);
            std::vector<GasPrimitive> bulk=driveRecord->gasTrace;std::vector<Vec3> traction=driveRecord->gasTraction;
            if(!terminalMaterial&&(bulk.size()!=state.surface.area.size()||traction.size()!=state.surface.area.size())){
                if(state.surface.area.empty()){bulk.clear();traction.clear();}else{report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;report.recoverable=false;error="CPU material requires actual face-ordered gas drive history";return false;}}
            if(terminalMaterial){
                if(bulk.size()<state.surface.area.size()||state.surface.gasFace.size()<state.surface.area.size()){report.recoverable=false;error="CPU material consumed gas trace/map missing";return false;}
                for(std::size_t f=0;f<state.film.size();++f)if(state.surface.gasFace[f]>=0&&f>=traction.size()){report.recoverable=false;error="CPU material consumed traction missing";return false;}
            }
            HostState trial;WallKnot beginning;CpuSurfaceResult surface;MaterialTransportResult transport;
            std::vector<ExchangePacket> packets;std::vector<Real> radiation;std::vector<CpuFilmDrive> drive;
            std::vector<CpuFilmForcing> forcing;std::vector<FilmQ> filmRate;std::vector<FilmRateBudget> filmRateBudget;
            std::vector<Vec3> pressureGradient;std::vector<Real> sideVolume;
            std::vector<Real> solidSweep,gasSweep;IntervalCursor trialCursor=cursor;
            SurfaceMesh filmEvaluationSurface=state.surface;filmEvaluationSurface.sweptEdgeArea.assign(state.surface.edgeOwner.size(),0);
            std::vector<CpuFilmDrive> refinementDrive;Real filmGuessDt=dt;
            bool accepted=false;WindowTime trialElapsed;
            MaterialFailureOrigin lastFailure=MaterialFailureOrigin::None;Real attemptedDt=dt;
            const int retries=std::max(p.tolerances.maxRetries,p.tolerances.maxCouplingIterations);
            const int lastRetry=terminalMaterial?std::max(0,retries):retries;
            for(int retry=0;retry<=lastRetry;++retry){
                lastFailure=MaterialFailureOrigin::None;attemptedDt=dt;
                const bool terminalLast=terminalMaterial&&retry==lastRetry;
                if(!terminalMaterial&&!(dt>0)){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU material substep does not advance";break;}
                if(dt!=filmGuessDt){filmEvaluationSurface=state.surface;filmEvaluationSurface.sweptEdgeArea.assign(state.surface.edgeOwner.size(),0);refinementDrive.clear();filmGuessDt=dt;}
                if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static&&retry>0)data_->wall->clearGeometry();
                error.clear();trial=state;trialElapsed=elapsed+dt;trial.time=windowAbsolute(window.begin,trialElapsed);trialCursor=cursor;
                if(!trialCursor.takeThroughLocal(trialElapsed,packets,radiation,error))break;
                ++accumulated.materialRhsEvaluations;
                bool transportCompleted=false;
                {bool numerical=false;if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,{},dt,controls.transportCfl,transport,error,&numerical)){
                    lastFailure=numerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;
                    if(!terminalMaterial||!numerical||!detail::completeMaterialTransport(state.solidMesh,state.solid,p,{},attemptedDt,controls.transportCfl,transport,error,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                    lastFailure=MaterialFailureOrigin::None;transportCompleted=true;
                }}
                if(dt>transport.dtLimit&&!terminalLast&&!transportCompleted){lastFailure=MaterialFailureOrigin::TransportNumerical;dt=transport.dtLimit;continue;}
                gasDrive(state,bulk,traction,drive);if(!refinementDrive.empty())drive=refinementDrive;
                std::vector<GasPrimitive> endpointBulk;std::vector<Vec3> endpointTraction;const GasIntervalRecord* endpointRecord=nullptr;
                if(trialElapsed==windowEnd){bool numerical=false;
                    if(!(terminalMaterial?baseTrace<true>(geometryEndpoint,p,endpointBulk,error,&numerical):baseTrace(geometryEndpoint,p,endpointBulk,error,&numerical))){lastFailure=numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;break;}}
                else{
                    if(!history.sampleGasDriveLocal(trialElapsed,endpointBulk,endpointTraction,error))break;
                    for(const auto& r:history.records())if(trialElapsed<r.localEnd){endpointRecord=&r;break;}
                    if(!endpointRecord){error="missing endpoint gas driving record";break;}
                }
                for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){
                    if(terminalMaterial&&f>=endpointBulk.size()){error="CPU material consumed endpoint trace missing";break;}
                    drive[f].endpointPressure=endpointBulk[f].pressure;}
                if(!error.empty())break;
                if(data_->wall&&p.enableFilm){
                    const auto& matching=driveRecord->gasWallMatching;
                    if(trialElapsed==windowEnd&&!endpointMatchingReady){
                        if(!data_->wall->prepareGeometry(geometryEndpoint.gasMesh,geometryEndpoint.surface.gasFace,error,wallGeometryConfig)
                            ||!data_->wall->sample(geometryEndpoint,p,endpointMatching,error)){report.recoverable=false;return false;}
                        endpointMatchingReady=true;
                    }
                    const auto& ending=trialElapsed==windowEnd?endpointMatching:endpointRecord->gasWallMatching;
                    if(!terminalMaterial&&(matching.size()!=state.surface.area.size()||ending.size()!=matching.size())){
                        report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;report.recoverable=false;error="film wall matching pressure history layout mismatch";return false;}
                    for(std::size_t f=0;f<drive.size();++f)if(state.surface.gasFace[f]>=0){
                        if(terminalMaterial&&(f>=matching.size()||f>=ending.size())){report.recoverable=false;error="film consumed wall matching pressure missing";return false;}
                        drive[f].pressure=matching[f].pressure;drive[f].endpointPressure=ending[f].pressure;
                    }
                }
                forcing.assign(state.film.size(),CpuFilmForcing{});
                for(const auto& packet:packets){if(!terminalMaterial&&(requiredConsumers(packet.kind)&ConsumeFilm)&&(packet.filmFace<0||static_cast<std::size_t>(packet.filmFace)>=forcing.size())){lastFailure=MaterialFailureOrigin::FilmNumerical;error="CPU gas history film receiver outside mesh";break;}filmForcing(packet,forcing);}
                if(!error.empty())break;
                if(!terminalMaterial&&radiation.size()!=state.surface.area.size()&&!radiation.empty()){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="CPU radiation face layout mismatch";break;}
                for(std::size_t f=0;f<radiation.size();++f)if(driveRecord->radiationReceiver[f]==ConsumeFilm){if(!terminalMaterial&&f>=forcing.size()){lastFailure=MaterialFailureOrigin::FilmNumerical;error="film radiation has no film owner";break;}forcing[f].radiationEnergy+=radiation[f];}
                if(!error.empty())break;
                HostState evaluation=state;evaluation.surface=filmEvaluationSurface;
                if(p.enableFilm){std::vector<FilmAux> profiles;
                    bool numerical=false;
                    if(!evaluateCpuFilmTransport(evaluation,p,dt,drive,filmRate,filmRateBudget,error,&profiles,&pressureGradient,&sideVolume,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        if(!terminalMaterial||!numerical||!detail::completeCpuFilmTransport(evaluation,p,attemptedDt,drive,filmRate,filmRateBudget,error,&profiles,&pressureGradient,&sideVolume,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    evaluation.filmAux=profiles;
                    for(std::size_t f=0;f<profiles.size();++f)evaluation.film[f].enthalpy+=pressureVolumeProduct(state.filmAux[f].pressure,state.film[f].mass/p.liquid.rho,profiles[f].pressure,state.film[f].mass/p.liquid.rho);
                }else{filmRate.clear();pressureGradient.assign(state.surface.area.size(),Vec3{});sideVolume.assign(state.surface.area.size(),0);}
                if(!data_->evaluate(evaluation,p,bulk,driveRecord->gasGradient,transport.rates,filmRate,dt,window.identity.sequence,surface,error,&pressureGradient,&sideVolume,&driveRecord->gasWallMatching,terminalMaterial)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
                bool shortenedForSource=false;
                // These extra responses only choose a shorter retry; the final
                // terminal attempt has no consumer for that proof-only work.
                for(const auto& record:history.records())if(!terminalLast&&record.localBegin>elapsed&&record.localBegin<trialElapsed){
                    CpuSurfaceResult response;++accumulated.boundaryDriveSamples;
                    if(!data_->evaluate(evaluation,p,record.gasTrace,record.gasGradient,transport.rates,filmRate,dt,window.identity.sequence,response,error,&pressureGradient,&sideVolume,&record.gasWallMatching,terminalLast)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}dt=boundedWindowDuration(record.localBegin,elapsed);error.clear();shortenedForSource=true;break;}
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
                    if(fraction>controls.driveSourceFraction&&!terminalLast){dt=boundedWindowDuration(record.localBegin,elapsed);shortenedForSource=true;break;}
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
                    bool numerical=false;
                    if(!finalizeCpuFilmPhasePacket(state.film[f],state.filmAux[f].pressure,filmRate[f],forcing[f],dt,bottom,p,phase,exactEvent,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        if(!terminalMaterial||!numerical||!detail::completeCpuFilmPhasePacket(state.film[f],state.filmAux[f].pressure,filmRate[f],forcing[f],attemptedDt,bottom,p,phase,exactEvent,error,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    forcing[f].exactPhaseEvent=forcing[f].exactPhaseEvent||exactEvent;
                    if(!error.empty())break;filmForcing(phase,forcing);}
                if(!error.empty())break;
                if(p.enableFilm)for(std::size_t f=0;f<forcing.size();++f){const Real remaining=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                    const Real scale=absValue(state.film[f].mass)+absValue(dt*filmRate[f].mass)+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass);
                    if((forcing[f].mass<0||forcing[f].phaseMass<0)&&absValue(remaining)<=16*std::numeric_limits<Real>::epsilon()*scale)forcing[f].exactPhaseEvent=true;
                }
                if(p.enableFilm){Real filmDt=dt;bool numerical=false;
                    if(!estimateCpuFilmStep(evaluation,p,forcing,drive,dt,dt,filmDt,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        if(!terminalMaterial||!numerical||!detail::completeCpuFilmStep(evaluation,p,forcing,drive,attemptedDt,attemptedDt,filmDt,error,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    if(filmDt<dt*(1-32*std::numeric_limits<Real>::epsilon())&&!terminalLast){lastFailure=MaterialFailureOrigin::FilmNumerical;dt=filmDt;continue;}}
                std::vector<Real> target(state.surface.area.size(),0);
                for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasSolid){
                    std::size_t match=0;
                    if(!packetAddressing.gasSolidIndex(state,packet,match,error))break;
                    const int donor=packet.solidCell;
                    MaterialPrimitive material;
                    if(!(terminalMaterial?recoverMaterial<true>(state.solid[donor],state.solidMesh.volumes[donor],p,material):recoverMaterial(state.solid[donor],state.solidMesh.volumes[donor],p,material))){
                        lastFailure=terminalMaterial?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical;error="surface donor EOS failed";break;}
                    for(int c=0;c<Nc;++c)if(packet.condensed[c]!=0){
                        if(!terminalMaterial&&(!finite(packet.condensed[c])||!finite(p.condensed[c].rho)||!(p.condensed[c].rho>0))){
                            lastFailure=MaterialFailureOrigin::GeometryNumerical;error="invalid surface recession constituent";break;}
                        target[match]-=packet.condensed[c]/p.condensed[c].rho/(1-material.porosity);
                    }
                    if(!error.empty())break;
                }
                for(const auto& phase:surface.phasePackets){MaterialPrimitive material;const int c=phase.solidCell;if(!(terminalMaterial?recoverMaterial<true>(state.solid[c],state.solidMesh.volumes[c],p,material):recoverMaterial(state.solid[c],state.solidMesh.volumes[c],p,material))){lastFailure=terminalMaterial?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical;error="phase donor EOS failed";break;}
                    // Heat/force-only packets do not select a condensed phase.
                    target[phase.filmFace]-=phase.mass==0 ? Real(0)
                        : phase.mass/p.condensed[p.material.phaseCondensed].rho/(1-material.porosity);}
                if(!error.empty())break;
                std::vector<FilmAux> aux=evaluation.filmAux;
                for(std::size_t f=0;f<aux.size();++f)if(state.surface.gasFace[f]>=0){const int sf=state.surface.solidFace[f],gf=state.surface.gasFace[f];const Real as=sf>=0?mag(state.solidMesh.areaVectors[sf]):0,ag=mag(state.gasMesh.areaVectors[gf]);
                    aux[f].solidNormalVelocity=as>0?target[f]/(as*dt):0;
                    const Real mass=p.enableFilm?dt*filmRate[f].mass+forcing[f].mass+forcing[f].phaseMass:0;
                    const Real side=p.enableFilm?sideVolume[f]:0;
                    if(p.enableFilm){Real endpointMass=state.film[f].mass+mass;const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(mass));if(absValue(endpointMass)<=roundoff&&forcing[f].exactPhaseEvent)endpointMass=0;if(!terminalMaterial&&(endpointMass<0||!finite(endpointMass))){lastFailure=MaterialFailureOrigin::GeometryNumerical;error="negative film mass in geometric proposal";break;}
                        if(!(terminalMaterial?detail::completeFilmThicknessFromMass(endpointMass,evaluation.surface.area[f],p.liquid.rho,aux[f].thickness):filmThicknessFromMass(endpointMass,evaluation.surface.area[f],p.liquid.rho,aux[f].thickness))){lastFailure=MaterialFailureOrigin::GeometryNumerical;error="invalid endpoint film thickness";break;}aux[f].area=evaluation.surface.area[f];}
                    aux[f].normalVelocity=p.enableFilm&&surface.physics[f].wet?(target[f]/dt+mass/(dt*p.liquid.rho)-side)/ag:aux[f].solidNormalVelocity;
                    aux[f].solidFront=state.filmAux[f].solidFront+dt*aux[f].solidNormalVelocity;aux[f].gasFront=state.filmAux[f].gasFront+dt*aux[f].normalVelocity;}
                if(!error.empty())break;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!target.empty()){SweepConstraintReport motion;
                    bool numerical=false;
                    if(!moveCoupledMeshesConstrained(state,aux,target,dt,trial.gasMesh,trial.solidMesh,trial.surface,motion,error,SweepConstraintControls{},p.tolerances,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::GeometryNumerical:MaterialFailureOrigin::None;
                        if(!terminalMaterial||!numerical||!detail::completeCoupledMeshes(state,aux,target,dt,trial.gasMesh,trial.solidMesh,trial.surface,motion,error,SweepConstraintControls{},p.tolerances,&numerical)){
                            if(!numerical)lastFailure=MaterialFailureOrigin::None;
                            if(motion.status==SweepConstraintStatus::Incompatible||motion.status==SweepConstraintStatus::SizeLimit||motion.status==SweepConstraintStatus::InvalidInput){report.failureOrigin=lastFailure;report.failedLocalDt=attemptedDt;report.recoverable=false;return false;}break;
                        }
                        lastFailure=MaterialFailureOrigin::None;
                    }
                }else{
                    const Real fraction=windowFraction(trialElapsed,WindowTime{},windowEnd,trial.time,window.begin,window.end);std::vector<Vec3> gp=base.gasMesh.points,sp=base.solidMesh.points;
                    if(!terminalMaterial&&(gp.size()!=geometryEndpoint.gasMesh.points.size()||sp.size()!=geometryEndpoint.solidMesh.points.size())){lastFailure=MaterialFailureOrigin::OwnerNumerical;error="material trajectory topology mismatch";break;}
                    if(terminalMaterial&&(gp.size()>geometryEndpoint.gasMesh.points.size()||sp.size()>geometryEndpoint.solidMesh.points.size())){error="material consumed trajectory coordinates missing";break;}
                    for(std::size_t i=0;i<gp.size();++i)gp[i]+=(geometryEndpoint.gasMesh.points[i]-gp[i])*fraction;
                    for(std::size_t i=0;i<sp.size();++i)sp[i]+=(geometryEndpoint.solidMesh.points[i]-sp[i])*fraction;
                    if(executedProgram){WallKnot executed;if(!intervalSampleWallLocal(*executedProgram,trialElapsed,executed,error))break;gp=executed.gasPoints;sp=executed.solidPoints;}
                    if(!stageGeometry(state.gasMesh,gp,dt,trial.gasMesh,gasSweep,lastFailure)||!stageGeometry(state.solidMesh,sp,dt,trial.solidMesh,solidSweep,lastFailure))break;
                    if(!state.surface.area.empty()&&!trajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,lastFailure))break;
                }
                if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){WallKnot executed;
                    if(!intervalSampleWallLocal(*executedProgram,trialElapsed,executed,error))break;
                    HostMesh constrainedGas=trial.gasMesh,constrainedSolid=trial.solidMesh;SurfaceMesh constrainedSurface=trial.surface;
                    const bool close=materialPointsClose(trial.gasMesh,executed.gasPoints,p.tolerances)&&materialPointsClose(trial.solidMesh,executed.solidPoints,p.tolerances);
                    if(close){if(!stageGeometry(state.gasMesh,executed.gasPoints,dt,trial.gasMesh,gasSweep,lastFailure)||!stageGeometry(state.solidMesh,executed.solidPoints,dt,trial.solidMesh,solidSweep,lastFailure)
                        ||!trajectorySurface(state,trial.gasMesh,trial.solidMesh,dt,trial.surface,lastFailure))break;
                        std::vector<Real> solveTarget;if(!(terminalMaterial?compensateMaterialSweepTargets<true>(state,target,solveTarget,error):compensateMaterialSweepTargets(state,target,solveTarget,error))){lastFailure=terminalMaterial?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical;break;}
                        const bool compatible=materialSweepsCompatible(state.solidMesh,state.surface,solveTarget,solidSweep,p.tolerances);
                        if(!compatible){trial.gasMesh=std::move(constrainedGas);trial.solidMesh=std::move(constrainedSolid);trial.surface=std::move(constrainedSurface);}
                    }
                }
                if(!stageGeometry(state.solidMesh,trial.solidMesh.points,dt,trial.solidMesh,solidSweep,lastFailure)
                    ||!stageGeometry(state.gasMesh,trial.gasMesh.points,dt,trial.gasMesh,gasSweep,lastFailure))break;
                if(p.enableFilm){HostState consistentFilm=state;consistentFilm.surface=trial.surface;std::vector<FilmQ> actualRates;std::vector<FilmRateBudget> actualBudgets;
                    bool numerical=false;
                    if(!evaluateCpuFilmTransport(consistentFilm,p,dt,drive,actualRates,actualBudgets,error,nullptr,nullptr,nullptr,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        if(!terminalMaterial||!numerical||!detail::completeCpuFilmTransport(consistentFilm,p,attemptedDt,drive,actualRates,actualBudgets,error,nullptr,nullptr,nullptr,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    bool same=actualRates.size()==filmRate.size();for(std::size_t f=0;same&&f<filmRate.size();++f){same=actualRates[f].mass==filmRate[f].mass&&actualRates[f].enthalpy==filmRate[f].enthalpy;
                        for(int species=0;species<Ns;++species)same=same&&actualRates[f].species[species]==filmRate[f].species[species];}
                    if(!same&&!terminalLast){lastFailure=MaterialFailureOrigin::FilmNumerical;filmEvaluationSurface=trial.surface;refinementDrive=drive;continue;}
                }
                if(p.enableFilm&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession){
                    for(std::size_t f=0;f<state.film.size();++f)if(state.surface.gasFace[f]>=0){
                        const int gf=state.surface.gasFace[f],sf=state.surface.solidFace[f];
                        Real updatedMass=(state.film[f].mass+dt*filmRate[f].mass+forcing[f].mass)+forcing[f].phaseMass;
                        const Real roundoff=16*std::numeric_limits<Real>::epsilon()*(absValue(state.film[f].mass)+absValue(dt*filmRate[f].mass)+absValue(forcing[f].mass)+absValue(forcing[f].phaseMass));
                        if(forcing[f].exactPhaseEvent&&absValue(updatedMass)<=roundoff)updatedMass=0;
                        if(!terminalMaterial&&!filmSweepsCompatible(state.film[f].mass,updatedMass,p.liquid.rho,gasSweep[gf],sf>=0?solidSweep[sf]:0,dt*sideVolume[f],p.tolerances)){
                            lastFailure=MaterialFailureOrigin::GeometryNumerical;
                            error="actual gas/solid/side swept volume does not match updated film mass";break;}
                    }
                    if(!error.empty())break;
                }
                ++accumulated.materialRhsEvaluations;transportCompleted=false;
                {bool numerical=false;if(!evaluateMaterialTransport(state.solidMesh,state.solid,p,solidSweep,dt,controls.transportCfl,transport,error,&numerical)){
                    lastFailure=numerical?MaterialFailureOrigin::TransportNumerical:MaterialFailureOrigin::None;
                    if(!terminalMaterial||!numerical||!detail::completeMaterialTransport(state.solidMesh,state.solid,p,solidSweep,attemptedDt,controls.transportCfl,transport,error,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;break;}
                    lastFailure=MaterialFailureOrigin::None;transportCompleted=true;
                }}
                if(dt>transport.dtLimit&&!terminalLast&&!transportCompleted){lastFailure=MaterialFailureOrigin::TransportNumerical;dt=transport.dtLimit;continue;}
                CpuMaterialReport stepReport;
                bool reactionOk=true;for(std::size_t c=0;c<state.solid.size();++c){std::uint64_t count=0;SolidQ reacted;
                    bool numerical=false;
                    if(!advanceMaterialReactions(state.solid[c],state.solidMesh.volumes[c],p,dt,controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps,reacted,count,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;
                        if(!terminalLast||!numerical||!detail::completeMaterialReactions(state.solid[c],state.solidMesh.volumes[c],p,attemptedDt,controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps,reacted,count,error,&numerical)){if(!numerical)lastFailure=MaterialFailureOrigin::None;reactionOk=false;break;}
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    trial.solid[c]=addSolid(reacted,transport.rates[c],dt);stepReport.reactionSteps+=count;}
                if(!reactionOk){dt*=.5;error.clear();continue;}
                std::vector<ExchangePacket> all=packets;all.insert(all.end(),surface.phasePackets.begin(),surface.phasePackets.end());
                Real packetRoundoff=0;std::vector<SolidQ> applied;
                bool packetNumerical=false;
                if(!applyMaterialPacketBatch(trial.solid,all,applied,packetRoundoff,error,&packetNumerical)){
                    lastFailure=packetNumerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;
                    if(!terminalLast||!packetNumerical||!detail::completeMaterialPacketBatch(trial.solid,all,applied,packetRoundoff,error,&packetNumerical)){
                        if(!packetNumerical)lastFailure=MaterialFailureOrigin::None;dt*=.5;error.clear();continue;
                    }
                    lastFailure=MaterialFailureOrigin::None;
                }
                trial.solid=std::move(applied);stepReport.numericalMassRoundoff+=packetRoundoff;
                for(const auto& packet:all)accountCpuPacket(packet,stepReport.budgetDelta);
                if(!error.empty())break;
                for(std::size_t f=0;f<radiation.size();++f){stepReport.budgetDelta.radiation+=radiation[f];if(driveRecord->radiationReceiver[f]==ConsumeSolid){const int c=state.surface.solidCell[f];if(!terminalMaterial&&c<0){lastFailure=MaterialFailureOrigin::LocalSourceNumerical;error="radiation has no material owner";break;}trial.solid[c].energy+=radiation[f];}}
                if(!error.empty())break;
                bool materialAdmissible=true;
                for(std::size_t c=0;c<trial.solid.size();++c){MaterialPrimitive material;
                    bool numerical=true;
                    if((!terminalLast&&!validSolidInventory(trial.solid[c]))||(trialElapsed<windowEnd
                        &&!(terminalLast?recoverMaterial<true>(trial.solid[c],trial.solidMesh.volumes[c],p,material,nullptr,-1,&numerical):recoverMaterial(trial.solid[c],trial.solidMesh.volumes[c],p,material,nullptr,-1,&numerical)))){
                        lastFailure=numerical?MaterialFailureOrigin::ReactionNumerical:MaterialFailureOrigin::None;materialAdmissible=false;break;
                    }
                    if(trialElapsed<windowEnd)trial.solid[c].porosity=material.porosity;
                }
                if(!materialAdmissible){dt*=.5;error.clear();continue;}
                if(p.enableFilm){HostState filmState=state;filmState.surface=trial.surface;CpuFilmCandidate film;
                    bool numerical=false;
                    if(!advanceCpuFilmCandidate(filmState,p,dt,forcing,drive,film,error,&numerical)){
                        lastFailure=numerical?MaterialFailureOrigin::FilmNumerical:MaterialFailureOrigin::None;
                        if(!terminalLast||!numerical||!detail::completeCpuFilmCandidate(filmState,p,attemptedDt,forcing,drive,film,error,&numerical)){
                            if(!numerical)lastFailure=MaterialFailureOrigin::None;dt*=.5;error.clear();continue;
                        }
                        lastFailure=MaterialFailureOrigin::None;
                    }
                    trial.film=std::move(film.film);trial.filmAux=std::move(film.filmAux);Budget filmBudget=film.report.budgetDelta;
                    for(int k=0;k<NParticipants;++k){filmBudget.exchangeMass[k]=0;filmBudget.exchangeEnergy[k]=0;filmBudget.exchangeMomentum[k]=Vec3{};for(int s=0;s<Ns;++s)filmBudget.exchangeSpecies[k][s]=0;}
                    filmBudget.filmKineticAdvection=0;stepReport.numericalMassRoundoff+=film.report.numericalMassRoundoff;addBudget(stepReport.budgetDelta,filmBudget);stepReport.filmSteps=1;
                    // Surface radiation is accounted by the interval owner above.
                    Real filmRadiation=0;for(const auto& f:forcing)filmRadiation+=f.radiationEnergy;stepReport.budgetDelta.radiation-=filmRadiation;
                    for(std::size_t f=0;f<trial.film.size();++f){trial.filmAux[f].solidFront=aux[f].solidFront;trial.filmAux[f].gasFront=aux[f].gasFront;trial.filmAux[f].solidNormalVelocity=aux[f].solidNormalVelocity;trial.filmAux[f].normalVelocity=aux[f].normalVelocity;
                        if(!terminalMaterial&&(state.film[f].mass==0)!=(trial.film[f].mass==0)&&trialElapsed<windowEnd){report=accumulated;report.failureOrigin=MaterialFailureOrigin::FilmNumerical;report.failedLocalDt=attemptedDt;report.eventTime=trial.time;report.eventDuration=windowDuration(trialElapsed,WindowTime{});error="film phase event requires an exact macro boundary";return false;}}}
                else trial.filmAux=aux;
                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession
                    &&!(terminalMaterial?updateMaterialSweepRemainder<true>(state,target,solidSweep,trial.solidMesh,trial.solidSweepRemainder,error)
                        :updateMaterialSweepRemainder(state,target,solidSweep,trial.solidMesh,trial.solidSweepRemainder,error))){lastFailure=terminalMaterial?MaterialFailureOrigin::None:MaterialFailureOrigin::GeometryNumerical;break;}
                for(std::size_t f=0;f<target.size();++f){
                    physicalSweepSum[f].add(target[f]);accumulated.solidTargetSweeps[f]=static_cast<Real>(physicalSweepSum[f].value());
                    const int face=state.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(solidSweep[face]);
                }
                stepReport.budgetDelta.bodyWork+=dt*transport.bodyPower;
                stepReport.budgetDelta.gclResidual+=maxValue(maximumGclResidual(state.gasMesh,trial.gasMesh,gasSweep),maximumGclResidual(state.solidMesh,trial.solidMesh,solidSweep));
                for(std::size_t f=0;f<state.solidMesh.owner.size();++f){const auto kind=state.solidMesh.boundaryKind[f];if(kind==BoundaryKind::Internal||kind==BoundaryKind::Periodic||kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;const auto& flux=transport.faceFlux[f];
                    stepReport.budgetDelta.boundaryEnergy+=dt*flux.energy;for(int c=0;c<Nc;++c){stepReport.budgetDelta.boundaryMass+=dt*flux.condensed[c];for(int e=0;e<p.nElements;++e)stepReport.budgetDelta.boundaryElements[e]+=dt*flux.condensed[c]*p.condensed[c].element[e];}
                    for(int s=0;s<Ns;++s){stepReport.budgetDelta.boundaryMass+=dt*flux.pore[s];stepReport.budgetDelta.boundarySpecies[s]+=dt*flux.pore[s];for(int e=0;e<p.nElements;++e)stepReport.budgetDelta.boundaryElements[e]+=dt*flux.pore[s]*p.species[s].element[e];}}
                beginning.time=state.time;beginning.localTime=elapsed;beginning.faces=surface.wall;beginning.gasPoints=state.gasMesh.points;beginning.solidPoints=state.solidMesh.points;
                if(program.knots.empty())program.knots.push_back(beginning);
                WallKnot ending=beginning;ending.time=trial.time;ending.localTime=trialElapsed;ending.gasPoints=trial.gasMesh.points;ending.solidPoints=trial.solidMesh.points;
                if(trialElapsed<windowEnd){CpuSurfaceResult endpointProposal;
                    if(!endpointRecord){error="missing endpoint wall matching record";break;}if(!data_->evaluate(trial,p,endpointBulk,endpointRecord->gasGradient,{},{},dt,window.identity.sequence,endpointProposal,error,nullptr,nullptr,&endpointRecord->gasWallMatching,terminalMaterial)){lastFailure=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;if(data_->terminalWallFailure){report.recoverable=false;return false;}break;}
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
                for(const auto& phase:surface.phasePackets){const int f=phase.filmFace,c=phase.solidCell;
                    MaterialPrimitive material;recoverMaterial(state.solid[c],state.solidMesh.volumes[c],p,material);
                    const double density=phase.mass==0?0:p.condensed[p.material.phaseCondensed].rho;
                    std::fprintf(stderr,"R3PHASE,window=%llu,micro=%llu,time=%a,dt=%a,face=%llu,film=%d,solid=%d,index=%d,mass=%a,rho=%a,porosity=%a,target=%a,energy=%a,heat=%a,mx=%a,my=%a,mz=%a,forcingMass=%a,forcingPhaseMass=%a,oldFilmM=%a,newFilmM=%a,oldFilmH=%a,newFilmH=%a,oldMeanX=%a,newMeanX=%a,oldSolidV=%a,newSolidV=%a\n",
                        static_cast<unsigned long long>(window.identity.sequence),static_cast<unsigned long long>(record.microSequence),state.time,dt,
                        static_cast<unsigned long long>(phase.face),f,c,p.material.phaseCondensed,phase.mass,density,material.porosity,target[f],phase.energy,phase.conductive,
                        phase.momentum.x,phase.momentum.y,phase.momentum.z,forcing[f].mass,forcing[f].phaseMass,state.film[f].mass,trial.film[f].mass,
                        state.film[f].enthalpy,trial.film[f].enthalpy,state.filmAux[f].meanVelocity.x,trial.filmAux[f].meanVelocity.x,state.solidMesh.volumes[c],trial.solidMesh.volumes[c]);
                }

                accumulated.numericalMassRoundoff+=stepReport.numericalMassRoundoff;accumulated.materialSteps++;accumulated.reactionSteps+=stepReport.reactionSteps;accumulated.filmSteps+=stepReport.filmSteps;accumulated.linearSolves+=stepReport.linearSolves;
                accumulated.nonlinearIterations+=stepReport.nonlinearIterations;accumulated.energyResidual+=stepReport.energyResidual;accumulated.minimumTemperature=minValue(accumulated.minimumTemperature,stepReport.minimumTemperature);addBudget(accumulated.budgetDelta,stepReport.budgetDelta);
                state=std::move(trial);elapsed=trialElapsed;cursor=trialCursor;accepted=true;break;
            }
            if(!accepted){if(error.empty())error="CPU material local retries exhausted";report=accumulated;report.failureOrigin=lastRetry<0?MaterialFailureOrigin::LocalSourceNumerical:lastFailure;report.failedLocalDt=attemptedDt;return false;}
        }
        // Lie/IMEX splitting: all conservative local sources are now in Utarget.
        // This is one true 3-D backward-Euler conduction candidate per window,
        // regardless of gas microstep or local reaction/film step count.
        if(!state.solid.empty()){
            const bool solved=terminalMaterial
                ?nativeConduction<true>(*data_->mesh,base,state,p,interval,controls,accumulated,error)
                :nativeConduction<false>(*data_->mesh,base,state,p,interval,controls,accumulated,error);
            if(!solved){report=accumulated;return false;}
        }
        for(const auto& r:history.records())for(const auto& packet:r.packets)if(requiredConsumers(packet.kind)&(ConsumeSolid|ConsumeFilm))++accumulated.budgetDelta.consumedPackets;
        state.filmStorage.resize(state.film.size());for(std::size_t f=0;f<state.film.size();++f){auto& storage=state.filmStorage[f];storage.step=window.identity.sequence;storage.stage=1;storage.geometry=state.gasMesh.geometryVersion;storage.face=state.surface.persistentId[f];storage.filmFace=f;storage.oldPV=base.filmAux[f].pressure*base.film[f].mass/p.liquid.rho;storage.newPV=state.filmAux[f].pressure*state.film[f].mass/p.liquid.rho;storage.consumed=true;}
        const auto samePoints=[](const HostMesh& a,const HostMesh& b){if(a.points.size()!=b.points.size())return false;for(std::size_t i=0;i<a.points.size();++i)if(a.points[i].x!=b.points[i].x||a.points[i].y!=b.points[i].y||a.points[i].z!=b.points[i].z)return false;return true;};
        if(samePoints(state.gasMesh,geometryEndpoint.gasMesh)&&samePoints(state.solidMesh,geometryEndpoint.solidMesh)){
            state.gasMesh=geometryEndpoint.gasMesh;state.solidMesh=geometryEndpoint.solidMesh;state.gasStages=geometryEndpoint.gasStages;state.solidStages=geometryEndpoint.solidStages;
            if(executedProgram&&p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
                // Re-certify the executed piecewise trajectory, not an invented
                // straight macro chord. Do not increment a provisional carry a
                // second time when adopting the gas owner's endpoint metadata.
                if(!terminalMaterial&&(executedProgram->knots.size()<2||executedProgram->knots.front().time!=window.begin
                    ||executedProgram->knots.back().time!=window.end)){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder trajectory interval mismatch";return false;}
                if(!terminalMaterial){
                    const auto& firstPoints=executedProgram->knots.front().solidPoints;
                    if(firstPoints.size()!=base.solidMesh.points.size()){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder base topology mismatch";return false;}
                    for(std::size_t i=0;i<firstPoints.size();++i)if(firstPoints[i].x!=base.solidMesh.points[i].x
                        ||firstPoints[i].y!=base.solidMesh.points[i].y||firstPoints[i].z!=base.solidMesh.points[i].z){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder base geometry mismatch";return false;}
                }
                HostMesh path=base.solidMesh;actualSweepSum.assign(base.surface.area.size(),IntervalDonorSum{});
                for(std::size_t k=1;k<executedProgram->knots.size();++k){HostMesh endpoint;std::vector<Real> measured;
                    const Real duration=windowDuration(knotOffset(window,executedProgram->knots[k]),knotOffset(window,executedProgram->knots[k-1]));
                    if(!stageGeometry(path,executedProgram->knots[k].solidPoints,duration,endpoint,measured,report.failureOrigin))return false;
                    for(std::size_t f=0;f<actualSweepSum.size();++f){const int face=base.surface.solidFace[f];if(face>=0)actualSweepSum[f].add(measured[face]);}
                    path=std::move(endpoint);
                }
                if(!terminalMaterial&&!samePoints(path,state.solidMesh)){report.failureOrigin=MaterialFailureOrigin::OwnerNumerical;error="executed material remainder endpoint mismatch";return false;}
            }
        }
        if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession&&!base.surface.solidFace.empty()){
            state.solidSweepRemainder.resize(base.surface.solidFace.size());
            for(std::size_t f=0;f<state.solidSweepRemainder.size();++f)state.solidSweepRemainder[f]=static_cast<Real>(
                (base.solidSweepRemainder.empty()?0:static_cast<long double>(base.solidSweepRemainder[f]))
                +physicalSweepSum[f].value()-actualSweepSum[f].value());
            if(!(terminalMaterial?validateMaterialSweepRemainder<true>(state,error):validateMaterialSweepRemainder(state,error))){report=accumulated;report.failureOrigin=MaterialFailureOrigin::GeometryNumerical;return false;}
        }
        state.gasChemistryAudit=geometryEndpoint.gasChemistryAudit;state.gasSstAudit=geometryEndpoint.gasSstAudit;
        state.gasWallDiagnostics=geometryEndpoint.gasWallDiagnostics;state.gasWallDiagnosticTime=geometryEndpoint.gasWallDiagnosticTime;
        state.gas=geometryEndpoint.gas;state.sst=geometryEndpoint.sst;state.particles=geometryEndpoint.particles;
        state.rejectedSteps=geometryEndpoint.rejectedSteps;state.gasVoidFraction=geometryEndpoint.gasVoidFraction;state.gasStages=geometryEndpoint.gasStages;
        state.gasMesh.boundaryPrimitive=geometryEndpoint.gasMesh.boundaryPrimitive;state.gasMesh.thermalBoundary=geometryEndpoint.gasMesh.thermalBoundary;state.gasMesh.boundarySst=geometryEndpoint.gasMesh.boundarySst;
        state.budget=geometryEndpoint.budget;addBudget(state.budget,accumulated.budgetDelta);state.time=window.end;program.donorPlan=accumulated.donorPlan;
        if(data_->wall&&p.meshMotion.policy!=MeshMotionPolicy::Static)data_->wall->clearGeometry();
        if(!program.knots.empty()){const auto& last=history.records().back();CpuSurfaceResult endpointSurface;
            std::vector<GasPrimitive> finalBulk;bool numerical=false;if(!(terminalMaterial?baseTrace<true>(state,p,finalBulk,error,&numerical):baseTrace(state,p,finalBulk,error,&numerical))){report=accumulated;report.failureOrigin=numerical?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;return false;}
            if(!data_->evaluate(state,p,finalBulk,last.gasGradient,{},{},interval,window.identity.sequence,endpointSurface,error,nullptr,nullptr,nullptr,terminalMaterial)){report=accumulated;report.failureOrigin=data_->surfaceNumericalFailure?MaterialFailureOrigin::SurfaceNumerical:MaterialFailureOrigin::None;report.recoverable=!data_->terminalWallFailure;return false;}
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
