#include "io/CoupledRadiation.H"
#include "gpu/Backend.H"
#include "../../../common/thermal/radiation/GpuParticleRadiationCoupler.H"
#include "../../../common/thermal/radiation/RadiationControls.H"
#include "../../../common/gpu/RadiationImplicitMath.H"
#include "IOdictionary.H"
#include "surfaceFields.H"
#include <set>
#include <iostream>
namespace chmt {
struct CoupledRadiation::Data {
    Foam::fvMesh& mesh;const ModelConfig& model;int frequency;
    Foam::IOdictionary properties;
    std::unique_ptr<Foam::gpuThermal::MieTable> table;
    std::unique_ptr<Foam::gpuThermal::MieDoRadiationSolver> solver;
    std::unique_ptr<Foam::gpuThermal::GpuParticleRadiationCoupler> coupler;
    Foam::labelList patches;Foam::volScalarField wallTemperature;
    Foam::List<Foam::scalarField> wallEmissivity;
    std::vector<Vec3> previousPoints;
    Data(Foam::Time& time,Foam::fvMesh& mesh_,const ModelConfig& config,const HostState& state,int every)
      :mesh(mesh_),model(config),frequency(every),
       properties(Foam::IOobject("radiationProperties",time.constant(),time,Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE)),
       wallTemperature(Foam::IOobject("chmtRadiationWallTemperature",time.timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE),
           mesh,Foam::dimensionedScalar("T",Foam::dimTemperature,300)){
        if(frequency<=0)throw std::runtime_error("invalid radiation coupling frequency");
        const auto& m=properties.subDict("model");Foam::fileName path(m.lookup("mieTable"));
        if(!path.isAbsolute())path=time.path()/path;
        table.reset(new Foam::gpuThermal::MieTable(path));
        std::set<int> ids;for(int f:state.surface.gasFace)if(f>=0)ids.insert(mesh.boundaryMesh().whichPatch(f));
        patches.setSize(ids.size());int k=0;for(int id:ids){if(id<0)throw std::runtime_error("radiation coupled face is internal");patches[k++]=id;}
        const auto& d=m.subDict("do");
        // The DOM boundary dictionary is the sole optical boundary source.
        // Match its absorptivity-over-emissivity precedence, including defaults.
        const auto optical=[](const Foam::dictionary& values,Foam::scalar prior) {
            return values.lookupOrDefault<Foam::scalar>("absorptivity",
                values.lookupOrDefault<Foam::scalar>("emissivity",prior));
        };
        Foam::scalar defaultEmissivity=d.lookupOrDefault<Foam::scalar>("boundaryAbsorptivity",1);
        const auto boundaries=d.subOrEmptyDict("boundaryRadiation");
        if(boundaries.found("default"))defaultEmissivity=optical(boundaries.subDict("default"),defaultEmissivity);
        wallEmissivity.setSize(mesh.boundary().size());
        forAll(wallEmissivity,p) {
            const auto& name=mesh.boundary()[p].name();
            const auto value=boundaries.found(name)?optical(boundaries.subDict(name),defaultEmissivity):defaultEmissivity;
            wallEmissivity[p].setSize(mesh.boundary()[p].size(),value);
        }

        solver.reset(new Foam::gpuThermal::MieDoRadiationSolver(mesh,d,*table,patches,Foam::gpuThermal::readRadiationControls(d)));
        coupler.reset(new Foam::gpuThermal::GpuParticleRadiationCoupler(mesh,*table,*solver,patches,
            config.particlePhase.minimumTemperature,config.particlePhase.maximumTemperature));
    }
};
CoupledRadiation::CoupledRadiation(Foam::Time& time,Foam::fvMesh& mesh,const ModelConfig& config,const HostState& state,int frequency)
 :data_(new Data(time,mesh,config,state,frequency)){}
CoupledRadiation::~CoupledRadiation()=default;
bool CoupledRadiation::readWarmStart(const std::string& path){return data_->coupler->readWarmStart(path);}
bool CoupledRadiation::writeWarmStart(const std::string& path)const{return data_->coupler->writeWarmStart(path);}

bool CoupledRadiation::advance(Backend& backend,const WallProgram& program,HostState& state,Real& externalHeat,std::string& error){
    auto& d=*data_;externalHeat=0;
    if(program.interval.identity.sequence%std::uint64_t(d.frequency))return true;
    const double elapsed=state.time-state.radiationTime;
    if(!(elapsed>0)){error="radiation interval does not advance";return false;}
    bool changed=d.previousPoints.size()!=state.gasMesh.points.size();
    if(!changed)for(std::size_t i=0;i<d.previousPoints.size();++i){const auto& a=d.previousPoints[i];const auto& b=state.gasMesh.points[i];if(a.x!=b.x||a.y!=b.y||a.z!=b.z){changed=true;break;}}
    if(changed){Foam::pointField points(state.gasMesh.points.size());
        forAll(points,i){const auto& p=state.gasMesh.points[i];points[i]=Foam::point(p.x,p.y,p.z);}
        d.mesh.movePoints(points);d.solver->updateGeometry();d.previousPoints=state.gasMesh.points;
    }
    ParticleRadiationState material;if(!borrowRadiationState(backend,material,error))return false;
    const int cells=d.mesh.nCells();Foam::scalarField mass(cells,0),temperature(cells,0),diameter(cells,0),epsilon(cells,0);
    if(!material.resident.enabled)for(int c=0;c<cells;++c){mass[c]=material.mass[c];temperature[c]=material.temperature[c];diameter[c]=material.diameter[c];
        epsilon[c]=mass[c]/(d.model.particlePhase.rhoSolid*d.mesh.V()[c]);}
    Foam::List<Foam::scalarField> emissivity(d.wallEmissivity);
    const auto& wall=program.knots.back().faces;
    for(std::size_t f=0;f<state.surface.gasFace.size();++f){const int face=state.surface.gasFace[f];if(face<0)continue;
        const int patch=d.mesh.boundaryMesh().whichPatch(face),local=face-d.mesh.boundaryMesh()[patch].start();
        d.wallTemperature.boundaryFieldRef()[patch][local]=wall[f].temperature;
        // Same exposed-area assumption as CHT; surface gas generation is independent.
        const double area=mag(d.mesh.Sf().boundaryField()[patch][local]);
        if(!material.resident.enabled)emissivity[patch][local]*=std::max(0.,1.-std::min(1.,material.occupiedArea[face]/area));
    }
    const Foam::gpuThermal::ParticleRadiationMaterialSnapshot snapshot{epsilon,mass,temperature,diameter,d.wallTemperature,emissivity,0.,elapsed,material.response,material.resident};
    const auto requested=d.coupler->solve(snapshot,false);
    ugkp::gpu::RadiationAcceptedEnergyDeviceView actual;
    if(!applyResidentRadiationEnergy(backend,requested.resident,actual,state,error))return false;
    // The material history owner still consumes boundary records. Publish only
    // its raw boundary energy here, never optical or per-cell heat fields.
    const auto result=d.coupler->collectBoundaryEnergy();
    std::vector<Real> wallHeat(state.surface.gasFace.size(),0);
    for(std::size_t f=0;f<wallHeat.size();++f){const int face=state.surface.gasFace[f];if(face<0)continue;
        const int patch=d.mesh.boundaryMesh().whichPatch(face),local=face-d.mesh.boundaryMesh()[patch].start();
        wallHeat[f]=result.solidWallRadiationEnergyJ[patch][local];}
    setWindowRadiationEnergy(backend,wallHeat);state.radiationTime=state.time;
    externalHeat=-result.escapedBoundaryRadiationEnergyJ;

    return true;
}
}
