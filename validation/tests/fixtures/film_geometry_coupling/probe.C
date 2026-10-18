
#include "fvCFD.H"
#include "io/MeshIO.H"
#include "materials/CpuMaterialDriver.H"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stdexcept>

using namespace chmt;

void check(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

bool sameBits(Real a, Real b)
{
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

int main(int argc, char** argv)
{
    Foam::argList::addOption("step", "scalar", "time step");
    Foam::argList::addOption("phaseCase", "word", "unselected, selected, or melting");
    Foam::argList args(argc, argv);
    Foam::Time runTime(Foam::Time::controlDictName, args);
    Foam::fvMesh mesh(Foam::IOobject(Foam::fvMesh::defaultRegion,
        runTime.timeName(), runTime, Foam::IOobject::MUST_READ));
    const Foam::word mode = args.optionLookupOrDefault<Foam::word>("phaseCase", "unselected");
    const bool coupled=mode=="coupled"||mode=="audit"||mode=="wideAudit";
    try {
        ModelConfig model;
        auto& p = model.physics;
        p.enableFilm = mode!="dry";
        if(mode=="audit"){p.tolerances.absoluteMass=0;p.tolerances.relativeMass=1e-315;}
        if(mode=="wideAudit"){p.tolerances.absoluteEnergy=1.6e308;p.tolerances.relativeEnergy=1.6e308;}
        p.spatialOrder = 1;
        p.gasConductivity = 2;
        p.gasViscosity = .01;
        p.liquidViscosity = 1;
        for (auto& s : p.species) {
            s.R = 287; s.cp0 = 1000; s.Tmin = 100; s.Tmax = 3000;
        }
        for (auto& s : p.condensed) {
            s.rho = 1000; s.cp0 = 1000; s.conductivity = 2;
            s.Tmin = 100; s.Tmax = 3000;
        }
        p.liquid = p.condensed[0];
        p.material.phaseFilmY[0] = 1;
        if (mode != "unselected") p.material.phaseCondensed = 0;
        if(coupled){p.meshMotion.policy=MeshMotionPolicy::CoupledRecession;p.gravity={10000,0,0};}
        if (mode == "melting") {
            p.material.enableMelting = true;
            p.meshMotion.policy = MeshMotionPolicy::CoupledRecession;
            p.meltTemperature = 600;
            p.liquid.e0 = 200000;
        }
        check(p.filmThermalMode == FilmThermalMode::ThicknessAveraged,
            "fixture must use ThicknessAveraged");
        check(mode != "unselected" || (p.material.phaseCondensed == -1
            && !p.material.enableMelting), "fixture must retain the unselected default");

        HostState base;
        base.solidMesh = io::importMesh(mesh, Foam::dictionary());
        base.gasMesh = base.solidMesh;
        for (auto& point : base.gasMesh.points) point.x += mode=="dry"?1:1.001;
        std::string error;
        check(rebuildGeometry(base.gasMesh, error), error);
        base.gasMesh.oldPoints = base.gasMesh.referencePoints = base.gasMesh.points;
        base.gasMesh.oldVolumes = base.gasMesh.volumes;
        const int sf = mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("right")].start();
        const int gf = mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("left")].start();
        base.solidMesh.boundaryKind[sf] = BoundaryKind::Interface;
        base.gasMesh.boundaryKind[gf] = BoundaryKind::Interface;
        check(rebuildGeometry(base.solidMesh, error), error);
        check(rebuildGeometry(base.gasMesh, error), error);
        base.solid.resize(mesh.nCells());GasPrimitive gas;gas.rho=1;gas.temperature=800;gas.pressure=229600;gas.Y[0]=1;
        for(int c=0;c<mesh.nCells();++c){base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];base.solid[c].energy=base.solid[c].condensed[0]*condensedE(p.condensed[0],700);base.gas.push_back(conservativeGas(gas,base.gasMesh.volumes[c],p));}
        Foam::IStringStream input("patch right; baseVelocity (0 0 0); prescribedTopTraction (0 0 0); prescribedPressureGradient (0 0 0);");Foam::dictionary spec(input);
        base.surface=io::importSurface(mesh,base.solidMesh,spec).mesh;auto& surface=base.surface;
        for(size_t f=0;f<surface.area.size();++f){surface.gasFace[f]=gf+f;surface.gasDistance[f]=.5;base.solidMesh.boundaryKind[sf+f]=BoundaryKind::Interface;base.gasMesh.boundaryKind[gf+f]=BoundaryKind::Interface;}
        check(rebuildGeometry(base.solidMesh,error),error);check(rebuildGeometry(base.gasMesh,error),error);
        base.film.resize(surface.area.size());base.filmAux.resize(surface.area.size());
        for(size_t f=0;f<base.film.size();++f){auto&q=base.film[f];q.mass=mode=="dry"?0:surface.area[f]*(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession?1:(1+.2*f));q.species[0]=q.mass;q.enthalpy=q.mass*liquidH(p,mode=="melting"?800:800+20*f,gas.pressure);check(recoverFilm(q,surface.area[f],gas.pressure,p,base.filmAux[f]),"film recovery");base.filmAux[f].normal={1,0,0};base.filmAux[f].baseVelocity={0,0,0};}
        CouplingInterval interval;
        interval.identity.sequence = 1;
        interval.end = args.optionLookupOrDefault<Foam::scalar>("step", .001);
        GasIntervalRecord record;
        record.end = interval.end; record.microSequence = 1;
        record.gasGeometry = base.gasMesh.geometryVersion;
        record.solidGeometry = base.solidMesh.geometryVersion;
        record.gasTrace.assign(surface.area.size(),gas); record.gasTraction.assign(surface.area.size(),coupled?Vec3{0,.01,0}:Vec3{});
        if(coupled)for(size_t f=0;f<record.gasTraction.size();++f)record.gasTraction[f]={0,1000*(1+Real(f)),0};
        IntervalHistory history;
        check(history.begin(interval, error) && history.appendAccepted(record, error), error);
        HostState endpoint = base;
        endpoint.time = interval.end;
        if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession)for(auto& point:endpoint.solidMesh.points)point.y*=1+2*interval.end;
        if(p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession)for(auto& point:endpoint.gasMesh.points)point.y*=1+2*interval.end;
        check(rebuildGeometry(endpoint.solidMesh,error),error);check(rebuildGeometry(endpoint.gasMesh,error),error);
        for(size_t c=0;c<endpoint.gas.size();++c)endpoint.gas[c]=conservativeGas(gas,endpoint.gasMesh.volumes[c],p);
        CpuMaterialDriver driver(model, &mesh);
        HostState result;
        WallProgram corrected;
        CpuMaterialReport report;
        const bool advanced = driver.advanceCandidate(base, endpoint, history,
            CpuMaterialControls{}, result, corrected, report, error);
        std::cout<<std::setprecision(17)<<"RESULT dt="<<interval.end<<" success="<<advanced<<" steps="<<report.materialSteps<<" error="<<error<<"\n";
        if(advanced){long double oldMass=0,newMass=0;
            for(const auto& q:base.solid)for(auto v:q.condensed)oldMass+=v;
            for(const auto& q:result.solid)for(auto v:q.condensed)newMass+=v;
            for(const auto& q:base.film)oldMass+=q.mass;
            for(const auto& q:result.film)newMass+=q.mass;
            std::cout<<"TOTAL_MASS "<<oldMass<<' '<<newMass<<"\n";
        }
        if(advanced)for(size_t f=0;f<result.film.size();++f)std::cout<<"FILM "<<f<<" mass="<<result.film[f].mass<<" old="<<base.film[f].mass<<" area="<<result.surface.area[f]<<" enthalpy="<<result.film[f].enthalpy<<"\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
