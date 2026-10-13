// Actual OpenFOAM material/film driver with prescribed, stage-matched gas history.
// This tests pressure adaptation and the ledger, not native gas execution.
#include "ablation/WallClosureHost.H"
#include "fvCFD.H"
#include "materials/CpuMaterialDriver.C"
#include "materials/ConductionContinuation.H"
#include "mesh/TrajectorySurface.H"
#include <cassert>
#include <iostream>
namespace {
chmt::HostMesh testMesh(const Foam::fvMesh& mesh){
    chmt::HostMesh out;
    for(const auto& p:mesh.points())out.points.push_back({p.x(),p.y(),p.z()});
    out.oldPoints=out.points;out.referencePoints=out.points;
    const int nf=mesh.faces().size();out.faceOffsets.push_back(0);
    out.neighbour.assign(nf,-1);out.periodicPartner.assign(nf,-1);
    out.boundaryKind.assign(nf,chmt::BoundaryKind::Internal);out.boundaryPrimitive.resize(nf);out.boundarySst.resize(nf);
    for(int f=0;f<nf;++f){out.owner.push_back(mesh.faceOwner()[f]);
        if(f<mesh.faceNeighbour().size())out.neighbour[f]=mesh.faceNeighbour()[f];
        for(int point:mesh.faces()[f])out.facePoints.push_back(point);
        out.faceOffsets.push_back(out.facePoints.size());out.faceIds.push_back(f);}
    for(const auto& patch:mesh.boundaryMesh()){assert(patch.type()=="wall");
        for(int f=0;f<patch.size();++f)out.boundaryKind[patch.start()+f]=chmt::BoundaryKind::NoSlip;}
    std::string error;assert(chmt::rebuildGeometry(out,error));out.oldVolumes=out.volumes;return out;
}
}
using namespace chmt;
int main(int argc,char** argv){
    Foam::argList::noParallel();Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
    Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
    ModelConfig model;auto& p=model.physics;p.enableFilm=true;p.gasMode=ugkwp::GasMode::MixtureFrozen;
    p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
    p.spatialOrder=1;p.minDt=1e-12;p.maxDt=1;p.cfl=.4;p.liquidViscosity=1;p.liquidReferencePressure=100000;p.meltTemperature=500;
    p.gasViscosity=.01;p.gasConductivity=1;
    for(auto& s:p.species){s.R=300;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.conductivity=1;c.Tmin=100;c.Tmax=3000;}
    p.liquid=p.condensed[0];p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1;
    HostState base;base.solidMesh=testMesh(mesh);base.gasMesh=testMesh(mesh);std::string error;
    const Real thickness=.001,initialPressure=80000,stagePressure=150000,endpointPressure=210000,dt=1e-4;
    for(auto& x:base.solidMesh.points)x.x-=1;for(auto& x:base.gasMesh.points)x.x+=thickness;
    for(auto* m:{&base.solidMesh,&base.gasMesh}){m->referencePoints=m->oldPoints=m->points;assert(rebuildGeometry(*m,error));m->oldVolumes=m->volumes;}
    for(std::size_t f=0;f<base.gasMesh.owner.size();++f)if(base.gasMesh.neighbour[f]<0)base.gasMesh.boundaryKind[f]=BoundaryKind::Slip;
    for(std::size_t sf=0;sf<base.solidMesh.owner.size();++sf)if(base.solidMesh.neighbour[sf]<0&&base.solidMesh.areaVectors[sf].x>0){
        int gf=-1;for(std::size_t f=0;f<base.gasMesh.owner.size();++f)if(base.gasMesh.neighbour[f]<0&&base.gasMesh.areaVectors[f].x<0
            &&std::abs(base.gasMesh.faceCentres[f].y-base.solidMesh.faceCentres[sf].y)<1e-12
            &&std::abs(base.gasMesh.faceCentres[f].z-base.solidMesh.faceCentres[sf].z)<1e-12)gf=f;
        assert(gf>=0);base.surface.solidFace.push_back(sf);base.surface.gasFace.push_back(gf);base.surface.solidCell.push_back(base.solidMesh.owner[sf]);base.surface.persistentId.push_back(sf);
        base.solidMesh.boundaryKind[sf]=base.gasMesh.boundaryKind[gf]=BoundaryKind::Interface;
    }
    assert(base.surface.gasFace.size()==4);base.filmAux.resize(4);for(auto& a:base.filmAux)a.thickness=thickness;
    assert(rebuildTrajectorySurface(base,base.gasMesh,base.solidMesh,dt,base.surface,error));
    base.solid.resize(mesh.nCells());for(std::size_t c=0;c<base.solid.size();++c){base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];base.solid[c].energy=base.solid[c].condensed[0]*500000;}
    auto setGas=[&](HostState& state,Real ownerP,Real matchP){state.gas.resize(state.gasMesh.volumes.size());
        for(std::size_t c=0;c<state.gas.size();++c){GasPrimitive w;w.temperature=500;w.pressure=state.gasMesh.cellCentres[c].x<thickness+.5?ownerP:matchP;w.rho=w.pressure/(300*500);w.Y[0]=1;
            state.gas[c]=conservativeGas(w,state.gasMesh.volumes[c],p);}};
    setGas(base,100000,stagePressure);base.film.resize(4);
    for(std::size_t f=0;f<4;++f){auto& q=base.film[f];q.mass=1000*thickness*base.surface.area[f];q.species[0]=q.mass;q.enthalpy=q.mass*liquidH(p,500,initialPressure);
        assert(recoverFilm(q,base.surface.area[f],initialPressure,p,base.filmAux[f]));}

    CouplingInterval interval;interval.identity.sequence=7;interval.end=dt;
    WallProgram program;CpuMaterialReport report;CpuMaterialDriver normal(model,&mesh);
    if(!normal.predictWall(base,interval,program,error,&report)){std::cerr<<error<<'\n';return 3;}
    assert(program.knots.size()==2&&program.knots.front().time==0&&program.knots.back().time==dt);
    assert(program.knots.front().faces.size()==4&&program.knots.back().gasPoints.size()==base.gasMesh.points.size());
    // Required host gas caloric tail: the ordinary EOS refuses supplied energy.
    ModelConfig host=model;host.physics.wallModel.family=static_cast<ugkwp::gaswall::WallFamily>(99);
    CpuMaterialDriver terminal(host,&mesh);HostState inconsistent=base;
    for(auto& q:inconsistent.gas)q.energy+=1e10;
    assert(!terminal.predictWall(inconsistent,interval,program,error,&report));
    assert(report.failureOrigin==MaterialFailureOrigin::SurfaceNumerical);
    if(!detail::ConductionContinuation::predict(terminal,inconsistent,interval,program,error,&report)){std::cerr<<error<<'\n';return 4;}
    assert(program.knots.size()==2&&program.knots.back().time==dt&&program.knots.back().faces.size()==4);
    for(const auto& face:program.knots.back().faces)assert(face.temperature==3000);
    // No recorded matching snapshot: consume the supplied gas through the
    // actual host sampling, closure contexts and CPU surface continuation.
    HostState supplied=base;for(auto& q:supplied.gas)q.energy=NAN;
    GasPrimitive absent;assert(!recoverGas(supplied.gas[0],supplied.gasMesh.volumes[0],p,absent));
    if(!detail::ConductionContinuation::predict(normal,supplied,interval,program,error,&report)){std::cerr<<error<<'\n';return 6;}
    assert(program.knots.size()==2&&program.knots.back().time==dt&&program.knots.back().faces.size()==4);
    for(const auto& face:program.knots.back().faces)assert(face.temperature!=0);
    supplied.gas.clear();assert(!detail::ConductionContinuation::predict(normal,supplied,interval,program,error,&report));
    ModelConfig moving=model;moving.physics.meshMotion.policy=MeshMotionPolicy::CoupledRecession;
    CpuMaterialDriver movingDriver(moving,&mesh);HostState movedBase=base;movedBase.solidSweepRemainder.assign(4,.01);
    assert(!movingDriver.predictWall(movedBase,interval,program,error,&report));
    assert(report.failureOrigin==MaterialFailureOrigin::GeometryNumerical);
    if(!detail::ConductionContinuation::predict(movingDriver,movedBase,interval,program,error,&report)){std::cerr<<error<<'\n';return 5;}
    assert(program.knots.size()==2&&program.knots.back().gasPoints.size()==base.gasMesh.points.size()&&program.knots.back().solidPoints.size()==base.solidMesh.points.size());
    assert(program.knots.back().time==dt);
    inconsistent.gas.clear();
    assert(!detail::ConductionContinuation::predict(terminal,inconsistent,interval,program,error,&report));
    assert(report.failureOrigin==MaterialFailureOrigin::None);
    std::cout<<"actual OF10 predictor: 4 faces, actual EOS/surface knots dt "<<dt<<", missing gas fails\n";
}
