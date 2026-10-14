// Actual OpenFOAM material/film driver with prescribed, stage-matched gas history.
// This tests pressure adaptation and the ledger, not native gas execution.
#include "ablation/WallClosureHost.H"
#include "tests/NativeMaterialProbe.H"
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
    WallClosureHost geometry;assert(geometry.prepareGeometry(base.gasMesh,base.surface.gasFace,error));
    std::vector<GasWallMatchingSample> samples;assert(geometry.sample(base,p,samples,error));for(const auto& s:samples)assert(std::abs(s.pressure-stagePressure)<1e-8);
    IntervalHistory history;CouplingInterval interval;interval.identity.sequence=1;interval.end=dt;assert(history.begin(interval,error));
    GasIntervalRecord record;record.microSequence=1;record.end=dt/2;record.gasTraceTime=0;record.gasGeometry=base.gasMesh.geometryVersion;
    record.gasWallMatching=samples;record.gasGradient.resize(4);record.gasTraction.resize(4);record.radiationEnergy.resize(4);record.radiationReceiver.assign(4,ConsumeFilm);
    for(int face:base.surface.gasFace){GasPrimitive w;const int c=base.gasMesh.owner[face];assert(recoverGas(base.gas[c],base.gasMesh.volumes[c],p,w));record.gasTrace.push_back(w);}
    assert(history.appendAccepted(record,error));
    auto second=record;second.microSequence=2;second.begin=dt/2;second.end=dt;second.gasTraceTime=dt/2;
    for(auto& sample:second.gasWallMatching)sample.pressure=180000;
    assert(history.appendAccepted(second,error));
    HostState endpoint=base;endpoint.time=dt;endpoint.gasMesh.geometryVersion+=1;setGas(endpoint,90000,endpointPressure);
    Real gasDelta=0;for(std::size_t c=0;c<base.gas.size();++c)gasDelta+=endpoint.gas[c].energy-base.gas[c].energy;
    endpoint.budget.boundaryEnergy=-gasDelta; // prescribed gas history supplied by its external owner
    // A supplied family identity previously refused the actual LowRe branch.
    // Terminal retains that branch's existing expressions; no new wall model.
    p.wallModel.family=static_cast<ugkwp::gaswall::WallFamily>(99);
    p.tolerances.maxRetries=0;p.tolerances.maxCouplingIterations=0;
    CpuMaterialDriver driver(model,&mesh);CpuMaterialControls controls;controls.maxSubstep=dt;controls.driveRelativeTolerance=1;
    HostState result;WallProgram corrected;CpuMaterialReport report;
    assert(!driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));
    assert(report.failureOrigin==MaterialFailureOrigin::SurfaceNumerical);
    chmt_test::resetMaterialObservations();
    if(!detail::ConductionContinuation::complete(driver,base,endpoint,history,controls,result,corrected,report,error,nullptr)){std::cerr<<error<<'\n';return 3;}
    assert(result.time==dt&&corrected.knots.back().faces.size()==4);
    for(const auto& face:corrected.knots.back().faces)assert(face.temperature==3000);
    assert(report.materialSteps==1&&chmt_test::filmCandidateCalls==1&&chmt_test::materialSolveCalls>0&&corrected.knots.size()>=2&&corrected.knots.front().faces.size()==4);
    chmt_test::physicalSnapshot("surface_terminal",result,corrected);
    std::cout<<"actual native surface early/history/endpoint publication time "<<result.time<<" material "<<report.materialSteps<<" film "<<chmt_test::filmCandidateCalls<<" solves "<<chmt_test::materialSolveCalls<<'\n';
}
