// Actual OF10 material owner with inherited remainder continuation; no GPU execution.
#include "tests/NativeMaterialProbe.H"
#include "materials/ConductionContinuation.H"
#include <cassert>
#include <iomanip>
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
    ModelConfig model;auto& p=model.physics;p.gasMode=ugkwp::GasMode::MixtureFrozen;
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=2;c.conductivity=1;c.Tmin=100;c.Tmax=3000;}
    for(auto& t:p.species){t.R=300;t.cp0=1000;t.Tmin=100;t.Tmax=3000;}
    p.liquid=p.condensed[0];p.spatialOrder=1;
    HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());
    for(std::size_t c=0;c<base.solid.size();++c){auto& q=base.solid[c];q.condensed[0]=1000*base.solidMesh.volumes[c];q.energy=q.condensed[0]*condensedE(p.condensed[0],400+25*c);}
    std::string error;const Real dt=.01;CpuMaterialControls controls;
    p.tolerances.maxRetries=0;p.tolerances.maxCouplingIterations=0;
    CouplingInterval interval;interval.end=dt;interval.identity.sequence=1;IntervalHistory history;assert(history.begin(interval,error));
    GasIntervalRecord record;record.end=dt;record.microSequence=1;assert(history.appendAccepted(record,error));
    HostState endpoint=base;endpoint.time=dt;CpuMaterialDriver driver(model,&mesh);HostState result;WallProgram corrected;CpuMaterialReport report;
    assert(driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));
    const auto normal=result;const auto normalReport=report;const auto normalSolves=chmt_test::materialSolveCalls;
    base.solidSweepRemainder={7};endpoint.solidSweepRemainder=base.solidSweepRemainder;
    assert(!driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));
    assert(report.failureOrigin==MaterialFailureOrigin::GeometryNumerical);
    chmt_test::resetMaterialObservations();
    if(!detail::ConductionContinuation::complete(driver,base,endpoint,history,controls,result,corrected,report,error,nullptr)){std::cerr<<error<<'\n';return 3;}
    assert(result.time==dt&&!corrected.knots.empty());assert(result.solidSweepRemainder==base.solidSweepRemainder);
    assert(chmt_test::materialSolveCalls==normalSolves&&report.materialSteps==normalReport.materialSteps);
    assert(result.solidMesh.volumes==normal.solidMesh.volumes&&result.solidMesh.oldVolumes==normal.solidMesh.oldVolumes);
    for(std::size_t c=0;c<base.solid.size();++c){assert(result.solid[c].energy==normal.solid[c].energy);assert(result.solid[c].condensed[0]==normal.solid[c].condensed[0]);}
    chmt_test::physicalSnapshot("sweep_terminal",result,corrected);
    std::cout<<"actual material inherited remainder publication paid_dt "<<dt<<" solves "<<chmt_test::materialSolveCalls<<" cells "<<result.solidMesh.volumes.size()<<'\n';
}
