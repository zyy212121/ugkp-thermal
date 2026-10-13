// Actual OF10 material owner with reaction terminal continuation; no GPU execution.
#include "materials/CpuMaterialDriver.C"
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
    p.enableReactions=true;p.nReactions=1;p.tolerances.maxRetries=0;p.tolerances.maxCouplingIterations=0;
    p.reactions[0].A=1;p.reactions[0].order[0]=1;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;
    CouplingInterval interval;interval.end=dt;interval.identity.sequence=1;IntervalHistory history;assert(history.begin(interval,error));
    GasIntervalRecord record;record.end=dt;record.microSequence=1;assert(history.appendAccepted(record,error));
    HostState endpoint=base;endpoint.time=dt;CpuMaterialDriver driver(model,&mesh);HostState result;WallProgram corrected;CpuMaterialReport report;
    controls.maxSubsteps=1;controls.reactionMaxSubstep=.004;
    assert(!driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));
    assert(report.failureOrigin==MaterialFailureOrigin::ReactionNumerical&&report.failedLocalDt==dt);
    if(!detail::ConductionContinuation::complete(driver,base,endpoint,history,controls,result,corrected,report,error,nullptr)){std::cerr<<error<<'\n';return 3;}
    assert(result.time==dt&&!corrected.knots.empty());
    for(std::size_t c=0;c<base.solid.size();++c){
        SolidQ reference;std::uint64_t count=0;bool numerical=false;
        assert(detail::completeMaterialReactions(base.solid[c],base.solidMesh.volumes[c],p,dt,.004,controls.reactionCfl,1,reference,count,error,&numerical));
        assert(count==3&&result.solid[c].condensed[0]==reference.condensed[0]);
        assert(result.solid[c].condensed[1]==reference.condensed[1]&&result.solid[c].progress[0]==reference.progress[0]);
    }
    std::cout<<"actual material reaction publication paid_dt "<<dt<<" steps "<<report.reactionSteps<<'\n';
}
