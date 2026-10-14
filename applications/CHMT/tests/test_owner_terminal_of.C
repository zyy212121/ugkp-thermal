// Actual OF10 material owner with transport/control continuation; no GPU execution.
#include "tests/NativeMaterialProbe.H"
#include "restart/Checkpoint.H"
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
#include "../../../tests/fixtures/owner_terminal/program.hpp"
static bool countAllocations=false;static unsigned long long ownerAllocations=0,ownerBytes=0;
void* operator new(std::size_t n){if(countAllocations){++ownerAllocations;ownerBytes+=n;}if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,std::size_t)noexcept{std::free(p);}
int main(int argc,char** argv){
    Foam::argList::noParallel();Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
    Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
    ModelConfig model;auto& p=model.physics;p.gasMode=ugkwp::GasMode::MixtureFrozen;
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=2;c.conductivity=1;c.Tmin=100;c.Tmax=3000;}
    for(auto& t:p.species){t.R=300;t.cp0=1000;t.Tmin=100;t.Tmax=3000;}
    p.liquid=p.condensed[0];p.spatialOrder=1;p.permeability=1e-14;p.poreViscosity=1e-5;
    HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());
    for(std::size_t c=0;c<base.solid.size();++c){auto& q=base.solid[c];q.condensed[0]=900*base.solidMesh.volumes[c];q.pore[0]=.1*base.solidMesh.volumes[c];q.energy=q.condensed[0]*condensedE(p.condensed[0],400+25*c)+q.pore[0]*speciesE(p.species[0],400+25*c);}
    std::string error;const Real dt=.01;CpuMaterialControls controls;
    p.tolerances.maxRetries=2;p.tolerances.maxCouplingIterations=2;
    CouplingInterval interval;interval.end=dt;interval.identity.sequence=1;IntervalHistory history;assert(history.begin(interval,error));
    GasIntervalRecord record;record.end=dt;record.microSequence=1;assert(history.appendAccepted(record,error));
    HostState endpoint=base;endpoint.time=dt;CpuMaterialDriver driver(model,&mesh);HostState result;WallProgram corrected;CpuMaterialReport report;
    countAllocations=true;assert(driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));countAllocations=false;
    const auto normal=result;const auto normalReport=report;const auto normalSolves=chmt_test::materialSolveCalls,normalRhs=chmt_test::materialRhsCalls;
    std::cout<<std::hexfloat<<"OWNER_PROGRAM ";programOut(corrected);std::cout<<"OWNER_COST "<<ownerAllocations<<' '<<ownerBytes<<'\n';
    assert(normal.gasWallDiagnostics.empty());
    chmt_test::physicalSnapshot("owner_normal",normal,corrected);

    if(const char* path=std::getenv("CHMT_PARITY_PATH"))assert(writeCheckpoint(path,model,normal,error));
    std::cout<<std::hexfloat<<"OWNER_NORMAL "<<normal.time<<' '<<normalReport.materialSteps<<' '<<normalRhs<<' '<<normalSolves<<'\n';
#if F2_TERMINAL
    for(int mode=0;mode<3;++mode){HostState changed=endpoint;IntervalHistory supplied=history;CpuMaterialControls requested=controls;
        if(mode==0)changed.solidMesh.topologyHash^=1;
        if(mode==1){assert(supplied.begin(interval,error));auto extra=record;extra.end=2*dt;assert(supplied.appendAccepted(extra,error));assert(!supplied.complete());}
        if(mode==2)requested.driveRelativeTolerance=-1;
        assert(!driver.advanceCandidate(base,changed,supplied,requested,result,corrected,report,error));
        assert(report.failureOrigin==MaterialFailureOrigin::OwnerNumerical);
        chmt_test::resetMaterialObservations();
        if(!detail::ConductionContinuation::complete(driver,base,changed,supplied,requested,result,corrected,report,error,nullptr)){std::cerr<<error<<'\n';return 3;}
        assert(result.time==dt&&corrected.knots.front().time==base.time&&corrected.knots.back().time==dt);
        assert(report.materialSteps==1&&chmt_test::materialRhsCalls==2&&chmt_test::materialSolveCalls==normalSolves);
        for(std::size_t c=0;c<base.solid.size();++c){assert(result.solid[c].energy==normal.solid[c].energy);assert(result.solid[c].condensed[0]==normal.solid[c].condensed[0]);}
    }
    auto interfaceTail=base;for(auto& kind:interfaceTail.solidMesh.boundaryKind)kind=BoundaryKind::Interface;interfaceTail.solidMesh.thermalBoundary.resize(1);
    CpuMaterialReport directReport;chmt_test::resetMaterialObservations();assert(nativeConduction<true>(mesh,base,interfaceTail,p,dt,controls,directReport,error));assert(chmt_test::materialSolveCalls>0);
    CpuMaterialDriver missing(model,nullptr);assert(!detail::ConductionContinuation::complete(missing,base,endpoint,history,controls,result,corrected,report,error,nullptr));assert(report.failureOrigin==MaterialFailureOrigin::None);
    IntervalHistory unavailable;assert(unavailable.begin(interval,error));assert(!detail::ConductionContinuation::complete(driver,base,endpoint,unavailable,controls,result,corrected,report,error,nullptr));assert(report.failureOrigin==MaterialFailureOrigin::None);
    std::cout<<"actual final-owner metadata/history/control tails and absent inputs passed\n";
#endif
}
