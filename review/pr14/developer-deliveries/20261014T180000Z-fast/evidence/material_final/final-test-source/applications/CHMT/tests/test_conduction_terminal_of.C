// Actual OF10 conduction function and full material owner; no gas/GPU execution.
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
long double solidEnergyChange(const HostState& before,const HostState& after){
    long double change=0;for(std::size_t c=0;c<before.solid.size();++c)change+=static_cast<long double>(after.solid[c].energy)-before.solid[c].energy;return change;
}
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
    HostState normal=base;CpuMaterialReport nr;
    assert(nativeConduction(mesh,base,normal,p,dt,controls,nr,error));
    const auto normalSolves=chmt_test::materialSolveCalls;
    HostState replay=base;CpuMaterialReport rr;chmt_test::resetMaterialObservations();
    assert(nativeConduction<true>(mesh,base,replay,p,dt,controls,rr,error));
    assert(normalSolves==chmt_test::materialSolveCalls&&solidEnergyChange(base,normal)==solidEnergyChange(base,replay)&&chmt_test::boundaryHeat(normal,p,dt)==chmt_test::boundaryHeat(replay,p,dt));
    for(std::size_t c=0;c<base.solid.size();++c)assert(normal.solid[c].energy==replay.solid[c].energy&&normal.solid[c].porosity==replay.solid[c].porosity);
    std::cout<<std::hexfloat<<"normal "<<normalSolves<<' '<<solidEnergyChange(base,normal)<<' '<<chmt_test::boundaryHeat(normal,p,dt);
    for(const auto& q:normal.solid)std::cout<<' '<<q.energy<<' '<<q.porosity;std::cout<<'\n';
    for(int iterations:{0,1}){
        CpuMaterialControls limited=controls;limited.nonlinearMaxIterations=iterations;limited.nonlinearRelativeTolerance=0;limited.energyAbsoluteTolerance=0;limited.energyRelativeTolerance=0;
        HostState stopped=base;CpuMaterialReport stoppedReport;
        assert(!nativeConduction(mesh,base,stopped,p,dt,limited,stoppedReport,error));
        assert(stoppedReport.failureOrigin==MaterialFailureOrigin::ConductionNumerical);
        HostState completed=base;CpuMaterialReport completedReport;chmt_test::resetMaterialObservations();
        assert(nativeConduction<true>(mesh,base,completed,p,dt,limited,completedReport,error));
        assert(chmt_test::materialSolveCalls==1);
        bool changed=false;for(std::size_t c=0;c<base.solid.size();++c){changed=changed||completed.solid[c].energy!=base.solid[c].energy;MaterialPrimitive actual;assert(recoverMaterial<true>(completed.solid[c],base.solidMesh.volumes[c],p,actual));assert(completed.solid[c].porosity==actual.porosity);}
        assert(changed);std::cout<<"terminal "<<iterations<<' '<<solidEnergyChange(base,completed)<<' '<<chmt_test::boundaryHeat(completed,p,dt)<<'\n';
    }
    {
        HostState heated=base;heated.solidMesh.thermalBoundary.assign(heated.solidMesh.owner.size(),ThermalBoundaryKind::ZeroGradient);
        for(std::size_t f=0;f<heated.solidMesh.owner.size();++f)if(heated.solidMesh.neighbour[f]<0){heated.solidMesh.thermalBoundary[f]=ThermalBoundaryKind::FixedValue;heated.solidMesh.boundaryPrimitive[f].temperature=900;}
        HostState evaluated=heated;CpuMaterialReport fluxReport;
        CpuMaterialControls limited=controls;limited.nonlinearMaxIterations=0;
        chmt_test::resetMaterialObservations();assert(nativeConduction<true>(mesh,heated,evaluated,p,dt,limited,fluxReport,error));
        assert(chmt_test::materialSolveCalls==1&&chmt_test::boundaryHeat(evaluated,p,dt)!=0);
        std::cout<<"actual fixed-boundary integral "<<chmt_test::boundaryHeat(evaluated,p,dt)<<'\n';
    }
    // Keep all indexed state arrays complete; omit the temperature that the
    // fixed boundary genuinely consumes, rather than invoking an out-of-bounds cell.
    HostState wrong=base;wrong.solidMesh.thermalBoundary.assign(mesh.nFaces(),ThermalBoundaryKind::FixedValue);
    wrong.solidMesh.boundaryPrimitive.clear();CpuMaterialReport wrongReport;chmt_test::resetMaterialObservations();
    assert(!nativeConduction<true>(mesh,base,wrong,p,dt,controls,wrongReport,error));
    assert(wrongReport.failureOrigin==MaterialFailureOrigin::None&&chmt_test::materialSolveCalls==0);
    assert(error=="CPU conduction consumed boundary temperature missing");
    CouplingInterval interval;interval.end=dt;interval.identity.sequence=1;IntervalHistory history;assert(history.begin(interval,error));
    GasIntervalRecord record;record.end=dt;record.microSequence=1;assert(history.appendAccepted(record,error));
    HostState endpoint=base;endpoint.time=dt;CpuMaterialDriver driver(model,&mesh);HostState result;WallProgram corrected;CpuMaterialReport report;
    controls.nonlinearMaxIterations=0;
    assert(!driver.advanceCandidate(base,endpoint,history,controls,result,corrected,report,error));
    assert(report.failureOrigin==MaterialFailureOrigin::ConductionNumerical);
    chmt_test::resetMaterialObservations();
    if(!detail::ConductionContinuation::complete(driver,base,endpoint,history,controls,result,corrected,report,error,nullptr)){std::cerr<<error<<'\n';return 3;}
    assert(chmt_test::materialSolveCalls==1&&result.time==dt&&!corrected.knots.empty());
    chmt_test::physicalSnapshot("conduction_terminal",result,corrected);
    std::cout<<"full material terminal publication "<<report.materialSteps<<' '<<chmt_test::materialSolveCalls<<'\n';
}
