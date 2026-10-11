#define main inherited_sweep_tests_main
#include "test_sweep_constraints.cpp"
#undef main
#include "restart/Checkpoint.H"
#include "materials/Reaction.H"
#include "mesh/EmptyExtrusion.H"
#include <unistd.h>

namespace {
HostMesh emptyMesh() {
    HostMesh mesh=twoHexahedra(0);
    for(std::size_t f=0;f<mesh.owner.size();++f)
        if(mesh.neighbour[f]<0 && absValue(mesh.areaVectors[f].y)>0)mesh.boundaryKind[f]=BoundaryKind::Empty;
    std::string error;require(rebuildGeometry(mesh,error),error);return mesh;
}
HostMesh checkTrajectory(const HostMesh& mesh,const std::vector<Vec3>& endpoint) {
    HostMesh moved;std::vector<Real> sweep;std::string error;
    require(makeStageGeometry(mesh,endpoint,1,moved,sweep,error),error);
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==BoundaryKind::Empty)
        require(sweep[f]==0,"certified Empty sweep is strict zero without clearing a computed value");
    require(maximumGclResidual(mesh,moved,sweep)<1e-11,"Empty trajectory shares volume/GCL geometry");
    for(int k=0;k<2;++k)for(int i=0;i<3;++i) {
        const int a=i+6*k,b=a+3;
        require(moved.points[a].x==moved.points[b].x && moved.points[a].z==moved.points[b].z,
            "paired-layer tangential coordinates stay identical");
        require(moved.points[a].y==mesh.points[a].y && moved.points[b].y==mesh.points[b].y,
            "both normal layer coordinates stay fixed");
    }
    return moved;
}
void legalTrajectories() {
    for(Real shift:{Real(0),Real(1000)}) {
        auto mesh=emptyMesh();for(auto& p:mesh.points)p+=Vec3{shift,shift,shift};
        std::string error;require(rebuildGeometry(mesh,error),error);
        auto endpoint=mesh.points;for(auto& p:endpoint)p+=Vec3{.03125,0,-.0625};
        mesh=checkTrajectory(mesh,endpoint);
        endpoint=mesh.points;for(auto& p:endpoint)p.x+=.125*(p.z-shift);
        mesh=checkTrajectory(mesh,endpoint);
        endpoint=mesh.points;for(std::size_t p=0;p<endpoint.size();++p)if(p>=6)endpoint[p].z-=.03125;
        checkTrajectory(mesh,endpoint);
    }
    for(int axis:{0,2}) {
        auto mesh=emptyMesh();
        for(auto& p:mesh.points)p=axis==0?Vec3{p.y,p.z,p.x}:Vec3{p.z,p.x,p.y};
        std::string error;require(rebuildGeometry(mesh,error),error);auto end=mesh.points;
        for(auto& p:end)p+=axis==0?Vec3{0,.03125,-.0625}:Vec3{.03125,-.0625,0};
        HostMesh moved;std::vector<Real> sweep;
        require(makeStageGeometry(mesh,end,1,moved,sweep,error),error);
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==BoundaryKind::Empty)
            require(sweep[f]==0,"all coordinate-axis extrusion orientations retain strict zero");
    }
}
void rejectUnrepresentedMotion() {
    const auto mesh=emptyMesh();std::string error;
    for(Real normal:{Real(1e-31),Real(1e-8)}) {
        auto endpoint=mesh.points;endpoint[0].y+=normal;
        HostMesh output;output.geometryVersion=776;std::vector<Real> sweep{123};
        require(!makeStageGeometry(mesh,endpoint,1,output,sweep,error),"Empty nonzero normal trajectory rejected");
        require(error.find("Empty")!=std::string::npos,"Empty support diagnostic: "+error);
        require(output.geometryVersion==776 && sweep==std::vector<Real>{123},"Empty rejection preserves outputs");
    }
    auto endpoint=mesh.points;endpoint[0].x+=1e-31;
    HostMesh output;std::vector<Real> sweep;
    require(!makeStageGeometry(mesh,endpoint,1,output,sweep,error),"unpaired tangential motion rejected");
    auto translated=mesh;for(auto& p:translated.points)p.y+=2;
    require(rebuildGeometry(translated,error),error);
    std::vector<Vec3> input(translated.points.size()),sentinel{{7,8,9}};
    std::vector<unsigned char> fixed(translated.points.size(),1);input[0].y=input[3].y=1e-31;
    require(!harmonicPointMotion(translated,input,fixed,sentinel,error),
        "normal input rejected before endpoint addition can round it away");
    require(sentinel.size()==1 && sentinel[0].x==7,"failed harmonic motion preserves output");
    auto rotated=mesh;
    for(auto& p:rotated.points)p={.8*p.x-.6*p.y,.6*p.x+.8*p.y,p.z};
    const auto version=rotated.geometryVersion;
    require(!rebuildGeometry(rotated,error),"rotated Empty is explicitly outside first supported representation");
    require(rotated.geometryVersion==version && error.find("Empty")!=std::string::npos,"unsupported Empty stays transactional");
    auto ordinary=twoHexahedra(0);for(auto& p:ordinary.points)p={.8*p.x-.6*p.y,.6*p.x+.8*p.y,p.z};
    require(rebuildGeometry(ordinary,error),"non-Empty rotated geometry is unaffected: "+error);
}
void pairedCorrectionDofs() {
    for(Real a:{Real(1e-100),Real(3),Real(1e100)}) {
        const Real b=2*a,scale=empty_detail::pairedScale(a,b),q=.75,alpha=q*scale;
        require(absValue((alpha/a)*(alpha/a)+(alpha/b)*(alpha/b)-q*q)<1e-15,
            "group scale preserves original pair-weighted minimum-norm metric");
    }
    const auto mesh=emptyMesh();std::vector<Vec3> direction(mesh.points.size()),zero(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1);std::vector<int> faces;
    for(std::size_t p=0;p<mesh.points.size();++p)if(mesh.points[p].z==1)direction[p]={0,0,1};
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.faceCentres[f].z==1 && mesh.areaVectors[f].z>0)faces.push_back(f);
    HostMesh output;std::vector<Real> sweep;SweepConstraintReport report;std::string error;
    require(constrainFaceSweeps(mesh,faces,{-.03,-.08},zero,fixed,direction,1,output,sweep,report,error),error);
    require(report.degreesOfFreedom==3,"one tangential correction DOF per paired layer vertex");
    checkTrajectory(mesh,output.points);
    direction[6].y=1e-31;output.geometryVersion=882;sweep={456};
    require(!constrainFaceSweeps(mesh,faces,{-.03,-.08},zero,fixed,direction,1,output,sweep,report,error),
        "tiny normal correction direction is not projected away");
    require(output.geometryVersion==882 && sweep==std::vector<Real>{456},"invalid Empty direction preserves outputs");
}
void coupledEmptyTrajectory() {
    auto base=coupledPair();std::string error;
    for(auto* mesh:{&base.gasMesh,&base.solidMesh}) {
        for(std::size_t f=0;f<mesh->owner.size();++f)
            if(mesh->neighbour[f]<0 && absValue(mesh->areaVectors[f].y)>0)mesh->boundaryKind[f]=BoundaryKind::Empty;
        require(rebuildGeometry(*mesh,error),error);
    }
    HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;
    require(moveCoupledMeshesConstrained(base,base.filmAux,{-.03,-.08},1,gas,solid,surface,report,error),error);
    checkTrajectory(base.solidMesh,solid.points);checkTrajectory(base.gasMesh,gas.points);
    require(surface.area.size()==2 && surface.sweptEdgeArea.size()>0,"coupled Empty surface uses the same trajectory");
}
void restartRepresentation() {
    ModelConfig model;auto& physics=model.physics;physics.enableGas=false;
    physics.minDt=1e-9;physics.maxDt=1;physics.cfl=.4;physics.modelFingerprint=1;
    for(int c=0;c<Nc;++c){auto& t=physics.condensed[c];t.rho=1000;t.cp0=1000;t.conductivity=2;t.Tmin=100;t.Tmax=2000;}
    HostState state;state.solidMesh=emptyMesh();auto& mesh=state.solidMesh;
    mesh.oldPoints=mesh.referencePoints=mesh.points;mesh.oldVolumes=mesh.volumes;
    mesh.boundaryPrimitive.resize(mesh.owner.size());mesh.boundarySst.resize(mesh.owner.size());
    state.solid.resize(mesh.volumes.size());
    for(std::size_t c=0;c<state.solid.size();++c) {
        state.solid[c].condensed[0]=1000*mesh.volumes[c];
        state.solid[c].energy=state.solid[c].condensed[0]*condensedE(physics.condensed[0],500);
    }
    state.time=1;state.acceptedSteps=state.commitSequence=1;state.lastAcceptedDt=1;state.nextDt=.1;
    for(auto& stage:state.solidStages) {
        stage.interval=1;stage.geometryVersion=mesh.geometryVersion;stage.topologyHash=mesh.topologyHash;
        stage.oldVolume=stage.newVolume=stage.evaluationVolume=mesh.volumes;stage.sweptVolume.assign(mesh.owner.size(),0);
        stage.areaVector=mesh.areaVectors;stage.cellCentre=mesh.cellCentres;stage.faceCentre=mesh.faceCentres;
        stage.oldPoints=stage.newPoints=mesh.points;
    }
    char root[]="/tmp/chmt-empty-restart.XXXXXX";require(mkdtemp(root)!=nullptr,"Empty restart temporary directory");
    const std::string path=std::string(root)+"/accepted";std::string error;HostState restored;
    require(writeCheckpoint(path,model,state,error),error);require(readCheckpoint(path,model,restored,error),error);
    auto endpoint=mesh.points;for(auto& p:endpoint)p.x+=.0625*p.z;
    const auto before=checkTrajectory(mesh,endpoint),after=checkTrajectory(restored.solidMesh,endpoint);
    require(before.volumes==after.volumes,"Empty representation reconstructed identically after restart");
    unlink((path+"/state.chmt").c_str());rmdir(path.c_str());rmdir(root);
}
}
int main() {
    rejectUnrepresentedMotion();legalTrajectories();pairedCorrectionDofs();coupledEmptyTrajectory();restartRepresentation();
    std::cout << "PASS: axis-aligned paired Empty extrusion, exact zero trajectories, DOFs, invalid motion and restart\n";
}
