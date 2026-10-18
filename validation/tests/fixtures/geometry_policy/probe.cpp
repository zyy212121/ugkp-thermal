#define main inherited_geometry_main
#include "../../../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "mesh/SweepWorkspace.H"
#include "mesh/EmptyExtrusion.H"
#include "DiagnosticsHost.H"
int main(int argc,char** argv) {
    const std::string test=argc>1?argv[1]:"";
    std::string error;HostMesh out;std::vector<Real> sweeps;
    if(test=="margin" || test=="tiny-margin") {
        const auto mesh=twoHexahedra(0);auto end=mesh.points;
        for(auto& p:end)p.z*=test=="tiny-margin"?1e-20:2e-15;
        ugkp::diagnostics::DeviceContext diagnostics{};ugkp::diagnostics::reset(diagnostics,1,34);
        Tolerances tol;tol.diagnostics=&diagnostics;
        require(makeStageGeometry(mesh,end,1,out,sweeps,error,tol),"positive thin trajectory: "+error);
        require(out.points[6].z==end[6].z && out.volumes[0]>0,"retain thin endpoint");
        auto snapshot=ugkp::diagnostics::takeSnapshot(diagnostics);
        require(!snapshot.hasFatal && !snapshot.warnings.empty(),"small positive margin warns");
        for(auto& p:end)p.z=-p.z;
        require(!makeStageGeometry(mesh,end,1,out,sweeps,error,tol),"negative volume remains an error");
    } else if(test=="rotation") {
        const auto mesh=twoHexahedra(0);auto end=mesh.points;
        const Real angle=2*std::acos(-1.)/3;
        for(auto& p:end)p={std::cos(angle)*p.x-std::sin(angle)*p.y,std::sin(angle)*p.x+std::cos(angle)*p.y,p.z};
        require(makeStageGeometry(mesh,end,1,out,sweeps,error),"noncollapsing 120-degree path: "+error);
        require(maximumGclResidual(mesh,out,sweeps)<1e-12,"rotated linear path independent GCL");
        for(auto v:out.volumes)require(v>0,"rotated endpoint positive");
    } else if(test=="caps") {
        const auto mesh=twoHexahedra(0);std::vector<Vec3> zero(mesh.points.size()),directions(mesh.points.size(),Vec3{0,0,1});
        std::vector<unsigned char> fixed(mesh.points.size(),1);SweepConstraintReport report;
        for(int policy=0;policy<5;++policy) {
            SweepConstraintControls controls;
            if(policy==0)controls.maximumWorkspaceBytes=1;
            if(policy==1)controls.maximumDegreesOfFreedom=1;
            if(policy==2)controls.maximumConstraints=1;
            if(policy==3)controls.maximumModeEntries=1;
            if(policy==4)controls.maximumWorkspaceBytes=0;
            require(constrainFaceSweeps(mesh,{0},{.01},zero,fixed,directions,1,out,sweeps,report,error,controls),"legacy cap must not reject: "+error);
            require(absValue(sweeps[0]-.01)<1e-10,"retained solve meets sweep");
            require(report.peakWorkspaceBytes>1,"actual allocation accounting retained");
        }
        std::size_t peak=0;sweep_detail::Budget budget(1,peak);
        {sweep_detail::Reals values(32,2,sweep_detail::Allocator<Real>(budget));require(values[31]==2 && peak>=256,"legacy byte cap does not reject allocator");}
        require(budget.live==0,"live accounting released");
        try{sweep_detail::product(std::size_t(-1),2);require(false,"actual overflow must throw");}catch(const sweep_detail::WorkspaceLimit&){}
    } else if(test=="empty") {
        auto mesh=twoHexahedra(0);
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0 && mesh.areaVectors[f].y!=0)mesh.boundaryKind[f]=BoundaryKind::Empty;
        require(rebuildGeometry(mesh,error),error);
        auto end=mesh.points;for(auto& p:end)p.x+=.125;
        require(makeStageGeometry(mesh,end,1,out,sweeps,error),error);
        for(std::size_t f=0;f<sweeps.size();++f)if(mesh.boundaryKind[f]==BoundaryKind::Empty)require(sweeps[f]==0,"supported Empty operator has exact zero flux");
        end[0].y+=1e-31;
        require(!makeStageGeometry(mesh,end,1,out,sweeps,error),"normal motion lacks Empty operator");
        require(error.find("Empty")!=std::string::npos,"representation error explains Empty");
    } else if(test=="empty-nonfinite") {
        auto mesh=twoHexahedra(0);int selected=-1;
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0 && mesh.areaVectors[f].y!=0){mesh.boundaryKind[f]=BoundaryKind::Empty;selected=static_cast<int>(f);}
        mesh.areaVectors[selected].y=std::numeric_limits<Real>::quiet_NaN();
        empty_detail::Extrusion extrusion;
        require(!empty_detail::build(mesh,extrusion,error),"nonfinite Empty area must not become a valid representation");
        require(error.find("nonfinite")!=std::string::npos,"specific consumed nonfinite Empty area diagnostic");
    } else if(test=="multiple-cover") {
        HostMesh mesh;mesh.faceOffsets={0};
        const Real pi=std::acos(-1.);
        for(int p=0;p<5;++p)mesh.points.push_back({std::cos(2*pi*p/5),std::sin(2*pi*p/5),0});
        mesh.points.push_back({0,0,1});mesh.points.push_back({0,0,-1});
        for(int p=0;p<5;++p)for(int side=0;side<2;++side) {
            const int a=(2*p)%5,b=(2*p+2)%5;
            mesh.facePoints.insert(mesh.facePoints.end(),side?std::initializer_list<int>{b,a,6}:std::initializer_list<int>{a,b,5});
            mesh.faceOffsets.push_back(static_cast<int>(mesh.facePoints.size()));
            mesh.owner.push_back(0);mesh.neighbour.push_back(-1);mesh.boundaryKind.push_back(BoundaryKind::Slip);
        }
        require(rebuildGeometry(mesh,error),"positive signed star fixture: "+error);
        auto end=mesh.points;for(auto& p:end)p.x+=.01;
        require(!makeStageGeometry(mesh,end,1,out,sweeps,error),"double-covered fan is still unsupported geometry");
        require(error.find("single star-shaped")!=std::string::npos,"topological degree error is retained");
    } else if(test=="parity") {
        const auto mesh=tetrahedralBulk();std::vector<int> faces;
        std::vector<Vec3> zero(mesh.points.size()),directions(mesh.points.size());
        std::vector<unsigned char> fixed(mesh.points.size(),1);fixed[4]=0;
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0)faces.push_back(static_cast<int>(f));
        for(int p=0;p<4;++p)directions[p]=normalized(mesh.points[p]-mesh.points[4]);
        SweepConstraintControls controls;SweepConstraintReport report;
        require(constrainFaceSweeps(mesh,faces,{-.01,-.02,-.015,-.01},zero,fixed,directions,1,out,sweeps,report,error,controls),error);
        std::cout<<std::hexfloat;
        for(auto p:out.points)std::cout<<p.x<<' '<<p.y<<' '<<p.z<<'\n';
        for(auto v:out.volumes)std::cout<<v<<'\n';
        for(auto v:sweeps)std::cout<<v<<'\n';
        std::cout<<report.rank<<' '<<report.iterations<<' '<<report.constraints<<'\n';
    } else require(false,"unknown case");
    std::cout<<"PASS "<<test<<'\n';
}
