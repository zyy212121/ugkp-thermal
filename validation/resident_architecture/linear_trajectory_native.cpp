// Compile the CPU oracle from a frozen pre-migration Geometry.C include root.
#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "linear_trajectory_probe.H"
int checks=0,valid=0,invalid=0,marginWarnings=0;
void compareTrajectory(HostMesh mesh,const std::vector<Vec3>& endpoint,bool expectWarning=false){
    std::string error;assert(chmt::rebuildGeometry(mesh,error));
    ugkp::diagnostics::DeviceContext diagnostic{};
    const bool expected=chmt::certifyLinearTrajectory(mesh,endpoint,error,&diagnostic);
    const auto& warning=diagnostic.warnings[1];
    // Independently compare the shared host specialization to the frozen
    // original function, including exact warning sample and count.
    const auto csr=topology(mesh);const auto before=points(mesh.points),after=points(endpoint);
    const device::TopologyView view{csr.faceOffsets.data(),csr.facePoints.data(),csr.cellOffsets.data(),csr.cellFaces.data(),
        csr.cellSigns.data(),csr.vertexOffsets.data(),csr.vertices.data()};
    bool moving=false;for(std::size_t p=0;p<endpoint.size();++p)moving=moving||chmt::mag(endpoint[p]-mesh.points[p])>0;
    int hostWarnings=0,hostFailure=-1,hostWarningCell=-1;double hostObserved=0,hostLimit=0;
    if(moving)for(int cell=0;cell<int(mesh.volumes.size());++cell){
        const auto capture=[&](double value,double limit,device::TrajectoryWarning){
            if(!hostWarnings){hostObserved=value;hostLimit=limit;hostWarningCell=cell;}++hostWarnings;};
        const auto failure=device::certifyLinearCell<long double>(cell,view,before.data(),after.data(),capture);
        if(failure!=device::TrajectoryFailure::None){hostFailure=cell;break;}
    }
    assert(expected==(hostFailure<0));assert(hostWarnings==int(warning.count));
    if(hostWarnings){assert(hostWarningCell==warning.sample.entityIndex);exact(hostObserved,warning.sample.observed);exact(hostLimit,warning.sample.reference);}
    const auto actual=trajectoryProbe::evaluate(topology(mesh),points(mesh.points),points(endpoint));
    int first=-1,warnings=0;for(std::size_t c=0;c<actual.size();++c){warnings+=actual[c].warnings;if(first<0&&actual[c].failure!=device::TrajectoryFailure::None)first=c;}
    if(expected!=(first<0)){std::cerr<<"trajectory acceptance differs at test "<<checks<<" native="<<error<<" GPU cell="<<first<<'\n';std::abort();}
    if(first>=0){const char* messages[]={"","zero-radius cell fan","nonfinite cell trajectory determinant","moving cell tetrahedral fan collapses or inverts","cell fan is not a single star-shaped volume"};
        std::string actualError;chmt::errorAt(actualError,messages[int(actual[first].failure)],first);assert(actualError==error);++invalid;
    }else ++valid;
    assert(warnings==int(warning.count));
    if(warnings){int c=0;while(!actual[c].warnings)++c;assert(c==warning.sample.entityIndex);
        exact(actual[c].tolerance,warning.sample.reference);
        assert(std::abs(actual[c].observed-warning.sample.observed)<=1e-12*std::max(std::abs(warning.sample.observed),std::numeric_limits<double>::min()));}
    if(expectWarning){assert(warnings>0);++marginWarnings;}
    ++checks;
}
HostMesh multipleCover(){
    HostMesh mesh;mesh.faceOffsets={0};const double pi=std::acos(-1.);
    for(int p=0;p<5;++p)mesh.points.push_back({std::cos(2*pi*p/5),std::sin(2*pi*p/5),0});
    mesh.points.push_back({0,0,1});mesh.points.push_back({0,0,-1});
    for(int p=0;p<5;++p)for(int side=0;side<2;++side){const int a=(2*p)%5,b=(2*p+2)%5;
        mesh.facePoints.insert(mesh.facePoints.end(),side?std::initializer_list<int>{b,a,6}:std::initializer_list<int>{a,b,5});
        mesh.faceOffsets.push_back(mesh.facePoints.size());mesh.owner.push_back(0);mesh.neighbour.push_back(-1);mesh.boundaryKind.push_back(chmt::BoundaryKind::Slip);}
    return mesh;
}
int main(){
    for(int cells:{1,7,129})for(double scale:{.001,1.,1024.})for(bool skew:{false,true}){
        const auto mesh=boxes(cells,scale,skew);compareTrajectory(mesh,mesh.points);
        for(int mode=0;mode<7;++mode){auto endpoint=mesh.points;
            for(auto& p:endpoint){
                if(mode==0)p.x+=.03125*scale;
                if(mode==1)p.z*=2e-15;
                if(mode==2)p.z*=1e-20;
                if(mode==3)p.z=-p.z;
                if(mode==4){p.x=-p.x;p.y=-p.y;}
                if(mode==5){p.x=-4*p.x;p.y=-p.y;}
                if(mode==6){const double x=p.x,y=p.y,a=2*std::acos(-1.)/3;p.x=std::cos(a)*x-std::sin(a)*y;p.y=std::sin(a)*x+std::cos(a)*y;}
            }
            compareTrajectory(mesh,endpoint,mode==1||mode==2);
        }
    }
    for(int failed:{0,1,5}){
        auto mixed=boxes(7,1.,false);auto end=mixed.points;
        for(std::size_t p=0;p<end.size();++p)end[p].z*=int(p/8)==failed?-1.:2e-15;
        compareTrajectory(mixed,end);
    }
    auto star=multipleCover();compareTrajectory(star,star.points);
    auto endpoint=star.points;for(auto& p:endpoint)p.x+=.01;compareTrajectory(star,endpoint);
    // A stationary non-star cell must still be checked if another cell moves.
    auto mesh=multipleCover();const auto cube=boxes(1,1.,false);const int offset=mesh.points.size();
    mesh.points.insert(mesh.points.end(),cube.points.begin(),cube.points.end());
    for(std::size_t f=0;f<cube.owner.size();++f){for(int k=cube.faceOffsets[f];k<cube.faceOffsets[f+1];++k)mesh.facePoints.push_back(offset+cube.facePoints[k]);
        mesh.faceOffsets.push_back(mesh.facePoints.size());mesh.owner.push_back(1);mesh.neighbour.push_back(-1);mesh.boundaryKind.push_back(chmt::BoundaryKind::Slip);}
    endpoint=mesh.points;for(std::size_t i=offset;i<endpoint.size();++i)endpoint[i].x+=.01;compareTrajectory(mesh,endpoint);
    HostMesh empty;empty.faceOffsets={0};compareTrajectory(empty,{});
    std::cout<<"PASS "<<checks<<" native/GPU trajectory certificates: valid="<<valid<<" invalid="<<invalid<<" thin warnings="<<marginWarnings<<"; rotation, interior inversion, multiple covering and globally moving meshes\n";
    return 0;
}
