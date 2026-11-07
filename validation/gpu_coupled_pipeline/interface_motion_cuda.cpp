#include "interface_motion_probe.H"
#include "mesh/Geometry.C" // independent frozen native scalar route
#define main historicalGeometryMain
#include "../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include <cstring>
namespace g=ugkp::gpu::geometry;
static std::vector<g::Point> pack(const std::vector<chmt::Vec3>& points){std::vector<g::Point> out;for(auto p:points)out.push_back({p.x,p.y,p.z});return out;}
static chmt::Vec3 unpack(g::Point p){return {p.x,p.y,p.z};}
static void exact(double a,double b){require((std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0,"interface motion arithmetic differs");}
static g::InterfaceGasPointMotionDefinition definition(const chmt::HostMesh& mesh,double depth){
    g::InterfaceGasPointMotionDefinition d;auto& gas=d.gas;gas.topology.faceOffsets=mesh.faceOffsets;gas.topology.facePoints=mesh.facePoints;
    gas.topology.vertexOffsets={0};for(std::size_t c=0;c<mesh.volumes.size();++c){std::set<int> vertices;
        for(int k=mesh.cellFaceOffsets[c];k<mesh.cellFaceOffsets[c+1];++k){const int f=mesh.cellFaces[k];for(int j=mesh.faceOffsets[f];j<mesh.faceOffsets[f+1];++j)vertices.insert(mesh.facePoints[j]);}
        gas.topology.vertices.insert(gas.topology.vertices.end(),vertices.begin(),vertices.end());gas.topology.vertexOffsets.push_back(gas.topology.vertices.size());}
    gas.owner=mesh.owner;gas.points=pack(mesh.points);gas.fixed.assign(mesh.points.size(),0);d.solidToGas.assign(mesh.points.size(),-1);
    std::vector<chmt::Vec3> normals(mesh.points.size());
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0){gas.wallFaces.push_back(f);d.solidFaces.push_back(f);
        for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k){const int p=mesh.facePoints[k];gas.fixed[p]=1;d.solidToGas[p]=p;normals[p]+=mesh.areaVectors[f];}}
    for(std::size_t p=0;p<gas.points.size();++p){const auto normal=chmt::normalized(normals[p]);gas.points[p]+=g::Point{normal.x,normal.y,normal.z}*depth;}
    gas.reference=gas.points;std::vector<std::set<int>> adjacency(mesh.points.size());
    for(std::size_t f=0;f<mesh.owner.size();++f){const int lo=mesh.faceOffsets[f],hi=mesh.faceOffsets[f+1];for(int k=lo;k<hi;++k){const int a=mesh.facePoints[k],b=mesh.facePoints[k+1==hi?lo:k+1];adjacency[a].insert(b);adjacency[b].insert(a);}}
    gas.offsets={0};for(const auto& row:adjacency){gas.neighbours.insert(gas.neighbours.end(),row.begin(),row.end());gas.offsets.push_back(gas.neighbours.size());}
    std::vector<chmt::Vec3> gasPoints;for(auto p:gas.points)gasPoints.push_back(unpack(p));gas.geometricTolerance=1e-12*chmt::meshScale(gasPoints);
    d.solidTopology=gas.topology;d.oldSolid=pack(mesh.points);d.pointThickness.assign(mesh.points.size(),depth);
    d.previousFilmThickness.assign(d.solidFaces.size(),depth);d.filmThickness=d.previousFilmThickness;return d;
}
static g::InterfaceGasPointMotionResult reference(const g::InterfaceGasPointMotionDefinition& d,const std::vector<g::Point>& end,const std::vector<g::Point>& areas){
    g::InterfaceGasPointMotionResult out;bool unchanged=true;
    for(std::size_t p=0;p<end.size();++p)unchanged=unchanged&&chmt::mag(unpack(end[p])-unpack(d.oldSolid[p]))==0;
    for(std::size_t i=0;i<d.filmThickness.size();++i)unchanged=unchanged&&d.filmThickness[i]==d.previousFilmThickness[i];
    out.status.changed=!unchanged;if(unchanged){out.points=d.gas.points;return out;}
    std::vector<chmt::Vec3> normals(end.size()),boundary(d.gas.points.size());auto fixed=d.gas.fixed;
    for(int f:d.solidFaces)for(int k=d.solidTopology.faceOffsets[f];k<d.solidTopology.faceOffsets[f+1];++k)normals[d.solidTopology.facePoints[k]]+=unpack(areas[f]);
    for(auto& n:normals)n=chmt::normalized(n);
    for(std::size_t s=0;s<d.solidToGas.size();++s)if(d.solidToGas[s]>=0){const int p=d.solidToGas[s];boundary[p]=unpack(end[s])+normals[s]*d.pointThickness[s]-unpack(d.gas.points[p]);fixed[p]=1;}
    chmt::HostMesh mesh;mesh.faceOffsets=d.gas.topology.faceOffsets;mesh.facePoints=d.gas.topology.facePoints;mesh.owner=d.gas.owner;
    for(auto p:d.gas.points)mesh.points.push_back(unpack(p));mesh.boundaryKind.assign(mesh.owner.size(),chmt::BoundaryKind::Slip);
    chmt::empty_detail::Extrusion extrusion;extrusion.axis=d.emptyAxis;extrusion.mate=d.emptyMate;
    if(d.emptyAxis>=0&&!mesh.boundaryKind.empty())mesh.boundaryKind[0]=chmt::BoundaryKind::Empty;
    std::vector<g::MotionPenalty> penalties(mesh.points.size());double minimum=std::numeric_limits<double>::infinity();
    const auto& t=d.gas.topology;
    for(int f:d.gas.wallFaces)if(f>=0){chmt::Vec3 c0{},c1{},a0{},a1{};const int lo=t.faceOffsets[f],hi=t.faceOffsets[f+1];
        for(int k=lo;k<hi;++k){const int v=t.facePoints[k];c0+=unpack(d.gas.reference[v]);c1+=mesh.points[v]+boundary[v];}c0=c0/double(hi-lo);c1=c1/double(hi-lo);
        for(int k=lo;k<hi;++k){const int v=t.facePoints[k],w=t.facePoints[k+1==hi?lo:k+1];a0+=chmt::cross(unpack(d.gas.reference[v])-c0,unpack(d.gas.reference[w])-c0);a1+=chmt::cross(mesh.points[v]+boundary[v]-c1,mesh.points[w]+boundary[w]-c1);}
        const auto n0=chmt::normalized(a0)*-1,n1=chmt::normalized(a1)*-1;const int cell=d.gas.owner[f];
        for(int k=t.vertexOffsets[cell];k<t.vertexOffsets[cell+1];++k){const int v=t.vertices[k];if(fixed[v])continue;const double h=chmt::dot(unpack(d.gas.reference[v])-c0,n0);
            if(h>1e-12*chmt::mag(unpack(d.gas.reference[v])-c0)){minimum=std::min(minimum,h);const double target=h-chmt::dot(mesh.points[v]-c1,n1);g::addNormalHeight(penalties[v],{n1.x,n1.y,n1.z},target,h);}}
    }
    std::string error;std::vector<chmt::Vec3> result;
    const bool ok=chmt::harmonicPointMotionRepresentedImpl(mesh,boundary,fixed,result,error,d.emptyAxis>=0?&extrusion:nullptr,nullptr,nullptr,penalties,minimum);
    using Failure=g::InterfaceGasMotionFailure;
    if(!ok){
        if(error.find("Empty layer vertices")!=std::string::npos)out.status.failure=Failure::EmptyFreePoint;
        else if(error.find("nonzero normal input")!=std::string::npos)out.status.failure=Failure::EmptyNormal;
        else if(error.find("paired prescribed")!=std::string::npos)out.status.failure=Failure::EmptyPair;
        else if(error.find("nonfinite prescribed")!=std::string::npos)out.status.failure=Failure::NonfiniteFixed;
        else if(error.find("isolated mesh point")!=std::string::npos)out.status.failure=Failure::IsolatedPoint;
        else if(error.find("Dirichlet anchor")!=std::string::npos)out.status.failure=Failure::NoAnchor;
        else out.status.failure=Failure::Harmonic;return out;
    }
    out.points=pack(result);out.status.constraints.minimumHeight=minimum;out.status.constraints.harmonicTolerance=d.gas.geometricTolerance;
    if(minimum>0)out.status.constraints.harmonicTolerance=std::min(d.gas.geometricTolerance,1e-10*minimum);return out;
}
static void compare(const g::InterfaceGasPointMotionResult& a,const g::InterfaceGasPointMotionResult& b){
    require(a.status.changed==b.status.changed&&a.status.failure==b.status.failure,"interface motion branch/failure mismatch");require(a.points.size()==b.points.size(),"interface output shape mismatch");
    for(std::size_t p=0;p<a.points.size();++p){exact(a.points[p].x,b.points[p].x);exact(a.points[p].y,b.points[p].y);exact(a.points[p].z,b.points[p].z);}
    if(a.status.changed&&a.status.failure==g::InterfaceGasMotionFailure::None){exact(a.status.constraints.minimumHeight,b.status.constraints.minimumHeight);exact(a.status.constraints.harmonicTolerance,b.status.constraints.harmonicTolerance);}
}
#ifndef UGKP_INTERFACE_MOTION_FIXTURES_ONLY
int main(){int cases=0;
    for(bool tetra:{false,true})for(double scale:{1e-3,1.,1e3})for(double depth:{0.,.001}){
        auto mesh=tetra?tetrahedralBulk():twoHexahedra(0);for(auto& p:mesh.points)p={scale*(p.x+7.1),scale*(p.y-4.3),scale*(p.z+2.2)};
        std::string error;require(chmt::rebuildGeometry(mesh,error),error);const auto areas=pack(mesh.areaVectors);
        auto d=definition(mesh,depth*scale);g::InterfaceGasPointMotion owner(d);
        for(int mode=0;mode<3;++mode){auto end=d.oldSolid;if(mode)for(std::size_t p=0;p<end.size();++p)end[p]+=g::Point{scale*.001*(1+p%2),-scale*.002*(1+p%3),scale*.003};
            if(mode==2)end[0].x=std::numeric_limits<double>::quiet_NaN();const auto expected=reference(d,end,areas);
            compare(expected,owner.solve(end,areas));compare(expected,probeInterfaceMotion(owner,d,end,areas));
            compare(expected,owner.solve(end,areas));++cases;}
        compare(reference(d,d.oldSolid,areas),owner.solve(d.oldSolid,areas));
        auto noAnchor=d;std::fill(noAnchor.solidToGas.begin(),noAnchor.solidToGas.end(),-1);std::fill(noAnchor.gas.fixed.begin(),noAnchor.gas.fixed.end(),0);
        g::InterfaceGasPointMotion unconstrained(noAnchor);auto end=d.oldSolid;end[0].x+=.001*scale;
        compare(reference(noAnchor,end,areas),unconstrained.solve(end,areas));compare(reference(noAnchor,d.oldSolid,areas),unconstrained.solve(d.oldSolid,areas));++cases;
        if(!tetra&&depth==0){auto empty=d;empty.emptyAxis=2;empty.emptyMate.resize(empty.gas.points.size());for(int p=0;p<12;++p)empty.emptyMate[p]=p<6?p+6:p-6;
            g::InterfaceGasPointMotion flat(empty);for(int mode=0;mode<3;++mode){auto end=empty.oldSolid;
                if(mode==0)for(auto& p:end)p.x+=.001*scale;if(mode==1)end[0].z+=.001*scale;if(mode==2)end[0].x+=.001*scale;
                compare(reference(empty,end,areas),flat.solve(end,areas));compare(reference(empty,end,areas),probeInterfaceMotion(flat,empty,end,areas));++cases;}}
    }
    std::cout<<"PASS "<<cases<<" frozen interface/relative-height/harmonic compositions, unchanged branch, Empty rejection, anchor failure, borrowed graph reuse and host-descriptor restoration\n";
}

#endif
