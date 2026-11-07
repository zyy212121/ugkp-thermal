#include "gpu/NormalHeightMotion.H"
#include "mesh/Geometry.C" // separately selected frozen native scalar operators
#include <algorithm>
#include <map>
#include <cstring>
#include <iostream>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
std::vector<g::MotionPenalty> residentNormalHeight(const g::Topology&,int,const std::vector<int>&,
    const std::vector<int>&,const std::vector<g::Point>&,const std::vector<g::Point>&,
    const std::vector<g::Point>&,const std::vector<unsigned char>&,double,g::NormalHeightMotionControl&);
g::NormalHeightPointMotionResult residentNormalHeightPointMotion(const g::NormalHeightPointMotionDefinition&,const std::vector<g::Point>&);
static void exact(double a,double b,int test){if(std::memcmp(&a,&b,sizeof(a))){
    std::cerr.precision(17);std::cerr<<test<<": mismatch "<<a<<" vs "<<b<<'\n';std::exit(1);}}
static void penalty(const g::MotionPenalty& a,const g::MotionPenalty& b,int test){
    const double expected[]={a.xx,a.xy,a.xz,a.yy,a.yz,a.zz,a.rhs.x,a.rhs.y,a.rhs.z};
    const double actual[]={b.xx,b.xy,b.xz,b.yy,b.yz,b.zz,b.rhs.x,b.rhs.y,b.rhs.z};
    for(int k=0;k<9;++k)exact(expected[k],actual[k],test);
}
static chmt::Vec3 point(g::Point p){return {p.x,p.y,p.z};}
int main(){int cases=0,composed=0;
    for(int n:{0,1,3,7})for(double height:{1e-13,1e-5,.1})for(double scale:{1e-3,1.,1e3})for(int mask=0;mask<3;++mask){
        g::Topology t;t.faceOffsets={0};t.vertexOffsets={0};std::vector<int> owner,walls;
        std::vector<g::Point> reference,points,displacement;std::vector<unsigned char> fixed;
        const auto index=[&](int x,int y,int z){return x+(n+1)*y+(n+1)*(n+1)*z;};
        if(n)for(int z=0;z<=n;++z)for(int y=0;y<=n;++y)for(int x=0;x<=n;++x){
            const double a=scale*x/n,b=height*scale*y/n,c=scale*z/n;
            const g::Point p{.8*a-.6*b+7.1*scale,.6*a+.8*b-4.3*scale,c+2.2*scale};
            reference.push_back(p);points.push_back(mask==2?p:p+g::Point{.001*b,.002*a,.003*c});
            displacement.push_back({.001*scale*(1.+x),-.002*scale*(1.+y),.0003*scale*z});
            fixed.push_back(mask==2||(mask==1&&(x==0||y==0)));}
        std::map<std::vector<int>,int> seen;
        for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
            const int a=index(x,y,z),b=index(x+1,y,z),d=index(x,y+1,z),e=index(x+1,y+1,z);
            const int A=index(x,y,z+1),B=index(x+1,y,z+1),D=index(x,y+1,z+1),E=index(x+1,y+1,z+1);
            std::vector<int> vertices={a,b,d,e,A,B,D,E};std::sort(vertices.begin(),vertices.end());
            const int cell=t.vertexOffsets.size()-1;t.vertices.insert(t.vertices.end(),vertices.begin(),vertices.end());t.vertexOffsets.push_back(t.vertices.size());
            const std::vector<std::vector<int>> polygons={{a,d,e,b},{A,B,E,D},{a,b,B,A},{d,D,E,e},{a,A,D,d},{b,e,E,B}};
            for(int side=0;side<6;++side){auto key=polygons[side];std::sort(key.begin(),key.end());auto found=seen.find(key);int f;
                if(found==seen.end()){f=owner.size();seen[key]=f;owner.push_back(cell);t.facePoints.insert(t.facePoints.end(),polygons[side].begin(),polygons[side].end());t.faceOffsets.push_back(t.facePoints.size());}
                else f=found->second;
                if((side==2&&y==0)||(side==4&&x==0))walls.push_back(f);
            }
        }
        if(!walls.empty())walls.push_back(walls.front());walls.insert(walls.begin(),-1);
        const int np=points.size();std::vector<g::MotionPenalty> expected(np),shared(np);
        double minimum=std::numeric_limits<double>::infinity(),sharedMinimum=minimum;
        const g::TopologyView topology{t.faceOffsets.data(),t.facePoints.data(),nullptr,nullptr,nullptr,t.vertexOffsets.data(),t.vertices.data()};
        for(int f:walls)if(f>=0){
            // Literal original buildGas relative-height calculation; independent
            // of the new helper and transposed device incidence traversal.
            chmt::Vec3 c0{},c1{},a0{},a1{};const int lo=t.faceOffsets[f],hi=t.faceOffsets[f+1];
            for(int k=lo;k<hi;++k){const int v=t.facePoints[k];c0+=point(reference[v]);c1+=point(points[v])+point(displacement[v]);}
            c0=c0/double(hi-lo);c1=c1/double(hi-lo);
            for(int k=lo;k<hi;++k){const int v=t.facePoints[k],w=t.facePoints[k+1==hi?lo:k+1];
                a0+=chmt::cross(point(reference[v])-c0,point(reference[w])-c0);
                a1+=chmt::cross(point(points[v])+point(displacement[v])-c1,point(points[w])+point(displacement[w])-c1);}
            const auto n0=chmt::normalized(a0)*-1,n1=chmt::normalized(a1)*-1;
            const auto wall=g::wallNormalHeightGeometry(topology,f,reference.data(),points.data(),displacement.data());
            const int cell=owner[f];for(int k=t.vertexOffsets[cell];k<t.vertexOffsets[cell+1];++k){const int v=t.vertices[k];if(fixed[v])continue;
                const double h=chmt::dot(point(reference[v])-c0,n0);
                if(h>1e-12*chmt::mag(point(reference[v])-c0)){minimum=std::min(minimum,h);
                    const double target=h-chmt::dot(point(points[v])-c1,n1);g::addNormalHeight(expected[v],{n1.x,n1.y,n1.z},target,h);}
                sharedMinimum=std::min(sharedMinimum,g::addWallNormalHeight(shared[v],wall,reference[v],points[v]));
            }
        }
        const double geometricTolerance=1e-12*scale;double tolerance=geometricTolerance;
        if(minimum>0)tolerance=std::min(tolerance,1e-10*minimum);
        g::NormalHeightMotionControl control;const auto got=residentNormalHeight(t,np,owner,walls,reference,points,displacement,fixed,geometricTolerance,control);
        exact(minimum,sharedMinimum,cases);exact(minimum,control.minimumHeight,cases);exact(tolerance,control.harmonicTolerance,cases);
        for(int p=0;p<np;++p){penalty(expected[p],shared[p],cases);penalty(expected[p],got[p],cases);}
        if(n&&scale==1.&&height>1e-13&&mask){
            chmt::HostMesh mesh;for(auto p:points)mesh.points.push_back(point(p));
            mesh.faceOffsets=t.faceOffsets;mesh.facePoints=t.facePoints;mesh.owner=owner;
            std::vector<std::set<int>> adjacency(np);
            for(std::size_t f=0;f<owner.size();++f){const int lo=t.faceOffsets[f],hi=t.faceOffsets[f+1];
                for(int k=lo;k<hi;++k){const int a=t.facePoints[k],b=t.facePoints[k+1==hi?lo:k+1];adjacency[a].insert(b);adjacency[b].insert(a);}}
            g::NormalHeightPointMotionDefinition definition;definition.topology=t;definition.owner=owner;definition.wallFaces=walls;
            definition.points=points;definition.reference=reference;definition.fixed=fixed;
            definition.geometricTolerance=1e-12*chmt::meshScale(mesh.points);definition.offsets={0};
            for(const auto& row:adjacency){definition.neighbours.insert(definition.neighbours.end(),row.begin(),row.end());definition.offsets.push_back(definition.neighbours.size());}
            std::vector<chmt::Vec3> boundary,original;for(auto p:displacement)boundary.push_back(point(p));std::string error;
            if(!chmt::harmonicPointMotionRepresentedImpl(mesh,boundary,fixed,original,error,nullptr,nullptr,nullptr,expected,minimum)){
                std::cerr<<"frozen harmonic failed "<<error<<'\n';return 1;}
            g::NormalHeightPointMotion solver(definition);
            const auto host=solver.solve(displacement),device=residentNormalHeightPointMotion(definition,displacement);
            if(host.motion.errorPoint>=0||device.motion.errorPoint>=0||host.motion.points.size()!=original.size()||device.motion.points.size()!=original.size())return 1;
            exact(host.motion.residual,device.motion.residual,cases);
            if(host.motion.iterations!=device.motion.iterations)return 1;
            exact(minimum,host.constraints.minimumHeight,cases);exact(minimum,device.constraints.minimumHeight,cases);
            const double expectedTolerance=minimum>0?std::min(definition.geometricTolerance,1e-10*minimum):definition.geometricTolerance;
            exact(expectedTolerance,host.constraints.harmonicTolerance,cases);exact(expectedTolerance,device.constraints.harmonicTolerance,cases);
            for(int p=0;p<np;++p)for(const auto& result:{host.motion.points[p],device.motion.points[p]}){
                exact(original[p].x,result.x,cases);exact(original[p].y,result.y,cases);exact(original[p].z,result.z,cases);}
            ++composed;
        }
        ++cases;
    }
    std::cout<<"PASS "<<composed<<" frozen near-wall plus harmonic compositions, host publication and device-selected graph replay\n";
    std::cout<<"PASS "<<cases<<" frozen relative-height penalty/tolerance configurations, empty meshes, translated/rotated/thin grids, overlapping walls and fixed masks\n";
}
