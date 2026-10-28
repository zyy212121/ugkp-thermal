#include "mesh/Geometry.C"
#include "gpu/PolyhedralGeometry.H"
#include <random>
#include <cstring>
#include <cassert>
#include <iostream>
namespace gpu=ugkp::gpu::geometry;
void exact(double a,double b){assert(std::memcmp(&a,&b,sizeof(a))==0);}
int main(){std::mt19937_64 rng(39152);std::uniform_real_distribution<double> random(-1,1);gpu::PolyhedralGeometry device;int count=0;
    for(int np:{0,3,17,129,2049})for(double scale:{1e-70,1e-4,1.,1e4,1e70})for(int variation=0;variation<4;++variation){
        gpu::Topology topology;topology.faceOffsets={0};std::vector<gpu::Point> areas;std::vector<gpu::BoundaryMotionFace> fronts;std::vector<unsigned char> side;
        for(int f=0;f<np;++f){for(int j=0;j<3;++j)topology.facePoints.push_back((f+j)%np);topology.faceOffsets.push_back(topology.facePoints.size());
            areas.push_back({random(rng)*scale,random(rng)*scale,random(rng)*scale});side.push_back(f%3!=0);
            if(f%3==0)fronts.push_back({f,random(rng)*1e-4,random(rng)*1e-4,.01*(variation+1),{random(rng)*.001,random(rng)*.001,random(rng)*.001}});}
        const double dt=.01;
        std::vector<chmt::Vec3> displacement(np),support(np),direction(np);std::vector<double> weight(np),thickness(np);std::vector<std::vector<chmt::Vec3>> normals(np);
        auto area=[&](int f){auto a=areas[f];return chmt::Vec3{a.x,a.y,a.z};};
        for(auto front:fronts){auto av=area(front.face);const double a=chmt::mag(av);const auto n=chmt::normalized(av);
            const chmt::Vec3 velocity{front.supportVelocity.x,front.supportVelocity.y,front.supportVelocity.z};const auto delta=n*(front.offset-front.previousOffset)+velocity*dt;
            for(int k=topology.faceOffsets[front.face];k<topology.faceOffsets[front.face+1];++k){int p=topology.facePoints[k];displacement[p]+=delta*a;support[p]+=velocity*(dt*a);direction[p]+=av;weight[p]+=a;thickness[p]+=front.thickness*a;}}
        for(int p=0;p<np;++p)if(weight[p]>0){displacement[p]=displacement[p]/weight[p];support[p]=support[p]/weight[p];thickness[p]/=weight[p];}
        for(int f=0;f<np;++f)if(side[f]){const auto normal=chmt::normalized(area(f));for(int k=topology.faceOffsets[f];k<topology.faceOffsets[f+1];++k){int p=topology.facePoints[k];if(weight[p]<=0)continue;
            auto q=normal;for(auto n:normals[p])q-=n*chmt::dot(q,n);if(chmt::mag(q)>1e-10)normals[p].push_back(chmt::normalized(q));}}
        for(int p=0;p<np;++p){auto recession=displacement[p]-support[p];for(auto n:normals[p]){recession-=n*chmt::dot(recession,n);direction[p]-=n*chmt::dot(direction[p],n);}
            if(normals[p].size()==3){recession={};direction[p]={};}displacement[p]=recession+support[p];direction[p]=chmt::normalized(direction[p]);}
        const auto out=device.boundaryMotion(topology,np,areas,fronts,side,dt);assert(out.size()==std::size_t(np));
        for(int p=0;p<np;++p){exact(displacement[p].x,out[p].displacement.x);exact(displacement[p].y,out[p].displacement.y);exact(displacement[p].z,out[p].displacement.z);
            exact(direction[p].x,out[p].direction.x);exact(direction[p].y,out[p].direction.y);exact(direction[p].z,out[p].direction.z);exact(weight[p],out[p].weight);exact(thickness[p],out[p].thickness);}
        ++count;
    }
    std::cout<<"PASS "<<count<<" ordered boundary recession, support, weighted thickness and side-plane projection configurations bitwise\n";
}
