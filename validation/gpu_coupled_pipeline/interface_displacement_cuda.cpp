#include "interface_displacement_probe.H"
#include "mesh/Geometry.C" // frozen scalar geometry/normalization
#include <random>
#include <cstring>
#include <iostream>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
static chmt::Vec3 native(g::Point p){return {p.x,p.y,p.z};}
static void exact(double a,double b,int test){if((std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0)return;std::cerr<<test<<": displacement mismatch\n";std::exit(1);}
int main(){std::mt19937_64 random(971432);std::uniform_real_distribution<double> value(-1,1);int cases=0;
    for(int n:{0,3,17,130,2001})for(double scale:{1e-80,1.,1e80})for(bool strided:{false,true}){
        g::Topology topology;topology.faceOffsets={0};std::vector<int> faces,mapping(n,-1);std::vector<g::Point> solid(n),area;std::vector<double> depth(n);
        const int ng=n?n+7:0;std::vector<g::Point> gas(ng);
        for(int p=0;p<n;++p){solid[p]={scale*value(random),scale*value(random),scale*value(random)};depth[p]=scale*.01*(1+value(random));if(p%3)mapping[p]=n-p+2;}
        if(n)solid[0]={};
        for(int f=0;f<(n?n/2+3:0);++f){for(int j=0;j<3;++j)topology.facePoints.push_back((f+j)%n);topology.faceOffsets.push_back(topology.facePoints.size());
            area.push_back({scale*scale*value(random),scale*scale*value(random),scale*scale*value(random)});if(f%2)faces.push_back(f);}
        std::reverse(faces.begin(),faces.end());
        std::vector<double> previous(faces.size(),.01*scale);
        for(auto& p:gas)p={scale*value(random),scale*value(random),scale*value(random)};
        for(int mode=0;mode<6;++mode){auto end=solid;auto film=previous;
            if(n&&mode==1)end[0].x=std::numeric_limits<double>::denorm_min();
            if(!film.empty()&&mode==2)film.front()+=.001*scale;
            if(mode==3)for(auto& p:end)p+=g::Point{.002*scale,-.003*scale,.001*scale};
            if(n&&mode==4)end.back().x=std::numeric_limits<double>::quiet_NaN();
            if(n&&mode==5)end[0].z+=.001*scale; // unmapped solid motion still changes the guard
            bool unchanged=true;for(int p=0;p<n;++p)unchanged=unchanged&&chmt::mag(native(end[p])-native(solid[p]))==0;
            for(std::size_t f=0;f<film.size();++f)unchanged=unchanged&&film[f]==previous[f];
            std::vector<chmt::Vec3> normals(n),expected(ng);
            for(int f:faces)for(int k=topology.faceOffsets[f];k<topology.faceOffsets[f+1];++k)normals[topology.facePoints[k]]+=native(area[f]);
            for(auto& normal:normals)normal=chmt::normalized(normal);
            for(int s=0;s<n;++s)if(mapping[s]>=0)expected[mapping[s]]=native(end[s])+normals[s]*depth[s]-native(gas[mapping[s]]);
            const auto got=probeInterfaceDisplacement(topology,ng,faces,mapping,solid,end,area,gas,depth,previous,film,strided);
            if(got.changed!=!unchanged){std::cerr<<cases<<": unchanged-offset guard mismatch\n";return 1;}
            for(int p=0;p<ng;++p){exact(expected[p].x,got.displacement[p].x,cases);exact(expected[p].y,got.displacement[p].y,cases);exact(expected[p].z,got.displacement[p].z,cases);}++cases;
        }
    }
    std::cout<<"PASS "<<cases<<" frozen interface displacement/unchanged-offset configurations, ordered shared normals, unmapped vertices, underflow motion and strided owner records\n";
}
