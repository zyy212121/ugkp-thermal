#include "sweep_planarity_probe.H"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstring>
namespace g=ugkp::gpu::geometry;
static SweepPlanarityProbe reference(const g::Topology& t,const std::vector<g::Point>& points,double coordinateUncertainty){
    SweepPlanarityProbe out;
    for(int f=0;f<int(t.faceOffsets.size())-1;++f){const int begin=t.faceOffsets[f],stop=t.faceOffsets[f+1];
        const int a=t.facePoints[begin],b=t.facePoints[begin+1],c=t.facePoints[begin+2];double faceLength=0;
        for(int k=begin+1;k<stop;++k){const double length=g::magnitude(points[t.facePoints[k]]-points[a]);faceLength=faceLength>length?faceLength:length;}
        const double scale=g::magnitude(g::cross(points[b]-points[a],points[c]-points[a]))*faceLength;
        for(int k=begin+3;k<stop;++k){const int d=t.facePoints[k];const double volume=g::dot(points[b]-points[a],g::cross(points[c]-points[a],points[d]-points[a]));
            const double uncertainty=64*std::numeric_limits<double>::epsilon()*scale+8*coordinateUncertainty*faceLength*faceLength;
            if(scale>0&&std::abs(volume)<=uncertainty){out.planar.push_back({a,b,c,d,scale});out.tolerance.push_back(32*std::numeric_limits<double>::epsilon()+8*coordinateUncertainty*faceLength*faceLength/scale);}}
    }return out;
}
int main(){int cases=0;
    for(int faces:{0,1,17,129,513})for(double scale:{1e-5,1.,1e5}){
        g::Topology t;t.faceOffsets={0};std::vector<g::Point> points;
        for(int f=0;f<faces;++f){const int vertices=3+f%6;for(int k=0;k<vertices;++k){const double angle=2*std::acos(-1.)*k/vertices;
            t.facePoints.push_back(points.size());points.push_back({scale*(3*f+std::cos(angle)),scale*std::sin(angle),scale*(.1*std::cos(angle)+.2*std::sin(angle))});}t.faceOffsets.push_back(points.size());}
        g::ResidentSweepPlanarity owner(t);
        for(int mode=0;mode<6;++mode){auto current=points;double uncertainty=0;
            if(mode==1)for(std::size_t p=0;p<current.size();++p)if(p%5==0)current[p].z+=.01*scale;
            if(mode==2)for(std::size_t p=0;p<current.size();++p)if(p%7==0)current[p].z+=1e-14*scale;
            if(mode==3){uncertainty=std::numeric_limits<double>::epsilon()*1e9;for(auto& p:current){p.x=(p.x+1e9)-1e9;p.y=(p.y-1e9)+1e9;p.z=(p.z+1e9)-1e9;}}
            if(mode==4)for(auto& p:current)p={0,0,0};
            if(mode==5)uncertainty=std::numeric_limits<double>::quiet_NaN();
            const auto expected=reference(t,current,uncertainty),actual=probeSweepPlanarity(owner,current,uncertainty);
            assert(expected.planar.size()==actual.planar.size());
            for(std::size_t i=0;i<expected.planar.size();++i){const auto a=expected.planar[i],b=actual.planar[i];
                assert(a.a==b.a&&a.b==b.b&&a.c==b.c&&a.d==b.d);assert(a.scale==b.scale);assert(expected.tolerance[i]==actual.tolerance[i]);}++cases;
        }
    }
    std::cout<<"PASS "<<cases<<" native ordered plane classifications, changing geometry, quantization uncertainty, degenerate faces, stable compaction and resident replay\n";
}
