#include "sweep_trial_owner_probe.H"
#include <iostream>
#include <cassert>
#include <climits>
#include <cstring>
namespace g=ugkp::gpu::geometry;
int main(){int cases=0;
    for(int count:{0,1,127,128,129,4097}){
        g::ResidentSweepTrial owner(count);std::vector<g::Point> points(count),correction(count);
        for(int mode=0;mode<6;++mode){
            for(int i=0;i<count;++i){points[i]={1+i*.01,-2-i*.02,3+i*.03};correction[i]={-.2,.5,-.7};}
            if(mode==1)for(auto& p:correction)p={};
            if(mode==2)for(auto& p:points)p={std::numeric_limits<double>::max(),0,-std::numeric_limits<double>::max()};
            if(mode==3&&count)correction[count/2].x=std::numeric_limits<double>::infinity();
            if(mode==4&&count)points[count-1].z=std::numeric_limits<double>::quiet_NaN();
            if(mode==5)for(auto& p:correction)p={1e-300,-1e-300,0};
            for(int backtrack:{0,1,50,1074,1075,INT_MAX,-1}){
                const double alpha=backtrack>=0?std::ldexp(1.,-backtrack):0;
                auto expected=points;bool representable=backtrack>=0,changed=false;int first=-1;
                for(int i=0;i<count;++i){const auto x=g::sweepTrialCoordinate(points[i].x,correction[i].x,alpha),y=g::sweepTrialCoordinate(points[i].y,correction[i].y,alpha),z=g::sweepTrialCoordinate(points[i].z,correction[i].z,alpha);
                    if(backtrack>=0&&x.representable&&y.representable&&z.representable)expected[i]={x.value,y.value,z.value};
                    else {representable=false;if(first<0)first=i;}
                    changed=changed||x.changed||y.changed||z.changed;
                }
                const auto actual=probeSweepTrialOwner(owner,points,correction,backtrack);
                assert(actual.status.alpha==alpha&&actual.status.representable==representable&&actual.status.firstInvalidPoint==first);
                assert(actual.status.changed==(representable&&changed));
                for(int i=0;i<count;++i)assert(std::memcmp(&expected[i],&actual.points[i],sizeof(g::Point))==0);++cases;
            }
        }
    }
    std::cout<<"PASS "<<cases<<" resident coordinate trials, native extended-range decisions, first failure, zero update, cross-block reduction and graph replay\n";
}
