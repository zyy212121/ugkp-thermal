#include "GpuCollisionCentralMoment.H"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
int main()
{
    float countA[8]={},countB[8]={};
    countA[7]=collisionCountWord(16777216);countB[7]=collisionCountWord(1);
    mergeCollisionMomentGroup(countA,countB);
    assert(collisionCountFromWord(countA[7])==16777217);
    double maxEnergyRelative=0;
    static_assert(sizeof(CollisionCentralMoment)==5*sizeof(float),"FP32 storage");
    for(float shift: {0.0f,1000.0f,100000.0f})
    {
        auto a=particleCollisionCentralMoment(3.0e-9f,shift,0,0,0.01f);
        assert(std::abs(collisionCentralTheta(a)-0.01f)<2e-9f);
        CollisionCentralMoment all{};
        double mass=0,px=0,py=0,pz=0,energy=0;
        std::vector<CollisionCentralMoment> terms;
        for(int i=0;i<4097;++i)
        {
            const float m=(1+i%7)*1.0e-9f;
            const float x=shift+float((i%13)-6)*0.0625f;
            const float y=float(i%7)*0.125f,z=-float(i%5)*0.25f;
            auto q=particleCollisionCentralMoment(m,x,y,z,0.01f);
            terms.push_back(q);mass+=m;px+=double(m)*x;py+=double(m)*y;pz+=double(m)*z;
        }
        const double ux=px/mass,uy=py/mass,uz=pz/mass;
        for(auto q:terms)energy+=0.5*double(q.mass)*((q.ux-ux)*(q.ux-ux)+(q.uy-uy)*(q.uy-uy)+(q.uz-uz)*(q.uz-uz))+q.q;
        // Partial groups, then a tree. This is an independent FP64 oracle,
        // not FP64 arithmetic or compensation in the production accumulator.
        std::vector<CollisionCentralMoment> partials;
        for(size_t begin=0;begin<terms.size();begin+=97)
        {
            CollisionCentralMoment q{};
            for(size_t i=begin;i<std::min(begin+97,terms.size());++i)q=mergeCollisionCentralMoments(q,terms[i]);
            partials.push_back(q);
        }
        while(partials.size()>1)
        {
            std::vector<CollisionCentralMoment> next;
            for(size_t i=0;i<partials.size();i+=2)
                next.push_back(i+1<partials.size()?mergeCollisionCentralMoments(partials[i],partials[i+1]):partials[i]);
            partials=next;
        }
        all=partials[0];
        assert(std::abs(all.mass-mass)<mass*2e-5);
        // Large boosts expose ordinary FP32 mean/writeback rounding; they do
        // not justify a promise of arbitrarily long bitwise conservation.
        maxEnergyRelative=std::max(maxEnergyRelative,std::abs(all.q-energy)/energy);
        assert(std::abs(all.q-energy)<energy*(shift>10000?0.03:0.001));
        assert(mergeCollisionCentralMoments({},a).q==a.q);
        assert(mergeCollisionCentralMoments(a,{}).ux==a.ux);
    }
    std::printf("PASS FP32 central host algebra; max Q relative=%.9g\n",maxEnergyRelative);
}
