#include "gasNumerics/GasWindowStep.H"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 const double initial=1e-8,tail=1e-23,stable=2.5722309259329196e-7,maximum=1e-6;
 auto step=ugkwp::selectGasWindowStep(initial,tail,stable,1.1,maximum,true);
 assert(step.accepted==tail && step.next==initial*1.1);
 // A split macro interval cannot grow or shrink the seed by itself.
 auto next=ugkwp::selectGasWindowStep(step.next,1e-6,stable,1.1,maximum,false);
 assert(next.accepted==step.next && next.next==step.next);
 double dt=initial,time=0;
 for(int i=0;i<1000;++i){
   auto s=ugkwp::selectGasWindowStep(dt,1.,stable,1.1,maximum,i==0);
   assert(s.accepted==initial*1.1);time+=s.accepted;dt=s.next;
 }
 assert(std::abs(time-1.1e-5)<1e-17);
 auto refreshed=ugkwp::selectGasWindowStep(dt,1.,stable,1.1,maximum,true);
 assert(refreshed.accepted==dt*1.1);
 auto reduced=ugkwp::selectGasWindowStep(dt,tail,2e-9,1.1,maximum,true);
 assert(reduced.accepted==tail && reduced.next==2e-9);
 auto capped=ugkwp::selectGasWindowStep(1e-6,2e-6,2e-6,1.1,maximum,true);
 assert(capped.accepted==maximum&&capped.next==maximum);
 std::cout<<"WINDOW_STEP_PASS refreshed_growth_only tail_seed_preserved\n";
}
