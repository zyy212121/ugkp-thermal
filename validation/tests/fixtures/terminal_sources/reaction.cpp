#include "materials/ReactionStepControl.H"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace chmt;
PhysicsConfig physics(){PhysicsConfig p;p.enableReactions=true;p.nReactions=1;
 for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.Tmin=100;c.Tmax=3000;}
 for(auto& s:p.species){s.R=300;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
 p.reactions[0].A=1;p.reactions[0].order[0]=1;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;return p;}
void equal(const SolidQ& a,const SolidQ& b){
 for(int i=0;i<Nc;++i)assert(a.condensed[i]==b.condensed[i]);
 for(int i=0;i<Ns;++i)assert(a.pore[i]==b.pore[i]);
 for(int i=0;i<Nr;++i)assert(a.progress[i]==b.progress[i]);
 assert(a.energy==b.energy&&a.porosity==b.porosity);
}
int main(){auto p=physics();SolidQ q;q.condensed[0]=1;q.energy=500000;
 SolidQ strict,terminal;Real ds=0,dt=0;bool as=false,at=false;
 strictEuler=terminalEuler=0;
 assert(reactionStepCandidate(q,.001,p,.01,.1,strict,ds,as));
 const auto normalCount=strictEuler;assert(terminalEuler==0);
 strictEuler=terminalEuler=0;
 assert(reactionStepCandidate<true>(q,.001,p,.01,.1,terminal,dt,at));
 equal(strict,terminal);assert(ds==dt&&as==at);assert(strictEuler==normalCount&&terminalEuler==0);
 std::cout<<"normal_euler "<<normalCount<<'\n';
 p.tolerances.absoluteTemperature=-1e100;
 strictEuler=terminalEuler=0;strict.energy=123;ds=456;
 assert(!reactionStepCandidate(q,.001,p,.01,.1,strict,ds,as));
 assert(strict.energy==123&&ds==456);const auto exhaustedCount=strictEuler;
 assert(exhaustedCount==120&&terminalEuler==0);
 strictEuler=terminalEuler=0;
 assert(reactionStepCandidate<true>(q,.001,p,.01,.1,terminal,dt,at));
 assert(strictEuler==exhaustedCount&&terminalEuler==1&&dt==.01&&at);
 SolidQ reference;Real rates[Nr]{};
 assert(advanceLocalReactions<true>(q,.001,p,.01,reference,rates));equal(terminal,reference);
 assert(terminal.condensed[0]==.99&&terminal.progress[0]==.01);
 std::cout<<"terminal_prefix_euler "<<exhaustedCount<<" extra_euler 1 paid_dt "<<dt<<'\n';
 bool numerical=true;assert(!boundedReactionRates<true>(q,q,.001,p,.01,nullptr,nullptr,-1,&numerical));assert(!numerical);
}
