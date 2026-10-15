#include "materials/ReactionStepControl.H"
#include <cassert>
#include <iostream>
#include <iomanip>
using namespace chmt;
int main(){PhysicsConfig p;p.enableReactions=true;p.nReactions=1;
 for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=2;c.Tmin=100;c.Tmax=3000;}
 for(auto& s:p.species){s.R=300;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
 p.reactions[0].A=10;p.reactions[0].order[0]=1;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;
 SolidQ q;q.condensed[0]=1;q.energy=caloricIntegral(0,1000,2,500);
 std::cout<<std::hexfloat;
 for(Real requested:{.001,.1,1.}){SolidQ out;Real dt=0;bool active=false;eulerCalls=0;
  assert(reactionStepCandidate(q,.001,p,requested,.4,out,dt,active));
  std::cout<<dt<<' '<<active<<' '<<eulerCalls<<' '<<out.energy<<' '<<out.porosity;
  for(auto x:out.condensed)std::cout<<' '<<x;
  for(auto x:out.pore)std::cout<<' '<<x;
  for(auto x:out.progress)std::cout<<' '<<x;
  std::cout<<'\n';q=out;
 }
}
