#include "TRANSPORT_OWNER"
#include <cassert>
#include <iostream>
using namespace chmt;
int main(){PhysicsConfig p;p.enableReactions=true;p.nReactions=1;
 for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.Tmin=100;c.Tmax=3000;}
 for(auto& s:p.species){s.R=300;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
 p.reactions[0].A=1;p.reactions[0].order[0]=1;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;
 SolidQ q;q.condensed[0]=1;q.energy=500000;SolidQ out;std::uint64_t count=0;std::string error;bool numerical=false;
 proposalCalls=0;
 assert(!advanceMaterialReactions(q,.001,p,1,0,.1,10,out,count,error,&numerical));
 assert(numerical&&error.find("advance time")!=std::string::npos&&proposalCalls==2);
 proposalCalls=0;
 assert(detail::completeMaterialReactions(q,.001,p,1,0,.1,10,out,count,error,&numerical));
 assert(count==2&&proposalCalls==2);
 SolidQ first,expected;Real rates[Nr]{};
 assert(advanceLocalReactions<true>(q,.001,p,.5,first,rates));
 assert(advanceLocalReactions<true>(first,.001,p,.5,expected,rates));
 for(int c=0;c<Nc;++c)assert(out.condensed[c]==expected.condensed[c]);
 for(int r=0;r<Nr;++r)assert(out.progress[r]==expected.progress[r]);
 assert(out.energy==expected.energy&&out.porosity==expected.porosity);
 // Inconsistent computed reaction state reaches the real packet sum; no clip
 // replaces a significant negative inventory on the terminal route.
 q.condensed[0]=-2;std::vector<SolidQ> batch;Real correction=0;
 assert(!applyMaterialPacketBatch({q},{},batch,correction,error,&numerical)&&numerical);
 assert(detail::completeMaterialPacketBatch({q},{},batch,correction,error,&numerical));
 assert(batch[0].condensed[0]==-2&&batch[0].energy==q.energy&&correction==0);
 std::cout<<"stagnation_real_euler paid_dt 0.5 interval 1 accepted 2; packet_tail negative preserved\n";
}
