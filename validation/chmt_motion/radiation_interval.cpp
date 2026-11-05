#include "coupling/IntervalContracts.H"
#include <cassert>
#include <iostream>
int main(){
 using namespace chmt;CouplingInterval w;w.begin=10;w.end=14;IntervalHistory h;std::string e;assert(h.begin(w,e));
 for(int k=0;k<2;++k){GasIntervalRecord r;r.begin=k?11:10;r.end=k?14:11;r.microSequence=k+1;r.radiationEnergy={k?3.:1.,k?-6.:-2.};r.radiationReceiver={ConsumeSolid,ConsumeSolid};assert(h.appendAccepted(r,e));}
 h.setExternalRadiationEnergy({7.,-3.});assert(h.complete());auto c=h.cursor();std::vector<ExchangePacket> p;std::vector<Real> rad;
 double sum0=0,sum1=0;for(double t:{.125,.5,1.,2.75,4.}){const double before=c.localTime().sum;assert(c.takeThroughLocal(WindowTime(t),p,rad,e));assert(std::abs(rad[0]-2.75*(t-before))<1e-14);sum0+=rad[0];sum1+=rad[1];}
 assert(c.complete()&&sum0==11.&&sum1==-11.);assert(h.begin(w,e)&&h.externalRadiationEnergy().empty());std::cout<<"RADIATION_INTERVAL_PASS\n";
}
