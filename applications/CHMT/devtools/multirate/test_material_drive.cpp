#include "materials/MaterialDrive.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
IntervalHistory make(int n,bool ramp){IntervalHistory h;CouplingInterval i;i.end=1;std::string e;assert(h.begin(i,e));for(int k=0;k<n;++k){GasIntervalRecord r;r.microSequence=k;r.begin=double(k)/n;r.end=double(k+1)/n;GasPrimitive g;g.rho=1;g.temperature=1+(ramp?(k+.5)/n:0);g.pressure=1;g.soundSpeed=1;g.Y[0]=1;r.gasTrace={g};r.gasTraction={{}};r.gasGradient.resize(1);assert(h.appendAccepted(r,e));}return h;}
Real integrate(const IntervalHistory& h,Real tolerance,Real maxStep,int& slabs){Real t=0,sum=0;std::string e;std::uint64_t samples=0;slabs=0;while(t<1){Real end=0;assert(materialDriveSlabEnd(h,t,std::min(1.,t+maxStep),tolerance,1,end,samples,e));std::vector<GasPrimitive> g;std::vector<Vec3> tr;assert(h.sampleGasDrive(t,g,tr,e));sum+=(end-t)*g[0].temperature;t=end;++slabs;}return sum;}
int main(){int one=0,many=0;assert(integrate(make(1,false),.05,1,one)==1);assert(integrate(make(1000,false),.05,1,many)==1);assert(one==1&&many==1);
 int coarse=0,tight=0,refined=0;const auto h=make(1000,true);const Real a=integrate(h,.1,1,coarse),b=integrate(h,.01,1,tight),c=integrate(h,.1,.01,refined);
 assert(std::abs(b-1.5)<std::abs(a-1.5));assert(std::abs(c-1.5)<std::abs(a-1.5));assert(coarse<20&&tight<200);
 std::cout<<"Boundary lag coalescing tests passed: constant refinement 1 vs "<<many<<" slabs; variable tighter controls improve forcing quadrature\n";
}
