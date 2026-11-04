#include "wall/GpuFiniteWallContact.H"
#include <cassert>
#include <iostream>
int main() {
 using Foam::gpuThermal::reboundContactRelease;
 for(double frozen:{0.,.25,2.}) {
   auto r=reboundContactRelease(1.,frozen,0.);
   assert(!r.detach && r.atRest==(frozen>0));
   r=reboundContactRelease(0.,frozen,0.);
   assert(r.detach && r.atRest==(frozen>0));
   r=reboundContactRelease(1.,frozen,3.);
   assert(r.detach && r.atRest==(frozen>0));
 }
 std::cout<<"PASS finite rebound release, frozen zero-speed, unfrozen reflected-speed\n";
}
