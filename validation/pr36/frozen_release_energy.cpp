#define __device__
#include "GpuFrozenParticleRelease.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
struct State {
 double pColdNodeSpecificEnthalpy[8],pCold2DNodeSpecificEnthalpy[64];
 double puxOld[1]={1},puyOld[1]={2},puzOld[1]={3};
 double pThermalEnthalpyResidual[1];float pT[1];
};
int main(){
 for(int nodes:{8,64}) for(double carry:{.125,-.125}) {
  State s;s.pThermalEnthalpyResidual[0]=carry;s.pT[0]=2000;
  for(int n=0;n<8;++n)s.pColdNodeSpecificEnthalpy[n]=2546000+n;
  for(int n=0;n<64;++n)s.pCold2DNodeSpecificEnthalpy[n]=2546000+n;
  const double expected=2546000+.5*(nodes-1)+carry+7;
  const double saved=frozenReleaseSpecificEnergy(s,0,nodes);assert(saved==expected);
  publishFrozenReleaseEnergy(s,0,saved);
  const double represented=ugkwp::ParticleCaloric{}.enthalpy(s.pT[0])+s.pThermalEnthalpyResidual[0];
  assert(std::abs(represented-expected)<1e-9);
 }
 std::cout<<"PASS frozen release profile enthalpy + recoil reservoir across free caloric conversion\n";
}
