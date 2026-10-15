#include <cstdio>
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
int main(){ugkwp::ParticleCaloric law;ugkwp::ParticleThermalCarry<double> p{3000,0};
 auto a=ugkwp::publishAffineRadiationThermalHeat(p,1.01,0.,200.,4000.,law);
 p={3000,0};auto b=ugkwp::publishAffineRadiationThermalHeat(p,1.,2000.,200.,4000.,law);
 p={4100,0};auto c=ugkwp::publishParticleThermalHeat(p,-1.,200.,4000.,law);
 std::printf("%d %d %d\n",a.valid,b.valid,c.valid);}
