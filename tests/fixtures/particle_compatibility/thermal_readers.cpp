#include <algorithm>
#include <cstdio>
#include <cmath>
#include "GpuPrecisionTypes.H"
#include "GpuParticleCaloric.H"
#include "GpuColdWallSolidification.H"
#define __device__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
template<class T>T finiteOr(T x,T y){return std::isfinite(x)?x:y;}
template<class T>T clampMin(T x,T y){return std::max(x,y);}
template<class T>T clampRange(T x,T lo,T hi){return std::min(std::max(x,lo),hi);}
namespace Foam {namespace gpuThermal {constexpr int particleWallMobile=0;}}
struct DeviceState {
 int nFaces=1,pStuck[1]{},compactPStuck[1]{},pStuckFaceId[1]{-1},compactPStuckFaceId[1]{-1},particleStuckCandidateMask[1]{};
 GpuReal pT[1]{300},compactPT[1]{300},pm[1]{2},pd[1]{.001},TpMin=200,TpMax=4000,particleDiameterFallback=.001;
 double pThermalEnthalpyResidual[1]{},compactPThermalEnthalpyResidual[1]{};
 float pColdNodeSpecificEnthalpy[8]{},pCold2DNodeSpecificEnthalpy[64]{},compactPColdNodeSpecificEnthalpy[8]{},compactPCold2DNodeSpecificEnthalpy[64]{};
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{2327,20,1.16e6,3990,1273,5.9,.25,0,0,12};
};
#include "operators/particleSpecificHeatDevice.cuh"
#include "GpuParticleCaloricDevice.cuh"
#include "GpuRadiationContributions.cuh"
int main(){
 DeviceState s;ugkwp::ParticleCaloric law;s.pThermalEnthalpyResidual[0]=law.enthalpyIncrement(300,1);
 auto r=particleRadiationContribution(s,0);printf("radiation in_range cache=300 effective=%.17g mass=%.17g temperatureMass=%.17g\n",particleEffectiveTemperatureDevice(s,0),double(r.mass),double(r.temperatureMass));
 s.pT[0]=4100;s.pThermalEnthalpyResidual[0]=0;printf("material outside_cache=4100 max=4000 moment=%.17g\n",double(particleMaterialEnthalpyContribution(s,0)));
 s.compactPT[0]=4100;s.compactPThermalEnthalpyResidual[0]=.0125;
 printf("compact moment=%a\n",materialEnthalpyMomentFromParticle<true>(s,0,GpuReal(2),GpuReal(4000),0));
 s.pT[0]=100;s.pThermalEnthalpyResidual[0]=.0125;
 printf("lower moment=%a radiation=%a\n",double(particleMaterialEnthalpyContribution(s,0)),double(particleRadiationContribution(s,0).temperatureMass));
 auto p=s.coldWallSolidificationParameters;GpuReal h=Foam::gpuThermal::coldWallSpecificEnthalpyJkg(GpuReal(50000),p);
 printf("inverse target=50000 h=%.17g inverse=%.17g valid_parameters=%d\n",double(h),double(Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK(h,p)),Foam::gpuThermal::validColdWallSolidificationParameters(p));
 for(double t:{1000.,2327.,4000.,20000.,50000.}){auto h0=Foam::gpuThermal::coldWallSpecificEnthalpyJkg(GpuReal(t),p);printf("inverse %a %a\n",t,double(Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK(h0,p)));}
}
