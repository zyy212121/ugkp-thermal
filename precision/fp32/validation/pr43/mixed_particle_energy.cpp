// Internal checkpoint 2A: production recovery/publication on a serial host.
// This is not a CUDA reduction test. Only the cell publication routine from
// GpuParticleMoments is executed; its unused parallel reduction shim aborts.
#include <cassert>
#include <initializer_list>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#ifndef FIXTURE_THERMAL
#define FIXTURE_THERMAL 1
#endif
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_THERMAL FIXTURE_THERMAL
#define GPU_RECOVERY_INLINE inline
#define __device__
#define __global__
#define __host__
#define __shared__
#define __forceinline__ inline
struct ThreadIndex { int x; };
ThreadIndex blockIdx{0}, threadIdx{0}, blockDim{1};
constexpr float OfSmall=2.22044604925031308085e-16f;
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
template<class R> R sqr3(R x,R y,R z){return x*x+y*y+z*z;}
#include "wall/GpuAluminaProperties.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalCarry.H"
#include "operators/particleSpecificHeatDevice.cuh"
#include "operators/particleTemperatureFromSpecificEnthalpyDevice.cuh"
struct DeviceState {
    int nCells=1,particleCapacity=1,csrHeavyReductionEnabled=0;
    int solveParticleTemperature=1;
    float rhoSolid=1,epsSMin=1e-12f,rhoMin=1e-20f;
    float TpMin=100,TpMax=5000,particleDiameterFallback=.001f;
    float particleThermalRho=1,particleCp=1000;
    double V[1]={1},rhoEs[1]={},rhoHp[1]={};
    double momRhoEP[1]={},momRhoHpP[1]={};
    float epsS[1]={1},rhoUsx[1]={},rhoUsy[1]={},rhoUsz[1]={};
    float rhoDs[1]={.001f},Usx[1]={},Usy[1]={},Usz[1]={};
    float theta[1]={},Tp[1]={1000},dMeanCell[1]={};
    float momRhoP[1]={1},momRhoUPx[1]={},momRhoUPy[1]={};
    float momRhoUPz[1]={},momRhoPD[1]={.001f};
    int cellParticleCount[2]={},cellParticleOffset[2]={0,1};
    int sortedParticleIndex[1]={0},pStatus[1]={1},pCellId[1]={0};
    float pm[1]={1},pux[1]={},puy[1]={},puz[1]={};
    float pTheta[1]={},pd[1]={.001f},pT[1]={1000};
    double pThermalEnthalpyResidual[1]={},compactPThermalEnthalpyResidual[1]={};
};
// Gas recovery is outside this fixture: these kernels call it before the
// solid operator, but none of the fixture state has gas data to modify.
void recoverGasPrimitiveCell(DeviceState&,int){}
double particleMomentThetaDevice(const DeviceState&s,int i){return s.pTheta[i];}
float particleHeatFactorDevice(const DeviceState&s){return s.particleCp;}
#include "operators/clearSolidCell.cuh"
#include "GpuMechanicalRecovery.cuh"
#include "operators/recoverPrimitivesKernel.cuh"
#include "operators/normalizeParticleMomentsAtomicKernel.cuh"
template<int N,class R> void blockReduceComponentSums(R(&)[N],R*){std::abort();}
double energyWarpPartials[8];
#define GPU_MOMENT_REAL double
#define GPU_MOMENT_R(x) double(x)
#define GPU_MOMENT_THERMAL FIXTURE_THERMAL
#define GPU_MOMENT_GATHER 0
#include "GpuParticleMoments.cuh"
int main(){
    static_assert(std::is_same<GpuReal,float>::value);
    static_assert(std::is_same<GpuEnergy,double>::value);
    using namespace Foam::gpuThermal;
    static_assert(std::is_same<decltype(aluminaSpecificHeatJkgK(1000.f)),float>::value);
    static_assert(std::is_same<decltype(aluminaSpecificHeatJkgK(1000.)),double>::value);
    static_assert(std::is_same<decltype(aluminaSpecificEnthalpyJkg(1000.f)),float>::value);
    static_assert(std::is_same<decltype(aluminaSpecificEnthalpyJkg(1000.)),double>::value);
    static_assert(std::is_same<decltype(aluminaTemperatureFromSpecificEnthalpyK(1e6f)),float>::value);
    static_assert(std::is_same<decltype(aluminaTemperatureFromSpecificEnthalpyK(1e6)),double>::value);
    const ugkwp::ParticleCaloric law;
    for(double t : {1000.,2327.,3000.125,4000.,4500.}) {
        const double h=law.enthalpy(t);
        const double tolerance=16*std::numeric_limits<double>::epsilon();
        assert(std::abs(particleSpecificEnthalpyDevice(t)-h)<=tolerance*std::abs(h));
        assert(std::abs(particleSpecificHeatDevice(t)-law.specificHeat(t))
            <=tolerance*law.specificHeat(t));
        ugkwp::ParticleThermalCarry<float> parcel{float(t),.001};
        const double represented=ugkwp::effectiveParticleEnthalpy(parcel,law);
        DeviceState material;
        material.pT[0]=parcel.temperature;
        material.pThermalEnthalpyResidual[0]=parcel.specificEnthalpyResidual;
        const double moment=materialEnthalpyMomentFromParticle(material,0,1.,double(parcel.temperature),0);
        assert(std::abs(moment-represented)<=tolerance*std::abs(represented));
        const double inverse=particleTemperatureFromSpecificEnthalpyDevice(represented);
        assert(std::abs(inverse-law.temperature(represented))<=tolerance*std::abs(inverse));
    }
    DeviceState s;
    const double energy=5e11+1.5;
    const double h=FIXTURE_THERMAL?particleSpecificEnthalpyDevice(1000.):1e6;
    const double enthalpy=h+.001;
    s.momRhoUPx[0]=1e6f;s.momRhoEP[0]=energy;s.momRhoHpP[0]=enthalpy;
    solidRecoveryFromParticleMomentsCell(s,0);
    assert(s.rhoEs[0]==energy&&s.theta[0]==1.f);
    assert(s.rhoHp[0]==enthalpy);
    recoverPrimitivesKernel(&s);
    assert(s.rhoEs[0]==energy&&s.theta[0]==1.f);
    assert(s.rhoHp[0]==enthalpy);
    const float cache=s.Tp[0];
    for(int n=0;n<4;++n)recoverPrimitivesKernel(&s);
    assert(s.rhoHp[0]==enthalpy&&s.Tp[0]==cache);

    // Genuine out-of-range/invalid repairs still operate, not a cache roundtrip.
    s.rhoHp[0]=std::numeric_limits<double>::quiet_NaN();
    recoverPrimitivesKernel(&s);
    assert(s.Tp[0]==s.TpMin&&std::isfinite(s.rhoHp[0]));
    s.rhoHp[0]=FIXTURE_THERMAL?particleSpecificEnthalpyDevice(6000.):6e6;
    recoverPrimitivesKernel(&s);
    assert(s.Tp[0]==s.TpMax);
    const double upper=FIXTURE_THERMAL?particleSpecificEnthalpyDevice(s.TpMax):5e6;
    assert(s.rhoHp[0]==upper);

    DeviceState atomic,segmented;
    atomic.V[0]=segmented.V[0]=3.0000000000000004;
    const double sums[8]={6,2,3,4,7e10+.125,.006,1e9+.001,1};
    atomic.momRhoP[0]=sums[0];atomic.momRhoUPx[0]=sums[1];
    atomic.momRhoUPy[0]=sums[2];atomic.momRhoUPz[0]=sums[3];
    atomic.momRhoEP[0]=sums[4];atomic.momRhoPD[0]=sums[5];
    atomic.momRhoHpP[0]=sums[6];
    normalizeParticleMomentsAtomicKernel(&atomic);
    publishParticleMomentsCell(segmented,0,sums);
    const double invV=1./atomic.V[0];
    assert(atomic.momRhoEP[0]==sums[4]*invV);
    assert(atomic.momRhoHpP[0]==sums[6]*invV);
    assert(atomic.momRhoEP[0]==segmented.momRhoEP[0]);
    assert(atomic.momRhoHpP[0]==segmented.momRhoHpP[0]);
}
