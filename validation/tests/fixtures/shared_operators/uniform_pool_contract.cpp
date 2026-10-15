#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#define __device__
#define __forceinline__ inline
#define asm(x) assert(false && x)
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) (x)
#define GPU_POOL_MASS_FALLBACK 1.
#define GPU_POOL_THETA_AFTER_REJECTION 1
#define GPU_POOL_PARTICLE_THETA(s,i) ((s).pTheta[i])
std::vector<int> events;
template<class T, int Read, int Write> struct Observed {
    T value{};
    struct Proxy {
        T& value;
        operator T() const { events.push_back(Read); return value; }
        Proxy& operator=(T x) { events.push_back(Write); value=x; return *this; }
    };
    Proxy operator[](int) { return {value}; }
};
struct DeviceState {
    int particleCapacity=1; int pCellId[1]={0};
    Observed<int,1,90> pStatus{{1}};
    Observed<unsigned long long,2,80> pRng{{123}};
    Observed<double,3,30> pm{{2}};
    Observed<double,4,40> pux{{1}},puy{{2}},puz{{3}};
    Observed<double,5,50> pTheta{{.75}};
    Observed<double,6,60> pd{{.25}};
    double thetaMin=1e-9,particleDiameterFallback=.1;
};
double finiteOr(double x,double fallback){return std::isfinite(x)?x:fallback;}
double clampMin(double x,double low){return x<low?low:x;}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
bool nonFiniteDevice(double x){return !std::isfinite(x);}
double uniform01Device(unsigned long long& x){++x;return .5;}
#include "GpuCollisionPoolParticle.cuh"
int main(){
  for(double probability:{0.,.25,1.}) {
    DeviceState actual,reference;
    double a[8]={},b[8]={};
    events.clear();
    accumulateOnePoolParticle<true>(actual,0,0,probability,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
    const auto observed=events;
    accumulateOnePoolParticle<true,false>(reference,0,0,probability,b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
    assert(std::memcmp(a,b,sizeof a)==0 && actual.pRng.value==reference.pRng.value && actual.pStatus.value==reference.pStatus.value);
    if(probability==1.) {
      assert(observed.size()>=2);
      // Accepted state is committed only after all physical loads/contributions.
      assert(observed[observed.size()-2]==80 && observed.back()==90);
    } else {
      assert(actual.pRng.value==124 && actual.pStatus.value==1);
      for(int event:observed) assert(event!=3 && event!=4 && event!=5 && event!=6);
    }
  }
  DeviceState plain; double c[8]={};
  accumulateOnePoolParticle<false>(plain,0,0,1.,c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]);
  assert(plain.pRng.value==123 && plain.pStatus.value==2 && c[0]==2 && c[7]==1);
  puts("PASS uniform Poisson access order, bitwise moments, RNG/status and non-Poisson semantics");
}
