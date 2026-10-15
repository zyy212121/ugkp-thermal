// Host scaffold only. All tested pressure arithmetic is included from production.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#define __device__
#define __global__
#define __shared__
#define __forceinline__ inline
using PressureReal = TEST_REAL;
using PressureTime = double;
using R = PressureReal;
using std::fabs;
using std::fmin;
using std::fmax;
using std::sqrt;
struct Dim { int x; };
Dim blockIdx{0}, blockDim{1}, threadIdx{0}, gridDim{1};
inline void __syncthreads() {}
// A block reduction is the identity for this harness's single executing lane.
// GPU reduction ordering and synchronization are deliberately not simulated.
template<int N> void blockReduceComponentSums(PressureReal (&)[N],PressureReal*) {
    if(blockDim.x!=1||threadIdx.x!=0)std::abort();
}
template<class T> T atomicAdd(T* address, T value) { T old=*address; *address+=value; return old; }
inline unsigned int atomicOr(unsigned int* address, unsigned int value) {
    unsigned int old=*address; *address|=value; return old;
}
inline unsigned int atomicCAS(unsigned int* address, unsigned int expected, unsigned int value) {
    unsigned int old=*address; if(old==expected)*address=value; return old;
}
template<class T> T clampMin(T value,T low) {return std::max(value,low);}
template<class T> T clampRange(T value,T low,T high) {return std::min(std::max(value,low),high);}
template<class T> bool finiteDevice(T value) {return std::isfinite(value);}
template<class T> T finiteOr(T value,T fallback) {return finiteDevice(value)?value:fallback;}
template<class T> T sqr3(T x,T y,T z) {return x*x+y*y+z*z;}
constexpr R OfVSmall=R(1e-30);
struct DeviceState {
    int nCells=1,nFaces=1,particleCapacity=4;
    int particleCount=2;
    int* particleCountDevice=&particleCount;
    R rhoMin=R(1e-20),particleDiameterFallback=1,TpMin=1,TpMax=1000;
    R epsSMin=R(1e-8),rhoSolid=1,thetaMin=R(.01),pressureKickFraction=10;
    R V[2]={1},cellLength[2]={1};
    R momRhoP[2]={2},momRhoUPx[2]={0},momRhoUPy[2]={0},momRhoUPz[2]={0},momRhoEP[2]={R(3.25)};
    R rhoUsx[2]={0},rhoUsy[2]={0},rhoUsz[2]={0},rhoEs[2]={R(3.25)};
    R Usx[2]={0},Usy[2]={0},Usz[2]={0},theta[2]={R(3.25/3)},epsS[2]={2};
    int cellParticleCount[3]={2};
    R momRhoPD[2]={2},momRhoHpP[2]={2};
    R pressureParticleMoments[14]={}; int pressureParticleCount[2]={};
    R pd[4]={1,1},pT[4]={1,1},compactPd[4]={1,1},compactPT[4]={1,1};
    R puxOld[4]={1,1},puyOld[4]={1,1},puzOld[4]={1,1};
    R compactPuxOld[4]={1,1},compactPuyOld[4]={1,1},compactPuzOld[4]={1,1};
    int cellPlaneStart[2]={0},cellPlaneCount[2]={1},cellFaceId[2]={0};
    int faceOwner[2]={0},faceNeighbour[2]={-1};
    R solidPressurePhiMomX[2]={-1},solidPressurePhiMomY[2]={R(-.25)},solidPressurePhiMomZ[2]={R(.5)},solidPressurePhiEnergy[2]={R(-.2)};
    R pressureDeltaMomX[2]={0},pressureDeltaMomY[2]={0},pressureDeltaMomZ[2]={0},pressureDeltaEnergy[2]={0};
    R pressureKickScale[2]={0};
    int pStatus[4]={1,1,0,0},pCellId[4]={0,0,-1,-1},pStuck[4]={0};
    R pm[4]={1,1,0,0},pux[4]={R(-.5),R(.5),0,0},puy[4]={0},puz[4]={0},pTheta[4]={1,1,0,0};
    int compactPStatus[4]={1,1,0,0},compactPCellId[4]={0,0,-1,-1},compactPStuck[4]={0};
    R compactPm[4]={1,1,0,0},compactPux[4]={R(-.5),R(.5),0,0},compactPuy[4]={0},compactPuz[4]={0},compactPTheta[4]={1,1,0,0};
    int preBaseCellOffset[3]={0,0,0},cellParticleOffset[3]={0,2,4},sortedParticleIndex[4]={0,1,2,3};
};

namespace Foam { namespace gpuThermal { constexpr int particleWallDeposited=1; } }
template<class T> T __shfl_down_sync(unsigned,T,int) { return 0; }
inline int __popc(unsigned x) { return __builtin_popcount(x); }
R pressureWarpPartials[8]={};
template<bool Compact> R particleMomentThetaDevice(const DeviceState& s,int i) {
    if((Compact?s.compactPStuck[i]:s.pStuck[i])!=0)return 0;
    return clampMin(finiteOr(Compact?s.compactPTheta[i]:s.pTheta[i],R(0)),R(0));
}
R particleSpecificEnthalpyDevice(R t) {return t;}
#include "GpuPressureConstrainedPhysics.cuh"
#include "GpuPressureCellTraversal.cuh"
#include "GpuPressureConstrainedAdapter.cuh"
#include "GpuPressureUnsortedConstrained.cuh"
#define GPU_OPERATOR_TIME PressureTime
#include "operators/applyCollisionalPressureProjectionKernel.cuh"
void check(bool yes,const char* message) {
    if(!yes){std::cerr<<message<<'\n';std::exit(1);}
}
void near(R actual,R expected,const char* message) {
    const R tol=R(256)*std::numeric_limits<R>::epsilon()*std::max(R(1),fabs(expected));
    if(!(fabs(actual-expected)<=tol)) {
        std::cerr<<message<<": actual="<<actual<<" expected="<<expected<<'\n';std::exit(1);
    }
}
void delta(DeviceState& s,R px,R py,R pz,R e) {
    s.solidPressurePhiMomX[0]=-px;s.solidPressurePhiMomY[0]=-py;
    s.solidPressurePhiMomZ[0]=-pz;s.solidPressurePhiEnergy[0]=-e;
}
void compactCopy(DeviceState& s) {
    for(int i=0;i<s.particleCapacity;++i) {
        s.compactPStatus[i]=s.pStatus[i];s.compactPCellId[i]=s.pCellId[i];s.compactPStuck[i]=s.pStuck[i];
        s.compactPm[i]=s.pm[i];s.compactPux[i]=s.pux[i];s.compactPuy[i]=s.puy[i];s.compactPuz[i]=s.puz[i];s.compactPTheta[i]=s.pTheta[i];
    }
}

template<bool Full>
void project(DeviceState& s,const std::string& route) {
    if(route=="unsorted") {
        ConstrainedPressureScratch<Full>::initialise(s,0);
        applyCollisionalPressureProjectionCellAtomicKernel(&s,1);
        applyCollisionalPressureProjectionParticlesAtomicKernel<Full>(&s);
        publishPressureParticleMomentsKernel<Full>(&s);
    } else if(route=="segments") {
        s.preBaseCellOffset[1]=1;s.cellParticleOffset[0]=1;
        accumulateCollisionalPressureKickByCellKernel<Full>(&s,1,0,1);
        applyCollisionalPressureProjectionSplitSegmentKernel<true,Full>(&s);
        applyCollisionalPressureProjectionSplitSegmentKernel<false,Full>(&s);
        publishPressureParticleMomentsKernel<Full>(&s);
    } else if(route=="split") {
        s.preBaseCellOffset[1]=1;s.cellParticleOffset[0]=1;
        applyUnifiedSplitPressureKernel<Full>(&s,1);
    } else if(route=="compact") applyCollisionalPressureProjectionKernel<Full,true>(&s,1);
    else applyCollisionalPressureProjectionKernel<Full,false>(&s,1);
}
void verifyActualClosure(const DeviceState& s,bool compact) {
    R sum[7]={};int count=0;
    for(int i=0;i<2;++i) {
        if(compact)accumulatePressureParticleMomentsDevice<true,true>(s,0,i,sum,count);
        else accumulatePressureParticleMomentsDevice<true,false>(s,0,i,sum,count);
    }
    near(s.momRhoUPx[0],sum[0]/s.V[0],"actual x momentum not published");
    near(s.momRhoUPy[0],sum[1]/s.V[0],"actual y momentum not published");
    near(s.momRhoUPz[0],sum[2]/s.V[0],"actual z momentum not published");
    near(s.momRhoEP[0],sum[3]/s.V[0],"actual particle energy not published");
    near(s.momRhoP[0],sum[4]/s.V[0],"actual density changed");
    check(s.cellParticleCount[0]==count,"actual particle count not published");
    check(s.theta[0]>=0,"negative closed theta");
}
int main(int argc,char** argv) {
    check(argc==2,"mode required");const std::string mode=argv[1];
    if(mode=="particle_clamp"||mode=="nonfinite_particle") {
        for(bool compact:{false,true}) {
            DeviceState s;s.pTheta[0]=mode=="particle_clamp"?R(-1):std::numeric_limits<R>::quiet_NaN();
            if(mode=="nonfinite_particle")s.pux[0]=std::numeric_limits<R>::infinity();
            compactCopy(s);
            if(compact)updateMobilePressureParticle<true>(s,0,R(0),R(0),R(0),R(1),R(0),R(0),R(1),R(2),R(4),true);
            else updateMobilePressureParticle<false>(s,0,R(0),R(0),R(0),R(1),R(0),R(0),R(1),R(2),R(4),true);
            check((compact?s.compactPTheta[0]:s.pTheta[0])==0,"original particle theta clamp missing");
            check(finiteDevice(compact?s.compactPux[0]:s.pux[0]),"original velocity fallback missing");
        }
    } else if(mode=="closure_clamp") {
        DeviceState s;R moments[7]={2,0,0,std::nextafter(R(1),R(0)),2,2,2};
        publishPressureParticleMomentsDevice<true>(s,0,moments,2);
        check(s.momRhoEP[0]==1&&s.rhoEs[0]==1&&s.theta[0]==0,"original negative internal-energy closure clamp missing");
    } else if(mode=="cold_initial") {
        for(const std::string route:{"sorted","compact","split","segments","unsorted"}) {
            DeviceState s;s.momRhoUPx[0]=2;s.momRhoEP[0]=std::nextafter(R(1),R(0));
            s.pux[0]=s.pux[1]=1;s.pTheta[0]=s.pTheta[1]=0;compactCopy(s);
            s.pressureKickScale[0]=pressureLocalConvexScale(s,0,1);
            check(s.pressureKickScale[0]==0,"invalid convex initial state must limit pressure to zero");
            scaleCollisionalPressureFaceFluxKernel(&s);project<true>(s,route);
            check(s.momRhoEP[0]==1&&s.theta[0]==0,"cold finite state did not reach original clamp closure");
        }
    } else if(mode=="limited_faces") {
        DeviceState s;s.nCells=2;s.nFaces=1;s.V[1]=2;s.cellLength[1]=1;
        s.cellPlaneCount[1]=1;s.cellFaceId[1]=0;s.cellPlaneStart[1]=1;
        s.faceNeighbour[0]=1;s.momRhoP[1]=2;s.momRhoEP[1]=R(3.25);
        s.pressureKickFraction=R(.01);
        for(int c=0;c<2;++c)s.pressureKickScale[c]=pressureLocalConvexScale(s,c,1);
        scaleCollisionalPressureFaceFluxKernel(&s);
        R a[4],b[4];pressureDeltaFromLimitedFaces(s,0,1,a);pressureDeltaFromLimitedFaces(s,1,1,b);
        for(int k=0;k<4;++k)near(a[k]*s.V[0]+b[k]*s.V[1],R(0),"shared face broke conservative balance");
        for(int c=0;c<2;++c){const R* d=c?b:a;
            check(pressure_convex::admissibleIncrement(s.momRhoP[c],R(0),R(0),R(0),s.momRhoEP[c],R(.03),R(.01),d[0],d[1],d[2],d[3]),"shared minimum violated convex constraint");}
    } else if(mode=="nonfinite_delta") {
        DeviceState s;
        s.solidPressurePhiMomX[0]=std::numeric_limits<R>::quiet_NaN();
        s.solidPressurePhiEnergy[0]=std::numeric_limits<R>::infinity();
        R d[4];pressureDeltaFromLimitedFaces(s,0,1,d);
        check(d[0]==0&&d[3]==0,"original finite delta fallback missing");
    } else if(mode=="scaling_subnormals") {
        // Gradual-underflow host contract only. FTZ/DAZ does not preserve
        // subnormal inputs, so the separate FTZ test uses normal inputs.
        const R tiny=std::numeric_limits<R>::denorm_min(),small=std::numeric_limits<R>::min();
        volatile R values[]={tiny,tiny*R(17),small/R(2),small,R(1),std::numeric_limits<R>::max()};
        for(int i=0;i<6;++i)for(int j=0;j<6;++j) {
            const R mass=values[i],energy=values[j];
            const long double expected=std::sqrt(2.L*mass*energy);
            const R actual=pressure_convex::momentumClosureScale(mass,energy);
            if(expected>static_cast<long double>(std::numeric_limits<R>::max()))check(!finiteDevice(actual),"subnormal matrix overflow was hidden");
            else check(finiteDevice(actual)&&std::fabs(static_cast<long double>(actual)-expected)<=std::max(static_cast<long double>(tiny),expected*4*std::numeric_limits<R>::epsilon()),"gradual-underflow closure scale lost representable value");
        }
        for(int i=0;i<4;++i) {
            const R x=values[i];
            check(pressure_convex::norm3(x,R(0),R(0))==x,"gradual-underflow norm lost nonzero component");
            const long double expected=std::sqrt(3.L)*x;
            const R actual=pressure_convex::norm3(x,x,x);
            check(std::fabs(static_cast<long double>(actual)-expected)<=std::max(static_cast<long double>(tiny),expected*4*std::numeric_limits<R>::epsilon()),"gradual-underflow diagonal norm lost precision");
        }
    } else if(mode=="scaling_boundaries"||mode=="scaling_boundaries_ftz") {
#if defined(__SSE__)
        const unsigned savedCsr=_mm_getcsr();
        if(mode=="scaling_boundaries_ftz")_mm_setcsr(savedCsr|0x8040u);
#endif
        const R small=std::numeric_limits<R>::min(),big=std::numeric_limits<R>::max();
        volatile R inputs[]={0,small,small*R(16),R(.5),R(1),R(2),big/R(2),big};
        for(int i=0;i<8;++i)for(int j=0;j<8;++j) {
            const R mass=inputs[i],energy=inputs[j];
            const long double expected=std::sqrt(2.L*mass*energy);
            const R actual=pressure_convex::momentumClosureScale(mass,energy);
            if(expected>static_cast<long double>(big))check(!finiteDevice(actual),"truly overflowing closure scale must remain nonfinite");
            else if(expected==0)check(actual==0,"zero closure scale must remain zero");
            else check(finiteDevice(actual)&&std::fabs(static_cast<long double>(actual)/expected-1)<=4*std::numeric_limits<R>::epsilon(),"scaled closure differs from independent wide-exponent reference");
        }
        for(R value:{small,small*R(16),R(1),big/R(2),big}) {
            volatile R input=value;const R x=input;
            check(pressure_convex::norm3(x,R(0),R(0))==x,"single-axis norm lost a finite normal component");
            for(int axes=2;axes<=3;++axes) {
                const long double expected=std::sqrt(static_cast<long double>(axes))*x;
                const R actual=pressure_convex::norm3(x,-x,axes==3?x:R(0));
                if(expected>static_cast<long double>(big))check(!finiteDevice(actual),"truly overflowing norm must remain nonfinite");
                else check(finiteDevice(actual)&&std::fabs(static_cast<long double>(actual)/expected-1)<=4*std::numeric_limits<R>::epsilon(),"scaled norm differs from independent wide-exponent reference");
            }
        }
        check(pressure_convex::norm3(R(0),R(0),R(0))==0,"zero norm changed");
        const R nan=std::numeric_limits<R>::quiet_NaN(),inf=std::numeric_limits<R>::infinity();
        for(R bad:{R(-1),nan,inf,-inf})for(R other:{R(0),R(1)}) {
            check(!finiteDevice(pressure_convex::momentumClosureScale(bad,other)),"invalid mass hidden by zero energy");
            check(!finiteDevice(pressure_convex::momentumClosureScale(other,bad)),"invalid energy hidden by zero mass");
        }
        for(R bad:{nan,inf,-inf}) {
            check(!finiteDevice(pressure_convex::norm3(bad,R(0),R(0))),"nonfinite x hidden in norm");
            check(!finiteDevice(pressure_convex::norm3(R(0),bad,R(0))),"nonfinite y hidden in norm");
            check(!finiteDevice(pressure_convex::norm3(R(0),R(0),bad)),"nonfinite z hidden in norm");
        }
#if defined(__SSE__)
        _mm_setcsr(savedCsr);
#endif
    } else if(mode=="invalid_face") {
        DeviceState s;s.cellFaceId[0]=10;
        check(pressureLocalConvexScale(s,0,1)==0,"invalid face index accepted");
    } else {
        std::string route=mode;const bool zero=mode.rfind("zero_",0)==0,mixed=mode.rfind("mixed_",0)==0;
        if(zero)route=mode.substr(5);if(mixed)route=mode.substr(6);
        for(bool full:{false,true}) {
            DeviceState s;if(zero)delta(s,0,0,0,0);
            if(mixed){s.pStuck[1]=Foam::gpuThermal::particleWallDeposited;s.pTheta[1]=R(77);}
            compactCopy(s);
            // No preflight is called. A stale delta cache must never be read.
            s.pressureDeltaMomX[0]=R(999);s.pressureDeltaEnergy[0]=R(-999);
            if(full)project<true>(s,route);else project<false>(s,route);
            verifyActualClosure(s,route=="compact");
            near(s.pressureDeltaMomX[0],-s.solidPressurePhiMomX[0],"limited face delta was not refreshed");
            if(mixed){check((route=="compact"?s.compactPux[1]:s.pux[1])==0,"stuck particle moved");
                check((route=="compact"?s.compactPuxOld[1]:s.puxOld[1])==0,"deposited old velocity not reset");
                check((route=="compact"?s.compactPTheta[1]:s.pTheta[1])==R(77),"contact-age storage was clamped");}
        }
    }
}
