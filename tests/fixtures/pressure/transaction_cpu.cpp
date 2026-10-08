// Host scaffold only. All tested pressure arithmetic is included from production.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
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
    R epsSMin=R(1e-8),rhoSolid=1,thetaMin=R(.01),pressureKickFraction=10;
    R V[2]={1},cellLength[2]={1};
    R momRhoP[2]={2},momRhoUPx[2]={0},momRhoUPy[2]={0},momRhoUPz[2]={0},momRhoEP[2]={R(3.25)};
    R rhoUsx[2]={0},rhoUsy[2]={0},rhoUsz[2]={0},rhoEs[2]={R(3.25)};
    R Usx[2]={0},Usy[2]={0},Usz[2]={0},theta[2]={R(3.25/3)},epsS[2]={2};
    int cellParticleCount[2]={2};
    int cellPlaneStart[2]={0},cellPlaneCount[2]={1},cellFaceId[2]={0};
    int faceOwner[2]={0},faceNeighbour[2]={-1};
    R solidPressurePhiMomX[2]={-1},solidPressurePhiMomY[2]={R(-.25)},solidPressurePhiMomZ[2]={R(.5)},solidPressurePhiEnergy[2]={R(-.2)};
    R pressureDeltaMomX[2]={0},pressureDeltaMomY[2]={0},pressureDeltaMomZ[2]={0},pressureDeltaEnergy[2]={0};
    R pressurePreviewMoments[20]={0},pressureKickScale[2]={0};
    unsigned int pressureFailure[3]={0};
    int pStatus[4]={1,1,0,0},pCellId[4]={0,0,-1,-1},pStuck[4]={0};
    R pm[4]={1,1,0,0},pux[4]={R(-.5),R(.5),0,0},puy[4]={0},puz[4]={0},pTheta[4]={1,1,0,0};
    int compactPStatus[4]={1,1,0,0},compactPCellId[4]={0,0,-1,-1},compactPStuck[4]={0};
    R compactPm[4]={1,1,0,0},compactPux[4]={R(-.5),R(.5),0,0},compactPuy[4]={0},compactPuz[4]={0},compactPTheta[4]={1,1,0,0};
    int preBaseCellOffset[3]={0,0,0},cellParticleOffset[3]={0,2,4},sortedParticleIndex[4]={0,1,2,3};
};
// Constraint policy is an application interface. Test both free and constrained
// records without importing thermal application types into the shared harness.
struct PressureConstraintPolicy {
    template<bool Compact> static bool stuck(const DeviceState& s,int i) {
        return (Compact?s.compactPStuck[i]:s.pStuck[i])!=0;
    }
};
#include "GpuPressureFailure.cuh"
#include "GpuPressureLimiter.cuh"
#include "GpuPressureParticleUpdate.cuh"
#include "GpuPressureCellTraversal.cuh"
#include "GpuPressureKickAccumulation.cuh"
#include "PressurePreflightHost.cuh"
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
void preview(DeviceState& s,const std::string& route) {
    preparePressurePreflightKernel(&s,1);
    if(route=="unsorted")previewPressureUnsortedKernel(&s);
    else if(route=="compact")previewPressureSortedKernel<false,true>(&s);
    else if(route=="split") {
        s.preBaseCellOffset[1]=1;s.cellParticleOffset[0]=1;
        previewPressureSortedKernel<true,false>(&s);
    } else previewPressureSortedKernel<false,false>(&s);
    auditPressurePreviewKernel(&s,1);
}
void applyMobileWriter(DeviceState& s,const std::string& route) {
    const auto q=pressurePreviewParameters(s,0,route=="unsorted");
    for(int i=0;i<2;++i) {
        if(route=="compact")updateMobilePressureParticle<true>(s,i,q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,q.theta1,q.thermalScale,q.thetaScale,q.resolved);
        else updateMobilePressureParticle<false>(s,i,q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,q.theta1,q.thermalScale,q.thetaScale,q.resolved);
    }
}
void actualMoments(const DeviceState& s,const std::string& route,R (&sum)[5]) {
    for(int i=0;i<2;++i) {
        const bool compact=route=="compact";
        const R m=compact?s.compactPm[i]:s.pm[i];
        const R x=compact?s.compactPux[i]:s.pux[i],y=compact?s.compactPuy[i]:s.puy[i],z=compact?s.compactPuz[i]:s.puz[i];
        const R theta=compact?s.compactPTheta[i]:s.pTheta[i];
        sum[0]+=m*x;sum[1]+=m*y;sum[2]+=m*z;
        sum[3]+=m*(R(.5)*(x*x+y*y+z*z)+R(1.5)*theta);sum[4]+=m;
    }
}
struct MustNotRunOperation {
    static constexpr bool closeInactiveCells=false;
    struct Shared {};
    struct Accumulator {};
    template<PressureDirectory> static void prepare(DeviceState&,int,PressureTime,Shared&,PressureParameters& q) {
        q={};check(false,"zero/failure transaction entered a physical writer");
    }
    template<bool,bool> static void visit(DeviceState&,int,int,bool,R,R,R,R,R,R,R,R,R,bool,Accumulator&) {
        check(false,"zero/failure transaction visited a particle");
    }
    template<PressureDirectory> static void finish(DeviceState&,int,const PressureParameters&,Accumulator&) {
        check(false,"zero/failure transaction published cell closure");
    }
};
void exerciseWriterGuards(DeviceState& s) {
    runCellPressureProjection<MustNotRunOperation,PressureDirectory::Full,false>(&s,1);
    runCellPressureProjection<MustNotRunOperation,PressureDirectory::Full,true>(&s,1);
    runCellPressureProjection<MustNotRunOperation,PressureDirectory::Split,false>(&s,1);
    runCellPressureProjection<MustNotRunOperation,PressureDirectory::Base,false>(&s,1);
    runCellPressureProjection<MustNotRunOperation,PressureDirectory::Injection,false>(&s,1);
    publishPressureCanonicalMomentsKernel(&s);
}
void assertFailure(DeviceState& s,unsigned int reason,const char* message) {
    check((s.pressureFailure[0]&reason)!=0,message);
    check(s.pressureFailure[1]==1,"first failure cell must be recorded as cell+1");
    check(s.pressureFailure[2]!=0,"first failure reason was omitted");
    DeviceState before=s;
    exerciseWriterGuards(s);
    check(std::memcmp(&s,&before,sizeof(s))==0,"failure transaction wrote physical state");
}
int main(int argc,char** argv) {
    check(argc==2,"one test mode required");
    const std::string mode=argv[1];
    if(mode.rfind("mobile_",0)==0) {
        const auto route=mode.substr(7);DeviceState s;
        if(route=="volume") {s.V[0]=2;s.momRhoP[0]/=2;s.momRhoEP[0]/=2;}
        const R target[5]={1,R(.25),R(-.5),s.momRhoEP[0]*s.V[0]+R(.2),2};
        preview(s,route);check(s.pressureFailure[0]==0,"all-mobile realizable transaction rejected");
        applyMobileWriter(s,route);
        R actual[5]={};actualMoments(s,route,actual);
        for(int k=0;k<5;++k) {
            near(actual[k],target[k],"real writer failed target closure");
            near(actual[k],s.pressurePreviewMoments[k],"preview and real writer differ");
        }
        publishPressureCanonicalMomentsKernel(&s);
        near(s.momRhoUPx[0],actual[0]/s.V[0],"canonical momentum differs from particles");
        near(s.momRhoEP[0],actual[3]/s.V[0],"canonical energy differs from particles");
        near(s.momRhoP[0],actual[4]/s.V[0],"canonical density differs from particles");
        near(s.theta[0],(actual[3]-(actual[0]*actual[0]+actual[1]*actual[1]+actual[2]*actual[2])/(R(2)*actual[4]))/(R(1.5)*actual[4]),"canonical internal energy closure differs");
    } else if(mode.rfind("mixed_",0)==0) {
        const auto route=mode.substr(6);DeviceState s;
        s.pux[0]=s.pux[1]=0;s.pTheta[0]=2;s.pTheta[1]=0;s.pStuck[1]=1;s.momRhoEP[0]=3;
        compactCopy(s);delta(s,1,0,0,0);preview(s,route);
        near(s.pressurePreviewMoments[0],R(.5),"mixed actual momentum fixture changed");
        near(s.pressurePreviewMoments[3],R(2.875),"mixed actual energy fixture changed");
        assertFailure(s,pressureUnrealizableParticles,"mixed free/stuck target was accepted despite nonconservation");
        near(s.pux[0],0,"preflight mutated mobile velocity");near(s.pTheta[0],2,"preflight mutated mobile theta");
    } else if(mode=="initial_mismatch") {
        for(int field=0;field<3;++field) {
            DeviceState s;
            s.thetaMin=R(.2); // Unresolved reconstruction hides the initial mismatch in q1.
            if(field==0){s.pux[0]+=R(.2);s.pux[1]+=R(.2);}
            if(field==1){s.pTheta[0]+=R(.2);s.pTheta[1]+=R(.2);}
            if(field==2){s.pm[0]+=R(.2);s.pm[1]+=R(.2);}
            preview(s,"sorted");assertFailure(s,pressureUnrealizableParticles,"initial particle/moment mismatch accepted");
        }
    } else if(mode=="nonfinite_initial") {
        for(R bad:{std::numeric_limits<R>::quiet_NaN(),std::numeric_limits<R>::infinity(),-std::numeric_limits<R>::infinity()}) {
            for(int field=0;field<5;++field) {
                DeviceState s;R* fields[]={s.momRhoP,s.momRhoUPx,s.momRhoUPy,s.momRhoUPz,s.momRhoEP};*fields[field]=bad;
                preparePressurePreflightKernel(&s,1);assertFailure(s,pressureBadInitial,"nonfinite initial moment sanitized");
            }
        }
    } else if(mode=="nonfinite_particle") {
        for(R bad:{std::numeric_limits<R>::quiet_NaN(),std::numeric_limits<R>::infinity(),-std::numeric_limits<R>::infinity()})
            for(int field=0;field<5;++field)for(const std::string route:{"sorted","compact","split","unsorted"})for(int zero=0;zero<2;++zero) {
                DeviceState s;R* fields[]={s.pm,s.pux,s.puy,s.puz,s.pTheta};*fields[field]=bad;
                if(zero)delta(s,0,0,0,0);
                compactCopy(s);preview(s,route);
                assertFailure(s,pressureBadParticle,"nonfinite particle input was silently sanitized or skipped");
            }
    } else if(mode=="negative_particle") {
        for(int field=0;field<2;++field) {
            DeviceState s;if(field==0)s.pm[0]=-1;else s.pTheta[0]=-1;
            preview(s,"sorted");assertFailure(s,pressureBadParticle,"negative raw particle input sanitized");
        }
    } else if(mode=="invalid_initial") {
        for(int field=0;field<5;++field) {
            DeviceState s;
            if(field==0)s.momRhoP[0]=-1;
            if(field==1)s.momRhoEP[0]=-1;
            if(field==2)s.momRhoUPx[0]=10;
            if(field==3)s.V[0]=0;
            if(field==4){s.momRhoP[0]=0;s.momRhoEP[0]=0;s.momRhoUPx[0]=1;}
            preparePressurePreflightKernel(&s,1);assertFailure(s,pressureBadInitial,"invalid initial state accepted");
        }
    } else if(mode=="particle_overflow") {
        DeviceState s;s.pux[0]=std::numeric_limits<R>::max();preview(s,"sorted");
        assertFailure(s,pressureBadParticle,"finite raw particle with overflowing candidate energy accepted");
    } else if(mode=="zero_initial_mismatch") {
        for(int field=0;field<3;++field) {
            DeviceState s;delta(s,0,0,0,0);
            if(field==0)s.pux[0]+=R(.2);
            if(field==1)s.pTheta[0]+=R(.2);
            if(field==2)s.pm[0]+=R(.2);
            preview(s,"sorted");assertFailure(s,pressureUnrealizableParticles,"zero delta concealed an initial particle/moment mismatch");
        }
    } else if(mode=="inactive") {
        for(int mismatch=0;mismatch<2;++mismatch) {
            DeviceState s;s.epsSMin=3;
            s.pressureKickScale[0]=pressureLocalConvexScale(s,0,1);
            check(s.pressureKickScale[0]==0,"inactive cell retained a pressure kick");
            scaleCollisionalPressureFaceFluxKernel(&s);
            if(mismatch)s.pm[0]+=R(.2);
            preview(s,"sorted");
            if(mismatch)assertFailure(s,pressureUnrealizableParticles,"inactive cell concealed an initial mass mismatch");
            else {
                check(s.pressureFailure[0]==0,"consistent inactive cell rejected");
                const DeviceState before=s;exerciseWriterGuards(s);
                check(std::memcmp(&s,&before,sizeof(s))==0,"inactive zero-kick cell was modified");
            }
        }
    } else if(mode=="empty") {
        for(int occupied=0;occupied<2;++occupied) {
            DeviceState s;s.momRhoP[0]=0;s.momRhoEP[0]=0;delta(s,0,0,0,0);
            if(!occupied){s.pStatus[0]=s.pStatus[1]=0;s.cellParticleOffset[1]=0;s.particleCount=0;}
            preview(s,"sorted");
            if(occupied)assertFailure(s,pressureUnrealizableParticles,"macro-empty cell with finite particle mass accepted");
            else {
                check(s.pressureFailure[0]==0,"truly empty cell rejected");
                const DeviceState before=s;exerciseWriterGuards(s);
                check(std::memcmp(&s,&before,sizeof(s))==0,"empty cell was modified");
            }
        }
    } else if(mode=="global_failure") {
        DeviceState s;s.nCells=2;s.nFaces=2;s.particleCount=4;
        s.V[1]=s.cellLength[1]=1;s.momRhoP[1]=2;s.momRhoEP[1]=R(3.25);
        s.rhoEs[1]=R(3.25);s.theta[1]=R(3.25/3);s.epsS[1]=2;s.cellParticleCount[1]=2;
        s.cellPlaneStart[1]=1;s.cellPlaneCount[1]=1;s.cellFaceId[1]=1;s.faceOwner[1]=1;s.faceNeighbour[1]=-1;
        s.solidPressurePhiMomX[1]=-1;s.solidPressurePhiMomY[1]=R(-.25);
        s.solidPressurePhiMomZ[1]=R(.5);s.solidPressurePhiEnergy[1]=R(-.2);
        for(int i=2;i<4;++i) {
            s.pStatus[i]=1;s.pCellId[i]=1;s.pm[i]=1;s.pux[i]=i==2?R(-.5):R(.5);s.pTheta[i]=1;
        }
        s.pTheta[3]=std::numeric_limits<R>::quiet_NaN();
        for(int c=0;c<2;++c){blockIdx.x=c;preparePressurePreflightKernel(&s,1);}
        blockIdx.x=0;previewPressureSortedKernel<false,false>(&s);
        check(s.pressureFailure[0]==0,"healthy first cell failed preview");
        blockIdx.x=1;previewPressureSortedKernel<false,false>(&s);
        check((s.pressureFailure[0]&pressureBadParticle)!=0&&s.pressureFailure[1]==2,"second-cell failure did not enter global decision");
        for(int c=0;c<2;++c){blockIdx.x=c;auditPressurePreviewKernel(&s,1);}
        const DeviceState before=s;
        for(int c=0;c<2;++c){blockIdx.x=c;exerciseWriterGuards(s);}
        check(std::memcmp(&s,&before,sizeof(s))==0,"one-cell failure allowed a physical writer in either cell");
        blockIdx.x=0;
    } else if(mode=="actual_floor"||mode=="actual_velocity") {
        DeviceState s;s.momRhoP[0]=1;s.momRhoUPx[0]=0;s.momRhoEP[0]=R(.5);s.thetaMin=0;s.pressureKickFraction=1;
        delta(s,1,0,0,0);preparePressurePreflightKernel(&s,1);
        check(s.pressureFailure[0]==0,"admissible target failed before actual-moment test");
        R* a=s.pressurePreviewMoments;a[0]=1;a[3]=R(.5);a[4]=1;a[5]=2;a[9]=R(.5);
        const R eps=std::numeric_limits<R>::epsilon();
        if(mode=="actual_floor")a[3]-=R(4)*eps;
        else {a[0]+=R(4)*eps;a[3]+=R(8)*eps;}
        // Both discrepancies fit conservation tolerance. They still must fail
        // the stricter floor/dU test on the aggregate actually being published.
        auditPressurePreviewKernel(&s,1);
        assertFailure(s,pressureUnrealizableParticles,"actual aggregate safety check relied only on target/tolerance");
    } else if(mode=="actual_particle_velocity") {
        DeviceState s;s.thetaMin=R(.2);s.pressureKickFraction=0;
        const R offset=R(8)*std::numeric_limits<R>::epsilon();
        s.pux[0]+=offset;delta(s,0,0,0,R(.2));
        preview(s,"sorted");
        check(s.pressurePreviewMoments[0]==0,"unresolved energy-only fixture must have zero final mean velocity");
        check(s.pressurePreviewMoments[6]>0,"fixture must retain nonzero actual initial mean velocity");
        assertFailure(s,pressureUnrealizableParticles,"zero dU cap ignored a velocity change relative to actual initial particles");
    } else if(mode.rfind("invalid_count_",0)==0) {
        const R first=mode=="invalid_count_negative"?R(-1):mode=="invalid_count_precision"?R(16777216):R(2147483648.0);
        const int trials=mode=="invalid_count_precision"?2:1;
        for(int trial=0;trial<trials;++trial) {
            const R count=first+R(2*trial);
            DeviceState s;s.pux[0]=s.pux[1]=0;s.momRhoEP[0]=3;delta(s,0,0,0,9);
            preview(s,"sorted");check(s.pressureFailure[0]==0,"exact count fixture rejected before corruption");
            s.pressurePreviewMoments[5]=count;auditPressurePreviewKernel(&s,1);
            if(sizeof(R)==8&&mode=="invalid_count_precision")
                check(s.pressureFailure[0]==0,"exactly representable double count rejected unnecessarily");
            else assertFailure(s,pressureUnrealizableParticles,"uncertifiable or out-of-range particle count accepted for integer publication");
        }
    } else if(mode=="derived_overflow") {
        DeviceState s;s.rhoSolid=std::numeric_limits<R>::min()/R(4);
        check(finiteDevice(s.rhoSolid)&&s.rhoSolid>0,"derived overflow fixture requires a finite positive rhoSolid");
        check(!finiteDevice(s.momRhoP[0]/s.rhoSolid),"fixture must overflow published epsS");
        preview(s,"sorted");
        assertFailure(s,pressureUnrealizableParticles,"finite accepted moments overflowed a canonical derived field");
    } else if(mode=="tiny_relative_tolerance") {
        const R tiny=sizeof(R)==4?R(1e-40):R(1e-60);
        DeviceState s;s.momRhoP[0]=R(1e-10);s.momRhoEP[0]=tiny;
        s.epsSMin=0;s.thetaMin=0;delta(s,0,0,0,tiny);
        preparePressurePreflightKernel(&s,1);
        check(s.pressureFailure[0]==0,"finite tiny-energy target rejected before closure audit");
        R* a=s.pressurePreviewMoments;
        a[3]=R(3)*tiny;a[4]=s.momRhoP[0];a[5]=1;a[9]=s.momRhoEP[0];
        const R target=s.momRhoEP[0]+s.pressureDeltaEnergy[0];
        check(a[3]>R(1.4)*target,"fixture must exceed relative roundoff by a material fraction");
        auditPressurePreviewKernel(&s,1);
        assertFailure(s,pressureUnrealizableParticles,"absolute tolerance floor accepted a material tiny-energy conservation error");
    } else if(mode=="canonical") {
        DeviceState s;preview(s,"sorted");check(s.pressureFailure[0]==0,"canonical fixture rejected");
        const R eps=std::numeric_limits<R>::epsilon();
        s.pressurePreviewMoments[0]+=R(2)*eps;s.pressurePreviewMoments[3]+=R(2)*eps;
        auditPressurePreviewKernel(&s,1);check(s.pressureFailure[0]==0,"roundoff-sized realized closure rejected");
        publishPressureCanonicalMomentsKernel(&s);
        check(s.momRhoUPx[0]==s.pressurePreviewMoments[0],"published requested momentum instead of accepted actual aggregate");
        check(s.momRhoEP[0]==s.pressurePreviewMoments[3],"published requested energy instead of accepted actual aggregate");
        check(s.rhoUsx[0]==s.momRhoUPx[0]&&s.rhoEs[0]==s.momRhoEP[0],"canonical diagnostic moments disagree");
    } else if(mode=="zero_identity") {
        for(const std::string route:{"sorted","compact","split","unsorted"}) {
            DeviceState s;delta(s,0,0,0,0);
            // Below-floor unresolved particles and a constrained record would
            // be altered by projection if either identity gate were missing.
            s.thetaMin=10;s.pressureKickFraction=0;s.pStuck[1]=1;s.pTheta[1]=0;
            s.momRhoEP[0]=s.rhoEs[0]=R(1.75);s.theta[0]=R(1.75/3);compactCopy(s);
            preview(s,route);check(s.pressureFailure[0]==0,"finite zero delta was not an identity");
            const DeviceState before=s;exerciseWriterGuards(s);
            check(std::memcmp(&s,&before,sizeof(s))==0,"zero delta changed state bit patterns");
            check(s.pux[0]==R(-.5)&&s.pux[1]==R(.5)&&s.pTheta[0]==1,"zero preflight mutated particles");
        }
    } else if(mode=="unsorted_recovery") {
        DeviceState s;s.momRhoP[0]=1;s.momRhoUPx[0]=R(.1);s.momRhoEP[0]=1;
        s.pressureDeltaMomX[0]=1;s.pressureDeltaEnergy[0]=R(.2);
        const auto l0=pressurePreviewParameters(s,0,true),l1=pressurePreviewParameters(s,0,false);
        check(l0.ux0==(s.momRhoUPx[0]+s.pressureDeltaMomX[0])-s.pressureDeltaMomX[0],"L0 must reconstruct q0 from rounded q1-delta");
        check(l1.ux0==s.momRhoUPx[0],"L1 must use original q0");
        check(l0.ux0!=l1.ux0,"rounding-sensitive L0/L1 fixture did not distinguish arithmetic");
        const auto recovered=recoverUnsortedPressureKinematics(s.momRhoP[0],l0.px1,l0.py1,l0.pz1,l0.e1,s.pressureDeltaMomX[0],s.pressureDeltaMomY[0],s.pressureDeltaMomZ[0],s.pressureDeltaEnergy[0],s.thetaMin);
        check(l0.resolved==recovered.resolved&&l0.thermalScale==recovered.thermalScale&&l0.thetaScale==recovered.thetaScale,"L0 preview diverged from production published-state reconstruction");
    } else if(mode=="invalid_status") {
        DeviceState s;s.pStatus[1]=2;preview(s,"sorted");
        assertFailure(s,pressureBadParticle,"nonzero unaccounted particle status accepted");
    } else if(mode=="invalid_directory") {
        for(int bad:{-1,4}){DeviceState s;s.sortedParticleIndex[1]=bad;preview(s,"sorted");assertFailure(s,pressureBadParticle,"invalid sorted particle index accepted");}
        DeviceState s;s.pCellId[1]=1;preview(s,"sorted");assertFailure(s,pressureBadParticle,"foreign-cell sorted particle accepted");
    } else if(mode=="limiter") {
        DeviceState s;s.momRhoP[0]=1;s.momRhoEP[0]=1;s.pressureKickFraction=100;delta(s,2,0,0,0);
        const R beta=pressureLocalConvexScale(s,0,1);check(beta>R(.70)&&beta<R(.71),"real local limiter did not constrain pressure face");
        s.pressureKickScale[0]=beta;scaleCollisionalPressureFaceFluxKernel(&s);preparePressurePreflightKernel(&s,1);
        check(s.pressureFailure[0]==0,"limited production face failed target preflight");
        for(int field=0;field<4;++field){DeviceState bad;R* fields[]={bad.solidPressurePhiMomX,bad.solidPressurePhiMomY,bad.solidPressurePhiMomZ,bad.solidPressurePhiEnergy};*fields[field]=std::numeric_limits<R>::quiet_NaN();pressureLocalConvexScale(bad,0,1);assertFailure(bad,pressureBadFace,"nonfinite raw face was sanitized");}
    } else check(false,"unknown test mode");
}
