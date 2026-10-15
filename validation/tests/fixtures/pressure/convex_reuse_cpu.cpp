// Execute the real production cell limiter with a host-only DeviceState scaffold.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#define __device__
#define __forceinline__ inline
using PressureReal=TEST_REAL;
using PressureTime=double;
using R=PressureReal;
using std::fmin;
using std::sqrt;
struct Dim { int x; };
Dim blockIdx{0},blockDim{1},threadIdx{0};
template<class T> T clampMin(T x,T y) { return std::max(x,y); }
template<class T> bool finiteDevice(T x) { return std::isfinite(x); }
template<class T> T finiteOr(T x,T fallback) { return finiteDevice(x)?x:fallback; }
constexpr R OfVSmall=R(1e-30);
struct DeviceState {
    int nCells=1,nFaces=6;
    R momRhoP[1]={2},momRhoUPx[1]={R(.25)},momRhoUPy[1]={R(.5)},momRhoUPz[1]={R(.75)},momRhoEP[1]={8};
    R V[1]={1},cellLength[1]={1},thetaMin=R(.01),pressureKickFraction=10,epsSMin=0,rhoSolid=1;
    int cellPlaneCount[1]={6},cellPlaneStart[1]={0},cellFaceId[6]={0,1,2,3,4,5};
    int faceOwner[6]={0,0,0,0,0,0},faceNeighbour[6]={-1,-1,-1,-1,-1,-1};
    R solidPressurePhiMomX[6],solidPressurePhiMomY[6],solidPressurePhiMomZ[6],solidPressurePhiEnergy[6];
    R pressureDeltaMomX[1],pressureDeltaMomY[1],pressureDeltaMomZ[1],pressureDeltaEnergy[1],pressureKickScale[1];
    R rhoUsx[1],rhoUsy[1],rhoUsz[1],rhoEs[1],Usx[1],Usy[1],Usz[1],theta[1];
    DeviceState() {
        for(int i=0;i<6;++i) {
            solidPressurePhiMomX[i]=R(.001);solidPressurePhiMomY[i]=R(.002);
            solidPressurePhiMomZ[i]=R(.003);solidPressurePhiEnergy[i]=R(.004);
        }
    }
};
int initialCalls=0,fractionCalls=0,rootCalls=0;
#include "GpuPressureKickAccumulation.cuh"
#include "convex_baseline.cuh"
void check(bool ok,const char* message) {
    if(!ok) { std::cerr<<message<<'\n';std::exit(1); }
}
bool same(R a,R b) { return std::memcmp(&a,&b,sizeof(R))==0; }
struct Face {
    R rho,px,py,pz,energy,floor,du,dx,dy,dz,de;
};
void compare(const Face& f) {
    const auto before=pressure_convex_baseline::limitFace(f.rho,f.px,f.py,f.pz,f.energy,f.floor,f.du,f.dx,f.dy,f.dz,f.de);
    const auto after=pressure_convex::limitFace(f.rho,f.px,f.py,f.pz,f.energy,f.floor,f.du,f.dx,f.dy,f.dz,f.de);
    check(before.valid==after.valid&&same(before.beta,after.beta),"public face limiter differs bitwise from frozen baseline");
    const auto initial=pressure_convex::initialState(f.rho,f.px,f.py,f.pz,f.energy,f.floor);
    const auto prepared=pressure_convex::limitFacePrepared(f.rho,f.px,f.py,f.pz,f.energy,initial,f.du,f.dx,f.dy,f.dz,f.de);
    check(before.valid==prepared.valid&&same(before.beta,prepared.beta),"prepared face limiter differs bitwise from frozen baseline");
}
R exponentSample(std::mt19937_64& gen) {
    const int low=std::numeric_limits<R>::min_exponent-std::numeric_limits<R>::digits;
    const int high=std::numeric_limits<R>::max_exponent-4;
    const int exponent=low+int(gen()%unsigned(high-low+1));
    return std::ldexp(R(1),exponent);
}
int main(int argc,char** argv) {
    check(argc==2,"mode required"); const std::string mode=argv[1];
    if(mode=="counts") {
        DeviceState s;
        const R beta=pressureLocalConvexScale(s,0,.125);
        std::cout<<"beta="<<beta<<" initial="<<initialCalls<<" fraction="<<fractionCalls<<" root="<<rootCalls<<'\n';
        check(beta==1,"counter fixture must take six nonzero, fully admissible faces");
        check(initialCalls==1,"initial cell state must be evaluated once, not once per face");
        check(fractionCalls==2+17*6,"rho/h decomposition must be shared by all face components and the range cap");
        check(rootCalls==3*6,"normalization denominator square root must be evaluated once per face");
    } else if(mode=="counts_fast_paths") {
        DeviceState s;
        for(int f=0;f<6;++f) s.solidPressurePhiMomX[f]=s.solidPressurePhiMomY[f]=s.solidPressurePhiMomZ[f]=s.solidPressurePhiEnergy[f]=0;
        check(pressureLocalConvexScale(s,0,.125)==1,"zero faces must preserve the identity");
        check(initialCalls==1&&fractionCalls==2&&rootCalls==0,"zero faces must skip normalization work");
        initialCalls=fractionCalls=rootCalls=0;
        s.momRhoUPx[0]=s.momRhoUPy[0]=s.momRhoUPz[0]=s.pressureKickFraction=0;
        for(int f=0;f<6;++f)s.solidPressurePhiEnergy[f]=R(.004);
        check(pressureLocalConvexScale(s,0,.125)==1,"zero momentum energy-only faces must remain admissible at zero dU");
        check(initialCalls==1&&fractionCalls==0&&rootCalls==0,"energy-only zero-momentum faces must skip normalization work");
    } else if(mode=="equivalence") {
        std::mt19937_64 gen(6149651);
        std::uniform_real_distribution<double> unit(0,1);
        for(int i=0;i<60000;++i) {
            const R scale=exponentSample(gen);
            Face f{R(2)*scale,R(.25)*scale,R(.5)*scale,R(.75)*scale,R(4)*scale,R(.01)*scale,R(10*unit(gen)),
                R(16*unit(gen)-8)*scale,R(16*unit(gen)-8)*scale,R(16*unit(gen)-8)*scale,R(16*unit(gen)-8)*scale};
            compare(f);
            // Independent density/energy exponents exercise rho*h overflow and
            // underflow even though each input and the initial state is valid.
            f.rho=exponentSample(gen);f.energy=exponentSample(gen);f.floor=0;
            f.px=f.py=f.pz=0;compare(f);
        }
        const R tiny=std::numeric_limits<R>::denorm_min(), small=std::numeric_limits<R>::min();
        const R huge=std::numeric_limits<R>::max();
        const std::vector<R> values={tiny,tiny*R(123),small,std::nextafter(small,R(0)),R(.125),R(.5),R(1),R(2),huge/R(8),huge};
        for(R rho:values)for(R h:values)for(R p:values)for(R sign:{R(-1),R(1)}) {
            const R before=pressure_convex_baseline::normalizedMomentum(sign*p,rho,h);
            const R after=pressure_convex::normalizedMomentum(sign*p,rho,h);
            check(same(before,after),"normalization changed division/exponent rounding");
            check(same(pressure_convex_baseline::momentumRangeCap(rho,h,p),pressure_convex::momentumRangeCap(rho,h,p)),"range cap differs bitwise");
        }
        for(R rho:values)for(R energy:values)for(R delta:values) {
            compare({rho,0,0,0,energy,0,1,delta,0,0,0});
            compare({rho,0,0,0,energy,0,1,0,0,0,-delta});
        }
        // ULP neighbours of the exact kinetic floor, the identity endpoint,
        // velocity cap and cancellation-safe quadratic/linear boundaries.
        for(R p:{R(0),R(1),R(-1)})for(R dp:{R(-2),R(-1),R(0),R(1),R(2)})
        for(R e:{std::nextafter(R(.5),R(0)),R(.5),std::nextafter(R(.5),R(1)),R(1)})
        for(R de:{R(0),R(-.5),R(-1),R(-1000000),tiny})
        for(R du:{R(0),std::nextafter(R(1),R(0)),R(1),std::nextafter(R(1),R(2)),huge})
            compare({1,p,0,0,e,0,du,dp,0,0,de});
        for(int field=0;field<11;++field)for(R bad:{-huge,-R(1),R(0),std::numeric_limits<R>::infinity(),-std::numeric_limits<R>::infinity(),std::numeric_limits<R>::quiet_NaN()}) {
            R x[11]={1,0,0,0,1,0,1,0,0,0,0};x[field]=bad;
            compare({x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10]});
        }
        std::cout<<"PASS bitwise random/extreme/ULP/invalid face and normalization reference comparisons\n";
    } else if(mode=="cells") {
        std::mt19937_64 gen(875613);std::uniform_real_distribution<double> unit(0,1);
        for(int i=0;i<12000;++i) {
            DeviceState s; const R scale=exponentSample(gen);
            s.momRhoP[0]=R(2)*scale;s.momRhoEP[0]=R(4)*scale;
            s.momRhoUPx[0]=R(.25)*scale;s.momRhoUPy[0]=R(.5)*scale;s.momRhoUPz[0]=R(.75)*scale;
            s.pressureKickFraction=R(10*unit(gen));
            for(int f=0;f<6;++f) {
                s.solidPressurePhiMomX[f]=R(16*unit(gen)-8)*scale;
                s.solidPressurePhiMomY[f]=R(16*unit(gen)-8)*scale;
                s.solidPressurePhiMomZ[f]=R(16*unit(gen)-8)*scale;
                s.solidPressurePhiEnergy[f]=R(16*unit(gen)-8)*scale;
                if(f%2) {s.faceOwner[f]=1;s.faceNeighbour[f]=0;}
            }
            const R before=baselinePressureLocalConvexScale(s,0,.125),after=pressureLocalConvexScale(s,0,.125);
            check(same(before,after),"production prepared cell limiter differs bitwise from baseline");
            // Both cells shorten a face by the shared minimum. Compare every
            // retained contribution, including opposite owner orientations.
            for(int f=0;f<6;++f) {
                const R neighbour=R(unit(gen));
                check(same(std::min(before,neighbour)*s.solidPressurePhiMomX[f],std::min(after,neighbour)*s.solidPressurePhiMomX[f]),"neighbour-minimum face contribution differs");
            }
        }
        DeviceState s;s.solidPressurePhiMomX[0]=R(10000);s.pressureKickFraction=0;
        for(int f=0;f<6;++f) {
            const R old=s.solidPressurePhiEnergy[f];s.solidPressurePhiEnergy[f]=std::numeric_limits<R>::quiet_NaN();
            check(same(baselinePressureLocalConvexScale(s,0,1),pressureLocalConvexScale(s,0,1)),"late invalid face was skipped after zero scale");
            s.solidPressurePhiEnergy[f]=old;
        }
        std::cout<<"PASS bitwise actual cell loop, neighbour minima and late invalid faces\n";
    }
}
