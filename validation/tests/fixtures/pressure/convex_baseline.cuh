// Frozen numerical reference from 93a7a35cc18b8b695e1e29a9332a158b9847d16c.
// Deliberately independent of the prepared-invariant implementation under test.
#pragma once
// Scalar algebra shared by the CUDA limiter and executable host regressions.
// The caller supplies one face delta divided by its positive convex weight.
#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define PRESSURE_CONVEX_HD __host__ __device__ __forceinline__
#else
#define PRESSURE_CONVEX_HD inline
#endif

namespace pressure_convex_baseline {
template<class Real> struct LimitResult { Real beta; bool valid; };
template<class Real> struct InitialState { Real internal; Real floor; bool valid; };

PRESSURE_CONVEX_HD float root(float x) { return ::sqrtf(x); }
PRESSURE_CONVEX_HD double root(double x) { return ::sqrt(x); }
PRESSURE_CONVEX_HD float fraction(float x,int* e) { return ::frexpf(x,e); }
PRESSURE_CONVEX_HD double fraction(double x,int* e) { return ::frexp(x,e); }
PRESSURE_CONVEX_HD float power(float x,int e) { return ::ldexpf(x,e); }
PRESSURE_CONVEX_HD double power(double x,int e) { return ::ldexp(x,e); }
PRESSURE_CONVEX_HD float fused(float x,float y,float z) { return ::fmaf(x,y,z); }
PRESSURE_CONVEX_HD double fused(double x,double y,double z) { return ::fma(x,y,z); }
PRESSURE_CONVEX_HD float inward(float x) { return ::nextafterf(x,0.0f); }
PRESSURE_CONVEX_HD double inward(double x) { return ::nextafter(x,0.0); }
template<class Real> PRESSURE_CONVEX_HD Real maximum(Real a,Real b) { return a>b?a:b; }
template<class Real> PRESSURE_CONVEX_HD Real minimum(Real a,Real b) { return a<b?a:b; }
template<class Real> PRESSURE_CONVEX_HD Real absolute(Real a) { return a<Real(0)?-a:a; }
template<class Real> PRESSURE_CONVEX_HD Real largest() { return sizeof(Real)==sizeof(float)?Real(FLT_MAX):Real(DBL_MAX); }
template<class Real> PRESSURE_CONVEX_HD Real epsilon() { return sizeof(Real)==sizeof(float)?Real(FLT_EPSILON):Real(DBL_EPSILON); }
template<class Real> PRESSURE_CONVEX_HD bool finite(Real a) { return a==a && a<=largest<Real>() && a>=-largest<Real>(); }
template<class Real> PRESSURE_CONVEX_HD Real maxComponent(Real x,Real y,Real z) {
    return maximum(absolute(x),maximum(absolute(y),absolute(z)));
}

// Scale before squaring: a normal nonzero component must not disappear when
// its square underflows, and a representable norm must not overflow early.
template<class Real> PRESSURE_CONVEX_HD Real norm3(Real x,Real y,Real z) {
    if(!finite(x)||!finite(y)||!finite(z)) return root(Real(-1));
    const Real m=maxComponent(x,y,z);
    if(m==Real(0)) return Real(0);
    x/=m;y/=m;z/=m;
    return m*root(fused(x,x,fused(y,y,z*z)));
}

// sqrt(2*mass*energy), without forming the possibly overflowing/underflowing
// product. Mantissas stay in [0.5,4); only the final result is rescaled.
template<class Real> PRESSURE_CONVEX_HD Real momentumClosureScale(Real mass,Real energy) {
    if(!finite(mass)||!finite(energy)||mass<Real(0)||energy<Real(0)) return root(Real(-1));
    if(mass==Real(0)||energy==Real(0)) return Real(0);
    int me=0,ee=0;
    const Real mm=fraction(mass,&me),em=fraction(energy,&ee);
    int e=me+ee;
    Real product=Real(2)*mm*em;
    if(e%2!=0) { product*=Real(2); --e; }
    return power(root(product),e/2);
}

// Squaring the raw momentum or multiplying rho by energy may overflow even
// when the final kinetic energy is finite. Keep their binary exponents apart.
template<class Real> PRESSURE_CONVEX_HD Real kinetic(Real rho,Real px,Real py,Real pz) {
    const Real m=maxComponent(px,py,pz);
    if(m==Real(0)) return Real(0);
    const Real x=px/m,y=py/m,z=pz/m;
    int me=0,re=0;
    const Real mm=fraction(m,&me),rm=fraction(rho,&re);
    const Real norm=fused(x,x,fused(y,y,z*z));
    return power((Real(.5)*mm*mm/rm)*norm,2*me-re);
}

template<class Real> PRESSURE_CONVEX_HD InitialState<Real> initialState
(Real rho,Real px,Real py,Real pz,Real energy,Real configuredFloor) {
    if(!finite(rho) || rho<=Real(0) || !finite(px) || !finite(py) || !finite(pz)
       || !finite(energy) || energy<Real(0) || !finite(configuredFloor) || configuredFloor<Real(0))
        return {Real(0),Real(0),false};
    const Real k=kinetic(rho,px,py,pz);
    const Real internal=energy-k;
    if(!finite(k) || !finite(internal) || internal<Real(0))
        return {Real(0),Real(0),false};
    // A valid cold state stays valid without adding energy to meet thetaMin.
    return {internal,minimum(configuredFloor,internal),true};
}

// min(1,a*b/c), with positive finite arguments, without intermediate overflow.
template<class Real> PRESSURE_CONVEX_HD Real boundedProductRatio(Real a,Real b,Real c) {
    if(a==Real(0) || b==Real(0)) return Real(0);
    int ae=0,be=0,ce=0;
    Real am=fraction(a,&ae),bm=fraction(b,&be),cm=fraction(c,&ce);
    const int e=ae+be-ce;
    if(e>3) return Real(1);
    return minimum(Real(1),power(am*bm/cm,e));
}

// Signed p/sqrt(rho*h), avoiding formation of either rho*h or its reciprocal.
template<class Real> PRESSURE_CONVEX_HD Real normalizedMomentum(Real p,Real rho,Real h) {
    if(p==Real(0)) return Real(0);
    int pe=0,re=0,he=0;
    Real pm=fraction(p,&pe),rm=fraction(rho,&re),hm=fraction(h,&he);
    int e=re+he;
    Real product=rm*hm;
    if(e%2!=0) { product*=Real(2); --e; }
    return power(pm/root(product),pe-e/2);
}

// A feasible endpoint has |delta P| <= (sqrt(2)+2)*sqrt(rho*h),
// h=max(E,|delta E|). Four is a loose upper bound, not a new physical limit.
// This cap makes the subsequent quadratic coefficients safely representable.
template<class Real> PRESSURE_CONVEX_HD Real momentumRangeCap(Real rho,Real h,Real m) {
    if(m==Real(0)) return Real(1);
    int re=0,he=0,me=0;
    Real rm=fraction(rho,&re),hm=fraction(h,&he),mm=fraction(m,&me);
    int e=re+he;
    Real product=rm*hm;
    if(e%2!=0) { product*=Real(2); --e; }
    const int out=e/2-me;
    if(out>3) return Real(1);
    return minimum(Real(1),power(Real(4)*root(product)/mm,out));
}

template<class Real> PRESSURE_CONVEX_HD bool admissibleIncrement
(Real rho,Real px,Real py,Real pz,Real energy,Real floor,Real maxDeltaU,
 Real dpx,Real dpy,Real dpz,Real dEnergy) {
    if(!finite(rho) || rho<=Real(0) || !finite(px) || !finite(py) || !finite(pz)
       || !finite(energy) || !finite(floor) || floor<Real(0)
       || !finite(maxDeltaU) || maxDeltaU<Real(0)
       || !finite(dpx) || !finite(dpy) || !finite(dpz) || !finite(dEnergy)) return false;
    const Real m=maxComponent(dpx,dpy,dpz);
    if(m>Real(0)) {
        if(maxDeltaU==Real(0)) return false;
        const Real x=dpx/m,y=dpy/m,z=dpz/m;
        const Real norm=root(fused(x,x,fused(y,y,z*z)));
        if(boundedProductRatio(rho,maxDeltaU/norm,m)<Real(1)) return false;
    }
    const Real x=px+dpx,y=py+dpy,z=pz+dpz,e=energy+dEnergy;
    if(!finite(x) || !finite(y) || !finite(z) || !finite(e)) return false;
    const Real k=kinetic(rho,x,y,z);
    return finite(k) && e>=floor && e-k>=floor;
}

template<class Real> PRESSURE_CONVEX_HD LimitResult<Real> limitFace
(
    Real rho,Real px,Real py,Real pz,Real energy,Real configuredFloor,Real maxDeltaU,
    Real dpx,Real dpy,Real dpz,Real dEnergy
) {
    const InitialState<Real> start=initialState(rho,px,py,pz,energy,configuredFloor);
    if(!start.valid || !finite(maxDeltaU) || maxDeltaU<Real(0)
       || !finite(dpx) || !finite(dpy) || !finite(dpz) || !finite(dEnergy))
        return {Real(0),false};
    const Real m=maxComponent(dpx,dpy,dpz);
    if(m==Real(0) && dEnergy==Real(0)) return {Real(1),true};

    Real cap=Real(1);
    if(m>Real(0)) {
        const Real x=dpx/m,y=dpy/m,z=dpz/m;
        const Real norm=root(fused(x,x,fused(y,y,z*z)));
        cap=boundedProductRatio(rho,maxDeltaU/norm,m);
    }
    const Real h=maximum(energy,absolute(dEnergy));
    if(h==Real(0) || cap==Real(0)) return {Real(0),true};
    cap=minimum(cap,momentumRangeCap(rho,h,m));
    if(dEnergy>Real(0)) cap=minimum(cap,(largest<Real>()-energy)/dEnergy);
    if(cap==Real(0)) return {Real(0),true};

    const Real x=normalizedMomentum(px,rho,h);
    const Real y=normalizedMomentum(py,rho,h);
    const Real z=normalizedMomentum(pz,rho,h);
    const Real dx=normalizedMomentum(cap*dpx,rho,h);
    const Real dy=normalizedMomentum(cap*dpy,rho,h);
    const Real dz=normalizedMomentum(cap*dpz,rho,h);
    const Real a=Real(.5)*fused(dx,dx,fused(dy,dy,dz*dz));
    const Real b=fused(cap,dEnergy/h,-fused(x,dx,fused(y,dy,z*dz)));
    const Real c=(start.internal-start.floor)/h;
    if(!finite(a) || !finite(b) || !finite(c)) return {Real(0),false};
    Real t=Real(1);
    if(fused(-a,t,fused(b,t,c))<Real(0)) {
        if(a==Real(0)) {
            t=b<Real(0)?c/(-b):Real(0);
        } else {
            const Real disc=root(fused(b,b,Real(4)*a*c));
            // The negative-b branch avoids subtracting nearly equal numbers.
            t=b>=Real(0)?(b+disc)/(Real(2)*a):(Real(2)*c)/(disc-b);
        }
        t=minimum(Real(1),maximum(Real(0),t));
    }
    Real beta=cap*t;
    // Never erode a fully admissible identity endpoint. For limited endpoints,
    // reserve a few ulps for the shared-face and final-cell arithmetic.
    if(beta<Real(1) && beta>Real(0))
        // A relative margin alone can round to the same subnormal value.
        beta=inward(beta*(Real(1)-Real(64)*epsilon<Real>()));
    if(!finite(beta)) return {Real(0),false};
    if(admissibleIncrement(rho,px,py,pz,energy,start.floor,maxDeltaU,
                           beta*dpx,beta*dpy,beta*dpz,beta*dEnergy))
        return {beta,true};
    // No per-face iterative solve. An uncertifiable rounded endpoint fails
    // closed; the unchanged, previously validated initial state is feasible.
    return {Real(0),true};
}
} // namespace pressure_convex_baseline
#undef PRESSURE_CONVEX_HD

// Each face is one fixed-weight convex substate (weight=1/faceCount).
// A neighbouring cell may only shorten that substate's safe segment.
// Invalid local inputs retain the original zero-kick fallback; the existing
// projection and closure still run their finite-value and nonnegative clamps.
__device__ inline PressureReal baselinePressureLocalConvexScale(DeviceState& s,int c,PressureTime dt)
{
    const PressureReal rho=s.momRhoP[c],px=s.momRhoUPx[c],py=s.momRhoUPy[c],pz=s.momRhoUPz[c],e=s.momRhoEP[c];
    if(!finiteDevice(rho)||rho<0||!finiteDevice(px)||!finiteDevice(py)||!finiteDevice(pz)||!finiteDevice(e)||e<0
       ||!finiteDevice(s.V[c])||s.V[c]<=0||!finiteDevice(s.cellLength[c])||s.cellLength[c]<=0
       ||!finiteDevice(s.thetaMin)||s.thetaMin<0||!finiteDevice(s.pressureKickFraction)||s.pressureKickFraction<0
       ||!finiteDevice(s.epsSMin)||s.epsSMin<0||!finiteDevice(s.rhoSolid)||s.rhoSolid<=0)
    {return 0;}
    if(rho==0)
    {
        return 0;
    }
    const PressureReal configuredFloor=PressureReal(1.5)*rho*s.thetaMin;
    const auto initial=pressure_convex_baseline::initialState(rho,px,py,pz,e,configuredFloor);
    if(!initial.valid){return 0;}
    if(rho<=s.epsSMin*s.rhoSolid)return 0;
    const int count=s.cellPlaneCount[c],start=s.cellPlaneStart[c];
    if(count<0||start<0){return 0;}
    const PressureReal maxDU=s.pressureKickFraction*s.cellLength[c]/dt;
    const PressureReal factor=dt/s.V[c];
    if(!finiteDevice(maxDU)||!finiteDevice(factor)){return 0;}
    PressureReal lambda=1;
    for(int j=0;j<count;++j)
    {
        const int f=s.cellFaceId[start+j];
        if(f<0||f>=s.nFaces||(s.faceOwner[f]!=c&&s.faceNeighbour[f]!=c))
        {return 0;}
        const PressureReal sign=s.faceOwner[f]==c?PressureReal(1):PressureReal(-1);
        const PressureReal weightInverse=PressureReal(count);
        const PressureReal dx=-factor*sign*s.solidPressurePhiMomX[f]*weightInverse;
        const PressureReal dy=-factor*sign*s.solidPressurePhiMomY[f]*weightInverse;
        const PressureReal dz=-factor*sign*s.solidPressurePhiMomZ[f]*weightInverse;
        const PressureReal de=-factor*sign*s.solidPressurePhiEnergy[f]*weightInverse;
        const auto face=pressure_convex_baseline::limitFace(rho,px,py,pz,e,configuredFloor,maxDU,dx,dy,dz,de);
        if(!face.valid){return 0;}
        lambda=fmin(lambda,face.beta);
    }
    return lambda;
}
