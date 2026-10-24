#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall { namespace detail {
// Quadratic-exact derivative at a nonuniform interior node.
template<class R> UGKWP_GAS_HD inline R centeredDerivative(R left,R center,R right,R hLeft,R hRight){
    return hRight*(center-left)/(hLeft*(hLeft+hRight))+hLeft*(right-center)/(hRight*(hLeft+hRight));
}
// Exact two-point conductance for a coefficient linear in physical distance:
// C=(integral_0^1 ds/a(s))^-1. Coincident/zero endpoints use their exact limits.
// Signed same-sign coefficients retain their mathematical sign; crossing zero
// has infinite resistance and is unavailable, rather than an alternate flux.
template<class R> UGKWP_GAS_HD inline R transportConductance(R a,R b){
    if(a==b)return a;
    if(a==0||b==0)return R(0);
    const R sign=a<0&&b<0?R(-1):R(1);
    a*=sign;b*=sign;
    if(!(a>0&&b>0))return std::numeric_limits<R>::quiet_NaN();
    const R low=min(a,b),high=max(a,b),difference=high-low;
    const bool ratioOverflows=low<R(1)&&difference>std::numeric_limits<R>::max()*low;
    const R logarithm=ratioOverflows?(::log(high)-::log(low)): ::log1p(difference/low);
    return sign*difference/logarithm;
}
// Integrate u/B in the momentum resistance coordinate t=log(mu/mu_a).
// Two Gauss points are exact when B is proportional to mu (including the
// constant-coefficient limit). This is a quadrature order, not a tuned switch.
template<class R> UGKWP_GAS_HD inline R thermalWorkVelocityWeight(R muA,R muB,R bA,R bB){
    if(!(muA>0&&muB>0&&bA>0&&bB>0))return R(.5);
    // Orient the resistance coordinate increasingly: log1p(-1) can otherwise
    // occur by rounding even though both endpoints are strictly positive.
    const bool reversed=muB<muA;
    if(reversed){R swap=muA;muA=muB;muB=swap;swap=bA;bA=bB;bB=swap;}
    const R difference=muB-muA;
    if(difference==0)return R(.5);
    const bool wide=muA<R(1)&&difference>std::numeric_limits<R>::max()*muA;
    const R logarithm=wide?(::log(muB)-::log(muA)): ::log1p(difference/muA);
    const R g=R(.2886751345948128822545743902509787),t[2]={R(.5)-g,R(.5)+g};
    R weights[2],logWeights[2];bool useLogs=wide;
    for(int j=0;j<2;++j){
        if(wide){
            const R logMu=::log(muA)+t[j]*logarithm,mu=::exp(logMu);
            // Compute both parts of the positive linear B interpolation in
            // log space. Its position may underflow while B*position does not.
            const R logPosition=logMu-::log(muB)+::log1p(-muA/mu)-::log1p(-muA/muB);
            const R position=::exp(logPosition);
            const R left=::log(bA)+::log1p(-position),right=::log(bB)+logPosition;
            const R largest=max(left,right);
            const R logB=largest+::log(::exp(left-largest)+::exp(right-largest));
            logWeights[j]=logMu-logB;
        }else{
            const R relative=::expm1(t[j]*logarithm);
            const R position=muA*relative/difference,mu=muA*(R(1)+relative);
            const R b=(R(1)-position)*bA+position*bB;
            weights[j]=mu/b;
            if(!(weights[j]>0&&weights[j]<=std::numeric_limits<R>::max())){
                logWeights[j]=::log(mu)-::log(b);useLogs=true;}
        }
    }
    if(useLogs){
        if(!wide)for(int j=0;j<2;++j)
            if(weights[j]>0&&weights[j]<=std::numeric_limits<R>::max())logWeights[j]=::log(weights[j]);
        const R largest=max(logWeights[0],logWeights[1]);
        for(int j=0;j<2;++j)weights[j]=::exp(logWeights[j]-largest);}
    else {const R largest=max(weights[0],weights[1]);
        for(int j=0;j<2;++j)weights[j]/=largest;}
    const R result=(weights[0]*t[0]+weights[1]*t[1])/(weights[0]+weights[1]);
    return reversed?R(1)-result:result;
}

// Positive power reconstruction in physical distance. Unlike a fixed y^-2
// fit this reproduces both the viscous omega~y^-2 and log-layer omega~y^-1
// laws, as well as linear data. The ordinary secant is the continuous choice
// when either endpoint is zero. No nonlinear candidate selection is involved.
template<class R> UGKWP_GAS_HD inline R powerValue(R a,R b,R ya,R yb,R y){
    if(!(a>0&&b>0&&ya>0))return a+(b-a)*(y-ya)/(yb-ya);
    const R exponent=::log(b/a)/::log(yb/ya);
    return a*::exp(exponent*::log(y/ya));
}
template<class R> UGKWP_GAS_HD inline R powerGradient(R a,R b,R ya,R yb,R y){
    if(!(a>0&&b>0&&ya>0))return (b-a)/(yb-ya);
    const R exponent=::log(b/a)/::log(yb/ya);
    return exponent*powerValue(a,b,ya,yb,y)/y;
}
template<class R> UGKWP_GAS_HD inline R powerIntegral(R a,R b,R ya,R yb,R lo,R hi){
    const R exponent=::log(b/a)/::log(yb/ya),e=exponent+R(1);
    const R value=powerValue(a,b,ya,yb,lo),logarithm=::log(hi/lo);
    return value*lo*(e==R(0)?logarithm:(::expm1(e*logarithm)/e));
}
template<class R> UGKWP_GAS_HD inline R omegaSquareAverage(R a,R b,R c,R ya,R yb,R yc){
    const R lo=(ya+yb)/R(2),hi=(yb+yc)/R(2);
    return (powerIntegral(a*a,b*b,ya,yb,lo,yb)+powerIntegral(b*b,c*c,yb,yc,yb,hi))/(hi-lo);
}
// Positive, fixed-geometry fitting. Exact for omega=A/y^2 and second-order
// consistent at fixed positive y. This is not a linear-exact face gradient.
template<class R> UGKWP_GAS_HD inline R omegaGradientFit(R left,R right){
    if(left<=0||right<=0)return R(1);
    const R mid=(left+right)/R(2),ratio=left*right/(mid*mid);return ratio*ratio;
}
// On positive-y intervals use omega_h = c0 + c1/y^2. Its midpoint
// derivative is omegaGradientFit(ya,yb)*(b-a)/(yb-ya): the coefficient
// is positive and independent of the states, even across a steep drop.
// These positive interpolation weights avoid subtracting large c0/c1 terms.
template<class R> UGKWP_GAS_HD inline R omegaInverseSquareValue(R a,R b,R ya,R yb,R y){
    if(y==ya||a==b)return a;
    if(y==yb)return b;
    const R wa=(ya/y)*(ya/y)*((yb-y)/(yb-ya))*((yb+y)/(yb+ya));
    const R wb=(yb/y)*(yb/y)*((y-ya)/(yb-ya))*((y+ya)/(yb+ya));
    return wa*a+wb*b;
}
// Exact integral of omega_h^2 on [lo,hi], written with positive endpoint
// weights. q<=1 avoids large ratios and cancellation in nearly flat data.
template<class R> UGKWP_GAS_HD inline R omegaInverseSquareIntegral(R a,R b,R ya,R yb,R lo,R hi){
    if(hi==lo)return R(0);
    const R left=omegaInverseSquareValue(a,b,ya,yb,lo);
    const R right=omegaInverseSquareValue(a,b,ya,yb,hi),q=lo/hi;
    const R mean=(q*(R(1)+R(3)*q)*left*left+R(4)*q*left*right
        +(R(3)+q)*right*right)/(R(3)*(R(1)+q)*(R(1)+q));
    return (hi-lo)*mean;
}
template<class R> UGKWP_GAS_HD inline R omegaInverseSquareAverage(R a,R b,R c,R ya,R yb,R yc){
    const R lo=(ya+yb)/R(2),hi=(yb+yc)/R(2);
    return (omegaInverseSquareIntegral(a,b,ya,yb,lo,yb)
        +omegaInverseSquareIntegral(b,c,yb,yc,yb,hi))/(hi-lo);
}
template<class R> UGKWP_GAS_HD inline R omegaDestructionFit(R y,R faceLeft,R faceRight){
    const R a=y/faceLeft,b=y/faceRight;
    return a*a*b*b*(faceRight/faceLeft+R(1)+faceLeft/faceRight)/R(3);
}
// Leading near-wall solution of mu*k''-m*k'-rho*betaStar*k/omegaZ^2=0,
// with omegaZ=a*(y+ell). Removing convection gives the exponential factor;
// the omitted constant potential is O((m*y1/mu)^2), requiring refinement.
// Both smooth-wall k~y^p and finite-blow k~y limits follow continuously.
template<class R> UGKWP_GAS_HD inline R nearWallKShape(R y,R y1,R ell,R p,R advection){
    if(y<=0)return R(0);
    R shape;
    if(ell<=0)shape=::pow(y/y1,p);
    else {const R exponent=R(2)*p-R(1);
        const R numerator=-::expm1(-exponent*::log1p(y/ell));
        const R denominator=-::expm1(-exponent*::log1p(y1/ell));
        shape=::pow((y+ell)/(y1+ell),p)*numerator/denominator;}
    return shape*::exp(advection*(y-y1));
}
template<class R> UGKWP_GAS_HD inline R nearWallKDerivative(R y,R y1,R ell,R p,R advection){
    const R shape=nearWallKShape(y,y1,ell,p,advection);
    if(ell<=0)return shape*(p/y+advection);
    const R exponent=R(2)*p-R(1);
    const R correction=exponent/((y+ell)*::expm1(exponent*::log1p(y/ell)));
    return shape*(p/(y+ell)+correction+advection);
}
template<class R> UGKWP_GAS_HD inline R nearWallKDerivativeAtWall(R y1,R ell,R p,R advection){
    if(ell<=0)return R(0);
    const R exponent=R(2)*p-R(1),denominator=-::expm1(-exponent*::log1p(y1/ell));
    return exponent/ell*::pow(ell/(y1+ell),p)/denominator*::exp(-advection*y1);
}
// q(x)=(log(1+x)-x)/(-x*x). Taylor evaluation avoids cancellation.
// Eighteen coefficients at |x|<=1/8 bound truncation below double epsilon.
template<class R> UGKWP_GAS_HD inline R logSlopeRemainder(R x){
    if(::fabs(x)<=R(0.125)){
      R q=R(1)/R(19);
      // Constant denominators keep the same Horner order without runtime divides.
      q=R(1)/R(18)-x*q;
      q=R(1)/R(17)-x*q;
      q=R(1)/R(16)-x*q;
      q=R(1)/R(15)-x*q;
      q=R(1)/R(14)-x*q;
      q=R(1)/R(13)-x*q;
      q=R(1)/R(12)-x*q;
      q=R(1)/R(11)-x*q;
      q=R(1)/R(10)-x*q;
      q=R(1)/R(9)-x*q;
      q=R(1)/R(8)-x*q;
      q=R(1)/R(7)-x*q;
      q=R(1)/R(6)-x*q;
      q=R(1)/R(5)-x*q;
      q=R(1)/R(4)-x*q;
      q=R(1)/R(3)-x*q;
      q=R(1)/R(2)-x*q;
      return q;
    }
    return (R(1)-::log1p(x)/x)/x;
}
// Geometry-fixed three-point derivative exact for {1,y,log(y)}. General
// smooth fields remain second-order; no velocity law or source is prescribed.
// The stencil touching y=0 retains its ordinary quadratic derivative.
template<class R> UGKWP_GAS_HD inline R logShearDerivative(R left,R center,R right,R hLeft,R hRight,R y){
    if(y>hLeft){
      const R qLeft=logSlopeRemainder(-hLeft/y);
      const R qRight=logSlopeRemainder(hRight/y);
      // Equivalent to b*qRight/(a*qLeft+b*qRight), a=hLeft/y,
      // b=hRight/y. Scale physical spacings to avoid ratio underflow.
      const R spacing=max(hLeft,hRight),a=hLeft/spacing,b=hRight/spacing;
      const R weight=b*qRight/(a*qLeft+b*qRight);
      return weight*(center-left)/hLeft+(R(1)-weight)*(right-center)/hRight;
    }
    return centeredDerivative(left,center,right,hLeft,hRight);
}
}}}
