#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall { namespace detail {

template<class R> UGKWP_GAS_HD inline R centeredDerivative(R left,R center,R right,R hLeft,R hRight){
    return hRight*(center-left)/(hLeft*(hLeft+hRight))+hLeft*(right-center)/(hRight*(hLeft+hRight));
}




template<class R> UGKWP_GAS_HD inline R transportConductance(R a,R b){
    if(a==b)return a;
    if(a==0||b==0)return R(0);
    const R sign=a<0&&b<0?R(-1):R(1);
    a*=sign;b*=sign;
    if(!(a>0&&b>0))return std::numeric_limits<R>::quiet_NaN();
    const R low=min(a,b),high=max(a,b),difference=high-low;
    const bool ratioOverflows=low<R(1)&&difference>std::numeric_limits<R>::max()*low;
    const R logarithm=ratioOverflows?(::logf(high)-::logf(low)): ::log1pf(difference/low);
    return sign*difference/logarithm;
}



template<class R> UGKWP_GAS_HD inline R thermalWorkVelocityWeight(R muA,R muB,R bA,R bB){
    if(!(muA>0&&muB>0&&bA>0&&bB>0))return R(.5f);


    const bool reversed=muB<muA;
    if(reversed){R swap=muA;muA=muB;muB=swap;swap=bA;bA=bB;bB=swap;}
    const R difference=muB-muA;
    if(difference==0)return R(.5f);
    const bool wide=muA<R(1)&&difference>std::numeric_limits<R>::max()*muA;
    const R logarithm=wide?(::logf(muB)-::logf(muA)): ::log1pf(difference/muA);
    const R g=R(.2886751345948128822545743902509787f),t[2]={R(.5f)-g,R(.5f)+g};
    R weights[2],logWeights[2];bool useLogs=wide;
    for(int j=0;j<2;++j){
        if(wide){
            const R logMu=::logf(muA)+t[j]*logarithm,mu=::expf(logMu);


            const R logPosition=logMu-::logf(muB)+::log1pf(-muA/mu)-::log1pf(-muA/muB);
            const R position=::expf(logPosition);
            const R left=::logf(bA)+::log1pf(-position),right=::logf(bB)+logPosition;
            const R largest=max(left,right);
            const R logB=largest+::logf(::expf(left-largest)+::expf(right-largest));
            logWeights[j]=logMu-logB;
        }else{
            const R relative=::expm1f(t[j]*logarithm);
            const R position=muA*relative/difference,mu=muA*(R(1)+relative);
            const R b=(R(1)-position)*bA+position*bB;
            weights[j]=mu/b;
            if(!(weights[j]>0&&weights[j]<=std::numeric_limits<R>::max())){
                logWeights[j]=::logf(mu)-::logf(b);useLogs=true;}
        }
    }
    if(useLogs){
        if(!wide)for(int j=0;j<2;++j)
            if(weights[j]>0&&weights[j]<=std::numeric_limits<R>::max())logWeights[j]=::logf(weights[j]);
        const R largest=max(logWeights[0],logWeights[1]);
        for(int j=0;j<2;++j)weights[j]=::expf(logWeights[j]-largest);}
    else {const R largest=max(weights[0],weights[1]);
        for(int j=0;j<2;++j)weights[j]/=largest;}
    const R result=(weights[0]*t[0]+weights[1]*t[1])/(weights[0]+weights[1]);
    return reversed?R(1)-result:result;
}





template<class R> UGKWP_GAS_HD inline R powerValue(R a,R b,R ya,R yb,R y){
    if(!(a>0&&b>0&&ya>0))return a+(b-a)*(y-ya)/(yb-ya);
    const R exponent=::logf(b/a)/::logf(yb/ya);
    return a*::expf(exponent*::logf(y/ya));
}
template<class R> UGKWP_GAS_HD inline R powerGradient(R a,R b,R ya,R yb,R y){
    if(!(a>0&&b>0&&ya>0))return (b-a)/(yb-ya);
    const R exponent=::logf(b/a)/::logf(yb/ya);
    return exponent*powerValue(a,b,ya,yb,y)/y;
}
template<class R> UGKWP_GAS_HD inline R powerIntegral(R a,R b,R ya,R yb,R lo,R hi){
    const R exponent=::logf(b/a)/::logf(yb/ya),e=exponent+R(1);
    const R value=powerValue(a,b,ya,yb,lo),logarithm=::logf(hi/lo);
    return value*lo*(e==R(0)?logarithm:(::expm1f(e*logarithm)/e));
}
template<class R> UGKWP_GAS_HD inline R omegaSquareAverage(R a,R b,R c,R ya,R yb,R yc){
    const R lo=(ya+yb)/R(2),hi=(yb+yc)/R(2);
    return (powerIntegral(a*a,b*b,ya,yb,lo,yb)+powerIntegral(b*b,c*c,yb,yc,yb,hi))/(hi-lo);
}


template<class R> UGKWP_GAS_HD inline R omegaGradientFit(R left,R right){
    if(left<=0||right<=0)return R(1);
    const R mid=(left+right)/R(2),ratio=left*right/(mid*mid);return ratio*ratio;
}




template<class R> UGKWP_GAS_HD inline R omegaInverseSquareValue(R a,R b,R ya,R yb,R y){
    if(y==ya||a==b)return a;
    if(y==yb)return b;
    const R wa=(ya/y)*(ya/y)*((yb-y)/(yb-ya))*((yb+y)/(yb+ya));
    const R wb=(yb/y)*(yb/y)*((y-ya)/(yb-ya))*((y+ya)/(yb+ya));
    return wa*a+wb*b;
}


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




template<class R> UGKWP_GAS_HD inline R nearWallKShape(R y,R y1,R ell,R p,R advection){
    if(y<=0)return R(0);
    R shape;
    if(ell<=0)shape=::powf(y/y1,p);
    else {const R exponent=R(2)*p-R(1);
        const R numerator=-::expm1f(-exponent*::log1pf(y/ell));
        const R denominator=-::expm1f(-exponent*::log1pf(y1/ell));
        shape=::powf((y+ell)/(y1+ell),p)*numerator/denominator;}
    return shape*::expf(advection*(y-y1));
}
template<class R> UGKWP_GAS_HD inline R nearWallKDerivative(R y,R y1,R ell,R p,R advection){
    const R shape=nearWallKShape(y,y1,ell,p,advection);
    if(ell<=0)return shape*(p/y+advection);
    const R exponent=R(2)*p-R(1);
    const R correction=exponent/((y+ell)*::expm1f(exponent*::log1pf(y/ell)));
    return shape*(p/(y+ell)+correction+advection);
}
template<class R> UGKWP_GAS_HD inline R nearWallKDerivativeAtWall(R y1,R ell,R p,R advection){
    if(ell<=0)return R(0);
    const R exponent=R(2)*p-R(1),denominator=-::expm1f(-exponent*::log1pf(y1/ell));
    return exponent/ell*::powf(ell/(y1+ell),p)/denominator*::expf(-advection*y1);
}


template<class R> UGKWP_GAS_HD inline R logSlopeRemainder(R x){
    if(::fabsf(x)<=R(0.125f)){
      R q=R(1)/R(19);

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
    return (R(1)-::log1pf(x)/x)/x;
}



template<class R> UGKWP_GAS_HD inline R logShearDerivative(R left,R center,R right,R hLeft,R hRight,R y){
    if(y>hLeft){
      const R qLeft=logSlopeRemainder(-hLeft/y);
      const R qRight=logSlopeRemainder(hRight/y);


      const R spacing=max(hLeft,hRight),a=hLeft/spacing,b=hRight/spacing;
      const R weight=b*qRight/(a*qLeft+b*qRight);
      return weight*(center-left)/hLeft+(R(1)-weight)*(right-center)/hRight;
    }
    return centeredDerivative(left,center,right,hLeft,hRight);
}
}}}
