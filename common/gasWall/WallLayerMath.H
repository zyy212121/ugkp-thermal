#pragma once
#include "WallModelTypes.H"
namespace ugkwp { namespace gaswall { namespace detail {
// Quadratic-exact derivative at a nonuniform interior node.
template<class R> UGKWP_GAS_HD inline R centeredDerivative(R left,R center,R right,R hLeft,R hRight){
    return hRight*(center-left)/(hLeft*(hLeft+hRight))+hLeft*(right-center)/(hRight*(hLeft+hRight));
}
// Positive, fixed-geometry fitting. Exact for omega=A/y^2 and second-order
// consistent at fixed positive y. This is not a linear-exact face gradient.
template<class R> UGKWP_GAS_HD inline R omegaGradientFit(R left,R right){
    if(left<=0||right<=0)return R(1);
    const R mid=(left+right)/R(2),ratio=left*right/(mid*mid);return ratio*ratio;
}
template<class R> UGKWP_GAS_HD inline R omegaDestructionFit(R y,R faceLeft,R faceRight){
    return y*y*y*y*(R(1)/(faceLeft*faceLeft*faceLeft)-R(1)/(faceRight*faceRight*faceRight))/(R(3)*(faceRight-faceLeft));
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
}}}
