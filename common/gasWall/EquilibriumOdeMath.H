#pragma once
#include "WallModelTypes.H"

namespace ugkwp { namespace gaswall { namespace detail {







template<class R> UGKWP_GAS_HD inline R equilibriumDynamicViscosity(
 R rho,R wallStress,R y,R molecularViscosity,R kappa,R aPlus){
    if(y==0||wallStress==0||kappa==0)return R(0);
    const R momentumScale=::sqrt(rho*wallStress);
    const R yStar=y*momentumScale/molecularViscosity;
    const R damping=-::expm1(-yStar/aPlus);
    return kappa*y*momentumScale*damping*damping;
}


template<class R> UGKWP_GAS_HD inline R equilibriumLocalStressViscosity(
 R rho,R wallStress,R localStress,R y,R molecularViscosity,R kappa,R aPlus){
    if(y==0||wallStress==0||localStress==0||kappa==0)return R(0);
    const R yStar=y*::sqrt(rho*wallStress)/molecularViscosity;
    const R damping=-::expm1(-yStar/aPlus);
    return kappa*y*::sqrt(rho*localStress)*damping*damping;
}
}}}
