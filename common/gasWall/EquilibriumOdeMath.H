#pragma once
#include "WallModelTypes.H"

namespace ugkwp { namespace gaswall { namespace detail {
// Semilocal EWM2 closure, Di Renzo & Urzay (CTR 2019, corrected 2020).
// Adapted from URANOS getmuT, commit 9c69c943c437ec8f11e0f70bd4c4deb64c17cd7a.
// Copyright (c) 2022, Department of Industrial Engineering,
// University of Padova, Italy. See third_party/uranos/LICENSE.
// This is dynamic viscosity. Transpiration enters the conservative transport
// equations; this equilibrium damping does not model every blowing-induced
// change to turbulence structure.
template<class R> UGKWP_GAS_HD inline R equilibriumDynamicViscosity(
 R rho,R wallStress,R y,R molecularViscosity,R kappa,R aPlus){
    if(y==0||wallStress==0||kappa==0)return R(0);
    const R momentumScale=::sqrt(rho*wallStress);
    const R yStar=y*momentumScale/molecularViscosity;
    const R damping=-::expm1(-yStar/aPlus);
    return kappa*y*momentumScale*damping*damping;
}
// The local-stress extension retains the baseline semilocal wall damping.
// It follows j*u-tau=constant, without adding a second advective source.
template<class R> UGKWP_GAS_HD inline R equilibriumLocalStressViscosity(
 R rho,R wallStress,R localStress,R y,R molecularViscosity,R kappa,R aPlus){
    if(y==0||wallStress==0||localStress==0||kappa==0)return R(0);
    const R yStar=y*::sqrt(rho*wallStress)/molecularViscosity;
    const R damping=-::expm1(-yStar/aPlus);
    return kappa*y*::sqrt(rho*localStress)*damping*damping;
}
}}}
