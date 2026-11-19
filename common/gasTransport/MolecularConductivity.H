#pragma once
#include "MolecularViscosity.H"
namespace ugkwp {


template<class Real,int N,class TableReal>
UGKWP_GAS_HD inline Real mixtureSpecificCp(const Real* fractions,Real temperature,
    const SpeciesThermoView<TableReal,N>& thermo) noexcept {
    Real cp=0;
    for(int j=0;j<N;++j)cp+=fractions[j]*speciesCp(j,temperature,thermo);
    return cp;
}
template<class Real>
UGKWP_GAS_HD inline Real conductivityFromViscosityCpPr(Real viscosity,Real cp,Real prandtl) noexcept {
    return viscosity*cp/prandtl;
}
}
