#pragma once
#include "GasStateView.H"
// Inert particles displace every intrinsic species by the identical gas mass
// factor. No species is created, consumed, or exchanged with the solid phase.
template<class State,class Real>
UGKWP_GAS_HD inline void scaleMixturePorositySpecies(State& s,int c,Real scale)
{
    if(ugkwp::mixtureGasActive(s))
        for(int k=0;k<ugkwp::GasStateTraits<State>::speciesCount;++k)
            s.gasSpecies.rho[k*s.nCells+c]*=scale;
}
