#pragma once
#include "GasStateView.H"
namespace ugkwp {



template<class State,class Real>
UGKWP_GAS_HD inline Real particleGasHeatCapacity(const State& s,int cell,Real volume,Real legacy) {
    if constexpr(GasStateTraits<State>::speciesCount>0)
        if(mixtureGasActive(s))
            return volume*Real(s.rho[cell])*(Real(s.gasSpecies.heatCapacity[cell])-Real(s.gasSpecies.gasConstant[cell]));
    return legacy;
}
template<class State,class Real>
UGKWP_GAS_HD inline Real particleGasMinimumInternalEnergy(const State& s,int cell,Real legacy) {
    if constexpr(GasStateTraits<State>::speciesCount>0)
        if(mixtureGasActive(s)) {
            constexpr int Ns=GasStateTraits<State>::speciesCount;
            Real partial[Ns];for(int k=0;k<Ns;++k)partial[k]=s.gasSpecies.rho[k*s.nCells+cell];
            Real lower,upper;mixtureTemperatureBounds(s.gasSpecies.thermo,s.gasSpecies.thermoControls,lower,upper);
            return mixtureEnergy(partial,lower,s.gasSpecies.thermo);
        }
    return legacy;
}
}
