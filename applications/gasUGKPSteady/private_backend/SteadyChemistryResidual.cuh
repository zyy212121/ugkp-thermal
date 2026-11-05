#pragma once
#include "../../../common/chemistry/GasRates.H"

// Instantaneous intrinsic species-density source, not a split-step increment.
// Formation-inclusive total energy has no separate chemical heat-release RHS.
template<class State, class Real>
__host__ __device__ inline bool steadyAddChemistryResidual
(const State& s, int c, Real* speciesResidual)
{
    if(s.gasSpecies.mode!=ugkwp::GasMode::MixtureChemistry)return true;
    constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;
    if constexpr(Ns>0)
    {
        Real concentration[Ns],production[Ns];
        for(int k=0;k<Ns;++k)
            concentration[k]=s.gasSpecies.rho[k*s.nCells+c]
                /s.gasSpecies.thermo.species[k].molarMass;
        ugkwp::ChemistryStatus status;
        if(!ugkwp::evaluateGasRates(Real(s.Tgas[c]),concentration,
            s.gasSpecies.thermo,s.gasSpecies.mechanism,production,nullptr,status))
            return false;
        for(int k=0;k<Ns;++k)
            speciesResidual[k]+=production[k]*s.gasSpecies.thermo.species[k].molarMass;
    }
    return true;
}
