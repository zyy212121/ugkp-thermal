#pragma once
#include "GasStateView.H"
template<class GasState>
__device__ GPU_OPERATOR_REAL molecularGasConductivity(const GasState& s)
{
    if constexpr (ugkwp::GasDirectConductivity<GasState>::value)
        if (ugkwp::gasHasDirectConductivity(s)) return s.gasThermalConductivity;
    return s.gasMu*s.gasCp/s.gasPrClamped;
}

template<class GasState>
__device__ GPU_OPERATOR_REAL molecularGasConductivity(const GasState& s,int c,GPU_OPERATOR_REAL temperature)
{
    // A constant viscosity does not imply a homogeneous caloric model.
    // Active mixtures use physical species Cp(T,Y), never bootstrap gasCp.
    if(!ugkwp::gasVariableViscosity(s) && !ugkwp::mixtureGasActive(s))
        return molecularGasConductivity(s);
    if constexpr(ugkwp::GasDirectConductivity<GasState>::value)
        if(ugkwp::gasHasDirectConductivity(s))return s.gasThermalConductivity;
    GPU_OPERATOR_REAL cp=s.gasCp;
    if constexpr(ugkwp::GasStateTraits<GasState>::speciesCount>0)
        if(ugkwp::mixtureGasActive(s)){
            cp=0;
            for(int j=0;j<ugkwp::GasStateTraits<GasState>::speciesCount;++j)
                cp+=s.gasSpecies.rho[j*s.nCells+c]/s.rho[c]
                    *ugkwp::speciesCp(j,temperature,s.gasSpecies.thermo);
        }
    return ugkwp::gasCellMolecularViscosityAt(s,c,temperature)*cp/s.gasPrClamped;
}
