#pragma once
#include "GasStateView.H"
#include "MolecularConductivity.H"
template<class GasState>
__device__ GPU_OPERATOR_REAL molecularGasConductivity(const GasState& s)
{
    if constexpr (ugkwp::GasDirectConductivity<GasState>::value)
        if (ugkwp::gasHasDirectConductivity(s)) return s.gasThermalConductivity;
    return ugkwp::conductivityFromViscosityCpPr(s.gasMu,s.gasCp,s.gasPrClamped);
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
            GPU_OPERATOR_REAL fractions[ugkwp::GasStateTraits<GasState>::speciesCount];
            for(int j=0;j<ugkwp::GasStateTraits<GasState>::speciesCount;++j)
                fractions[j]=s.gasSpecies.rho[j*s.nCells+c]/s.rho[c];
            cp=ugkwp::mixtureSpecificCp(fractions,temperature,s.gasSpecies.thermo);
        }
    return ugkwp::conductivityFromViscosityCpPr(ugkwp::gasCellMolecularViscosityAt(s,c,temperature),
        cp,ugkwp::gasCellMolecularPrandtl(s,c));
}
