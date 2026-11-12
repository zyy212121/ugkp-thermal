#ifndef CHMT_DRY_MATERIAL_TYPES_H
#define CHMT_DRY_MATERIAL_TYPES_H
#include "Portability.H"
#include "../../../../common/GpuPrecisionTypes.H"
#include <array>
#include <cstddef>

namespace chmt { namespace dry {
using PhysicalReal=GpuReal;
using ThermalReal=GpuReal;

// Masses are extensive [kg], energy [J] includes formation energies of ALL
// constituents. Darcy kinetic energy is neglected in this LTE material model.
template<std::size_t Nc, std::size_t Ns>
struct MaterialState {
    std::array<PhysicalReal,Nc> condensedMass{};
    std::array<PhysicalReal,Ns> poreMass{};
    ThermalReal energy=0;
};

// Signed mass produced per CHOSEN reaction extent unit [kg/extent]. Examples:
// mass-normalized pyrolysis uses kg/kg; molar surface kinetics uses kg/mol.
// Extent-rate and coefficient units must match; neither is silently converted.
// These are mass production coefficients, not dimensionless molar stoichiometry.
template<std::size_t Nc, std::size_t Ns>
struct Stoichiometry {
    std::array<PhysicalReal,Nc> condensed{};
    std::array<PhysicalReal,Ns> gas{};
};

struct CondensedThermo {
    PhysicalReal density=0;                 // kg/m3 of constituent (not bulk density)
    ThermalReal cp0=0, cp1=0;              // e(T)=e0+cp0*T+cp1*T*T/2
    ThermalReal formationEnergy=0;         // J/kg; same reference convention as gas
    ThermalReal conductivity=0;           // W/(m K)
    CHMT_DRY_HD ThermalReal internalEnergy(ThermalReal T) const {
        return formationEnergy+cp0*T+ThermalReal(.5)*cp1*T*T;
    }
    CHMT_DRY_HD ThermalReal heatCapacity(ThermalReal T) const {return cp0+cp1*T;}
};

// Gas calorics are borrowed through a caller-supplied functor with methods
// internalEnergy(s,T), heatCapacityCv(s,T), enthalpy(s,T), gasConstant(s).
// The eventual CHMT adapter must use the existing shared gas thermo tables,
// including their formation reference, NOT fit a second gas EOS here.
struct CaloricState {ThermalReal energy=0, capacity=0;};
struct MaterialPrimitive {
    ThermalReal temperature=0;PhysicalReal porosity=0, pressure=0, poreDensity=0;
    ThermalReal conductivity=0;
};
} }
#endif
