#ifndef CHMT_DRY_MATERIAL_TYPES_H
#define CHMT_DRY_MATERIAL_TYPES_H
#include <array>
#include <cstddef>

namespace chmt { namespace dry {
// CPU-only material mathematics. No gas solver, device ABI, film, or CHT state.
// Masses are extensive [kg], energy [J] includes formation energies of ALL
// constituents. Darcy kinetic energy is neglected in this LTE material model.
template<std::size_t Nc, std::size_t Ns>
struct MaterialState {
    std::array<double,Nc> condensedMass{};
    std::array<double,Ns> poreMass{};
    double energy=0;
};

// Signed mass produced per CHOSEN reaction extent unit [kg/extent]. Examples:
// mass-normalized pyrolysis uses kg/kg; molar surface kinetics uses kg/mol.
// Extent-rate and coefficient units must match; neither is silently converted.
// These are mass production coefficients, not dimensionless molar stoichiometry.
template<std::size_t Nc, std::size_t Ns>
struct Stoichiometry {
    std::array<double,Nc> condensed{};
    std::array<double,Ns> gas{};
};

struct CondensedThermo {
    double density=0;                 // kg/m3 of constituent (not bulk density)
    double cp0=0, cp1=0;              // e(T)=e0+cp0*T+cp1*T*T/2
    double formationEnergy=0;         // J/kg; same reference convention as gas
    double conductivity=0;           // W/(m K)
    double internalEnergy(double T) const {
        return formationEnergy+cp0*T+0.5*cp1*T*T;
    }
    double heatCapacity(double T) const {return cp0+cp1*T;}
};

// Gas calorics are borrowed through a caller-supplied functor with methods
// internalEnergy(s,T), heatCapacityCv(s,T), enthalpy(s,T), gasConstant(s).
// The eventual CHMT adapter must use the existing shared gas thermo tables,
// including their formation reference, NOT fit a second gas EOS here.
struct CaloricState {double energy=0, capacity=0;};
struct MaterialPrimitive {
    double temperature=0, porosity=0, pressure=0, poreDensity=0;
    double conductivity=0;
};
} }
#endif
