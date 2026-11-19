#ifndef CHMT_DRY_MATERIAL_TYPES_H
#define CHMT_DRY_MATERIAL_TYPES_H
#include "Portability.H"
#include <array>
#include <cstddef>

namespace chmt { namespace dry {



template<std::size_t Nc, std::size_t Ns>
struct MaterialState {
    std::array<double,Nc> condensedMass{};
    std::array<double,Ns> poreMass{};
    double energy=0;
};





template<std::size_t Nc, std::size_t Ns>
struct Stoichiometry {
    std::array<double,Nc> condensed{};
    std::array<double,Ns> gas{};
};

struct CondensedThermo {
    double density=0;
    double cp0=0, cp1=0;
    double formationEnergy=0;
    double conductivity=0;
    CHMT_DRY_HD double internalEnergy(double T) const {
        return formationEnergy+cp0*T+0.5*cp1*T*T;
    }
    CHMT_DRY_HD double heatCapacity(double T) const {return cp0+cp1*T;}
};





struct CaloricState {double energy=0, capacity=0;};
struct MaterialPrimitive {
    double temperature=0, porosity=0, pressure=0, poreDensity=0;
    double conductivity=0;
};
} }
#endif
