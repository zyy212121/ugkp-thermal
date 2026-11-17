#ifndef CHMT_DRY_MATERIAL_TYPES_H
#define CHMT_DRY_MATERIAL_TYPES_H
#include "Portability.H"
#include "../../../../common/GpuPrecisionTypes.H"
#include <array>
#include <cstddef>

namespace chmt { namespace dry {
using PhysicalReal=GpuReal;
using ThermalReal=GpuReal;



template<std::size_t Nc, std::size_t Ns>
struct MaterialState {
    std::array<PhysicalReal,Nc> condensedMass{};
    std::array<PhysicalReal,Ns> poreMass{};
    ThermalReal energy=0;
};





template<std::size_t Nc, std::size_t Ns>
struct Stoichiometry {
    std::array<PhysicalReal,Nc> condensed{};
    std::array<PhysicalReal,Ns> gas{};
};

struct CondensedThermo {
    PhysicalReal density=0;
    ThermalReal cp0=0, cp1=0;
    ThermalReal formationEnergy=0;
    ThermalReal conductivity=0;
    CHMT_DRY_HD ThermalReal internalEnergy(ThermalReal T) const {
        return formationEnergy+cp0*T+ThermalReal(.5)*cp1*T*T;
    }
    CHMT_DRY_HD ThermalReal heatCapacity(ThermalReal T) const {return cp0+cp1*T;}
};





struct CaloricState {ThermalReal energy=0, capacity=0;};
struct MaterialPrimitive {
    ThermalReal temperature=0;PhysicalReal porosity=0, pressure=0, poreDensity=0;
    ThermalReal conductivity=0;
};
} }
#endif
