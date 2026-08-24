#ifndef GpuThermalLedgerMath_H
#define GpuThermalLedgerMath_H

namespace Foam
{
namespace GpuThermalLedgerMath
{

struct EnergyExchange
{
    double deltaGasEnergy;
    double gasWallEnergy;
};

constexpr EnergyExchange finalGasWallEnergyExchange
(
    const double dt,
    const double finalGasPhiRhoE
) noexcept
{
    return
    {
        -dt*finalGasPhiRhoE,
        dt*finalGasPhiRhoE
    };
}

}                                  
}                  

#endif
