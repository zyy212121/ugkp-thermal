#ifndef CHMT_GPU_DEVICEVIEWS_H
#define CHMT_GPU_DEVICEVIEWS_H
#include "core/Configuration.H"
namespace chmt {
// Material-side thermal-column POD only. Gas and SST use common GasStateView.
struct NormalThermalView {
    int nFaces=0,nLayers=0; // nFaces is number of normal thermal columns.
    const int* offsets=nullptr; // nFaces+1 CSR layer offsets.
    const Real* coordinate=nullptr;const Real* thickness=nullptr; // nLayers, m.
    const Real* area=nullptr; // nFaces, m^2 reference column area.
    const Real* pressure=nullptr;const Real* pressureRate=nullptr;
    // nFaces, Pa and Pa/s. pressureRate is prescribed material Dp/Dt along
    // the same normal-column material trajectory used by mass/enthalpy update.
    // Its V*Dp/Dt true-H source is evolved here; do NOT add a second film
    // delta(pV) source to normal enthalpy. U diagnostics subtract actual pV.
    // When normal true H is authoritative, film H aggregates these same layers;
    // it must not receive an independent parallel pressure-volume conversion.
    const Real* mass=nullptr;const Real* oldVolume=nullptr;const Real* newVolume=nullptr; // nLayers: mass [kg], old/new layer volume [m^3].
    Real* enthalpy=nullptr;Real* rhs=nullptr;Real* temperature=nullptr;
    Real* liquidFraction=nullptr; // nLayers, dimensionless, persistent phase state.
    Real* bottomHeatRate=nullptr;Real* topHeatRate=nullptr; // nFaces, W inward.
    Real* acceptedWallHeat=nullptr; // nFaces, cumulative accepted net wall heat, J.
    // RHS/corrector kernels may write trial heat RATES only. They must never
    // increment acceptedWallHeat. Backend commits stage-1 integral exactly once.
    const Real* bottomTemperature=nullptr;const Real* topTemperature=nullptr;
    const Real* bottomHeatFlux=nullptr;const Real* topHeatFlux=nullptr;
    // Boundary arrays have nFaces entries: temperature [K], inward heat flux [W/m^2].
    // Exactly one boundary temperature or signed inward heat-flux array is
    // provided per side; absent side means no boundary prescription is supplied.
};
static_assert(std::is_trivially_copyable<NormalThermalView>::value,"NormalThermalView device ABI");
}
#endif
