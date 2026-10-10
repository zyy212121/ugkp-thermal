#pragma once
#include "GasModel.H"

namespace ugkwp
{
// Status belongs to trial storage; any nonzero cell/face status rejects the
// entire trial. Applications must never publish a partially accepted update.
enum class GasTransportCode : int
{
    Success = 0,
    InvalidStorage,
    InvalidComposition,
    InvalidThermodynamics,
    UnsupportedConfiguration,
    InvalidGeometry,
    NonFiniteState,
    NegativeInventory,
    ChemistryFailure
};
struct GasCapabilityRequest
{
    GasMode mode = GasMode::SingleLegacy;
    int fluxScheme = 1; // existing create codes 1..9, not Scheme enum ordinals
    int reconstruction = 0;
    int limiter = 0;
    int timeIntegrator = 1;
    int turbulenceModel = 0;
    bool movingGeometry = false;
    bool particleCoupling = false;
    int sstWallTreatment = 0; // 0 wall-resolved low-Re, 1 high-Re wall functions
};
struct GasCapabilityResult
{
    GasTransportCode code;
    const char* message;
    UGKWP_GAS_HD explicit operator bool() const noexcept
    { return code == GasTransportCode::Success; }
};
UGKWP_GAS_HD inline GasCapabilityResult validateGasCapabilities(const GasCapabilityRequest& r)
{
    if (r.fluxScheme < 1 || r.fluxScheme > 9 || r.reconstruction < 0 || r.reconstruction > 2
        || r.limiter < 0 || r.limiter > 2 || r.timeIntegrator < 1 || r.timeIntegrator > 3
        || r.turbulenceModel < 0 || r.turbulenceModel > 3)
        return {GasTransportCode::UnsupportedConfiguration, "invalid gas numerical selector"};
    if (r.movingGeometry && r.timeIntegrator != 1)
        return {GasTransportCode::UnsupportedConfiguration, "ALE currently requires Euler with certified stage sweeps and volumes"};
    if (r.movingGeometry && r.reconstruction == 2)
        return {GasTransportCode::UnsupportedConfiguration, "ALE energyLimitedLinear correction is not yet verified"};
    if (r.movingGeometry && r.particleCoupling)
        return {GasTransportCode::UnsupportedConfiguration, "ALE particle void-fraction closure is not enabled"};
    if (r.mode == GasMode::SingleLegacy)
        return {GasTransportCode::Success, "legacy gas capabilities"};
    if (r.mode != GasMode::MixtureFrozen && r.mode != GasMode::MixtureChemistry)
        return {GasTransportCode::UnsupportedConfiguration, "invalid gas mixture mode"};
    if (r.fluxScheme != 1 && r.fluxScheme != 2)
        return {GasTransportCode::UnsupportedConfiguration, "mixture species splitting is currently verified only for Rusanov and Kurganov/HLL"};
    if (r.reconstruction != 0 && r.reconstruction != 1)
        return {GasTransportCode::UnsupportedConfiguration, "mixture reconstruction supports firstOrder or composition-aware MUSCL"};
    if (r.turbulenceModel == 3 && r.sstWallTreatment != 0)
        return {GasTransportCode::UnsupportedConfiguration, "mixture SST currently requires wall-resolved low-Re thermal treatment"};
    if (r.particleCoupling)
        return {GasTransportCode::UnsupportedConfiguration, "mixture particle thermal and void-fraction closure is not enabled"};
    return {GasTransportCode::Success, "mixture shared Rusanov or Kurganov/HLL with firstOrder or MUSCL"};
}

} // namespace ugkwp
