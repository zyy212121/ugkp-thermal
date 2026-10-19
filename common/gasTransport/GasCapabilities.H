#pragma once
#include "GasModel.H"

namespace ugkwp
{


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
    ChemistryFailure,
    SourceStepLimit,
    BoundaryLayerFailure // terminal wall BVP failure; independent of bulk chemistry
};
// One producer/host contract for implemented conservative mixture fluxes.
UGKWP_GAS_HD inline bool mixtureGasFluxImplemented(int createCode)
{ return createCode==1 || createCode==2 || createCode==4; }

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
    int sstWallTreatment = 0; // 0 lowRe, 1 wallFunction, 2 boundaryLayer
    int boundaryLayerModel = 1; // 0 restricted constantTransport, 1 reactingSst
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
    if (r.sstWallTreatment == 2)
    {
        if (r.mode == GasMode::SingleLegacy || r.particleCoupling)
            return {GasTransportCode::UnsupportedConfiguration, "boundaryLayer requires shared mixture thermo and gas-only storage"};
        if (r.boundaryLayerModel != 0 && r.boundaryLayerModel != 1)
            return {GasTransportCode::UnsupportedConfiguration, "selected boundaryLayer model has no implementation"};
    }
    if (r.movingGeometry && r.particleCoupling)
        return {GasTransportCode::UnsupportedConfiguration, "ALE particle void-fraction closure is not enabled"};
    if (r.mode == GasMode::SingleLegacy)
        return {GasTransportCode::Success, "legacy gas capabilities"};
    if (r.particleCoupling)
        return {GasTransportCode::UnsupportedConfiguration, "mixture particle thermal and void-fraction closure is not enabled"};
    return {GasTransportCode::Success, "mixture shared Rusanov, Kurganov/HLL or HLLC with firstOrder or MUSCL"};
}

} // namespace ugkwp
