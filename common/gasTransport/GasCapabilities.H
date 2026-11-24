#pragma once
#include "GasModel.H"
#include "GasBuildConfig.H"
#include "../gasWall/WallFamily.H"

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
    BoundaryLayerFailure
};

UGKWP_GAS_HD inline bool mixtureGasFluxImplemented(int createCode)
{ return createCode==1 || createCode==2 || createCode==4 || createCode==8; }

struct GasCapabilityRequest
{
    GasMode mode = GasMode::SingleLegacy;
    int fluxScheme = 1;
    int reconstruction = 0;
    int limiter = 0;
    int timeIntegrator = 1;
    int turbulenceModel = 0;
    bool movingGeometry = false;
    bool particleCoupling = false;

    bool pairedMixtureParticleCoupling = false;
    bool residentParticleGeometry = false;
    bool residentParticleWallCoupling = false;
    int sstWallTreatment = 0;
    int boundaryLayerModel = 1;
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
        if (r.mode==GasMode::SingleLegacy && !(compiledGasSpecies==1 && r.boundaryLayerModel==2))
            return {GasTransportCode::UnsupportedConfiguration, "boundaryLayer requires shared species thermo or a single-gas equilibriumOde adapter"};
        if (r.particleCoupling&&!r.residentParticleWallCoupling)
            return {GasTransportCode::UnsupportedConfiguration, "boundaryLayer particle exchange is not configured"};
        if (r.boundaryLayerModel != 0 && r.boundaryLayerModel != 1 && r.boundaryLayerModel != 2)
            return {GasTransportCode::UnsupportedConfiguration, "selected boundaryLayer model has no implementation"};
    }
    if (r.movingGeometry && r.particleCoupling&&!r.residentParticleGeometry)
        return {GasTransportCode::UnsupportedConfiguration, "ALE particle void-fraction closure is not enabled"};
    if (r.mode == GasMode::SingleLegacy)
        return {GasTransportCode::Success, "legacy gas capabilities"};
    if (r.particleCoupling && !(r.pairedMixtureParticleCoupling && r.mode==GasMode::MixtureFrozen))
        return {GasTransportCode::UnsupportedConfiguration, "mixture particle thermal and void-fraction closure is not enabled"};
    return {GasTransportCode::Success, "mixture shared Rusanov, Kurganov/HLL, HLLC or SLAU2 with firstOrder or MUSCL"};
}

}
