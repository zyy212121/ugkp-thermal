#ifndef CHMT_CORE_CONFIGURATION_H
#define CHMT_CORE_CONFIGURATION_H
#include "Types.H"
#include "../../../common/gasTransport/GasModel.H"
#include "../../../common/chemistry/ChemistryTypes.H"
#include "../../../common/gasWall/WallModelTypes.H"
#include <array>
#include <string>

namespace chmt {
struct SstConfig {
    Real a1=0.31,betaStar=0.09,beta1=0.075,beta2=0.0828,gamma1=5.0/9.0,gamma2=0.44;
    Real sigmaK1=0.85,sigmaK2=1.0,sigmaOmega1=0.5,sigmaOmega2=0.856;
    Real turbulentPrandtl=0.9,turbulentSchmidt=0.7,productionLimit=10.0;
    Real minimumK=0,minimumOmega=0;
};
enum class EvaporationLaw { Disabled, HertzKnudsen };
struct MaterialPolicyConfig {
    Real minimumPorosity=0,maximumPorosity=1;
    int phaseCondensed=-1;
    bool enableMelting=false,enableSurfaceReactions=false,enablePoreOutflow=false;
    Real phaseFilmY[Ns]{};
    int nSurfaceReactions=0;
    Reaction surfaceReactions[Nr]{};
    Real surfaceGasOrder[Nr][Ns]{};
    EvaporationLaw evaporation=EvaporationLaw::Disabled;
    Real evaporationAccommodation[Ns]{};
    Real saturationPressureScale[Ns]{},saturationTemperature[Ns]{};
    Real gasContactResistance=0,solidContactResistance=0; // m^2 K/W.
};
enum class ParticleContactMode { ElasticRebound, Deposit };
struct ParticlePolicyConfig {
    int condensedIndex=-1,maxFaceCrossings=32;
    Real heatTransferCoefficient=0,contactHeatTransferCoefficient=0; // W/(m^2 K).
    Real contactDuration=0,emissivity=0; // s, dimensionless.
    ParticleContactMode contact=ParticleContactMode::ElasticRebound;
};
enum class MeshMotionPolicy { Static, CoupledRecession, PrescribedSinusoidal };
struct MeshMotionConfig {
    MeshMotionPolicy policy=MeshMotionPolicy::Static;
    Vec3 amplitude{},spatialWaveNumber{},spatialOrigin{};
    Real angularFrequency=0,timeOrigin=0;
};
inline ugkwp::gaswall::WallModelConfig<Real> defaultWallModel() {
    ugkwp::gaswall::WallModelConfig<Real> value;value.family=ugkwp::gaswall::WallFamily::LowRe;return value;
}
struct PhysicsConfig {
    ugkwp::gaswall::WallModelConfig<Real> wallModel=defaultWallModel();
    int wallWorkspaceSlots=0; // zero resolves a bounded pool from the actual device
    bool enableGas=true;
    ugkwp::GasMode gasMode=ugkwp::GasMode::SingleLegacy;
    int singleGasSpecies=-1; // Active physical material species in legacy five-field transport.
    std::uint64_t gasThermoFingerprint=0,gasMechanismFingerprint=0;
    ugkwp::ChemistryControls<Real> gasChemistryControls{};
    MeshMotionConfig meshMotion{};
    ParticlePolicyConfig particle{};
    MaterialPolicyConfig material{};
    SpeciesThermo species[Ns]{};CondensedThermo condensed[Nc]{};Reaction reactions[Nr]{};int nReactions=0,nElements=0;
    CondensedThermo liquid{}; // e0 is h_ref's T=0 intercept for THIS table.
    Real liquidViscosity=0,liquidReferencePressure=0,meltTemperature=0;
    Real permeability=0,poreViscosity=0,gasViscosity=0,gasConductivity=0,gasDiffusivity[Ns]{};
    Real emissivity=0,ambientTemperature=0;Vec3 gravity{};
    bool enableFilm=false,enableReactions=false,enableParticles=false,enableSst=false,enableRadiation=false;
    FilmThermalMode filmThermalMode=FilmThermalMode::ThicknessAveraged;
    ReconstructionMode reconstruction=ReconstructionMode::LimitedLinear;
    SstConfig sst{};Tolerances tolerances{};
    Real minDt=0,maxDt=0,cfl=0;int spatialOrder=2;
    std::uint64_t modelFingerprint=0,speciesFingerprint=defaultSpeciesHash,schemaFingerprint=checkpointSchema;
};
inline bool validateWallModelConfig(const PhysicsConfig& p,std::string& error) {
    const auto& w=p.wallModel;using namespace ugkwp::gaswall;
    if(w.family==WallFamily::LowRe){error.clear();return true;}
    if(w.family!=WallFamily::BoundaryLayer){error="CHMT ordinary wallFunction coupling is not enabled";return false;}
    if(p.material.gasContactResistance!=0){error="boundaryLayer requires zero gas contact resistance";return false;}
    if(p.gasMode==ugkwp::GasMode::SingleLegacy||!(p.gasViscosity>0)||!(p.gasConductivity>0)){
        error="boundaryLayer requires shared mixture thermo and positive molecular transport";return false;}
    if(p.wallWorkspaceSlots<0||p.wallWorkspaceSlots>4096||w.nodes<4||w.nodes>128||w.maxIterations<=0
        ||!finite(w.relativeTolerance)||w.relativeTolerance<=0||!finite(w.absoluteTolerance)||w.absoluteTolerance<0
        ||!finite(w.stretch)||w.stretch<1){error="invalid boundaryLayer solver controls";return false;}
    if(w.model==BoundaryLayerModel::ConstantTransport){
        if(p.enableSst||p.gasMode!=ugkwp::GasMode::MixtureFrozen){error="constantTransport is a frozen laminar analytic limit";return false;}
    }else if(w.model!=BoundaryLayerModel::ReactingSst){
        error="unsupported finite-rate boundaryLayer implementation";return false;}
    error.clear();return true;
}
struct ModelConfig {
    PhysicsConfig physics{};
    std::array<std::string,Ns> speciesNames{};
    std::array<std::string,Nc> condensedNames{};
    std::array<std::string,Ne> elementNames{};
    std::string modelName{},materialSource{},mechanismSource{},sourceFingerprint{},baseCommit{};
    std::string gasMechanismSource{},gasMechanismSourceSha256{},gasPhase{};
    std::string modelFingerprint{},speciesFingerprint{},schemaFingerprint{};
    ModelConfig() { for(int s=0;s<Ns;++s)speciesNames[s]="S"+std::to_string(s); }
};
template<class Names>
inline std::uint64_t speciesOrderHash(const Names& names) {
    std::uint64_t hash=14695981039346656037ULL;
    for(const auto& name:names){for(unsigned char byte:name){hash^=byte;hash*=1099511628211ULL;}hash*=1099511628211ULL;}
    return hash;
}
inline bool validateSpeciesSet(const ModelConfig& config,std::string& error) {
    for(int s=0;s<Ns;++s){
        if(config.speciesNames[s].empty()){error="empty runtime gas species name";return false;}
        for(int j=0;j<s;++j)if(config.speciesNames[s]==config.speciesNames[j]){error="duplicate runtime gas species name";return false;}
    }
    if(config.physics.speciesFingerprint!=speciesOrderHash(config.speciesNames)){
        error="configuration runtime species name/order hash mismatch";return false;
    }
    error.clear();return true;
}
static_assert(std::is_trivially_copyable<PhysicsConfig>::value,"PhysicsConfig must be copied to CUDA without STL");
} // namespace chmt
#endif
