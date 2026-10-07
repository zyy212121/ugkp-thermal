#ifndef CHMT_CORE_CONFIGURATION_H
#define CHMT_CORE_CONFIGURATION_H
#include "Types.H"
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
struct PhysicsConfig {
    bool enableGas=true;
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
    std::uint64_t modelFingerprint=0,speciesFingerprint=compiledSpeciesHash,schemaFingerprint=checkpointSchema;
};
struct ModelConfig {
    PhysicsConfig physics{};
    std::array<std::string,Ns> speciesNames{};
    std::array<std::string,Nc> condensedNames{};
    std::array<std::string,Ne> elementNames{};
    std::string modelName{},materialSource{},mechanismSource{},sourceFingerprint{},baseCommit{};
    std::string modelFingerprint{},speciesFingerprint{},schemaFingerprint{};
    ModelConfig() { for(int s=0;s<Ns;++s)speciesNames[s]=compiledSpeciesNames[s]; }
};
inline bool validateSpeciesSet(const ModelConfig& config,std::string& error) {
    if(config.physics.speciesFingerprint!=compiledSpeciesHash){error="configuration species fingerprint does not match compiled species set";return false;}
    for(int s=0;s<Ns;++s)if(config.speciesNames[s]!=compiledSpeciesNames[s]){
        error="configuration species "+std::to_string(s)+" name/order mismatch: expected "+compiledSpeciesNames[s];return false;
    }
    error.clear();return true;
}
static_assert(std::is_trivially_copyable<PhysicsConfig>::value,"PhysicsConfig must be copied to CUDA without STL");
} // namespace chmt
#endif
