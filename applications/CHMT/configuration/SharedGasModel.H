#ifndef CHMT_CONFIGURATION_SHAREDGASMODEL_H
#define CHMT_CONFIGURATION_SHAREDGASMODEL_H
#include "core/Configuration.H"
#include "../../../common/gasTransport/GasModelIO.H"
namespace chmt {
// Canonical host tables are copied into checkpoint-safe CHMT POD metadata. All
// evaluations still call common MixtureThermo; no caloric formulas live here.
inline bool bindSharedGasModel(const ugkwp::GasModelConfiguration& shared,
    ModelConfig& model,std::string& error) {
    auto fail=[&](const char* message){error=message;return false;};
    if(shared.mode==ugkwp::GasMode::SingleLegacy)
        return fail("CHMT single-gas binding requires the common legacy thermophysical adapter, which is not wired yet");
    if(shared.mode!=ugkwp::GasMode::MixtureFrozen&&shared.mode!=ugkwp::GasMode::MixtureChemistry)
        return fail("invalid shared gas mode");
    if(shared.elementComposition.size()!=shared.species.size()*shared.elementNames.size())
        return fail("shared gas element table dimensions differ");
    if(shared.speciesNames.size()!=Ns||shared.species.size()!=Ns
        ||shared.diffusionCoefficients.size()!=Ns||shared.elementNames.size()>Ne)
        return fail("shared gas model dimensions differ from the compiled CHMT contract");
    if(shared.speciesOrderHash!=speciesOrderHash(shared.speciesNames))
        return fail("shared gas model runtime species name/order hash mismatch");
    if(model.physics.gasThermoFingerprint!=0&&model.physics.speciesFingerprint!=shared.speciesOrderHash)
        return fail("configured gas species identity is immutable");
    const auto thermo=shared.thermoView<Ns>();
    if(ugkwp::validateSpeciesThermoView(thermo)!=ugkwp::ThermoStatus::Success)
        return fail("shared gas model contains invalid thermodynamic tables");
    ModelConfig candidate=model;auto& physics=candidate.physics;
    if(!shared.elementNames.empty()) {
        if(physics.nElements!=static_cast<int>(shared.elementNames.size()))
            return fail("shared gas/material element dimensions differ");
        for(int e=0;e<physics.nElements;++e)if(candidate.elementNames[e]!=shared.elementNames[e])
            return fail("shared gas/material element order differs");
    }
    physics.gasMode=shared.mode;physics.gasThermoFingerprint=shared.thermoHash;
    physics.speciesFingerprint=shared.speciesOrderHash;
    candidate.speciesFingerprint=std::to_string(shared.speciesOrderHash);
    physics.gasChemistryControls=shared.chemistryControls;
    physics.sst.turbulentSchmidt=shared.turbulentSchmidt;
    candidate.gasMechanismSource=shared.mechanism;candidate.gasPhase=shared.phase;
    for(int s=0;s<Ns;++s){
        const auto& source=shared.species[s];auto& target=physics.species[s];
        target=SpeciesThermo{};target.model=source.model;
        target.R=ugkwp::universalGasConstant<Real>()/source.molarMass;
        target.Tmin=source.minTemperature;target.Tmax=source.maxTemperature;target.Tmid=source.midTemperature;
        target.referencePressure=source.referencePressure;target.referenceTemperature=source.referenceTemperature;
        target.referenceEntropy=source.referenceEntropy;target.hasEntropyReference=source.hasEntropyReference;
        const auto* coefficients=shared.coefficients.data()+source.coefficientOffset;
        if(source.model==ugkwp::SpeciesThermoModel::LinearCp){target.cp0=coefficients[0];target.cp1=coefficients[1];target.e0=coefficients[2];}
        else for(int i=0;i<14;++i)target.nasa7[i]=coefficients[i];
        for(int e=0;e<static_cast<int>(shared.elementNames.size());++e)
            target.element[e]=shared.elementComposition[s*shared.elementNames.size()+e]/source.molarMass;
        candidate.speciesNames[s]=shared.speciesNames[s];
        physics.gasDiffusivity[s]=shared.diffusionCoefficients[s];
    }
    model=std::move(candidate);error.clear();return true;
}
// Single transport retains all real material species metadata but transports
// exactly one active gas. There is no padded or silently collapsed mixture.
inline bool bindSingleGasModel(const ugkwp::GasModelConfiguration& materialGas,
    const std::string& activeName,ModelConfig& model,std::string& error) {
    if(materialGas.mode!=ugkwp::GasMode::MixtureFrozen){error="single gas requires frozen canonical materialGasProperties metadata";return false;}
    ModelConfig candidate=model;
    if(!bindSharedGasModel(materialGas,candidate,error))return false;
    int active=-1;for(int s=0;s<Ns;++s)if(candidate.speciesNames[s]==activeName)active=s;
    if(active<0){error="singleGasSpecies is absent from materialGasProperties";return false;}
    const auto& thermo=candidate.physics.species[active];
    if(thermo.model!=ugkwp::SpeciesThermoModel::LinearCp||thermo.cp1!=0||thermo.e0!=0||thermo.cp0<=thermo.R){
        error="legacy single gas requires constant Cp, positive Cv and zero formation-energy offset";return false;}
    for(int s=0;s<Ns;++s)if(candidate.physics.gasDiffusivity[s]!=0){error="single gas metadata cannot enable multicomponent diffusion";return false;}
    candidate.physics.gasMode=ugkwp::GasMode::SingleLegacy;candidate.physics.singleGasSpecies=active;
    model=std::move(candidate);error.clear();return true;
}

}
#endif
