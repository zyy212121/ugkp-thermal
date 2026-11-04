#ifndef CHMT_CONFIGURATION_SHAREDGASMODEL_H
#define CHMT_CONFIGURATION_SHAREDGASMODEL_H
#include "core/Configuration.H"
#include "../../../common/gasTransport/GasModelIO.H"
namespace chmt {
// Canonical host tables are copied into checkpoint-safe CHMT POD metadata. All
// evaluations still call common MixtureThermo; no caloric formulas live here.
inline bool bindSharedGasModel(const ugkwp::GasModelConfiguration& shared,
    ModelConfig& model,std::string& error) {
    // Validate owning tables before any indexed access or model mutation.
    if(shared.species.size()!=Ns||shared.speciesNames.size()!=Ns
        ||shared.diffusionCoefficients.size()!=Ns){
        error="shared gas species tables must match compiled CHMT species count";return false;
    }
    const std::size_t elements=shared.elementNames.size();
    if(elements>Ne||shared.elementComposition.size()!=std::size_t(Ns)*elements){
        error="shared gas element tables exceed CHMT capacity or have inconsistent dimensions";return false;
    }
    for(const auto& source:shared.species){
        const std::size_t required=source.model==ugkwp::SpeciesThermoModel::LinearCp?3:
            source.model==ugkwp::SpeciesThermoModel::NASA7?14:0;
        if(!required||source.coefficientOffset<0
            ||std::size_t(source.coefficientOffset)>shared.coefficients.size()
            ||required>shared.coefficients.size()-std::size_t(source.coefficientOffset)){
            error="shared gas thermodynamic coefficient range or model is invalid";return false;
        }
    }
    auto& physics=model.physics;
    physics.gasMode=shared.mode;physics.gasThermoFingerprint=shared.thermoHash;
    physics.gasChemistryControls=shared.chemistryControls;
    physics.sst.turbulentSchmidt=shared.turbulentSchmidt;
    model.gasMechanismSource=shared.mechanism;model.gasPhase=shared.phase;
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
        model.speciesNames[s]=shared.speciesNames[s];
        physics.gasDiffusivity[s]=shared.diffusionCoefficients[s];
    }
    error.clear();return true;
}
// Single transport retains all real material species metadata but transports
// exactly one active gas. There is no padded or silently collapsed mixture.
inline bool bindSingleGasModel(const ugkwp::GasModelConfiguration& materialGas,
    const std::string& activeName,ModelConfig& model,std::string& error) {
    ModelConfig candidate=model;
    if(!bindSharedGasModel(materialGas,candidate,error))return false;
    int active=-1;for(int s=0;s<Ns;++s)if(candidate.speciesNames[s]==activeName)active=s;
    if(active<0){error="singleGasSpecies is absent from materialGasProperties";return false;}
    candidate.physics.gasMode=ugkwp::GasMode::SingleLegacy;candidate.physics.singleGasSpecies=active;
    model=std::move(candidate);error.clear();return true;
}

}
#endif
