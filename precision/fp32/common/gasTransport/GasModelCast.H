#pragma once
#include "MolecularViscosity.H"
#include "../chemistry/ChemistryTypes.H"
namespace ugkwp {
template<class D,class S> struct GasModelCaster { static D cast(const S& s){return static_cast<D>(s);} };
template<class D,class S> struct GasModelCaster<SpeciesThermoData<D>,SpeciesThermoData<S>> { static SpeciesThermoData<D> cast(const SpeciesThermoData<S>& s){ SpeciesThermoData<D> d;
 d.model=s.model;
 d.coefficientOffset=s.coefficientOffset;
 d.molarMass=s.molarMass;
 d.minTemperature=s.minTemperature;
 d.midTemperature=s.midTemperature;
 d.maxTemperature=s.maxTemperature;
 d.referencePressure=s.referencePressure;
 d.referenceTemperature=s.referenceTemperature;
 d.referenceEntropy=s.referenceEntropy;
 d.hasEntropyReference=s.hasEntropyReference;
 return d; } };
template<class D,class S> struct GasModelCaster<GasArrheniusData<D>,GasArrheniusData<S>> { static GasArrheniusData<D> cast(const GasArrheniusData<S>& s){ GasArrheniusData<D> d;
 d.preExponential=s.preExponential;
 d.temperatureExponent=s.temperatureExponent;
 d.activationEnergy=s.activationEnergy;
 return d; } };
template<class D,class S> struct GasModelCaster<GasTroeData<D>,GasTroeData<S>> { static GasTroeData<D> cast(const GasTroeData<S>& s){ GasTroeData<D> d;
 d.alpha=s.alpha;
 d.T3=s.T3;
 d.T1=s.T1;
 d.T2=s.T2;
 d.hasT2=s.hasT2;
 return d; } };
template<class D,class S> struct GasModelCaster<GasStoichTerm<D>,GasStoichTerm<S>> { static GasStoichTerm<D> cast(const GasStoichTerm<S>& s){ GasStoichTerm<D> d;
 d.species=s.species;
 d.coefficient=s.coefficient;
 return d; } };
template<class D,class S> struct GasModelCaster<GasColliderEfficiency<D>,GasColliderEfficiency<S>> { static GasColliderEfficiency<D> cast(const GasColliderEfficiency<S>& s){ GasColliderEfficiency<D> d;
 d.species=s.species;
 d.efficiency=s.efficiency;
 return d; } };
template<class D,class S> struct GasModelCaster<SutherlandSpecies<D>,SutherlandSpecies<S>> { static SutherlandSpecies<D> cast(const SutherlandSpecies<S>& s){ SutherlandSpecies<D> d;
 d.muRef=s.muRef;
 d.TRef=s.TRef;
 d.S=s.S;
 d.molarMass=s.molarMass;
 return d; } };
template<class D,class S> struct GasModelCaster<PowerLawDiffusivity<D>,PowerLawDiffusivity<S>> { static PowerLawDiffusivity<D> cast(const PowerLawDiffusivity<S>& s){ PowerLawDiffusivity<D> d;
 d.DRef=s.DRef;
 d.TRef=s.TRef;
 d.pRef=s.pRef;
 d.temperatureExponent=s.temperatureExponent;
 d.pressureExponent=s.pressureExponent;
 return d; } };
template<class D,class S> struct GasModelCaster<GasReactionData<D>,GasReactionData<S>> { static GasReactionData<D> cast(const GasReactionData<S>& s){ GasReactionData<D> d;
 d.type=s.type;
 d.reversible=s.reversible;
 d.duplicate=s.duplicate;
 d.sourceIndex=s.sourceIndex;
 d.reactantOffset=s.reactantOffset;
 d.reactantCount=s.reactantCount;
 d.productOffset=s.productOffset;
 d.productCount=s.productCount;
 d.efficiencyOffset=s.efficiencyOffset;
 d.efficiencyCount=s.efficiencyCount;
 d.defaultEfficiency=s.defaultEfficiency;
 d.highRate=GasModelCaster<GasArrheniusData<D>,GasArrheniusData<S>>::cast(s.highRate);
 d.lowRate=GasModelCaster<GasArrheniusData<D>,GasArrheniusData<S>>::cast(s.lowRate);
 d.troe=GasModelCaster<GasTroeData<D>,GasTroeData<S>>::cast(s.troe);
 return d; } };
template<class D,class S> inline D castGasModel(const S& s){return GasModelCaster<D,S>::cast(s);}
template<class D,class S,int N> inline SpeciesThermoView<D,N> castSpeciesThermoViewMetadata(const SpeciesThermoView<S,N>& s){ SpeciesThermoView<D,N> d;
 d.coefficientCount=s.coefficientCount;
 d.elementCount=s.elementCount;
 d.speciesOrderHash=s.speciesOrderHash;
 d.thermoHash=s.thermoHash;
 d.diagnostics=s.diagnostics;
 return d; }
template<class D,class S,int N> inline GasMechanismView<D,N> castGasMechanismViewMetadata(const GasMechanismView<S,N>& s){ GasMechanismView<D,N> d;
 d.reactionCount=s.reactionCount;
 d.reactantTermCount=s.reactantTermCount;
 d.productTermCount=s.productTermCount;
 d.efficiencyCount=s.efficiencyCount;
 d.independentRank=s.independentRank;
 d.referencePressure=s.referencePressure;
 d.speciesOrderHash=s.speciesOrderHash;
 d.mechanismHash=s.mechanismHash;
 return d; }
template<class D,class S> inline MolecularViscosityView<D> castMolecularViscosityMetadata(const MolecularViscosityView<S>& s){ MolecularViscosityView<D> d;
 d.model=s.model;d.count=s.count;d.compositionSpecies=s.compositionSpecies;d.compositionTRef=s.compositionTRef;d.prandtlSpecies=s.prandtlSpecies;d.diffusionCount=s.diffusionCount;
 for(int i=0;i<3;++i){d.muCoefficients[i]=s.muCoefficients[i];d.sCoefficients[i]=s.sCoefficients[i];d.prandtlCoefficients[i]=s.prandtlCoefficients[i];} return d; }
}
