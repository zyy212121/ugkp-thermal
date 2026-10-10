#ifndef UGKWP_GAS_MECHANISM_IO_H
#define UGKWP_GAS_MECHANISM_IO_H

#include "GasModelIO.H"
#include "../chemistry/MechanismValidation.H"
#include <limits>

namespace ugkwp
{
// Host-owned canonical SI tables. A view is rebuilt against each host/device
// allocation; neither this object nor any borrowed pointer is an IPC payload.
struct GasMechanismConfiguration
{
    std::string phase;
    std::string sourceSha256;
    std::vector<std::string> speciesNames;
    std::vector<GasReactionData<double>> reactions;
    std::vector<GasStoichTerm<double>> reactants;
    std::vector<GasStoichTerm<double>> products;
    std::vector<GasColliderEfficiency<double>> efficiencies;
    std::vector<double> stoichiometricBasis;
    int independentRank = 0;
    double referencePressure = 0;
    std::uint64_t speciesOrderHash = 0;
    std::uint64_t mechanismHash = 0;

    template<int Ns>
    GasMechanismView<double, Ns> mechanismView() const
    {
        if (speciesNames.size() != static_cast<std::size_t>(Ns))
            throw std::invalid_argument("mechanism species count differs from compiled backend Ns");
        GasMechanismView<double, Ns> view;
        view.reactions = reactions.data();
        view.reactionCount = static_cast<int>(reactions.size());
        view.reactants = reactants.data();
        view.reactantTermCount = static_cast<int>(reactants.size());
        view.products = products.data();
        view.productTermCount = static_cast<int>(products.size());
        view.efficiencies = efficiencies.data();
        view.efficiencyCount = static_cast<int>(efficiencies.size());
        view.stoichiometricBasis = stoichiometricBasis.data();
        view.independentRank = independentRank;
        view.referencePressure = referencePressure;
        view.speciesOrderHash = speciesOrderHash;
        view.mechanismHash = mechanismHash;
        return view;
    }
};

namespace gasModelIoDetail
{
inline std::uint64_t unsigned64(const Dictionary& dict, const std::string& key)
{
    const std::string value = word(dict, key);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw error("'" + key + "' must be an unsigned decimal 64-bit integer");
    try { return static_cast<std::uint64_t>(std::stoull(value)); }
    catch (const std::exception&) { throw error("'" + key + "' exceeds unsigned 64-bit range"); }
}
inline bool boolean(const Dictionary& dict, const std::string& key)
{
    const std::string value = word(dict, key);
    if (value == "true") return true;
    if (value == "false") return false;
    throw error("'" + key + "' must be true or false");
}
inline GasArrheniusData<double> arrhenius(const Dictionary& dict, const std::string& key)
{
    const auto values = numbers(dict, key);
    if (values.size() != 3) throw error("'" + key + "' requires (A b Ea) in SI molar units");
    GasArrheniusData<double> rate;
    rate.preExponential = values[0];
    rate.temperatureExponent = values[1];
    rate.activationEnergy = values[2];
    return rate;
}
inline void stoich(const Dictionary& dict, const std::string& key, std::vector<GasStoichTerm<double>>& terms, int speciesCount)
{
    const auto values = numbers(dict, key);
    if (values.size()%2) throw error("'" + key + "' must contain flat species/coefficient pairs");
    for (std::size_t i=0; i<values.size(); i+=2)
    {
        GasStoichTerm<double> term;
        term.species = integer(values[i], key);
        if (term.species >= speciesCount) throw error("species index outside mechanism order in '" + key + "'");
        term.coefficient = values[i+1];
        terms.push_back(term);
    }
}
inline std::uint64_t mechanismIdentity(const GasMechanismConfiguration& model)
{
    std::uint64_t hash = model.speciesOrderHash;
    hashScalar(hash, model.referencePressure);
    hashScalar(hash, model.independentRank);
    for (double value : model.stoichiometricBasis) hashScalar(hash, value);
    for (const auto& reaction : model.reactions)
    {
        hashByte(hash, static_cast<unsigned char>(reaction.type));
        hashByte(hash, reaction.reversible ? 1 : 0);
        hashByte(hash, reaction.duplicate ? 1 : 0);
        hashScalar(hash, reaction.sourceIndex);
        for (const auto& rate : {reaction.highRate, reaction.lowRate})
        {
            hashScalar(hash, rate.preExponential);
            hashScalar(hash, rate.temperatureExponent);
            hashScalar(hash, rate.activationEnergy);
        }
        hashScalar(hash, reaction.defaultEfficiency);
        hashScalar(hash, reaction.troe.alpha);
        hashScalar(hash, reaction.troe.T3);
        hashScalar(hash, reaction.troe.T1);
        hashScalar(hash, reaction.troe.T2);
        hashByte(hash, reaction.troe.hasT2 ? 1 : 0);
        hashScalar(hash, reaction.reactantCount);
        for (int i=0; i<reaction.reactantCount; ++i)
        {
            const auto& term = model.reactants[reaction.reactantOffset+i];
            hashScalar(hash, term.species); hashScalar(hash, term.coefficient);
        }
        hashScalar(hash, reaction.productCount);
        for (int i=0; i<reaction.productCount; ++i)
        {
            const auto& term = model.products[reaction.productOffset+i];
            hashScalar(hash, term.species); hashScalar(hash, term.coefficient);
        }
        hashScalar(hash, reaction.efficiencyCount);
        for (int i=0; i<reaction.efficiencyCount; ++i)
        {
            const auto& term = model.efficiencies[reaction.efficiencyOffset+i];
            hashScalar(hash, term.species); hashScalar(hash, term.efficiency);
        }
    }
    return hash;
}
} // namespace gasModelIoDetail

template<int Ns>
inline GasMechanismConfiguration parseGasMechanismProperties
(
    const std::string& text,
    const SpeciesThermoView<double, Ns>& thermo,
    const std::string& expectedPhase
)
{
    using namespace gasModelIoDetail;
    Tokens tokens(text);
    const Dictionary dict = tokens.dictionary();
    allowed(dict, {"FoamFile", "schemaVersion", "units", "phase", "species", "referencePressure", "speciesOrderHash", "mechanismHash", "sourceSha256", "independentRank", "stoichiometricBasis", "reactions"}, "mechanism");
    if (scalar(dict, "schemaVersion") != 1) throw error("mechanism requires schemaVersion 1");
    if (word(dict, "units") != "siMolar") throw error("mechanism units must be siMolar (mol, m, s, kg, J, K, Pa)");
    GasMechanismConfiguration model;
    model.phase = word(dict, "phase");
    if (expectedPhase.empty() || model.phase != expectedPhase) throw error("mechanism phase does not match the explicitly selected phase");
    model.speciesNames = list(dict, "species");
    model.speciesOrderHash = unsigned64(dict, "speciesOrderHash");
    if (model.speciesNames.size() != static_cast<std::size_t>(Ns)
        || speciesIdentity(model.speciesNames) != model.speciesOrderHash
        || model.speciesOrderHash != thermo.speciesOrderHash)
        throw error("mechanism species count/order/hash does not match the thermodynamic model");
    std::set<std::string> unique;
    for (const auto& name : model.speciesNames)
        if (!unique.insert(name).second) throw error("duplicate mechanism species '"+name+"'");
    model.referencePressure = scalar(dict, "referencePressure");
    model.mechanismHash = unsigned64(dict, "mechanismHash");
    if (dict.count("sourceSha256"))
    {
        model.sourceSha256 = word(dict, "sourceSha256");
        if (model.sourceSha256.size() != 64 || model.sourceSha256.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw error("sourceSha256 must contain 64 lowercase hexadecimal characters");
    }
    model.independentRank = integer(scalar(dict, "independentRank"), "independentRank", 1);
    if (model.independentRank > Ns) throw error("independentRank exceeds species count");
    model.stoichiometricBasis = numbers(dict, "stoichiometricBasis");
    if (model.stoichiometricBasis.size() != static_cast<std::size_t>(Ns*model.independentRank))
        throw error("stoichiometricBasis must have Ns*independentRank entries in species-major order");
    const Dictionary reactionDict = subdict(dict, "reactions");
    if (reactionDict.empty()) throw error("enabled chemistry requires a nonempty mechanism");
    for (std::size_t index=0; index<reactionDict.size(); ++index)
    {
        const std::string name = "reaction"+std::to_string(index);
        const Dictionary data = subdict(reactionDict, name);
        allowed(data, {"type", "reversible", "duplicate", "sourceIndex", "reactants", "products", "highRate", "lowRate", "defaultEfficiency", "efficiencies", "troe", "troeHasT2"}, name);
        GasReactionData<double> reaction;
        const std::string type = word(data, "type");
        if (type == "elementary") reaction.type = GasReactionType::Elementary;
        else if (type == "thirdBody") reaction.type = GasReactionType::ThirdBody;
        else if (type == "Lindemann") reaction.type = GasReactionType::Lindemann;
        else if (type == "Troe") reaction.type = GasReactionType::Troe;
        else throw error("unsupported reaction type '" + type + "' in " + name);
        reaction.reversible = boolean(data, "reversible");
        reaction.duplicate = boolean(data, "duplicate");
        reaction.sourceIndex = integer(scalar(data, "sourceIndex"), "sourceIndex");
        if (reaction.sourceIndex != static_cast<int>(index)) throw error("sourceIndex must match canonical contiguous reaction order");
        reaction.highRate = arrhenius(data, "highRate");
        reaction.lowRate = arrhenius(data, "lowRate");
        reaction.defaultEfficiency = scalar(data, "defaultEfficiency");
        const auto troe = numbers(data, "troe");
        if (troe.size() != 4) throw error("troe requires (alpha T3 T1 T2)");
        reaction.troe.alpha=troe[0]; reaction.troe.T3=troe[1]; reaction.troe.T1=troe[2]; reaction.troe.T2=troe[3];
        reaction.troe.hasT2 = boolean(data, "troeHasT2");
        reaction.reactantOffset = static_cast<int>(model.reactants.size());
        stoich(data, "reactants", model.reactants, Ns);
        reaction.reactantCount = static_cast<int>(model.reactants.size())-reaction.reactantOffset;
        reaction.productOffset = static_cast<int>(model.products.size());
        stoich(data, "products", model.products, Ns);
        reaction.productCount = static_cast<int>(model.products.size())-reaction.productOffset;
        reaction.efficiencyOffset = static_cast<int>(model.efficiencies.size());
        const auto efficiencies = numbers(data, "efficiencies");
        if (efficiencies.size()%2) throw error("efficiencies must contain flat species/efficiency pairs");
        for (std::size_t i=0; i<efficiencies.size(); i+=2)
        {
            GasColliderEfficiency<double> term;
            term.species = integer(efficiencies[i], "efficiencies");
            if (term.species >= Ns) throw error("efficiency species index outside mechanism order");
            term.efficiency = efficiencies[i+1];
            model.efficiencies.push_back(term);
        }
        reaction.efficiencyCount = static_cast<int>(model.efficiencies.size())-reaction.efficiencyOffset;
        model.reactions.push_back(reaction);
    }
    if (mechanismIdentity(model) != model.mechanismHash) throw error("mechanismHash does not match canonical SI tables");
    const ChemistryStatus status = validateGasMechanismView(thermo, model.mechanismView<Ns>());
    if (!status) throw error("mechanism physical validation failed at reaction " + std::to_string(status.reaction));
    return model;
}

inline std::string resolveGasMechanismPath
(
    const GasModelConfiguration& model,
    const std::string& constantDirectory
)
{
    if (model.mode != GasMode::MixtureChemistry || model.mechanism.empty())
        throw gasModelIoDetail::error("a mechanism path requires gasMode mixtureChemistry");
    if (model.mechanism.front() == '/') return model.mechanism;
    if (constantDirectory.empty()) throw gasModelIoDetail::error("case constant directory is empty");
    return constantDirectory + (constantDirectory.back() == '/' ? "" : "/") + model.mechanism;
}

template<int Ns>
inline GasMechanismConfiguration readGasMechanismProperties
(
    const std::string& path,
    const SpeciesThermoView<double, Ns>& thermo,
    const std::string& expectedPhase
)
{
    std::ifstream input(path);
    if (!input) throw gasModelIoDetail::error("cannot read required mechanism '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw gasModelIoDetail::error("failed reading mechanism '" + path + "'");
    return parseGasMechanismProperties<Ns>(text, thermo, expectedPhase);
}
} // namespace ugkwp
#endif
