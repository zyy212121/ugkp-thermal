#ifndef UGKWP_GAS_MECHANISM_IO_H
#define UGKWP_GAS_MECHANISM_IO_H

#include "GasModelIO.H"
#include "../chemistry/MechanismValidation.H"
#include <limits>
#include <algorithm>

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
    if (values.size() < 3) throw error("'" + key + "' requires (A b Ea) in SI molar units");
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
    const std::string& expectedPhase,
    const std::vector<std::string>& thermoSpeciesNames = {}
)
{
    using namespace gasModelIoDetail;
    Tokens tokens(text);
    const Dictionary dict = tokens.dictionary();
    (void)expectedPhase;
    (void)thermo;
    GasMechanismConfiguration model;
    if (dict.count("phase")) model.phase = word(dict, "phase");
    model.speciesNames = list(dict, "species");


    std::vector<int> speciesMap(thermoSpeciesNames.empty()
        ? std::max(model.speciesNames.size(),static_cast<std::size_t>(Ns))
        : model.speciesNames.size(), -1);
    if (thermoSpeciesNames.empty())
    {
        for (std::size_t s=0; s<speciesMap.size(); ++s) speciesMap[s]=static_cast<int>(s);
    }
    else
    {
        std::vector<bool> used(thermoSpeciesNames.size(), false);
        for (std::size_t s=0; s<model.speciesNames.size(); ++s)
            for (std::size_t t=0; t<thermoSpeciesNames.size(); ++t)
                if (!used[t] && model.speciesNames[s]==thermoSpeciesNames[t])
                {speciesMap[s]=static_cast<int>(t);used[t]=true;break;}
    }
    model.referencePressure = scalar(dict, "referencePressure");
    model.independentRank = integer(scalar(dict, "independentRank"), "independentRank");
    if (model.independentRank > Ns) throw error("independentRank exceeds species count");
    model.stoichiometricBasis = numbers(dict, "stoichiometricBasis");
    const Dictionary reactionDict = subdict(dict, "reactions");
    std::vector<std::string> reactionNames;
    std::map<std::string,bool> orderedNames;

    for (std::size_t index=0; index<reactionDict.size(); ++index)
    {
        const std::string name="reaction"+std::to_string(index);
        if (reactionDict.count(name)) {reactionNames.push_back(name);orderedNames[name]=true;}
    }
    for (const auto& entry : reactionDict)
        if (!orderedNames.count(entry.first))
            reactionNames.push_back(entry.first);
    for (const std::string& name : reactionNames)
    {
        const Dictionary data = subdict(reactionDict, name);
        GasReactionData<double> reaction;
        const std::string type = word(data, "type");
        if (type == "elementary") reaction.type = GasReactionType::Elementary;
        else if (type == "thirdBody") reaction.type = GasReactionType::ThirdBody;
        else if (type == "Lindemann") reaction.type = GasReactionType::Lindemann;
        else if (type == "Troe") reaction.type = GasReactionType::Troe;
        else throw error("unsupported reaction type '" + type + "' in " + name);
        reaction.reversible = boolean(data, "reversible");
        if (data.count("duplicate")) reaction.duplicate = boolean(data, "duplicate");
        if (data.count("sourceIndex")) reaction.sourceIndex = integer(scalar(data, "sourceIndex"), "sourceIndex", std::numeric_limits<int>::min());
        reaction.highRate = arrhenius(data, "highRate");
        if (reaction.type==GasReactionType::Lindemann || reaction.type==GasReactionType::Troe)
            reaction.lowRate = arrhenius(data, "lowRate");
        if (reaction.type!=GasReactionType::Elementary)
            reaction.defaultEfficiency = scalar(data, "defaultEfficiency");
        if (reaction.type==GasReactionType::Troe)
        {
            const auto troe = numbers(data, "troe");
            reaction.troe.hasT2 = boolean(data, "troeHasT2");
            if (troe.size() < (reaction.troe.hasT2 ? 4U : 3U))
                throw error("troe is missing alpha, T3, T1 or the selected T2 term");
            reaction.troe.alpha=troe[0];reaction.troe.T3=troe[1];reaction.troe.T1=troe[2];
            if (reaction.troe.hasT2) reaction.troe.T2=troe[3];
        }
        reaction.reactantOffset = static_cast<int>(model.reactants.size());
        stoich(data, "reactants", model.reactants, static_cast<int>(speciesMap.size()));
        reaction.reactantCount = static_cast<int>(model.reactants.size())-reaction.reactantOffset;
        reaction.productOffset = static_cast<int>(model.products.size());
        stoich(data, "products", model.products, static_cast<int>(speciesMap.size()));
        reaction.productCount = static_cast<int>(model.products.size())-reaction.productOffset;
        reaction.efficiencyOffset = static_cast<int>(model.efficiencies.size());
        const auto efficiencies = reaction.type!=GasReactionType::Elementary
            ? numbers(data, "efficiencies") : std::vector<double>{};
        if (efficiencies.size()%2) throw error("efficiencies must contain flat species/efficiency pairs");
        for (std::size_t i=0; i<efficiencies.size(); i+=2)
        {
            GasColliderEfficiency<double> term;
            term.species = integer(efficiencies[i], "efficiencies");
            if (static_cast<std::size_t>(term.species) >= speciesMap.size()) throw error("efficiency species index outside mechanism order");
            term.efficiency = efficiencies[i+1];
            model.efficiencies.push_back(term);
        }
        reaction.efficiencyCount = static_cast<int>(model.efficiencies.size())-reaction.efficiencyOffset;
        model.reactions.push_back(reaction);
    }
    for (auto* terms : {&model.reactants, &model.products}) for (auto& term : *terms)
    {
        const int mapped=speciesMap[term.species];
        if (mapped<0) throw error("reaction uses a species absent from the compiled thermo model");
        term.species=mapped;
    }
    for (auto& term : model.efficiencies)
    {
        const int mapped=speciesMap[term.species];
        if (mapped<0) throw error("collider uses a species absent from the compiled thermo model");
        term.species=mapped;
    }


    if (model.independentRank)
    {
        const auto sourceBasis=model.stoichiometricBasis;
        const std::size_t rank=model.independentRank;
        if (sourceBasis.size()<static_cast<std::size_t>(Ns)*rank)
            throw error("stoichiometricBasis is missing required compiled rows");
        const std::size_t rows=std::min(speciesMap.size(),sourceBasis.size()/rank);
        std::vector<bool> usedRows(rows,false);
        std::size_t nextRow=0;
        const auto appendRow=[&](std::size_t source)
        {
            for (std::size_t j=0;j<rank;++j)
                model.stoichiometricBasis[nextRow*rank+j]=sourceBasis[source*rank+j];
            usedRows[source]=true;
            ++nextRow;
        };
        const std::size_t targets=std::max(static_cast<std::size_t>(Ns),thermoSpeciesNames.size());
        for (std::size_t target=0;target<targets;++target)
        {
            const auto found=std::find(speciesMap.begin(),speciesMap.end(),static_cast<int>(target));
            const std::size_t source=found-speciesMap.begin();
            if (found==speciesMap.end() || source>=rows)
            {
                if (target<static_cast<std::size_t>(Ns))
                    throw error("stoichiometricBasis is missing the mapped species row");
                continue;
            }
            appendRow(source);
        }

        for (std::size_t source=0;source<rows;++source)
            if (!usedRows[source]) appendRow(source);
    }
    model.speciesOrderHash=speciesIdentity(thermoSpeciesNames.empty()?model.speciesNames:thermoSpeciesNames);
    model.mechanismHash=mechanismIdentity(model);
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
    const std::string& expectedPhase,
    const std::vector<std::string>& thermoSpeciesNames = {}
)
{
    std::ifstream input(path);
    if (!input) throw gasModelIoDetail::error("cannot read required mechanism '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw gasModelIoDetail::error("failed reading mechanism '" + path + "'");
    return parseGasMechanismProperties<Ns>(text, thermo, expectedPhase, thermoSpeciesNames);
}
} // namespace ugkwp
#endif
