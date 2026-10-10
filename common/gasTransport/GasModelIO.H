#ifndef UGKWP_GAS_MODEL_IO_H
#define UGKWP_GAS_MODEL_IO_H

// Host-only constant/gasModelProperties parser, shared by all application shells.
// This owning object is never passed to a kernel or serialized as a raw struct.
#include "GasModel.H"
#include "MixtureThermo.H"
#include "../chemistry/ChemistryTypes.H"
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace ugkwp
{

enum class GasDiffusionModel { None, Constant };

struct GasModelConfiguration
{
    GasMode mode = GasMode::SingleLegacy;
    std::vector<std::string> speciesNames;
    std::vector<SpeciesThermoData<double>> species;
    std::vector<double> coefficients;
    std::vector<std::string> elementNames;
    std::vector<double> elementComposition;
    GasDiffusionModel diffusionModel = GasDiffusionModel::None;
    std::vector<double> diffusionCoefficients; // m2/s, in the exact species order
    double turbulentSchmidt = 0.7;
    ChemistryControls<double> chemistryControls;
    std::string mechanism;
    std::string phase;
    std::uint64_t speciesOrderHash = 0;
    std::uint64_t thermoHash = 0;

    template<int Ns>
    SpeciesThermoView<double, Ns> thermoView() const
    {
        if (species.size() != static_cast<std::size_t>(Ns))
            throw std::invalid_argument("gas model species count differs from compiled backend Ns");
        SpeciesThermoView<double, Ns> view;
        view.species = species.data();
        view.coefficients = coefficients.data();
        view.coefficientCount = static_cast<int>(coefficients.size());
        view.elementCount = static_cast<int>(elementNames.size());
        view.elementComposition = elementComposition.empty() ? nullptr : elementComposition.data();
        view.speciesOrderHash = speciesOrderHash;
        view.thermoHash = thermoHash;
        return view;
    }
};

namespace gasModelIoDetail
{
struct Entry
{
    std::vector<std::string> values;
    std::map<std::string, Entry> children;
    bool dictionary = false;
    bool list = false;
};
using Dictionary = std::map<std::string, Entry>;

inline std::runtime_error error(const std::string& message)
{
    return std::runtime_error("constant/gasModelProperties: " + message);
}

class Tokens
{
    std::vector<std::string> tokens_;
    std::size_t position_ = 0;
public:
    explicit Tokens(const std::string& text)
    {
        std::size_t i = 0;
        while (i < text.size())
        {
            const unsigned char ch = static_cast<unsigned char>(text[i]);
            if (std::isspace(ch)) { ++i; continue; }
            if (text.compare(i, 2, "//") == 0)
            {
                i = text.find('\n', i + 2);
                if (i == std::string::npos) break;
                continue;
            }
            if (text.compare(i, 2, "/*") == 0)
            {
                const std::size_t end = text.find("*/", i + 2);
                if (end == std::string::npos) throw error("unterminated comment");
                i = end + 2; continue;
            }
            if (std::strchr("{}();", ch))
            { tokens_.push_back(text.substr(i++, 1)); continue; }
            if (ch == '"')
            {
                std::string value;
                ++i;
                while (i < text.size() && text[i] != '"')
                {
                    if (text[i] == '\\')
                    {
                        ++i;
                        if (i == text.size()) throw error("unterminated quoted value");
                    }
                    value += text[i++];
                }
                if (i == text.size()) throw error("unterminated quoted value");
                ++i; tokens_.push_back(value); continue;
            }
            if (ch == '#' || ch == '$')
                throw error("preprocessor directives require an expanded host dictionary");
            const std::size_t begin = i;
            while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))
                && !std::strchr("{}();", text[i])
                && text.compare(i, 2, "//") != 0 && text.compare(i, 2, "/*") != 0)
                ++i;
            tokens_.push_back(text.substr(begin, i - begin));
        }
    }
    bool empty() const { return position_ == tokens_.size(); }
    std::string peek() const { return empty() ? std::string() : tokens_[position_]; }
    std::string take()
    {
        if (empty()) throw error("unexpected end of dictionary");
        return tokens_[position_++];
    }
    Dictionary dictionary(bool nested = false)
    {
        Dictionary result;
        while (!empty() && peek() != "}")
        {
            const std::string key = take();
            if (key == ";" || key == "(" || key == ")" || key == "{")
                throw error("invalid dictionary key '" + key + "'");
            Entry entry;
            if (peek() == "{")
            {
                take(); entry.dictionary = true; entry.children = dictionary(true);
                if (peek() == ";") take();
            }
            else if (peek() == "(")
            {
                take(); entry.list = true;
                while (!empty() && peek() != ")")
                {
                    const std::string value = take();
                    if (value == "(" || value == "{" || value == "}" || value == ";")
                        throw error("nested or malformed list for '" + key + "'");
                    entry.values.push_back(value);
                }
                if (take() != ")" || take() != ";") throw error("missing list terminator for '" + key + "'");
            }
            else
            {
                while (!empty() && peek() != ";")
                {
                    if (peek() == "}" || peek() == "{" || peek() == "(" || peek() == ")")
                        throw error("missing semicolon for '" + key + "'");
                    entry.values.push_back(take());
                }
                if (take() != ";") throw error("missing semicolon for '" + key + "'");
            }
            if (!result.emplace(key, entry).second) throw error("duplicate key '" + key + "'");
        }
        if (nested)
        {
            if (take() != "}") throw error("unterminated subdictionary");
        }
        else if (!empty()) throw error("unexpected closing brace");
        return result;
    }
};

inline void allowed(const Dictionary& dict, const std::set<std::string>& keys, const std::string& context)
{
    for (const auto& entry : dict)
        if (!keys.count(entry.first)) throw error("unknown " + context + " key '" + entry.first + "'");
}
inline const Entry& required(const Dictionary& dict, const std::string& key)
{
    const auto found = dict.find(key);
    if (found == dict.end()) throw error("required entry '" + key + "' is missing");
    return found->second;
}
inline std::string word(const Dictionary& dict, const std::string& key)
{
    const Entry& entry = required(dict, key);
    if (entry.dictionary || entry.list || entry.values.size() != 1 || entry.values[0].empty())
        throw error("'" + key + "' must be one value");
    return entry.values[0];
}
inline const Dictionary& subdict(const Dictionary& dict, const std::string& key)
{
    const Entry& entry = required(dict, key);
    if (!entry.dictionary) throw error("'" + key + "' must be a dictionary");
    return entry.children;
}
inline const std::vector<std::string>& list(const Dictionary& dict, const std::string& key)
{
    const Entry& entry = required(dict, key);
    if (!entry.list) throw error("'" + key + "' must be a list");
    return entry.values;
}
inline double number(const std::string& value, const std::string& key)
{
    char* end = nullptr;
    errno = 0;
    const double result = std::strtod(value.c_str(), &end);
    if (end != value.c_str() + value.size() || end == value.c_str()
        || errno == ERANGE || !std::isfinite(result))
        throw error("'" + key + "' must contain finite numeric values");
    return result;
}
inline double scalar(const Dictionary& dict, const std::string& key)
{ return number(word(dict, key), key); }
inline std::vector<double> numbers(const Dictionary& dict, const std::string& key)
{
    std::vector<double> result;
    for (const std::string& value : list(dict, key)) result.push_back(number(value, key));
    return result;
}
inline int integer(double value, const std::string& key, int minimum = 0)
{
    if (value < minimum || value > std::numeric_limits<int>::max() || std::floor(value) != value)
        throw error("'" + key + "' must be an in-range integer");
    return static_cast<int>(value);
}
inline void readNumericControls
(
    const Dictionary& dict,
    const std::map<std::string, double*>& reals,
    const std::map<std::string, int*>& integers,
    const std::string& context,
    bool hasThermo = false
)
{
    for (const auto& entry : dict)
    {
        const auto real = reals.find(entry.first);
        const auto whole = integers.find(entry.first);
        if (real != reals.end()) *real->second = scalar(dict, entry.first);
        else if (whole != integers.end()) *whole->second = integer(scalar(dict, entry.first), entry.first);
        else if (!(hasThermo && entry.first == "thermo"))
            throw error("unknown " + context + " key '" + entry.first + "'");
    }
}
inline ChemistryControls<double> chemistryControls(const Dictionary& dict)
{
    ChemistryControls<double> c;
    readNumericControls(dict,
        {{"relativeTolerance", &c.relativeTolerance},
         {"absoluteMassFractionTolerance", &c.absoluteMassFractionTolerance},
         {"absoluteTemperatureTolerance", &c.absoluteTemperatureTolerance},
         {"initialStep", &c.initialStep}, {"minimumStep", &c.minimumStep},
         {"maximumStep", &c.maximumStep}, {"newtonTolerance", &c.newtonTolerance},
         {"absoluteMassTolerance", &c.absoluteMassTolerance},
         {"relativeConservationTolerance", &c.relativeConservationTolerance},
         {"absoluteElementTolerance", &c.absoluteElementTolerance}},
        {{"maximumSteps", &c.maximumSteps}, {"maximumRejectedSteps", &c.maximumRejectedSteps},
         {"maximumNewtonIterations", &c.maximumNewtonIterations},
         {"maximumLineSearchIterations", &c.maximumLineSearchIterations}},
        "chemistryControls", true);
    if (dict.count("thermo"))
    {
        const Dictionary thermo = subdict(dict, "thermo");
        readNumericControls(thermo,
            {{"minTemperature", &c.thermo.minTemperature}, {"maxTemperature", &c.thermo.maxTemperature},
             {"absoluteEnergyTolerance", &c.thermo.absoluteEnergyTolerance},
             {"relativeEnergyTolerance", &c.thermo.relativeEnergyTolerance},
             {"absoluteTemperatureTolerance", &c.thermo.absoluteTemperatureTolerance},
             {"relativeTemperatureTolerance", &c.thermo.relativeTemperatureTolerance}},
            {{"maximumIterations", &c.thermo.maximumIterations}}, "chemistryControls/thermo");
    }
    if (!validateChemistryControls(c)) throw error("invalid chemistryControls");
    return c;
}
inline void hashByte(std::uint64_t& hash, unsigned char value)
{ hash = (hash ^ value)*UINT64_C(1099511628211); }
inline void hashWord(std::uint64_t& hash, const std::string& value)
{
    for (unsigned char ch : value) hashByte(hash, ch);
    hashByte(hash, 0);
}
inline void hashScalar(std::uint64_t& hash, double value)
{
    // Canonical IEEE-754 little-endian byte order, independent of host endian.
    // Signed zero has the same physical identity.
    if (value == 0) value = 0;
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "gas model requires binary64 doubles");
    std::memcpy(&bits, &value, sizeof(value));
    for (int i = 0; i != 8; ++i) hashByte(hash, static_cast<unsigned char>(bits >> (8*i)));
}
inline std::uint64_t speciesIdentity(const std::vector<std::string>& names)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const auto& name : names) hashWord(hash, name);
    return hash;
}
inline void identities(GasModelConfiguration& model)
{
    model.speciesOrderHash = speciesIdentity(model.speciesNames);
    model.thermoHash = model.speciesOrderHash;
    for (const auto& sp : model.species)
    {
        hashByte(model.thermoHash, static_cast<unsigned char>(sp.model));
        hashScalar(model.thermoHash, sp.molarMass);
        hashScalar(model.thermoHash, sp.minTemperature);
        hashScalar(model.thermoHash, sp.midTemperature);
        hashScalar(model.thermoHash, sp.maxTemperature);
        hashScalar(model.thermoHash, sp.referencePressure);
        hashScalar(model.thermoHash, sp.referenceTemperature);
        hashScalar(model.thermoHash, sp.referenceEntropy);
        hashByte(model.thermoHash, sp.hasEntropyReference ? 1 : 0);
    }
    for (double coefficient : model.coefficients) hashScalar(model.thermoHash, coefficient);
    for (const auto& name : model.elementNames) hashWord(model.thermoHash, name);
    for (double count : model.elementComposition) hashScalar(model.thermoHash, count);
}
} // namespace gasModelIoDetail

inline GasModelConfiguration parseGasModelProperties(const std::string& text, bool present = true)
{
    using namespace gasModelIoDetail;
    GasModelConfiguration model;
    if (!present) return model;
    Tokens tokens(text);
    const Dictionary dict = tokens.dictionary();
    allowed(dict, {"FoamFile", "schemaVersion", "gasMode", "species", "speciesThermo", "elements", "diffusion", "mechanism", "phase", "chemistryControls"}, "gas model");
    if (dict.count("schemaVersion") && scalar(dict, "schemaVersion") != 1)
        throw error("only schemaVersion 1 is supported");
    const std::string mode = word(dict, "gasMode");
    if (mode == "single")
    {
        allowed(dict, {"FoamFile", "schemaVersion", "gasMode"}, "single gas model");
        return model;
    }
    if (mode == "mixtureFrozen") model.mode = GasMode::MixtureFrozen;
    else if (mode == "mixtureChemistry") model.mode = GasMode::MixtureChemistry;
    else throw error("unknown gasMode '" + mode + "'; use single, mixtureFrozen or mixtureChemistry");
    if (model.mode == GasMode::MixtureChemistry)
    {
        model.mechanism = word(dict, "mechanism");
        model.phase = word(dict, "phase");
        if (dict.count("chemistryControls"))
            model.chemistryControls = chemistryControls(subdict(dict, "chemistryControls"));
    }
    else if (dict.count("mechanism") || dict.count("phase") || dict.count("chemistryControls"))
        throw error("mechanism/phase/chemistryControls entries require gasMode mixtureChemistry");
    model.speciesNames = list(dict, "species");
    if (model.speciesNames.empty()) throw error("species list must not be empty");
    std::set<std::string> names;
    for (const std::string& name : model.speciesNames)
    {
        if (!names.insert(name).second) throw error("duplicate species '" + name + "'");
        if (name.empty() || name.find_first_of("/\\.;(){} \t\n") != std::string::npos)
            throw error("species name is not a valid field identifier: '" + name + "'");
    }
    const Dictionary speciesThermo = subdict(dict, "speciesThermo");
    allowed(speciesThermo, names, "speciesThermo");
    if (dict.count("elements"))
    {
        model.elementNames = list(dict, "elements");
        std::set<std::string> unique;
        for (const auto& name : model.elementNames)
            if (!unique.insert(name).second) throw error("duplicate element '" + name + "'");
    }
    for (const std::string& name : model.speciesNames)
    {
        const Dictionary data = subdict(speciesThermo, name);
        allowed(data, {"model", "molarMass", "minTemperature", "midTemperature", "maxTemperature", "referencePressure", "referenceTemperature", "referenceEntropy", "coefficients", "atoms"}, "species " + name);
        SpeciesThermoData<double> sp;
        const std::string tag = word(data, "model");
        if (tag == "linearCp") sp.model = SpeciesThermoModel::LinearCp;
        else if (tag == "NASA7") sp.model = SpeciesThermoModel::NASA7;
        else throw error("unsupported thermo model '" + tag + "' for " + name);
        sp.molarMass = scalar(data, "molarMass");
        sp.minTemperature = scalar(data, "minTemperature");
        sp.maxTemperature = scalar(data, "maxTemperature");
        if (sp.molarMass <= 0) throw error("molarMass must be positive for " + name);
        if (sp.minTemperature <= 0 || sp.maxTemperature <= sp.minTemperature)
            throw error("invalid temperature interval for " + name);
        sp.referencePressure = data.count("referencePressure") ? scalar(data, "referencePressure") : 101325;
        if (sp.referencePressure <= 0) throw error("referencePressure must be positive for " + name);
        const std::vector<double> coefficients = numbers(data, "coefficients");
        const std::size_t expected = sp.model == SpeciesThermoModel::LinearCp ? 3 : 14;
        if (coefficients.size() != expected) throw error("wrong thermo coefficient count for " + name);
        if (sp.model == SpeciesThermoModel::NASA7)
        {
            if (!data.count("referencePressure")) throw error("NASA7 requires explicit referencePressure for " + name);
            sp.midTemperature = scalar(data, "midTemperature");
            if (sp.midTemperature <= sp.minTemperature || sp.midTemperature >= sp.maxTemperature)
                throw error("invalid NASA7 midpoint temperature for " + name);
            if (data.count("referenceTemperature") || data.count("referenceEntropy"))
                throw error("NASA7 entropy is defined by its coefficients for " + name);
        }
        else
        {
            if (data.count("midTemperature")) throw error("linearCp does not use midTemperature for " + name);
            if (data.count("referenceTemperature") != data.count("referenceEntropy"))
                throw error("linearCp entropy requires both referenceTemperature and referenceEntropy for " + name);
            sp.hasEntropyReference = data.count("referenceTemperature") != 0;
            if (sp.hasEntropyReference)
            {
                if (!data.count("referencePressure")) throw error("linearCp entropy requires explicit referencePressure for " + name);
                sp.referenceTemperature = scalar(data, "referenceTemperature");
                sp.referenceEntropy = scalar(data, "referenceEntropy");
                if (sp.referenceTemperature <= 0) throw error("referenceTemperature must be positive for " + name);
            }
        }
        sp.coefficientOffset = static_cast<int>(model.coefficients.size());
        model.coefficients.insert(model.coefficients.end(), coefficients.begin(), coefficients.end());
        model.species.push_back(sp);
        if (!model.elementNames.empty())
        {
            const auto atoms = numbers(data, "atoms");
            if (atoms.size() != model.elementNames.size()) throw error("atoms count differs from elements for " + name);
            for (double count : atoms) if (count < 0) throw error("atoms must be nonnegative for " + name);
            model.elementComposition.insert(model.elementComposition.end(), atoms.begin(), atoms.end());
        }
        else if (data.count("atoms")) throw error("atoms require an ordered elements list");
    }
    double commonMinimum = model.species.front().minTemperature;
    double commonMaximum = model.species.front().maxTemperature;
    for (const auto& sp : model.species)
    {
        if (sp.minTemperature > commonMinimum) commonMinimum = sp.minTemperature;
        if (sp.maxTemperature < commonMaximum) commonMaximum = sp.maxTemperature;
    }
    if (commonMinimum >= commonMaximum) throw error("species tables have no common temperature interval");
    // Use the common thermo validator for the entire valid interval; the parser
    // does not own a second caloric polynomial/positivity implementation.
    for (const auto& sp : model.species)
    {
        SpeciesThermoView<double, 1> view;
        view.species = &sp;
        view.coefficients = model.coefficients.data();
        view.coefficientCount = static_cast<int>(model.coefficients.size());
        const ThermoStatus status = validateSpeciesThermoView(view);
        if (status != ThermoStatus::Success)
            throw error("invalid thermodynamic model or nonpositive heat capacity over its temperature interval");
    }
    const Dictionary diffusion = subdict(dict, "diffusion");
    allowed(diffusion, {"model", "coefficients", "turbulentSchmidt"}, "diffusion");
    const std::string diffusionTag = word(diffusion, "model");
    model.diffusionCoefficients.assign(model.species.size(), 0);
    if (diffusionTag == "constant")
    {
        model.diffusionModel = GasDiffusionModel::Constant;
        model.diffusionCoefficients = numbers(diffusion, "coefficients");
        if (model.diffusionCoefficients.size() != model.species.size())
            throw error("diffusion coefficients must match the exact species count/order");
        for (double value : model.diffusionCoefficients)
            if (value < 0) throw error("diffusion coefficients must be nonnegative");
    }
    else if (diffusionTag != "none") throw error("unsupported diffusion model '" + diffusionTag + "'");
    else if (diffusion.count("coefficients")) throw error("diffusion model none must not contain coefficients");
    if (diffusion.count("turbulentSchmidt")) model.turbulentSchmidt = scalar(diffusion, "turbulentSchmidt");
    if (model.turbulentSchmidt <= 0) throw error("turbulentSchmidt must be positive");
    identities(model);
    return model;
}

inline GasModelConfiguration readGasModelProperties(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
    {
        struct stat info;
        if (::stat(path.c_str(), &info) != 0 && errno == ENOENT) return GasModelConfiguration{};
        throw gasModelIoDetail::error("cannot read '" + path + "'");
    }
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw gasModelIoDetail::error("failed reading '" + path + "'");
    return parseGasModelProperties(text);
}

} // namespace ugkwp
#endif
