#ifndef UGKWP_GAS_MODEL_H
#define UGKWP_GAS_MODEL_H

#include <cstdint>

#if defined(__CUDACC__)
#define UGKWP_GAS_HD __host__ __device__
#else
#define UGKWP_GAS_HD
#endif

namespace ugkwp
{

// Runtime physics selection. Ns remains part of the generated state/ABI identity.
enum class GasMode : unsigned char
{
    SingleLegacy,
    MixtureFrozen,
    MixtureChemistry
};

enum class SpeciesThermoModel : unsigned char { LinearCp, NASA7 };

template<class Real>
UGKWP_GAS_HD constexpr Real universalGasConstant() noexcept
{
    return Real(8.31446261815324); // J/(mol K); exact SI N_A*k_B
}

// LinearCp coefficients: cp0 [J/(kg K)], cp1 [J/(kg K^2)], e0 [J/kg].
// h=e0+cp0*T+cp1*T*T/2. e=h-R*T. This is an energy intercept at T=0,
// not a value of h or e at referenceTemperature. Entropy has its own reference.
// NASA7 coefficients: two contiguous seven-term dimensionless arrays, low then
// high. T==midTemperature selects the low range, as in Cantera's NasaPoly2.
template<class Real>
struct SpeciesThermoData
{
    SpeciesThermoModel model = SpeciesThermoModel::LinearCp;
    int coefficientOffset = 0;
    Real molarMass = Real(0);              // kg/mol
    Real minTemperature = Real(0);         // K
    Real midTemperature = Real(0);         // K, NASA7 only
    Real maxTemperature = Real(0);         // K
    Real referencePressure = Real(0);      // Pa; standard-state entropy pressure
    Real referenceTemperature = Real(0);   // K, LinearCp only
    Real referenceEntropy = Real(0);       // J/(kg K), LinearCp only
    bool hasEntropyReference = false;     // explicit LinearCp entropy; never inferred
};

// Borrowed immutable storage. Host and device owners construct separate views;
// borrowed addresses MUST NOT be serialized into a protocol or checkpoint.
// elementComposition is species-major: [species*elementCount+element], atoms
// per molecule (equivalently mol atoms/mol species), not mol atoms/kg.
template<class Real, int Ns>
struct SpeciesThermoView
{
    static_assert(Ns > 0, "A gas model needs at least one species");
    static constexpr int speciesCount = Ns;
    const SpeciesThermoData<Real>* species = nullptr;
    const Real* coefficients = nullptr;
    int coefficientCount = 0;
    int elementCount = 0;
    const Real* elementComposition = nullptr;
    std::uint64_t speciesOrderHash = 0;
    std::uint64_t thermoHash = 0;
};

enum class GasReactionType : unsigned char
{
    Elementary,
    ThirdBody,
    Lindemann,
    Troe
};

template<class Real>
struct GasArrheniusData
{
    Real preExponential = Real(0);     // SI mol,m,s; order-dependent units
    Real temperatureExponent = Real(0);
    Real activationEnergy = Real(0);   // J/mol; finite negative values are valid
};

template<class Real>
struct GasTroeData
{
    Real alpha = Real(0);
    Real T3 = Real(0);                 // K
    Real T1 = Real(0);                 // K
    Real T2 = Real(0);                 // K, optional exp(-T2/T) term
    bool hasT2 = false;
};

template<class Real>
struct GasStoichTerm
{
    int species = -1;
    Real coefficient = Real(0);        // nonnegative reactant/product order
};

template<class Real>
struct GasColliderEfficiency
{
    int species = -1;
    Real efficiency = Real(1);
};

// Reactants and products are separate sparse lists, retaining kinetic order and
// explicit colliders on both sides. Reversible means Gibbs-derived reverse rate.
// Unsupported rate types/custom orders must be rejected by the host importer.
// Duplicate channels remain distinct entries and are summed, never deduplicated.
template<class Real>
struct GasReactionData
{
    GasReactionType type = GasReactionType::Elementary;
    bool reversible = false;
    bool duplicate = false;
    int sourceIndex = -1;
    int reactantOffset = 0;
    int reactantCount = 0;
    int productOffset = 0;
    int productCount = 0;
    int efficiencyOffset = 0;
    int efficiencyCount = 0;
    GasArrheniusData<Real> highRate;
    GasArrheniusData<Real> lowRate;      // separate low-pressure Arrhenius law
    Real defaultEfficiency = Real(1);
    GasTroeData<Real> troe;
};

template<class Real, int Ns>
struct GasMechanismView
{
    static_assert(Ns > 0, "A mechanism needs at least one species");
    static constexpr int speciesCount = Ns;
    const GasReactionData<Real>* reactions = nullptr;
    int reactionCount = 0;              // independent of material reaction count
    const GasStoichTerm<Real>* reactants = nullptr;
    int reactantTermCount = 0;
    const GasStoichTerm<Real>* products = nullptr;
    int productTermCount = 0;
    const GasColliderEfficiency<Real>* efficiencies = nullptr;
    int efficiencyCount = 0;
    // Dimensionless MOLAR stoichiometric columns, species-major [Ns*rank].
    // Reduced reactor coordinates are molar-concentration increments [mol/m^3].
    const Real* stoichiometricBasis = nullptr;
    int independentRank = 0;           // 0 <= independentRank <= Ns
    Real referencePressure = Real(0);   // Pa, common standard pressure for Kc
    std::uint64_t speciesOrderHash = 0;
    std::uint64_t mechanismHash = 0;
};

enum class ThermoStatus : unsigned char
{
    Success,
    InvalidModel,
    InvalidComposition,
    NonFiniteInput,
    TemperatureOutOfRange,
    NonPositiveHeatCapacity,
    InvalidControls,
    EnergyOutOfRange,
    IterationLimit
};

template<class Real>
struct ThermoInversionControls
{
    Real minTemperature = Real(0);       // optional tighter lower bound
    Real maxTemperature = Real(0);       // zero uses table upper bound
    Real absoluteEnergyTolerance = Real(1e-10);       // J
    Real relativeEnergyTolerance = Real(1e-10);
    Real absoluteTemperatureTolerance = Real(1e-9);   // K
    Real relativeTemperatureTolerance = Real(1e-10);
    int maximumIterations = 100;
};

// One cell's immutable closed-reactor input. No momentum, density layout, mesh,
// application type or borrowed state pointer. The interval is a separate argument
// to advanceClosedReactor; do not put a transport time step into the model view.
template<class Real, int Ns>
struct ClosedReactorInput
{
    static_assert(Ns > 0, "A reactor needs at least one species");
    Real speciesMass[Ns] = {};          // kg, all Ns species including exact zeros
    Real totalMass = Real(0);           // kg, unchanged by homogeneous chemistry
    Real internalEnergy = Real(0);      // J, formation-inclusive, may be negative
    Real volume = Real(0);              // m^3
};

template<class Real, int Ns>
struct ClosedReactorResult
{
    Real speciesMass[Ns] = {};          // kg; only published on complete success
    Real temperature = Real(0);         // K
};

} // namespace ugkwp

#endif
