#ifndef UGKWP_GAS_MODEL_H
#define UGKWP_GAS_MODEL_H

#include <cstdint>
#include "../Diagnostics.H"

#if defined(__CUDACC__)
#define UGKWP_GAS_HD __host__ __device__
#else
#define UGKWP_GAS_HD
#endif

namespace ugkwp
{


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
    return Real(8.31446261815324);
}






template<class Real>
struct SpeciesThermoData
{
    SpeciesThermoModel model = SpeciesThermoModel::LinearCp;
    int coefficientOffset = 0;
    Real molarMass = Real(0);
    Real minTemperature = Real(0);
    Real midTemperature = Real(0);
    Real maxTemperature = Real(0);
    Real referencePressure = Real(0);
    Real referenceTemperature = Real(0);
    Real referenceEntropy = Real(0);
    bool hasEntropyReference = false;
};





template<class Real, int Ns>
struct SpeciesThermoView
{
    static constexpr int speciesCount = Ns;
    const SpeciesThermoData<Real>* species = nullptr;
    const Real* coefficients = nullptr;
    int coefficientCount = 0;
    int elementCount = 0;
    const Real* elementComposition = nullptr;
    std::uint64_t speciesOrderHash = 0;
    std::uint64_t thermoHash = 0;
    ugkp::diagnostics::DeviceContext* diagnostics = nullptr;
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
    Real preExponential = Real(0);
    Real temperatureExponent = Real(0);
    Real activationEnergy = Real(0);
};

template<class Real>
struct GasTroeData
{
    Real alpha = Real(0);
    Real T3 = Real(0);
    Real T1 = Real(0);
    Real T2 = Real(0);
    bool hasT2 = false;
};

template<class Real>
struct GasStoichTerm
{
    int species = -1;
    Real coefficient = Real(0);
};

template<class Real>
struct GasColliderEfficiency
{
    int species = -1;
    Real efficiency = Real(1);
};





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
    GasArrheniusData<Real> lowRate;
    Real defaultEfficiency = Real(1);
    GasTroeData<Real> troe;
};

template<class Real, int Ns>
struct GasMechanismView
{
    static constexpr int speciesCount = Ns;
    const GasReactionData<Real>* reactions = nullptr;
    int reactionCount = 0;
    const GasStoichTerm<Real>* reactants = nullptr;
    int reactantTermCount = 0;
    const GasStoichTerm<Real>* products = nullptr;
    int productTermCount = 0;
    const GasColliderEfficiency<Real>* efficiencies = nullptr;
    int efficiencyCount = 0;


    const Real* stoichiometricBasis = nullptr;
    int independentRank = 0;
    Real referencePressure = Real(0);
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
    Real minTemperature = Real(0);
    Real maxTemperature = Real(0);
    Real absoluteEnergyTolerance = Real(1e-10);
    Real relativeEnergyTolerance = Real(1e-10);
    Real absoluteTemperatureTolerance = Real(1e-9);
    Real relativeTemperatureTolerance = Real(1e-10);
    int maximumIterations = 100;
    bool explicitRelativeEnergyTolerance = false;
    bool explicitRelativeTemperatureTolerance = false;
};


template<class To,class From>
UGKWP_GAS_HD inline ThermoInversionControls<To> castThermoControls
(const ThermoInversionControls<From>& input) noexcept
{
    ThermoInversionControls<To> out;
    out.minTemperature=To(input.minTemperature);out.maxTemperature=To(input.maxTemperature);
    out.absoluteEnergyTolerance=To(input.absoluteEnergyTolerance);
    out.relativeEnergyTolerance=To(input.relativeEnergyTolerance);
    out.absoluteTemperatureTolerance=To(input.absoluteTemperatureTolerance);
    out.relativeTemperatureTolerance=To(input.relativeTemperatureTolerance);
    out.maximumIterations=input.maximumIterations;
    out.explicitRelativeEnergyTolerance=input.explicitRelativeEnergyTolerance;
    out.explicitRelativeTemperatureTolerance=input.explicitRelativeTemperatureTolerance;
    return out;
}




template<class Real, int Ns>
struct ClosedReactorInput
{
    Real speciesMass[Ns] = {};
    Real totalMass = Real(0);
    Real internalEnergy = Real(0);
    Real volume = Real(0);
};

template<class Real, int Ns>
struct ClosedReactorResult
{
    Real speciesMass[Ns] = {};
    Real temperature = Real(0);
};

}

#endif
