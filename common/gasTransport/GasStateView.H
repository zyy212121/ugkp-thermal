#pragma once
#include <type_traits>
#include <limits>
#include <utility>
#include "GasModel.H"
#include "GasGeometryPolicy.H"
#include "GasSstAudit.H"
#include "GasBoundaryLayerState.H"
#include "../chemistry/ChemistryTypes.H"

namespace ugkwp
{

// A non-owning snapshot of the existing density-storage gas operator contract.
// Conserved fields are rho, rho*u and rho*E, not integrated cell inventories.
// Allocation, lifetime, device upload and scalar refresh belong to the caller.
// In particular, a copied view does not track later pointer/scalar assignments.
// The SST fields are optional at runtime when sstConfigured is zero; they do
// not introduce particle, scheduler, host container or CUDA runtime ownership.
// Wall exposure and accepted wall energy remain separate caller capabilities.
template<class Real, class Time, class SstCoefficients>
struct GasStateView
{
    using RealType = Real;
    using TimeType = Time;
    using SstCoefficientsType = SstCoefficients;
    static constexpr int speciesCount = 0;
    GasGeometryState<Real> gasGeometry;
    GasSstAuditState<Real> gasSstAudit;
    GasBoundaryLayerState<Real> gasBoundaryLayer;
    Real gasThermalConductivity = Real(-1); // negative: legacy mu*cp/Pr; otherwise W/(m K)

#define UGKWP_GAS_STATE_FIELD(Type, Name) Type Name{};
#include "GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
};

// Optional species capability. Storage uses exactly Ns species, never padded
// capacity. Cell arrays use [species*nCells+cell], face arrays use
// [species*nFaces+face]; scalar primitive/status fields are per-cell/per-face.
// rho and initial have units kg/m^3; flux is signed kg/s. grad* are gradients
// of mass fraction [1/m]. All arrays belong to the caller's trial allocation.
template<class Real, int Ns>
struct GasSpeciesState
{
    static_assert(Ns > 0, "species storage requires a positive exact count");
    static constexpr int speciesCount = Ns;
    using RealType = Real;
    GasMode mode = GasMode::SingleLegacy;
    SpeciesThermoView<Real, Ns> thermo;
    ThermoInversionControls<Real> thermoControls;
    GasMechanismView<Real, Ns> mechanism;
    ChemistryControls<Real> chemistryControls;
    ChemistryAudit<Real, Ns>* chemistryAudit = nullptr; // per-cell trial audit
    ChemistryStatus* chemistryStatus = nullptr;        // per-cell attempted status
    Real densityClosureTolerance = Real(64)*std::numeric_limits<Real>::epsilon();
    Real* rho = nullptr;
    Real* initial = nullptr;
    Real* flux = nullptr;
    Real* gradX = nullptr;
    Real* gradY = nullptr;
    Real* gradZ = nullptr;
    Real* limiter = nullptr;             // one jointly admissible scale per cell
    Real* boundaryMassFraction = nullptr;
    int* compositionBoundaryFixed = nullptr; // per face, 0 extrapolated, 1 fixed
    Real* positivityScale = nullptr;     // species-specific donor scale per cell
    const Real* diffusivity = nullptr;   // Ns molecular D values, m^2/s; null=off
    Real turbulentSchmidt = Real(0.7);
    Real* soundSpeed = nullptr;
    Real* heatCapacity = nullptr;        // mixture cp, J/(kg K)
    Real* gasConstant = nullptr;         // mixture R, J/(kg K)
    int* cellStatus = nullptr;
    int* faceStatus = nullptr;
};

// Legacy DeviceState types require no added field, allocation or runtime work.
template<class State, class = void>
struct GasStateTraits { static constexpr int speciesCount = 0; };

template<class State>
struct GasStateTraits<State, std::void_t<decltype(std::declval<State>().gasSpecies)>>
{
    using SpeciesState = decltype(std::declval<State>().gasSpecies);
    static constexpr int speciesCount = SpeciesState::speciesCount;
};

template<class State>
UGKWP_GAS_HD inline bool mixtureGasActive(const State& state)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        return state.gasSpecies.mode != GasMode::SingleLegacy;
    return false;
}

template<class State,class=void> struct GasDirectConductivity : std::false_type {};
template<class State> struct GasDirectConductivity<State,
    std::void_t<decltype(std::declval<State>().gasThermalConductivity)>> : std::true_type {};

template<class State>
UGKWP_GAS_HD inline bool gasHasDirectConductivity(const State& state)
{
    if constexpr (GasDirectConductivity<State>::value)
        return state.gasThermalConductivity >= 0;
    return false;
}

template<class State,class=void> struct GasGeometryCapability : std::false_type {};
template<class State> struct GasGeometryCapability<State,
    std::void_t<decltype(std::declval<State>().gasGeometry)>> : std::true_type {};
template<class State>
UGKWP_GAS_HD inline bool gasMovingGeometry(const State& state)
{
    if constexpr (GasGeometryCapability<State>::value) return state.gasGeometry.enabled;
    return false;
}

// Bind legacy storage (or any conforming gas-only storage) without copying
// field data. Pointer assignment also rejects accidental precision changes.
// This is a host-side binding operation; upload the returned POD as needed.
template<class State>
inline auto makeGasStateView(const State& state)
    -> GasStateView
    <
        typename std::remove_pointer<decltype(state.rho)>::type,
        typename std::remove_pointer<decltype(state.pressureScheduleTimes)>::type,
        decltype(state.sstCoefficients)
    >
{
    using Real = typename std::remove_pointer<decltype(state.rho)>::type;
    using Time = typename std::remove_pointer<decltype(state.pressureScheduleTimes)>::type;
    using SstCoefficients = decltype(state.sstCoefficients);
    GasStateView<Real, Time, SstCoefficients> view{};
#define UGKWP_GAS_STATE_FIELD(Type, Name) view.Name = state.Name;
#include "GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    if constexpr (GasDirectConductivity<State>::value)
        view.gasThermalConductivity=state.gasThermalConductivity;
    if constexpr (GasGeometryCapability<State>::value) view.gasGeometry=state.gasGeometry;
    if constexpr (GasSstAuditCapability<State>::value) view.gasSstAudit=state.gasSstAudit;
    if constexpr (GasBoundaryLayerCapability<State>::value) view.gasBoundaryLayer=state.gasBoundaryLayer;
    return view;
}

} // namespace ugkwp
