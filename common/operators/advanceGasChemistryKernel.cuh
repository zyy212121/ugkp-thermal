#pragma once

#include "../gasTransport/GasStateView.H"
#include "../gasTransport/GasCapabilities.H"
#include "../chemistry/LocalStiffIntegrator.H"

namespace ugkwp
{

template<class GasState>
using GasChemistryStateReal = typename std::remove_pointer
<decltype(std::declval<GasState>().rho)>::type;

namespace gasChemistryAdapterDetail
{
// Only this cell's disposable trial diagnostics are changed on failure. A prior
// transport failure is never replaced, and accepted diagnostics live elsewhere.
template<class SpeciesState>
UGKWP_GAS_HD inline ChemistryStatus recordFailure
(
    SpeciesState& species,
    const int cell,
    const ChemistryStatus status
) noexcept
{
    if (species.chemistryStatus)
        species.chemistryStatus[cell] = status;
    if (species.cellStatus && species.cellStatus[cell] == 0)
        species.cellStatus[cell] = int(GasTransportCode::ChemistryFailure);
    return status;
}
} // namespace gasChemistryAdapterDetail

// The single density-storage-to-closed-reactor adapter. Both application
// adapters supply the same gas-only view. All arrays must belong to a disposable
// trial: a successful cell may publish to that trial before another cell fails.
// The caller must reject the WHOLE trial if any cell/face status is nonzero.
//
// stageVolume is explicit: Vold for the first chemical half and Vnew for the
// second. This routine does not read state.V, change geometry, or add p*dV work.
// rho, rhoUx/y/z and formation-inclusive rhoE are never written. Primitive
// recovery remains the caller's common recovery stage after successful chemistry.
// Optional audit storage is overwritten per successful call, never accumulated:
// the orchestrator owns half-stage identity and accepted diagnostic publication.
template<class GasState, class Time>
UGKWP_GAS_HD inline ChemistryStatus advanceGasChemistryCell
(
    GasState& state,
    const int cell,
    const Time chemicalInterval,
    const GasChemistryStateReal<GasState> stageVolume
) noexcept
{
    ChemistryStatus status;
    if constexpr (GasStateTraits<GasState>::speciesCount == 0)
    {
        // No species field, allocation, metadata access, or floating-point work
        // is required by the unchanged legacy path.
        return status;
    }
    else
    {
        using Real = GasChemistryStateReal<GasState>;
        constexpr int Ns = GasStateTraits<GasState>::speciesCount;
        auto& species = state.gasSpecies;
        if (species.mode == GasMode::SingleLegacy
            || species.mode == GasMode::MixtureFrozen)
            return status;

        // Invalid indices cannot safely address diagnostic arrays either.
        if (cell < 0 || cell >= state.nCells)
        {
            status.code = ChemistryCode::InvalidComposition;
            return status;
        }
        if (species.cellStatus && species.cellStatus[cell] != 0)
        {
            if (species.chemistryStatus && !species.chemistryStatus[cell])
                return species.chemistryStatus[cell];
            status.code = ChemistryCode::InvalidComposition;
            return status;
        }
        if (species.chemistryStatus && !species.chemistryStatus[cell])
            return species.chemistryStatus[cell];

        if (species.mode != GasMode::MixtureChemistry
            || !species.rho || !species.chemistryStatus || !species.cellStatus
            || !state.rho || !state.rhoUx || !state.rhoUy || !state.rhoUz
            || !state.rhoE)
        {
            status.code = ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        const Real interval = static_cast<Real>(chemicalInterval);
        if (!gasFinite(chemicalInterval) || chemicalInterval < Time(0)
            || !gasFinite(interval)
            || (chemicalInterval > Time(0) && !(interval > Real(0)))
            || !gasFinite(stageVolume) || !(stageVolume > Real(0)))
        {
            status.code = ChemistryCode::NonFiniteInput;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        const Real density = state.rho[cell];
        if (!gasFinite(density) || !(density > Real(0))
            || !gasFinite(species.densityClosureTolerance)
            || species.densityClosureTolerance < Real(0))
        {
            status.code = ChemistryCode::InvalidComposition;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        ClosedReactorInput<Real, Ns> input;
        input.totalMass = density*stageVolume;
        input.volume = stageVolume;
        Real densitySum = Real(0);
        for (int s = 0; s < Ns; ++s)
        {
            const Real partialDensity = species.rho[s*state.nCells + cell];
            if (!gasFinite(partialDensity) || partialDensity < Real(0))
            {
                status.code = ChemistryCode::InvalidComposition;
                return gasChemistryAdapterDetail::recordFailure(species, cell, status);
            }
            densitySum += partialDensity;
            input.speciesMass[s] = partialDensity*stageVolume;
        }
        if (!gasFinite(densitySum)
            || gasThermoDetail::absValue(densitySum - density)
                > species.densityClosureTolerance*density)
        {
            status.code = ChemistryCode::ConservationFailure;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        const Real ux = state.rhoUx[cell]/density;
        const Real uy = state.rhoUy[cell]/density;
        const Real uz = state.rhoUz[cell]/density;
        const Real kineticDensity = Real(0.5)*density*(ux*ux + uy*uy + uz*uz);
        input.internalEnergy = (state.rhoE[cell] - kineticDensity)*stageVolume;
        // Negative formation-inclusive U is valid; the common caloric inversion
        // determines admissibility. No heat source, energy floor, or repair.
        if (!gasFinite(ux) || !gasFinite(uy) || !gasFinite(uz)
            || !gasFinite(input.internalEnergy))
        {
            status.code = ChemistryCode::NonFiniteInput;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        ClosedReactorResult<Real, Ns> result;
        ChemistryAudit<Real, Ns> localAudit;
        status = advanceClosedReactor
        (
            input, interval, species.thermo, species.mechanism,
            species.chemistryControls, result, localAudit
        );
        if (!status)
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);

        // Validate all converted outputs before the first write. Independent
        // normalization/clipping would violate the conservative reactor contract.
        Real candidateDensity[Ns];
        densitySum = Real(0);
        for (int s = 0; s < Ns; ++s)
        {
            candidateDensity[s] = result.speciesMass[s]/stageVolume;
            if (!gasFinite(candidateDensity[s]) || candidateDensity[s] < Real(0))
            {
                status.code = ChemistryCode::NonFiniteInput;
                return gasChemistryAdapterDetail::recordFailure(species, cell, status);
            }
            densitySum += candidateDensity[s];
        }
        if (!gasFinite(densitySum)
            || gasThermoDetail::absValue(densitySum - density)
                > species.densityClosureTolerance*density)
        {
            status.code = ChemistryCode::ConservationFailure;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        // A zero interval must not change density bits through a round trip with
        // an arbitrary volume. All validation and diagnostics still take place.
        if (interval != Real(0))
            for (int s = 0; s < Ns; ++s)
                species.rho[s*state.nCells + cell] = candidateDensity[s];
        if (species.chemistryAudit)
            species.chemistryAudit[cell] = localAudit;
        species.chemistryStatus[cell] = status;
        return status;
    }
}

} // namespace ugkwp

#if defined(__CUDACC__)
// No application-specific chemical body: a CUDA launch merely assigns cells to
// the same tested host/device adapter. Stage volumes are mandatory and immutable.
template<class GasState, class Time>
__global__ void advanceGasChemistryKernel
(
    GasState* state,
    const Time chemicalInterval,
    const ugkwp::GasChemistryStateReal<GasState>* stageVolumes
)
{
    if (!state) return;
    const int cell = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell >= state->nCells) return;
    using Real = ugkwp::GasChemistryStateReal<GasState>;
    const Real volume = stageVolumes
        ? stageVolumes[cell] : Real(0);
    ugkwp::advanceGasChemistryCell(*state, cell, chemicalInterval, volume);
}
#endif
