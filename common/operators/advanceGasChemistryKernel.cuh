#pragma once

#include "../gasTransport/GasStateView.H"
#include "../gasTransport/GasCapabilities.H"
#include "../gasTransport/GasBudgetAudit.H"
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
// Actual execution failures are reported to the trial owner.
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
    const GasChemistryStateReal<GasState> stageVolume,
    const bool nativeRefinement=false
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
        using StorageReal = GasChemistryStateReal<GasState>;
        using Real = GasEnergyScalar<GasState>;
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
        if(nativeRefinement && species.cellStatus && species.cellStatus[cell]!=0)
        {
            if(species.chemistryStatus && !species.chemistryStatus[cell])return species.chemistryStatus[cell];
            status.code=ChemistryCode::InvalidComposition;
            return status;
        }
        // For public calls, old numerical diagnostics do not replace executing this
        // cell. Missing storage and unavailable arithmetic are checked where
        // their actual inputs/outputs are needed below.
        if (species.mode != GasMode::MixtureChemistry
            || !species.rho || !species.chemistryStatus || !species.cellStatus
            || !state.rho || !state.rhoUx || !state.rhoUy || !state.rhoUz
            || !state.rhoE)
        {
            status.code = ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }

        // Retain a wider Time operand through stage products and coverage.
        const Time interval = chemicalInterval;
        if (!gasFinite(interval) || interval < Time(0)
            || !gasFinite(stageVolume) || stageVolume == StorageReal(0))
        {
            status.code = ChemistryCode::NonFiniteInput;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }
        const Real density = state.rho[cell];

        ClosedReactorInput<Real, Ns> input;
        input.totalMass = density*stageVolume;
        input.volume = stageVolume;
        Real densitySum = Real(0);
        for (int s = 0; s < Ns; ++s)
        {
            const Real partialDensity = species.rho[s*state.nCells + cell];
            if (!gasFinite(partialDensity))
            {
                status.code = ChemistryCode::InvalidComposition;
                return gasChemistryAdapterDetail::recordFailure(species, cell, status);
            }
            densitySum += partialDensity;
            input.speciesMass[s] = partialDensity*stageVolume;
        }
        const Real inputDensityDefect = densitySum - density;
        const Real ux = Real(state.rhoUx[cell])/density;
        const Real uy = Real(state.rhoUy[cell])/density;
        const Real uz = Real(state.rhoUz[cell])/density;
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
            castChemistryControls<Real>(species.chemistryControls), result, localAudit
        );
        if (!status)
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);

        // Convert the actual publication before the first write. A zero-time
        // call retains the original density bits, not their volume round trip.
        StorageReal candidateDensity[Ns];
        densitySum = Real(0);
        for (int s = 0; s < Ns; ++s)
        {
            candidateDensity[s] = interval == Time(0)
                ? species.rho[s*state.nCells + cell]
                : result.speciesMass[s]/stageVolume;
            if (!gasFinite(candidateDensity[s]))
            {
                status.code = ChemistryCode::NonFiniteInput;
                return gasChemistryAdapterDetail::recordFailure(species, cell, status);
            }
            densitySum += candidateDensity[s];
        }
        // Recover one populated dependent coordinate only when conversion
        // roundoff pushes an admissible input beyond the unchanged closure gate.
        // Do not normalize/clamp species or hide a larger integration defect.
        if (interval != Time(0) && gasFinite(densitySum)
            && gasThermoDetail::absValue(inputDensityDefect)<=species.densityClosureTolerance*density
            && gasThermoDetail::absValue(densitySum - density)
                > species.densityClosureTolerance*density
            && gasThermoDetail::absValue(densitySum - density)
                <= gasThermoDetail::absValue(inputDensityDefect)
                    + Real(8*Ns)*std::numeric_limits<StorageReal>::epsilon()*density)
        {
            int dependent = 0;
            for (int s = 1; s < Ns; ++s)
                if (candidateDensity[s] > candidateDensity[dependent])
                    dependent = s;
            Real independent = Real(0);
            for (int s = 0; s < Ns; ++s)
                if (s != dependent) independent += candidateDensity[s];
            const StorageReal original = candidateDensity[dependent];
            const Real originalSum = densitySum;
            candidateDensity[dependent] = density - independent;
            densitySum = Real(0);
            for (int s = 0; s < Ns; ++s) densitySum += candidateDensity[s];
            if (!gasFinite(candidateDensity[dependent]) || candidateDensity[dependent] < StorageReal(0)
                || !gasFinite(densitySum)
                || gasThermoDetail::absValue(densitySum-density)>species.densityClosureTolerance*density)
            {
                candidateDensity[dependent] = original;
                densitySum = originalSum;
            }
        }
        // Requested diagnostics describe the rounded masses actually published.
        // An unavailable/overflowing diagnostic never cancels finite state.
        if (species.chemistryAudit)
        {
            Real publishedMass[Ns], massSum = Real(0);
            for (int s = 0; s < Ns; ++s)
            {
                publishedMass[s] = Real(candidateDensity[s])*Real(stageVolume);
                massSum += publishedMass[s];
                localAudit.speciesMassChange[s] = publishedMass[s] - input.speciesMass[s];
            }
            localAudit.massResidual = massSum - input.totalMass;
            localAudit.maximumElementResidual = Real(0);
            if (species.thermo.elementCount > 0 && !species.thermo.elementComposition)
                localAudit.maximumElementResidual = gasThermoDetail::invalidValue<Real>();
            else for (int e = 0; e < species.thermo.elementCount; ++e)
            {
                Real before = Real(0), after = Real(0);
                for (int s = 0; s < Ns; ++s)
                {
                    const Real atoms = species.thermo.elementComposition
                        [s*species.thermo.elementCount + e]/species.thermo.species[s].molarMass;
                    before += input.speciesMass[s]*atoms;
                    after += publishedMass[s]*atoms;
                }
                const Real residual = gasThermoDetail::absValue(after - before);
                if (gasFinite(localAudit.maximumElementResidual))
                    localAudit.maximumElementResidual = gasFinite(residual)
                        ? gasThermoDetail::maxValue(localAudit.maximumElementResidual, residual) : residual;
            }
            localAudit.energyResidual = mixtureEnergy
                (publishedMass, result.temperature, species.thermo) - input.internalEnergy;
        }

        // A zero interval must not change density bits through a round trip with
        // an arbitrary volume. Requested diagnostics still describe its masses.
        if (interval != Time(0))
        {
            gasBudgetChemistry(state,cell,candidateDensity,stageVolume);
            for (int s = 0; s < Ns; ++s)
                species.rho[s*state.nCells + cell] = candidateDensity[s];
        }
        if (species.chemistryAudit)
            species.chemistryAudit[cell] = castChemistryAudit<StorageReal>(localAudit);
        species.chemistryStatus[cell] = status;
        // Do not erase a separate transport/geometry/wall operation's error.
        // Their producers classify execution versus policy in their own task.
        const auto previous=static_cast<GasTransportCode>(species.cellStatus[cell]);
        if(previous==GasTransportCode::Success || previous==GasTransportCode::InvalidComposition
            || previous==GasTransportCode::InvalidThermodynamics || previous==GasTransportCode::NegativeInventory
            || previous==GasTransportCode::ChemistryFailure || previous==GasTransportCode::SourceStepLimit)
            species.cellStatus[cell] = 0;
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
    const ugkwp::GasChemistryStateReal<GasState>* stageVolumes,
    const bool nativeRefinement=false
)
{
    if (!state) return;
    const int cell = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell >= state->nCells) return;
    using Real = ugkwp::GasChemistryStateReal<GasState>;
    const Real volume = stageVolumes
        ? stageVolumes[cell] : Real(0);
    ugkwp::advanceGasChemistryCell(*state, cell, chemicalInterval, volume, nativeRefinement);
}
#endif
