#pragma once

#include "../gasTransport/GasStateView.H"
#include "../gasTransport/GasCapabilities.H"
#include "../gasTransport/GasGeometryValidation.H"
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
    const ChemistryStatus status,
    const bool unavailable=true
) noexcept
{
    if (species.chemistryStatus)
        species.chemistryStatus[cell] = status;
    if (species.cellStatus && species.cellStatus[cell]!=int(GasTransportCode::InvalidStorage)
        && (unavailable || species.cellStatus[cell]==0))
        species.cellStatus[cell] = int(unavailable?GasTransportCode::InvalidStorage:GasTransportCode::ChemistryFailure);
    return status;
}




template<class TableReal,int Ns>
UGKWP_GAS_HD inline bool unavailableTrialModel
(
    const SpeciesThermoView<TableReal,Ns>& thermo,
    const GasMechanismView<TableReal,Ns>& mechanism,
    const ChemistryStatus& status
) noexcept
{
    if (status.thermoCode == ThermoStatus::InvalidModel)
    {
        if (!thermo.species || !thermo.coefficients) return true;
        for (int s = 0; s < Ns; ++s)
            if (thermo.species[s].model != SpeciesThermoModel::LinearCp
                && thermo.species[s].model != SpeciesThermoModel::NASA7) return true;


        return false;
    }
    if (status.code == ChemistryCode::InvalidModel
        && mechanism.independentRank > 0 && !mechanism.stoichiometricBasis) return true;
    if (status.attemptedSteps > 0)
    {
        if (mechanism.reactionCount > 0 && !mechanism.reactions) return true;
        if (status.reaction >= 0 && mechanism.reactions)
            return !gasChemistryDetail::reactionInputAvailable<TableReal,Ns,true>
                (mechanism,status.reaction);
    }
    return false;
}
} // namespace gasChemistryAdapterDetail

// The single density-storage-to-closed-reactor adapter. Both application
// adapters supply the same gas-only view. All arrays must belong to a disposable
// trial: a successful cell may publish to that trial before another cell fails.

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
    const bool nativeRefinement=false,
    bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    ChemistryStatus status;
    if constexpr (GasStateTraits<GasState>::speciesCount == 0)
    {
        // No species field, allocation, metadata access, or floating-point work
        // is required by the unchanged legacy path.
        if(evaluated)*evaluated=true;
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
        {
            if(evaluated)*evaluated=true;
            return status;
        }

        if(nativeRefinement && species.cellStatus && species.cellStatus[cell]!=0)
        {
            if(species.chemistryStatus && !species.chemistryStatus[cell])return species.chemistryStatus[cell];
            status.code=ChemistryCode::InvalidComposition;
            return status;
        }



        if (!species.rho || !species.cellStatus
            || !state.rho || !state.rhoUx || !state.rhoUy || !state.rhoUz
            || !state.rhoE)
        {
            status.code = ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(species, cell, status);
        }


        const Time interval = chemicalInterval;
        if (!gasFinite(interval) || interval < Time(0)
            || !gasFinite(stageVolume) || stageVolume == StorageReal(0))
        {
            status.code = ChemistryCode::NonFiniteInput;
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
        }

        ClosedReactorResult<Real, Ns> result;
        const ChemistryStatus inputDiagnostic=status;
        bool reactorAvailable=false;
        if (nativeRefinement)
            status = advanceClosedReactorTrial
            (
                input, interval, species.thermo, species.mechanism,
                castChemistryControls<Real>(species.chemistryControls), result, &reactorAvailable
            );
        else
            status = advanceClosedReactor
            (
                input, interval, species.thermo, species.mechanism,
                castChemistryControls<Real>(species.chemistryControls), result, &reactorAvailable
            );
        if (!reactorAvailable)
            return gasChemistryAdapterDetail::recordFailure
            (
                species, cell, status,
                !nativeRefinement || gasChemistryAdapterDetail::unavailableTrialModel
                    (species.thermo,species.mechanism,status)
            );
        gasChemistryDetail::retainDiagnostic(status,inputDiagnostic);


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
        // A zero interval must not change density bits through a round trip with

        if (interval != Time(0))
        {
            for (int s = 0; s < Ns; ++s)
                species.rho[s*state.nCells + cell] = candidateDensity[s];
        }
        if(species.chemistryStatus)species.chemistryStatus[cell] = status;
        if(evaluated)*evaluated=true;
        if(!status)return gasChemistryAdapterDetail::recordFailure(species,cell,status,false);
        return status;
    }
}



template<class GasState,class Time>
UGKWP_GAS_HD inline ChemistryStatus advanceGasChemistryCellWithVolumes
(GasState& state,int cell,Time interval,const GasChemistryStateReal<GasState>* volumes,
 bool nativeRefinement=false,bool* evaluated=nullptr) noexcept
{
    if(evaluated)*evaluated=false;
    if constexpr(GasStateTraits<GasState>::speciesCount==0){if(evaluated)*evaluated=true;return ChemistryStatus{};}
    else {
        auto& species=state.gasSpecies;
        if(species.mode==GasMode::SingleLegacy || species.mode==GasMode::MixtureFrozen)
        {if(evaluated)*evaluated=true;return ChemistryStatus{};}
        if(!volumes){ChemistryStatus status;status.code=ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(species,cell,status);}
        return advanceGasChemistryCell(state,cell,interval,volumes[cell],nativeRefinement,evaluated);
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
    ugkwp::advanceGasChemistryCellWithVolumes(*state, cell, chemicalInterval, stageVolumes, nativeRefinement);
}
#endif
