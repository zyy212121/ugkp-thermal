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
// Preserve the first fatal and keep warning-only numerical residuals usable.
template<class State>
UGKWP_GAS_HD inline ChemistryStatus recordFailure
(
    State& state,
    const int cell,
    const ChemistryStatus status,
    const bool unavailable=true
) noexcept
{
    auto& species=state.gasSpecies;
    using namespace ugkp::diagnostics;
    const bool warning=gasChemistryDetail::hasUsableChemistryResult(status);
    const Code code=warning
        ?(status.code==ChemistryCode::ConservationFailure?Code::ConservationResidual:Code::AccuracyResidual)
        :(status.code==ChemistryCode::InvalidModel?Code::MissingOperator:
          (status.code==ChemistryCode::NonFiniteInput || status.thermoCode==ThermoStatus::NonFiniteInput?Code::NonFinite:Code::InvalidPhysicalState));
    auto record=makeRecord(code,2201,22,0,EntityKind::Cell,cell,status.reaction,
        status.observed,status.reference,status.observed,
        warning?(status.code==ChemistryCode::ConservationFailure?Quantity::Conservation:Quantity::Accuracy):
        (status.code==ChemistryCode::SingularJacobian?Quantity::LinearSystem:
         status.code==ChemistryCode::TemperatureFailure?Quantity::Temperature:
         status.code==ChemistryCode::InvalidComposition?Quantity::MassFraction:Quantity::Unspecified));
    // A coarse solver status does not establish a zero pivot versus nonfinite
    // matrix/RHS. Preserve its category without inventing a narrower cause.
    record.metadata=static_cast<std::uint32_t>(status.code)&detailCodeMask;
    if(warning)recordWarning(state.diagnostics,record);
    else recordFatal(state.diagnostics,record);
    if (cell>=0 && cell<state.nCells)
    {
        if(species.chemistryStatus)species.chemistryStatus[cell]=status;
        if(!warning && species.cellStatus)
            species.cellStatus[cell]=int(unavailable?GasTransportCode::InvalidStorage:GasTransportCode::ChemistryFailure);
    }
    return status;
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
        auto thermo=species.thermo;thermo.diagnostics=state.diagnostics;
        if (species.mode == GasMode::SingleLegacy
            || species.mode == GasMode::MixtureFrozen)
        {
            if(evaluated)*evaluated=true;
            return status;
        }

        (void)nativeRefinement;
        using namespace ugkp::diagnostics;
        if(hasFatal(state.diagnostics)){status.code=ChemistryCode::NonFiniteInput;return status;}
        if(cell<0 || cell>=state.nCells)
        {
            recordFatal(state.diagnostics,makeRecord(Code::InvalidIndex,2202,22,0,EntityKind::Cell,cell,-1,cell,state.nCells,0,Quantity::Index));
            status.code=ChemistryCode::InvalidComposition;return status;
        }
        if (!species.rho || !species.cellStatus
            || !state.rho || !state.rhoUx || !state.rhoUy || !state.rhoUz
            || !state.rhoE)
        {
            status.code = ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(state, cell, status);
        }


        const Time interval = chemicalInterval;
        if (!gasFinite(interval) || interval < Time(0)
            || !gasFinite(stageVolume) || stageVolume == StorageReal(0))
        {
            status.code = ChemistryCode::NonFiniteInput;
            recordFatal(state.diagnostics,makeRecord(stageVolume==StorageReal(0)?Code::ZeroDenominator:Code::NonFinite,2203,22,0,EntityKind::Cell,cell,-1,stageVolume,0,0,Quantity::Denominator));
            return status;
        }
        const Real density = state.rho[cell];

        if(!gasFinite(density) || density==Real(0))
        {
            recordFatal(state.diagnostics,makeRecord(density==Real(0)?Code::ZeroDenominator:Code::NonFinite,10031,22,0,EntityKind::Cell,cell,-1,density,0,0,Quantity::Density));
            status.code=ChemistryCode::NonFiniteInput;return status;
        }
        ClosedReactorInput<Real, Ns> input;
        input.totalMass = density*stageVolume;
        input.volume = stageVolume;
        Real densitySum = Real(0);
        for (int s = 0; s < Ns; ++s)
        {
            const Real partialDensity = species.rho[s*state.nCells + cell];
            if (!gasFinite(partialDensity))
            {
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,2205,22,0,EntityKind::Cell,cell,s,partialDensity,0,0,Quantity::MassFraction));
                status.code = ChemistryCode::InvalidComposition;return status;
            }
            const Real y=partialDensity/density;
            if(!gasFinite(y)){
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,10032,22,0,EntityKind::Cell,cell,s,y,0,0,Quantity::MassFraction));
                status.code=ChemistryCode::NonFiniteInput;return status;
            }
            const Real clipped=gasThermoDetail::maxValue(Real(0),gasThermoDetail::minValue(Real(1),y));
            if(y!=clipped)recordWarning(state.diagnostics,makeRecord(Code::FiniteYClip,10033,22,0,EntityKind::Cell,cell,s,y,0,clipped,Quantity::MassFraction));
            densitySum += partialDensity;
            if(!gasFinite(densitySum)){
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,10034,22,0,EntityKind::Cell,cell,s,densitySum,0,0,Quantity::Conservation));
                status.code=ChemistryCode::NonFiniteInput;return status;
            }
            input.speciesMass[s] = (clipped!=y?clipped*density:partialDensity)*stageVolume;
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
            return gasChemistryAdapterDetail::recordFailure(state,cell,status);
        }

        ClosedReactorResult<Real, Ns> result;
        const ChemistryStatus inputDiagnostic=status;
        bool reactorAvailable=false;
        status = advanceClosedReactor
        (
            input, interval, thermo, species.mechanism,
            castChemistryControls<Real>(species.chemistryControls), result, &reactorAvailable
        );
        if (!reactorAvailable)
            return gasChemistryAdapterDetail::recordFailure
            (
                state, cell, status
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
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,2206,22,0,EntityKind::Cell,cell,s,candidateDensity[s],0,0,Quantity::MassFraction));
                status.code=ChemistryCode::NonFiniteInput;return status;
            }
            const Real y=Real(candidateDensity[s])/density;
            if(!gasFinite(y)){
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,10035,22,0,EntityKind::Cell,cell,s,y,0,0,Quantity::MassFraction));
                status.code=ChemistryCode::NonFiniteInput;return status;
            }
            const Real clipped=gasThermoDetail::maxValue(Real(0),gasThermoDetail::minValue(Real(1),y));
            if(y!=clipped){
                recordWarning(state.diagnostics,makeRecord(Code::FiniteYClip,10036,22,0,EntityKind::Cell,cell,s,y,0,clipped,Quantity::MassFraction));
                candidateDensity[s]=StorageReal(clipped*density);
            }
            densitySum += candidateDensity[s];
            if(!gasFinite(densitySum)){
                recordFatal(state.diagnostics,makeRecord(Code::NonFinite,10037,22,0,EntityKind::Cell,cell,s,densitySum,0,0,Quantity::Conservation));
                status.code=ChemistryCode::NonFiniteInput;return status;
            }
        }
        const Real densityResidual=densitySum-density;
        if(gasFinite(densityResidual) && gasThermoDetail::absValue(densityResidual)>species.densityClosureTolerance*gasThermoDetail::absValue(density))
            recordWarning(state.diagnostics,makeRecord(Code::ConservationResidual,2204,22,0,EntityKind::Cell,cell,-1,densityResidual,species.densityClosureTolerance*gasThermoDetail::absValue(density),0,Quantity::Conservation));
        if(gasFinite(inputDensityDefect) && gasThermoDetail::absValue(inputDensityDefect)>species.densityClosureTolerance*gasThermoDetail::absValue(density))
            recordWarning(state.diagnostics,makeRecord(Code::ConservationResidual,10038,22,0,EntityKind::Cell,cell,-1,inputDensityDefect,species.densityClosureTolerance*gasThermoDetail::absValue(density),inputDensityDefect,Quantity::Conservation));
        // A zero interval must not change density bits through a round trip with

        if (interval != Time(0))
        {
            for (int s = 0; s < Ns; ++s)
                species.rho[s*state.nCells + cell] = candidateDensity[s];
        }
        if(species.chemistryStatus)species.chemistryStatus[cell] = status;
        if(evaluated)*evaluated=true;
        if(!status)return gasChemistryAdapterDetail::recordFailure(state,cell,status,false);
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
        if(cell<0 || cell>=state.nCells){
            using namespace ugkp::diagnostics;
            recordFatal(state.diagnostics,makeRecord(Code::InvalidIndex,10039,22,0,EntityKind::Cell,cell,-1,cell,state.nCells,0,Quantity::Index));
            ChemistryStatus status;status.code=ChemistryCode::InvalidComposition;return status;
        }
        if(!volumes){ChemistryStatus status;status.code=ChemistryCode::InvalidModel;
            return gasChemistryAdapterDetail::recordFailure(state,cell,status);}
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
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int cell = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell >= state->nCells) return;
    ugkwp::advanceGasChemistryCellWithVolumes(*state, cell, chemicalInterval, stageVolumes, nativeRefinement);
}
#endif
