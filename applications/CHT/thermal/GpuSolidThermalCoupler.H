#ifndef GPU_THERMAL_GPU_SOLID_THERMAL_COUPLER_H
#define GPU_THERMAL_GPU_SOLID_THERMAL_COUPLER_H

#include "GpuSolidThermalProperties.H"
#include "../../../common/gpu/SparseConjugateGradient.H"
#include "../../../common/gpu/SolidThermalProperties.H"
#include "../../../common/gpu/ImplicitDiffusion.H"
#include "autoPtr.H"
#include "fvMesh.H"
#include "labelList.H"
#include "scalarField.H"
#include "volFields.H"

#include <stdexcept>
#include <memory>

namespace Foam
{
class Time;
class GpuParticleResidentSolver;

namespace gpuThermal
{

struct ThermalExchangeRestartState;

class GpuSolidThermalSolveError
:
    public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct GpuSolidThermalControls
{
    label nonlinearMaxIterations{50};
    scalar nonlinearRelativeTolerance{1e-10};
    scalar energyRelativeTolerance{1e-8};
    scalar energyAbsoluteTolerance{1e-10};
    bool nonlinearAitken{false};
    scalar nonlinearInitialRelaxation{1};
    scalar nonlinearMinimumRelaxation{0.05};
    scalar nonlinearMaximumRelaxation{1};
};

scalar clampedAitkenRelaxation
(
    const scalarField& previousResidual,
    const scalarField& currentResidual,
    scalar previousRelaxation,
    scalar minimumRelaxation,
    scalar maximumRelaxation
);

scalarField coupledPatchOwnerTemperature
(
    const volScalarField& solidTemperature,
    label patchI
);

// Gas/contact/radiation need the accepted surface, not the owner node.
scalarField coupledPatchSurfaceTemperature
(
    const volScalarField& solidTemperature,
    label patchI
);

class GpuSolidThermalCandidate
{
    autoPtr<volScalarField> temperature_;

public:
    scalar integratedEnergyResidualJ{0};
    label nonlinearIterations{0};

    GpuSolidThermalCandidate
    (
        autoPtr<volScalarField> temperature,
        scalar integratedEnergyResidualJ,
        label nonlinearIterations
    );

    const volScalarField& temperature() const;

    friend class GpuSolidThermalCandidateSolver;
};

class GpuSolidThermalCandidateSolver
{
    fvMesh& mesh_;
    const GpuSolidThermalProperties& properties_;
    labelList coupledPatchIds_;
    labelList ownerConductivityPatchIds_;
    GpuSolidThermalControls controls_;
    dictionary normalLinearControls_, finalLinearControls_;
    const dictionary& selectedLinearControls() const;
    mutable std::unique_ptr<ugkp::gpu::SparseConjugateGradient> sparse_;
    mutable std::unique_ptr<ugkp::gpu::SolidThermalProperties> deviceProperties_;
    mutable std::unique_ptr<ugkp::gpu::ImplicitDiffusion> residentDiffusion_;
    mutable std::unique_ptr<ugkp::gpu::SolidThermalIterationGraph> residentGraph_;
    mutable bool residentAdvanced_=false;
    GpuSolidThermalCandidate solveCandidate(const volScalarField&,const List<scalarField>&,scalar,
        const ugkp::gpu::ThermalInterfaceEnergyDeviceView&,const ugkp::gpu::ThermalInterfaceEnergyDeviceView&,
        bool prepareOnly,ugkp::diagnostics::DeviceContext*) const;

public:
    GpuSolidThermalCandidateSolver
    (
        fvMesh& mesh,
        const GpuSolidThermalProperties& properties,
        const labelList& coupledPatchIds,
        const GpuSolidThermalControls& controls,
        const labelList& ownerConductivityPatchIds = labelList()
    );

    GpuSolidThermalCandidate solveTemporarySolidCandidate
    (
        const volScalarField& registeredTemperature,
        const List<scalarField>& interfaceEnergyJ,
        scalar deltaTExchange,
        const ugkp::gpu::ThermalInterfaceEnergyDeviceView& residentEnergy = {},
        const ugkp::gpu::ThermalInterfaceEnergyDeviceView& appendedEnergy = {}
    ) const;

    bool prepareResident(const volScalarField&,const ugkp::gpu::ThermalInterfaceEnergyDeviceView&,
        ugkp::diagnostics::DeviceContext*,const ugkp::gpu::ThermalInterfaceEnergyDeviceView& appended={}) const;
    ugkp::gpu::SolidThermalIterationGraph& prepareResidentContact(label patch) const;
    void acceptResidentContact() const { residentAdvanced_=true; }
    bool residentPrepared() const { return bool(residentGraph_); }
    bool residentAdvanced() const { return residentAdvanced_; }
    void advanceResident(void* producerStream) const;
    void publishResident(volScalarField&) const;

    ugkp::gpu::ThermalSolidContactDeviceView contactState(const volScalarField&,label patch,bool initialise) const;
    ugkp::gpu::ThermalInterfaceStateDeviceView interfaceState(const labelList& contactPairIds,label pairs,bool includeEmissivity=false,ugkp::diagnostics::DeviceContext* context=nullptr) const;

    void publishSolidCandidate
    (
        const GpuSolidThermalCandidate& candidate,
        volScalarField& registeredTemperature
    ) const;
};

#ifdef UGKWP_USE_CUDA

class GpuSolidThermalCouplerData;

struct GpuThermalCouplingResult
{
    bool coupled{false};
    bool radiationCoupled{false};
    scalar elapsedCouplingTimeS{0};
    scalar elapsedRadiationTimeS{0};
};

class GpuSolidThermalCoupler
{
    std::unique_ptr<GpuSolidThermalCouplerData> data_;

public:
    GpuSolidThermalCoupler
    (
        Time& runTime,
        fvMesh& fluidMesh,
        GpuParticleResidentSolver& resident,
        const dictionary& couplingDictionary,
        const ThermalExchangeRestartState& validatedRestartState
    );

    ~GpuSolidThermalCoupler();

    void initialiseWallTemperatureBeforeFirstAdvance(volScalarField& Tgas);

    GpuThermalCouplingResult exchangeIfDue
    (
        Time& runTime,
        volScalarField& rho,
        volVectorField& rhoU,
        volScalarField& rhoE,
        volVectorField& U,
        volScalarField& p,
        volScalarField& Tgas,
        volScalarField& epsilonS,
        volVectorField& rhoUs,
        volScalarField& rhoEs,
        volScalarField& rhoDs,
        volScalarField& rhoHp,
        volVectorField& Us,
        volScalarField& theta,
        volScalarField& Tp,
        volScalarField& dMeanCell
    );

    label persistAtWriteTime
    (
        Time& runTime,
        volScalarField& rho,
        volVectorField& rhoU,
        volScalarField& rhoE,
        volVectorField& U,
        volScalarField& p,
        volScalarField& Tgas,
        volScalarField& epsilonS,
        volVectorField& rhoUs,
        volScalarField& rhoEs,
        volScalarField& rhoDs,
        volScalarField& rhoHp,
        volVectorField& Us,
        volScalarField& theta,
        volScalarField& Tp,
        volScalarField& dMeanCell,
        bool thermalCouplingWrite
    );
};

#endif

}                        
}                  

#endif
