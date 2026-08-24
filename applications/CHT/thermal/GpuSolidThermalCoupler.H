#ifndef GPU_THERMAL_GPU_SOLID_THERMAL_COUPLER_H
#define GPU_THERMAL_GPU_SOLID_THERMAL_COUPLER_H

#include "GpuSolidThermalProperties.H"
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
    GpuSolidThermalControls controls_;

public:
    GpuSolidThermalCandidateSolver
    (
        fvMesh& mesh,
        const GpuSolidThermalProperties& properties,
        const labelList& coupledPatchIds,
        const GpuSolidThermalControls& controls
    );

    GpuSolidThermalCandidate solveTemporarySolidCandidate
    (
        const volScalarField& registeredTemperature,
        const List<scalarField>& interfaceEnergyJ,
        scalar deltaTExchange
    ) const;

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
    label particleCount{0};
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
        volScalarField& dMeanCell
    );
};

#endif

}                        
}                  

#endif
