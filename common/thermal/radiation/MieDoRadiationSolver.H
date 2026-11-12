#ifndef GPU_THERMAL_MIE_DO_RADIATION_SOLVER_H
#define GPU_THERMAL_MIE_DO_RADIATION_SOLVER_H

#include "MieTable.H"
#include "gpu/RadiationResidentView.H"
#include "dictionary.H"
#include "fvMesh.H"
#include "labelList.H"
#include "scalarField.H"
#include "volFields.H"

#include <memory>
#include <stdexcept>

namespace Foam
{
namespace gpuThermal
{

struct MieDoControls
{
    label nTheta;
    label nPhi;
    label maxSourceIterations{300};
    scalar sourceRelativeTolerance{1e-10};
    scalar phaseBalanceTolerance{1e-12};
    label maxPhaseBalanceIterations{200};
    scalar radiationActiveEpsilonMin{0};
    scalar conservationRelativeTolerance{0};
    scalar maxPhaseScalingCorrection{0};
    label angleParallelThreads{1};
};

struct MieDoSnapshot
{
    const scalarField& epsilonS;
    const scalarField& particleTemperatureK;
    const scalarField& representativeDiameterM;
    const volScalarField& wallTemperatureK;
    const List<scalarField>& coupledWallEmissivity;
    ugkp::gpu::RadiationParticleDeviceView resident{};
};

struct MieDoResult
{
    scalarField absorptionCoefficientInvM;
    scalarField scatteringCoefficientInvM;
    scalarField extinctionCoefficientInvM;
    scalarField asymmetryFactor;
    scalarField incidentRadiationG_Wm2;
    scalarField particlePowerDensityWm3;
    List<scalarField> boundaryQrOutwardWm2;
    scalar integratedParticlePowerW;
    scalar integratedBoundaryPowerW;
    scalar conservationResidualW;
    label sourceIterations;
    label angleThreadsUsed;
    ugkp::gpu::RadiationDeviceView resident{};
};

class MieDoSolveError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class MieDoRadiationSolverData;
class MieDoRadiationWorkspaceData;

class MieDoRadiationWorkspace
{
    std::unique_ptr<MieDoRadiationWorkspaceData> data_;

    explicit MieDoRadiationWorkspace
    (
        std::unique_ptr<MieDoRadiationWorkspaceData> data
    ) noexcept;

    friend class MieDoRadiationSolver;

public:
    ~MieDoRadiationWorkspace();

    MieDoRadiationWorkspace(MieDoRadiationWorkspace&& other) noexcept;
    MieDoRadiationWorkspace& operator=
    (
        MieDoRadiationWorkspace&& other
    ) noexcept;

    MieDoRadiationWorkspace(const MieDoRadiationWorkspace&) = delete;
    MieDoRadiationWorkspace& operator=
    (
        const MieDoRadiationWorkspace&
    ) = delete;
};

class MieDoRadiationSolver
{
    std::shared_ptr<const MieDoRadiationSolverData> data_;

public:
    MieDoRadiationSolver
    (
        const fvMesh& mesh,
        const dictionary& radiationProperties,
        const MieTable& mieTable,
        const labelList& coupledSolidPatchIds,
        const MieDoControls& controls
    );

    // Call once after an accepted mesh motion, before the next radiation solve.
    void updateGeometry();
    const fvMesh& mesh() const noexcept;
    const MieTable& mieTable() const noexcept;
    const labelList& coupledSolidPatchIds() const noexcept;
    const MieDoControls& controls() const noexcept;
    void validateSnapshot(const MieDoSnapshot& snapshot) const;

    MieDoRadiationWorkspace makeWorkspace() const;
    MieDoResult solve(const MieDoSnapshot& snapshot) const;
    MieDoResult solve
    (
        const MieDoSnapshot& snapshot,
        MieDoRadiationWorkspace& workspace,
        bool exportFields=true
    ) const;
    MieDoResult collect(MieDoRadiationWorkspace&) const;
};

}                        
}                  

#endif
