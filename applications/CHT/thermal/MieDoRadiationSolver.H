#ifndef GPU_THERMAL_MIE_DO_RADIATION_SOLVER_H
#define GPU_THERMAL_MIE_DO_RADIATION_SOLVER_H

#include "MieTable.H"
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
    label maxSourceIterations;
    scalar sourceRelativeTolerance;
    scalar phaseBalanceTolerance;
    label maxPhaseBalanceIterations;
    scalar radiationActiveEpsilonMin;
    scalar conservationRelativeTolerance;
    scalar maxPhaseScalingCorrection;
    label angleParallelThreads;
};

struct MieDoSnapshot
{
    const scalarField& epsilonS;
    const scalarField& particleTemperatureK;
    const scalarField& representativeDiameterM;
    const volScalarField& wallTemperatureK;
    const List<scalarField>& coupledWallEmissivity;
};

struct MieDoResult
{
    scalarField incidentRadiationG_Wm2;
    scalarField particlePowerDensityWm3;
    List<scalarField> boundaryQrOutwardWm2;
    scalar integratedParticlePowerW;
    scalar integratedBoundaryPowerW;
    scalar conservationResidualW;
    label sourceIterations;
    label angleThreadsUsed;
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
        MieDoRadiationWorkspace& workspace
    ) const;
};

}                        
}                  

#endif
