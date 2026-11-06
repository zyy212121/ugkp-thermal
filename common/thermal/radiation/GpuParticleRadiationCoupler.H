#ifndef GPU_THERMAL_GPU_PARTICLE_RADIATION_COUPLER_H
#define GPU_THERMAL_GPU_PARTICLE_RADIATION_COUPLER_H

#include "MieDoRadiationSolver.H"
#include "ParticleRadiationImplicitMath.H"
#include "../../../common/gpu/ParticleRadiationMaterial.H"
#include "fvMesh.H"
#include "labelList.H"
#include "scalarField.H"
#include "volFields.H"

#include <functional>

namespace Foam
{
namespace gpuThermal
{

struct ParticleRadiationMaterialSnapshot
{
    const scalarField& epsilonS;
    const scalarField& particleMassInCellKg;
    const scalarField& particleMeanTemperatureK;
    const scalarField& representativeDiameterM;
    const volScalarField& wallTemperatureK;
    const List<scalarField>& coupledWallEmissivity;
    scalar particleHeatFactorJkgK;
    scalar elapsedCouplingTimeS;
    ugkp::gpu::ParticleRadiationResponseView response{};
    ugkp::gpu::RadiationParticleDeviceView resident{};
};

struct ParticleRadiationMaterialResult
{
    scalarField absorptionCoefficientInvM;
    scalarField scatteringCoefficientInvM;
    scalarField extinctionCoefficientInvM;
    scalarField asymmetryFactor;
    scalarField finalMeanTemperatureK;
    scalarField deltaParticleRadiationEnergyJ;
    List<scalarField> solidWallRadiationEnergyJ;
    scalar escapedBoundaryRadiationEnergyJ;
    scalar radiativeEnergyMagnitudeJ=0,acceptedParticleRadiationEnergyJ=0;
    ugkp::gpu::RadiationMaterialDeviceView resident{};
};

namespace detail
{

scalar clampParticleRadiationTableValue
(
    scalar value,
    scalar minimum,
    scalar maximum
);

struct ParticleRadiationTransactionContext
{
    const fvMesh& mesh;
    const MieTable& mieTable;
    const labelList& coupledSolidPatchIds;
    scalar particleTemperatureMinK;
    scalar particleTemperatureMaxK;
    scalar radiationActiveEpsilonMin;
    ugkp::gpu::ParticleRadiationMaterial* materialDevice = nullptr;
    bool exportFields = true;
};

scalar particleCellCapacityJPerK
(
    const ParticleRadiationMaterialSnapshot& snapshot,
    label cellI
);

void validateParticleRadiationSnapshot
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot
);

ParticleRadiationMaterialResult solveValidatedParticleRadiationTransaction
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    const std::function<MieDoResult(const scalarField&)>& solveRadiation
);

template<class MakeWorkspace, class SolveRadiation>
ParticleRadiationMaterialResult solveParticleRadiationTransaction
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    MakeWorkspace makeWorkspace,
    SolveRadiation solveRadiation
)
{
    scalarField canonicalTemperatureK(snapshot.particleMeanTemperatureK);
    scalarField canonicalDiameterM(snapshot.representativeDiameterM);
    // Physical temperature drives black-body emission in both DO and the
    // material response. Mie queries clamp their optical coordinates only.
    const bool canonicalCellArraysHaveMeshSize =
        snapshot.epsilonS.size() >= context.mesh.nCells()
     && canonicalTemperatureK.size() >= context.mesh.nCells()
     && canonicalDiameterM.size() >= context.mesh.nCells();
    if (!snapshot.resident.enabled && canonicalCellArraysHaveMeshSize)
    {
        for (label cellI = 0; cellI < context.mesh.nCells(); ++cellI)
        {
            if (snapshot.epsilonS[cellI] > context.radiationActiveEpsilonMin)
            {
                if (canonicalDiameterM[cellI] > scalar(0))
                {
                    canonicalDiameterM[cellI] = clampParticleRadiationTableValue
                    (
                        canonicalDiameterM[cellI],
                        context.mieTable.minDiameterM(),
                        context.mieTable.maxDiameterM()
                    );
                }
            }
        }
    }
    const ParticleRadiationMaterialSnapshot canonicalSnapshot
    {
        snapshot.epsilonS,
        snapshot.particleMassInCellKg,
        canonicalTemperatureK,
        canonicalDiameterM,
        snapshot.wallTemperatureK,
        snapshot.coupledWallEmissivity,
        snapshot.particleHeatFactorJkgK,
        snapshot.elapsedCouplingTimeS,
        snapshot.response,
        snapshot.resident
    };

    validateParticleRadiationSnapshot(context, canonicalSnapshot);

    auto workspace = makeWorkspace();
    const std::function<MieDoResult(const scalarField&)> boundRadiationSolve =
    [&](const scalarField& trialTemperatureK)
    {
        const MieDoSnapshot trialSnapshot
        {
            canonicalSnapshot.epsilonS,
            trialTemperatureK,
            canonicalSnapshot.representativeDiameterM,
            canonicalSnapshot.wallTemperatureK,
            canonicalSnapshot.coupledWallEmissivity,
            canonicalSnapshot.resident
        };
        return solveRadiation(workspace, trialSnapshot);
    };

    return solveValidatedParticleRadiationTransaction
    (
        context,
        canonicalSnapshot,
        boundRadiationSolve
    );
}

}                    

class GpuParticleRadiationCoupler
{
    const fvMesh& mesh_;
    const MieTable& mieTable_;
    const MieDoRadiationSolver& radiationSolver_;
    const labelList coupledSolidPatchIds_;
    const scalar particleTemperatureMinK_;
    const scalar particleTemperatureMaxK_;
    mutable MieDoRadiationWorkspace radiationWorkspace_;
    std::unique_ptr<ugkp::gpu::ParticleRadiationMaterial> materialDevice_;

public:
    GpuParticleRadiationCoupler
    (
        const fvMesh& mesh,
        const MieTable& mieTable,
        const MieDoRadiationSolver& radiationSolver,
        const labelList& coupledSolidPatchIds,
        scalar particleTemperatureMinK,
        scalar particleTemperatureMaxK
    );

    ParticleRadiationMaterialResult solve
    (
        const ParticleRadiationMaterialSnapshot& snapshot,
        bool exportFields=true
    ) const;
    ParticleRadiationMaterialResult collect() const;
    void settleAcceptedEnergy(const ugkp::gpu::RadiationAcceptedEnergyDeviceView&) const;
    ParticleRadiationMaterialResult collectBoundaryEnergy() const;
};

}                        
}                  

#endif
