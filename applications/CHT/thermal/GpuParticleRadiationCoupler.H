#ifndef GPU_THERMAL_GPU_PARTICLE_RADIATION_COUPLER_H
#define GPU_THERMAL_GPU_PARTICLE_RADIATION_COUPLER_H

#include "MieDoRadiationSolver.H"
#include "ParticleRadiationImplicitMath.H"
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
    scalar globalConservationResidualJ;
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
    const scalar legalTemperatureMinimumK = max
    (
        context.particleTemperatureMinK,
        context.mieTable.minTemperatureK()
    );
    const scalar legalTemperatureMaximumK = min
    (
        context.particleTemperatureMaxK,
        context.mieTable.maxTemperatureK()
    );
    const bool canonicalCellArraysHaveMeshSize =
        snapshot.epsilonS.size() == context.mesh.nCells()
     && canonicalTemperatureK.size() == context.mesh.nCells()
     && canonicalDiameterM.size() == context.mesh.nCells();
    if (canonicalCellArraysHaveMeshSize)
    {
        forAll(canonicalTemperatureK, cellI)
        {
            if (snapshot.epsilonS[cellI] > context.radiationActiveEpsilonMin)
            {
                canonicalTemperatureK[cellI] = clampParticleRadiationTableValue
                (
                    canonicalTemperatureK[cellI],
                    legalTemperatureMinimumK,
                    legalTemperatureMaximumK
                );
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
        snapshot.elapsedCouplingTimeS
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
            canonicalSnapshot.coupledWallEmissivity
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
        const ParticleRadiationMaterialSnapshot& snapshot
    ) const;
};

}                        
}                  

#endif
