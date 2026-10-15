#include "GpuParticleRadiationCoupler.H"

#include "MieDoRadiationMath.H"
#include "surfaceFields.H"
#include "wallPolyPatch.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace Foam
{
namespace gpuThermal
{
namespace
{
[[noreturn]] void materialFailure
(
    const std::string& reason,
    const label outerIteration = 0,
    const label cellI = -1,
    const label patchI = -1,
    const label faceI = -1,
    const scalar residual = std::numeric_limits<scalar>::quiet_NaN()
)
{
    std::ostringstream message;
    message.precision(17);
    message
        << reason
        << "; outerIteration=" << outerIteration
        << "; cell=" << cellI
        << "; patch=" << patchI
        << "; face=" << faceI
        << "; residual=" << residual;
    throw ParticleRadiationSolveError(message.str());
}

bool isCoupledPatch
(
    const labelList& coupledSolidPatchIds,
    const label patchI
)
{
    forAll(coupledSolidPatchIds, coupledI)
    {
        if (coupledSolidPatchIds[coupledI] == patchI)
        {
            return true;
        }
    }
    return false;
}

void validateRadiationResult
(
    const detail::ParticleRadiationTransactionContext& context,
    const MieDoResult& result,
    const label outerIteration
)
{
    const fvMesh& mesh = context.mesh;
    if
    (
        result.incidentRadiationG_Wm2.size() < mesh.nCells()
     || result.particlePowerDensityWm3.size() < mesh.nCells()
     || result.absorptionCoefficientInvM.size() < mesh.nCells()
     || result.scatteringCoefficientInvM.size() < mesh.nCells()
     || result.extinctionCoefficientInvM.size() < mesh.nCells()
     || result.asymmetryFactor.size() < mesh.nCells()
     || result.boundaryQrOutwardWm2.size() < mesh.boundary().size()
    )
    {
        materialFailure("DO result sizes do not match the transaction mesh", outerIteration);
    }

    forAll(mesh.boundary(), patchI)
    {
        if
        (
            result.boundaryQrOutwardWm2[patchI].size()
         < mesh.boundary()[patchI].size()
        )
        {
            materialFailure
            (
                "DO boundary flux patch size mismatch",
                outerIteration,
                -1,
                patchI
            );
        }
    }
}

}             

namespace detail
{
scalar clampParticleRadiationTableValue
(
    const scalar value,
    const scalar minimum,
    const scalar maximum
)
{
    if
    (
        !finiteParticleRadiationScalar(value)
     || !finiteParticleRadiationScalar(minimum)
     || !finiteParticleRadiationScalar(maximum)
     || minimum > maximum
    )
    {
        return value;
    }

    if (value < minimum)
    {
        return minimum;
    }
    if (value > maximum)
    {
        return maximum;
    }
    return value;
}

scalar particleCellCapacityJPerK
(
    const ParticleRadiationMaterialSnapshot& snapshot,
    const label cellI
)
{
    return
        snapshot.particleMassInCellKg[cellI]*snapshot.particleHeatFactorJkgK;
}

void validateParticleRadiationSnapshot
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot
)
{
    const fvMesh& mesh = context.mesh;

    if
    (
        snapshot.epsilonS.size() < mesh.nCells()
     || snapshot.particleMassInCellKg.size() < mesh.nCells()
     || snapshot.particleMeanTemperatureK.size() < mesh.nCells()
     || snapshot.representativeDiameterM.size() < mesh.nCells()
    )
    {
        materialFailure("particle-radiation snapshot cell arrays do not match fvMesh::nCells");
    }
    if (snapshot.coupledWallEmissivity.size() < mesh.boundary().size()
        || snapshot.wallTemperatureK.boundaryField().size() < mesh.boundary().size())
    {
        materialFailure("particle-radiation emissivity patch count mismatch");
    }

    forAll(context.coupledSolidPatchIds, coupledI)
    {
        const label patchI = context.coupledSolidPatchIds[coupledI];
        if
        (
            patchI < 0
         || patchI >= mesh.boundary().size()
        )
        {
            materialFailure("invalid coupled solid patch id", 0, -1, patchI);
        }
    }

    forAll(mesh.boundary(), patchI)
    {
        const scalarField& wallTemperature =
            snapshot.wallTemperatureK.boundaryField()[patchI];
        const scalarField& emissivity = snapshot.coupledWallEmissivity[patchI];
        if
        (
            wallTemperature.size() < mesh.boundary()[patchI].size()
         || emissivity.size() < mesh.boundary()[patchI].size()
        )
        {
            materialFailure("wall-temperature/emissivity face count mismatch", 0, -1, patchI);
        }
    }
}

ParticleRadiationMaterialResult solveValidatedParticleRadiationTransaction
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    const std::function<MieDoResult(const scalarField&)>& solveRadiation
)
try
{
    ParticleRadiationMaterialResult result;
    result.deltaParticleRadiationEnergyJ.setSize
    (
        context.mesh.nCells(), scalar(0)
    );
    result.solidWallRadiationEnergyJ.setSize
    (
        context.mesh.boundary().size()
    );
    forAll(context.mesh.boundary(), patchI)
    {
        result.solidWallRadiationEnergyJ[patchI].setSize
        (
            context.mesh.boundary()[patchI].size(), scalar(0)
        );
    }
    result.escapedBoundaryRadiationEnergyJ = scalar(0);
    const scalar couplingDeltaT = snapshot.elapsedCouplingTimeS;

    scalarField currentTemperatureK(snapshot.particleMeanTemperatureK);
    currentTemperatureK.setSize(context.mesh.nCells());
    const MieDoResult radiationResult = solveRadiation(currentTemperatureK);
    validateRadiationResult(context, radiationResult, 1);
    result.absorptionCoefficientInvM = radiationResult.absorptionCoefficientInvM;
    result.scatteringCoefficientInvM = radiationResult.scatteringCoefficientInvM;
    result.extinctionCoefficientInvM = radiationResult.extinctionCoefficientInvM;
    result.asymmetryFactor = radiationResult.asymmetryFactor;

    const scalar legalMinimumTemperatureK = max
    (
        context.particleTemperatureMinK,
        context.mieTable.minTemperatureK()
    );
    const scalar legalMaximumTemperatureK = min
    (
        context.particleTemperatureMaxK,
        context.mieTable.maxTemperatureK()
    );
    scalar acceptedParticleEnergyJ = scalar(0);
    scalar grossTransactionEnergyJ = scalar(0);
    forAll(currentTemperatureK, cellI)
    {
        const scalar rawDeltaEnergyJ = couplingDeltaT
           *context.mesh.V()[cellI]
           *radiationResult.particlePowerDensityWm3[cellI];

        const bool radiationActive =
            snapshot.epsilonS[cellI] > context.radiationActiveEpsilonMin;
        if (!radiationActive)
        {
            continue;
        }

        const scalar capacityJPerK = particleCellCapacityJPerK(snapshot, cellI);

        const scalar oldTemperatureK = snapshot.particleMeanTemperatureK[cellI];
        const scalar incidentRadiationG_Wm2 =
            radiationResult.incidentRadiationG_Wm2[cellI];
        const scalar representativeDiameterM =
            snapshot.representativeDiameterM[cellI];
        const scalar epsilonS = snapshot.epsilonS[cellI];
        const scalar cellVolumeM3 = context.mesh.V()[cellI];
        const auto radiativeBalanceAtTemperature =
        [&](const scalar temperatureK)
        {
            const MieOpticalSample optical =
                context.mieTable.query(representativeDiameterM, temperatureK);
            return incidentRadiationG_Wm2
              - fourPiConstant()*bandBlackBodyIntensity
                (
                    temperatureK,
                    optical.planckBandFraction
                );
        };
        const auto powerAtTemperatureW =
        [&](const scalar temperatureK)
        {
            const MieOpticalSample optical =
                context.mieTable.query(representativeDiameterM, temperatureK);
            const scalar absorptionCoefficient =
                scalar(3)*epsilonS*optical.Qabs
               /(scalar(2)*representativeDiameterM);
            const scalar powerW = cellVolumeM3*particleRadiationPowerDensity
            (
                absorptionCoefficient,
                incidentRadiationG_Wm2,
                bandBlackBodyIntensity
                (
                    temperatureK,
                    optical.planckBandFraction
                )
            );

            return powerW;
        };

        detail::ImplicitTemperatureRootResult root
        {
            oldTemperatureK,
            scalar(0),
            0
        };
        try
        {
            root = detail::solveBoundedImplicitMaterialTemperature
            (
                oldTemperatureK,
                legalMinimumTemperatureK,
                legalMaximumTemperatureK,
                capacityJPerK,
                couplingDeltaT,
                100,
                scalar(1e-10),
                scalar(1e-9),
                radiativeBalanceAtTemperature,
                powerAtTemperatureW
            );
        }
        catch (const ParticleRadiationSolveError& error)
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "implicit radiation cell solve failed: " << error.what()
                << "; epsilonS=" << epsilonS
                << "; particleMassKg=" << snapshot.particleMassInCellKg[cellI]
                << "; capacityJPerK=" << capacityJPerK
                << "; representativeDiameterM=" << representativeDiameterM
                << "; incidentRadiationG_Wm2=" << incidentRadiationG_Wm2
                << "; rawDeltaEnergyJ=" << rawDeltaEnergyJ;
            materialFailure(reason.str(), 1, cellI);
        }
        const scalar acceptedEnergyJ =
            capacityJPerK*(root.temperatureK - oldTemperatureK);

        currentTemperatureK[cellI] = root.temperatureK;
        result.deltaParticleRadiationEnergyJ[cellI] = acceptedEnergyJ;
        acceptedParticleEnergyJ += acceptedEnergyJ;
        grossTransactionEnergyJ += mag(acceptedEnergyJ);
    }

    List<scalarField> rawBoundaryEnergyJ(context.mesh.boundary().size());
    scalar rawBoundaryTotalEnergyJ = scalar(0);
    forAll(context.mesh.boundary(), patchI)
    {
        const vectorField& faceAreas =
            context.mesh.Sf().boundaryField()[patchI];
        rawBoundaryEnergyJ[patchI].setSize(faceAreas.size(), scalar(0));
        forAll(faceAreas, faceI)
        {
            const scalar boundaryEnergyJ = couplingDeltaT
               *mag(faceAreas[faceI])
               *radiationResult.boundaryQrOutwardWm2[patchI][faceI];

            rawBoundaryEnergyJ[patchI][faceI] = boundaryEnergyJ;
            rawBoundaryTotalEnergyJ += boundaryEnergyJ;
            grossTransactionEnergyJ += mag(boundaryEnergyJ);
        }
    }

    const detail::ConservativeBoundaryEnergyScaleResult boundaryScale =
        detail::conservativeBoundaryEnergyScale
        (
            acceptedParticleEnergyJ,
            rawBoundaryTotalEnergyJ,
            grossTransactionEnergyJ,
            scalar(1e-10)
        );

    forAll(context.mesh.boundary(), patchI)
    {
        const bool coupled = isCoupledPatch
        (
            context.coupledSolidPatchIds,
            patchI
        );
        forAll(rawBoundaryEnergyJ[patchI], faceI)
        {
            const scalar boundaryEnergyJ =
                boundaryScale.scale*rawBoundaryEnergyJ[patchI][faceI];

            if (coupled)
            {
                result.solidWallRadiationEnergyJ[patchI][faceI] =
                    boundaryEnergyJ;
            }
            else
            {
                result.escapedBoundaryRadiationEnergyJ += boundaryEnergyJ;
            }
        }
    }

    result.finalMeanTemperatureK = currentTemperatureK;
    return result;

}
catch (const ParticleRadiationSolveError&)
{
    throw;
}
catch (const std::exception& error)
{
    throw ParticleRadiationSolveError
    (
        std::string("particle-radiation transaction failed: ") + error.what()
    );
}

}                    

GpuParticleRadiationCoupler::GpuParticleRadiationCoupler
(
    const fvMesh& mesh,
    const MieTable& mieTable,
    const MieDoRadiationSolver& radiationSolver,
    const labelList& coupledSolidPatchIds,
    const scalar particleTemperatureMinK,
    const scalar particleTemperatureMaxK
)
:
    mesh_(mesh),
    mieTable_(mieTable),
    radiationSolver_(radiationSolver),
    coupledSolidPatchIds_(coupledSolidPatchIds),
    particleTemperatureMinK_(particleTemperatureMinK),
    particleTemperatureMaxK_(particleTemperatureMaxK),
    radiationWorkspace_(radiationSolver.makeWorkspace())
{
}

ParticleRadiationMaterialResult GpuParticleRadiationCoupler::solve
(
    const ParticleRadiationMaterialSnapshot& snapshot
) const
try
{
    const detail::ParticleRadiationTransactionContext context
    {
        mesh_,
        mieTable_,
        coupledSolidPatchIds_,
        particleTemperatureMinK_,
        particleTemperatureMaxK_,
        radiationSolver_.controls().radiationActiveEpsilonMin
    };

    return detail::solveParticleRadiationTransaction
    (
        context,
        snapshot,
        [&]()
        {
            return &radiationWorkspace_;
        },
        [&](MieDoRadiationWorkspace*& workspace, const MieDoSnapshot& trial)
        {
            return radiationSolver_.solve(trial, *workspace);
        }
    );
}
catch (const ParticleRadiationSolveError&)
{
    throw;
}
catch (const std::exception& error)
{
    throw ParticleRadiationSolveError
    (
        std::string("particle-radiation coupler failed: ") + error.what()
    );
}

}                        
}                  
