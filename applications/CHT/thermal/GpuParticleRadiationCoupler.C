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
    if
    (
        !detail::finiteParticleRadiationScalar(result.integratedParticlePowerW)
     || !detail::finiteParticleRadiationScalar(result.integratedBoundaryPowerW)
     || !detail::finiteParticleRadiationScalar(result.conservationResidualW)
    )
    {
        materialFailure("DO integrated powers are nonfinite", outerIteration);
    }

    for (label cellI = 0; cellI < mesh.nCells(); ++cellI)
    {
        const scalar absorption = result.absorptionCoefficientInvM[cellI];
        const scalar scattering = result.scatteringCoefficientInvM[cellI];
        const scalar extinction = result.extinctionCoefficientInvM[cellI];
        const scalar asymmetry = result.asymmetryFactor[cellI];
        if
        (
            !detail::finiteParticleRadiationScalar
            (
                result.incidentRadiationG_Wm2[cellI]
            )
         || !detail::finiteParticleRadiationScalar
            (
                result.particlePowerDensityWm3[cellI]
            )
         || !detail::finiteParticleRadiationScalar(absorption)
         || !detail::finiteParticleRadiationScalar(scattering)
         || !detail::finiteParticleRadiationScalar(extinction)
         || !detail::finiteParticleRadiationScalar(asymmetry)
        )
        {
            materialFailure
            (
                "DO cell radiation result is nonfinite",
                outerIteration,
                cellI
            );
        }
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
        for (label faceI = 0; faceI < mesh.boundary()[patchI].size(); ++faceI)
        {
            if
            (
                !detail::finiteParticleRadiationScalar
                (
                    result.boundaryQrOutwardWm2[patchI][faceI]
                )
            )
            {
                materialFailure
                (
                    "DO boundary flux is nonfinite",
                    outerIteration,
                    -1,
                    patchI,
                    faceI
                );
            }
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
        !finiteParticleRadiationScalar(context.radiationActiveEpsilonMin)
     || !finiteParticleRadiationScalar(context.particleTemperatureMinK)
     || !finiteParticleRadiationScalar(context.particleTemperatureMaxK)
    )
    {
        materialFailure("invalid particle-radiation transaction controls");
    }

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

    for (label cellI = 0; cellI < mesh.nCells(); ++cellI)
    {
        const scalar epsilon = snapshot.epsilonS[cellI];
        const scalar massKg = snapshot.particleMassInCellKg[cellI];
        const scalar temperatureK = snapshot.particleMeanTemperatureK[cellI];
        const scalar diameterM = snapshot.representativeDiameterM[cellI];
        if (!finiteParticleRadiationScalar(epsilon))
        {
            materialFailure("epsilonS is nonfinite", 0, cellI);
        }
        if (!finiteParticleRadiationScalar(massKg))
        {
            materialFailure("particle mass is nonfinite", 0, cellI);
        }
        if (!finiteParticleRadiationScalar(temperatureK))
        {
            materialFailure("particle temperature is nonfinite", 0, cellI);
        }
        if (!finiteParticleRadiationScalar(diameterM))
        {
            materialFailure("representative diameter is nonfinite", 0, cellI);
        }
        const bool radiationActive =
            epsilon > context.radiationActiveEpsilonMin;
        if
        (
            massKg == scalar(0)
         && radiationActive
        )
        {
            materialFailure("nonzero optical material has zero microscopic particle mass", 0, cellI);
        }
        const scalar capacityJPerK = particleCellCapacityJPerK(snapshot, cellI);
        if
        (
            !finiteParticleRadiationScalar(capacityJPerK)
         || (radiationActive && capacityJPerK == scalar(0))
        )
        {
            materialFailure("particle cell thermal capacity is invalid", 0, cellI);
        }
        if
        (
            !finiteParticleRadiationScalar(mesh.V()[cellI])
        )
        {
            materialFailure("particle-radiation cell volume is invalid", 0, cellI);
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
        for (label faceI = 0; faceI < mesh.boundary()[patchI].size(); ++faceI)
        {
            if (!finiteParticleRadiationScalar(wallTemperature[faceI]))
            {
                materialFailure("wall temperature is nonfinite", 0, -1, patchI, faceI);
            }
            if
            (
                !finiteParticleRadiationScalar(emissivity[faceI])
            )
            {
                materialFailure("coupled-wall emissivity is nonfinite", 0, -1, patchI, faceI);
            }
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
    result.globalConservationResidualJ = scalar(0);
    const scalar couplingDeltaT = snapshot.elapsedCouplingTimeS;
    if (!finiteParticleRadiationScalar(couplingDeltaT))
    {
        materialFailure("invalid implicit radiation coupling duration");
    }

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
        if (!finiteParticleRadiationScalar(rawDeltaEnergyJ))
        {
            materialFailure
            (
                "raw radiation particle energy is nonfinite",
                1,
                cellI
            );
        }

        const bool radiationActive =
            snapshot.epsilonS[cellI] > context.radiationActiveEpsilonMin;
        if (!radiationActive)
        {
            if (rawDeltaEnergyJ != scalar(0))
            {
                materialFailure
                (
                    "inactive cell received radiation energy",
                    1,
                    cellI
                );
            }
            continue;
        }

        const scalar capacityJPerK = particleCellCapacityJPerK(snapshot, cellI);
        if (capacityJPerK == scalar(0))
        {
            materialFailure
            (
                "active cell has zero radiation thermal capacity",
                1,
                cellI
            );
        }

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
            if
            (
                !finiteParticleRadiationScalar(absorptionCoefficient)
                 || !finiteParticleRadiationScalar(powerW)
            )
            {
                materialFailure
                (
                    "implicit radiation material power is invalid",
                    1,
                    cellI
                );
            }
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
        if
        (
            !finiteParticleRadiationScalar(root.temperatureK)
         || !finiteParticleRadiationScalar(acceptedEnergyJ)
        )
        {
            materialFailure
            (
                "implicit radiation material solution is invalid",
                1,
                cellI,
                -1,
                -1,
                root.temperatureK
            );
        }
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
            if (!finiteParticleRadiationScalar(boundaryEnergyJ))
            {
                materialFailure
                (
                    "raw radiation boundary energy is nonfinite",
                    1,
                    -1,
                    patchI,
                    faceI
                );
            }
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

    scalar appliedBoundaryEnergyJ = scalar(0);
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
            if (!finiteParticleRadiationScalar(boundaryEnergyJ))
            {
                materialFailure("applied radiation boundary energy is nonfinite", 1, -1, patchI, faceI);
            }
            appliedBoundaryEnergyJ += boundaryEnergyJ;
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

    // Retain the original diagnostic arithmetic after the one physical scaling.
    // A newly admitted cancellation may have no second diagnostic divisor;
    // in that case report the actual accepted-plus-applied sum, without refusal.
    try
    {
        const detail::ConservativeBoundaryEnergyScaleResult appliedBalance =
            detail::conservativeBoundaryEnergyScale
            (
                acceptedParticleEnergyJ,
                appliedBoundaryEnergyJ,
                grossTransactionEnergyJ,
                scalar(1e-10)
            );
        result.globalConservationResidualJ = appliedBalance.residualJ;
    }
    catch (const ParticleRadiationSolveError&)
    {
        result.globalConservationResidualJ = acceptedParticleEnergyJ + appliedBoundaryEnergyJ;
        if (!finiteParticleRadiationScalar(result.globalConservationResidualJ))
        {
            materialFailure("applied radiation-boundary energy is nonfinite");
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
    if
    (
        !detail::finiteParticleRadiationScalar(particleTemperatureMinK)
     || !detail::finiteParticleRadiationScalar(particleTemperatureMaxK)
    )
    {
        materialFailure("invalid particle-radiation coupler controls/bounds");
    }
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
