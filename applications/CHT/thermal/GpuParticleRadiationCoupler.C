#include "GpuParticleRadiationCoupler.H"

#include "MieDoRadiationMath.H"
#include "surfaceFields.H"
#include "wallPolyPatch.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <set>
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

bool sameLabels(const labelList& first, const labelList& second)
{
    if (first.size() != second.size())
    {
        return false;
    }
    forAll(first, i)
    {
        if (first[i] != second[i])
        {
            return false;
        }
    }
    return true;
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
        result.incidentRadiationG_Wm2.size() != mesh.nCells()
     || result.particlePowerDensityWm3.size() != mesh.nCells()
     || result.absorptionCoefficientInvM.size() != mesh.nCells()
     || result.scatteringCoefficientInvM.size() != mesh.nCells()
     || result.extinctionCoefficientInvM.size() != mesh.nCells()
     || result.asymmetryFactor.size() != mesh.nCells()
     || result.boundaryQrOutwardWm2.size() != mesh.boundary().size()
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

    forAll(result.incidentRadiationG_Wm2, cellI)
    {
        const scalar absorption = result.absorptionCoefficientInvM[cellI];
        const scalar scattering = result.scatteringCoefficientInvM[cellI];
        const scalar extinction = result.extinctionCoefficientInvM[cellI];
        const scalar asymmetry = result.asymmetryFactor[cellI];
        const scalar extinctionTolerance = scalar(1024)
           *std::numeric_limits<scalar>::epsilon()
           *max(scalar(1), mag(extinction));
        if
        (
            !detail::finiteParticleRadiationScalar
            (
                result.incidentRadiationG_Wm2[cellI]
            )
         || result.incidentRadiationG_Wm2[cellI] < scalar(0)
         || !detail::finiteParticleRadiationScalar
            (
                result.particlePowerDensityWm3[cellI]
            )
         || !detail::finiteParticleRadiationScalar(absorption)
         || !detail::finiteParticleRadiationScalar(scattering)
         || !detail::finiteParticleRadiationScalar(extinction)
         || !detail::finiteParticleRadiationScalar(asymmetry)
         || absorption < scalar(0)
         || scattering < scalar(0)
         || extinction < scalar(0)
         || mag(extinction - absorption - scattering) > extinctionTolerance
         || asymmetry < scalar(-1)
         || asymmetry > scalar(1)
        )
        {
            materialFailure
            (
                "DO cell radiation result is negative/nonfinite",
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
         != mesh.boundary()[patchI].size()
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
        forAll(result.boundaryQrOutwardWm2[patchI], faceI)
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

#if 0                                                                         
scalar relativeTemperatureChange
(
    const scalarField& previous,
    const scalarField& current
)
{
    scalar maximumChange = scalar(0);
    forAll(current, cellI)
    {
        maximumChange = max
        (
            maximumChange,
            mag(current[cellI] - previous[cellI])
           /max(max(mag(current[cellI]), mag(previous[cellI])), scalar(1))
        );
    }
    return maximumChange;
}

bool fullyCoupledMaterialResidualsPass
(
    const detail::ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    const scalarField& temperatureK,
    const MieDoResult& radiationResult,
    const label outerIteration
)
{
    const scalarField& volumes = context.mesh.V();
    bool allPassed = true;
    forAll(temperatureK, cellI)
    {
        const scalar capacityJPerK =
            detail::particleCellCapacityJPerK(snapshot, cellI);
        const scalar capacityEnergyJ = capacityJPerK
           *(temperatureK[cellI] - snapshot.particleMeanTemperatureK[cellI]);
        const scalar radiationEnergyJ = snapshot.elapsedCouplingTimeS
           *volumes[cellI]*radiationResult.particlePowerDensityWm3[cellI];
        const scalar residualJ = capacityEnergyJ - radiationEnergyJ;
        const scalar scaleJ = max
        (
            max(mag(capacityEnergyJ), mag(radiationEnergyJ)),
            std::numeric_limits<scalar>::min()
        );
        const scalar relativeResidual = mag(residualJ)/scaleJ;
        if
        (
            !detail::finiteParticleRadiationScalar(relativeResidual)
         || relativeResidual > context.controls.materialEnergyRelativeTolerance
        )
        {
            allPassed = false;
        }

        const bool radiationInactive =
            snapshot.epsilonS[cellI] <= context.radiationActiveEpsilonMin;
        const bool pureScattering =
            !radiationInactive && cellIsPureScattering(context, snapshot, cellI);
        if
        (
            (radiationInactive || pureScattering)
         &&
            (
                temperatureK[cellI] != snapshot.particleMeanTemperatureK[cellI]
             || radiationResult.particlePowerDensityWm3[cellI] != scalar(0)
            )
        )
        {
            materialFailure
            (
                "inactive/pure-scattering cell changed material energy",
                outerIteration,
                cellI,
                -1,
                -1,
                residualJ
            );
        }
    }
    return allPassed;
}

scalarField solveFrozenRadiationCells
(
    const detail::ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    const MieDoResult& frozenRadiation,
    const label outerIteration
)
{
    scalarField newTemperatureK(snapshot.particleMeanTemperatureK);
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

    forAll(newTemperatureK, cellI)
    {
        if (snapshot.epsilonS[cellI] <= context.radiationActiveEpsilonMin)
        {
            continue;
        }
        if (cellIsPureScattering(context, snapshot, cellI))
        {
            continue;
        }

        const scalar oldTemperatureK = snapshot.particleMeanTemperatureK[cellI];
        const scalar incidentRadiationG_Wm2 =
            frozenRadiation.incidentRadiationG_Wm2[cellI];
        const auto radiativeBalanceAtTemperature =
        [&](const scalar temperatureK)
        {
            const scalar bandFraction =
                context.mieTable.planckBandFraction(temperatureK);
            return incidentRadiationG_Wm2
              - fourPiConstant()
               *bandBlackBodyIntensity(temperatureK, bandFraction);
        };
        const scalar cellVolumeM3 = context.mesh.V()[cellI];
        const auto powerAtTemperatureW =
        [&](const scalar temperatureK)
        {
            return cellVolumeM3*materialPowerDensityWm3
            (
                context,
                snapshot,
                cellI,
                temperatureK,
                incidentRadiationG_Wm2
            );
        };
        const detail::ImplicitTemperatureRootResult root =
            detail::solveBoundedImplicitMaterialTemperature
            (
                oldTemperatureK,
                legalMinimumTemperatureK,
                legalMaximumTemperatureK,
                detail::particleCellCapacityJPerK(snapshot, cellI),
                snapshot.elapsedCouplingTimeS,
                context.controls.maxCellRootIterations,
                context.controls.temperatureRelativeTolerance,
                context.controls.materialEnergyRelativeTolerance,
                radiativeBalanceAtTemperature,
                powerAtTemperatureW
            );
        newTemperatureK[cellI] = root.temperatureK;
        if
        (
            newTemperatureK[cellI] < legalMinimumTemperatureK
         || newTemperatureK[cellI] > legalMaximumTemperatureK
        )
        {
            materialFailure
            (
                "implicit material root left the legal temperature interval",
                outerIteration,
                cellI
            );
        }
    }
    return newTemperatureK;
}
#endif

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
     || context.radiationActiveEpsilonMin < scalar(0)
     || !finiteParticleRadiationScalar(context.particleTemperatureMinK)
     || !finiteParticleRadiationScalar(context.particleTemperatureMaxK)
     || context.particleTemperatureMinK >= context.particleTemperatureMaxK
    )
    {
        materialFailure("invalid particle-radiation transaction controls");
    }

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
    if (legalMinimumTemperatureK >= legalMaximumTemperatureK)
    {
        materialFailure("GPU and MIE table temperature intervals do not overlap");
    }
    if
    (
        !finiteParticleRadiationScalar(snapshot.particleHeatFactorJkgK)
     || snapshot.particleHeatFactorJkgK <= scalar(0)
     || !finiteParticleRadiationScalar(snapshot.elapsedCouplingTimeS)
     || snapshot.elapsedCouplingTimeS <= scalar(0)
    )
    {
        materialFailure("particle heat factor or actual elapsed coupling time is invalid");
    }
    if
    (
        snapshot.epsilonS.size() != mesh.nCells()
     || snapshot.particleMassInCellKg.size() != mesh.nCells()
     || snapshot.particleMeanTemperatureK.size() != mesh.nCells()
     || snapshot.representativeDiameterM.size() != mesh.nCells()
    )
    {
        materialFailure("particle-radiation snapshot cell arrays do not match fvMesh::nCells");
    }
    if (&snapshot.wallTemperatureK.mesh() != &mesh)
    {
        materialFailure("particle-radiation wall temperature belongs to a different mesh");
    }
    if (snapshot.coupledWallEmissivity.size() != mesh.boundary().size())
    {
        materialFailure("particle-radiation emissivity patch count mismatch");
    }

    std::set<label> uniqueCoupledPatches;
    forAll(context.coupledSolidPatchIds, coupledI)
    {
        const label patchI = context.coupledSolidPatchIds[coupledI];
        if
        (
            patchI < 0
         || patchI >= mesh.boundary().size()
         || !uniqueCoupledPatches.insert(patchI).second
         || !isA<wallPolyPatch>(mesh.boundaryMesh()[patchI])
        )
        {
            materialFailure("invalid coupled solid patch id", 0, -1, patchI);
        }
    }

    forAll(snapshot.epsilonS, cellI)
    {
        const scalar epsilon = snapshot.epsilonS[cellI];
        const scalar massKg = snapshot.particleMassInCellKg[cellI];
        const scalar temperatureK = snapshot.particleMeanTemperatureK[cellI];
        const scalar diameterM = snapshot.representativeDiameterM[cellI];
        if (!finiteParticleRadiationScalar(epsilon) || epsilon < scalar(0))
        {
            materialFailure("epsilonS is negative/nonfinite", 0, cellI);
        }
        if (!finiteParticleRadiationScalar(massKg) || massKg < scalar(0))
        {
            materialFailure("particle mass is negative/nonfinite", 0, cellI);
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
        if
        (
            radiationActive
         &&
            (
                temperatureK < legalMinimumTemperatureK
             || temperatureK > legalMaximumTemperatureK
             || diameterM < context.mieTable.minDiameterM()
             || diameterM > context.mieTable.maxDiameterM()
            )
        )
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "active optical cell lies outside table/GPU bounds"
                << "; epsilonS=" << epsilon
                << "; particleMassKg=" << massKg
                << "; particleTemperatureK=" << temperatureK
                << "; legalTemperatureK=[" << legalMinimumTemperatureK
                << ',' << legalMaximumTemperatureK << ']'
                << "; representativeDiameterM=" << diameterM
                << "; legalDiameterM=[" << context.mieTable.minDiameterM()
                << ',' << context.mieTable.maxDiameterM() << ']';
            materialFailure(reason.str(), 0, cellI);
        }
        if
        (
            !radiationActive
         && massKg > scalar(0)
         &&
            (
                temperatureK < context.particleTemperatureMinK
             || temperatureK > context.particleTemperatureMaxK
             || diameterM <= scalar(0)
            )
        )
        {
            materialFailure("inactive nonempty particle cell lies outside GPU material bounds", 0, cellI);
        }
        const scalar capacityJPerK = particleCellCapacityJPerK(snapshot, cellI);
        if
        (
            !finiteParticleRadiationScalar(capacityJPerK)
         || capacityJPerK < scalar(0)
         || (radiationActive && capacityJPerK <= scalar(0))
        )
        {
            materialFailure("particle cell thermal capacity is invalid", 0, cellI);
        }
        if
        (
            !finiteParticleRadiationScalar(mesh.V()[cellI])
         || mesh.V()[cellI] <= scalar(0)
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
            wallTemperature.size() != mesh.boundary()[patchI].size()
         || emissivity.size() != mesh.boundary()[patchI].size()
        )
        {
            materialFailure("wall-temperature/emissivity face count mismatch", 0, -1, patchI);
        }
        forAll(wallTemperature, faceI)
        {
            if (!finiteParticleRadiationScalar(wallTemperature[faceI]))
            {
                materialFailure("wall temperature is nonfinite", 0, -1, patchI, faceI);
            }
            if
            (
                !finiteParticleRadiationScalar(emissivity[faceI])
             || emissivity[faceI] < scalar(0)
             || emissivity[faceI] > scalar(1)
            )
            {
                materialFailure("coupled-wall emissivity is outside [0,1]", 0, -1, patchI, faceI);
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
    result.finalMeanTemperatureK = snapshot.particleMeanTemperatureK;
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
    if (!finiteParticleRadiationScalar(couplingDeltaT) || couplingDeltaT <= scalar(0))
    {
        materialFailure("invalid implicit radiation coupling duration");
    }

    scalarField currentTemperatureK(snapshot.particleMeanTemperatureK);
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
        if (capacityJPerK <= scalar(0))
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
             || absorptionCoefficient < scalar(0)
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
         || root.temperatureK < legalMinimumTemperatureK
         || root.temperatureK > legalMaximumTemperatureK
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

    const detail::ConservativeBoundaryEnergyScaleResult appliedBalance =
        detail::conservativeBoundaryEnergyScale
        (
            acceptedParticleEnergyJ,
            appliedBoundaryEnergyJ,
            grossTransactionEnergyJ,
            scalar(1e-10)
        );
    result.globalConservationResidualJ = appliedBalance.residualJ;
    result.finalMeanTemperatureK = currentTemperatureK;
    return result;

#if 0                                                                        
    const scalarField oldTemperatureK(snapshot.particleMeanTemperatureK);
    scalarField trialTemperatureK(oldTemperatureK);
    MieDoResult radiationResult = solveRadiation(trialTemperatureK);
    validateRadiationResult(context, radiationResult, 0);

    scalarField acceptedTemperatureK;
    label acceptedOuterIterations = 0;
    bool converged = false;

    for
    (
        label outerIteration = 1;
        outerIteration <= context.controls.maxOuterIterations;
        ++outerIteration
    )
    {
        scalarField nextTemperatureK = solveFrozenRadiationCells
        (
            context,
            snapshot,
            radiationResult,
            outerIteration
        );
        MieDoResult nextRadiationResult = solveRadiation(nextTemperatureK);
        validateRadiationResult(context, nextRadiationResult, outerIteration);

        const scalar maximumRelativeTemperatureChange =
            relativeTemperatureChange(trialTemperatureK, nextTemperatureK);
        const bool materialResidualsPass = fullyCoupledMaterialResidualsPass
        (
            context,
            snapshot,
            nextTemperatureK,
            nextRadiationResult,
            outerIteration
        );
        if
        (
            maximumRelativeTemperatureChange
         <= context.controls.temperatureRelativeTolerance
         && materialResidualsPass
        )
        {
            acceptedTemperatureK.transfer(nextTemperatureK);
            radiationResult = std::move(nextRadiationResult);
            acceptedOuterIterations = outerIteration;
            converged = true;
            break;
        }

        trialTemperatureK.transfer(nextTemperatureK);
        radiationResult = std::move(nextRadiationResult);
    }

    if (!converged)
    {
        materialFailure
        (
            "particle-radiation outer iteration reached its limit without convergence",
            context.controls.maxOuterIterations
        );
    }

    MieDoResult finalRadiationResult = solveRadiation(acceptedTemperatureK);
    validateRadiationResult
    (
        context,
        finalRadiationResult,
        acceptedOuterIterations
    );
    if
    (
        !fullyCoupledMaterialResidualsPass
        (
            context,
            snapshot,
            acceptedTemperatureK,
            finalRadiationResult,
            acceptedOuterIterations
        )
    )
    {
        materialFailure
        (
            "final DO solve failed the fully coupled material residual",
            acceptedOuterIterations
        );
    }

    ParticleRadiationMaterialResult result;
    result.finalMeanTemperatureK = acceptedTemperatureK;
    result.deltaParticleRadiationEnergyJ.setSize(context.mesh.nCells(), scalar(0));
    result.solidWallRadiationEnergyJ.setSize(context.mesh.boundary().size());
    result.escapedBoundaryRadiationEnergyJ = scalar(0);
    result.globalConservationResidualJ = scalar(0);
    result.outerIterations = acceptedOuterIterations;

    scalar totalParticleEnergyJ = scalar(0);
    scalar grossEnergyJ = scalar(0);
    scalar roundoffReferenceEnergyJ = scalar(0);
    forAll(result.deltaParticleRadiationEnergyJ, cellI)
    {
        const scalar capacityJPerK = particleCellCapacityJPerK(snapshot, cellI);
        result.deltaParticleRadiationEnergyJ[cellI] = capacityJPerK
           *(
                result.finalMeanTemperatureK[cellI]
              - snapshot.particleMeanTemperatureK[cellI]
            );
        totalParticleEnergyJ += result.deltaParticleRadiationEnergyJ[cellI];
        grossEnergyJ += mag(result.deltaParticleRadiationEnergyJ[cellI]);
        if (snapshot.epsilonS[cellI] > context.radiationActiveEpsilonMin)
        {
            const MieOpticalSample optical = context.mieTable.query
            (
                snapshot.representativeDiameterM[cellI],
                result.finalMeanTemperatureK[cellI]
            );
            const scalar absorptionCoefficient =
                scalar(3)*snapshot.epsilonS[cellI]*optical.Qabs
               /(
                    scalar(2)
                   *snapshot.representativeDiameterM[cellI]
                );
            const scalar emissionIntensity = bandBlackBodyIntensity
            (
                result.finalMeanTemperatureK[cellI],
                optical.planckBandFraction
            );
            roundoffReferenceEnergyJ += snapshot.elapsedCouplingTimeS
               *context.mesh.V()[cellI]*absorptionCoefficient
               *(
                    mag(finalRadiationResult.incidentRadiationG_Wm2[cellI])
                  + fourPiConstant()*mag(emissionIntensity)
                );
        }
    }

    scalar totalBoundaryEnergyJ = scalar(0);
    forAll(context.mesh.boundary(), patchI)
    {
        const vectorField& faceArea =
            context.mesh.Sf().boundaryField()[patchI];
        const labelUList& faceCells = context.mesh.boundary()[patchI].faceCells();
        result.solidWallRadiationEnergyJ[patchI].setSize
        (
            faceArea.size(),
            scalar(0)
        );
        const bool coupled = isCoupledPatch
        (
            context.coupledSolidPatchIds,
            patchI
        );
        forAll(faceArea, faceI)
        {
            const scalar outwardPowerW =
                mag(faceArea[faceI])
               *finalRadiationResult.boundaryQrOutwardWm2[patchI][faceI];
            const scalar boundaryEnergyJ =
                snapshot.elapsedCouplingTimeS*outwardPowerW;
            totalBoundaryEnergyJ += boundaryEnergyJ;
            grossEnergyJ += mag(boundaryEnergyJ);
            roundoffReferenceEnergyJ += snapshot.elapsedCouplingTimeS
               *mag(faceArea[faceI])
               *(
                    mag(finalRadiationResult.boundaryQrOutwardWm2[patchI][faceI])
                  + mag
                    (
                        finalRadiationResult.incidentRadiationG_Wm2
                        [faceCells[faceI]]
                    )
                );
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

    result.globalConservationResidualJ =
        totalParticleEnergyJ + totalBoundaryEnergyJ;
    const scalar machineRoundoffToleranceJ =
        scalar(4096)*std::numeric_limits<scalar>::epsilon()
       *max(roundoffReferenceEnergyJ, grossEnergyJ);
    const scalar globalToleranceJ =
        context.controls.globalConservationRelativeTolerance
       *grossEnergyJ
      + machineRoundoffToleranceJ;
    if
    (
        !finiteParticleRadiationScalar(result.globalConservationResidualJ)
     || mag(result.globalConservationResidualJ) > globalToleranceJ
    )
    {
        materialFailure
        (
            "global particle-plus-boundary radiation energy conservation failed",
            acceptedOuterIterations,
            -1,
            -1,
            -1,
            result.globalConservationResidualJ
        );
    }

    return result;
#endif
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
    if (&mesh != &radiationSolver.mesh())
    {
        materialFailure("particle-radiation coupler mesh identity mismatch");
    }
    if (!mieTable.sharesDataWith(radiationSolver.mieTable()))
    {
        materialFailure("particle-radiation coupler MIE table storage identity mismatch");
    }
    if
    (
        !sameLabels
        (
            coupledSolidPatchIds,
            radiationSolver.coupledSolidPatchIds()
        )
    )
    {
        materialFailure("particle-radiation coupler solid patch identity mismatch");
    }
    if
    (
        !detail::finiteParticleRadiationScalar(particleTemperatureMinK)
     || !detail::finiteParticleRadiationScalar(particleTemperatureMaxK)
     || particleTemperatureMinK >= particleTemperatureMaxK
     || max(particleTemperatureMinK, mieTable.minTemperatureK())
        >= min(particleTemperatureMaxK, mieTable.maxTemperatureK())
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
