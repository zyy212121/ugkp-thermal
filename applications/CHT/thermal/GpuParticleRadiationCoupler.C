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

    ugkp::gpu::ParticleRadiationMaterialInput input;
    const label cells=context.mesh.nCells();
    input.epsilon.assign(snapshot.epsilonS.begin(),snapshot.epsilonS.begin()+cells);
    input.mass.assign(snapshot.particleMassInCellKg.begin(),snapshot.particleMassInCellKg.begin()+cells);
    input.temperature.assign(currentTemperatureK.begin(),currentTemperatureK.end());
    input.diameter.assign(snapshot.representativeDiameterM.begin(),snapshot.representativeDiameterM.begin()+cells);
    input.incident.assign(radiationResult.incidentRadiationG_Wm2.begin(),radiationResult.incidentRadiationG_Wm2.begin()+cells);
    input.volume.assign(context.mesh.V().begin(),context.mesh.V().end());
    input.heatFactor=snapshot.particleHeatFactorJkgK;input.elapsed=couplingDeltaT;
    input.minimumTemperature=max(context.particleTemperatureMinK,context.mieTable.minTemperatureK());
    input.maximumTemperature=min(context.particleTemperatureMaxK,context.mieTable.maxTemperatureK());
    input.activeEpsilonMinimum=context.radiationActiveEpsilonMin;
    forAll(context.mesh.boundary(),patchI){const bool coupled=isCoupledPatch(context.coupledSolidPatchIds,patchI);
        const vectorField& areas=context.mesh.Sf().boundaryField()[patchI];
        forAll(areas,faceI){input.boundaryArea.push_back(mag(areas[faceI]));input.boundaryFlux.push_back(radiationResult.boundaryQrOutwardWm2[patchI][faceI]);input.coupledBoundary.push_back(coupled);}}
    // Production couplers retain the table and workspace. Standalone callers
    // own one temporary workspace and execute the identical CUDA operator.
    std::unique_ptr<ugkp::gpu::ParticleRadiationMaterial> temporary;
    auto* device=context.materialDevice;
    if(!device){temporary.reset(new ugkp::gpu::ParticleRadiationMaterial(context.mieTable.exportGpuTable()));device=temporary.get();}
    const auto completed=device->solve(input);
    result.finalMeanTemperatureK.setSize(cells);
    for(label cellI=0;cellI<cells;++cellI){result.finalMeanTemperatureK[cellI]=completed.temperature[cellI];result.deltaParticleRadiationEnergyJ[cellI]=completed.energy[cellI];}
    std::size_t boundary=0;
    forAll(result.solidWallRadiationEnergyJ,patchI)forAll(result.solidWallRadiationEnergyJ[patchI],faceI)
        result.solidWallRadiationEnergyJ[patchI][faceI]=completed.boundaryEnergy[boundary++];
    result.escapedBoundaryRadiationEnergyJ=completed.escapedEnergy;
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
    radiationWorkspace_(radiationSolver.makeWorkspace()),
    materialDevice_(new ugkp::gpu::ParticleRadiationMaterial(mieTable.exportGpuTable()))
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
        radiationSolver_.controls().radiationActiveEpsilonMin,
        materialDevice_.get()
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
