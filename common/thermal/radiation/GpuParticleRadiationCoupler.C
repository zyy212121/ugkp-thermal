
#include "FoamSolverScheme.H"
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
ugkp::gpu::ParticleRadiationSolveControls startupMaterialControls(const fvMesh& mesh)
{
    ugkp::gpu::ParticleRadiationSolveControls controls;
    const auto& scheme=ugkp::solverControls::startupScheme(mesh.time());
    scheme.read("radiation.material","boundedMaximumIterations",controls.boundedMaximumIterations,"effectiveDefault",mesh.name());
    scheme.read("radiation.material","boundedTemperatureRelativeTolerance",controls.boundedTemperatureRelativeTolerance,"effectiveDefault",mesh.name());
    scheme.read("radiation.material","boundedEnergyRelativeTolerance",controls.boundedEnergyRelativeTolerance,"effectiveDefault",mesh.name());
    scheme.read("radiation.material","caloricMaximumIterations",controls.caloricMaximumIterations,"effectiveDefault",mesh.name());
    scheme.read("radiation.material","caloricEnergyRelativeTolerance",controls.caloricEnergyRelativeTolerance,"effectiveDefault",mesh.name());
    scheme.read("radiation.material","boundaryEnergyRelativeTolerance",controls.boundaryEnergyRelativeTolerance,"effectiveDefault",mesh.name());
    return controls;
}
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
    if(result.resident.enabled){
        if(result.resident.cells!=mesh.nCells()||result.resident.boundaryFaces!=mesh.nFaces()-mesh.nInternalFaces())
            materialFailure("resident DO result sizes do not match transaction mesh",outerIteration);
        return;
    }
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

namespace {
List<scalarField> mapBoundaryEnergy(const fvMesh& mesh,const std::vector<RadiationReal>& boundary) {
    List<scalarField> result(mesh.boundary().size());
    std::size_t index=0;
    forAll(result,patchI){
        auto& energy=result[patchI];
        energy.setSize(mesh.boundary()[patchI].size());
        forAll(energy,faceI)energy[faceI]=boundary[index++];
    }
    return result;
}
}

namespace detail
{
RadiationReal clampParticleRadiationTableValue
(
    const RadiationReal value,
    const RadiationReal minimum,
    const RadiationReal maximum
)
{
    if
    (
        !std::isfinite(value)
     || !std::isfinite(minimum)
     || !std::isfinite(maximum)
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

RadiationReal particleCellCapacityJPerK
(
    const ParticleRadiationMaterialSnapshot& snapshot,
    const label cellI
)
{
    return
        RadiationReal(snapshot.particleMassInCellKg[cellI])*RadiationReal(snapshot.particleHeatFactorJkgK);
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
        snapshot.resident.enabled ? snapshot.resident.cells != mesh.nCells() :
        (snapshot.epsilonS.size() < mesh.nCells()
     || snapshot.particleMassInCellKg.size() < mesh.nCells()
     || snapshot.particleMeanTemperatureK.size() < mesh.nCells()
     || snapshot.representativeDiameterM.size() < mesh.nCells())
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

static ugkp::gpu::ParticleRadiationMaterialInput materialWindowInput
(const ParticleRadiationTransactionContext& context,const ParticleRadiationMaterialSnapshot& snapshot) {
    ugkp::gpu::ParticleRadiationMaterialInput input;
    const label cells=context.mesh.nCells();
    input.particleResident=snapshot.resident;
    input.volume.assign(context.mesh.V().begin(),context.mesh.V().end());
    if(!snapshot.resident.enabled){
        input.epsilon.assign(snapshot.epsilonS.begin(),snapshot.epsilonS.begin()+cells);
        input.temperature.assign(snapshot.particleMeanTemperatureK.begin(),snapshot.particleMeanTemperatureK.begin()+cells);
        input.diameter.assign(snapshot.representativeDiameterM.begin(),snapshot.representativeDiameterM.begin()+cells);
        input.mass.assign(snapshot.particleMassInCellKg.begin(),snapshot.particleMassInCellKg.begin()+cells);
    }
    input.response=snapshot.response;
    input.heatFactor=snapshot.particleHeatFactorJkgK;input.elapsed=snapshot.elapsedCouplingTimeS;
    input.minimumTemperature=context.particleTemperatureMinK;
    input.maximumTemperature=context.particleTemperatureMaxK;
    input.activeEpsilonMinimum=context.radiationActiveEpsilonMin;
    forAll(context.mesh.boundary(),patchI){const bool coupled=isCoupledPatch(context.coupledSolidPatchIds,patchI);
        const vectorField& areas=context.mesh.Sf().boundaryField()[patchI];
        forAll(areas,faceI){input.boundaryArea.push_back(mag(areas[faceI]));input.coupledBoundary.push_back(coupled);}}
    return input;
}

ParticleRadiationMaterialResult solveValidatedParticleRadiationTransaction
(
    const ParticleRadiationTransactionContext& context,
    const ParticleRadiationMaterialSnapshot& snapshot,
    const std::function<MieDoResult(const scalarField&, ugkp::gpu::RadiationThermalCouplingDeviceView, ugkp::gpu::RadiationThermalIterationHook)>& solveRadiation
)
try
{
    ParticleRadiationMaterialResult result;
    const label cells=context.mesh.nCells();
    scalarField currentTemperatureK;
    if(!snapshot.resident.enabled){
        currentTemperatureK=snapshot.particleMeanTemperatureK;
        currentTemperatureK.setSize(cells);
    }
    const auto input=materialWindowInput(context,snapshot);
    std::unique_ptr<ugkp::gpu::ParticleRadiationMaterial> temporary;
    auto* device=context.materialDevice;
    if(!device){temporary.reset(new ugkp::gpu::ParticleRadiationMaterial(context.mieTable.exportGpuTable(),startupMaterialControls(context.mesh)));device=temporary.get();}
    const auto thermal=device->beginAnchoredWindow(input);
    const auto hook=[device](ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency,void* stream,
        const ugkp::gpu::RadiationThermalIterationDeviceView& view){
        return device->appendAnchoredIteration(graph,dependency,stream,view);
    };
    const MieDoResult radiationResult=solveRadiation(currentTemperatureK,thermal,hook);
    validateRadiationResult(context,radiationResult,1);
    result.absorptionCoefficientInvM=radiationResult.absorptionCoefficientInvM;
    result.scatteringCoefficientInvM=radiationResult.scatteringCoefficientInvM;
    result.extinctionCoefficientInvM=radiationResult.extinctionCoefficientInvM;
    result.asymmetryFactor=radiationResult.asymmetryFactor;
    const auto completed=device->finishAnchoredWindow(radiationResult.resident,context.exportFields||bool(temporary));
    if(!context.exportFields&&!temporary){result.resident=completed.resident;return result;}
    result.finalMeanTemperatureK.setSize(cells);
    result.deltaParticleRadiationEnergyJ.setSize(cells);
    for(label cellI=0;cellI<cells;++cellI){result.finalMeanTemperatureK[cellI]=completed.temperature[cellI];result.deltaParticleRadiationEnergyJ[cellI]=completed.energy[cellI];}
    result.solidWallRadiationEnergyJ=mapBoundaryEnergy(context.mesh,completed.boundaryEnergy);
    result.escapedBoundaryRadiationEnergyJ=completed.escapedEnergy;

    if(!temporary)result.resident=completed.resident;
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
    particleTemperatureMinK_(RadiationReal(particleTemperatureMinK)),
    particleTemperatureMaxK_(RadiationReal(particleTemperatureMaxK)),
    radiationWorkspace_(radiationSolver.makeWorkspace()),
    materialDevice_(new ugkp::gpu::ParticleRadiationMaterial(mieTable.exportGpuTable(),startupMaterialControls(mesh)))
{
}

ParticleRadiationMaterialResult GpuParticleRadiationCoupler::solve
(
    const ParticleRadiationMaterialSnapshot& snapshot,
    bool exportFields
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
        materialDevice_.get(),exportFields
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
            return radiationSolver_.solve(trial, *workspace,exportFields);
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

ParticleRadiationParentBinding GpuParticleRadiationCoupler::prepareParent
(const ParticleRadiationMaterialSnapshot& snapshot,const double* deviceElapsed) const {
    if(!snapshot.resident.enabled||!deviceElapsed)
        materialFailure("parent material response requires resident particles and an interval clock");
    const detail::ParticleRadiationTransactionContext context{
        mesh_,mieTable_,coupledSolidPatchIds_,particleTemperatureMinK_,particleTemperatureMaxK_,
        radiationSolver_.controls().radiationActiveEpsilonMin,materialDevice_.get(),false};
    detail::validateParticleRadiationSnapshot(context,snapshot);
    const auto input=detail::materialWindowInput(context,snapshot);
    auto* material=materialDevice_.get();
    const auto thermal=material->prepareParentWindow(input,deviceElapsed);
    const auto hook=[material](ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency,
        void* stream,const ugkp::gpu::RadiationThermalIterationDeviceView& view){
        return material->appendAnchoredIteration(graph,dependency,stream,view);
    };
    const MieDoSnapshot radiationInput{snapshot.epsilonS,snapshot.particleMeanTemperatureK,
        snapshot.representativeDiameterM,snapshot.wallTemperatureK,snapshot.coupledWallEmissivity,
        snapshot.resident,thermal,hook};
    auto& radiation=radiationSolver_.prepareParent(radiationInput,radiationWorkspace_);
    return {&radiation,material};
}

void GpuParticleRadiationCoupler::freezeWarmStart(ugkp::gpu::RadiationWarmStart& output,void* nextWriter) const {
    radiationSolver_.freezeWarmStart(radiationWorkspace_,output,nextWriter);
}
bool GpuParticleRadiationCoupler::readWarmStart(const std::string& path) const {
    return radiationSolver_.readWarmStart(radiationWorkspace_,path);
}
bool GpuParticleRadiationCoupler::writeWarmStart(const std::string& path) const {
    return radiationSolver_.writeWarmStart(radiationWorkspace_,path);
}
List<scalarField> GpuParticleRadiationCoupler::collectBoundaryEnergy() const {
    return mapBoundaryEnergy(mesh_,materialDevice_->collectBoundaryEnergy());
}

}
}
