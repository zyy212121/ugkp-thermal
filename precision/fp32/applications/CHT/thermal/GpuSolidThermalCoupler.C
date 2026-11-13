#include "FoamSolverScheme.H"
#include "FoamThermalDiscretization.H"
#include "../../../common/thermal/radiation/RadiationControls.H"
#include "../../../common/thermal/ThermalCouplingSchedule.H"
#include "GpuSolidThermalCoupler.H"
#include "../../../common/gasBoundary/FoamInput.H"
#include "../../../common/DiagnosticsFoam.H"
#include "../../../common/DiagnosticsCatalogue.H"
#include "../../../common/gpu/FoamSparseSolve.H"
#include "../../../common/gpu/FoamImplicitDiffusion.H"
#include "ThermalCouplingSchedule.H"
#include "ThermalHostDiagnostics.H"

#include "fixedGradientFvPatchFields.H"
#include "directFvPatchFieldMapper.H"
#include "entry.H"
#include "ITstream.H"
#include "token.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "SolverPerformance.H"
#include "zeroGradientFvPatchFields.H"
#include "surfaceFields.H"
#include "OSspecific.H"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <map>

#ifdef UGKWP_USE_CUDA
#include "GpuParticleRadiationCoupler.H"
#include "GpuSolidPatchMapper.H"
#include "GpuThermalExchangeState.H"
#include "MieDoRadiationSolver.H"
#include "MieTable.H"
#include "../gpu/GpuResidentStrict.H"
#include "IOdictionary.H"
#include "OFstream.H"
#include "PtrList.H"
#endif

namespace Foam
{
namespace gpuThermal
{
namespace
{
[[noreturn]] void solidFailure(const std::string& reason)
{
    throw GpuSolidThermalSolveError(reason);
}

bool finiteScalar(const scalar value)
{
    return std::isfinite(static_cast<double>(value));
}

GpuReal relativeTemperatureChange
(
    const scalarField& previous,
    const scalarField& current
)
{
    GpuReal maximum=0;
    forAll(current,cellI){
        const GpuReal now=GpuReal(current[cellI]),before=GpuReal(previous[cellI]);
        maximum=std::max(maximum,std::abs(now-before)/std::max(std::max(std::abs(now),std::abs(before)),GpuReal(1)));
    }
    return maximum;
}

bool isCoupledPatch(const labelList& coupledPatchIds, const label patchI)
{
    forAll(coupledPatchIds, coupledI)
    {
        if (coupledPatchIds[coupledI] == patchI)
        {
            return true;
        }
    }
    return false;
}

autoPtr<volScalarField> thermalFieldOnMesh
(
    const IOobject& io,
    fvMesh& mesh,
    const volScalarField& source
)
{
    if (&source.mesh() == &mesh)
    {
        return autoPtr<volScalarField>(new volScalarField(io, source));
    }
    if
    (
        source.size() < mesh.nCells()
     || source.boundaryField().size() < mesh.boundary().size()
    )
    {
        solidFailure("solid field is missing required mesh values");
    }
    autoPtr<volScalarField> result
    (
        new volScalarField(io, mesh, source.dimensions())
    );
    forAll(result->primitiveField(), cellI)
    {
        result->primitiveFieldRef()[cellI] = source[cellI];
    }
    forAll(mesh.boundary(), patchI)
    {
        const fvPatch& patch = mesh.boundary()[patchI];
        if (source.boundaryField()[patchI].size() < patch.size())
        {
            solidFailure("solid field is missing required boundary values");
        }
        labelList addressing(patch.size());
        forAll(addressing, faceI) addressing[faceI] = faceI;
        const directFvPatchFieldMapper mapper(addressing);
        result->boundaryFieldRef().set
        (
            patchI,
            fvPatchScalarField::New
            (
                source.boundaryField()[patchI], patch,
                result->internalField(), mapper
            ).ptr()
        );
    }
    return result;
}

class ScalarSolverLogSilencer
{
    int previous_;

public:
    ScalarSolverLogSilencer()
    :
        previous_(SolverPerformance<scalar>::debug)
    {
        SolverPerformance<scalar>::debug = 0;
    }

    ~ScalarSolverLogSilencer()
    {
        SolverPerformance<scalar>::debug = previous_;
    }
};

}

scalar clampedAitkenRelaxation
(
    const scalarField& previousResidual,
    const scalarField& currentResidual,
    const scalar previousRelaxation,
    const scalar minimumRelaxation,
    const scalar maximumRelaxation
)
{
    if
    (
        previousResidual.size() < currentResidual.size()
    )
    {
        solidFailure("invalid Aitken relaxation inputs");
    }

    GpuReal numerator = GpuReal(0);
    GpuReal denominator = GpuReal(0);
    forAll(currentResidual, cellI)
    {
        const GpuReal difference =
            GpuReal(currentResidual[cellI]) - GpuReal(previousResidual[cellI]);
        numerator += GpuReal(previousResidual[cellI])*difference;
        denominator += difference*difference;
    }
    if (!finiteScalar(denominator) || denominator <= std::numeric_limits<GpuReal>::min())
    {
        return std::min
        (
            std::max(GpuReal(previousRelaxation), GpuReal(minimumRelaxation)),
            GpuReal(maximumRelaxation)
        );
    }

    const GpuReal candidate =
        -GpuReal(previousRelaxation)*numerator/denominator;
    if (!finiteScalar(candidate))
    {
        return std::min
        (
            std::max(GpuReal(previousRelaxation), GpuReal(minimumRelaxation)),
            GpuReal(maximumRelaxation)
        );
    }
    return std::min(std::max(candidate,GpuReal(minimumRelaxation)),GpuReal(maximumRelaxation));
}

scalarField coupledPatchOwnerTemperature
(
    const volScalarField& solidTemperature,
    const label patchI
)
{
    if
    (
        patchI < 0
     || patchI >= solidTemperature.mesh().boundary().size()
    )
    {
        solidFailure("invalid solid coupling patch for owner temperature");
    }
    return scalarField
    (
        solidTemperature.boundaryField()[patchI].patchInternalField()
    );
}

scalarField coupledPatchSurfaceTemperature
(
    const volScalarField& solidTemperature,
    const label patchI
)
{
    if (patchI < 0 || patchI >= solidTemperature.boundaryField().size())
    {
        solidFailure("invalid solid coupling patch for surface temperature");
    }
    const fvPatchScalarField& patch = solidTemperature.boundaryField()[patchI];
    if (!isA<fixedGradientFvPatchScalarField>(patch))
    {
        solidFailure("coupled solid surface must be a fixedGradient patch");
    }


    return scalarField(patch);
}

GpuSolidThermalCandidate::GpuSolidThermalCandidate
(
    autoPtr<volScalarField> temperature,
    const scalar residual,
    const label iterations
)
:
    temperature_(temperature),
    integratedEnergyResidualJ(residual),
    nonlinearIterations(iterations)
{}

const volScalarField& GpuSolidThermalCandidate::temperature() const
{
    return temperature_();
}

namespace {
dictionary startupSolidLinearControls(const fvMesh& mesh,const char* module)
{
    dictionary result;
    const auto& scheme=ugkp::solverControls::startupScheme(mesh.time());
    label maximum=1000;
    label minimum=0;
    scalar tolerance=1e-6;
    scalar relative=0;
    scheme.read(module,"maxIter",maximum,"effectiveDefault",mesh.name());
    scheme.read(module,"minIter",minimum,"effectiveDefault",mesh.name());
    scheme.read(module,"tolerance",tolerance,"effectiveDefault",mesh.name());
    scheme.read(module,"relTol",relative,"effectiveDefault",mesh.name());
    result.set("maxIter",maximum);result.set("minIter",minimum);
    result.set("tolerance",tolerance);result.set("relTol",relative);
    return result;
}
}
const dictionary& GpuSolidThermalCandidateSolver::selectedLinearControls() const
{
    return !ugkp::solverControls::thermalDiscretization(mesh_).steady()&&mesh_.data::lookupOrDefault<bool>("finalIteration",false)
        ?finalLinearControls_:normalLinearControls_;
}

GpuSolidThermalCandidateSolver::GpuSolidThermalCandidateSolver
(
    fvMesh& mesh,
    const GpuSolidThermalProperties& properties,
    const labelList& coupledPatchIds,
    const GpuSolidThermalControls& controls,
    const labelList& ownerConductivityPatchIds
)
:
    mesh_(mesh),
    properties_(properties),
    coupledPatchIds_(coupledPatchIds),
    ownerConductivityPatchIds_(ownerConductivityPatchIds),
    controls_(controls)
{
    // FP32 defaults retain the approved engineering stopping budgets.
    // Preserve looser requests and leave invalid inputs to existing handling.
    constexpr GpuReal relativeBudget=1e-5f,energyRelativeBudget=1e-4f;
    if(std::isfinite(controls_.nonlinearRelativeTolerance)
        &&controls_.nonlinearRelativeTolerance>=0
        &&controls_.nonlinearRelativeTolerance<relativeBudget)
        controls_.nonlinearRelativeTolerance=relativeBudget;
    if(std::isfinite(controls_.energyRelativeTolerance)
        &&controls_.energyRelativeTolerance>=0
        &&controls_.energyRelativeTolerance<energyRelativeBudget)
        controls_.energyRelativeTolerance=energyRelativeBudget;

    const auto& scheme=ugkp::solverControls::startupScheme(mesh_.time());
    scheme.read("solid.nonlinear","nonlinearMaxIterations",controls_.nonlinearMaxIterations,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","nonlinearRelativeTolerance",controls_.nonlinearRelativeTolerance,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","energyRelativeTolerance",controls_.energyRelativeTolerance,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","energyAbsoluteTolerance",controls_.energyAbsoluteTolerance,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","nonlinearAitken",controls_.nonlinearAitken,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","nonlinearInitialRelaxation",controls_.nonlinearInitialRelaxation,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","nonlinearMinimumRelaxation",controls_.nonlinearMinimumRelaxation,"effectiveDefault",mesh_.name());
    scheme.read("solid.nonlinear","nonlinearMaximumRelaxation",controls_.nonlinearMaximumRelaxation,"effectiveDefault",mesh_.name());
    (void)ugkp::solverControls::thermalDiscretization(mesh_);
    normalLinearControls_=startupSolidLinearControls(mesh_,"solid.linear.normal");
    finalLinearControls_=startupSolidLinearControls(mesh_,"solid.linear.final");
    Info<< "FP32 solid engineering stopping budgets: region=" << mesh_.name()
        << " configuredRelative=" << controls.nonlinearRelativeTolerance
        << " effectiveRelative=" << controls_.nonlinearRelativeTolerance
        << " configuredEnergyRelative=" << controls.energyRelativeTolerance
        << " effectiveEnergyRelative=" << controls_.energyRelativeTolerance
        << " energyAbsolute=" << controls_.energyAbsoluteTolerance
        << " (FP32 defaults followed by explicit constant/scheme overrides)"
        << nl;
}

GpuSolidThermalCandidate GpuSolidThermalCandidateSolver::solveTemporarySolidCandidate
(const volScalarField& temperature,const List<scalarField>& energy,scalar interval,
 const ugkp::gpu::ThermalInterfaceEnergyDeviceView& resident,const ugkp::gpu::ThermalInterfaceEnergyDeviceView& appended) const
{
    return solveCandidate(temperature,energy,interval,resident,appended,false,nullptr);
}
bool GpuSolidThermalCandidateSolver::prepareResident(const volScalarField& temperature,
    const ugkp::gpu::ThermalInterfaceEnergyDeviceView& energy,ugkp::diagnostics::DeviceContext* context,
    const ugkp::gpu::ThermalInterfaceEnergyDeviceView& appended) const
{
    if(!energy.enabled||!energy.interval)return false;
    List<scalarField> initialEnergy(coupledPatchIds_.size());
    forAll(initialEnergy,i)initialEnergy[i].setSize(mesh_.boundary()[coupledPatchIds_[i]].size(),scalar(0));
    // Configure coefficients and storage without advancing the initial field.
    // Each replay reads its actual interval from the shared device ledger.
    solveCandidate(temperature,initialEnergy,1,energy,appended,true,context);
    return bool(residentGraph_);
}
ugkp::gpu::SolidThermalIterationGraph& GpuSolidThermalCandidateSolver::prepareResidentContact(label patch) const
{
    if(!residentGraph_)solidFailure("solid contact resident owner is not prepared");
    const auto& cells=mesh_.boundary()[patch].faceCells();
    residentGraph_->prepareContactWindow(std::vector<int>(cells.begin(),cells.end()));return *residentGraph_;
}
void GpuSolidThermalCandidateSolver::advanceResident(void* producerStream) const
{
    if(!residentGraph_)solidFailure("solid resident owner is not prepared");
    residentGraph_->enqueue(producerStream);residentAdvanced_=true;
}
void GpuSolidThermalCandidateSolver::publishResident(volScalarField& temperature) const
{
    if(!residentGraph_||!residentAdvanced_)return;
    residentDiffusion_->downloadSolution(temperature.primitiveFieldRef().begin(),temperature.size());
    std::vector<double> boundary,normal;residentDiffusion_->downloadBoundaryState(boundary,normal);
    label offset=0;forAll(temperature.boundaryField(),p){auto& patch=temperature.boundaryFieldRef()[p];
        if(patch.type()=="fixedGradient"){
            auto& gradient=refCast<fixedGradientFvPatchScalarField>(patch).gradient();
            forAll(gradient,f)gradient[f]=normal[offset+f];
        }
        forAll(patch,f)patch[f]=boundary[offset+f];
        offset+=patch.size();
    }
}

GpuSolidThermalCandidate
GpuSolidThermalCandidateSolver::solveCandidate
(
    const volScalarField& registeredTemperature,
    const List<scalarField>& interfaceEnergyJ,
    const scalar deltaTExchange,
    const ugkp::gpu::ThermalInterfaceEnergyDeviceView& residentEnergy,
    const ugkp::gpu::ThermalInterfaceEnergyDeviceView& appendedEnergy,
    const bool prepareOnly,ugkp::diagnostics::DeviceContext* context
) const
{
    if
    (
        interfaceEnergyJ.size() < coupledPatchIds_.size()
    )
    {
        solidFailure("invalid solid candidate mesh, energy list, or exchange time");
    }

    autoPtr<volScalarField> candidate = thermalFieldOnMesh
    (
        IOobject
        (
            "TsolidCandidate", mesh_.time().timeName(), mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, false
        ),
        mesh_, registeredTemperature
    );

    GpuReal totalInterfaceEnergyJ = GpuReal(0);
    forAll(coupledPatchIds_, coupledI)
    {
        const label patchI = coupledPatchIds_[coupledI];
        if
        (
            patchI < 0
         || patchI >= mesh_.boundary().size()
         || interfaceEnergyJ[coupledI].size() < mesh_.boundary()[patchI].size()
         || !isA<fixedGradientFvPatchScalarField>
            (candidate->boundaryField()[patchI])
        )
        {
            solidFailure("coupled solid patch must be valid fixedGradient with matching energy");
        }
        forAll(mesh_.boundary()[patchI], faceI)
        {
            const scalar energyJ = interfaceEnergyJ[coupledI][faceI];

            totalInterfaceEnergyJ += GpuReal(energyJ);
        }
    }

    const scalarField oldInternal(candidate->primitiveField());
    scalarField previous(candidate->primitiveField());
    bool havePreviousFixedPointResidual = false;
    GpuReal nonlinearRelaxation = controls_.nonlinearInitialRelaxation;
    const dimensionedScalar dt("deltaTExchange", dimTime, deltaTExchange);
    const dimensionSet rhoCpDimensions = dimEnergy/dimVolume/dimTemperature;
    const dimensionSet kappaDimensions = dimPower/dimLength/dimTemperature;
    GpuReal finalResidual = std::numeric_limits<GpuReal>::quiet_NaN();
    GpuReal finalTemperatureResidual =
        std::numeric_limits<GpuReal>::quiet_NaN();
    GpuReal finalSurfaceResidual = std::numeric_limits<GpuReal>::quiet_NaN();
    GpuReal finalSurfaceFluxResidual = std::numeric_limits<GpuReal>::quiet_NaN();
    GpuReal finalEnergyLimit = scalar(0);

    // Each gas surface needs one initial snapshot. The convergence pass below
    // updates it in place for the next iteration; owner-only patches need none.
    List<scalarField> previousSurface(coupledPatchIds_.size());
    forAll(coupledPatchIds_, coupledI)
    {
        if
        (
            isCoupledPatch
            (
                ownerConductivityPatchIds_, coupledPatchIds_[coupledI]
            )
        )
        {
            continue;
        }
        previousSurface[coupledI] =
            candidate->boundaryField()[coupledPatchIds_[coupledI]];
    }

    ugkp::gpu::installGpuImplicitDiffusion();
    if (!sparse_) sparse_.reset(new ugkp::gpu::SparseConjugateGradient);
    if (!residentDiffusion_) residentDiffusion_.reset(new ugkp::gpu::ImplicitDiffusion);
    residentDiffusion_->uploadSolution(candidate->primitiveField().begin(),candidate->size());
    labelList residentBoundaryOffset(mesh_.boundary().size());label residentBoundaryCount=0;
    forAll(candidate->boundaryField(),p){residentBoundaryOffset[p]=residentBoundaryCount;residentBoundaryCount+=candidate->boundaryField()[p].size();}
    std::vector<double> residentInterfaceEnergy(residentBoundaryCount,0);std::vector<int> residentInterfaceSelected(residentBoundaryCount,0),residentInterfaceFaces;
    forAll(coupledPatchIds_,i)forAll(interfaceEnergyJ[i],f){const label b=residentBoundaryOffset[coupledPatchIds_[i]]+f;
        residentInterfaceEnergy[b]=interfaceEnergyJ[i][f];residentInterfaceSelected[b]=1;residentInterfaceFaces.push_back(b);}
    // The gas ledger is the leading mapper prefix. A three-region solid may
    // append its independently computed solid-contact patch after that prefix.
    residentDiffusion_->setInterfaceEnergy(residentInterfaceFaces,residentEnergy,appendedEnergy);
    bool residentConvergence=false;
    std::vector<int> convergenceSurfaceIndex(residentBoundaryCount,-2);
    std::vector<double> initialBoundaryTemperature;initialBoundaryTemperature.reserve(residentBoundaryCount);
    forAll(candidate->boundaryField(),p)forAll(candidate->boundaryField()[p],f)initialBoundaryTemperature.push_back(candidate->boundaryField()[p][f]);
    int surfaceSlot=0;
    forAll(coupledPatchIds_,i)forAll(mesh_.boundary()[coupledPatchIds_[i]],f){
        const int b=residentBoundaryOffset[coupledPatchIds_[i]]+f;
        convergenceSurfaceIndex[b]=isCoupledPatch(ownerConductivityPatchIds_,coupledPatchIds_[i])?-1:surfaceSlot++;
    }
    mesh_.schemes().setFluxRequired(candidate->name());
    for
    (
        label iteration = 1;
        iteration <= max(label(1), controls_.nonlinearMaxIterations);
        ++iteration
    )
    {
        volScalarField Csec
        (
            IOobject
            (
                "solidCsecCandidate", mesh_.time().timeName(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar("zero", rhoCpDimensions, scalar(0))
        );
        volScalarField kappa
        (
            IOobject
            (
                "solidKappaCandidate", mesh_.time().timeName(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar("zero", kappaDimensions, scalar(0))
        );
        if (!deviceProperties_)
        {
            const auto tables = properties_.deviceTables();
            const auto& knots = properties_.caloricKnots();
            deviceProperties_.reset(new ugkp::gpu::SolidThermalProperties
                (tables.data(), knots.begin(), knots.size()));
        }
        if(iteration==1){
            deviceProperties_->setResidentCellAnchors(residentDiffusion_->deviceSolution(),oldInternal.size(),residentDiffusion_->deviceValues().readyStream);
            deviceProperties_->setConvergenceBoundary(convergenceSurfaceIndex,initialBoundaryTemperature);
        }
        std::vector<ugkp::gpu::ThermalPropertyQuery> propertyQueries;
        propertyQueries.reserve(mesh_.nFaces()-mesh_.nInternalFaces());
        std::vector<int> propertySources;propertySources.reserve(residentBoundaryCount);
        forAll(mesh_.boundary(), patchI)
        {
            const bool ownerConductivity = isCoupledPatch(ownerConductivityPatchIds_, patchI);
            const auto& cells = mesh_.boundary()[patchI].faceCells();
            forAll(mesh_.boundary()[patchI], faceI)
            {
                const scalar T = ownerConductivity ? (*candidate)[cells[faceI]]
                    : candidate->boundaryField()[patchI][faceI];
                propertyQueries.push_back({GpuReal(T), GpuReal(T)});
                propertySources.push_back(ownerConductivity?cells[faceI]:-1-residentBoundaryOffset[patchI]-faceI);
            }
        }
        std::vector<ugkp::gpu::ThermalPropertyResult> propertyResults;
        deviceProperties_->evaluateResidentCells(propertyQueries,residentDiffusion_->deviceSolution(),nullptr,mesh_.nCells(),
            ugkp::gpu::ThermalCapacity|ugkp::gpu::ThermalConductivity,true,residentDiffusion_->deviceValues().readyStream,propertyResults,
            iteration>1?residentDiffusion_->deviceBoundaryTemperature():nullptr,&propertySources,!residentConvergence);
        const auto checkPropertyResults = [&](const std::vector<ugkp::gpu::ThermalPropertyResult>& results)
        {
        for (std::size_t i=0; i<results.size(); ++i)
        {
            const auto& status = results[i].status;
            if (status.error != ugkp::gpu::ThermalPropertyError::None)
                FatalErrorInFunction << "invalid consumed solid property; region=" << mesh_.name()
                    << "; entry=" << i << "; property=" << status.property
                    << "; temperature=" << status.temperature << "; value=" << status.value
                    << "; kind=" << static_cast<int>(status.error) << exit(FatalError);
            if (status.extrapolated && hostDiagnosticWarningsEnabled())
                WarningInFunction << "finite solid table extrapolation; legacy outOfBounds=error"
                    << "; region=" << mesh_.name() << "; entry=" << i
                    << "; propertyMask=" << status.extrapolated << '\n';
        }
        };
        const auto checkResidentProperties=[&](){
            const auto diagnostic=deviceProperties_->diagnostic();
            if(diagnostic.firstError>=0){const auto& status=diagnostic.error;
                FatalErrorInFunction << "invalid consumed solid property; region=" << mesh_.name()
                    << "; entry=" << diagnostic.firstError << "; property=" << status.property
                    << "; temperature=" << status.temperature << "; value=" << status.value << exit(FatalError);}
            if(diagnostic.firstExtrapolation>=0&&hostDiagnosticWarningsEnabled())WarningInFunction
                << "finite solid table extrapolation; legacy outOfBounds=error; region=" << mesh_.name()
                << "; firstEntry=" << diagnostic.firstExtrapolation << endl;
        };
        if(residentConvergence)checkResidentProperties();else checkPropertyResults(propertyResults);
        std::size_t propertyIndex = 0;
        if(!residentConvergence){
        forAll(candidate->primitiveField(), cellI)
        {
            Csec[cellI] = propertyResults[propertyIndex].capacity;
            kappa[cellI] = propertyResults[propertyIndex++].conductivity;
        }
        forAll(mesh_.boundary(), patchI) forAll(mesh_.boundary()[patchI], faceI)
        {
            Csec.boundaryFieldRef()[patchI][faceI] = propertyResults[propertyIndex].capacity;
            kappa.boundaryFieldRef()[patchI][faceI] = propertyResults[propertyIndex++].conductivity;
        }

        forAll(coupledPatchIds_, coupledI)
        {
            const label patchI = coupledPatchIds_[coupledI];
            fixedGradientFvPatchScalarField& interfacePatch =
                refCast<fixedGradientFvPatchScalarField>
                (
                    candidate->boundaryFieldRef()[patchI]
                );
            forAll(interfacePatch, faceI)
            {
                const GpuReal area = GpuReal(mesh_.magSf().boundaryField()[patchI][faceI]);

                const GpuReal denominator =
                    area*GpuReal(deltaTExchange)*GpuReal(kappa.boundaryField()[patchI][faceI]);

                interfacePatch.gradient()[faceI] =
                    GpuReal(interfaceEnergyJ[coupledI][faceI])/denominator;
            }
        }
        candidate->correctBoundaryConditions();
        }

        const autoPtr<volScalarField> oldTemperature = thermalFieldOnMesh
        (
            IOobject
            (
                "TsolidOldCandidate", mesh_.time().timeName(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_, registeredTemperature
        );
        if (!sparse_) sparse_.reset(new ugkp::gpu::SparseConjugateGradient);
        if (!residentDiffusion_) residentDiffusion_.reset(new ugkp::gpu::ImplicitDiffusion);
        const auto deviceFields=deviceProperties_->deviceFields();
        residentDiffusion_->prescribeBoundaryEnergy(residentInterfaceEnergy,residentInterfaceSelected,deltaTExchange,
            deviceFields.conductivity+mesh_.nCells(),deviceFields.readyStream);
        auto equation=ugkp::gpu::residentImplicitThermalEquation
        (
            *residentDiffusion_,*sparse_,kappa,*candidate,Csec,oldTemperature(),deltaTExchange,true,
            deviceFields.capacity,deviceProperties_->deviceCellAnchors(),deviceFields.readyStream,deviceFields.conductivity,true,nullptr,nullptr,deviceFields.conductivity+mesh_.nCells()
        );
        residentConvergence=residentDiffusion_->residentIteration();
        if(iteration==1&&residentConvergence){
            const dictionary& linearControls=selectedLinearControls();
            if(equation.configureResidentLocalSolve(linearControls)){
                const int coefficientPlan=deviceProperties_->prepareResidentEvaluation(propertyQueries,propertySources,
                    mesh_.nCells(),ugkp::gpu::ThermalCapacity|ugkp::gpu::ThermalConductivity,true,0);
                std::vector<ugkp::gpu::ThermalPropertyQuery> completionQueries;std::vector<int> completionSources;
                forAll(coupledPatchIds_,i){const label patch=coupledPatchIds_[i];
                    if(isCoupledPatch(ownerConductivityPatchIds_,patch))continue;
                    forAll(candidate->boundaryField()[patch],f){
                        completionQueries.push_back({0,0,ugkp::gpu::ThermalConductivity});
                        completionSources.push_back(-1-residentBoundaryOffset[patch]-f);}}
                const int convergencePlan=deviceProperties_->prepareResidentEvaluation(completionQueries,completionSources,
                    mesh_.nCells(),ugkp::gpu::ThermalEnergy|ugkp::gpu::ThermalTemperatureChange,false,1);
                equation.residentBoundaryFlux();
                ugkp::gpu::SolidThermalResidentProblem problem;problem.cells=mesh_.nCells();problem.surfaceProperties=completionQueries.size();
                problem.coefficientPlan=coefficientPlan;problem.convergencePlan=convergencePlan;problem.interval=deltaTExchange;
                problem.controls={controls_.nonlinearMaxIterations,controls_.nonlinearAitken,GpuReal(controls_.nonlinearRelativeTolerance),
                    GpuReal(controls_.energyAbsoluteTolerance),GpuReal(controls_.energyRelativeTolerance),GpuReal(controls_.nonlinearInitialRelaxation),
                    GpuReal(controls_.nonlinearMinimumRelaxation),GpuReal(controls_.nonlinearMaximumRelaxation),std::numeric_limits<GpuReal>::min()};
                if(prepareOnly){
                    ugkp::boundary::Definition schedule;schedule.faces.resize(residentBoundaryCount);
                    std::vector<int> signs(residentBoundaryCount,1);bool scheduled=false;
                    forAll(candidate->boundaryField(),p){const auto& patch=candidate->boundaryField()[p];
                        if(patch.type()!="uniformJump")continue;
                        const auto& cyclic=refCast<const cyclicFvPatchScalarField>(patch);
                        const label owner=cyclic.cyclicPatch().owner()?p:cyclic.cyclicPatch().neighbFvPatch().index();
                        const auto configuration=ugkp::boundary::foam::configuration(candidate->boundaryField()[owner]);
                        ugkp::boundary::foam::function<scalar>(schedule,configuration,"jumpTable",ugkp::boundary::Temperature,
                            schedule.faces,residentBoundaryOffset[p],patch.size());
                        forAll(patch,f)signs[residentBoundaryOffset[p]+f]=cyclic.cyclicPatch().owner()?1:-1;
                        scheduled=true;
                    }
                    if(scheduled)residentDiffusion_->configureBoundaryJumpSchedule(schedule,signs,mesh_.time().value());
                    // Matrix preparation prescribes zero initial exchange only
                    // to establish coefficients. Preserve the accepted restart
                    // boundary state for the first nonlinear window candidate.
                    residentDiffusion_->initializeBoundaryTemperature(initialBoundaryTemperature);
                    problem.deviceInterval=residentEnergy.interval;problem.diagnostics=context;
                    residentGraph_.reset(new ugkp::gpu::SolidThermalIterationGraph(*deviceProperties_,*residentDiffusion_,*sparse_,problem));
                    return GpuSolidThermalCandidate(candidate,0,0);
                }
                ugkp::gpu::SolidThermalIterationGraph graph(*deviceProperties_,*residentDiffusion_,*sparse_,problem);
                graph.enqueue(residentDiffusion_->deviceValues().readyStream);
                ugkp::diagnostics::DeviceContext completed{};
                const auto result=graph.collect(&completed);
                ugkp::diagnostics::dispatch(ugkp::diagnostics::takeSnapshot(completed,ugkp::diagnostics::sourceCatalogue()),ugkp::diagnostics::makeFoamSink());
                if(!result.valid)solidFailure("invalid consumed state in resident solid nonlinear iteration");
                equation.publishSolution();
                return GpuSolidThermalCandidate(candidate,result.energyResidual.binary64(),result.iteration);
            }
        }
        if(prepareOnly)return GpuSolidThermalCandidate(candidate,0,0);
        solverPerformance linearPerformance;
        {
            ScalarSolverLogSilencer silenceSolverLog;
            if (!sparse_) sparse_.reset(new ugkp::gpu::SparseConjugateGradient);
            const dictionary& linearControls = selectedLinearControls();
            linearPerformance = equation.solve(linearControls,!residentConvergence&&!controls_.nonlinearAitken);
        }
        if
        (
            !finiteScalar(linearPerformance.finalResidual())
        )
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "solid linear solve returned a nonfinite residual"
                << "; region=" << mesh_.name()
                << "; nonlinearIteration=" << iteration
                << "; solver=" << linearPerformance.solverName()
                << "; iterations=" << linearPerformance.nIterations()
                << "; initialResidual="
                << linearPerformance.initialResidual()
                << "; finalResidual=" << linearPerformance.finalResidual()
                << "; singular=" << linearPerformance.singular();
            FatalErrorInFunction
                << reason.str().c_str() << "; time=" << mesh_.time().timeName()
                << exit(FatalError);
        }
        if (!linearPerformance.converged() && hostDiagnosticWarningsEnabled())
        {
            const dictionary& linearControls = selectedLinearControls();
            const scalar absoluteLimit = linearControls.lookupOrDefault<scalar>("tolerance", 1e-6);
            const scalar relativeLimit = linearControls.lookupOrDefault<scalar>("relTol", 0)
                *linearPerformance.initialResidual();
            WarningInFunction
                << "solid linear solve did not converge; region=" << mesh_.name()
                << "; time=" << mesh_.time().timeName()
                << "; nonlinearIteration=" << iteration
                << "; iterations=" << linearPerformance.nIterations()
                << "; residual=" << linearPerformance.finalResidual()
                << "; absoluteLimit=" << absoluteLimit
                << "; relativeLimit=" << relativeLimit
                << "; retaining computed candidate" << endl;
        }
        if (controls_.nonlinearAitken)
        {
            nonlinearRelaxation = deviceProperties_->relaxResident
            (
                residentDiffusion_->deviceSolution(),residentDiffusion_->devicePreviousSolution(),previous.size(),
                havePreviousFixedPointResidual, nonlinearRelaxation,
                controls_.nonlinearMinimumRelaxation, controls_.nonlinearMaximumRelaxation, std::numeric_limits<GpuReal>::min(),
                residentDiffusion_->deviceValues().readyStream
            );
            if(!residentConvergence)equation.publishSolution();
            havePreviousFixedPointResidual = true;
        }
        if(residentConvergence)residentDiffusion_->refreshBoundaryTemperature();
        else candidate->correctBoundaryConditions();
        finalSurfaceResidual = scalar(0);
        finalSurfaceFluxResidual = scalar(0);
        // Preserve the original surface-then-cell consumption order while
        // evaluating the completed iterate's conductivity and caloric integral.
        propertyQueries.clear();propertySources.clear();
        forAll(coupledPatchIds_, coupledI)
        {
            const label patchI=coupledPatchIds_[coupledI];
            if (isCoupledPatch(ownerConductivityPatchIds_, patchI)) continue;
            forAll(candidate->boundaryField()[patchI], faceI){
                propertyQueries.push_back({GpuReal(candidate->boundaryField()[patchI][faceI]), 0,ugkp::gpu::ThermalConductivity});
                propertySources.push_back(-1-residentBoundaryOffset[patchI]-faceI);
            }
        }
        const std::size_t surfacePropertyCount=propertyQueries.size();
        deviceProperties_->evaluateResidentCells(propertyQueries,residentDiffusion_->deviceSolution(),residentDiffusion_->devicePreviousSolution(),mesh_.nCells(),
            ugkp::gpu::ThermalEnergy|ugkp::gpu::ThermalTemperatureChange,false,residentDiffusion_->deviceValues().readyStream,propertyResults,
            residentDiffusion_->deviceBoundaryTemperature(),&propertySources,!residentConvergence);
        if(residentConvergence)checkResidentProperties();else checkPropertyResults(propertyResults);
        if(residentConvergence){
            const auto flux=equation.residentBoundaryFlux();
            const auto metrics=deviceProperties_->convergence(mesh_.nCells(),surfacePropertyCount,
                residentDiffusion_->deviceVolume(),flux,residentDiffusion_->deviceBoundaryTemperature(),
                residentDiffusion_->deviceBoundaryGradient(),residentDiffusion_->deviceBoundaryEnergy(),
                residentDiffusion_->deviceBoundaryArea(),deltaTExchange,residentDiffusion_->deviceValues().readyStream);
            const GpuReal enthalpy=metrics.enthalpyChange.value,external=metrics.externalEnergy.value;
            const GpuReal energyResidual=enthalpy-totalInterfaceEnergyJ-external;
            const GpuReal energyScale=std::max(std::max(std::abs(enthalpy),std::abs(totalInterfaceEnergyJ)+std::abs(external)),1.0f);
            const GpuReal limit=GpuReal(controls_.energyAbsoluteTolerance)+GpuReal(controls_.energyRelativeTolerance)*energyScale;
            finalResidual=GpuReal(energyResidual);finalEnergyLimit=GpuReal(limit);
            finalTemperatureResidual=metrics.temperatureChange;finalSurfaceResidual=metrics.surfaceChange;finalSurfaceFluxResidual=metrics.surfaceFluxChange;
            const bool converged=finalTemperatureResidual<=GpuReal(controls_.nonlinearRelativeTolerance)
                &&max(finalSurfaceResidual,finalSurfaceFluxResidual)<=GpuReal(controls_.nonlinearRelativeTolerance)&&std::abs(energyResidual)<=limit;
            if(converged||iteration==max(label(1),controls_.nonlinearMaxIterations)){
                equation.publishSolution();
                if(converged)return GpuSolidThermalCandidate(candidate,finalResidual,iteration);
            }
            continue;
        }
        propertyIndex=0;
        forAll(coupledPatchIds_, coupledI)
        {
            const label patchI = coupledPatchIds_[coupledI];
            if (isCoupledPatch(ownerConductivityPatchIds_, patchI))
            {
                continue;
            }
            const fixedGradientFvPatchScalarField& patch =
                refCast<const fixedGradientFvPatchScalarField>
                (
                    candidate->boundaryField()[patchI]
                );
            const scalarField& gradient = patch.gradient();
            const scalarField& area = mesh_.magSf().boundaryField()[patchI];
            scalarField& previousPatchSurface = previousSurface[coupledI];
            forAll(patch, faceI)
            {
                const GpuReal surface = patch[faceI];
                const GpuReal previousSurfaceTemperature =
                    previousPatchSurface[faceI];
                finalSurfaceResidual = max
                (
                    finalSurfaceResidual,
                    mag(surface - previousSurfaceTemperature)
                   /max
                    (
                        max(mag(surface), mag(previousSurfaceTemperature)),
                        GpuReal(1)
                    )
                );
                // The completed-interval ledger remains prescribed. Iterate
                // only its solid surface closure k(Tw)*grad(T) = E/(A*dt).
                const GpuReal prescribedFlux = GpuReal(interfaceEnergyJ[coupledI][faceI])
                   /(GpuReal(area[faceI])*GpuReal(deltaTExchange));
                const GpuReal surfaceFlux =
                    propertyResults[propertyIndex++].conductivity*GpuReal(gradient[faceI]);
                finalSurfaceFluxResidual = max
                (
                    finalSurfaceFluxResidual,
                    mag(surfaceFlux - prescribedFlux)
                   /max(max(mag(surfaceFlux), mag(prescribedFlux)), GpuReal(1))
                );
                previousPatchSurface[faceI] = surface;
            }
        }
        const auto matrixFlux = equation.boundaryFlux();
        GpuReal externalBoundaryEnergyIntoSolidJ = GpuReal(0);
        forAll(matrixFlux, patchI)
        {
            if (isCoupledPatch(coupledPatchIds_, patchI))
            {
                continue;
            }
            forAll(matrixFlux[patchI], faceI)
            {
                const scalar faceFluxW =
                    matrixFlux[patchI][faceI];

                externalBoundaryEnergyIntoSolidJ -= GpuReal(deltaTExchange)*GpuReal(faceFluxW);
            }
        }

        GpuReal enthalpyChangeJ = GpuReal(0);
        finalTemperatureResidual = scalar(0);
        forAll(candidate->primitiveField(), cellI)
        {
            const auto& value=propertyResults[surfacePropertyCount+cellI];
            enthalpyChangeJ += GpuReal(mesh_.V()[cellI])*value.energy;
            finalTemperatureResidual=max(finalTemperatureResidual, value.relativeChange);
        }
        finalResidual =
            enthalpyChangeJ
          - totalInterfaceEnergyJ
          - externalBoundaryEnergyIntoSolidJ;

        const GpuReal energyScale = max
        (
            max
            (
                mag(enthalpyChangeJ),
                mag(totalInterfaceEnergyJ)
              + mag(externalBoundaryEnergyIntoSolidJ)
            ),
            GpuReal(1)
        );
        const bool temperatureConverged =
            finalTemperatureResidual <= GpuReal(controls_.nonlinearRelativeTolerance);
        finalEnergyLimit = GpuReal(controls_.energyAbsoluteTolerance)
          + GpuReal(controls_.energyRelativeTolerance)*energyScale;
        const bool energyConverged = mag(finalResidual) <= finalEnergyLimit;
        const bool surfaceConverged =
            max(finalSurfaceResidual, finalSurfaceFluxResidual)
         <= GpuReal(controls_.nonlinearRelativeTolerance);
        if (temperatureConverged && surfaceConverged && energyConverged)
        {
            return GpuSolidThermalCandidate(candidate, finalResidual, iteration);
        }
        previous = candidate->primitiveField();
    }

    if (hostDiagnosticWarningsEnabled())
    {
        WarningInFunction
            << "solid nonlinear iteration budget exhausted; region=" << mesh_.name()
            << "; time=" << mesh_.time().timeName()
            << "; iterations=" << max(label(1), controls_.nonlinearMaxIterations)
            << "; temperatureResidual=" << finalTemperatureResidual
            << "; surfaceTemperatureResidual=" << finalSurfaceResidual
            << "; surfaceFluxResidual=" << finalSurfaceFluxResidual
            << "; relativeLimit=" << controls_.nonlinearRelativeTolerance
            << "; energyResidualJ=" << finalResidual
            << "; energyLimitJ=" << finalEnergyLimit
            << "; retaining computed candidate" << endl;
    }

    return GpuSolidThermalCandidate
    (
        candidate, finalResidual, max(label(1), controls_.nonlinearMaxIterations)
    );
}

ugkp::gpu::ThermalSolidContactDeviceView GpuSolidThermalCandidateSolver::contactState
(const volScalarField& temperature,const label patch,const bool initialise) const
{
    if(!residentDiffusion_)residentDiffusion_.reset(new ugkp::gpu::ImplicitDiffusion);
    if(initialise)residentDiffusion_->uploadSolution(temperature.primitiveField().begin(),temperature.size());
    if(!deviceProperties_){const auto tables=properties_.deviceTables();const auto& knots=properties_.caloricKnots();
        deviceProperties_.reset(new ugkp::gpu::SolidThermalProperties(tables.data(),knots.begin(),knots.size()));}
    const auto& owners=mesh_.boundary()[patch].faceCells();std::vector<int> cells(owners.begin(),owners.end());
    std::vector<ugkp::gpu::ThermalPropertyResult> status;
    const auto view=deviceProperties_->contactState(cells,residentDiffusion_->deviceTemperature(),residentDiffusion_->deviceValues().readyStream,status);
    for(std::size_t i=0;i<status.size();++i){const auto& value=status[i].status;
        if(value.error!=ugkp::gpu::ThermalPropertyError::None)FatalErrorInFunction
            << "invalid consumed solid contact conductivity; region=" << mesh_.name() << "; face=" << i
            << "; temperature=" << value.temperature << "; value=" << value.value << exit(FatalError);
        if(value.extrapolated&&hostDiagnosticWarningsEnabled())WarningInFunction
            << "finite solid contact conductivity extrapolation; region=" << mesh_.name() << "; face=" << i << endl;}
    return view;
}

ugkp::gpu::ThermalInterfaceStateDeviceView GpuSolidThermalCandidateSolver::interfaceState
(
    const labelList& contactPairIds,const label pairs,const bool includeEmissivity,ugkp::diagnostics::DeviceContext* context
) const
{
    if(!residentDiffusion_||!deviceProperties_||!residentDiffusion_->deviceBoundaryTemperature())return {};
    labelList offsets(mesh_.boundary().size());label count=0;
    forAll(mesh_.boundary(),p){offsets[p]=count;count+=mesh_.boundary()[p].size();}
    std::vector<int> faces,contact;
    for(label i=0;i<pairs;++i)forAll(mesh_.boundary()[coupledPatchIds_[i]],f){
        faces.push_back(offsets[coupledPatchIds_[i]]+f);contact.push_back(isCoupledPatch(contactPairIds,i)?1:0);}
    std::vector<ugkp::gpu::ThermalPropertyResult> status;
    const auto view=deviceProperties_->interfaceState(faces,contact,residentDiffusion_->deviceBoundaryTemperature(),
        residentDiffusion_->deviceValues().readyStream,status,includeEmissivity,context);
    for(std::size_t i=0;i<status.size();++i){const auto& value=status[i].status;
        if(value.error!=ugkp::gpu::ThermalPropertyError::None)FatalErrorInFunction
            << "invalid consumed contact-wall property; region=" << mesh_.name() << "; face=" << i
            << "; property=" << value.property << "; temperature=" << value.temperature
            << "; value=" << value.value << exit(FatalError);
        if(value.extrapolated&&hostDiagnosticWarningsEnabled())WarningInFunction
            << "finite contact-wall property extrapolation; region=" << mesh_.name() << "; face=" << i << endl;
    }
    return view;
}

void GpuSolidThermalCandidateSolver::publishSolidCandidate
(
    const GpuSolidThermalCandidate& candidate,
    volScalarField& registeredTemperature
) const
{
    if
    (
        candidate.temperature().size() < registeredTemperature.size()
     || candidate.temperature().boundaryField().size()
        < registeredTemperature.boundaryField().size()
    )
    {
        solidFailure("solid candidate is missing required publication values");
    }
    registeredTemperature.primitiveFieldRef() =
        candidate.temperature().primitiveField();
    forAll(registeredTemperature.boundaryField(), patchI)
    {
        if (candidate.temperature().boundaryField()[patchI].size()
            < registeredTemperature.boundaryField()[patchI].size())
        {
            solidFailure("solid candidate is missing required boundary publication values");
        }
        registeredTemperature.boundaryFieldRef()[patchI] ==
            candidate.temperature().boundaryField()[patchI];
        if
        (
            isA<fixedGradientFvPatchScalarField>
            (registeredTemperature.boundaryField()[patchI])
         && isA<fixedGradientFvPatchScalarField>
            (candidate.temperature().boundaryField()[patchI])
        )
        {
            refCast<fixedGradientFvPatchScalarField>
            (
                registeredTemperature.boundaryFieldRef()[patchI]
            ).gradient() =
                refCast<const fixedGradientFvPatchScalarField>
                (
                    candidate.temperature().boundaryField()[patchI]
                ).gradient();
        }
    }
    registeredTemperature.correctBoundaryConditions();
}

}
}

#ifdef UGKWP_USE_CUDA

namespace Foam
{
namespace gpuThermal
{
namespace
{
const dictionary& exactSubDictionary
(
    const dictionary& parent,
    const word& name
)
{
    const entry* item = parent.lookupEntryPtr(name, false, false);
    if (item == nullptr || !item->isDict())
    {
        solidFailure("missing required dictionary " + std::string(name.c_str()));
    }
    return item->dict();
}

template<class Value>
Value exactLookup(const dictionary& dict, const word& name)
{
    if (!dict.found(name, false, false))
    {
        solidFailure("missing required entry " + std::string(name.c_str()));
    }
    return dict.lookup<Value>(name, false, false);
}

List<SolidPatchPair> readPatchPairs(const dictionary& coupling)
{
    if (!coupling.found("coupledInterfaces", false, false))
    {
        solidFailure("missing required coupledInterfaces list");
    }
    PtrList<dictionary> pairDictionaries
    (
        coupling.lookup("coupledInterfaces", false, false)
    );
    if (pairDictionaries.empty())
    {
        solidFailure("coupledInterfaces must contain at least one pair");
    }
    List<SolidPatchPair> result(pairDictionaries.size());
    forAll(pairDictionaries, pairI)
    {
        result[pairI].fluidPatch =
            exactLookup<word>(pairDictionaries[pairI], "fluidPatch");
        result[pairI].solidPatch =
            exactLookup<word>(pairDictionaries[pairI], "solidPatch");
    }
    return result;
}

List<SolidPatchPair> readFluidSolidPatchPairs
(
    const dictionary& coupling,
    const word& primarySolidRegion
)
{
    PtrList<dictionary> entries
    (
        coupling.lookup("fluidSolidInterfaces", false, false)
    );
    if (entries.empty())
    {
        solidFailure("fluidSolidInterfaces must contain at least one pair");
    }
    List<SolidPatchPair> result(entries.size());
    forAll(entries, pairI)
    {
        if
        (
            exactLookup<word>(entries[pairI], "solidRegion")
         != primarySolidRegion
         || exactLookup<word>(entries[pairI], "mapping")
         != "conservativeNonConformal"
        )
        {
            solidFailure("fluid-solid interface must target the primary solid with conservativeNonConformal mapping");
        }
        result[pairI].fluidPatch =
            exactLookup<word>(entries[pairI], "fluidPatch");
        result[pairI].solidPatch =
            exactLookup<word>(entries[pairI], "solidPatch");
    }
    return result;
}

SolidPatchPair readSolidSolidPatchPair
(
    const dictionary& coupling,
    const word& firstRegion,
    const word& secondRegion
)
{
    PtrList<dictionary> entries
    (
        coupling.lookup("solidSolidInterfaces", false, false)
    );
    if (entries.size() != 1)
    {
        solidFailure("v2 requires exactly one solid-solid interface");
    }
    const dictionary& entry = entries[0];
    if (exactLookup<word>(entry, "mapping") != "oneToOneConformal")
    {
        solidFailure("solid-solid interface requires the oneToOneConformal operator");
    }
    const word first = exactLookup<word>(entry, "firstRegion");
    const word second = exactLookup<word>(entry, "secondRegion");
    SolidPatchPair result;
    if (first == firstRegion && second == secondRegion)
    {
        result.fluidPatch = exactLookup<word>(entry, "firstPatch");
        result.solidPatch = exactLookup<word>(entry, "secondPatch");
    }
    else if (second == firstRegion && first == secondRegion)
    {
        result.fluidPatch = exactLookup<word>(entry, "secondPatch");
        result.solidPatch = exactLookup<word>(entry, "firstPatch");
    }
    else solidFailure("solid-solid interface is missing its selected region endpoint");
    return result;
}



}

class GpuSolidThermalCouplerData
{
public:
    Time& runTime;
    fvMesh& fluidMesh;
    GpuParticleResidentSolver& resident;
    dictionary resolvedConfiguration;
    ThermalExchangeWriteEventStateMachine writeState;
    bool threeRegion;
    word auxiliarySolidRegion;
    autoPtr<fvMesh> auxiliarySolidMesh;
    autoPtr<volScalarField> TauxiliarySolid;
    autoPtr<GpuSolidThermalProperties> auxiliaryProperties;
    autoPtr<GpuSolidThermalCandidateSolver> auxiliaryCandidateSolver;
    autoPtr<OneToOneConformalSolidPatchMapper> solidInterfaceMapper;
    std::unique_ptr<ugkp::gpu::SolidThermalContact> solidContact;
    std::unique_ptr<ugkp::gpu::SolidContactIterationGraph> solidContactGraph;
    word solidRegion;
    autoPtr<fvMesh> solidMesh;
    autoPtr<volScalarField> Tsolid;
    autoPtr<GpuSolidThermalProperties> properties;
    autoPtr<GpuSolidPatchMapper> mapper;
    autoPtr<MieTable> mieTable;
    autoPtr<MieDoRadiationSolver> radiationSolver;
    autoPtr<GpuParticleRadiationCoupler> particleRadiationCoupler;
    scalar lastRadiationIntervalS=0;
    autoPtr<GpuSolidThermalCandidateSolver> solidCandidateSolver;
    bool radiationEnabled;
    GpuReal rhoSolid;
    GpuReal particleHeatFactor;
    scalar solidContactCouplingInterval;
    label radiationCouplingFrequency;
    label solidInterfaceMaxIterations;
    GpuReal solidInterfaceTemperatureTolerance;
    GpuReal solidInterfaceEnergyTolerance;
    bool wallInitialised;
    labelList particleContactPairIds;
    labelList particleContactFluidPatchIds;
    autoPtr<surfaceScalarField> gasConvectiveWallHeatFlux;
    autoPtr<surfaceScalarField> particleRadiationWallHeatFlux;
    autoPtr<volScalarField> particleRadiationAbsorptionCoefficient;
    autoPtr<volScalarField> particleRadiationScatteringCoefficient;
    autoPtr<volScalarField> particleRadiationExtinctionCoefficient;
    autoPtr<volScalarField> particleRadiationAsymmetryFactor;
    autoPtr<surfaceScalarField> particleStuckWallHeatFlux;
    autoPtr<surfaceScalarField> particleReflectedWallHeatFlux;

    GpuSolidThermalCouplerData
    (
        Time& runTimeValue,
        fvMesh& fluidMeshValue,
        GpuParticleResidentSolver& residentValue,
        const dictionary& configuration,
        const ThermalExchangeRestartState& restartState
    )
    :
        runTime(runTimeValue),
        fluidMesh(fluidMeshValue),
        resident(residentValue),
        resolvedConfiguration(configuration),
        writeState(restartState),
        threeRegion(false),
        auxiliarySolidRegion(),
        auxiliarySolidMesh(nullptr),
        TauxiliarySolid(nullptr),
        auxiliaryProperties(nullptr),
        auxiliaryCandidateSolver(nullptr),
        solidInterfaceMapper(nullptr),
        solidRegion(),
        solidMesh(nullptr),
        Tsolid(nullptr),
        properties(nullptr),
        mapper(nullptr),
        mieTable(nullptr),
        radiationSolver(nullptr),
        particleRadiationCoupler(nullptr),
        solidCandidateSolver(nullptr),
        radiationEnabled(true),
        rhoSolid(0),
        particleHeatFactor(0),
        solidContactCouplingInterval(0),
        radiationCouplingFrequency(1),
        solidInterfaceMaxIterations(1),
        solidInterfaceTemperatureTolerance(0),
        solidInterfaceEnergyTolerance(0),
        wallInitialised(false),
        particleContactPairIds(),
        particleContactFluidPatchIds(),
        gasConvectiveWallHeatFlux(nullptr),
        particleRadiationWallHeatFlux(nullptr),
        particleRadiationAbsorptionCoefficient(nullptr),
        particleRadiationScatteringCoefficient(nullptr),
        particleRadiationExtinctionCoefficient(nullptr),
        particleRadiationAsymmetryFactor(nullptr),
        particleStuckWallHeatFlux(nullptr),
        particleReflectedWallHeatFlux(nullptr)
    {
        const dictionary& coupling =
            exactSubDictionary(resolvedConfiguration, "solidThermalCoupling");
        const entry* versionEntry =
            coupling.lookupEntryPtr("formatVersion", false, false);
        label formatVersion = 0;
        if (versionEntry && versionEntry->isStream())
        {
            const ITstream& version = versionEntry->stream();
            if (!version.empty() && version[0].isLabel())
            {
                formatVersion = version[0].labelToken();
            }
        }


        threeRegion = formatVersion == 2
          || (formatVersion != 1
           && coupling.found("solidSolidInterfaces", false, false));
        const auto schedule=ugkp::thermal::readCouplingSchedule(coupling,runTime);
        radiationEnabled=schedule.radiationEnabled;
        solidContactCouplingInterval=schedule.interval;
        radiationCouplingFrequency=schedule.radiationFrequency;
        if
        (
            radiationEnabled && radiationCouplingFrequency == 0
        )
        {
            solidFailure("active radiation frequency is zero");
        }
        if (threeRegion)
        {
            solidRegion = exactLookup<word>(coupling, "primarySolidRegion");
            const auto& interfaceScheme=ugkp::solverControls::startupScheme(runTime);
            interfaceScheme.read("solid.interface","solidInterfaceMaxIterations",solidInterfaceMaxIterations,"effectiveDefault",solidRegion);
            interfaceScheme.read("solid.interface","solidInterfaceTemperatureRelativeTolerance",solidInterfaceTemperatureTolerance,"effectiveDefault",solidRegion);
            interfaceScheme.read("solid.interface","solidInterfaceEnergyRelativeTolerance",solidInterfaceEnergyTolerance,"effectiveDefault",solidRegion);

            PtrList<dictionary> solidInterfaces
            (
                coupling.lookup("solidSolidInterfaces", false, false)
            );
            if (solidInterfaces.size() != 1)
            {
                solidFailure("v2 requires exactly one solid-solid interface");
            }
            const word first = exactLookup<word>(solidInterfaces[0], "firstRegion");
            const word second = exactLookup<word>(solidInterfaces[0], "secondRegion");
            if (second == solidRegion) auxiliarySolidRegion = first;
            else if (first == solidRegion) auxiliarySolidRegion = second;
            else solidFailure("solid-solid interface has no primary solid endpoint");
        }
        else
        {
            if (exactLookup<word>(coupling, "mapping") != "oneToOneConformal")
            {
                solidFailure("v1 requires oneToOneConformal mapping");
            }
            solidRegion = exactLookup<word>(coupling, "solidRegion");
        }
        solidMesh.reset
        (
            new fvMesh
            (
                IOobject
                (
                    solidRegion, runTime.timeName(), runTime,
                    IOobject::MUST_READ, IOobject::AUTO_WRITE, true
                ),
                false
            )
        );
        Tsolid.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "T", runTime.timeName(), solidMesh(),
                    IOobject::MUST_READ, IOobject::AUTO_WRITE, true
                ),
                solidMesh()
            )
        );
        if (threeRegion)
        {
            auxiliarySolidMesh.reset
            (
                new fvMesh
                (
                    IOobject
                    (
                        auxiliarySolidRegion, runTime.timeName(), runTime,
                        IOobject::MUST_READ, IOobject::AUTO_WRITE, true
                    ),
                    false
                )
            );
            TauxiliarySolid.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        "T", runTime.timeName(), auxiliarySolidMesh(),
                        IOobject::MUST_READ, IOobject::AUTO_WRITE, true
                    ),
                    auxiliarySolidMesh()
                )
            );
            const dictionary& regions =
                exactSubDictionary(coupling, "solidRegions");
            properties.reset
            (
                new GpuSolidThermalProperties
                (
                    exactSubDictionary
                    (
                        exactSubDictionary(regions, solidRegion),
                        "properties"
                    )
                )
            );
            auxiliaryProperties.reset
            (
                new GpuSolidThermalProperties
                (
                    exactSubDictionary
                    (
                        exactSubDictionary(regions, auxiliarySolidRegion),
                        "properties"
                    )
                )
            );
            const List<SolidPatchPair> pairs
            (
                readFluidSolidPatchPairs(coupling, solidRegion)
            );
            mapper.reset
            (
                new ConservativeNearestSolidPatchMapper
                (
                    fluidMesh, solidMesh(), pairs
                )
            );
            List<SolidPatchPair> solidPairs(1);
            solidPairs[0] = readSolidSolidPatchPair
            (
                coupling, auxiliarySolidRegion, solidRegion
            );
            solidInterfaceMapper.reset
            (
                new OneToOneConformalSolidPatchMapper
                (
                    auxiliarySolidMesh(), solidMesh(), solidPairs, true
                )
            );
        }
        else
        {
            properties.reset
            (
                new GpuSolidThermalProperties
                (
                    exactSubDictionary(coupling, "properties")
                )
            );
            const List<SolidPatchPair> pairs(readPatchPairs(coupling));
            mapper.reset
            (
                new OneToOneConformalSolidPatchMapper
                (
                    fluidMesh, solidMesh(), pairs
                )
            );
        }
        forAll(mapper->solidPatchIds(), pairI)
        {
            const label patchI = mapper->solidPatchIds()[pairI];
            if (!isA<fixedGradientFvPatchScalarField>(Tsolid->boundaryField()[patchI]))
            {
                solidFailure("coupled solid T patch must use fixedGradient");
            }
        }
        if (threeRegion)
        {
            const label auxiliaryPatchI =
                solidInterfaceMapper->fluidPatchIds()[0];
            const label primaryPatchI =
                solidInterfaceMapper->solidPatchIds()[0];
            if
            (
                !isA<fixedGradientFvPatchScalarField>
                (TauxiliarySolid->boundaryField()[auxiliaryPatchI])
             || !isA<fixedGradientFvPatchScalarField>
                (Tsolid->boundaryField()[primaryPatchI])
            )
            {
                solidFailure("solid-solid T patches must use fixedGradient");
            }
        }

        const dictionary& particleContract = resolvedConfiguration.found
        (
            "resolvedParticleContract", false, false
        )
          ? exactSubDictionary(resolvedConfiguration, "resolvedParticleContract")
          : exactSubDictionary(coupling, "resolvedParticleContract");
        rhoSolid = exactLookup<scalar>(particleContract, "rhoSolid");
        particleHeatFactor =
            exactLookup<scalar>(particleContract, "particleHeatFactorJkgK");
        const scalar TpMin = exactLookup<scalar>(particleContract, "TpMin");
        const scalar TpMax = exactLookup<scalar>(particleContract, "TpMax");
        if (radiationEnabled)
        {
            const dictionary& radiation =
                exactSubDictionary(coupling, "radiation");
            fileName tablePath(exactLookup<fileName>(radiation, "mieTable"));
            if (!tablePath.isAbsolute())
            {
                tablePath = runTime.path()/tablePath;
            }
            mieTable.reset(new MieTable(tablePath));
            const dictionary& doDictionary =
                exactSubDictionary(radiation, "do");
            radiationSolver.reset
            (
                new MieDoRadiationSolver
                (
                    fluidMesh, doDictionary, mieTable(),
                    mapper->fluidPatchIds(), readRadiationControls(doDictionary)
                )
            );
            particleRadiationCoupler.reset
            (
                new GpuParticleRadiationCoupler
                (
                    fluidMesh, mieTable(), radiationSolver(),
                    mapper->fluidPatchIds(), TpMin, TpMax
                )
            );
            particleRadiationAbsorptionCoefficient.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        "particleRadiationAbsorptionCoefficient",
                        runTime.timeName(), fluidMesh,
                        IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE, true
                    ),
                    fluidMesh,
                    dimensionedScalar("zero", dimless/dimLength, scalar(0))
                )
            );
            particleRadiationScatteringCoefficient.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        "particleRadiationScatteringCoefficient",
                        runTime.timeName(), fluidMesh,
                        IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE, true
                    ),
                    fluidMesh,
                    dimensionedScalar("zero", dimless/dimLength, scalar(0))
                )
            );
            particleRadiationExtinctionCoefficient.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        "particleRadiationExtinctionCoefficient",
                        runTime.timeName(), fluidMesh,
                        IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE, true
                    ),
                    fluidMesh,
                    dimensionedScalar("zero", dimless/dimLength, scalar(0))
                )
            );
            particleRadiationAsymmetryFactor.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        "particleRadiationAsymmetryFactor",
                        runTime.timeName(), fluidMesh,
                        IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE, true
                    ),
                    fluidMesh,
                    dimensionedScalar("zero", dimless, scalar(0))
                )
            );
        }
        labelList primaryCoupledPatchIds(mapper->solidPatchIds());
        labelList primaryOwnerConductivityPatchIds;
        if (threeRegion)
        {
            const label oldSize = primaryCoupledPatchIds.size();
            primaryCoupledPatchIds.setSize(oldSize + 1);
            primaryCoupledPatchIds[oldSize] =
                solidInterfaceMapper->solidPatchIds()[0];
            primaryOwnerConductivityPatchIds = solidInterfaceMapper->solidPatchIds();
        }
        solidCandidateSolver.reset
        (
            new GpuSolidThermalCandidateSolver
            (
                solidMesh(), properties(), primaryCoupledPatchIds,
                GpuSolidThermalControls{}, primaryOwnerConductivityPatchIds
            )
        );
        if (threeRegion)
        {
            auxiliaryCandidateSolver.reset
            (
                new GpuSolidThermalCandidateSolver
                (
                    auxiliarySolidMesh(), auxiliaryProperties(),
                    solidInterfaceMapper->fluidPatchIds(),
                    GpuSolidThermalControls{},
                    solidInterfaceMapper->fluidPatchIds()
                )
            );
        }
        resident.configureGasWallEnergyLedger(fluidMesh, mapper->fluidPatchIds());
        labelList interfaceSolidSizes(mapper->size());
        List<labelList> interfaceDestinations(mapper->size());
        scalarField interfaceAreas(resident.configuredGasWallEnergyFaceIds().size());
        label interfaceOffset = 0;
        forAll(interfaceSolidSizes, pairI)
        {
            interfaceSolidSizes[pairI] =
                solidMesh->boundary()[mapper->solidPatchIds()[pairI]].size();
            interfaceDestinations[pairI] = mapper->solidFaceForFluidFace(pairI);
            const scalarField& area =
                fluidMesh.magSf().boundaryField()[mapper->fluidPatchIds()[pairI]];
            forAll(area, faceI) interfaceAreas[interfaceOffset++] = area[faceI];
        }
        resident.configureInterfaceExchange
        (
            interfaceSolidSizes, interfaceDestinations, interfaceAreas
        );

        gasConvectiveWallHeatFlux.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    "gasConvectiveWallHeatFlux",
                    runTime.timeName(),
                    fluidMesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE,
                    true
                ),
                fluidMesh,
                dimensionedScalar
                (
                    "zero", dimPower/dimArea, scalar(0)
                )
            )
        );
        particleRadiationWallHeatFlux.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    "particleRadiationWallHeatFlux",
                    runTime.timeName(),
                    fluidMesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE,
                    true
                ),
                fluidMesh,
                dimensionedScalar
                (
                    "zero", dimPower/dimArea, scalar(0)
                )
            )
        );
        particleStuckWallHeatFlux.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    resident.particleStuckWallHeatFluxFieldName(),
                    runTime.timeName(),
                    fluidMesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE,
                    true
                ),
                fluidMesh,
                dimensionedScalar("zero", dimPower/dimArea, scalar(0))
            )
        );
        particleReflectedWallHeatFlux.reset
        (
            new surfaceScalarField
            (
                IOobject
                (
                    resident.particleReflectedWallHeatFluxFieldName(),
                    runTime.timeName(),
                    fluidMesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE,
                    true
                ),
                fluidMesh,
                dimensionedScalar("zero", dimPower/dimArea, scalar(0))
            )
        );
        if (resident.particleWallHeatTransferEnabled())
        {
            particleContactPairIds.setSize(mapper->size());
            particleContactFluidPatchIds.setSize(mapper->size());
            label nContactPairs = 0;
            forAll(mapper->fluidPatchIds(), pairI)
            {
                const label fluidPatchI = mapper->fluidPatchIds()[pairI];
                if
                (
                    resident.particleWallContactEnabledOnPatch
                    (
                        fluidMesh, fluidPatchI
                    )
                )
                {
                    particleContactPairIds[nContactPairs] = pairI;
                    particleContactFluidPatchIds[nContactPairs] = fluidPatchI;
                    ++nContactPairs;
                }
            }
            particleContactPairIds.setSize(nContactPairs);
            particleContactFluidPatchIds.setSize(nContactPairs);

            forAll(fluidMesh.boundary(), patchI)
            {
                if
                (
                    resident.particleWallContactEnabledOnPatch(fluidMesh, patchI)
                 && !isCoupledPatch(mapper->fluidPatchIds(), patchI)
                )
                {
                    solidFailure
                    (
                        "Every heat-transferring stuck wall patch must belong "
                        "to a CHT coupled solid interface"
                    );
                }
            }
            if (particleContactPairIds.empty())
            {
                solidFailure
                (
                    "Particle-wall heat transfer is enabled but the "
                    "intersection of CHT coupled patches and stuck wall "
                    "patches is empty"
                );
            }
        }
        restorePendingWallEnergyLedgers(restartState);
    }

    void restorePendingWallEnergyLedgers
    (
        const ThermalExchangeRestartState& restartState
    )
    {
        const fileName pendingPath
        (
            runTime.timePath()
           /ThermalRestartPreflight::pendingWallEnergyObjectName()
        );
        if (!Foam::isFile(pendingPath))
        {



            const bool legacyReset = restartState.formatVersion >= 1
                                  && restartState.formatVersion <= 3;
            if (!restartState.initialState && !legacyReset)
            {
                solidFailure("restart is missing its required pending wall-energy ledger");
            }
            return;
        }
        const PendingWallEnergyLedgerSnapshot pending =
            ThermalRestartPreflight::readPendingWallEnergyLedgers
            (
                runTime.timePath(),
                runTime.value(),
                restartState.completedSimulationTimeS,
                restartState.fluidMeshTopologySha1,
                fluidMesh.nFaces()
            );
        const labelList& configuredFaces =
            resident.configuredGasWallEnergyFaceIds();
        const bool sameOrder = pending.faceIds.size() == configuredFaces.size()
         && (configuredFaces.empty() || std::memcmp
            (pending.faceIds.begin(), configuredFaces.begin(),
             static_cast<std::size_t>(configuredFaces.size())*sizeof(label)) == 0);
        if (sameOrder)
        {
            resident.restorePendingWallEnergyLedgers
            (
                pending.gasConvectiveEnergyJ,
                pending.particleDepositedEnergyJ,
                pending.particleReflectedEnergyJ
            );
            return;
        }
        std::map<label, label> rowForFace;
        forAll(configuredFaces, faceI) rowForFace.emplace(configuredFaces[faceI], -1);
        forAll(pending.faceIds, entryI)
        {
            const auto found = rowForFace.find(pending.faceIds[entryI]);
            if (found == rowForFace.end())
            {
                solidFailure("pending wall-energy entry has no configured destination face");
            }


            found->second = entryI;
        }
        scalarField gas(configuredFaces.size());
        scalarField deposited(configuredFaces.size());
        scalarField reflected(configuredFaces.size());
        forAll(configuredFaces, faceI)
        {
            const label row = rowForFace.find(configuredFaces[faceI])->second;
            if (row < 0)
            {
                solidFailure("pending wall-energy ledger is missing a required configured face");
            }
            gas[faceI] = pending.gasConvectiveEnergyJ[row];
            deposited[faceI] = pending.particleDepositedEnergyJ[row];
            reflected[faceI] = pending.particleReflectedEnergyJ[row];
        }
        resident.restorePendingWallEnergyLedgers(gas, deposited, reflected);
    }

    PendingWallEnergyLedgerSnapshot pendingWallEnergySnapshot
    (
        const ThermalExchangeRestartState& state
    ) const
    {
        PendingWallEnergyLedgerSnapshot pending;
        pending.checkpointSimulationTimeS = runTime.value();
        pending.completedExchangeSimulationTimeS =
            state.completedSimulationTimeS;
        pending.fluidMeshTopologySha1 = state.fluidMeshTopologySha1;
        pending.nMeshFaces = fluidMesh.nFaces();
        pending.faceIds = resident.configuredGasWallEnergyFaceIds();
        resident.peekPendingWallEnergyLedgers
        (
            pending.gasConvectiveEnergyJ,
            pending.particleDepositedEnergyJ,
            pending.particleReflectedEnergyJ
        );
        return pending;
    }

    std::pair<fileName, fileName> writePendingWallEnergyTemporary
    (
        const PendingWallEnergyLedgerSnapshot& pending
    ) const
    {
        const fileName finalPath
        (
            runTime.timePath()
           /ThermalRestartPreflight::pendingWallEnergyObjectName()
        );
        const fileName temporaryPath(finalPath + ".tmp");
        std::ofstream os(temporaryPath.c_str(), std::ios::out | std::ios::trunc);
        if (!os)
        {
            solidFailure("cannot create temporary pending wall-energy ledger");
        }
        if
        (
            pending.gasConvectiveEnergyJ.size() != pending.faceIds.size()
         || pending.particleDepositedEnergyJ.size() != pending.faceIds.size()
         || pending.particleReflectedEnergyJ.size() != pending.faceIds.size()
        )
        {
            solidFailure("pending wall-energy ledger arrays have inconsistent sizes");
        }
        os.setf(std::ios::scientific);
        os.precision(std::numeric_limits<scalar>::max_digits10);
        os << "FoamFile\n{\nversion 2.0;\nformat ascii;\nclass dictionary;\n"
           << "location \"" << runTime.timeName() << "\";\n"
           << "object gpuPendingWallEnergyLedgers;\n}\n\n"
           << "formatVersion 1;\n"
           << "checkpointSimulationTimeS "
           << pending.checkpointSimulationTimeS << ";\n"
           << "completedExchangeSimulationTimeS "
           << pending.completedExchangeSimulationTimeS << ";\n"
           << "nMeshFaces " << pending.nMeshFaces << ";\n"
           << "nEntries " << pending.faceIds.size() << ";\n\n"
           << "entries\n(\n";
        GpuReal gasTotal = scalar(0);
        GpuReal depositedTotal = scalar(0);
        GpuReal reflectedTotal = scalar(0);
        forAll(pending.faceIds, entryI)
        {
            const GpuReal gas = pending.gasConvectiveEnergyJ[entryI];
            const GpuReal deposited = pending.particleDepositedEnergyJ[entryI];
            const GpuReal reflected = pending.particleReflectedEnergyJ[entryI];

            gasTotal += gas;
            depositedTotal += deposited;
            reflectedTotal += reflected;
            os << "    (" << pending.faceIds[entryI] << " " << gas << " "
               << deposited << " " << reflected << ")\n";
        }
        os << ");\n\n"
           << "gasConvectiveEnergyJ " << gasTotal << ";\n"
           << "particleDepositedEnergyJ " << depositedTotal << ";\n"
           << "particleReflectedEnergyJ " << reflectedTotal << ";\n";
        os.flush();
        os.close();
        if (!os)
        {
            solidFailure("failed completing temporary pending wall-energy ledger");
        }
        return std::make_pair(temporaryPath, finalPath);
    }

    void mapSolidWallTemperatureToFluid(volScalarField& Tgas,bool publishOnly=false)
    {
        const auto deviceInterface=publishOnly?ugkp::gpu::ThermalInterfaceStateDeviceView{}:
            solidCandidateSolver->interfaceState(particleContactPairIds,mapper->size(),radiationEnabled,resident.deviceDiagnostics());
        if(deviceInterface.enabled){
            resident.updateSolidInterfaceState(deviceInterface);
            if(!fluidMesh.time().writeTime())return;
        }
        List<scalarField> contactWallTemperature(particleContactPairIds.size());
        label nextContactI = 0;
        forAll(mapper->fluidPatchIds(), pairI)
        {
            const label fluidPatchI = mapper->fluidPatchIds()[pairI];
            const label solidPatchI = mapper->solidPatchIds()[pairI];
            if (!Tgas.boundaryField()[fluidPatchI].fixesValue())
            {
                solidFailure("coupled fluid T patch must be fixed-value");
            }
            scalarField solidWallTemperature
            (
                coupledPatchSurfaceTemperature(Tsolid(), solidPatchI)
            );
            const scalarField mapped = mapper->mapSolidScalarToFluid
            (
                pairI, solidWallTemperature
            );
            Tgas.boundaryFieldRef()[fluidPatchI] == mapped;
            // Contact pair ids are built in ascending mapper order. Retain
            // only their accepted surfaces, without a second face copy/read.
            if
            (
                nextContactI < particleContactPairIds.size()
             && particleContactPairIds[nextContactI] == pairI
            )
            {
                contactWallTemperature[nextContactI].transfer
                (
                    solidWallTemperature
                );
                ++nextContactI;
            }
        }
        if(publishOnly||deviceInterface.enabled)return;
        resident.updateSolidWallTemperatures
        (
            fluidMesh, Tgas, mapper->fluidPatchIds()
        );
        if (particleContactPairIds.empty())
        {
            return;
        }

        List<scalarField> mappedWallEffusivity(fluidMesh.boundary().size());
        forAll(particleContactPairIds, contactI)
        {
            const label pairI = particleContactPairIds[contactI];
            const label fluidPatchI = mapper->fluidPatchIds()[pairI];
            const scalarField& solidWallTemperature =
                contactWallTemperature[contactI];
            scalarField solidWallEffusivity(solidWallTemperature.size());
            forAll(solidWallEffusivity, faceI)
            {
                const GpuReal Tw = GpuReal(solidWallTemperature[faceI]);
                const GpuReal rhoWall = properties->rho(Tw);
                const GpuReal cpWall = properties->Cp(Tw);
                const GpuReal kappaWall = properties->kappa(Tw);
                const GpuReal squaredEffusivity = rhoWall*cpWall*kappaWall;

                solidWallEffusivity[faceI] = std::sqrt(squaredEffusivity);
            }
            mappedWallEffusivity[fluidPatchI] =
                mapper->mapSolidScalarToFluid(pairI, solidWallEffusivity);
        }
        resident.updateParticleWallEffusivities
        (
            fluidMesh, mappedWallEffusivity, particleContactFluidPatchIds
        );
    }

    List<scalarField> fluidWallEmissivity() const
    {
        List<scalarField> result(fluidMesh.boundary().size());
        forAll(result, patchI)
        {
            result[patchI].setSize(fluidMesh.boundary()[patchI].size(), scalar(0));
        }
        forAll(mapper->fluidPatchIds(), pairI)
        {
            const label fluidPatchI = mapper->fluidPatchIds()[pairI];
            const label solidPatchI = mapper->solidPatchIds()[pairI];
            const scalarField solidTemperature
            (
                coupledPatchSurfaceTemperature(Tsolid(), solidPatchI)
            );
            scalarField solidEmissivity(solidTemperature.size());
            forAll(solidEmissivity, faceI)
            {
                solidEmissivity[faceI] = properties->emissivity
                (
                    solidTemperature[faceI]
                );
            }
            result[fluidPatchI] = mapper->mapSolidScalarToFluid
            (
                pairI, solidEmissivity
            );
        }
        return result;
    }

};

GpuSolidThermalCoupler::GpuSolidThermalCoupler
(
    Time& runTime,
    fvMesh& fluidMesh,
    GpuParticleResidentSolver& resident,
    const dictionary& couplingDictionary,
    const ThermalExchangeRestartState& validatedRestartState
)
:
    data_
    (
        new GpuSolidThermalCouplerData
        (
            runTime, fluidMesh, resident, couplingDictionary,
            validatedRestartState
        )
    )
{}

GpuSolidThermalCoupler::~GpuSolidThermalCoupler() = default;

void GpuSolidThermalCoupler::initialiseWallTemperatureBeforeFirstAdvance
(
    volScalarField& Tgas
)
{
    data_->mapSolidWallTemperatureToFluid(Tgas);
    if(!data_->threeRegion)data_->solidCandidateSolver->prepareResident(data_->Tsolid(),data_->resident.interfaceEnergyDeviceState(),data_->resident.deviceDiagnostics());
    else
    {
        const label auxiliaryPatchI=data_->solidInterfaceMapper->fluidPatchIds()[0];
        const label primaryPatchI=data_->solidInterfaceMapper->solidPatchIds()[0];
        const auto& auxiliaryPatch=data_->auxiliarySolidMesh->boundary()[auxiliaryPatchI];
        const auto& primaryPatch=data_->solidMesh->boundary()[primaryPatchI];
        const auto nearest=data_->solidInterfaceMapper->solidFaceForFluidFace(0);
        data_->solidContact.reset(new ugkp::gpu::SolidThermalContact);
        data_->solidContact->begin(std::vector<int>(nearest.begin(),nearest.end()),
            std::vector<double>(primaryPatch.deltaCoeffs().begin(),primaryPatch.deltaCoeffs().end()),
            std::vector<double>(auxiliaryPatch.deltaCoeffs().begin(),auxiliaryPatch.deltaCoeffs().end()),
            std::vector<double>(auxiliaryPatch.magSf().begin(),auxiliaryPatch.magSf().end()));
        const auto gasEnergy=data_->resident.interfaceEnergyDeviceState();
        const auto primaryEnergy=data_->solidContact->primaryEnergy(gasEnergy.interval);
        const auto auxiliaryEnergy=data_->solidContact->auxiliaryEnergy(gasEnergy.interval);
        const bool primaryPrepared=data_->solidCandidateSolver->prepareResident(data_->Tsolid(),gasEnergy,
            data_->resident.deviceDiagnostics(),primaryEnergy);
        const bool auxiliaryPrepared=data_->auxiliaryCandidateSolver->prepareResident(data_->TauxiliarySolid(),auxiliaryEnergy,
            data_->resident.deviceDiagnostics());
        if(primaryPrepared&&auxiliaryPrepared){
            auto& primary=data_->solidCandidateSolver->prepareResidentContact(primaryPatchI);
            auto& auxiliary=data_->auxiliaryCandidateSolver->prepareResidentContact(auxiliaryPatchI);
            data_->solidContactGraph.reset(new ugkp::gpu::SolidContactIterationGraph(primary,auxiliary,*data_->solidContact,
                {data_->solidInterfaceMaxIterations,GpuReal(data_->solidInterfaceTemperatureTolerance),GpuReal(data_->solidInterfaceEnergyTolerance)},1,gasEnergy.interval));
        }
    }
    data_->wallInitialised = true;
}

GpuThermalCouplingResult GpuSolidThermalCoupler::exchangeIfDue
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
)
{
    GpuThermalCouplingResult result;
    if (!data_->wallInitialised)
    {
        initialiseWallTemperatureBeforeFirstAdvance(Tgas);
    }
    const scalar elapsedSinceCoupling =
        runTime.value()
      - data_->writeState.committedState().completedSimulationTimeS;

    if
    (
        !ugkpcht::thermalEventIsDue
        (
            elapsedSinceCoupling,
            data_->solidContactCouplingInterval
        )
    )
    {
        return result;
    }
    const ThermalExchangeWriteDecision decision =
        data_->writeState.prepareWriteEvent
        (
            true, runTime.timeIndex(), runTime.value()
        );
    if (!decision.exchangeRequired)
    {
        return result;
    }
    const scalar deltaTExchange = decision.deltaTExchangeS;
    const scalar elapsedSinceRadiation =
        runTime.value()
      - data_->writeState.committedState()
            .completedRadiationSimulationTimeS;

    const bool radiationDue =
        data_->radiationEnabled
     && decision.exchangeSequence
          % static_cast<std::uint64_t>(data_->radiationCouplingFrequency)
        == 0;
    bool commitPhaseStarted = false;

    try
    {
        scalarField particleMassInCellKg;
        scalarField particleRadiationEpsilonS;
        scalarField particleRadiationMeanTemperatureK;
        scalarField particleRadiationDiameterM;
        autoPtr<ParticleRadiationMaterialResult> radiation;
        if (radiationDue)
        {
            const auto residentRadiation=data_->resident.radiationParticleDeviceState();
            // Configured wall emissivity is combined with resident contact
            // coverage by the shared radiation input kernel.
            List<scalarField> emissivity;
            if(residentRadiation.boundaryEmissivity){
                emissivity.setSize(data_->fluidMesh.boundary().size());
                forAll(emissivity,p)emissivity[p].setSize(data_->fluidMesh.boundary()[p].size(),scalar(0));
            }else emissivity=data_->fluidWallEmissivity();
            const ParticleRadiationMaterialSnapshot radiationSnapshot
            {
                particleRadiationEpsilonS,
                particleMassInCellKg,
                particleRadiationMeanTemperatureK,
                particleRadiationDiameterM,
                Tgas,
                emissivity,
                data_->particleHeatFactor,
                elapsedSinceRadiation,
                data_->resident.particleRadiationResponse(),
                residentRadiation
            };
            radiation.reset
            (
                new ParticleRadiationMaterialResult
                (
                    data_->particleRadiationCoupler->solve(radiationSnapshot,false)
                )
            );
            data_->lastRadiationIntervalS=elapsedSinceRadiation;
        }

        // Physical input remains resident from DO material response through the
        // shared wall ledger and mapper. Host fields below are output/audit mirrors.
        scalarField compactRadiationEnergy;
        List<scalarField> solidEnergy;
        List<scalarField> gasHeatFlux;
        List<scalarField> depositedHeatFlux;
        List<scalarField> reflectedHeatFlux;
        ugkp::gpu::ThermalInterfaceEnergyDeviceView residentSolidEnergy;
        const bool residentSolid=data_->solidCandidateSolver->residentPrepared()&&(!data_->threeRegion||bool(data_->solidContactGraph));
        if(residentSolid){
            data_->resident.prepareInterfaceExchangeDevice(deltaTExchange,radiationDue?&radiation().resident:nullptr);
            residentSolidEnergy=data_->resident.interfaceEnergyDeviceState();
        }else data_->resident.previewInterfaceExchange
        (
            deltaTExchange, compactRadiationEnergy, solidEnergy,
            gasHeatFlux, depositedHeatFlux, reflectedHeatFlux,
            radiationDue ? &radiation().resident : nullptr, &residentSolidEnergy
        );
        autoPtr<GpuSolidThermalCandidate> solidCandidate;
        autoPtr<GpuSolidThermalCandidate> auxiliaryCandidate;
        if(residentSolid){
            if(data_->threeRegion){
                data_->solidContactGraph->enqueue(residentSolidEnergy.producerStream);
                data_->solidCandidateSolver->acceptResidentContact();data_->auxiliaryCandidateSolver->acceptResidentContact();
            }else data_->solidCandidateSolver->advanceResident(residentSolidEnergy.producerStream);
        }
        else if (!data_->threeRegion)
        {
            GpuSolidThermalCandidate candidate =
                data_->solidCandidateSolver->solveTemporarySolidCandidate
                (
                    data_->Tsolid(), solidEnergy, deltaTExchange, residentSolidEnergy
                );
            solidCandidate.reset(new GpuSolidThermalCandidate(candidate));
        }
        else
        {
            const label auxiliaryPatchI =
                data_->solidInterfaceMapper->fluidPatchIds()[0];
            const label primaryPatchI =
                data_->solidInterfaceMapper->solidPatchIds()[0];
            const fvPatch& auxiliaryPatch =
                data_->auxiliarySolidMesh->boundary()[auxiliaryPatchI];
            const fvPatch& primaryPatch =
                data_->solidMesh->boundary()[primaryPatchI];
            if(!data_->solidContact)data_->solidContact.reset(new ugkp::gpu::SolidThermalContact);
            const auto nearest=data_->solidInterfaceMapper->solidFaceForFluidFace(0);
            data_->solidContact->begin(std::vector<int>(nearest.begin(),nearest.end()),
                std::vector<double>(primaryPatch.deltaCoeffs().begin(),primaryPatch.deltaCoeffs().end()),
                std::vector<double>(auxiliaryPatch.deltaCoeffs().begin(),auxiliaryPatch.deltaCoeffs().end()),
                std::vector<double>(auxiliaryPatch.magSf().begin(),auxiliaryPatch.magSf().end()));
            auto primaryState=data_->solidCandidateSolver->contactState(data_->Tsolid(),primaryPatchI,true);
            auto auxiliaryState=data_->auxiliaryCandidateSolver->contactState(data_->TauxiliarySolid(),auxiliaryPatchI,true);
            for(label interfaceIteration=0;interfaceIteration<max(label(1),data_->solidInterfaceMaxIterations);++interfaceIteration)
            {
                ugkp::gpu::ThermalInterfaceEnergyDeviceView primaryContactEnergy,auxiliaryContactEnergy;
                std::vector<double> primaryAudit,auxiliaryAudit;
                data_->solidContact->evaluate(primaryState,auxiliaryState,deltaTExchange,
                    primaryContactEnergy,auxiliaryContactEnergy,primaryAudit,auxiliaryAudit);
                // These endpoint mirrors retain native custom-boundary semantics
                // and the candidate's audit. Device segments feed standard RHSs.
                List<scalarField> auxiliaryEnergyList(1),primaryEnergy(data_->mapper->size()+1);
                auxiliaryEnergyList[0].setSize(auxiliaryAudit.size());std::copy(auxiliaryAudit.begin(),auxiliaryAudit.end(),auxiliaryEnergyList[0].begin());
                forAll(solidEnergy,pairI)primaryEnergy[pairI]=solidEnergy[pairI];
                primaryEnergy.last().setSize(primaryAudit.size());std::copy(primaryAudit.begin(),primaryAudit.end(),primaryEnergy.last().begin());
                GpuSolidThermalCandidate nextAuxiliary=data_->auxiliaryCandidateSolver->solveTemporarySolidCandidate
                    (data_->TauxiliarySolid(),auxiliaryEnergyList,deltaTExchange,auxiliaryContactEnergy);
                GpuSolidThermalCandidate nextPrimary=data_->solidCandidateSolver->solveTemporarySolidCandidate
                    (data_->Tsolid(),primaryEnergy,deltaTExchange,residentSolidEnergy,primaryContactEnergy);
                primaryState=data_->solidCandidateSolver->contactState(data_->Tsolid(),primaryPatchI,false);
                auxiliaryState=data_->auxiliaryCandidateSolver->contactState(data_->TauxiliarySolid(),auxiliaryPatchI,false);
                const auto convergence=data_->solidContact->convergence(primaryState,auxiliaryState);
                auxiliaryCandidate.reset(new GpuSolidThermalCandidate(nextAuxiliary));
                solidCandidate.reset(new GpuSolidThermalCandidate(nextPrimary));
                if(convergence.temperatureChange<=data_->solidInterfaceTemperatureTolerance
                    &&convergence.energyDifference/convergence.energyScale<=data_->solidInterfaceEnergyTolerance)break;
                if(interfaceIteration+1==max(label(1),data_->solidInterfaceMaxIterations)&&hostDiagnosticWarningsEnabled())
                    WarningInFunction << "solid interface iteration budget exhausted; time=" << data_->fluidMesh.time().timeName()
                        << "; iterations=" << interfaceIteration+1 << "; temperatureResidual=" << convergence.temperatureChange
                        << "; temperatureLimit=" << data_->solidInterfaceTemperatureTolerance
                        << "; energyResidual=" << convergence.energyDifference/convergence.energyScale
                        << "; energyLimit=" << data_->solidInterfaceEnergyTolerance << "; retaining computed candidates" << endl;
            }
        }

        commitPhaseStarted = true;
        if (radiationDue)
        {
            data_->resident.applyParticleRadiationDeviceEnergy(radiation().resident);
        }
        data_->resident.resetGasWallEnergy();
        data_->resident.resetParticleWallHeatLedgers();
        if(!residentSolid){
        data_->gasConvectiveWallHeatFlux->primitiveFieldRef() = scalar(0);
        forAll(data_->gasConvectiveWallHeatFlux->boundaryField(), patchI)
        {
            data_->gasConvectiveWallHeatFlux->boundaryFieldRef()[patchI] =
                scalar(0);
        }
        forAll(data_->mapper->fluidPatchIds(), pairI)
        {
            const label patchI = data_->mapper->fluidPatchIds()[pairI];
            auto& flux =
                data_->gasConvectiveWallHeatFlux->boundaryFieldRef()[patchI];
            flux = gasHeatFlux[pairI];
        }
        data_->particleStuckWallHeatFlux->primitiveFieldRef() = scalar(0);
        data_->particleReflectedWallHeatFlux->primitiveFieldRef() = scalar(0);
        forAll(data_->particleStuckWallHeatFlux->boundaryField(), patchI)
        {
            data_->particleStuckWallHeatFlux->boundaryFieldRef()[patchI] =
                scalar(0);
            data_->particleReflectedWallHeatFlux->boundaryFieldRef()[patchI] =
                scalar(0);
        }
        forAll(data_->mapper->fluidPatchIds(), pairI)
        {
            const label patchI = data_->mapper->fluidPatchIds()[pairI];
            data_->particleStuckWallHeatFlux->boundaryFieldRef()[patchI] =
                depositedHeatFlux[pairI];
            data_->particleReflectedWallHeatFlux->boundaryFieldRef()[patchI] =
                reflectedHeatFlux[pairI];
        }
        data_->solidCandidateSolver->publishSolidCandidate
        (
            solidCandidate(), data_->Tsolid()
        );
        }
        if (data_->threeRegion&&!residentSolid)
        {
            data_->auxiliaryCandidateSolver->publishSolidCandidate
            (
                auxiliaryCandidate(), data_->TauxiliarySolid()
            );
        }
        if(residentSolid&&runTime.writeTime()){
            data_->solidCandidateSolver->publishResident(data_->Tsolid());
            if(data_->threeRegion)data_->auxiliaryCandidateSolver->publishResident(data_->TauxiliarySolid());
        }
        data_->mapSolidWallTemperatureToFluid(Tgas);
        if (radiationDue)
        {
            data_->resident.refreshParticleEnthalpyAfterRadiation();
        }

        if (runTime.writeTime())
        {
            data_->resident.downloadToHostMirror
            (
                runTime, rho, rhoU, rhoE, U, p, Tgas, epsilonS, rhoUs, rhoEs,
                rhoDs, rhoHp, Us, theta, Tp, dMeanCell, true
            );
        }
        ThermalExchangeRestartState completed =
            data_->writeState.committedState();
        completed.formatVersion = 4;
        completed.initialState = false;
        completed.exchangeSequence = decision.exchangeSequence;
        completed.previousExchangeSimulationTimeS =
            data_->writeState.committedState().completedSimulationTimeS;
        completed.completedSimulationTimeS = decision.simulationTimeS;
        if (radiationDue)
        {
            completed.completedRadiationSimulationTimeS =
                decision.simulationTimeS;
        }
        completed.completedTimeIndex = decision.timeIndex;
        completed.gasWallLedgerConsumed = true;
        completed.particleWallLedgerConsumed = true;
        completed.particleRadiationApplied = true;
        completed.solidStateUpdated = true;
        completed.wallTemperatureUploaded = true;
        completed.particleMomentsRebuilt = true;
        data_->writeState.commitPreparedExchange(completed);
        result.coupled = true;
        result.radiationCoupled = radiationDue;
        result.elapsedCouplingTimeS = deltaTExchange;
        result.elapsedRadiationTimeS =
            radiationDue ? elapsedSinceRadiation : scalar(0);
        return result;
    }
    catch (...)
    {
        if (!commitPhaseStarted && data_->writeState.hasPendingExchange())
        {
            data_->writeState.cancelPreparedExchange();
        }
        throw;
    }
}

label GpuSolidThermalCoupler::persistAtWriteTime
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
    volScalarField& dMeanCell,
    const bool thermalCouplingWrite
)
{
    if
    (
        thermalCouplingWrite
     || data_->writeState.committedState().completedSimulationTimeS
        != runTime.value()
    )
    {
        data_->resident.downloadToHostMirror
        (
            runTime, rho, rhoU, rhoE, U, p, Tgas, epsilonS, rhoUs, rhoEs,
            rhoDs, rhoHp, Us, theta, Tp, dMeanCell, true
        );
    }
    if(data_->solidCandidateSolver->residentAdvanced()){
        data_->solidCandidateSolver->publishResident(data_->Tsolid());
        if(data_->threeRegion&&data_->solidContactGraph)data_->auxiliaryCandidateSolver->publishResident(data_->TauxiliarySolid());
        // The preview survives consumption of the accumulated ledgers. Publish
        // the last completed exchange, never a recomputation from reset ledgers.
        List<scalarField> gas,deposited,reflected;
        data_->resident.collectInterfaceExchange(gas,deposited,reflected);
        data_->gasConvectiveWallHeatFlux->primitiveFieldRef()=scalar(0);
        data_->particleStuckWallHeatFlux->primitiveFieldRef()=scalar(0);
        data_->particleReflectedWallHeatFlux->primitiveFieldRef()=scalar(0);
        forAll(data_->gasConvectiveWallHeatFlux->boundaryField(),p){
            data_->gasConvectiveWallHeatFlux->boundaryFieldRef()[p]=scalar(0);
            data_->particleStuckWallHeatFlux->boundaryFieldRef()[p]=scalar(0);
            data_->particleReflectedWallHeatFlux->boundaryFieldRef()[p]=scalar(0);
        }
        forAll(data_->mapper->fluidPatchIds(),i){const label p=data_->mapper->fluidPatchIds()[i];
            data_->gasConvectiveWallHeatFlux->boundaryFieldRef()[p]=gas[i];
            data_->particleStuckWallHeatFlux->boundaryFieldRef()[p]=deposited[i];
            data_->particleReflectedWallHeatFlux->boundaryFieldRef()[p]=reflected[i];
        }
    }
    const GpuThermalRestartMirrorMetadata mirrors =
        data_->resident.stageThermalRestartMirrors
        (
            runTime,
            thermalCouplingWrite
        );
    ThermalExchangeRestartState completed =
        data_->writeState.committedState();
    completed.formatVersion = 4;
    const PendingWallEnergyLedgerSnapshot pendingWallEnergy =
        data_->pendingWallEnergySnapshot(completed);
    const std::pair<fileName, fileName> preparedPendingWallEnergy =
        data_->writePendingWallEnergyTemporary(pendingWallEnergy);
    IOdictionary stateObject
    (
        IOobject
        (
            ThermalRestartPreflight::stateObjectName(),
            runTime.timeName(), runTime,
            IOobject::NO_READ, IOobject::AUTO_WRITE, true
        ),
        ThermalRestartPreflight::stateDictionary(completed)
    );
    if(data_->lastRadiationIntervalS>0){
        const auto radiation=data_->particleRadiationCoupler->collect();
        data_->particleRadiationAbsorptionCoefficient->primitiveFieldRef()=radiation.absorptionCoefficientInvM;
        data_->particleRadiationScatteringCoefficient->primitiveFieldRef()=radiation.scatteringCoefficientInvM;
        data_->particleRadiationExtinctionCoefficient->primitiveFieldRef()=radiation.extinctionCoefficientInvM;
        data_->particleRadiationAsymmetryFactor->primitiveFieldRef()=radiation.asymmetryFactor;
        data_->particleRadiationWallHeatFlux->primitiveFieldRef()=scalar(0);
        forAll(data_->particleRadiationWallHeatFlux->boundaryField(),patchI)
            data_->particleRadiationWallHeatFlux->boundaryFieldRef()[patchI]=scalar(0);
        forAll(data_->mapper->fluidPatchIds(),pairI){
            const label patchI=data_->mapper->fluidPatchIds()[pairI];
            auto& flux=data_->particleRadiationWallHeatFlux->boundaryFieldRef()[patchI];
            const auto& energy=radiation.solidWallRadiationEnergyJ[patchI];
            const auto& area=data_->fluidMesh.magSf().boundaryField()[patchI];
            forAll(flux,faceI)flux[faceI]=energy[faceI]/(area[faceI]*data_->lastRadiationIntervalS);
        }
    }
    data_->mapSolidWallTemperatureToFluid(Tgas,true);
    data_->resident.downloadSharedGasSpecies(rho);
    const bool fieldWriteSucceeded =
        thermalCouplingWrite && !runTime.writeTime()
      ? runTime.writeNow()
      : runTime.write();
    if (!fieldWriteSucceeded)
    {
        FatalIOErrorInFunction(runTime.controlDict())
            << "ordinary OpenFOAM field write failed during thermal persistence; time="
            << runTime.timeName() << exit(FatalIOError);
    }
    ThermalRestartPreflight::commitManifestAtomically
    (
        preparedPendingWallEnergy.first,
        preparedPendingWallEnergy.second
    );
    return mirrors.particleCount;
}

}
}

#endif
