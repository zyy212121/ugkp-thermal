#include "GpuSolidThermalCoupler.H"
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

scalar relativeTemperatureChange
(
    const scalarField& previous,
    const scalarField& current
)
{
    scalar maximum = scalar(0);
    forAll(current, cellI)
    {
        maximum = max
        (
            maximum,
            mag(current[cellI] - previous[cellI])
           /max(max(mag(current[cellI]), mag(previous[cellI])), scalar(1))
        );
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

    scalar numerator = scalar(0);
    scalar denominator = scalar(0);
    forAll(currentResidual, cellI)
    {
        const scalar difference =
            currentResidual[cellI] - previousResidual[cellI];
        numerator += previousResidual[cellI]*difference;
        denominator += difference*difference;
    }
    if (!finiteScalar(denominator) || denominator <= VSMALL)
    {
        return min
        (
            max(previousRelaxation, minimumRelaxation),
            maximumRelaxation
        );
    }

    const scalar candidate =
        -previousRelaxation*numerator/denominator;
    if (!finiteScalar(candidate))
    {
        return min
        (
            max(previousRelaxation, minimumRelaxation),
            maximumRelaxation
        );
    }
    return min(max(candidate, minimumRelaxation), maximumRelaxation);
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
}

GpuSolidThermalCandidate
GpuSolidThermalCandidateSolver::solveTemporarySolidCandidate
(
    const volScalarField& registeredTemperature,
    const List<scalarField>& interfaceEnergyJ,
    const scalar deltaTExchange
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

    scalar totalInterfaceEnergyJ = scalar(0);
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

            totalInterfaceEnergyJ += energyJ;
        }
    }

    const scalarField oldInternal(candidate->primitiveField());
    scalarField previous(candidate->primitiveField());
    bool havePreviousFixedPointResidual = false;
    scalar nonlinearRelaxation = controls_.nonlinearInitialRelaxation;
    const dimensionedScalar dt("deltaTExchange", dimTime, deltaTExchange);
    const dimensionSet rhoCpDimensions = dimEnergy/dimVolume/dimTemperature;
    const dimensionSet kappaDimensions = dimPower/dimLength/dimTemperature;
    scalar finalResidual = std::numeric_limits<scalar>::quiet_NaN();
    scalar finalTemperatureResidual =
        std::numeric_limits<scalar>::quiet_NaN();
    scalar finalSurfaceResidual = std::numeric_limits<scalar>::quiet_NaN();
    scalar finalSurfaceFluxResidual = std::numeric_limits<scalar>::quiet_NaN();
    scalar finalEnergyLimit = scalar(0);

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
        std::vector<ugkp::gpu::ThermalPropertyQuery> propertyQueries;
        propertyQueries.reserve(mesh_.nCells() + mesh_.nFaces() - mesh_.nInternalFaces());
        forAll(candidate->primitiveField(), cellI)
            propertyQueries.push_back({(*candidate)[cellI], oldInternal[cellI]});
        forAll(mesh_.boundary(), patchI)
        {
            const bool ownerConductivity = isCoupledPatch(ownerConductivityPatchIds_, patchI);
            const auto& cells = mesh_.boundary()[patchI].faceCells();
            forAll(mesh_.boundary()[patchI], faceI)
            {
                const scalar T = ownerConductivity ? (*candidate)[cells[faceI]]
                    : candidate->boundaryField()[patchI][faceI];
                propertyQueries.push_back({T, T});
            }
        }
        std::vector<ugkp::gpu::ThermalPropertyResult> propertyResults;
        deviceProperties_->evaluate(propertyQueries, propertyResults);
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
        checkPropertyResults(propertyResults);
        std::size_t propertyIndex = 0;
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
                const scalar area = mesh_.magSf().boundaryField()[patchI][faceI];

                const scalar denominator =
                    area*deltaTExchange*kappa.boundaryField()[patchI][faceI];

                interfacePatch.gradient()[faceI] =
                    interfaceEnergyJ[coupledI][faceI]/denominator;
            }
        }
        candidate->correctBoundaryConditions();

        const autoPtr<volScalarField> oldTemperature = thermalFieldOnMesh
        (
            IOobject
            (
                "TsolidOldCandidate", mesh_.time().timeName(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_, registeredTemperature
        );
        fvScalarMatrix equation
        (
            ugkp::gpu::gpuImplicitThermalEquation(kappa, *candidate, Csec, oldTemperature(), deltaTExchange, true)
        );
        solverPerformance linearPerformance;
        {
            ScalarSolverLogSilencer silenceSolverLog;
            if (!sparse_) sparse_.reset(new ugkp::gpu::SparseConjugateGradient);
            const dictionary& linearControls = mesh_.solution().solverDict
            (
                candidate->select
                (
                    !mesh_.schemes().steady()
                 && mesh_.data::lookupOrDefault<bool>("finalIteration", false)
                )
            );
            linearPerformance = ugkp::gpu::solveFoamScalarMatrix(equation, linearControls, *sparse_);
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
            const dictionary& linearControls = mesh_.solution().solverDict
            (
                candidate->select
                (
                    !mesh_.schemes().steady()
                 && mesh_.data::lookupOrDefault<bool>("finalIteration", false)
                )
            );
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
            nonlinearRelaxation = deviceProperties_->relax
            (
                candidate->primitiveFieldRef().begin(), previous.begin(), previous.size(),
                havePreviousFixedPointResidual, nonlinearRelaxation,
                controls_.nonlinearMinimumRelaxation, controls_.nonlinearMaximumRelaxation, VSMALL
            );
            havePreviousFixedPointResidual = true;
        }
        candidate->correctBoundaryConditions();
        finalSurfaceResidual = scalar(0);
        finalSurfaceFluxResidual = scalar(0);
        // Preserve the original surface-then-cell consumption order while
        // evaluating the completed iterate's conductivity and caloric integral.
        propertyQueries.clear();
        forAll(coupledPatchIds_, coupledI)
        {
            const label patchI=coupledPatchIds_[coupledI];
            if (isCoupledPatch(ownerConductivityPatchIds_, patchI)) continue;
            forAll(candidate->boundaryField()[patchI], faceI)
                propertyQueries.push_back({candidate->boundaryField()[patchI][faceI], 0,
                    ugkp::gpu::ThermalConductivity});
        }
        const std::size_t surfacePropertyCount=propertyQueries.size();
        forAll(candidate->primitiveField(), cellI)
            propertyQueries.push_back({(*candidate)[cellI], oldInternal[cellI],
                ugkp::gpu::ThermalEnergy|ugkp::gpu::ThermalTemperatureChange, previous[cellI]});
        deviceProperties_->evaluate(propertyQueries, propertyResults);
        checkPropertyResults(propertyResults);
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
                const scalar surface = patch[faceI];
                const scalar previousSurfaceTemperature =
                    previousPatchSurface[faceI];
                finalSurfaceResidual = max
                (
                    finalSurfaceResidual,
                    mag(surface - previousSurfaceTemperature)
                   /max
                    (
                        max(mag(surface), mag(previousSurfaceTemperature)),
                        scalar(1)
                    )
                );
                // The completed-interval ledger remains prescribed. Iterate
                // only its solid surface closure k(Tw)*grad(T) = E/(A*dt).
                const scalar prescribedFlux = interfaceEnergyJ[coupledI][faceI]
                   /(area[faceI]*deltaTExchange);
                const scalar surfaceFlux =
                    propertyResults[propertyIndex++].conductivity*gradient[faceI];
                finalSurfaceFluxResidual = max
                (
                    finalSurfaceFluxResidual,
                    mag(surfaceFlux - prescribedFlux)
                   /max(max(mag(surfaceFlux), mag(prescribedFlux)), scalar(1))
                );
                previousPatchSurface[faceI] = surface;
            }
        }
        const tmp<surfaceScalarField> matrixFlux = equation.flux();
        scalar externalBoundaryEnergyIntoSolidJ = scalar(0);
        forAll(matrixFlux().boundaryField(), patchI)
        {
            if (isCoupledPatch(coupledPatchIds_, patchI))
            {
                continue;
            }
            forAll(matrixFlux().boundaryField()[patchI], faceI)
            {
                const scalar faceFluxW =
                    matrixFlux().boundaryField()[patchI][faceI];

                externalBoundaryEnergyIntoSolidJ -= deltaTExchange*faceFluxW;
            }
        }

        scalar enthalpyChangeJ = scalar(0);
        finalTemperatureResidual = scalar(0);
        forAll(candidate->primitiveField(), cellI)
        {
            const auto& value=propertyResults[surfacePropertyCount+cellI];
            enthalpyChangeJ += mesh_.V()[cellI]*value.energy;
            finalTemperatureResidual=max(finalTemperatureResidual, value.relativeChange);
        }
        finalResidual =
            enthalpyChangeJ
          - totalInterfaceEnergyJ
          - externalBoundaryEnergyIntoSolidJ;

        const scalar energyScale = max
        (
            max
            (
                mag(enthalpyChangeJ),
                mag(totalInterfaceEnergyJ)
              + mag(externalBoundaryEnergyIntoSolidJ)
            ),
            scalar(1)
        );
        const bool temperatureConverged =
            finalTemperatureResidual <= controls_.nonlinearRelativeTolerance;
        finalEnergyLimit = controls_.energyAbsoluteTolerance
          + controls_.energyRelativeTolerance*energyScale;
        const bool energyConverged = mag(finalResidual) <= finalEnergyLimit;
        const bool surfaceConverged =
            max(finalSurfaceResidual, finalSurfaceFluxResidual)
         <= controls_.nonlinearRelativeTolerance;
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

MieDoControls readDoControls(const dictionary& doDictionary)
{
    return MieDoControls
    {
        exactLookup<label>(doDictionary, "nTheta"),
        exactLookup<label>(doDictionary, "nPhi"),
        exactLookup<label>(doDictionary, "maxSourceIterations"),
        exactLookup<scalar>(doDictionary, "sourceRelativeTolerance"),
        exactLookup<scalar>(doDictionary, "phaseBalanceTolerance"),
        exactLookup<label>(doDictionary, "maxPhaseBalanceIterations"),
        exactLookup<scalar>(doDictionary, "radiationActiveEpsilonMin"),
        exactLookup<scalar>(doDictionary, "conservationRelativeTolerance"),
        exactLookup<scalar>(doDictionary, "maxPhaseScalingCorrection"),
        doDictionary.lookupOrDefault<label>("angleParallelThreads", 1)
    };
}

GpuSolidThermalControls readSolidControls(const dictionary& coupling)
{
    GpuSolidThermalControls result;
    result.nonlinearMaxIterations =
        exactLookup<label>(coupling, "nonlinearMaxIterations");
    result.nonlinearRelativeTolerance =
        exactLookup<scalar>(coupling, "nonlinearRelativeTolerance");
    result.energyRelativeTolerance =
        exactLookup<scalar>(coupling, "energyRelativeTolerance");
    result.energyAbsoluteTolerance =
        exactLookup<scalar>(coupling, "energyAbsoluteTolerance");
    result.nonlinearAitken =
        coupling.lookupOrDefault<bool>("nonlinearAitken", false);
    result.nonlinearInitialRelaxation =
        coupling.lookupOrDefault<scalar>
        (
            "nonlinearInitialRelaxation", scalar(1)
        );
    result.nonlinearMinimumRelaxation =
        coupling.lookupOrDefault<scalar>
        (
            "nonlinearMinimumRelaxation", scalar(0.05)
        );
    result.nonlinearMaximumRelaxation =
        coupling.lookupOrDefault<scalar>
        (
            "nonlinearMaximumRelaxation", scalar(1)
        );
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
    word solidRegion;
    autoPtr<fvMesh> solidMesh;
    autoPtr<volScalarField> Tsolid;
    autoPtr<GpuSolidThermalProperties> properties;
    autoPtr<GpuSolidPatchMapper> mapper;
    autoPtr<MieTable> mieTable;
    autoPtr<MieDoRadiationSolver> radiationSolver;
    autoPtr<GpuParticleRadiationCoupler> particleRadiationCoupler;
    autoPtr<GpuSolidThermalCandidateSolver> solidCandidateSolver;
    bool radiationEnabled;
    scalar rhoSolid;
    scalar particleHeatFactor;
    scalar solidContactCouplingInterval;
    label radiationCouplingFrequency;
    label solidInterfaceMaxIterations;
    scalar solidInterfaceTemperatureTolerance;
    scalar solidInterfaceEnergyTolerance;
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
        const scalar legacyCouplingInterval =
            exactLookup<scalar>(coupling, "couplingInterval");
        radiationEnabled =
            coupling.lookupOrDefault<bool>("radiationEnabled", true);
        solidContactCouplingInterval = coupling.lookupOrDefault<scalar>
        (
            "solidContactCouplingInterval", legacyCouplingInterval
        );
        radiationCouplingFrequency = coupling.lookupOrDefault<label>
        (
            "radiationCouplingFrequency", 1
        );
        if
        (
            radiationEnabled && radiationCouplingFrequency == 0
        )
        {
            solidFailure("active radiation frequency is zero");
        }
        if (threeRegion)
        {
            solidInterfaceMaxIterations =
                exactLookup<label>(coupling, "solidInterfaceMaxIterations");
            solidInterfaceTemperatureTolerance = exactLookup<scalar>
            (
                coupling, "solidInterfaceTemperatureRelativeTolerance"
            );
            solidInterfaceEnergyTolerance = exactLookup<scalar>
            (
                coupling, "solidInterfaceEnergyRelativeTolerance"
            );
            solidRegion = exactLookup<word>(coupling, "primarySolidRegion");
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
                    mapper->fluidPatchIds(), readDoControls(doDictionary)
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
                readSolidControls(coupling), primaryOwnerConductivityPatchIds
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
                    readSolidControls(coupling),
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
        scalar gasTotal = scalar(0);
        scalar depositedTotal = scalar(0);
        scalar reflectedTotal = scalar(0);
        forAll(pending.faceIds, entryI)
        {
            const scalar gas = pending.gasConvectiveEnergyJ[entryI];
            const scalar deposited = pending.particleDepositedEnergyJ[entryI];
            const scalar reflected = pending.particleReflectedEnergyJ[entryI];

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

    void mapSolidWallTemperatureToFluid(volScalarField& Tgas)
    {
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
                const scalar Tw = solidWallTemperature[faceI];
                const scalar rhoWall = properties->rho(Tw);
                const scalar cpWall = properties->Cp(Tw);
                const scalar kappaWall = properties->kappa(Tw);
                const scalar squaredEffusivity = rhoWall*cpWall*kappaWall;

                solidWallEffusivity[faceI] = sqrt(squaredEffusivity);
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
            data_->resident.downloadToHostMirror
            (
                runTime, rho, rhoU, rhoE, U, p, Tgas, epsilonS, rhoUs, rhoEs,
                rhoDs, rhoHp, Us, theta, Tp, dMeanCell, true
            );
            data_->resident.downloadMobileParticleRadiationMaterial
            (
                data_->fluidMesh,
                particleMassInCellKg,
                particleRadiationMeanTemperatureK,
                particleRadiationDiameterM
            );
            particleRadiationEpsilonS.setSize
            (
                data_->fluidMesh.nCells(), scalar(0)
            );
            forAll(particleMassInCellKg, cellI)
            {
                particleRadiationEpsilonS[cellI] =
                    particleMassInCellKg[cellI]
                   /(data_->rhoSolid*data_->fluidMesh.V()[cellI]);
            }
            List<scalarField> emissivity(data_->fluidWallEmissivity());
            const scalarField occupiedContactArea
            (
                data_->resident.particleWallOccupiedContactArea()
            );
            forAll(data_->mapper->fluidPatchIds(), pairI)
            {
                const label patchI = data_->mapper->fluidPatchIds()[pairI];
                const polyPatch& patch =
                    data_->fluidMesh.boundaryMesh()[patchI];
                forAll(patch, faceI)
                {
                    const label globalFaceI = patch.start() + faceI;
                    const scalar faceArea = mag
                    (
                        data_->fluidMesh.Sf().boundaryField()[patchI][faceI]
                    );
                    const scalar occupiedArea =
                        occupiedContactArea[globalFaceI];

                    const scalar exposedFraction = max
                    (
                        scalar(0),
                        scalar(1) - min(occupiedArea/faceArea, scalar(1))
                    );
                    emissivity[patchI][faceI] *= exposedFraction;
                }
            }
            const ParticleRadiationMaterialSnapshot radiationSnapshot
            {
                particleRadiationEpsilonS,
                particleMassInCellKg,
                particleRadiationMeanTemperatureK,
                particleRadiationDiameterM,
                Tgas,
                emissivity,
                data_->particleHeatFactor,
                elapsedSinceRadiation
            };
            radiation.reset
            (
                new ParticleRadiationMaterialResult
                (
                    data_->particleRadiationCoupler->solve(radiationSnapshot)
                )
            );
            data_->particleRadiationAbsorptionCoefficient->primitiveFieldRef() =
                radiation().absorptionCoefficientInvM;
            data_->particleRadiationScatteringCoefficient->primitiveFieldRef() =
                radiation().scatteringCoefficientInvM;
            data_->particleRadiationExtinctionCoefficient->primitiveFieldRef() =
                radiation().extinctionCoefficientInvM;
            data_->particleRadiationAsymmetryFactor->primitiveFieldRef() =
                radiation().asymmetryFactor;
        }

        scalarField compactRadiationEnergy;
        if (radiationDue)
        {
            compactRadiationEnergy.setSize
            (
                data_->resident.configuredGasWallEnergyFaceIds().size()
            );
            label offset = 0;
            forAll(data_->mapper->fluidPatchIds(), pairI)
            {
                const label patchI = data_->mapper->fluidPatchIds()[pairI];
                const scalarField& energy =
                    radiation().solidWallRadiationEnergyJ[patchI];
                forAll(energy, faceI) compactRadiationEnergy[offset++] = energy[faceI];
            }
        }
        List<scalarField> solidEnergy;
        List<scalarField> gasHeatFlux;
        List<scalarField> depositedHeatFlux;
        List<scalarField> reflectedHeatFlux;
        data_->resident.previewInterfaceExchange
        (
            deltaTExchange, compactRadiationEnergy, solidEnergy,
            gasHeatFlux, depositedHeatFlux, reflectedHeatFlux
        );
        autoPtr<GpuSolidThermalCandidate> solidCandidate;
        autoPtr<GpuSolidThermalCandidate> auxiliaryCandidate;
        if (!data_->threeRegion)
        {
            GpuSolidThermalCandidate candidate =
                data_->solidCandidateSolver->solveTemporarySolidCandidate
                (
                    data_->Tsolid(), solidEnergy, deltaTExchange
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
            scalarField auxiliaryTemperature
            (
                coupledPatchOwnerTemperature
                (
                    data_->TauxiliarySolid(), auxiliaryPatchI
                )
            );
            scalarField primaryTemperature
            (
                coupledPatchOwnerTemperature(data_->Tsolid(), primaryPatchI)
            );
            scalarField previousEnergy(auxiliaryPatch.size(), scalar(0));

            for
            (
                label interfaceIteration = 0;
                interfaceIteration < max(label(1), data_->solidInterfaceMaxIterations);
                ++interfaceIteration
            )
            {
                const scalarField mappedPrimaryTemperature
                (
                    data_->solidInterfaceMapper->mapSolidScalarToFluid
                    (
                        0,
                        primaryTemperature
                    )
                );
                scalarField primaryKappa(primaryTemperature.size(), scalar(0));
                scalarField primaryDelta(primaryTemperature.size(), scalar(0));
                forAll(primaryTemperature, faceI)
                {
                    primaryKappa[faceI] =
                        data_->properties->kappa(primaryTemperature[faceI]);
                    primaryDelta[faceI] = primaryPatch.deltaCoeffs()[faceI];
                }
                const scalarField mappedPrimaryKappa
                (
                    data_->solidInterfaceMapper->mapSolidScalarToFluid
                    (
                        0,
                        primaryKappa
                    )
                );
                const scalarField mappedPrimaryDelta
                (
                    data_->solidInterfaceMapper->mapSolidScalarToFluid
                    (
                        0,
                        primaryDelta
                    )
                );
                scalarField auxiliaryEnergy(auxiliaryPatch.size(), scalar(0));
                forAll(auxiliaryEnergy, faceI)
                {
                    const scalar kAux =
                        data_->auxiliaryProperties->kappa(auxiliaryTemperature[faceI]);
                    const scalar dAux = auxiliaryPatch.deltaCoeffs()[faceI];
                    const scalar area = auxiliaryPatch.magSf()[faceI];
                    const scalar auxiliaryDenominator = kAux*dAux;
                    const scalar primaryDenominator =
                        mappedPrimaryKappa[faceI]*mappedPrimaryDelta[faceI];

                    const scalar resistance =
                        scalar(1)/auxiliaryDenominator
                      + scalar(1)/primaryDenominator;

                    auxiliaryEnergy[faceI] =
                        area*deltaTExchange
                       *(
                            mappedPrimaryTemperature[faceI]
                          - auxiliaryTemperature[faceI]
                        )
                       /resistance;
                }

                List<scalarField> auxiliaryEnergyList(1);
                auxiliaryEnergyList[0] = auxiliaryEnergy;
                List<scalarField> primaryEnergy(data_->mapper->size() + 1);
                forAll(solidEnergy, pairI)
                {
                    primaryEnergy[pairI] = solidEnergy[pairI];
                }
                scalarField negativeAuxiliaryEnergy(-auxiliaryEnergy);
                primaryEnergy.last() =
                    data_->solidInterfaceMapper->mapIntegratedFaceEnergyToSolid
                    (
                        0, negativeAuxiliaryEnergy
                    );

                GpuSolidThermalCandidate nextAuxiliary =
                    data_->auxiliaryCandidateSolver->solveTemporarySolidCandidate
                    (
                        data_->TauxiliarySolid(),
                        auxiliaryEnergyList,
                        deltaTExchange
                    );
                GpuSolidThermalCandidate nextPrimary =
                    data_->solidCandidateSolver->solveTemporarySolidCandidate
                    (
                        data_->Tsolid(), primaryEnergy, deltaTExchange
                    );

                const scalarField nextAuxiliaryTemperature
                (
                    coupledPatchOwnerTemperature
                    (
                        nextAuxiliary.temperature(), auxiliaryPatchI
                    )
                );
                const scalarField nextPrimaryTemperature
                (
                    coupledPatchOwnerTemperature
                    (
                        nextPrimary.temperature(), primaryPatchI
                    )
                );
                const scalar temperatureResidual = max
                (
                    relativeTemperatureChange
                    (
                        auxiliaryTemperature, nextAuxiliaryTemperature
                    ),
                    relativeTemperatureChange
                    (
                        primaryTemperature, nextPrimaryTemperature
                    )
                );
                scalar energyDifference = scalar(0);
                scalar energyScale = scalar(1);
                forAll(auxiliaryEnergy, faceI)
                {
                    energyDifference +=
                        mag(auxiliaryEnergy[faceI] - previousEnergy[faceI]);
                    energyScale += mag(auxiliaryEnergy[faceI]);
                }

                auxiliaryCandidate.reset
                (
                    new GpuSolidThermalCandidate(nextAuxiliary)
                );
                solidCandidate.reset(new GpuSolidThermalCandidate(nextPrimary));
                auxiliaryTemperature = nextAuxiliaryTemperature;
                primaryTemperature = nextPrimaryTemperature;
                previousEnergy = auxiliaryEnergy;
                if
                (
                    temperatureResidual
                 <= data_->solidInterfaceTemperatureTolerance
                 && energyDifference/energyScale
                 <= data_->solidInterfaceEnergyTolerance
                )
                {
                    break;
                }
                if
                (
                    interfaceIteration + 1
                 == max(label(1), data_->solidInterfaceMaxIterations)
                 && hostDiagnosticWarningsEnabled()
                )
                {
                    WarningInFunction
                        << "solid interface iteration budget exhausted; time="
                        << data_->fluidMesh.time().timeName()
                        << "; iterations=" << interfaceIteration + 1
                        << "; temperatureResidual=" << temperatureResidual
                        << "; temperatureLimit=" << data_->solidInterfaceTemperatureTolerance
                        << "; energyResidual=" << energyDifference/energyScale
                        << "; energyLimit=" << data_->solidInterfaceEnergyTolerance
                        << "; retaining computed candidates" << endl;
                }
            }

        }

        commitPhaseStarted = true;
        if (radiationDue)
        {
            data_->resident.applyParticleRadiationEnergy
            (
                data_->fluidMesh,
                radiation().deltaParticleRadiationEnergyJ,
                particleMassInCellKg,
                particleRadiationMeanTemperatureK,
                data_->particleHeatFactor
            );
        }
        data_->resident.resetGasWallEnergy();
        data_->resident.resetParticleWallHeatLedgers();
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
        if (radiationDue)
        {
            data_->particleRadiationWallHeatFlux->primitiveFieldRef() =
                scalar(0);
            forAll
            (
                data_->particleRadiationWallHeatFlux->boundaryField(), patchI
            )
            {
                data_->particleRadiationWallHeatFlux->boundaryFieldRef()
                    [patchI] = scalar(0);
            }
            forAll(data_->mapper->fluidPatchIds(), pairI)
            {
                const label patchI = data_->mapper->fluidPatchIds()[pairI];
                auto& flux =
                    data_->particleRadiationWallHeatFlux->boundaryFieldRef()
                    [patchI];
                const scalarField& energy =
                    radiation().solidWallRadiationEnergyJ[patchI];
                const scalarField& area =
                    data_->fluidMesh.magSf().boundaryField()[patchI];
                forAll(flux, faceI)
                {
                    flux[faceI] =
                        energy[faceI]/(area[faceI]*elapsedSinceRadiation);
                }
            }
        }
        data_->solidCandidateSolver->publishSolidCandidate
        (
            solidCandidate(), data_->Tsolid()
        );
        if (data_->threeRegion)
        {
            data_->auxiliaryCandidateSolver->publishSolidCandidate
            (
                auxiliaryCandidate(), data_->TauxiliarySolid()
            );
        }
        data_->mapSolidWallTemperatureToFluid(Tgas);
        if (radiationDue)
        {
            data_->resident.refreshParticleEnthalpyAfterRadiation();
        }

        if (radiationDue || runTime.writeTime())
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
        result.particleCount = data_->resident.particleCount();
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
