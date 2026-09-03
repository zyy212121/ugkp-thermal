#include "GpuSolidThermalCoupler.H"
#include "ThermalCouplingSchedule.H"

#include "fixedGradientFvPatchFields.H"
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
        previousResidual.size() != currentResidual.size()
     || !finiteScalar(previousRelaxation)
     || !finiteScalar(minimumRelaxation)
     || !finiteScalar(maximumRelaxation)
     || minimumRelaxation <= scalar(0)
     || maximumRelaxation < minimumRelaxation
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
    const GpuSolidThermalControls& controls
)
:
    mesh_(mesh),
    properties_(properties),
    coupledPatchIds_(coupledPatchIds),
    controls_(controls)
{
    if
    (
        controls_.nonlinearMaxIterations <= 0
     || !finiteScalar(controls_.nonlinearRelativeTolerance)
     || controls_.nonlinearRelativeTolerance < scalar(0)
     || !finiteScalar(controls_.energyRelativeTolerance)
     || controls_.energyRelativeTolerance < scalar(0)
     || !finiteScalar(controls_.energyAbsoluteTolerance)
     || controls_.energyAbsoluteTolerance < scalar(0)
     || !finiteScalar(controls_.nonlinearInitialRelaxation)
     || !finiteScalar(controls_.nonlinearMinimumRelaxation)
     || !finiteScalar(controls_.nonlinearMaximumRelaxation)
     || controls_.nonlinearMinimumRelaxation <= scalar(0)
     || controls_.nonlinearMaximumRelaxation
        < controls_.nonlinearMinimumRelaxation
     || controls_.nonlinearMaximumRelaxation > scalar(1)
     || controls_.nonlinearInitialRelaxation
        < controls_.nonlinearMinimumRelaxation
     || controls_.nonlinearInitialRelaxation
        > controls_.nonlinearMaximumRelaxation
     || coupledPatchIds_.empty()
    )
    {
        solidFailure("invalid solid thermal controls or empty coupled patch list");
    }
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
        &registeredTemperature.mesh() != &mesh_
     || interfaceEnergyJ.size() != coupledPatchIds_.size()
     || !finiteScalar(deltaTExchange)
     || deltaTExchange <= scalar(0)
    )
    {
        solidFailure("invalid solid candidate mesh, energy list, or exchange time");
    }

    scalar totalInterfaceEnergyJ = scalar(0);
    forAll(coupledPatchIds_, coupledI)
    {
        const label patchI = coupledPatchIds_[coupledI];
        if
        (
            patchI < 0
         || patchI >= mesh_.boundary().size()
         || interfaceEnergyJ[coupledI].size() != mesh_.boundary()[patchI].size()
         || !isA<fixedGradientFvPatchScalarField>
            (registeredTemperature.boundaryField()[patchI])
        )
        {
            solidFailure("coupled solid patch must be valid fixedGradient with matching energy");
        }
        forAll(interfaceEnergyJ[coupledI], faceI)
        {
            const scalar energyJ = interfaceEnergyJ[coupledI][faceI];
            if (!finiteScalar(energyJ))
            {
                solidFailure("non-finite solid interface energy");
            }
            totalInterfaceEnergyJ += energyJ;
        }
    }
    if (!finiteScalar(totalInterfaceEnergyJ))
    {
        solidFailure("non-finite integrated solid interface energy");
    }

    autoPtr<volScalarField> candidate
    (
        new volScalarField
        (
            IOobject
            (
                "TsolidCandidate",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            registeredTemperature
        )
    );
    const scalarField oldInternal(registeredTemperature.primitiveField());
    scalarField previous(candidate->primitiveField());
    scalarField previousFixedPointResidual(previous.size(), scalar(0));
    bool havePreviousFixedPointResidual = false;
    scalar nonlinearRelaxation = controls_.nonlinearInitialRelaxation;
    const dimensionedScalar dt("deltaTExchange", dimTime, deltaTExchange);
    const dimensionSet rhoCpDimensions = dimEnergy/dimVolume/dimTemperature;
    const dimensionSet kappaDimensions = dimPower/dimLength/dimTemperature;
    scalar finalResidual = std::numeric_limits<scalar>::quiet_NaN();
    scalar finalTemperatureResidual =
        std::numeric_limits<scalar>::quiet_NaN();
    scalar finalEnergyRelativeResidual =
        std::numeric_limits<scalar>::quiet_NaN();
    scalar finalRawFixedPointResidual =
        std::numeric_limits<scalar>::quiet_NaN();

    mesh_.schemes().setFluxRequired(candidate->name());
    for
    (
        label iteration = 1;
        iteration <= controls_.nonlinearMaxIterations;
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
        forAll(candidate->primitiveField(), cellI)
        {
            Csec[cellI] = properties_.Csec((*candidate)[cellI], oldInternal[cellI]);
            kappa[cellI] = properties_.kappa((*candidate)[cellI]);
        }
        forAll(mesh_.boundary(), patchI)
        {
            const bool coupledPatch = isCoupledPatch(coupledPatchIds_, patchI);
            const scalarField patchOwnerTemperature
            (
                candidate->boundaryField()[patchI].patchInternalField()
            );
            forAll(mesh_.boundary()[patchI], faceI)
            {
                                                                          
                                                                             
                                                                               
                                                                               
                                                                            
                                                                            
                                                        
                const scalar patchTemperature = coupledPatch
                  ? patchOwnerTemperature[faceI]
                  : candidate->boundaryField()[patchI][faceI];
                Csec.boundaryFieldRef()[patchI][faceI] =
                    properties_.Csec(patchTemperature, patchTemperature);
                kappa.boundaryFieldRef()[patchI][faceI] =
                    properties_.kappa(patchTemperature);
            }
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
                if (!finiteScalar(area) || area <= scalar(0))
                {
                    solidFailure("invalid coupled solid face area");
                }
                interfacePatch.gradient()[faceI] =
                    interfaceEnergyJ[coupledI][faceI]
                   /(area*deltaTExchange*kappa.boundaryField()[patchI][faceI]);
            }
        }
        candidate->correctBoundaryConditions();

        volScalarField oldTemperature
        (
            IOobject
            (
                "TsolidOldCandidate", mesh_.time().timeName(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            registeredTemperature
        );
        fvScalarMatrix equation
        (
            fvm::Sp(Csec/dt, *candidate)
          - fvm::laplacian(kappa, *candidate)
         == (Csec/dt)*oldTemperature
        );
        solverPerformance linearPerformance;
        {
            ScalarSolverLogSilencer silenceSolverLog;
            linearPerformance = equation.solve();
        }
        if
        (
            !linearPerformance.converged()
         || linearPerformance.singular()
         || !finiteScalar(linearPerformance.finalResidual())
        )
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "solid linear solve did not converge"
                << "; region=" << mesh_.name()
                << "; nonlinearIteration=" << iteration
                << "; solver=" << linearPerformance.solverName()
                << "; iterations=" << linearPerformance.nIterations()
                << "; initialResidual="
                << linearPerformance.initialResidual()
                << "; finalResidual=" << linearPerformance.finalResidual()
                << "; singular=" << linearPerformance.singular();
            solidFailure(reason.str());
        }
        if (controls_.nonlinearAitken)
        {
            scalarField currentFixedPointResidual
            (
                candidate->primitiveField() - previous
            );
            finalRawFixedPointResidual =
                relativeTemperatureChange
                (
                    previous,
                    candidate->primitiveField()
                );
            if (havePreviousFixedPointResidual)
            {
                nonlinearRelaxation = clampedAitkenRelaxation
                (
                    previousFixedPointResidual,
                    currentFixedPointResidual,
                    nonlinearRelaxation,
                    controls_.nonlinearMinimumRelaxation,
                    controls_.nonlinearMaximumRelaxation
                );
            }
            candidate->primitiveFieldRef() =
                previous + nonlinearRelaxation*currentFixedPointResidual;
            candidate->correctBoundaryConditions();
            previousFixedPointResidual = currentFixedPointResidual;
            havePreviousFixedPointResidual = true;
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
                if (!finiteScalar(faceFluxW))
                {
                    solidFailure("solid matrix returned non-finite boundary flux");
                }
                externalBoundaryEnergyIntoSolidJ -= deltaTExchange*faceFluxW;
            }
        }

        scalar enthalpyChangeJ = scalar(0);
        scalar minimumCandidateTemperature = GREAT;
        label minimumCandidateCell = -1;
        forAll(candidate->primitiveField(), cellI)
        {
            if (!finiteScalar((*candidate)[cellI]))
            {
                solidFailure("solid candidate solve produced non-finite temperature");
            }
            if ((*candidate)[cellI] < minimumCandidateTemperature)
            {
                minimumCandidateTemperature = (*candidate)[cellI];
                minimumCandidateCell = cellI;
            }
            enthalpyChangeJ += mesh_.V()[cellI]
               *properties_.deltaHv((*candidate)[cellI], oldInternal[cellI]);
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
        finalTemperatureResidual =
            relativeTemperatureChange(previous, candidate->primitiveField());
        finalEnergyRelativeResidual = mag(finalResidual)/energyScale;
        const bool temperatureConverged =
            finalTemperatureResidual <= controls_.nonlinearRelativeTolerance;
        const bool energyConverged =
            mag(finalResidual)
         <= controls_.energyAbsoluteTolerance
          + controls_.energyRelativeTolerance*energyScale;
        if (temperatureConverged && energyConverged)
        {
            if (minimumCandidateTemperature < scalar(1))
            {
                std::ostringstream reason;
                reason.precision(17);
                reason
                    << "nonphysical converged solid candidate"
                    << "; region=" << mesh_.name()
                    << "; nonlinearIterations=" << iteration
                    << "; cell=" << minimumCandidateCell
                    << "; temperatureK=" << minimumCandidateTemperature;
                solidFailure(reason.str());
            }
            return GpuSolidThermalCandidate(candidate, finalResidual, iteration);
        }
        previous = candidate->primitiveField();
    }

    std::ostringstream reason;
    reason.precision(17);
    scalar minimumCandidateTemperature = GREAT;
    scalar maximumCandidateTemperature = -GREAT;
    forAll(candidate->primitiveField(), cellI)
    {
        minimumCandidateTemperature = min
        (
            minimumCandidateTemperature,
            (*candidate)[cellI]
        );
        maximumCandidateTemperature = max
        (
            maximumCandidateTemperature,
            (*candidate)[cellI]
        );
    }
    reason
        << "solid candidate nonlinear solve did not converge"
        << "; region=" << mesh_.name()
        << "; iterations=" << controls_.nonlinearMaxIterations
        << "; temperatureResidual=" << finalTemperatureResidual
        << "; energyRelativeResidual=" << finalEnergyRelativeResidual
        << "; energyResidualJ=" << finalResidual
        << "; rawFixedPointResidual=" << finalRawFixedPointResidual
        << "; relaxation=" << nonlinearRelaxation
        << "; Tmin=" << minimumCandidateTemperature
        << "; Tmax=" << maximumCandidateTemperature;
    solidFailure(reason.str());
}

void GpuSolidThermalCandidateSolver::publishSolidCandidate
(
    const GpuSolidThermalCandidate& candidate,
    volScalarField& registeredTemperature
) const
{
    if
    (
        &registeredTemperature.mesh() != &mesh_
     || candidate.temperature().size() != registeredTemperature.size()
    )
    {
        solidFailure("cannot publish a solid candidate for a different mesh");
    }
    registeredTemperature.primitiveFieldRef() =
        candidate.temperature().primitiveField();
    forAll(registeredTemperature.boundaryField(), patchI)
    {
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
    if
    (
        exactLookup<word>(entry, "firstRegion") != firstRegion
     || exactLookup<word>(entry, "secondRegion") != secondRegion
     || exactLookup<word>(entry, "mapping") != "oneToOneConformal"
    )
    {
        solidFailure("solid-solid interface region order or mapping is invalid");
    }
    SolidPatchPair result;
    result.fluidPatch = exactLookup<word>(entry, "firstPatch");
    result.solidPatch = exactLookup<word>(entry, "secondPatch");
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

scalarField flattenPatchFields(const List<scalarField>& patches)
{
    label count = 0;
    forAll(patches, patchI)
    {
        count += patches[patchI].size();
    }
    scalarField result(count);
    label offset = 0;
    forAll(patches, patchI)
    {
        forAll(patches[patchI], faceI)
        {
            std::memcpy
            (
                &result[offset++],
                &patches[patchI][faceI],
                sizeof(scalar)
            );
        }
    }
    return result;
}

void requireBitwiseGasWallPreviewMatch
(
    const List<scalarField>& preview,
    const scalarField& consumed
)
{
    const scalarField flatPreview(flattenPatchFields(preview));
    if
    (
        flatPreview.size() != consumed.size()
     || (flatPreview.size() > 0
      && std::memcmp
        (
            flatPreview.begin(), consumed.begin(),
            static_cast<std::size_t>(flatPreview.size())*sizeof(scalar)
        ) != 0)
    )
    {
        solidFailure("consumed gas-wall ledger differs byte-for-byte from preview");
    }
}

void requireBitwiseParticleWallPreviewMatch
(
    const scalarField& preview,
    const scalarField& consumed
)
{
    if
    (
        preview.size() != consumed.size()
     || (preview.size() > 0
      && std::memcmp
        (
            preview.begin(), consumed.begin(),
            static_cast<std::size_t>(preview.size())*sizeof(scalar)
        ) != 0)
    )
    {
        solidFailure
        (
            "consumed particle-wall ledger differs byte-for-byte from preview"
        );
    }
}

void writeManifestStateEntries
(
    std::ostream& os,
    const ThermalExchangeRestartState& state
)
{
    os.setf(std::ios::scientific);
    os.precision(std::numeric_limits<scalar>::max_digits10);
    os << "formatVersion " << state.formatVersion << ";\n"
       << "initialState " << (state.initialState ? "true" : "false") << ";\n"
       << "exchangeSequence \"" << state.exchangeSequence << "\";\n"
       << "completedTimeIndex " << state.completedTimeIndex << ";\n"
       << "completedSimulationTimeS " << state.completedSimulationTimeS << ";\n";
    if (state.formatVersion >= 3)
    {
        os << "completedRadiationSimulationTimeS "
           << state.completedRadiationSimulationTimeS << ";\n";
    }
    os << "previousExchangeSimulationTimeS "
       << state.previousExchangeSimulationTimeS << ";\n"
       << "fluidMeshTopologySha1 \"" << state.fluidMeshTopologySha1 << "\";\n"
       << "solidMeshTopologySha1 \"" << state.solidMeshTopologySha1 << "\";\n";
    if (state.formatVersion >= 2)
    {
        os << "auxiliarySolidMeshTopologySha1 \""
           << state.auxiliarySolidMeshTopologySha1 << "\";\n";
    }
    os << "couplingConfigurationSha1 \"" << state.couplingConfigurationSha1 << "\";\n"
       << "wallTemperatureSha1UsedForCompletedInterval \""
       << state.wallTemperatureSha1UsedForCompletedInterval << "\";\n"
       << "newlyUploadedWallTemperatureSha1 \""
       << state.newlyUploadedWallTemperatureSha1 << "\";\n"
       << "gasWallLedgerConsumed "
       << (state.gasWallLedgerConsumed ? "true" : "false") << ";\n";
    if (state.formatVersion >= 2)
    {
        os << "particleWallLedgerConsumed "
           << (state.particleWallLedgerConsumed ? "true" : "false")
           << ";\n";
    }
    os << "particleRadiationApplied "
       << (state.particleRadiationApplied ? "true" : "false") << ";\n"
       << "solidStateUpdated "
       << (state.solidStateUpdated ? "true" : "false") << ";\n"
       << "wallTemperatureUploaded "
       << (state.wallTemperatureUploaded ? "true" : "false") << ";\n"
       << "particleMomentsRebuilt "
       << (state.particleMomentsRebuilt ? "true" : "false") << ";\n"
       << "particleContactEnergyJ " << state.particleContactEnergyJ << ";\n";
}

void writeArtifact
(
    std::ostream& os,
    const word& role,
    const fileName& relativePath,
    const word& sha1,
    const label count
)
{
    os << "    (" << role << " \"" << relativePath << "\" \""
       << sha1 << "\" " << count << ")\n";
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
    word currentWallTemperatureSha1;
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
        currentWallTemperatureSha1(),
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
        const label formatVersion = exactLookup<label>(coupling, "formatVersion");
        if (formatVersion != 1 && formatVersion != 2)
        {
            solidFailure("unsupported solid thermal formatVersion");
        }
        threeRegion = formatVersion == 2;
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
            !std::isfinite(solidContactCouplingInterval)
         || solidContactCouplingInterval <= scalar(0)
         || radiationCouplingFrequency <= 0
        )
        {
            solidFailure("thermal coupling interval and radiation frequency must be strictly positive");
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
            if
            (
                solidInterfaceMaxIterations <= 0
             || solidInterfaceTemperatureTolerance < scalar(0)
             || solidInterfaceEnergyTolerance < scalar(0)
            )
            {
                solidFailure("invalid solid-solid interface iteration controls");
            }
            if (exactLookup<word>(coupling, "fluidRegion") != fluidMesh.name())
            {
                solidFailure("configured fluidRegion does not match the active fluid mesh");
            }
            solidRegion = exactLookup<word>(coupling, "primarySolidRegion");
            PtrList<dictionary> solidInterfaces
            (
                coupling.lookup("solidSolidInterfaces", false, false)
            );
            if (solidInterfaces.size() != 1)
            {
                solidFailure("v2 requires exactly one solid-solid interface");
            }
            auxiliarySolidRegion =
                exactLookup<word>(solidInterfaces[0], "firstRegion");
            if
            (
                exactLookup<word>(solidInterfaces[0], "secondRegion")
             != solidRegion
            )
            {
                solidFailure("primarySolidRegion must be the second solid-solid region");
            }
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
        const label particleCapacity =
            exactLookup<label>(particleContract, "particleCapacity");
        if
        (
            resident.rhoSolid() != rhoSolid
         || resident.particleHeatFactorJkgK() != particleHeatFactor
         || resident.particleTemperatureMin() != TpMin
         || resident.particleTemperatureMax() != TpMax
         || resident.particleCapacity() != particleCapacity
        )
        {
            solidFailure("resolvedParticleContract does not exactly match resident initialisation");
        }

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
        if (threeRegion)
        {
            const label oldSize = primaryCoupledPatchIds.size();
            primaryCoupledPatchIds.setSize(oldSize + 1);
            primaryCoupledPatchIds[oldSize] =
                solidInterfaceMapper->solidPatchIds()[0];
        }
        solidCandidateSolver.reset
        (
            new GpuSolidThermalCandidateSolver
            (
                solidMesh(), properties(), primaryCoupledPatchIds,
                readSolidControls(coupling)
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
                    readSolidControls(coupling)
                )
            );
        }
        resident.configureGasWallEnergyLedger(fluidMesh, mapper->fluidPatchIds());
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
            if (restartState.completedSimulationTimeS != runTime.value())
            {
                solidFailure
                (
                    "lagged CHT restart is missing its pending wall-energy ledger"
                );
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
        if
        (
            pending.faceIds.size() != configuredFaces.size()
         ||
            (
                !configuredFaces.empty()
             && std::memcmp
                (
                    pending.faceIds.begin(),
                    configuredFaces.begin(),
                    static_cast<std::size_t>(configuredFaces.size())
                   *sizeof(label)
                ) != 0
            )
        )
        {
            solidFailure
            (
                "pending wall-energy ledger faces do not exactly match the "
                "configured CHT coupled faces"
            );
        }
        resident.restorePendingWallEnergyLedgers
        (
            pending.gasConvectiveEnergyJ,
            pending.particleDepositedEnergyJ,
            pending.particleReflectedEnergyJ
        );
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
           << "fluidMeshTopologySha1 \""
           << pending.fluidMeshTopologySha1 << "\";\n"
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
            if
            (
                !finiteScalar(gas)
             || !finiteScalar(deposited)
             || !finiteScalar(reflected)
            )
            {
                solidFailure("non-finite pending wall-energy ledger value");
            }
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
        forAll(mapper->fluidPatchIds(), pairI)
        {
            const label fluidPatchI = mapper->fluidPatchIds()[pairI];
            const label solidPatchI = mapper->solidPatchIds()[pairI];
            if (!Tgas.boundaryField()[fluidPatchI].fixesValue())
            {
                solidFailure("coupled fluid T patch must be fixed-value");
            }
            const scalarField solidWallTemperature
            (
                coupledPatchOwnerTemperature(Tsolid(), solidPatchI)
            );
            const scalarField mapped = mapper->mapSolidScalarToFluid
            (
                pairI, solidWallTemperature
            );
            Tgas.boundaryFieldRef()[fluidPatchI] == mapped;
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
            const label solidPatchI = mapper->solidPatchIds()[pairI];
            const scalarField solidWallTemperature
            (
                coupledPatchOwnerTemperature(Tsolid(), solidPatchI)
            );
            scalarField solidWallEffusivity(solidWallTemperature.size());
            forAll(solidWallEffusivity, faceI)
            {
                const scalar Tw = solidWallTemperature[faceI];
                const scalar rhoWall = properties->rho(Tw);
                const scalar cpWall = properties->Cp(Tw);
                const scalar kappaWall = properties->kappa(Tw);
                if
                (
                    !finiteScalar(rhoWall) || !finiteScalar(cpWall)
                 || !finiteScalar(kappaWall) || rhoWall <= 0
                 || cpWall <= 0 || kappaWall <= 0
                )
                {
                    solidFailure
                    (
                        "solidRegionProperties rho/Cp/kappa must evaluate to "
                        "finite positive values on every stuck CHT wall face"
                    );
                }
                solidWallEffusivity[faceI] = sqrt
                (
                    rhoWall*cpWall*kappaWall
                );
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
                coupledPatchOwnerTemperature(Tsolid(), solidPatchI)
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

    std::pair<fileName, fileName> writeManifestTemporary
    (
        const ThermalExchangeRestartState& state,
        const GpuThermalRestartMirrorMetadata& mirrors,
        const PendingWallEnergyLedgerSnapshot& pendingWallEnergy
    )
    {
        const fileName timePath(runTime.timePath());
        const fileName statePath(timePath/ThermalRestartPreflight::stateObjectName());
        const fileName manifestFinal
        (
            timePath/ThermalRestartPreflight::manifestObjectName()
        );
        const fileName manifestTemp(manifestFinal + ".tmp");
        std::ofstream os(manifestTemp.c_str(), std::ios::out | std::ios::trunc);
        if (!os)
        {
            solidFailure("cannot create temporary thermal restart manifest");
        }
        os << "FoamFile\n{\nversion 2.0;\nformat ascii;\nclass dictionary;\n"
           << "object thermalExchangeManifest;\n}\n\n";
        writeManifestStateEntries(os, state);
        os << "thermalExchangeStateFile \"thermalExchangeState\";\n"
           << "thermalExchangeStateSha1 \""
           << ThermalRestartPreflight::fileSha1(statePath) << "\";\n"
           << "solidFieldName T;\nartifacts\n(\n";
        writeArtifact(os, "particleRestart", mirrors.particleFile,
            mirrors.particleSha1, mirrors.particleCount);
        writeArtifact(os, "epsGPrevRestart", mirrors.epsGPrevFile,
            mirrors.epsGPrevSha1, mirrors.epsGPrevCount);
        writeArtifact(os, "sourceResidualRestart", mirrors.sourceResidualFile,
            mirrors.sourceResidualSha1, mirrors.sourceResidualCount);
        const fileName pendingWallEnergyFile
        (
            ThermalRestartPreflight::pendingWallEnergyObjectName()
        );
        writeArtifact
        (
            os,
            "pendingWallEnergyLedgers",
            pendingWallEnergyFile,
            ThermalRestartPreflight::fileSha1
            (
                timePath/pendingWallEnergyFile
            ),
            pendingWallEnergy.faceIds.size()
        );
        const fileName solidRelative(solidRegion/"T");
        writeArtifact(os, "solidTemperature", solidRelative,
            ThermalRestartPreflight::fileSha1(timePath/solidRelative), -1);
        if (threeRegion)
        {
            const fileName auxiliaryRelative(auxiliarySolidRegion/"T");
            writeArtifact
            (
                os, "auxiliarySolidTemperature", auxiliaryRelative,
                ThermalRestartPreflight::fileSha1(timePath/auxiliaryRelative), -1
            );
        }
        const word roles[] =
        {
            "fluidRho", "fluidRhoU", "fluidRhoE", "fluidU", "fluidP",
            "fluidT", "fluidEpsilonS", "fluidRhoUs", "fluidRhoEs",
            "fluidRhoDs", "fluidRhoHp", "fluidUs", "fluidTheta",
            "fluidTp", "fluidDMeanCell"
        };
        const word files[] =
        {
            "rho", "rhoU", "rhoE", "U", "p", "T", "epsilonS", "rhoUs",
            "rhoEs", "rhoDs", "rhoHp", "Us", "theta", "Tp", "dMeanCell"
        };
        const fileName fluidPrefix =
            fluidMesh.name() == fvMesh::defaultRegion
          ? fileName::null
          : fileName(fluidMesh.name());
        for (label i = 0; i < 15; ++i)
        {
            const fileName relative = fluidPrefix.empty()
              ? fileName(files[i])
              : fluidPrefix/files[i];
            writeArtifact(os, roles[i], relative,
                ThermalRestartPreflight::fileSha1(timePath/relative), -1);
        }
        os << ");\n";
        os.flush();
        os.close();
        if (!os)
        {
            solidFailure("failed completing temporary thermal restart manifest");
        }
        return std::make_pair(manifestTemp, manifestFinal);
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
    if (data_->wallInitialised)
    {
        solidFailure("solid wall temperature startup upload may occur only once");
    }
    data_->mapSolidWallTemperatureToFluid(Tgas);
    data_->currentWallTemperatureSha1 =
        ThermalRestartPreflight::canonicalWallTemperatureSha1
        (
            Tgas,
            data_->mapper->fluidPatchIds()
        );
    if
    (
        !data_->writeState.committedState().initialState
     && data_->currentWallTemperatureSha1
        != data_->writeState.committedState().newlyUploadedWallTemperatureSha1
    )
    {
        solidFailure("reconstructed wall temperature checksum differs from restart state");
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
    if
    (
        &runTime != &data_->runTime
     || !data_->wallInitialised
    )
    {
        solidFailure("thermal exchange requires the owned post-advance event");
    }
    const scalar elapsedSinceCoupling =
        runTime.value()
      - data_->writeState.committedState().completedSimulationTimeS;
    if (!std::isfinite(elapsedSinceCoupling))
    {
        solidFailure("elapsed coupling time is nonfinite");
    }
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
    if
    (
        !std::isfinite(elapsedSinceRadiation)
     || elapsedSinceRadiation < scalar(0)
    )
    {
        data_->writeState.cancelPreparedExchange();
        solidFailure("elapsed radiation coupling time is invalid");
    }
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
                    if
                    (
                        !std::isfinite(faceArea)
                     || !std::isfinite(occupiedArea)
                     || faceArea <= scalar(0)
                     || occupiedArea < scalar(0)
                     || occupiedArea > faceArea*(scalar(1) + scalar(1e-12))
                    )
                    {
                        solidFailure
                        (
                            "invalid instantaneous particle-wall radiation "
                            "occupied area"
                        );
                    }
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

        const List<scalarField> gasPreview =
            data_->resident.peekGasWallEnergy();
        scalarField particleDepositedPreview;
        scalarField particleReflectedPreview;
        data_->resident.peekParticleWallHeatLedgers
        (
            particleDepositedPreview, particleReflectedPreview
        );
        scalarField particleContactPreview(particleDepositedPreview);
        particleContactPreview += particleReflectedPreview;
        List<scalarField> solidEnergy(data_->mapper->size());
        scalar particleContactEnergyJ = scalar(0);
        forAll(particleContactPreview, faceI)
        {
            if (!finiteScalar(particleContactPreview[faceI]))
            {
                solidFailure("non-finite particle-wall contact energy");
            }
            particleContactEnergyJ += particleContactPreview[faceI];
        }
        forAll(solidEnergy, pairI)
        {
            const label fluidPatchI = data_->mapper->fluidPatchIds()[pairI];
            const polyPatch& fluidPatch =
                data_->fluidMesh.boundaryMesh()[fluidPatchI];
            scalarField combined(gasPreview[pairI]);
            forAll(combined, faceI)
            {
                if (radiationDue)
                {
                    combined[faceI] +=
                        radiation().solidWallRadiationEnergyJ
                        [fluidPatchI][faceI];
                }
                if (!particleContactPreview.empty())
                {
                    combined[faceI] += particleContactPreview
                    [
                        fluidPatch.start() + faceI
                    ];
                }
            }
            solidEnergy[pairI] =
                data_->mapper->mapIntegratedFaceEnergyToSolid(pairI, combined);
        }
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
            bool interfaceConverged = false;

            for
            (
                label interfaceIteration = 0;
                interfaceIteration < data_->solidInterfaceMaxIterations;
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
                    const scalar resistance =
                        scalar(1)/(kAux*dAux)
                      + scalar(1)
                       /(mappedPrimaryKappa[faceI]*mappedPrimaryDelta[faceI]);
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
                    interfaceConverged = true;
                    break;
                }
            }
            if (!interfaceConverged)
            {
                solidFailure("solid-solid interface iteration did not converge");
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
        scalarField consumedGasWallEnergy;
        data_->resident.downloadAndResetGasWallEnergy(consumedGasWallEnergy);
        requireBitwiseGasWallPreviewMatch(gasPreview, consumedGasWallEnergy);
        scalarField consumedParticleDepositedEnergy;
        scalarField consumedParticleReflectedEnergy;
        data_->resident.downloadAndResetParticleWallHeatLedgers
        (
            consumedParticleDepositedEnergy,
            consumedParticleReflectedEnergy
        );
        requireBitwiseParticleWallPreviewMatch
        (
            particleDepositedPreview, consumedParticleDepositedEnergy
        );
        requireBitwiseParticleWallPreviewMatch
        (
            particleReflectedPreview, consumedParticleReflectedEnergy
        );
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
            const scalarField& energy = gasPreview[pairI];
            const scalarField& area =
                data_->fluidMesh.magSf().boundaryField()[patchI];
            forAll(flux, faceI)
            {
                flux[faceI] = energy[faceI]/(area[faceI]*deltaTExchange);
            }
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
            const polyPatch& patch = data_->fluidMesh.boundaryMesh()[patchI];
            auto& depositedFlux =
                data_->particleStuckWallHeatFlux->boundaryFieldRef()[patchI];
            auto& reflectedFlux =
                data_->particleReflectedWallHeatFlux->boundaryFieldRef()[patchI];
            const scalarField& area =
                data_->fluidMesh.magSf().boundaryField()[patchI];
            forAll(depositedFlux, faceI)
            {
                const label globalFaceI = patch.start() + faceI;
                if (!particleDepositedPreview.empty())
                {
                    depositedFlux[faceI] = particleDepositedPreview[globalFaceI]
                        /(area[faceI]*deltaTExchange);
                    reflectedFlux[faceI] = particleReflectedPreview[globalFaceI]
                        /(area[faceI]*deltaTExchange);
                }
            }
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
        const word newWallSha1 =
            ThermalRestartPreflight::canonicalWallTemperatureSha1
            (
                Tgas,
                data_->mapper->fluidPatchIds()
            );
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
        completed.wallTemperatureSha1UsedForCompletedInterval =
            data_->currentWallTemperatureSha1;
        completed.newlyUploadedWallTemperatureSha1 = newWallSha1;
        completed.gasWallLedgerConsumed = true;
        completed.particleWallLedgerConsumed = true;
        completed.particleRadiationApplied = true;
        completed.solidStateUpdated = true;
        completed.wallTemperatureUploaded = true;
        completed.particleMomentsRebuilt = true;
        completed.particleContactEnergyJ = particleContactEnergyJ;
        ThermalRestartPreflight::validateStateSemantics(completed);
        data_->writeState.commitPreparedExchange(completed);
        data_->currentWallTemperatureSha1 = newWallSha1;
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
        &runTime != &data_->runTime
     ||
        (
            !runTime.writeTime()
         &&
            (
                !thermalCouplingWrite
             || data_->writeState.committedState().completedSimulationTimeS
                != runTime.value()
            )
        )
     || !data_->wallInitialised
     || data_->writeState.hasPendingExchange()
    )
    {
        solidFailure("thermal persistence requires a completed owned writeTime event");
    }

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
    const bool fieldWriteSucceeded =
        thermalCouplingWrite && !runTime.writeTime()
      ? runTime.writeNow()
      : runTime.write();
    if (!fieldWriteSucceeded)
    {
        solidFailure("ordinary OpenFOAM field write failed during thermal persistence");
    }
    ThermalRestartPreflight::commitManifestAtomically
    (
        preparedPendingWallEnergy.first,
        preparedPendingWallEnergy.second
    );
    ThermalRestartPreflight::validateFiniteRestartFields
    (
        data_->fluidMesh, data_->solidMesh()
    );
    if (data_->threeRegion)
    {
        ThermalRestartPreflight::validateFiniteRestartFields
        (
            data_->fluidMesh, data_->auxiliarySolidMesh()
        );
    }
    const std::pair<fileName, fileName> preparedManifest =
        data_->writeManifestTemporary
        (
            completed,
            mirrors,
            pendingWallEnergy
        );
    ThermalRestartPreflight::commitManifestAtomically
    (
        preparedManifest.first,
        preparedManifest.second
    );
    return mirrors.particleCount;
}

}                        
}                  

#endif
