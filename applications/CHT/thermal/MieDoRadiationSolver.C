#include "MieDoRadiationSolver.H"

#include "MieDoRadiationMath.H"
#include "gpu/DiscreteOrdinateRadiation.H"
#include "ThermalHostDiagnostics.H"
#include "Pstream.H"
#include "surfaceFields.H"
#include "wallPolyPatch.H"

#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Foam
{
namespace gpuThermal
{
namespace
{
struct PatchRadiationControls
{
    word type;
    bool hasSpecifiedTemperature;
    scalar specifiedTemperatureK;
    scalar absorptivity;
    scalar fixedIntensity;
};

[[noreturn]] void solveFailure
(
    const std::string& reason,
    const label iteration = 0,
    const scalar residual = std::numeric_limits<scalar>::quiet_NaN(),
    const label cell = -1,
    const label patch = -1,
    const label face = -1
)
{
    std::ostringstream message;
    message.precision(17);
    message
        << reason
        << "; iteration=" << iteration
        << "; residual=" << residual
        << "; cell=" << cell
        << "; patch=" << patch
        << "; face=" << face;
    throw MieDoSolveError(message.str());
}

inline bool isSpecularType(const word& type)
{
    return
        type == "symmetry"
     || type == "symmetryPlane"
     || type == "specular"
     || type == "specularReflection";
}

inline bool isDiffuseType(const word& type)
{
    return type == "diffuseGrey" || type == "greyDiffuse";
}

inline bool isSupportedRadiationType(const word& type)
{
    return
        isDiffuseType(type)
     || type == "blackBody"
     || type == "black"
     || type == "zeroIncoming"
     || type == "zero"
     || type == "fixedValue"
     || type == "fixedI"
     || isSpecularType(type);
}

inline bool isSupportedSerialPatch(const word& patchType)
{
    return
        patchType == "wall"
     || patchType == "patch"
     || patchType == "symmetry"
     || patchType == "symmetryPlane"
     || patchType == "wedge";
}

bool readScalarIfPresent
(
    const dictionary& dict,
    const word& name,
    scalar& value
)
{
    if (!dict.found(name))
    {
        return false;
    }
    value = readScalar(dict.lookup(name));
    return true;
}

bool readWordIfPresent
(
    const dictionary& dict,
    const word& name,
    word& value
)
{
    if (!dict.found(name))
    {
        return false;
    }
    value = word(dict.lookup(name));
    return true;
}

void applyPatchDictionary
(
    const dictionary& dict,
    PatchRadiationControls& controls
)
{
    readWordIfPresent(dict, "type", controls.type);

    scalar temperature = controls.specifiedTemperatureK;
    if
    (
        readScalarIfPresent(dict, "T", temperature)
     || readScalarIfPresent(dict, "temperature", temperature)
    )
    {
        controls.hasSpecifiedTemperature = true;
        controls.specifiedTemperatureK = temperature;
    }

    if (!readScalarIfPresent(dict, "absorptivity", controls.absorptivity))
    {
        readScalarIfPresent(dict, "emissivity", controls.absorptivity);
    }
    if (!readScalarIfPresent(dict, "I", controls.fixedIntensity))
    {
        readScalarIfPresent(dict, "IFixed", controls.fixedIntensity);
    }
}

void validatePatchControls
(
    const PatchRadiationControls& controls,
    const label patchI,
    const word& patchName
)
{
    if (!isSupportedRadiationType(controls.type))
    {
        solveFailure
        (
            "unsupported radiation boundary type '"
          + std::string(controls.type.c_str()) + "' on patch '"
          + std::string(patchName.c_str()) + "'",
            0,
            std::numeric_limits<scalar>::quiet_NaN(),
            -1,
            patchI,
            -1
        );
    }

}

void buildAngularQuadrature
(
    const label nTheta,
    const label nPhi,
    vectorField& directions,
    scalarField& weights
)
{
    const label nDirections = nTheta*nPhi;
    directions.setSize(nDirections);
    weights.setSize(nDirections);

    const scalar pi = fourPiConstant()/scalar(4);
    const scalar deltaTheta = pi/scalar(nTheta);
    const scalar deltaPhi = scalar(2)*pi/scalar(nPhi);
    scalar weightSum = 0;
    label directionI = 0;

    for (label thetaI = 0; thetaI < nTheta; ++thetaI)
    {
        const scalar theta = (scalar(thetaI) + scalar(0.5))*deltaTheta;
        const scalar sinTheta = Foam::sin(theta);
        const scalar cosTheta = Foam::cos(theta);

        for (label phiI = 0; phiI < nPhi; ++phiI)
        {
            const scalar phi = (scalar(phiI) + scalar(0.5))*deltaPhi;
            directions[directionI] = vector
            (
                sinTheta*Foam::cos(phi),
                sinTheta*Foam::sin(phi),
                cosTheta
            );
            weights[directionI] = sinTheta*deltaTheta*deltaPhi;
            weightSum += weights[directionI];
            ++directionI;
        }
    }

    forAll(weights, i)
    {
        weights[i] *= fourPiConstant()/weightSum;
    }
}

label nearestDirection
(
    const vector& target,
    const vectorField& directions
)
{
    label best = 0;
    scalar bestDot = -GREAT;
    forAll(directions, directionI)
    {
        const scalar alignment = target & directions[directionI];
        if (alignment > bestDot)
        {
            bestDot = alignment;
            best = directionI;
        }
    }
    return best;
}

}             

class MieDoRadiationSolverData
{
public:
    const fvMesh& mesh;
    const MieTable mieTable;
    const labelList coupledPatchIds;
    const MieDoControls controls;
    boolList isCoupledPatch;
    List<PatchRadiationControls> patchControls;
    vectorField directions;
    scalarField weights;
    List<scalarField> directionInternalFlux;
    List<scalarField> directionTransportDiagonal;
    List<List<scalarField>> directionBoundaryFlux;
    List<scalarField> diffuseDenominator;
    scalar sourceRelaxation;
    scalar linearSolverTolerance;
    scalar linearSolverRelativeTolerance;
    label linearSolverMaxIterations;

    MieDoRadiationSolverData
    (
        const fvMesh& meshValue,
        const dictionary& radiationProperties,
        const MieTable& tableValue,
        const labelList& coupledPatchIdsValue,
        const MieDoControls& controlsValue
    )
    :
        mesh(meshValue),
        mieTable(tableValue),
        coupledPatchIds(coupledPatchIdsValue),
        controls(controlsValue),
        isCoupledPatch(mesh.boundary().size(), false),
        patchControls(mesh.boundary().size()),
        diffuseDenominator(mesh.boundary().size()),
        sourceRelaxation
        (
            radiationProperties.lookupOrDefault<scalar>
            (
                "sourceRelaxation",
                scalar(1)
            )
        ),
        linearSolverTolerance
        (
            radiationProperties.lookupOrDefault<scalar>
            (
                "linearSolverTolerance",
                min(scalar(1e-12), controls.sourceRelativeTolerance*scalar(0.01))
            )
        ),
        linearSolverRelativeTolerance
        (
            radiationProperties.lookupOrDefault<scalar>
            (
                "linearSolverRelTol",
                scalar(0)
            )
        ),
        linearSolverMaxIterations
        (
            radiationProperties.lookupOrDefault<label>
            (
                "linearSolverMaxIterations",
                1000
            )
        )
    {
        if (Pstream::parRun())
        {
            solveFailure("MieDoRadiationSolver requires a reconstructed serial fvMesh");
        }
        if
        (
            controls.nTheta < 1
         || controls.nPhi < 1
        )
        {
            solveFailure("invalid Mie DO controls");
        }

        forAll(coupledPatchIds, coupledI)
        {
            const label patchI = coupledPatchIds[coupledI];
            if (patchI < 0 || patchI >= mesh.boundary().size())
            {
                solveFailure("coupled solid patch id is out of range", 0, scalar(0), -1, patchI, -1);
            }
            isCoupledPatch[patchI] = true;
        }

        const word defaultType = radiationProperties.lookupOrDefault<word>
        (
            "IBoundaryCondition",
            word("diffuseGrey")
        );
        const scalar defaultAbsorptivity =
            radiationProperties.lookupOrDefault<scalar>
            (
                "boundaryAbsorptivity",
                scalar(1)
            );
        const scalar defaultFixedIntensity =
            radiationProperties.lookupOrDefault<scalar>("IFixed", scalar(0));

        PatchRadiationControls defaults
        {
            defaultType,
            false,
            scalar(0),
            defaultAbsorptivity,
            defaultFixedIntensity
        };

        const dictionary* boundaryRadiation = nullptr;
        if (radiationProperties.found("boundaryRadiation"))
        {
            boundaryRadiation = &radiationProperties.subDict("boundaryRadiation");
            if (boundaryRadiation->found("default"))
            {
                applyPatchDictionary(boundaryRadiation->subDict("default"), defaults);
            }
        }

        forAll(mesh.boundary(), patchI)
        {
            if (mesh.boundary()[patchI].size() == 0)
            {
                continue;
            }
            const word polyType = mesh.boundaryMesh()[patchI].type();
            if (!isSupportedSerialPatch(polyType))
            {
                solveFailure
                (
                    "unsupported serial fvPatch type '" + std::string(polyType.c_str()) + "'",
                    0,
                    scalar(0),
                    -1,
                    patchI,
                    -1
                );
            }

            patchControls[patchI] = defaults;
            const word& patchName = mesh.boundary()[patchI].name();
            if (boundaryRadiation && boundaryRadiation->found(patchName))
            {
                applyPatchDictionary
                (
                    boundaryRadiation->subDict(patchName),
                    patchControls[patchI]
                );
            }
            if (polyType == "wedge")
            {
                patchControls[patchI].type = "symmetry";
            }
            validatePatchControls(patchControls[patchI], patchI, patchName);

            if (isCoupledPatch[patchI] && !isDiffuseType(patchControls[patchI].type))
            {
                solveFailure
                (
                    "coupled solid patch requires diffuseGrey radiation type",
                    0,
                    scalar(0),
                    -1,
                    patchI,
                    -1
                );
            }

        }

        buildAngularQuadrature
        (
            controls.nTheta,
            controls.nPhi,
            directions,
            weights
        );

        const labelUList& owner = mesh.owner();
        const labelUList& neighbour = mesh.neighbour();
        const vectorField& internalAreas = mesh.Sf().primitiveField();
        directionInternalFlux.setSize(directions.size());
        directionTransportDiagonal.setSize(directions.size());
        directionBoundaryFlux.setSize(directions.size());
        forAll(directions, directionI)
        {
            scalarField& internalFlux = directionInternalFlux[directionI];
            scalarField& transportDiagonal =
                directionTransportDiagonal[directionI];
            List<scalarField>& boundaryFlux =
                directionBoundaryFlux[directionI];
            internalFlux.setSize(mesh.nInternalFaces());
            transportDiagonal.setSize(mesh.nCells(), scalar(0));
            boundaryFlux.setSize(mesh.boundary().size());

            forAll(internalFlux, faceI)
            {
                const scalar orientedAreaFlux =
                    directions[directionI] & internalAreas[faceI];
                internalFlux[faceI] = orientedAreaFlux;
                const InternalFaceUpwindCoefficients coefficients =
                    internalFaceUpwindCoefficients(orientedAreaFlux);
                transportDiagonal[owner[faceI]] += coefficients.ownerDiag;
                transportDiagonal[neighbour[faceI]] += coefficients.neighbourDiag;
            }

            forAll(mesh.boundary(), patchI)
            {
                const labelUList& faceCells =
                    mesh.boundary()[patchI].faceCells();
                const vectorField& faceAreas =
                    mesh.Sf().boundaryField()[patchI];
                boundaryFlux[patchI].setSize(faceCells.size());
                forAll(faceCells, faceI)
                {
                    const scalar orientedAreaFlux =
                        directions[directionI] & faceAreas[faceI];
                    boundaryFlux[patchI][faceI] = orientedAreaFlux;
                    if (orientedAreaFlux > scalar(0))
                    {
                        transportDiagonal[faceCells[faceI]] += orientedAreaFlux;
                    }
                }
            }
        }

        forAll(mesh.boundary(), patchI)
        {
            const vectorField& faceAreas = mesh.Sf().boundaryField()[patchI];
            diffuseDenominator[patchI].setSize(faceAreas.size(), scalar(0));
            forAll(faceAreas, faceI)
            {
                const scalar areaMagnitude = mag(faceAreas[faceI]);

                const vector outwardNormal = faceAreas[faceI]/areaMagnitude;
                scalar denominator = 0;
                forAll(directions, directionI)
                {
                    const scalar cosineOut = directions[directionI] & outwardNormal;
                    if (cosineOut < scalar(0))
                    {
                        denominator += weights[directionI]*(-cosineOut);
                    }
                }

                diffuseDenominator[patchI][faceI] = denominator;
            }
        }
    }

    scalar boundaryTemperature
    (
        const MieDoSnapshot& snapshot,
        const label patchI,
        const label faceI
    ) const
    {
        if (!isCoupledPatch[patchI] && patchControls[patchI].hasSpecifiedTemperature)
        {
            return patchControls[patchI].specifiedTemperatureK;
        }
        return snapshot.wallTemperatureK.boundaryField()[patchI][faceI];
    }

    scalar boundaryEmissivity
    (
        const MieDoSnapshot& snapshot,
        const label patchI,
        const label faceI
    ) const
    {
        if (isCoupledPatch[patchI])
        {
            return snapshot.coupledWallEmissivity[patchI][faceI];
        }
        return patchControls[patchI].absorptivity;
    }

    void validateSnapshot(const MieDoSnapshot& snapshot) const
    {
        const label nCells = mesh.nCells();
        if
        (
            snapshot.epsilonS.size() < nCells
         || snapshot.particleTemperatureK.size() < nCells
         || snapshot.representativeDiameterM.size() < nCells
        )
        {
            solveFailure("Mie DO snapshot cell arrays do not match fvMesh::nCells");
        }
        if (snapshot.coupledWallEmissivity.size() < mesh.boundary().size()
            || snapshot.wallTemperatureK.boundaryField().size() < mesh.boundary().size())
        {
            solveFailure("coupled-wall emissivity list does not match boundary patch count");
        }

        forAll(mesh.boundary(), patchI)
        {
            const scalarField& temperature =
                snapshot.wallTemperatureK.boundaryField()[patchI];
            if (temperature.size() < mesh.boundary()[patchI].size())
            {
                solveFailure("wall-temperature patch size mismatch", 0, scalar(0), -1, patchI, -1);
            }

            if (isCoupledPatch[patchI])
            {
                const scalarField& emissivity = snapshot.coupledWallEmissivity[patchI];
                if (emissivity.size() < mesh.boundary()[patchI].size())
                {
                    solveFailure("coupled-wall emissivity patch size mismatch", 0, scalar(0), -1, patchI, -1);
                }

            }
        }
    }
};

namespace
{
ugkp::gpu::RadiationGeometry radiationGeometry(const MieDoRadiationSolverData& data)
{
    using ugkp::gpu::RadiationBoundary;
    ugkp::gpu::RadiationGeometry result;
    const fvMesh& mesh=data.mesh;
    result.cells=mesh.nCells();result.directions=data.directions.size();
    forAll(mesh.owner(),face)result.owner.push_back(mesh.owner()[face]);
    forAll(mesh.neighbour(),face)result.neighbour.push_back(mesh.neighbour()[face]);
    result.cellOffsets.push_back(0);
    forAll(mesh.cells(),cellI){for(const label face:mesh.cells()[cellI])if(face<mesh.nInternalFaces())result.cellFaces.push_back(face);
        result.cellOffsets.push_back(result.cellFaces.size());}
    forAll(mesh.V(),cellI)result.volume.push_back(mesh.V()[cellI]);
    for(const scalar weight:data.weights)result.weights.push_back(weight);
    for(const vector& a:data.directions)for(const vector& b:data.directions)result.directionCosines.push_back(max(scalar(-1),min(scalar(1),a&b)));
    forAll(mesh.boundary(),patchI){const word& type=data.patchControls[patchI].type;
        RadiationBoundary kind=RadiationBoundary::Diffuse;
        if(isSpecularType(type))kind=RadiationBoundary::Specular;
        else if(type=="zeroIncoming"||type=="zero")kind=RadiationBoundary::Zero;
        else if(type=="fixedValue"||type=="fixedI")kind=RadiationBoundary::Fixed;
        else if(type=="blackBody"||type=="black")kind=RadiationBoundary::Black;
        forAll(mesh.boundary()[patchI],faceI){
            result.boundaryOwner.push_back(mesh.boundary()[patchI].faceCells()[faceI]);result.boundaryPatch.push_back(patchI);result.boundaryFace.push_back(faceI);
            result.boundaryType.push_back(kind);result.boundaryArea.push_back(mag(mesh.Sf().boundaryField()[patchI][faceI]));
            result.diffuseDenominator.push_back(data.diffuseDenominator[patchI][faceI]);result.fixedIntensity.push_back(data.patchControls[patchI].fixedIntensity);
        }
    }
    forAll(data.directions,directionI){
        for(const scalar flux:data.directionInternalFlux[directionI])result.internalFlux.push_back(flux);
        for(const scalar diagonal:data.directionTransportDiagonal[directionI])result.transportDiagonal.push_back(diagonal);
        forAll(mesh.boundary(),patchI)forAll(mesh.boundary()[patchI],faceI){
            const vector area=mesh.Sf().boundaryField()[patchI][faceI];
            result.boundaryFlux.push_back(data.directionBoundaryFlux[directionI][patchI][faceI]);
            result.boundaryCosine.push_back(data.directions[directionI]&(area/mag(area)));
            const vector normal=area/max(mag(area),VSMALL);
            vector reflected=data.directions[directionI]-scalar(2)*(data.directions[directionI]&normal)*normal;
            const scalar magnitude=mag(reflected);if(magnitude>VSMALL)reflected/=magnitude;
            result.reflectedDirection.push_back(nearestDirection(reflected,data.directions));
        }
    }
    return result;
}
ugkp::gpu::RadiationControls radiationControls(const MieDoRadiationSolverData& data)
{
    ugkp::gpu::RadiationControls result;
    result.activeEpsilonMinimum=data.controls.radiationActiveEpsilonMin;result.phaseTolerance=data.controls.phaseBalanceTolerance;
    result.sourceTolerance=data.controls.sourceRelativeTolerance;result.linearTolerance=data.linearSolverTolerance;
    result.linearRelativeTolerance=data.linearSolverRelativeTolerance;result.sourceRelaxation=data.sourceRelaxation;result.small=SMALL;
    result.phaseIterations=data.controls.maxPhaseBalanceIterations;result.sourceIterations=data.controls.maxSourceIterations;result.linearIterations=data.linearSolverMaxIterations;
    result.parallelAngles=min(max(data.controls.angleParallelThreads,label(1)),max(data.directions.size(),label(1)))>1;
    return result;
}
void copyRadiationField(const std::vector<double>& source,scalarField& target)
{
    target.setSize(source.size());forAll(target,i)target[i]=source[i];
}
}
class MieDoRadiationWorkspaceData
{
public:
    std::shared_ptr<const MieDoRadiationSolverData> owner;
    std::unique_ptr<ugkp::gpu::DiscreteOrdinateRadiation> gpu;
    bool intensitiesInitialized=false;
    explicit MieDoRadiationWorkspaceData(const MieDoRadiationSolver&,const std::shared_ptr<const MieDoRadiationSolverData>&){}
    void prepare(const std::shared_ptr<const MieDoRadiationSolverData>& data)
    {
        if(owner==data)return;
        std::vector<double> warm;
        const bool matching=owner&&owner->mesh.nCells()==data->mesh.nCells()&&owner->directions.size()==data->directions.size();
        if(matching&&intensitiesInitialized)gpu->downloadIntensities(warm);
        std::unique_ptr<ugkp::gpu::DiscreteOrdinateRadiation> next(new ugkp::gpu::DiscreteOrdinateRadiation
            (radiationGeometry(*data),data->mieTable.exportGpuTable(),radiationControls(*data)));
        if(!warm.empty())next->uploadIntensities(warm);
        intensitiesInitialized=matching&&intensitiesInitialized;gpu=std::move(next);owner=data;
    }
};

MieDoRadiationWorkspace::MieDoRadiationWorkspace
(
    std::unique_ptr<MieDoRadiationWorkspaceData> data
) noexcept
:
    data_(std::move(data))
{}

MieDoRadiationWorkspace::~MieDoRadiationWorkspace() = default;

MieDoRadiationWorkspace::MieDoRadiationWorkspace
(
    MieDoRadiationWorkspace&& other
) noexcept = default;

MieDoRadiationWorkspace& MieDoRadiationWorkspace::operator=
(
    MieDoRadiationWorkspace&& other
) noexcept = default;

MieDoRadiationSolver::MieDoRadiationSolver
(
    const fvMesh& mesh,
    const dictionary& radiationProperties,
    const MieTable& mieTable,
    const labelList& coupledSolidPatchIds,
    const MieDoControls& controls
)
:
    data_
    (
        std::make_shared<MieDoRadiationSolverData>
        (
            mesh,
            radiationProperties,
            mieTable,
            coupledSolidPatchIds,
            controls
        )
    )
{}

const fvMesh& MieDoRadiationSolver::mesh() const noexcept
{
    return data_->mesh;
}

const MieTable& MieDoRadiationSolver::mieTable() const noexcept
{
    return data_->mieTable;
}

const labelList& MieDoRadiationSolver::coupledSolidPatchIds() const noexcept
{
    return data_->coupledPatchIds;
}

const MieDoControls& MieDoRadiationSolver::controls() const noexcept
{
    return data_->controls;
}

void MieDoRadiationSolver::validateSnapshot
(
    const MieDoSnapshot& snapshot
) const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    data_->validateSnapshot(snapshot);
}

MieDoRadiationWorkspace MieDoRadiationSolver::makeWorkspace() const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    return MieDoRadiationWorkspace
    (
        std::unique_ptr<MieDoRadiationWorkspaceData>
        (
            new MieDoRadiationWorkspaceData(*this, data_)
        )
    );
}

MieDoResult MieDoRadiationSolver::solve(const MieDoSnapshot& snapshot) const
{
    MieDoRadiationWorkspace workspace = makeWorkspace();
    return solve(snapshot, workspace);
}

MieDoResult MieDoRadiationSolver::solve
(
    const MieDoSnapshot& snapshot,
    MieDoRadiationWorkspace& workspace
) const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    if (!workspace.data_)
    {
        solveFailure("Mie DO workspace has no storage");
    }
    const MieDoRadiationSolverData& data = *data_;
    data.validateSnapshot(snapshot);
    MieDoRadiationWorkspaceData& scratch = *workspace.data_;
    scratch.prepare(data_);
    const fvMesh& mesh=data.mesh;
    const label nCells=mesh.nCells();
    ugkp::gpu::RadiationSnapshot input;
    input.epsilon.assign(snapshot.epsilonS.begin(),snapshot.epsilonS.begin()+nCells);
    input.temperature.assign(snapshot.particleTemperatureK.begin(),snapshot.particleTemperatureK.begin()+nCells);
    input.diameter.assign(snapshot.representativeDiameterM.begin(),snapshot.representativeDiameterM.begin()+nCells);
    forAll(mesh.boundary(),patchI)forAll(mesh.boundary()[patchI],faceI){
        input.wallTemperature.push_back(data.boundaryTemperature(snapshot,patchI,faceI));
        input.emissivity.push_back(data.boundaryEmissivity(snapshot,patchI,faceI));
    }
    const bool warm=scratch.intensitiesInitialized;scratch.intensitiesInitialized=false;
    ugkp::gpu::RadiationResult output;
    try {output=scratch.gpu->solve(input,warm);}
    catch(const ugkp::gpu::RadiationError& error){solveFailure(error.what(),error.iteration,error.value,error.cell,error.patch,error.face);}
    catch(const std::exception& error){solveFailure(std::string("CUDA Mie DO solve failed: ")+error.what());}
    if(hostDiagnosticWarningsEnabled()){
        for(label cellI=0;cellI<nCells;++cellI){const auto& phase=output.phaseMetrics[cellI];
            if(phase.rowError>data.controls.phaseBalanceTolerance||phase.columnError>data.controls.phaseBalanceTolerance)
                WarningInFunction<<"DO phase balance iteration budget exhausted; time="<<mesh.time().timeName()<<"; cell="<<cellI
                    <<"; iterations="<<phase.iterations<<"; rowResidual="<<phase.rowError<<"; columnResidual="<<phase.columnError
                    <<"; limit="<<data.controls.phaseBalanceTolerance<<"; retaining computed phase probabilities"<<endl;
        }
        for(std::size_t direction=0;direction<output.directionWarnings.size();++direction){const auto& warning=output.directionWarnings[direction];
            if(warning.linearIteration>0)WarningInFunction<<"DO linear iteration budget exhausted; time="<<mesh.time().timeName()
                <<"; direction="<<direction<<"; sourceIteration="<<warning.sourceIteration<<"; linearIterations="<<warning.linearIteration
                <<"; residual="<<warning.residual<<"; limit="<<warning.limit<<"; retaining computed intensity"<<endl;
        }
        if(output.maximumRelativeChange>data.controls.sourceRelativeTolerance)
            WarningInFunction<<"DO source iteration budget exhausted; time="<<mesh.time().timeName()<<"; iterations="<<output.sourceIterations
                <<"; residual="<<output.maximumRelativeChange<<"; limit="<<data.controls.sourceRelativeTolerance<<"; retaining computed intensities"<<endl;
    }
    MieDoResult result;
    copyRadiationField(output.absorption,result.absorptionCoefficientInvM);copyRadiationField(output.scattering,result.scatteringCoefficientInvM);
    copyRadiationField(output.extinction,result.extinctionCoefficientInvM);copyRadiationField(output.asymmetry,result.asymmetryFactor);
    copyRadiationField(output.incident,result.incidentRadiationG_Wm2);copyRadiationField(output.power,result.particlePowerDensityWm3);
    result.boundaryQrOutwardWm2.setSize(mesh.boundary().size());result.integratedParticlePowerW=0;result.integratedBoundaryPowerW=0;
    result.sourceIterations=output.sourceIterations;result.angleThreadsUsed=1;
    forAll(result.particlePowerDensityWm3,cellI)result.integratedParticlePowerW+=result.particlePowerDensityWm3[cellI]*mesh.V()[cellI];
    std::size_t index=0;
    forAll(mesh.boundary(),patchI){auto& values=result.boundaryQrOutwardWm2[patchI];values.setSize(mesh.boundary()[patchI].size());
        forAll(values,faceI){values[faceI]=output.boundaryFlux[index++];result.integratedBoundaryPowerW+=values[faceI]*mag(mesh.Sf().boundaryField()[patchI][faceI]);}}
    result.conservationResidualW=result.integratedParticlePowerW+result.integratedBoundaryPowerW;
    scratch.intensitiesInitialized=true;return result;
}

}
}
