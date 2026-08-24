#include "MieDoRadiationSolver.H"

#include "MieDoRadiationMath.H"
#include "Pstream.H"
#include "surfaceFields.H"
#include "wallPolyPatch.H"

#include <cmath>
#include <exception>
#include <limits>
#include <set>
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

inline bool finiteScalar(const scalar value)
{
    return std::isfinite(value);
}

scalar validatedNonnegativeIntensity
(
    const scalar value,
    const scalar referenceMagnitude,
    const std::string& location,
    const label iteration = 0,
    const label cell = -1,
    const label patch = -1,
    const label face = -1
)
{
    if (!finiteScalar(value))
    {
        solveFailure
        (
            "nonfinite radiation intensity at " + location,
            iteration,
            value,
            cell,
            patch,
            face
        );
    }
    if (value >= scalar(0))
    {
        return value;
    }

    const scalar roundoffTolerance =
        scalar(256)*std::numeric_limits<scalar>::epsilon()
       *max(mag(referenceMagnitude), scalar(1));
    if (value >= -roundoffTolerance)
    {
        return scalar(0);
    }

    solveFailure
    (
        "negative radiation intensity exceeds roundoff at " + location,
        iteration,
        value,
        cell,
        patch,
        face
    );
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
    if
    (
        !finiteScalar(controls.absorptivity)
     || controls.absorptivity < scalar(0)
     || controls.absorptivity > scalar(1)
    )
    {
        solveFailure("boundary absorptivity is outside [0,1]", 0, controls.absorptivity, -1, patchI, -1);
    }
    if (!finiteScalar(controls.fixedIntensity) || controls.fixedIntensity < scalar(0))
    {
        solveFailure("fixed boundary intensity is negative or nonfinite", 0, controls.fixedIntensity, -1, patchI, -1);
    }
    if
    (
        controls.hasSpecifiedTemperature
     && (!finiteScalar(controls.specifiedTemperatureK) || controls.specifiedTemperatureK <= scalar(0))
    )
    {
        solveFailure("specified boundary temperature is nonpositive or nonfinite", 0, controls.specifiedTemperatureK, -1, patchI, -1);
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

    if (!finiteScalar(weightSum) || weightSum <= scalar(0))
    {
        solveFailure("angular quadrature has a nonpositive weight sum");
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

scalar relativeChange
(
    const scalarField& oldValues,
    const scalarField& newValues
)
{
    scalar maxDifference = 0;
    scalar maxNew = SMALL;
    forAll(newValues, cellI)
    {
        maxDifference = max(maxDifference, mag(newValues[cellI] - oldValues[cellI]));
        maxNew = max(maxNew, mag(newValues[cellI]));
    }
    return maxDifference/maxNew;
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
         || controls.maxSourceIterations < 1
         || !finiteScalar(controls.sourceRelativeTolerance)
         || controls.sourceRelativeTolerance <= scalar(0)
         || !finiteScalar(controls.phaseBalanceTolerance)
         || controls.phaseBalanceTolerance <= scalar(0)
         || controls.maxPhaseBalanceIterations < 1
         || !finiteScalar(controls.radiationActiveEpsilonMin)
         || controls.radiationActiveEpsilonMin < scalar(0)
         || !finiteScalar(controls.conservationRelativeTolerance)
         || controls.conservationRelativeTolerance <= scalar(0)
         || !finiteScalar(controls.maxPhaseScalingCorrection)
         || controls.maxPhaseScalingCorrection < scalar(0)
         || controls.angleParallelThreads < 1
         || !finiteScalar(sourceRelaxation)
         || sourceRelaxation <= scalar(0)
         || sourceRelaxation > scalar(1)
         || !finiteScalar(linearSolverTolerance)
         || linearSolverTolerance <= scalar(0)
         || !finiteScalar(linearSolverRelativeTolerance)
         || linearSolverRelativeTolerance < scalar(0)
         || linearSolverMaxIterations < 1
        )
        {
            solveFailure("invalid Mie DO controls");
        }

        std::set<label> uniquePatches;
        forAll(coupledPatchIds, coupledI)
        {
            const label patchI = coupledPatchIds[coupledI];
            if (patchI < 0 || patchI >= mesh.boundary().size())
            {
                solveFailure("coupled solid patch id is out of range", 0, scalar(0), -1, patchI, -1);
            }
            if (!uniquePatches.insert(patchI).second)
            {
                solveFailure("duplicate coupled solid patch id", 0, scalar(0), -1, patchI, -1);
            }
            if (!isA<wallPolyPatch>(mesh.boundaryMesh()[patchI]))
            {
                solveFailure("coupled solid radiation patch is not a wall", 0, scalar(0), -1, patchI, -1);
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
                if (!finiteScalar(areaMagnitude) || areaMagnitude <= VSMALL)
                {
                    solveFailure("radiation boundary face has zero/nonfinite area", 0, areaMagnitude, -1, patchI, faceI);
                }
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
                if (!finiteScalar(denominator) || denominator <= VSMALL)
                {
                    solveFailure("angular grid has no incoming direction on boundary face", 0, denominator, -1, patchI, faceI);
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

    scalar incomingIntensity
    (
        const MieDoSnapshot& snapshot,
        const List<scalarField>& intensities,
        const List<scalarField>& incidentFlux,
        const label directionI,
        const label patchI,
        const label faceI,
        const label cellI
    ) const
    {
        const word& type = patchControls[patchI].type;

        if (isSpecularType(type))
        {
            const vectorField& faceAreas = mesh.Sf().boundaryField()[patchI];
            const vector normal = faceAreas[faceI]/max(mag(faceAreas[faceI]), VSMALL);
            vector reflected =
                directions[directionI]
              - scalar(2)*(directions[directionI] & normal)*normal;
            const scalar reflectedMagnitude = mag(reflected);
            if (reflectedMagnitude > VSMALL)
            {
                reflected /= reflectedMagnitude;
            }
            return validatedNonnegativeIntensity
            (
                intensities[nearestDirection(reflected, directions)][cellI],
                intensities[directionI][cellI],
                "specular boundary reflection",
                0,
                cellI,
                patchI,
                faceI
            );
        }
        if (type == "zeroIncoming" || type == "zero")
        {
            return scalar(0);
        }
        if (type == "fixedValue" || type == "fixedI")
        {
            return patchControls[patchI].fixedIntensity;
        }

        const scalar wallTemperature = boundaryTemperature(snapshot, patchI, faceI);
        const scalar bandFraction = mieTable.planckBandFraction(wallTemperature);
        const scalar blackIntensity =
            bandBlackBodyIntensity(wallTemperature, bandFraction);

        if (type == "blackBody" || type == "black")
        {
            return blackIntensity;
        }

        const scalar emissivity = boundaryEmissivity(snapshot, patchI, faceI);
        return
            emissivity*blackIntensity
          + (scalar(1) - emissivity)
           *validatedNonnegativeIntensity
            (
                incidentFlux[patchI][faceI],
                incidentFlux[patchI][faceI],
                "diffuse reflected incident flux",
                0,
                cellI,
                patchI,
                faceI
            )
           /diffuseDenominator[patchI][faceI];
    }

    void computeIncidentFlux
    (
        const List<scalarField>& intensities,
        List<scalarField>& incidentFlux
    ) const
    {
        forAll(mesh.boundary(), patchI)
        {
            incidentFlux[patchI] = scalar(0);
            const labelUList& faceCells = mesh.boundary()[patchI].faceCells();
            const vectorField& faceAreas = mesh.Sf().boundaryField()[patchI];

            forAll(faceCells, faceI)
            {
                const scalar areaMagnitude = mag(faceAreas[faceI]);
                const vector normal = faceAreas[faceI]/areaMagnitude;
                scalar flux = 0;
                forAll(directions, directionI)
                {
                    const scalar cosineOut = directions[directionI] & normal;
                    if (cosineOut > scalar(0))
                    {
                        flux +=
                            weights[directionI]*cosineOut
                           *validatedNonnegativeIntensity
                            (
                                intensities[directionI][faceCells[faceI]],
                                intensities[directionI][faceCells[faceI]],
                                "incident boundary-flux reconstruction",
                                0,
                                faceCells[faceI],
                                patchI,
                                faceI
                            );
                    }
                }
                incidentFlux[patchI][faceI] = flux;
            }
        }
    }

    void validateSnapshot(const MieDoSnapshot& snapshot) const
    {
        const label nCells = mesh.nCells();
        if
        (
            snapshot.epsilonS.size() != nCells
         || snapshot.particleTemperatureK.size() != nCells
         || snapshot.representativeDiameterM.size() != nCells
        )
        {
            solveFailure("Mie DO snapshot cell arrays do not match fvMesh::nCells");
        }
        if (&snapshot.wallTemperatureK.mesh() != &mesh)
        {
            solveFailure("Mie DO wall-temperature field belongs to a different fvMesh");
        }
        if (snapshot.coupledWallEmissivity.size() != mesh.boundary().size())
        {
            solveFailure("coupled-wall emissivity list does not match boundary patch count");
        }

        forAll(snapshot.epsilonS, cellI)
        {
            const scalar epsilon = snapshot.epsilonS[cellI];
            const scalar temperature = snapshot.particleTemperatureK[cellI];
            const scalar diameter = snapshot.representativeDiameterM[cellI];
            if (!finiteScalar(epsilon) || epsilon < scalar(0))
            {
                solveFailure("epsilonS is negative or nonfinite", 0, epsilon, cellI);
            }
            if (!finiteScalar(temperature) || !finiteScalar(diameter))
            {
                solveFailure("particle temperature/diameter is nonfinite", 0, scalar(0), cellI);
            }
            if (epsilon > controls.radiationActiveEpsilonMin)
            {
                if (diameter <= scalar(0))
                {
                    solveFailure("active radiation cell has nonpositive diameter", 0, diameter, cellI);
                }
            }
        }

        forAll(mesh.boundary(), patchI)
        {
            const scalarField& temperature =
                snapshot.wallTemperatureK.boundaryField()[patchI];
            if (temperature.size() != mesh.boundary()[patchI].size())
            {
                solveFailure("wall-temperature patch size mismatch", 0, scalar(0), -1, patchI, -1);
            }
            forAll(temperature, faceI)
            {
                if (!finiteScalar(temperature[faceI]))
                {
                    solveFailure("wall temperature is nonfinite", 0, temperature[faceI], -1, patchI, faceI);
                }
            }

            if (isCoupledPatch[patchI])
            {
                const scalarField& emissivity = snapshot.coupledWallEmissivity[patchI];
                if (emissivity.size() != mesh.boundary()[patchI].size())
                {
                    solveFailure("coupled-wall emissivity patch size mismatch", 0, scalar(0), -1, patchI, -1);
                }
                forAll(emissivity, faceI)
                {
                    if
                    (
                        !finiteScalar(emissivity[faceI])
                     || emissivity[faceI] < scalar(0)
                     || emissivity[faceI] > scalar(1)
                    )
                    {
                        solveFailure("coupled-wall emissivity is outside [0,1]", 0, emissivity[faceI], -1, patchI, faceI);
                    }
                }
            }
        }
    }
};

class MieDoRadiationWorkspaceData
{
public:
    const MieDoRadiationSolver* solverIdentity;
    std::shared_ptr<const MieDoRadiationSolverData> solverData;
    scalarField absorption;
    scalarField scattering;
    scalarField emissionIntensity;
    List<List<scalarField>> phaseProbability;
    List<scalarField> rawPhaseKernel;
    scalarField phaseScaling;
    scalarField nextPhaseScaling;
    List<scalarField> intensities;
    List<scalarField> frozenIntensities;
    List<scalarField> incidentFlux;
    List<scalarField> diagonal;
    List<scalarField> source;
    List<scalarField> solved;
    List<scalarField> linearResidual;
    bool intensitiesInitialized;

    explicit MieDoRadiationWorkspaceData
    (
        const MieDoRadiationSolver& solver,
        const std::shared_ptr<const MieDoRadiationSolverData>& data
    )
    :
        solverIdentity(&solver),
        solverData(data),
        intensitiesInitialized(false)
    {}

    void prepare(const MieDoRadiationSolverData& data)
    {
        const label nCells = data.mesh.nCells();
        const label nDirections = data.directions.size();

        absorption.setSize(nCells);
        scattering.setSize(nCells);
        emissionIntensity.setSize(nCells);
        absorption = scalar(0);
        scattering = scalar(0);
        emissionIntensity = scalar(0);

        phaseProbability.setSize(nCells);
        rawPhaseKernel.setSize(nDirections);
        forAll(rawPhaseKernel, directionI)
        {
            rawPhaseKernel[directionI].setSize(nDirections);
        }
        phaseScaling.setSize(nDirections);
        nextPhaseScaling.setSize(nDirections);

        const bool intensityShapeMatches =
            intensities.size() == nDirections
         &&
            (
                nDirections == 0
             || intensities[0].size() == nCells
            );
        intensities.setSize(nDirections);
        frozenIntensities.setSize(nDirections);
        diagonal.setSize(nDirections);
        source.setSize(nDirections);
        solved.setSize(nDirections);
        linearResidual.setSize(nDirections);
        forAll(intensities, directionI)
        {
            intensities[directionI].setSize(nCells);
            frozenIntensities[directionI].setSize(nCells);
            diagonal[directionI].setSize(nCells);
            source[directionI].setSize(nCells);
            solved[directionI].setSize(nCells);
            linearResidual[directionI].setSize(nCells);
        }
        if (!intensityShapeMatches)
        {
            intensitiesInitialized = false;
        }

        incidentFlux.setSize(data.mesh.boundary().size());
        forAll(incidentFlux, patchI)
        {
            incidentFlux[patchI].setSize(data.mesh.boundary()[patchI].size());
            incidentFlux[patchI] = scalar(0);
        }

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
    if
    (
        !workspace.data_
     || workspace.data_->solverIdentity != this
     || workspace.data_->solverData.get() != data_.get()
    )
    {
        solveFailure("Mie DO workspace belongs to a different solver");
    }
    const MieDoRadiationSolverData& data = *data_;
    data.validateSnapshot(snapshot);
    MieDoRadiationWorkspaceData& scratch = *workspace.data_;
    scratch.prepare(data);

    const fvMesh& mesh = data.mesh;
    const label nCells = mesh.nCells();
    const label nDirections = data.directions.size();
    scalarField& absorption = scratch.absorption;
    scalarField& scattering = scratch.scattering;
    scalarField& emissionIntensity = scratch.emissionIntensity;
    List<List<scalarField>>& phaseProbability = scratch.phaseProbability;
    scalar initialIntensity = 0;

    try
    {
        forAll(absorption, cellI)
        {
            const scalar epsilon = snapshot.epsilonS[cellI];
            if (epsilon <= data.controls.radiationActiveEpsilonMin)
            {
                continue;
            }

            const scalar diameter = snapshot.representativeDiameterM[cellI];
            const scalar temperature = snapshot.particleTemperatureK[cellI];
            const MieOpticalSample optical = data.mieTable.query(diameter, temperature);
            absorption[cellI] = scalar(3)*epsilon*optical.Qabs/(scalar(2)*diameter);
            scattering[cellI] = scalar(3)*epsilon*optical.Qsca/(scalar(2)*diameter);
            emissionIntensity[cellI] =
                bandBlackBodyIntensity(temperature, optical.planckBandFraction);
            initialIntensity = max(initialIntensity, emissionIntensity[cellI]);

            if (scattering[cellI] > scalar(0))
            {
                List<scalarField>& rawKernel = scratch.rawPhaseKernel;
                for (label rowI = 0; rowI < nDirections; ++rowI)
                {
                    for (label colI = 0; colI < nDirections; ++colI)
                    {
                        const scalar mu = max
                        (
                            scalar(-1),
                            min
                            (
                                scalar(1),
                                data.directions[rowI] & data.directions[colI]
                            )
                        );
                        rawKernel[rowI][colI] =
                            data.mieTable.phase(diameter, temperature, mu)
                           /fourPiConstant();
                    }
                }

                const PhaseBalanceMetrics phaseMetrics =
                    balanceSymmetricPhaseKernelInPlace
                (
                    rawKernel,
                    data.weights,
                    data.controls.phaseBalanceTolerance,
                    data.controls.maxPhaseBalanceIterations,
                    scratch.phaseScaling,
                    scratch.nextPhaseScaling,
                    phaseProbability[cellI]
                );
                if
                (
                    phaseMetrics.maxRelativeScalingCorrection
                  > data.controls.maxPhaseScalingCorrection
                )
                {
                    solveFailure
                    (
                        "phase diagonal scaling correction exceeds configured accuracy gate",
                        phaseMetrics.iterations,
                        phaseMetrics.maxRelativeScalingCorrection,
                        cellI
                    );
                }
            }
        }

        forAll(mesh.boundary(), patchI)
        {
            const word& type = data.patchControls[patchI].type;
            if
            (
                isDiffuseType(type)
             || type == "blackBody"
             || type == "black"
            )
            {
                forAll(mesh.boundary()[patchI], faceI)
                {
                    const scalar wallT = data.boundaryTemperature(snapshot, patchI, faceI);
                    const scalar wallIb = bandBlackBodyIntensity
                    (
                        wallT,
                        data.mieTable.planckBandFraction(wallT)
                    );
                    initialIntensity = max(initialIntensity, wallIb);
                }
            }
            else if (type == "fixedValue" || type == "fixedI")
            {
                initialIntensity = max(initialIntensity, data.patchControls[patchI].fixedIntensity);
            }
        }
    }
    catch (const MieDoSolveError&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        solveFailure(std::string("offline MIE_TABLE coefficient/phase query failed: ") + error.what());
    }

    List<scalarField>& intensities = scratch.intensities;
    const bool useWarmStart = scratch.intensitiesInitialized;
    scratch.intensitiesInitialized = false;
    if (!useWarmStart)
    {
        forAll(intensities, directionI)
        {
            intensities[directionI] = initialIntensity;
        }
    }
    else
    {
        forAll(intensities, directionI)
        {
            forAll(intensities[directionI], cellI)
            {
                (void)validatedNonnegativeIntensity
                (
                    intensities[directionI][cellI],
                    initialIntensity,
                    "warm-start radiation intensity",
                    0,
                    cellI
                );
            }
        }
    }

    List<scalarField>& frozenIntensities = scratch.frozenIntensities;
    List<scalarField>& incidentFlux = scratch.incidentFlux;
    List<scalarField>& directionDiagonal = scratch.diagonal;
    List<scalarField>& directionSource = scratch.source;
    List<scalarField>& directionSolved = scratch.solved;
    List<scalarField>& directionLinearResidual = scratch.linearResidual;

    scalar finalSourceResidual = GREAT;
    scalar maximumRelativeChange = scalar(0);
    label sourceIterations = 0;
    label angleThreadsUsed = 1;
    bool sourceFinished = false;
    std::exception_ptr parallelError;
    std::vector<std::exception_ptr> directionErrors
    (
        static_cast<std::size_t>(nDirections)
    );
    const label requestedThreads = min
    (
        data.controls.angleParallelThreads,
        max(nDirections, label(1))
    );
    const bool parallelAngles = requestedThreads > 1;
    const scalarField& volumes = mesh.V();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

#ifdef _OPENMP
    #pragma omp parallel num_threads(requestedThreads) shared(angleThreadsUsed, sourceFinished, parallelError, directionErrors, maximumRelativeChange, finalSourceResidual, sourceIterations, frozenIntensities, intensities, incidentFlux)
#endif
    {
#ifdef _OPENMP
        #pragma omp single
        {
            angleThreadsUsed = omp_get_num_threads();
        }
#endif
        for
        (
            label iteration = 1;
            iteration <= data.controls.maxSourceIterations;
            ++iteration
        )
        {
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                maximumRelativeChange = scalar(0);
                std::fill
                (
                    directionErrors.begin(),
                    directionErrors.end(),
                    std::exception_ptr()
                );
                try
                {
                    if (parallelAngles)
                    {
                        frozenIntensities = intensities;
                        data.computeIncidentFlux
                        (
                            frozenIntensities,
                            incidentFlux
                        );
                    }
                    else
                    {
                        data.computeIncidentFlux(intensities, incidentFlux);
                    }
                }
                catch (...)
                {
                    parallelError = std::current_exception();
                }
            }

            if (!parallelError)
            {
#ifdef _OPENMP
                #pragma omp for schedule(static) reduction(max:maximumRelativeChange)
#endif
                for (label directionI = 0; directionI < nDirections; ++directionI)
                {
                  try
                  {
                    const List<scalarField>& sourceIntensities =
                        parallelAngles ? frozenIntensities : intensities;
                    const scalarField& oldIntensity =
                        sourceIntensities[directionI];
                    scalarField& diagonal = directionDiagonal[directionI];
                    scalarField& source = directionSource[directionI];
            const scalarField& internalFlux =
                data.directionInternalFlux[directionI];
            diagonal = data.directionTransportDiagonal[directionI];
            source = scalar(0);
            forAll(diagonal, cellI)
            {
                diagonal[cellI] +=
                    (absorption[cellI] + scattering[cellI])*volumes[cellI];
            }

            forAll(mesh.boundary(), patchI)
            {
                const labelUList& faceCells = mesh.boundary()[patchI].faceCells();
                forAll(faceCells, faceI)
                {
                    const scalar orientedAreaFlux =
                        data.directionBoundaryFlux[directionI][patchI][faceI];
                    const label cellI = faceCells[faceI];
                    if (orientedAreaFlux < scalar(0))
                    {
                        const scalar incoming = data.incomingIntensity
                        (
                            snapshot,
                            sourceIntensities,
                            incidentFlux,
                            directionI,
                            patchI,
                            faceI,
                            cellI
                        );
                        source[cellI] -= orientedAreaFlux*incoming;
                    }
                }
            }

            forAll(source, cellI)
            {
                scalar scatteringMean = 0;
                if (scattering[cellI] > scalar(0))
                {
                    const List<scalarField>& probability = phaseProbability[cellI];
                    for (label otherDirectionI = 0; otherDirectionI < nDirections; ++otherDirectionI)
                    {
                        scatteringMean +=
                            probability[directionI][otherDirectionI]
                           *validatedNonnegativeIntensity
                            (
                                sourceIntensities[otherDirectionI][cellI],
                                initialIntensity,
                                "table-Mie scattering source",
                                iteration,
                                cellI
                            );
                    }
                }
                source[cellI] += volumes[cellI]
                   *(
                        absorption[cellI]*emissionIntensity[cellI]
                      + scattering[cellI]*scatteringMean
                    );
            }

            forAll(diagonal, cellI)
            {
                if (!finiteScalar(diagonal[cellI]) || diagonal[cellI] <= VSMALL)
                {
                    solveFailure
                    (
                        "DO upwind linear system has a nonpositive diagonal",
                        iteration,
                        diagonal[cellI],
                        cellI
                    );
                }
            }

            scalarField& solved = directionSolved[directionI];
            solved = oldIntensity;
            const auto linearResidual = [&](const scalarField& values)->scalar
            {
                scalarField& residual =
                    directionLinearResidual[directionI];
                residual = scalar(0);
                scalar maximumMagnitude = SMALL;
                scalar maximumResidual = 0;
                forAll(residual, cellI)
                {
                    residual[cellI] = diagonal[cellI]*values[cellI] - source[cellI];
                    maximumMagnitude = max
                    (
                        maximumMagnitude,
                        max
                        (
                            mag(diagonal[cellI]*values[cellI]),
                            mag(source[cellI])
                        )
                    );
                }
                forAll(internalFlux, faceI)
                {
                    const scalar orientedAreaFlux = internalFlux[faceI];
                    if (orientedAreaFlux > scalar(0))
                    {
                        residual[neighbour[faceI]] -=
                            orientedAreaFlux*values[owner[faceI]];
                        maximumMagnitude = max
                        (
                            maximumMagnitude,
                            mag(orientedAreaFlux*values[owner[faceI]])
                        );
                    }
                    else if (orientedAreaFlux < scalar(0))
                    {
                        residual[owner[faceI]] +=
                            orientedAreaFlux*values[neighbour[faceI]];
                        maximumMagnitude = max
                        (
                            maximumMagnitude,
                            mag(orientedAreaFlux*values[neighbour[faceI]])
                        );
                    }
                }
                forAll(residual, cellI)
                {
                    maximumResidual = max(maximumResidual, mag(residual[cellI]));
                }
                return maximumResidual/maximumMagnitude;
            };

            const scalar initialLinearResidual = linearResidual(solved);
            const scalar linearTarget = max
            (
                data.linearSolverTolerance,
                data.linearSolverRelativeTolerance*initialLinearResidual
            );
            scalar finalLinearResidual = initialLinearResidual;
            label linearIteration = 0;

            while
            (
                finalLinearResidual > linearTarget
             && linearIteration < data.linearSolverMaxIterations
            )
            {
                ++linearIteration;
                forAll(solved, cellI)
                {
                    scalar rightHandSide = source[cellI];
                    const cell& cellFaces = mesh.cells()[cellI];
                    forAll(cellFaces, localFaceI)
                    {
                        const label faceI = cellFaces[localFaceI];
                        if (faceI >= mesh.nInternalFaces())
                        {
                            continue;
                        }
                        const scalar orientedAreaFlux = internalFlux[faceI];
                        if (cellI == owner[faceI] && orientedAreaFlux < scalar(0))
                        {
                            rightHandSide -= orientedAreaFlux*solved[neighbour[faceI]];
                        }
                        else if
                        (
                            cellI == neighbour[faceI]
                         && orientedAreaFlux > scalar(0)
                        )
                        {
                            rightHandSide += orientedAreaFlux*solved[owner[faceI]];
                        }
                    }

                    solved[cellI] = validatedNonnegativeIntensity
                    (
                        rightHandSide/diagonal[cellI],
                        initialIntensity,
                        "FV upwind linear solve",
                        iteration,
                        cellI
                    );
                }
                finalLinearResidual = linearResidual(solved);
            }

            if (!finiteScalar(finalLinearResidual) || finalLinearResidual > linearTarget)
            {
                solveFailure
                (
                    "FV upwind linear solve did not converge",
                    iteration,
                    finalLinearResidual
                );
            }

            forAll(solved, cellI)
            {
                solved[cellI] =
                    (scalar(1) - data.sourceRelaxation)*oldIntensity[cellI]
                  + data.sourceRelaxation*solved[cellI];
            }

            const scalar directionRelativeChange =
                relativeChange(oldIntensity, solved);
            maximumRelativeChange = max
            (
                maximumRelativeChange,
                directionRelativeChange
            );
            intensities[directionI] = solved;
                  }
                  catch (...)
                  {
                      directionErrors[static_cast<std::size_t>(directionI)] =
                          std::current_exception();
                  }
                }
            }
            else
            {
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

#ifdef _OPENMP
            #pragma omp single
#endif
            {
                if (!parallelError)
                {
                    for (label directionI = 0; directionI < nDirections; ++directionI)
                    {
                        if
                        (
                            directionErrors
                            [static_cast<std::size_t>(directionI)]
                        )
                        {
                            parallelError = directionErrors
                            [static_cast<std::size_t>(directionI)];
                            break;
                        }
                    }
                }
                finalSourceResidual = maximumRelativeChange;
                sourceIterations = iteration;
                sourceFinished =
                    bool(parallelError)
                 || maximumRelativeChange
                    <= data.controls.sourceRelativeTolerance;
            }

#ifdef _OPENMP
            #pragma omp barrier
#endif
            if (sourceFinished)
            {
                break;
            }
        }
    }

    if (parallelError)
    {
        std::rethrow_exception(parallelError);
    }

    if (finalSourceResidual > data.controls.sourceRelativeTolerance)
    {
        solveFailure
        (
            "DO source iteration reached its limit without convergence",
            sourceIterations,
            finalSourceResidual
        );
    }

    data.computeIncidentFlux(intensities, incidentFlux);

    MieDoResult result;
    result.incidentRadiationG_Wm2.setSize(nCells, scalar(0));
    result.particlePowerDensityWm3.setSize(nCells, scalar(0));
    result.boundaryQrOutwardWm2.setSize(mesh.boundary().size());
    result.integratedParticlePowerW = scalar(0);
    result.integratedBoundaryPowerW = scalar(0);
    result.conservationResidualW = scalar(0);
    result.sourceIterations = sourceIterations;
    result.angleThreadsUsed = angleThreadsUsed;

    for (label directionI = 0; directionI < nDirections; ++directionI)
    {
        forAll(result.incidentRadiationG_Wm2, cellI)
        {
            result.incidentRadiationG_Wm2[cellI] +=
                data.weights[directionI]
               *validatedNonnegativeIntensity
                (
                    intensities[directionI][cellI],
                    initialIntensity,
                    "incident-radiation moment",
                    sourceIterations,
                    cellI
                );
        }
    }

    forAll(result.particlePowerDensityWm3, cellI)
    {
        result.particlePowerDensityWm3[cellI] =
            particleRadiationPowerDensity
            (
                absorption[cellI],
                result.incidentRadiationG_Wm2[cellI],
                emissionIntensity[cellI]
            );
        result.integratedParticlePowerW +=
            result.particlePowerDensityWm3[cellI]*volumes[cellI];
    }

    forAll(mesh.boundary(), patchI)
    {
        const labelUList& faceCells = mesh.boundary()[patchI].faceCells();
        const vectorField& faceAreas = mesh.Sf().boundaryField()[patchI];
        result.boundaryQrOutwardWm2[patchI].setSize(faceCells.size(), scalar(0));

        forAll(faceCells, faceI)
        {
            const scalar areaMagnitude = mag(faceAreas[faceI]);
            const vector normal = faceAreas[faceI]/areaMagnitude;
            scalar qrOutward = 0;

            for (label directionI = 0; directionI < nDirections; ++directionI)
            {
                const scalar cosineOut = data.directions[directionI] & normal;
                scalar faceIntensity = 0;
                if (cosineOut > scalar(0))
                {
                    faceIntensity = validatedNonnegativeIntensity
                    (
                        intensities[directionI][faceCells[faceI]],
                        initialIntensity,
                        "outgoing boundary intensity",
                        sourceIterations,
                        faceCells[faceI],
                        patchI,
                        faceI
                    );
                }
                else if (cosineOut < scalar(0))
                {
                    faceIntensity = data.incomingIntensity
                    (
                        snapshot,
                        intensities,
                        incidentFlux,
                        directionI,
                        patchI,
                        faceI,
                        faceCells[faceI]
                    );
                }
                const scalar directionalFlux =
                    data.weights[directionI]*cosineOut*faceIntensity;
                qrOutward += directionalFlux;
            }

            result.boundaryQrOutwardWm2[patchI][faceI] = qrOutward;
            result.integratedBoundaryPowerW += qrOutward*areaMagnitude;
        }
    }

    result.conservationResidualW =
        result.integratedParticlePowerW + result.integratedBoundaryPowerW;
    scratch.intensitiesInitialized = true;
    return result;
}

}                        
}                  
