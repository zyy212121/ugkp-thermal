#include "GpuSolidPatchMapper.H"

#include "Pstream.H"
#include "cyclicAMIPolyPatch.H"
#include "cyclicPolyPatch.H"
#include "mappedWallPolyPatch.H"
#include "wallPolyPatch.H"

#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <string>

namespace Foam
{
namespace gpuThermal
{

namespace
{

constexpr scalar areaRelativeTolerance = scalar(1e-8);
constexpr scalar centreRelativeTolerance = scalar(1e-9);
constexpr scalar oppositeNormalTolerance = scalar(1e-10);

[[noreturn]] void mappingFailure
(
    const SolidPatchPair& pair,
    const label firstBadFace,
    const std::string& reason
)
{
    std::ostringstream message;
    message.precision(17);
    message
        << "oneToOneConformal solid patch mapping rejected: " << reason
        << "; fluidPatch=" << pair.fluidPatch
        << "; solidPatch=" << pair.solidPatch
        << "; firstBadFace=" << firstBadFace;
    throw SolidPatchMappingError(message.str());
}

bool finiteScalar(const scalar value)
{
    return std::isfinite(value);
}

bool finiteVector(const vector& value)
{
    return
        finiteScalar(value.x())
     && finiteScalar(value.y())
     && finiteScalar(value.z());
}

void includePoints
(
    const pointField& points,
    point& minimum,
    point& maximum
)
{
    forAll(points, pointI)
    {
        for (direction component = 0; component < vector::nComponents; ++component)
        {
            minimum[component] = min(minimum[component], points[pointI][component]);
            maximum[component] = max(maximum[component], points[pointI][component]);
        }
    }
}

scalar combinedDomainLength
(
    const fvMesh& fluidMesh,
    const fvMesh& solidMesh,
    const SolidPatchPair& diagnosticPair
)
{
    if (fluidMesh.points().empty() || solidMesh.points().empty())
    {
        mappingFailure(diagnosticPair, -1, "fluid and solid meshes must contain points");
    }

    point minimum = fluidMesh.points()[0];
    point maximum = minimum;
    includePoints(fluidMesh.points(), minimum, maximum);
    includePoints(solidMesh.points(), minimum, maximum);
    const scalar length = mag(maximum - minimum);
    if (!finiteScalar(length))
    {
        mappingFailure(diagnosticPair, -1, "combined mesh domain length is nonfinite");
    }
    return length;
}

void rejectUnsupportedPatchType
(
    const polyPatch& patch,
    const bool fluidSide,
    const SolidPatchPair& pair,
    const bool allowMappedWall
)
{
    const word& patchType = patch.type();
    const std::string side = fluidSide ? "fluid" : "solid";
    if (patchType == mappedWallPolyPatch::typeName && !allowMappedWall)
    {
        mappingFailure
        (
            pair,
            patch.size() == 0 ? -1 : 0,
            side + " patch type mappedWall is unsupported in v1"
        );
    }
    if (patchType == cyclicAMIPolyPatch::typeName)
    {
        mappingFailure
        (
            pair,
            patch.size() == 0 ? -1 : 0,
            side + " patch type cyclicAMI is unsupported in v1"
        );
    }
    if (patchType == cyclicPolyPatch::typeName)
    {
        mappingFailure
        (
            pair,
            patch.size() == 0 ? -1 : 0,
            side + " patch type cyclic is unsupported in v1"
        );
    }
    if
    (
        !isA<wallPolyPatch>(patch)
     || (patchType != wallPolyPatch::typeName
      && !(allowMappedWall && patchType == mappedWallPolyPatch::typeName))
    )
    {
        mappingFailure
        (
            pair,
            patch.size() == 0 ? -1 : 0,
            side + " patch must be an ordinary wall; actual type="
          + std::string(patchType.c_str())
        );
    }
}

void validateGeometry
(
    const fvPatch& fluidPatch,
    const fvPatch& solidPatch,
    const scalar centreTolerance,
    const SolidPatchPair& pair
)
{
    if (fluidPatch.size() != solidPatch.size())
    {
        mappingFailure
        (
            pair,
            min(fluidPatch.size(), solidPatch.size()),
            "local face count mismatch: fluid="
          + std::to_string(fluidPatch.size())
          + ", solid=" + std::to_string(solidPatch.size())
        );
    }

    const vectorField& fluidCentres = fluidPatch.Cf();
    const vectorField& solidCentres = solidPatch.Cf();
    const vectorField& fluidAreas = fluidPatch.Sf();
    const vectorField& solidAreas = solidPatch.Sf();
    const scalarField& fluidAreaMagnitudes = fluidPatch.magSf();
    const scalarField& solidAreaMagnitudes = solidPatch.magSf();

    forAll(fluidCentres, faceI)
    {
        const scalar fluidArea = fluidAreaMagnitudes[faceI];
        const scalar solidArea = solidAreaMagnitudes[faceI];
        if
        (
            !finiteScalar(fluidArea)
         || !finiteScalar(solidArea)
         || fluidArea <= scalar(0)
         || solidArea <= scalar(0)
        )
        {
            mappingFailure(pair, faceI, "face area is nonpositive or nonfinite");
        }
        const scalar areaScale = max(fluidArea, solidArea);
        if (mag(fluidArea - solidArea) > areaRelativeTolerance*areaScale)
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "face area mismatch: fluid=" << fluidArea
                << ", solid=" << solidArea
                << ", relativeTolerance=" << areaRelativeTolerance;
            mappingFailure(pair, faceI, reason.str());
        }

        if
        (
            !finiteVector(fluidCentres[faceI])
         || !finiteVector(solidCentres[faceI])
        )
        {
            mappingFailure(pair, faceI, "face centre is nonfinite");
        }
        const scalar centreSeparation =
            mag(fluidCentres[faceI] - solidCentres[faceI]);
        if
        (
            !finiteScalar(centreSeparation)
         || centreSeparation > centreTolerance
        )
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "face centre mismatch: separation=" << centreSeparation
                << ", tolerance=" << centreTolerance;
            mappingFailure(pair, faceI, reason.str());
        }

        if
        (
            !finiteVector(fluidAreas[faceI])
         || !finiteVector(solidAreas[faceI])
        )
        {
            mappingFailure(pair, faceI, "face normal is nonfinite");
        }
        const vector fluidUnitNormal = fluidAreas[faceI]/fluidArea;
        const vector solidUnitNormal = solidAreas[faceI]/solidArea;
        const scalar oppositeResidual =
            scalar(1) + (fluidUnitNormal & solidUnitNormal);
        if
        (
            !finiteScalar(oppositeResidual)
         || oppositeResidual > oppositeNormalTolerance
        )
        {
            std::ostringstream reason;
            reason.precision(17);
            reason
                << "face normal mismatch: 1+nFluid.nSolid="
                << oppositeResidual
                << ", tolerance=" << oppositeNormalTolerance;
            mappingFailure(pair, faceI, reason.str());
        }
    }
}

void validateSourceSize
(
    const scalarField& source,
    const label expectedSize,
    const SolidPatchPair& pair,
    const std::string& direction
)
{
    if (source.size() != expectedSize)
    {
        mappingFailure
        (
            pair,
            min(source.size(), expectedSize),
            direction + " source face count mismatch: source="
          + std::to_string(source.size())
          + ", expected=" + std::to_string(expectedSize)
        );
    }
}

void requireBitwiseCopy
(
    const scalarField& source,
    const scalarField& mapped,
    const SolidPatchPair& pair,
    const std::string& direction
)
{
    forAll(source, faceI)
    {
        if (std::memcmp(&source[faceI], &mapped[faceI], sizeof(scalar)) != 0)
        {
            mappingFailure
            (
                pair,
                faceI,
                direction + " changed a face value instead of direct local-index copy"
            );
        }
    }
}

}             

OneToOneConformalSolidPatchMapper::OneToOneConformalSolidPatchMapper
(
    const fvMesh& fluidMesh,
    const fvMesh& solidMesh,
    const List<SolidPatchPair>& patchPairs,
    const bool allowMappedWall
)
:
    fluidMesh_(fluidMesh),
    solidMesh_(solidMesh),
    patchPairs_(patchPairs),
    fluidPatchIds_(patchPairs.size(), -1),
    solidPatchIds_(patchPairs.size(), -1)
{
    const SolidPatchPair emptyDiagnosticPair;
    if (patchPairs_.empty())
    {
        mappingFailure
        (
            emptyDiagnosticPair,
            -1,
            "at least one fluid-solid patch pair is required"
        );
    }
    const SolidPatchPair& firstPair = patchPairs_[0];
    if
    (
        Pstream::parRun()
     || fluidMesh_.time().processorCase()
     || solidMesh_.time().processorCase()
    )
    {
        mappingFailure
        (
            firstPair,
            0,
            "v1 oneToOneConformal mapping is serial only; decomposed execution is unsupported"
        );
    }

    const scalar domainLength =
        combinedDomainLength(fluidMesh_, solidMesh_, firstPair);
    const scalar centreTolerance =
        centreRelativeTolerance*max(domainLength, scalar(1));
    std::set<label> usedFluidPatchIds;
    std::set<label> usedSolidPatchIds;
    std::set<label> coveredFluidFaces;
    std::set<label> coveredSolidFaces;

    forAll(patchPairs_, pairI)
    {
        const SolidPatchPair& pair = patchPairs_[pairI];
        const label fluidPatchId =
            fluidMesh_.boundary().findPatchID(pair.fluidPatch);
        const label solidPatchId =
            solidMesh_.boundary().findPatchID(pair.solidPatch);
        if (fluidPatchId < 0 || solidPatchId < 0)
        {
            std::string reason;
            if (fluidPatchId < 0 && solidPatchId < 0)
            {
                reason = "fluid and solid patches do not exist";
            }
            else if (fluidPatchId < 0)
            {
                reason = "fluid patch does not exist";
            }
            else
            {
                reason = "solid patch does not exist";
            }
            mappingFailure(pair, -1, reason);
        }

        fluidPatchIds_[pairI] = fluidPatchId;
        solidPatchIds_[pairI] = solidPatchId;
        const fvPatch& fluidPatch = fluidMesh_.boundary()[fluidPatchId];
        const fvPatch& solidPatch = solidMesh_.boundary()[solidPatchId];
        rejectUnsupportedPatchType(fluidPatch.patch(), true, pair, allowMappedWall);
        rejectUnsupportedPatchType(solidPatch.patch(), false, pair, allowMappedWall);

        if (!usedFluidPatchIds.insert(fluidPatchId).second)
        {
            mappingFailure(pair, fluidPatch.size() == 0 ? -1 : 0, "duplicate fluid patch");
        }
        if (!usedSolidPatchIds.insert(solidPatchId).second)
        {
            mappingFailure(pair, solidPatch.size() == 0 ? -1 : 0, "duplicate solid patch");
        }
        forAll(fluidPatch, faceI)
        {
            if (!coveredFluidFaces.insert(fluidPatch.patch().start() + faceI).second)
            {
                mappingFailure(pair, faceI, "duplicate fluid boundary face coverage");
            }
        }
        forAll(solidPatch, faceI)
        {
            if (!coveredSolidFaces.insert(solidPatch.patch().start() + faceI).second)
            {
                mappingFailure(pair, faceI, "duplicate solid boundary face coverage");
            }
        }

        validateGeometry(fluidPatch, solidPatch, centreTolerance, pair);
    }
}

const SolidPatchPair& OneToOneConformalSolidPatchMapper::checkedPair
(
    const label pairI
) const
{
    if (pairI < 0 || pairI >= patchPairs_.size())
    {
        const SolidPatchPair invalidPair;
        mappingFailure
        (
            invalidPair,
            -1,
            "patch-pair index is outside [0," + std::to_string(patchPairs_.size()) + ")"
        );
    }
    return patchPairs_[pairI];
}

label OneToOneConformalSolidPatchMapper::size() const
{
    return patchPairs_.size();
}

const List<SolidPatchPair>&
OneToOneConformalSolidPatchMapper::patchPairs() const
{
    return patchPairs_;
}

const labelList& OneToOneConformalSolidPatchMapper::fluidPatchIds() const
{
    return fluidPatchIds_;
}

const labelList& OneToOneConformalSolidPatchMapper::solidPatchIds() const
{
    return solidPatchIds_;
}

scalarField OneToOneConformalSolidPatchMapper::mapIntegratedFaceEnergyToSolid
(
    const label pairI,
    const scalarField& fluidFaceEnergyJ
) const
{
    const SolidPatchPair& pair = checkedPair(pairI);
    validateSourceSize
    (
        fluidFaceEnergyJ,
        fluidMesh_.boundary()[fluidPatchIds_[pairI]].size(),
        pair,
        "fluid-to-solid integrated-energy"
    );

    scalarField solidFaceEnergyJ(fluidFaceEnergyJ);
    requireBitwiseCopy
    (
        fluidFaceEnergyJ,
        solidFaceEnergyJ,
        pair,
        "fluid-to-solid integrated-energy mapping"
    );

    scalar fluidTotalJ = scalar(0);
    scalar solidTotalJ = scalar(0);
    scalar sumAbsoluteJ = scalar(0);
    forAll(fluidFaceEnergyJ, faceI)
    {
        if (!finiteScalar(fluidFaceEnergyJ[faceI]))
        {
            mappingFailure(pair, faceI, "integrated face energy is nonfinite");
        }
        fluidTotalJ += fluidFaceEnergyJ[faceI];
        solidTotalJ += solidFaceEnergyJ[faceI];
        sumAbsoluteJ += mag(fluidFaceEnergyJ[faceI]);
    }
    if
    (
        !finiteScalar(fluidTotalJ)
     || !finiteScalar(solidTotalJ)
     || !finiteScalar(sumAbsoluteJ)
    )
    {
        mappingFailure(pair, -1, "integrated face-energy DP sum overflowed");
    }
    const scalar sumScale = max
    (
        max(sumAbsoluteJ, mag(fluidTotalJ)),
        max(mag(solidTotalJ), std::numeric_limits<scalar>::min())
    );
    const scalar summationTolerance =
        scalar(64)*std::numeric_limits<scalar>::epsilon()*sumScale;
    if (mag(fluidTotalJ - solidTotalJ) > summationTolerance)
    {
        mappingFailure
        (
            pair,
            -1,
            "integrated face-energy total changed beyond DP summation tolerance"
        );
    }
    return solidFaceEnergyJ;
}

scalarField OneToOneConformalSolidPatchMapper::mapSolidScalarToFluid
(
    const label pairI,
    const scalarField& solidFaceValues
) const
{
    const SolidPatchPair& pair = checkedPair(pairI);
    validateSourceSize
    (
        solidFaceValues,
        solidMesh_.boundary()[solidPatchIds_[pairI]].size(),
        pair,
        "solid-to-fluid scalar"
    );
    scalarField fluidFaceValues(solidFaceValues);
    requireBitwiseCopy
    (
        solidFaceValues,
        fluidFaceValues,
        pair,
        "solid-to-fluid scalar mapping"
    );
    return fluidFaceValues;
}

ConservativeNearestSolidPatchMapper::ConservativeNearestSolidPatchMapper
(
    const fvMesh& fluidMesh,
    const fvMesh& solidMesh,
    const List<SolidPatchPair>& patchPairs
)
:
    fluidMesh_(fluidMesh),
    solidMesh_(solidMesh),
    patchPairs_(patchPairs),
    fluidPatchIds_(patchPairs.size(), -1),
    solidPatchIds_(patchPairs.size(), -1),
    nearestSolidFaceForFluidFace_(patchPairs.size())
{
    const SolidPatchPair emptyPair;
    if (patchPairs_.empty())
    {
        mappingFailure(emptyPair, -1, "at least one nonconformal pair is required");
    }
    if
    (
        Pstream::parRun()
     || fluidMesh_.time().processorCase()
     || solidMesh_.time().processorCase()
    )
    {
        mappingFailure(patchPairs_[0], 0, "conservative nearest mapping is serial only");
    }

    std::set<label> usedFluid;
    std::set<label> usedSolid;
    forAll(patchPairs_, pairI)
    {
        const SolidPatchPair& pair = patchPairs_[pairI];
        const label fluidPatchI =
            fluidMesh_.boundary().findPatchID(pair.fluidPatch);
        const label solidPatchI =
            solidMesh_.boundary().findPatchID(pair.solidPatch);
        if (fluidPatchI < 0 || solidPatchI < 0)
        {
            mappingFailure(pair, -1, "nonconformal patch does not exist");
        }
        if
        (
            !usedFluid.insert(fluidPatchI).second
         || !usedSolid.insert(solidPatchI).second
        )
        {
            mappingFailure(pair, 0, "duplicate nonconformal patch coverage");
        }

        const fvPatch& fluidPatch = fluidMesh_.boundary()[fluidPatchI];
        const fvPatch& solidPatch = solidMesh_.boundary()[solidPatchI];
        if
        (
            !isA<wallPolyPatch>(fluidPatch.patch())
         || !isA<wallPolyPatch>(solidPatch.patch())
         || fluidPatch.size() == 0
         || solidPatch.size() == 0
        )
        {
            mappingFailure(pair, 0, "nonconformal patches must be non-empty wall patches");
        }

        fluidPatchIds_[pairI] = fluidPatchI;
        solidPatchIds_[pairI] = solidPatchI;
        labelList& nearest = nearestSolidFaceForFluidFace_[pairI];
        nearest.setSize(fluidPatch.size(), -1);
        const vectorField& fluidCentres = fluidPatch.Cf();
        const vectorField& solidCentres = solidPatch.Cf();
        const vectorField& fluidAreas = fluidPatch.Sf();
        const vectorField& solidAreas = solidPatch.Sf();
        const scalarField& fluidMagSf = fluidPatch.magSf();
        const scalarField& solidMagSf = solidPatch.magSf();

        forAll(fluidCentres, fluidFaceI)
        {
            if
            (
                !finiteVector(fluidCentres[fluidFaceI])
             || !finiteScalar(fluidMagSf[fluidFaceI])
             || fluidMagSf[fluidFaceI] <= scalar(0)
            )
            {
                mappingFailure(pair, fluidFaceI, "invalid fluid face geometry");
            }
            scalar bestDistance = std::numeric_limits<scalar>::max();
            label bestSolidFace = -1;
            forAll(solidCentres, solidFaceI)
            {
                if
                (
                    !finiteVector(solidCentres[solidFaceI])
                 || !finiteScalar(solidMagSf[solidFaceI])
                 || solidMagSf[solidFaceI] <= scalar(0)
                )
                {
                    mappingFailure(pair, solidFaceI, "invalid solid face geometry");
                }
                const scalar distance =
                    magSqr(fluidCentres[fluidFaceI] - solidCentres[solidFaceI]);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestSolidFace = solidFaceI;
                }
            }
            if (bestSolidFace < 0 || !finiteScalar(bestDistance))
            {
                mappingFailure(pair, fluidFaceI, "no finite nearest solid face");
            }
            const vector nf = fluidAreas[fluidFaceI]/fluidMagSf[fluidFaceI];
            const vector ns = solidAreas[bestSolidFace]/solidMagSf[bestSolidFace];
            if (!finiteVector(nf) || !finiteVector(ns) || (nf & ns) > scalar(-0.5))
            {
                mappingFailure(pair, fluidFaceI, "nearest face normals are not opposed");
            }
            nearest[fluidFaceI] = bestSolidFace;
        }
    }
}

const SolidPatchPair& ConservativeNearestSolidPatchMapper::checkedPair
(
    const label pairI
) const
{
    if (pairI < 0 || pairI >= patchPairs_.size())
    {
        const SolidPatchPair invalidPair;
        mappingFailure(invalidPair, -1, "nonconformal pair index is out of range");
    }
    return patchPairs_[pairI];
}

label ConservativeNearestSolidPatchMapper::size() const
{
    return patchPairs_.size();
}

const List<SolidPatchPair>&
ConservativeNearestSolidPatchMapper::patchPairs() const
{
    return patchPairs_;
}

const labelList& ConservativeNearestSolidPatchMapper::fluidPatchIds() const
{
    return fluidPatchIds_;
}

const labelList& ConservativeNearestSolidPatchMapper::solidPatchIds() const
{
    return solidPatchIds_;
}

scalarField ConservativeNearestSolidPatchMapper::mapIntegratedFaceEnergyToSolid
(
    const label pairI,
    const scalarField& fluidFaceEnergyJ
) const
{
    const SolidPatchPair& pair = checkedPair(pairI);
    validateSourceSize
    (
        fluidFaceEnergyJ,
        fluidMesh_.boundary()[fluidPatchIds_[pairI]].size(),
        pair,
        "nonconformal fluid-to-solid integrated-energy"
    );
    scalarField result
    (
        solidMesh_.boundary()[solidPatchIds_[pairI]].size(),
        scalar(0)
    );
    const labelList& nearest = nearestSolidFaceForFluidFace_[pairI];
    forAll(fluidFaceEnergyJ, fluidFaceI)
    {
        if (!finiteScalar(fluidFaceEnergyJ[fluidFaceI]))
        {
            mappingFailure(pair, fluidFaceI, "integrated face energy is nonfinite");
        }
        result[nearest[fluidFaceI]] += fluidFaceEnergyJ[fluidFaceI];
    }
    return result;
}

scalarField ConservativeNearestSolidPatchMapper::mapSolidScalarToFluid
(
    const label pairI,
    const scalarField& solidFaceValues
) const
{
    const SolidPatchPair& pair = checkedPair(pairI);
    validateSourceSize
    (
        solidFaceValues,
        solidMesh_.boundary()[solidPatchIds_[pairI]].size(),
        pair,
        "nonconformal solid-to-fluid scalar"
    );
    scalarField result(nearestSolidFaceForFluidFace_[pairI].size());
    forAll(result, fluidFaceI)
    {
        const scalar value =
            solidFaceValues[nearestSolidFaceForFluidFace_[pairI][fluidFaceI]];
        if (!finiteScalar(value))
        {
            mappingFailure(pair, fluidFaceI, "solid scalar is nonfinite");
        }
        result[fluidFaceI] = value;
    }
    return result;
}

}                        
}                  
