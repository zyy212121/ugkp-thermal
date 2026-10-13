#include "GpuSolidPatchMapper.H"

#include "Pstream.H"
#include "cyclicAMIPolyPatch.H"
#include "cyclicPolyPatch.H"
#include "mappedWallPolyPatch.H"
#include "wallPolyPatch.H"

#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

namespace Foam
{
namespace gpuThermal
{
namespace
{
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

void requireLocalCorrespondence
(
    const fvPatch& fluidPatch,
    const fvPatch& solidPatch,
    const SolidPatchPair& pair
)
{
    if (fluidPatch.size() != solidPatch.size())
    {
        mappingFailure(pair, min(fluidPatch.size(), solidPatch.size()),
            "local-index map is missing a corresponding face");
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
    if (source.size() < expectedSize)
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
    (void)allowMappedWall;
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
        requireLocalCorrespondence(fluidPatch, solidPatch, pair);
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

    const label count = solidMesh_.boundary()[solidPatchIds_[pairI]].size();
    scalarField solidFaceEnergyJ(count);
    std::copy_n(fluidFaceEnergyJ.begin(), count, solidFaceEnergyJ.begin());
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
    const label count = fluidMesh_.boundary()[fluidPatchIds_[pairI]].size();
    scalarField fluidFaceValues(count);
    std::copy_n(solidFaceValues.begin(), count, fluidFaceValues.begin());
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
        const fvPatch& fluidPatch = fluidMesh_.boundary()[fluidPatchI];
        const fvPatch& solidPatch = solidMesh_.boundary()[solidPatchI];
        fluidPatchIds_[pairI] = fluidPatchI;
        solidPatchIds_[pairI] = solidPatchI;
        labelList& nearest = nearestSolidFaceForFluidFace_[pairI];
        nearest.setSize(fluidPatch.size(), -1);
        const vectorField& fluidCentres = fluidPatch.Cf();
        const vectorField& solidCentres = solidPatch.Cf();
        forAll(fluidCentres, fluidFaceI)
        {
            scalar bestDistance = std::numeric_limits<scalar>::max();
            label bestSolidFace = -1;
            forAll(solidCentres, solidFaceI)
            {
                const scalar distance =
                    magSqr(fluidCentres[fluidFaceI] - solidCentres[solidFaceI]);
                if (finiteScalar(distance)
                 && (bestSolidFace < 0 || distance < bestDistance))
                {
                    bestDistance = distance;
                    bestSolidFace = solidFaceI;
                }
            }
            if (bestSolidFace < 0 && !solidCentres.empty())
            {
                // Only the formerly refused path needs a scaled comparison.
                // A shared scale preserves ordering among huge finite centres;
                // subtract first whenever possible to retain small differences.
                scalar scale = scalar(0);
                for (direction component = 0; component < vector::nComponents; ++component)
                {
                    scale = std::fmax(scale, mag(fluidCentres[fluidFaceI][component]));
                }
                forAll(solidCentres, solidFaceI)
                {
                    for (direction component = 0; component < vector::nComponents; ++component)
                    {
                        scale = std::fmax(scale, mag(solidCentres[solidFaceI][component]));
                    }
                }
                forAll(solidCentres, solidFaceI)
                {
                    scalar scaledDistance = scalar(0);
                    for (direction component = 0; component < vector::nComponents; ++component)
                    {
                        const scalar fluid = fluidCentres[fluidFaceI][component];
                        const scalar solid = solidCentres[solidFaceI][component];
                        const scalar difference = fluid - solid;
                        const scalar scaledDifference = std::isfinite(difference)
                          ? difference/scale : fluid/scale - solid/scale;
                        scaledDistance += scaledDifference*scaledDifference;
                    }
                    if
                    (
                        bestSolidFace < 0
                     || (std::isfinite(scaledDistance) && !std::isfinite(bestDistance))
                     || scaledDistance < bestDistance
                    )
                    {
                        bestDistance = scaledDistance;
                        bestSolidFace = solidFaceI;
                    }
                }
            }
            if (bestSolidFace < 0)
            {
                mappingFailure(pair, fluidFaceI, "no available nearest solid face");
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
    forAll(nearest, fluidFaceI)
    {
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
    checkedPair(pairI);
    scalarField result(nearestSolidFaceForFluidFace_[pairI].size());
    forAll(result, fluidFaceI)
    {
        const scalar value =
            solidFaceValues[nearestSolidFaceForFluidFace_[pairI][fluidFaceI]];

        result[fluidFaceI] = value;
    }
    return result;
}

}
}
