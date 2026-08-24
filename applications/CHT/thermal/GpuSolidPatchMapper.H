#ifndef GPU_THERMAL_GPU_SOLID_PATCH_MAPPER_H
#define GPU_THERMAL_GPU_SOLID_PATCH_MAPPER_H

#include "List.H"
#include "fvMesh.H"
#include "labelList.H"
#include "scalarField.H"
#include "word.H"

#include <stdexcept>
#include <string>

namespace Foam
{
namespace gpuThermal
{

class SolidPatchMappingError : public std::runtime_error
{
public:
    explicit SolidPatchMappingError(const std::string& message)
    :
        std::runtime_error(message)
    {}
};

struct SolidPatchPair
{
    word fluidPatch;
    word solidPatch;
};

class GpuSolidPatchMapper
{
public:
    virtual ~GpuSolidPatchMapper() = default;

    virtual label size() const = 0;
    virtual const List<SolidPatchPair>& patchPairs() const = 0;
    virtual const labelList& fluidPatchIds() const = 0;
    virtual const labelList& solidPatchIds() const = 0;

    virtual scalarField mapIntegratedFaceEnergyToSolid
    (
        label pairI,
        const scalarField& fluidFaceEnergyJ
    ) const = 0;

    virtual scalarField mapSolidScalarToFluid
    (
        label pairI,
        const scalarField& solidFaceValues
    ) const = 0;
};

class OneToOneConformalSolidPatchMapper final
:
    public GpuSolidPatchMapper
{
    const fvMesh& fluidMesh_;
    const fvMesh& solidMesh_;
    const List<SolidPatchPair> patchPairs_;
    labelList fluidPatchIds_;
    labelList solidPatchIds_;

    const SolidPatchPair& checkedPair(label pairI) const;

public:
    OneToOneConformalSolidPatchMapper
    (
        const fvMesh& fluidMesh,
        const fvMesh& solidMesh,
        const List<SolidPatchPair>& patchPairs,
        bool allowMappedWall = false
    );

    label size() const override;
    const List<SolidPatchPair>& patchPairs() const override;
    const labelList& fluidPatchIds() const override;
    const labelList& solidPatchIds() const override;

    scalarField mapIntegratedFaceEnergyToSolid
    (
        label pairI,
        const scalarField& fluidFaceEnergyJ
    ) const override;

    scalarField mapSolidScalarToFluid
    (
        label pairI,
        const scalarField& solidFaceValues
    ) const override;
};

class ConservativeNearestSolidPatchMapper final
:
    public GpuSolidPatchMapper
{
    const fvMesh& fluidMesh_;
    const fvMesh& solidMesh_;
    const List<SolidPatchPair> patchPairs_;
    labelList fluidPatchIds_;
    labelList solidPatchIds_;
    List<labelList> nearestSolidFaceForFluidFace_;

    const SolidPatchPair& checkedPair(label pairI) const;

public:
    ConservativeNearestSolidPatchMapper
    (
        const fvMesh& fluidMesh,
        const fvMesh& solidMesh,
        const List<SolidPatchPair>& patchPairs
    );

    label size() const override;
    const List<SolidPatchPair>& patchPairs() const override;
    const labelList& fluidPatchIds() const override;
    const labelList& solidPatchIds() const override;

    scalarField mapIntegratedFaceEnergyToSolid
    (
        label pairI,
        const scalarField& fluidFaceEnergyJ
    ) const override;

    scalarField mapSolidScalarToFluid
    (
        label pairI,
        const scalarField& solidFaceValues
    ) const override;
};

}                        
}                  

#endif
