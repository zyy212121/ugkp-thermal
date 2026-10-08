#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"
#include "fixedGradientFvPatchFields.H"
#include "GpuSolidThermalCoupler.H"
#include "DynamicList.H"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace Foam;
using namespace Foam::gpuThermal;

namespace Foam
{
namespace gpuThermal
{

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void near(scalar actual, scalar expected)
{
    require(std::isfinite(actual) && std::abs(actual-expected) <= 1e-12*std::max(scalar(1),std::abs(expected)), "mapped value differs from accepted surface");
}
bool finiteScalar(scalar value) { return std::isfinite(value); }
[[noreturn]] void solidFailure(const std::string& reason) { throw GpuSolidThermalSolveError(reason); }
#include "PatchMembership.inc"

label surfaceReads = 0;
scalarField recordedSurfaceTemperature(const volScalarField& temperature, label patchI)
{
    ++surfaceReads;
    return coupledPatchSurfaceTemperature(temperature, patchI);
}

// The actual mapping method is CUDA-build-only. The host harness substitutes
// only its mapper and GPU upload boundary; all temperature/property operations
// and the production method body are compiled unchanged.
struct Mapper
{
    labelList patches;
    explicit Mapper(const labelList& ids) : patches(ids) {}
    const labelList& fluidPatchIds() const { return patches; }
    const labelList& solidPatchIds() const { return patches; }
    scalarField mapSolidScalarToFluid(label, const scalarField& values) const
    {
        scalarField mapped(values);
        std::reverse(mapped.begin(), mapped.end());
        return mapped;
    }
};
struct Resident
{
    List<scalarField> temperatures;
    List<scalarField> effusivities;
    label uploads = 0;
    void updateSolidWallTemperatures(const fvMesh& mesh, const volScalarField& gas, const labelList& ids)
    {
        require(uploads == 0, "temperature upload order changed");
        temperatures.setSize(mesh.boundary().size());
        forAll(ids, i) temperatures[ids[i]] = gas.boundaryField()[ids[i]];
        ++uploads;
    }
    void updateParticleWallEffusivities(const fvMesh&, const List<scalarField>& values, const labelList&)
    {
        require(uploads == 1, "effusivity upload precedes temperature upload");
        effusivities = values;
        ++uploads;
    }
};
struct MappingHarness
{
    const fvMesh& fluidMesh;
    volScalarField& solid;
    Mapper* mapper;
    const GpuSolidThermalProperties* properties;
    Resident resident;
    labelList particleContactPairIds;
    labelList particleContactFluidPatchIds;
    MappingHarness(const fvMesh& mesh, volScalarField& temperature, Mapper& mapping,
                   const GpuSolidThermalProperties& material)
    : fluidMesh(mesh), solid(temperature), mapper(&mapping), properties(&material) {}
    const volScalarField& Tsolid() const { return solid; }
#define coupledPatchSurfaceTemperature recordedSurfaceTemperature
#include "WallMapping.inc"
#undef coupledPatchSurfaceTemperature
};

void testSnapshots(const volScalarField& solid, const labelList& coupledPatchIds_)
{
    // Empty owner set, mixed gas/owner set, and all-owner auxiliary solid.
    for (label ownerCount=0; ownerCount<=coupledPatchIds_.size(); ++ownerCount)
    {
        labelList ownerConductivityPatchIds_(ownerCount);
        forAll(ownerConductivityPatchIds_, i) ownerConductivityPatchIds_[i]=coupledPatchIds_[i];
        autoPtr<volScalarField> candidate(new volScalarField(solid));
#include "SurfaceSnapshot.inc"
        forAll(coupledPatchIds_, i)
        {
            if (i < ownerCount)
            {
                require(previousSurface[i].empty(), "unused owner-conductivity surface was copied");
            }
            else
            {
                const scalarField& expected=solid.boundaryField()[coupledPatchIds_[i]];
                require(previousSurface[i].size()==expected.size(), "gas surface snapshot missing");
                forAll(expected, f) near(previousSurface[i][f], expected[f]);
            }
        }
    }
}

void testBoundaryProperties(const volScalarField& solid, const labelList& patches,
                            const GpuSolidThermalProperties& properties_)
{
    const fvMesh& mesh_=solid.mesh();
    for (label ownerCount=0; ownerCount<=patches.size(); ++ownerCount)
    {
        labelList ownerConductivityPatchIds_(ownerCount);
        forAll(ownerConductivityPatchIds_,i) ownerConductivityPatchIds_[i]=patches[i];
        autoPtr<volScalarField> candidate(new volScalarField(solid));
        volScalarField Csec(solid), kappa(solid);
        label ownerGathers=0;
#include "BoundaryProperties.inc"
        require(ownerGathers==ownerCount, "unused owner temperature was gathered for a surface-property patch");
        forAll(patches,i)
        {
            const label patchI=patches[i];
            forAll(solid.boundaryField()[patchI],f)
            {
                const scalar T=i<ownerCount?300:solid.boundaryField()[patchI][f];
                near(kappa.boundaryField()[patchI][f],0.02*T);
                near(Csec.boundaryField()[patchI][f],1000);
            }
        }
    }
}
}
}

int main(int argc, char** argv)
{
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject(fvMesh::defaultRegion, runTime.timeName(), runTime, IOobject::MUST_READ));
    IOdictionary config(IOobject("testProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    GpuSolidThermalProperties properties(config.subDict("properties"));
    volScalarField solid(IOobject("solid",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,
                         dimensionedScalar("T",dimTemperature,300),wordList(mesh.boundary().size(),"fixedGradient"));
    volScalarField gas(IOobject("gas",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,
                       dimensionedScalar("T",dimTemperature,100),wordList(mesh.boundary().size(),"fixedValue"));
    labelList pairs(3);
    pairs[0]=mesh.boundaryMesh().findPatchID("right");
    pairs[1]=mesh.boundaryMesh().findPatchID("left");
    pairs[2]=mesh.boundaryMesh().findPatchID("sides");
    forAll(mesh.boundary(),p)
    {
        refCast<fixedGradientFvPatchScalarField>(solid.boundaryFieldRef()[p]).gradient()=20;
    }
    solid.correctBoundaryConditions();
    const word mode(config.lookup("testMode"));
    if (mode == "snapshot")
    {
        testSnapshots(solid,pairs);
        std::cout << "PASS snapshot\n";
        return 0;
    }
    if (mode == "properties")
    {
        testBoundaryProperties(solid,pairs,properties);
        std::cout << "PASS properties\n";
        return 0;
    }
    Mapper mapper(pairs);
    MappingHarness harness(mesh,solid,mapper,properties);
    // Include disabled contact, sparse/non-leading contact, and all-contact.
    for (unsigned mask : {0u,1u,2u,5u,7u})
    {
        DynamicList<label> contactPairs, contactPatches;
        forAll(pairs, i) if (mask & (1u<<i))
        {
            contactPairs.append(i);
            contactPatches.append(pairs[i]);
        }
        harness.particleContactPairIds=contactPairs;
        harness.particleContactFluidPatchIds=contactPatches;
        for (label invocation=0; invocation<2; ++invocation)
        {
            solid.primitiveFieldRef()=300+20*invocation;
            forAll(mesh.boundary(), p)
            {
                scalarField& gradient=refCast<fixedGradientFvPatchScalarField>(solid.boundaryFieldRef()[p]).gradient();
                forAll(gradient,f) gradient[f]=10*(1+p+f+invocation);
            }
            solid.correctBoundaryConditions();
            surfaceReads=0;
            harness.resident=Resident();
            harness.mapSolidWallTemperatureToFluid(gas);
            require(surfaceReads==pairs.size(), "accepted surface was gathered or validated more than once per mapping");
            require(harness.resident.uploads==(mask?2:1), "unexpected GPU upload count");
            forAll(pairs,i)
            {
                const scalarField& surface=solid.boundaryField()[pairs[i]];
                forAll(surface,f)
                {
                    const scalar Tw=surface[surface.size()-1-f];
                    near(harness.resident.temperatures[pairs[i]][f],Tw);
                    if (mask & (1u<<i))
                    {
                        // Fixture rho=1000, Cp=1, kappa(T)=0.02*T.
                        near(harness.resident.effusivities[pairs[i]][f],std::sqrt(20*Tw));
                    }
                }
            }
        }
    }
    // First-pass validation still rejects stale accepted state before upload.
    solid.boundaryFieldRef()[pairs[2]][0]+=1;
    harness.resident=Resident();
    bool rejected=false;
    try { harness.mapSolidWallTemperatureToFluid(gas); }
    catch (const GpuSolidThermalSolveError&) { rejected=true; }
    require(rejected && harness.resident.uploads==0, "invalid surface reached GPU upload");
    std::cout << "PASS mapping\n";
}
