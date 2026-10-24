#ifndef UGKWP_GPU_BACKEND_TYPES_H
#define UGKWP_GPU_BACKEND_TYPES_H
#include <cstdint>
#include "../../../common/gasWall/WallResolutionDefaults.H"
// Typed in-process boundary between OpenFOAM and the resident CUDA owner.
namespace ugkwpGpu {
static constexpr std::uint32_t gasModelApiVersion = 1;
static constexpr std::uint64_t maxGasModelTextBytes = 8ULL*1024ULL*1024ULL;
struct GasModelCapabilitiesV1
{
    std::uint32_t version;
    std::uint32_t compiledSpecies;
    std::uint32_t modeMask;
    std::uint32_t reserved;
};
struct GasModelConfigureArgsV1
{
    std::uint32_t version;
    std::uint32_t mode;
    std::uint32_t speciesCount;
    std::uint32_t reserved;
    std::uint64_t speciesOrderHash;
    std::uint64_t thermoHash;
    std::uint64_t mechanismHash;
    std::uint64_t modelBytes;
    std::uint64_t mechanismBytes;
};
struct GasSpeciesIdentityV1
{
    std::uint32_t version;
    std::uint32_t speciesCount;
    std::uint64_t speciesOrderHash;
    std::uint64_t thermoHash;
    std::uint64_t mechanismHash;
};
inline bool sameGasSpeciesIdentity(const GasSpeciesIdentityV1& a, const GasSpeciesIdentityV1& b)
{
    return a.version==b.version && a.speciesCount==b.speciesCount
        && a.speciesOrderHash==b.speciesOrderHash && a.thermoHash==b.thermoHash
        && a.mechanismHash==b.mechanismHash;
}

struct BoundaryLayerConfigV1
{
    std::uint32_t version=1,speciesCount=0,model=1,nodes=ugkwp::gaswall::defaultWallNodes,maxIterations=60;
    std::uint32_t wallCount=0,quadratureCount=0,matchingCount=0,workspaceSlots=0,reservedAudit=0;
    double relativeTolerance=1e-8,absoluteTolerance=1e-10,stretch=ugkwp::gaswall::defaultWallStretch;
    double turbulentPrandtl=.9,turbulentSchmidt=.7;
};


static constexpr std::uint32_t boundaryLayerDiagnosticScalars=25;
}
#endif
