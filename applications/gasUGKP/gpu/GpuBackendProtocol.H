#ifndef UGKWP_GPU_BACKEND_PROTOCOL_H
#define UGKWP_GPU_BACKEND_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ugkwpGpuIpc
{

static constexpr std::uint32_t magic = 0x47554750U;          
static constexpr std::uint16_t protocolMajor = 7;
static constexpr std::uint16_t protocolMinor = 0;
static constexpr std::uint64_t maxPayloadBytes = 32ULL*1024ULL*1024ULL*1024ULL;

enum class Op : std::uint32_t
{
    create = 1,
    uploadMesh = 2,
    uploadBoundarySources = 3,
    configureScheduledInlet = 4,
    downloadSourceResidualMass = 5,
    uploadSourceResidualMass = 6,
    uploadGasBoundaryFields = 7,
    uploadFields = 8,
    advance = 9,
    computeGasCourant = 10,
    advanceGasOnly = 11,
    downloadFields = 12,
    downloadGasBoundaryFields = 13,
    uploadParticleRestartMirror = 14,
    downloadParticleRestartMirror = 15,
    downloadEpsGPrev = 16,
    uploadEpsGPrev = 17,
    downloadNut = 18,
    release = 19,
    configureSst = 20,
    downloadSst = 21,
    queryGasModelCapabilitiesV1 = 22,
    configureGasModelV1 = 23,
    uploadSpeciesV1 = 24,
    downloadSpeciesV1 = 25,
    uploadSpeciesBoundaryV1 = 26,
    configureBoundaryLayerV1 = 27,
    downloadBoundaryLayerV1 = 28
};

enum class GasFluxScheme : std::int32_t
{
    RusanovTadmor = 1,
    HllKurganov = 2,
    Hlle = 3,
    Hllc = 4,
    Roe = 5,
    Hllem = 6,
    HllcAdc = 7,
    Slau2 = 8,
    Slau2_2 = 9
};

enum class GasReconstruction : std::int32_t
{
    firstOrder = 0,
    MUSCL = 1,
    OpenFoamEnergyLimitedLinear = 2
};

enum class GasLimiter : std::int32_t
{
    none = 0,
    barthJespersen = 1,
    venkatakrishnan = 2
};

enum class GasTimeIntegrator : std::int32_t
{
    Euler = 1,
    SSPRK2 = 2,
    SSPRK3 = 3
};

enum class TurbulenceModel : std::int32_t
{
    laminar = 0,
    WALE = 1,
    Smagorinsky = 2,
    kOmegaSST = 3
};

struct RequestHeader
{
    std::uint32_t magicValue;
    std::uint16_t major;
    std::uint16_t minor;
    std::uint32_t operation;
    std::uint32_t reserved;
    std::uint64_t payloadBytes;
};

struct ResponseHeader
{
    std::uint32_t magicValue;
    std::uint16_t major;
    std::uint16_t minor;
    std::int32_t status;
    std::uint32_t errorBytes;
    std::uint64_t payloadBytes;
};

struct CreateArgs
{
    std::int32_t nCells;
    std::int32_t nFaces;
    std::int32_t nInternalFaces;
    std::int32_t nCellPlanes;
    std::int32_t particleCapacity;
    std::int32_t maxFaceWalkHops;
    double injectionParcelMass;
    std::uint64_t rngSeed;
    double gammaGas;
    double Rgas;
    double rhoSolid;
    std::int32_t solveParticleTemperature;
    std::int32_t particleGasHeatTransferModelId;
    double particleThermalRho;
    double particleCp;
    double gasMu;
    double gasPr;
    double particleDiameterFallback;
    double particleDiameterMin;
    double particleDiameterMax;
    double particleDiameterSigma;
    double injectionTheta;
    double rhoMin;
    double TgasMin;
    double epsSMin;
    double thetaMin;
    double TpMin;
    double TpMax;
    std::int32_t collisionalPressureEnabled;
    double collisionalRestitution;
    double pressureKickFraction;
    std::int32_t jammingPressureEnabled;
    double packingFraction;
    std::int32_t packingProjectionIterations;
    std::int32_t gasFluxScheme;
    std::int32_t gasReconstruction;
    std::int32_t gasLimiter;
    std::int32_t gasTimeIntegrator;
    std::int32_t gasRobustFallback;
    std::int32_t turbulenceModel;
    double lesDeltaCoeff;
    double turbulentPrandtl;
    double waleCw;
    double smagorinskyCs;
    double maxDiffusionNumber;
    std::int32_t csrCellLocalPathEnabled;
    std::int32_t csrHeavyReductionMode;
    std::int32_t csrHeavyAutoInterval;
    std::int32_t particleBlockThreads;
    std::int32_t reductionBlockThreads;
    std::int32_t csrWarpAggregatedBinning;
    std::int32_t csrSplitPreDirectoryEnabled;
    std::int32_t dragModel;
    double dragParameter0;
    double dragParameter1;
    double dragParameter2;
    double dragParameter3;
    double gravityX;
    double gravityY;
    double gravityZ;
};

struct CountArgs
{
    std::int32_t count;
};

struct ScheduledInletConfigArgs
{
    std::int32_t count;
    std::int32_t nPressureRows;
    std::int32_t nVolumeFractionRows;
    double inletTemperature;
};

struct AdvanceArgs
{
    double dt;
    double simulationTime;
};

struct ScalarArgs
{
    double value;
};

struct CourantArgs
{
    double dt;
    double targetMaxCo;
    double scheduleTime;
};

struct SstConfigArgs
{
    double alphaK1;
    double alphaK2;
    double alphaOmega1;
    double alphaOmega2;
    double beta1;
    double beta2;
    double betaStar;
    double gamma1;
    double gamma2;
    double a1;
    double b1;
    double c1;
    double kMin;
    double omegaMin;
    double maxSourceNumber;
    std::int32_t wallTreatment;
    double wallKappa;
    double wallE;
    double wallCmu;
};

// Optional extension operations leave every v7 legacy payload unchanged.
// These wire objects contain fixed-width scalar fields only, with no padding;
// model tables cross IPC as bounded canonical dictionary bytes, never pointers.
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
    return a.version == gasModelApiVersion && b.version == a.version
        && a.speciesCount == b.speciesCount && a.speciesOrderHash == b.speciesOrderHash
        && a.thermoHash == b.thermoHash && a.mechanismHash == b.mechanismHash;
}

inline bool validCount(const std::int32_t count, const std::int32_t limit)
{
    return count >= 0 && count <= limit;
}

inline bool addBytes(std::uint64_t& total, const std::uint64_t count,
                     const std::uint64_t elementBytes)
{
    if (count != 0 && elementBytes > maxPayloadBytes/count)
    {
        return false;
    }
    const std::uint64_t bytes = count*elementBytes;
    if (bytes > maxPayloadBytes - total)
    {
        return false;
    }
    total += bytes;
    return true;
}

template<class T>
inline bool addArrayBytes(std::uint64_t& total, const std::uint64_t count)
{
    return addBytes(total, count, sizeof(T));
}

inline bool gasModelPayloadBytes(const GasModelConfigureArgsV1& args,
                                 std::uint32_t compiledSpecies, std::uint64_t& bytes)
{
    if (args.version != gasModelApiVersion || args.reserved != 0
        || args.speciesCount == 0 || args.speciesCount != compiledSpecies
        || (args.mode != 1 && args.mode != 2) || args.speciesOrderHash == 0 || args.thermoHash == 0
        || args.modelBytes == 0 || args.modelBytes > maxGasModelTextBytes
        || args.mechanismBytes > maxGasModelTextBytes
        || (args.mode == 1 && (args.mechanismBytes != 0 || args.mechanismHash != 0))
        || (args.mode == 2 && (args.mechanismBytes == 0 || args.mechanismHash == 0))) return false;
    bytes = sizeof(GasModelConfigureArgsV1);
    return addBytes(bytes, args.modelBytes, 1) && addBytes(bytes, args.mechanismBytes, 1);
}
inline bool gasSpeciesPayloadBytes(const GasSpeciesIdentityV1& identity,
                                   std::uint64_t cells, std::uint64_t& bytes)
{
    if (identity.version != gasModelApiVersion || identity.speciesCount == 0
        || identity.speciesOrderHash == 0 || identity.thermoHash == 0
        || cells > maxPayloadBytes/identity.speciesCount/sizeof(double)) return false;
    bytes = sizeof(GasSpeciesIdentityV1);
    return addArrayBytes<double>(bytes, cells*identity.speciesCount);
}
// Optional wall-family extension; every pre-existing wire layout is unchanged.
struct BoundaryLayerConfigV1
{
    std::uint32_t version=1,speciesCount=0,model=1,nodes=24,maxIterations=60;
    std::uint32_t wallCount=0,quadratureCount=0,matchingCount=0,workspaceSlots=0,budgetAudit=0;
    double relativeTolerance=1e-8,absoluteTolerance=1e-10,stretch=2;
    double turbulentPrandtl=.9,turbulentSchmidt=.7;
};
static constexpr std::uint32_t boundaryLayerDiagnosticScalars=25;
inline bool boundaryLayerPayloadBytes(const BoundaryLayerConfigV1& a,
    std::uint32_t faces,std::uint32_t species,std::uint64_t& bytes)
{
    if(a.version!=1 || a.budgetAudit>1 || !a.speciesCount || a.speciesCount!=species
        || a.model>1 || a.nodes<4 || a.nodes>128 || !a.maxIterations
        || !a.wallCount || a.wallCount>faces || a.workspaceSlots>4096
        || a.quadratureCount<a.wallCount || a.matchingCount<a.wallCount
        || a.quadratureCount>std::uint32_t(std::numeric_limits<int>::max())
        || a.matchingCount>std::uint32_t(std::numeric_limits<int>::max()))return false;
    bytes=sizeof(a);
    return addArrayBytes<int>(bytes,std::uint64_t(a.wallCount)+2ULL*(a.wallCount+1ULL)+a.matchingCount)
        && addArrayBytes<double>(bytes,7ULL*a.wallCount+2ULL*a.quadratureCount+a.matchingCount
            +std::uint64_t(a.wallCount)*a.speciesCount);
}
static_assert(sizeof(BoundaryLayerConfigV1)==80,"unexpected wall config ABI");
static_assert(sizeof(GasModelCapabilitiesV1) == 16, "unexpected model capability wire padding");
static_assert(sizeof(GasModelConfigureArgsV1) == 56, "unexpected model configure wire padding");
static_assert(sizeof(GasSpeciesIdentityV1) == 32, "unexpected species identity wire padding");

static_assert(sizeof(RequestHeader) == 24, "unexpected request header ABI");
static_assert(sizeof(ResponseHeader) == 24, "unexpected response header ABI");
static_assert(sizeof(CreateArgs) == 392, "unexpected create payload ABI");
static_assert(sizeof(CountArgs) == 4, "unexpected count payload ABI");
static_assert
(
    sizeof(ScheduledInletConfigArgs) == 24,
    "unexpected inlet configuration payload ABI"
);
static_assert(sizeof(AdvanceArgs) == 16, "unexpected advance payload ABI");
static_assert(sizeof(ScalarArgs) == 8, "unexpected scalar payload ABI");
static_assert(sizeof(CourantArgs) == 24, "unexpected Courant payload ABI");
static_assert(sizeof(SstConfigArgs) == 152, "unexpected SST config ABI");

}                         

#endif
