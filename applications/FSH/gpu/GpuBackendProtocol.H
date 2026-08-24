#ifndef UGKWP_GPU_BACKEND_PROTOCOL_H
#define UGKWP_GPU_BACKEND_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ugkwpGpuIpc
{

static constexpr std::uint32_t magic = 0x46534842U;          
static constexpr std::uint16_t protocolMajor = 6;
static constexpr std::uint16_t protocolMinor = 0;
static constexpr std::uint64_t maxPayloadBytes = 32ULL*1024ULL*1024ULL*1024ULL;

enum class Op : std::uint32_t
{
    create = 1,
    uploadMesh = 2,
    configureParticleStuckModel = 3,
    uploadBoundarySources = 4,
    configureScheduledInlet = 5,
    downloadSourceResidualMass = 6,
    uploadSourceResidualMass = 7,
    uploadGasBoundaryFields = 8,
    uploadFields = 9,
    advance = 10,
    computeGasCourant = 11,
    advanceGasOnly = 12,
    downloadFields = 13,
    downloadGasBoundaryFields = 14,
    uploadParticleRestartMirror = 15,
    downloadParticleRestartMirror = 16,
    downloadEpsGPrev = 17,
    uploadEpsGPrev = 18,
    downloadNut = 19,
    downloadAndResetParticleWallHeatLedgers = 20,
    release = 21,
    configureSst = 22,
    downloadSst = 23
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
    double gasMu;
    double gasPr;
    std::int32_t dragModelId;
    std::int32_t particleGasHeatTransferModelId;
    double dragResidualRe;
    double gravityX;
    double gravityY;
    double gravityZ;
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
};

struct CountArgs
{
    std::int32_t count;
};

struct ParticleStuckConfigArgs
{
    std::int32_t nFaces;
    std::int32_t heatTransferEnabled;
    double sommerfeldThreshold;
    double maximumCoverage;
    double depositionHeatTransferEfficiency;
    double reflectionHeatTransferEfficiency;
    double adhesionEnergyScale;
    double wallDensityKgM3;
    double wallSpecificHeatJkgK;
    double wallConductivityWmK;
    double contactAngleDegree;
    std::int32_t wallTransientResistance;
    std::int32_t nonlinearIterations;
    double meltingTemperatureK;
    double mushyRangeK;
    double latentHeatJkg;
    double solidDensityKgM3;
    double solidSpecificHeatJkgK;
    double solidThermalConductivityWmK;
    double pinningThicknessFraction;
    double interfaceResistanceM2KW;
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

static_assert(sizeof(RequestHeader) == 24, "unexpected request header ABI");
static_assert(sizeof(ResponseHeader) == 24, "unexpected response header ABI");
static_assert(sizeof(CreateArgs) == 352, "unexpected create payload ABI");
static_assert(sizeof(CountArgs) == 4, "unexpected count payload ABI");
static_assert
(
    sizeof(ParticleStuckConfigArgs) == 152,
    "unexpected particle stuck-model payload ABI"
);
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
