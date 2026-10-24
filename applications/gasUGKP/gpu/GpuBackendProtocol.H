#ifndef UGKWP_GPU_BACKEND_PROTOCOL_H
#define UGKWP_GPU_BACKEND_PROTOCOL_H

#include <cstddef>
#include "../../../common/gasWall/WallResolutionDefaults.H"
#include <cstdint>
#include <limits>
#include <array>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace ugkwpGpuIpc
{

static constexpr std::uint32_t magic = 0x47554750U;          
static constexpr std::uint16_t protocolMajor = 9;
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
    downloadBoundaryLayerV1 = 28,
    enableBoundaryLayerProfilesV1 = 29,
    downloadBoundaryLayerProfilesV1 = 30
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
    std::uint64_t diagnosticSequence = 0;
    std::int64_t diagnosticStep = -1;
    double diagnosticTime = 0;
};

struct ResponseHeader
{
    std::uint32_t magicValue;
    std::uint16_t major;
    std::uint16_t minor;
    std::int32_t status;
    std::uint32_t errorBytes;
    std::uint64_t payloadBytes;
    std::uint64_t diagnosticBytes;
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
    std::int32_t wallTreatment;
    double wallKappa;
    double wallE;
    double wallCmu;
};

// Optional extension operations leave every v7 legacy payload unchanged.
// These wire objects contain fixed-width scalar fields only, with no padding;

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


inline bool wireLittleEndian()
{ const std::uint16_t one=1; return *reinterpret_cast<const unsigned char*>(&one)==1; }
template<class T> inline void encodeWireScalar(unsigned char* bytes,const T& value)
{
    std::memcpy(bytes,&value,sizeof(T));
    if (!wireLittleEndian()) std::reverse(bytes,bytes+sizeof(T));
}
template<class T> inline void decodeWireScalar(const unsigned char* bytes,T& value)
{
    unsigned char storage[sizeof(T)]; std::memcpy(storage,bytes,sizeof(T));
    if (!wireLittleEndian()) std::reverse(storage,storage+sizeof(T));
    std::memcpy(&value,storage,sizeof(T));
}
template<class T> struct WireCodec
{
    static constexpr std::size_t size=sizeof(T);
    static constexpr bool nativeLayout=true;
    static void encode(unsigned char* b,const T& v) { encodeWireScalar(b,v); }
    static void decode(const unsigned char* b,T& v) { decodeWireScalar(b,v); }
};
template<> struct WireCodec<RequestHeader>
{
    static constexpr std::size_t size=48;
    static constexpr bool nativeLayout=sizeof(RequestHeader)==size
        && offsetof(RequestHeader,magicValue)==0
        && offsetof(RequestHeader,major)==4
        && offsetof(RequestHeader,minor)==6
        && offsetof(RequestHeader,operation)==8
        && offsetof(RequestHeader,reserved)==12
        && offsetof(RequestHeader,payloadBytes)==16
        && offsetof(RequestHeader,diagnosticSequence)==24
        && offsetof(RequestHeader,diagnosticStep)==32
        && offsetof(RequestHeader,diagnosticTime)==40;
    static void encode(unsigned char* b,const RequestHeader& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.magicValue);
        encodeWireScalar(b+4,v.major);
        encodeWireScalar(b+6,v.minor);
        encodeWireScalar(b+8,v.operation);
        encodeWireScalar(b+12,v.reserved);
        encodeWireScalar(b+16,v.payloadBytes);
        encodeWireScalar(b+24,v.diagnosticSequence);
        encodeWireScalar(b+32,v.diagnosticStep);
        encodeWireScalar(b+40,v.diagnosticTime);
    }
    static void decode(const unsigned char* b,RequestHeader& v)
    {
        decodeWireScalar(b+0,v.magicValue);
        decodeWireScalar(b+4,v.major);
        decodeWireScalar(b+6,v.minor);
        decodeWireScalar(b+8,v.operation);
        decodeWireScalar(b+12,v.reserved);
        decodeWireScalar(b+16,v.payloadBytes);
        decodeWireScalar(b+24,v.diagnosticSequence);
        decodeWireScalar(b+32,v.diagnosticStep);
        decodeWireScalar(b+40,v.diagnosticTime);
    }
};
template<> struct WireCodec<ResponseHeader>
{
    static constexpr std::size_t size=32;
    static constexpr bool nativeLayout=sizeof(ResponseHeader)==size
        && offsetof(ResponseHeader,magicValue)==0
        && offsetof(ResponseHeader,major)==4
        && offsetof(ResponseHeader,minor)==6
        && offsetof(ResponseHeader,status)==8
        && offsetof(ResponseHeader,errorBytes)==12
        && offsetof(ResponseHeader,payloadBytes)==16
        && offsetof(ResponseHeader,diagnosticBytes)==24;
    static void encode(unsigned char* b,const ResponseHeader& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.magicValue);
        encodeWireScalar(b+4,v.major);
        encodeWireScalar(b+6,v.minor);
        encodeWireScalar(b+8,v.status);
        encodeWireScalar(b+12,v.errorBytes);
        encodeWireScalar(b+16,v.payloadBytes);
        encodeWireScalar(b+24,v.diagnosticBytes);
    }
    static void decode(const unsigned char* b,ResponseHeader& v)
    {
        decodeWireScalar(b+0,v.magicValue);
        decodeWireScalar(b+4,v.major);
        decodeWireScalar(b+6,v.minor);
        decodeWireScalar(b+8,v.status);
        decodeWireScalar(b+12,v.errorBytes);
        decodeWireScalar(b+16,v.payloadBytes);
        decodeWireScalar(b+24,v.diagnosticBytes);
    }
};
template<> struct WireCodec<CreateArgs>
{
    static constexpr std::size_t size=384;
    static constexpr bool nativeLayout=sizeof(CreateArgs)==size
        && offsetof(CreateArgs,nCells)==0
        && offsetof(CreateArgs,nFaces)==4
        && offsetof(CreateArgs,nInternalFaces)==8
        && offsetof(CreateArgs,nCellPlanes)==12
        && offsetof(CreateArgs,particleCapacity)==16
        && offsetof(CreateArgs,maxFaceWalkHops)==20
        && offsetof(CreateArgs,injectionParcelMass)==24
        && offsetof(CreateArgs,rngSeed)==32
        && offsetof(CreateArgs,gammaGas)==40
        && offsetof(CreateArgs,Rgas)==48
        && offsetof(CreateArgs,rhoSolid)==56
        && offsetof(CreateArgs,solveParticleTemperature)==64
        && offsetof(CreateArgs,particleGasHeatTransferModelId)==68
        && offsetof(CreateArgs,particleThermalRho)==72
        && offsetof(CreateArgs,particleCp)==80
        && offsetof(CreateArgs,gasMu)==88
        && offsetof(CreateArgs,gasPr)==96
        && offsetof(CreateArgs,particleDiameterFallback)==104
        && offsetof(CreateArgs,particleDiameterMin)==112
        && offsetof(CreateArgs,particleDiameterMax)==120
        && offsetof(CreateArgs,particleDiameterSigma)==128
        && offsetof(CreateArgs,injectionTheta)==136
        && offsetof(CreateArgs,rhoMin)==144
        && offsetof(CreateArgs,TgasMin)==152
        && offsetof(CreateArgs,epsSMin)==160
        && offsetof(CreateArgs,thetaMin)==168
        && offsetof(CreateArgs,TpMin)==176
        && offsetof(CreateArgs,TpMax)==184
        && offsetof(CreateArgs,collisionalPressureEnabled)==192
        && offsetof(CreateArgs,collisionalRestitution)==200
        && offsetof(CreateArgs,pressureKickFraction)==208
        && offsetof(CreateArgs,jammingPressureEnabled)==216
        && offsetof(CreateArgs,packingFraction)==224
        && offsetof(CreateArgs,packingProjectionIterations)==232
        && offsetof(CreateArgs,gasFluxScheme)==236
        && offsetof(CreateArgs,gasReconstruction)==240
        && offsetof(CreateArgs,gasLimiter)==244
        && offsetof(CreateArgs,gasTimeIntegrator)==248
        && offsetof(CreateArgs,gasRobustFallback)==252
        && offsetof(CreateArgs,turbulenceModel)==256
        && offsetof(CreateArgs,lesDeltaCoeff)==264
        && offsetof(CreateArgs,turbulentPrandtl)==272
        && offsetof(CreateArgs,waleCw)==280
        && offsetof(CreateArgs,smagorinskyCs)==288
        && offsetof(CreateArgs,csrCellLocalPathEnabled)==296
        && offsetof(CreateArgs,csrHeavyReductionMode)==300
        && offsetof(CreateArgs,csrHeavyAutoInterval)==304
        && offsetof(CreateArgs,particleBlockThreads)==308
        && offsetof(CreateArgs,reductionBlockThreads)==312
        && offsetof(CreateArgs,csrWarpAggregatedBinning)==316
        && offsetof(CreateArgs,csrSplitPreDirectoryEnabled)==320
        && offsetof(CreateArgs,dragModel)==324
        && offsetof(CreateArgs,dragParameter0)==328
        && offsetof(CreateArgs,dragParameter1)==336
        && offsetof(CreateArgs,dragParameter2)==344
        && offsetof(CreateArgs,dragParameter3)==352
        && offsetof(CreateArgs,gravityX)==360
        && offsetof(CreateArgs,gravityY)==368
        && offsetof(CreateArgs,gravityZ)==376;
    static void encode(unsigned char* b,const CreateArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.nCells);
        encodeWireScalar(b+4,v.nFaces);
        encodeWireScalar(b+8,v.nInternalFaces);
        encodeWireScalar(b+12,v.nCellPlanes);
        encodeWireScalar(b+16,v.particleCapacity);
        encodeWireScalar(b+20,v.maxFaceWalkHops);
        encodeWireScalar(b+24,v.injectionParcelMass);
        encodeWireScalar(b+32,v.rngSeed);
        encodeWireScalar(b+40,v.gammaGas);
        encodeWireScalar(b+48,v.Rgas);
        encodeWireScalar(b+56,v.rhoSolid);
        encodeWireScalar(b+64,v.solveParticleTemperature);
        encodeWireScalar(b+68,v.particleGasHeatTransferModelId);
        encodeWireScalar(b+72,v.particleThermalRho);
        encodeWireScalar(b+80,v.particleCp);
        encodeWireScalar(b+88,v.gasMu);
        encodeWireScalar(b+96,v.gasPr);
        encodeWireScalar(b+104,v.particleDiameterFallback);
        encodeWireScalar(b+112,v.particleDiameterMin);
        encodeWireScalar(b+120,v.particleDiameterMax);
        encodeWireScalar(b+128,v.particleDiameterSigma);
        encodeWireScalar(b+136,v.injectionTheta);
        encodeWireScalar(b+144,v.rhoMin);
        encodeWireScalar(b+152,v.TgasMin);
        encodeWireScalar(b+160,v.epsSMin);
        encodeWireScalar(b+168,v.thetaMin);
        encodeWireScalar(b+176,v.TpMin);
        encodeWireScalar(b+184,v.TpMax);
        encodeWireScalar(b+192,v.collisionalPressureEnabled);
        encodeWireScalar(b+200,v.collisionalRestitution);
        encodeWireScalar(b+208,v.pressureKickFraction);
        encodeWireScalar(b+216,v.jammingPressureEnabled);
        encodeWireScalar(b+224,v.packingFraction);
        encodeWireScalar(b+232,v.packingProjectionIterations);
        encodeWireScalar(b+236,v.gasFluxScheme);
        encodeWireScalar(b+240,v.gasReconstruction);
        encodeWireScalar(b+244,v.gasLimiter);
        encodeWireScalar(b+248,v.gasTimeIntegrator);
        encodeWireScalar(b+252,v.gasRobustFallback);
        encodeWireScalar(b+256,v.turbulenceModel);
        encodeWireScalar(b+264,v.lesDeltaCoeff);
        encodeWireScalar(b+272,v.turbulentPrandtl);
        encodeWireScalar(b+280,v.waleCw);
        encodeWireScalar(b+288,v.smagorinskyCs);
        encodeWireScalar(b+296,v.csrCellLocalPathEnabled);
        encodeWireScalar(b+300,v.csrHeavyReductionMode);
        encodeWireScalar(b+304,v.csrHeavyAutoInterval);
        encodeWireScalar(b+308,v.particleBlockThreads);
        encodeWireScalar(b+312,v.reductionBlockThreads);
        encodeWireScalar(b+316,v.csrWarpAggregatedBinning);
        encodeWireScalar(b+320,v.csrSplitPreDirectoryEnabled);
        encodeWireScalar(b+324,v.dragModel);
        encodeWireScalar(b+328,v.dragParameter0);
        encodeWireScalar(b+336,v.dragParameter1);
        encodeWireScalar(b+344,v.dragParameter2);
        encodeWireScalar(b+352,v.dragParameter3);
        encodeWireScalar(b+360,v.gravityX);
        encodeWireScalar(b+368,v.gravityY);
        encodeWireScalar(b+376,v.gravityZ);
    }
    static void decode(const unsigned char* b,CreateArgs& v)
    {
        decodeWireScalar(b+0,v.nCells);
        decodeWireScalar(b+4,v.nFaces);
        decodeWireScalar(b+8,v.nInternalFaces);
        decodeWireScalar(b+12,v.nCellPlanes);
        decodeWireScalar(b+16,v.particleCapacity);
        decodeWireScalar(b+20,v.maxFaceWalkHops);
        decodeWireScalar(b+24,v.injectionParcelMass);
        decodeWireScalar(b+32,v.rngSeed);
        decodeWireScalar(b+40,v.gammaGas);
        decodeWireScalar(b+48,v.Rgas);
        decodeWireScalar(b+56,v.rhoSolid);
        decodeWireScalar(b+64,v.solveParticleTemperature);
        decodeWireScalar(b+68,v.particleGasHeatTransferModelId);
        decodeWireScalar(b+72,v.particleThermalRho);
        decodeWireScalar(b+80,v.particleCp);
        decodeWireScalar(b+88,v.gasMu);
        decodeWireScalar(b+96,v.gasPr);
        decodeWireScalar(b+104,v.particleDiameterFallback);
        decodeWireScalar(b+112,v.particleDiameterMin);
        decodeWireScalar(b+120,v.particleDiameterMax);
        decodeWireScalar(b+128,v.particleDiameterSigma);
        decodeWireScalar(b+136,v.injectionTheta);
        decodeWireScalar(b+144,v.rhoMin);
        decodeWireScalar(b+152,v.TgasMin);
        decodeWireScalar(b+160,v.epsSMin);
        decodeWireScalar(b+168,v.thetaMin);
        decodeWireScalar(b+176,v.TpMin);
        decodeWireScalar(b+184,v.TpMax);
        decodeWireScalar(b+192,v.collisionalPressureEnabled);
        decodeWireScalar(b+200,v.collisionalRestitution);
        decodeWireScalar(b+208,v.pressureKickFraction);
        decodeWireScalar(b+216,v.jammingPressureEnabled);
        decodeWireScalar(b+224,v.packingFraction);
        decodeWireScalar(b+232,v.packingProjectionIterations);
        decodeWireScalar(b+236,v.gasFluxScheme);
        decodeWireScalar(b+240,v.gasReconstruction);
        decodeWireScalar(b+244,v.gasLimiter);
        decodeWireScalar(b+248,v.gasTimeIntegrator);
        decodeWireScalar(b+252,v.gasRobustFallback);
        decodeWireScalar(b+256,v.turbulenceModel);
        decodeWireScalar(b+264,v.lesDeltaCoeff);
        decodeWireScalar(b+272,v.turbulentPrandtl);
        decodeWireScalar(b+280,v.waleCw);
        decodeWireScalar(b+288,v.smagorinskyCs);
        decodeWireScalar(b+296,v.csrCellLocalPathEnabled);
        decodeWireScalar(b+300,v.csrHeavyReductionMode);
        decodeWireScalar(b+304,v.csrHeavyAutoInterval);
        decodeWireScalar(b+308,v.particleBlockThreads);
        decodeWireScalar(b+312,v.reductionBlockThreads);
        decodeWireScalar(b+316,v.csrWarpAggregatedBinning);
        decodeWireScalar(b+320,v.csrSplitPreDirectoryEnabled);
        decodeWireScalar(b+324,v.dragModel);
        decodeWireScalar(b+328,v.dragParameter0);
        decodeWireScalar(b+336,v.dragParameter1);
        decodeWireScalar(b+344,v.dragParameter2);
        decodeWireScalar(b+352,v.dragParameter3);
        decodeWireScalar(b+360,v.gravityX);
        decodeWireScalar(b+368,v.gravityY);
        decodeWireScalar(b+376,v.gravityZ);
    }
};
template<> struct WireCodec<CountArgs>
{
    static constexpr std::size_t size=4;
    static constexpr bool nativeLayout=sizeof(CountArgs)==size
        && offsetof(CountArgs,count)==0;
    static void encode(unsigned char* b,const CountArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.count);
    }
    static void decode(const unsigned char* b,CountArgs& v)
    {
        decodeWireScalar(b+0,v.count);
    }
};
template<> struct WireCodec<ScheduledInletConfigArgs>
{
    static constexpr std::size_t size=24;
    static constexpr bool nativeLayout=sizeof(ScheduledInletConfigArgs)==size
        && offsetof(ScheduledInletConfigArgs,count)==0
        && offsetof(ScheduledInletConfigArgs,nPressureRows)==4
        && offsetof(ScheduledInletConfigArgs,nVolumeFractionRows)==8
        && offsetof(ScheduledInletConfigArgs,inletTemperature)==16;
    static void encode(unsigned char* b,const ScheduledInletConfigArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.count);
        encodeWireScalar(b+4,v.nPressureRows);
        encodeWireScalar(b+8,v.nVolumeFractionRows);
        encodeWireScalar(b+16,v.inletTemperature);
    }
    static void decode(const unsigned char* b,ScheduledInletConfigArgs& v)
    {
        decodeWireScalar(b+0,v.count);
        decodeWireScalar(b+4,v.nPressureRows);
        decodeWireScalar(b+8,v.nVolumeFractionRows);
        decodeWireScalar(b+16,v.inletTemperature);
    }
};
template<> struct WireCodec<AdvanceArgs>
{
    static constexpr std::size_t size=16;
    static constexpr bool nativeLayout=sizeof(AdvanceArgs)==size
        && offsetof(AdvanceArgs,dt)==0
        && offsetof(AdvanceArgs,simulationTime)==8;
    static void encode(unsigned char* b,const AdvanceArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.dt);
        encodeWireScalar(b+8,v.simulationTime);
    }
    static void decode(const unsigned char* b,AdvanceArgs& v)
    {
        decodeWireScalar(b+0,v.dt);
        decodeWireScalar(b+8,v.simulationTime);
    }
};
template<> struct WireCodec<ScalarArgs>
{
    static constexpr std::size_t size=8;
    static constexpr bool nativeLayout=sizeof(ScalarArgs)==size
        && offsetof(ScalarArgs,value)==0;
    static void encode(unsigned char* b,const ScalarArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.value);
    }
    static void decode(const unsigned char* b,ScalarArgs& v)
    {
        decodeWireScalar(b+0,v.value);
    }
};
template<> struct WireCodec<CourantArgs>
{
    static constexpr std::size_t size=16;
    static constexpr bool nativeLayout=sizeof(CourantArgs)==size
        && offsetof(CourantArgs,dt)==0
        && offsetof(CourantArgs,scheduleTime)==8;
    static void encode(unsigned char* b,const CourantArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.dt);
        encodeWireScalar(b+8,v.scheduleTime);
    }
    static void decode(const unsigned char* b,CourantArgs& v)
    {
        decodeWireScalar(b+0,v.dt);
        decodeWireScalar(b+8,v.scheduleTime);
    }
};
template<> struct WireCodec<SstConfigArgs>
{
    static constexpr std::size_t size=144;
    static constexpr bool nativeLayout=sizeof(SstConfigArgs)==size
        && offsetof(SstConfigArgs,alphaK1)==0
        && offsetof(SstConfigArgs,alphaK2)==8
        && offsetof(SstConfigArgs,alphaOmega1)==16
        && offsetof(SstConfigArgs,alphaOmega2)==24
        && offsetof(SstConfigArgs,beta1)==32
        && offsetof(SstConfigArgs,beta2)==40
        && offsetof(SstConfigArgs,betaStar)==48
        && offsetof(SstConfigArgs,gamma1)==56
        && offsetof(SstConfigArgs,gamma2)==64
        && offsetof(SstConfigArgs,a1)==72
        && offsetof(SstConfigArgs,b1)==80
        && offsetof(SstConfigArgs,c1)==88
        && offsetof(SstConfigArgs,kMin)==96
        && offsetof(SstConfigArgs,omegaMin)==104
        && offsetof(SstConfigArgs,wallTreatment)==112
        && offsetof(SstConfigArgs,wallKappa)==120
        && offsetof(SstConfigArgs,wallE)==128
        && offsetof(SstConfigArgs,wallCmu)==136;
    static void encode(unsigned char* b,const SstConfigArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.alphaK1);
        encodeWireScalar(b+8,v.alphaK2);
        encodeWireScalar(b+16,v.alphaOmega1);
        encodeWireScalar(b+24,v.alphaOmega2);
        encodeWireScalar(b+32,v.beta1);
        encodeWireScalar(b+40,v.beta2);
        encodeWireScalar(b+48,v.betaStar);
        encodeWireScalar(b+56,v.gamma1);
        encodeWireScalar(b+64,v.gamma2);
        encodeWireScalar(b+72,v.a1);
        encodeWireScalar(b+80,v.b1);
        encodeWireScalar(b+88,v.c1);
        encodeWireScalar(b+96,v.kMin);
        encodeWireScalar(b+104,v.omegaMin);
        encodeWireScalar(b+112,v.wallTreatment);
        encodeWireScalar(b+120,v.wallKappa);
        encodeWireScalar(b+128,v.wallE);
        encodeWireScalar(b+136,v.wallCmu);
    }
    static void decode(const unsigned char* b,SstConfigArgs& v)
    {
        decodeWireScalar(b+0,v.alphaK1);
        decodeWireScalar(b+8,v.alphaK2);
        decodeWireScalar(b+16,v.alphaOmega1);
        decodeWireScalar(b+24,v.alphaOmega2);
        decodeWireScalar(b+32,v.beta1);
        decodeWireScalar(b+40,v.beta2);
        decodeWireScalar(b+48,v.betaStar);
        decodeWireScalar(b+56,v.gamma1);
        decodeWireScalar(b+64,v.gamma2);
        decodeWireScalar(b+72,v.a1);
        decodeWireScalar(b+80,v.b1);
        decodeWireScalar(b+88,v.c1);
        decodeWireScalar(b+96,v.kMin);
        decodeWireScalar(b+104,v.omegaMin);
        decodeWireScalar(b+112,v.wallTreatment);
        decodeWireScalar(b+120,v.wallKappa);
        decodeWireScalar(b+128,v.wallE);
        decodeWireScalar(b+136,v.wallCmu);
    }
};
template<> struct WireCodec<GasModelCapabilitiesV1>
{
    static constexpr std::size_t size=16;
    static constexpr bool nativeLayout=sizeof(GasModelCapabilitiesV1)==size
        && offsetof(GasModelCapabilitiesV1,version)==0
        && offsetof(GasModelCapabilitiesV1,compiledSpecies)==4
        && offsetof(GasModelCapabilitiesV1,modeMask)==8
        && offsetof(GasModelCapabilitiesV1,reserved)==12;
    static void encode(unsigned char* b,const GasModelCapabilitiesV1& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.version);
        encodeWireScalar(b+4,v.compiledSpecies);
        encodeWireScalar(b+8,v.modeMask);
        encodeWireScalar(b+12,v.reserved);
    }
    static void decode(const unsigned char* b,GasModelCapabilitiesV1& v)
    {
        decodeWireScalar(b+0,v.version);
        decodeWireScalar(b+4,v.compiledSpecies);
        decodeWireScalar(b+8,v.modeMask);
        decodeWireScalar(b+12,v.reserved);
    }
};
template<> struct WireCodec<GasModelConfigureArgsV1>
{
    static constexpr std::size_t size=56;
    static constexpr bool nativeLayout=sizeof(GasModelConfigureArgsV1)==size
        && offsetof(GasModelConfigureArgsV1,version)==0
        && offsetof(GasModelConfigureArgsV1,mode)==4
        && offsetof(GasModelConfigureArgsV1,speciesCount)==8
        && offsetof(GasModelConfigureArgsV1,reserved)==12
        && offsetof(GasModelConfigureArgsV1,speciesOrderHash)==16
        && offsetof(GasModelConfigureArgsV1,thermoHash)==24
        && offsetof(GasModelConfigureArgsV1,mechanismHash)==32
        && offsetof(GasModelConfigureArgsV1,modelBytes)==40
        && offsetof(GasModelConfigureArgsV1,mechanismBytes)==48;
    static void encode(unsigned char* b,const GasModelConfigureArgsV1& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.version);
        encodeWireScalar(b+4,v.mode);
        encodeWireScalar(b+8,v.speciesCount);
        encodeWireScalar(b+12,v.reserved);
        encodeWireScalar(b+16,v.speciesOrderHash);
        encodeWireScalar(b+24,v.thermoHash);
        encodeWireScalar(b+32,v.mechanismHash);
        encodeWireScalar(b+40,v.modelBytes);
        encodeWireScalar(b+48,v.mechanismBytes);
    }
    static void decode(const unsigned char* b,GasModelConfigureArgsV1& v)
    {
        decodeWireScalar(b+0,v.version);
        decodeWireScalar(b+4,v.mode);
        decodeWireScalar(b+8,v.speciesCount);
        decodeWireScalar(b+12,v.reserved);
        decodeWireScalar(b+16,v.speciesOrderHash);
        decodeWireScalar(b+24,v.thermoHash);
        decodeWireScalar(b+32,v.mechanismHash);
        decodeWireScalar(b+40,v.modelBytes);
        decodeWireScalar(b+48,v.mechanismBytes);
    }
};
template<> struct WireCodec<GasSpeciesIdentityV1>
{
    static constexpr std::size_t size=32;
    static constexpr bool nativeLayout=sizeof(GasSpeciesIdentityV1)==size
        && offsetof(GasSpeciesIdentityV1,version)==0
        && offsetof(GasSpeciesIdentityV1,speciesCount)==4
        && offsetof(GasSpeciesIdentityV1,speciesOrderHash)==8
        && offsetof(GasSpeciesIdentityV1,thermoHash)==16
        && offsetof(GasSpeciesIdentityV1,mechanismHash)==24;
    static void encode(unsigned char* b,const GasSpeciesIdentityV1& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.version);
        encodeWireScalar(b+4,v.speciesCount);
        encodeWireScalar(b+8,v.speciesOrderHash);
        encodeWireScalar(b+16,v.thermoHash);
        encodeWireScalar(b+24,v.mechanismHash);
    }
    static void decode(const unsigned char* b,GasSpeciesIdentityV1& v)
    {
        decodeWireScalar(b+0,v.version);
        decodeWireScalar(b+4,v.speciesCount);
        decodeWireScalar(b+8,v.speciesOrderHash);
        decodeWireScalar(b+16,v.thermoHash);
        decodeWireScalar(b+24,v.mechanismHash);
    }
};
template<> struct WireCodec<BoundaryLayerConfigV1>
{
    static constexpr std::size_t size=80;
    static constexpr bool nativeLayout=sizeof(BoundaryLayerConfigV1)==size
        && offsetof(BoundaryLayerConfigV1,version)==0
        && offsetof(BoundaryLayerConfigV1,speciesCount)==4
        && offsetof(BoundaryLayerConfigV1,model)==8
        && offsetof(BoundaryLayerConfigV1,nodes)==12
        && offsetof(BoundaryLayerConfigV1,maxIterations)==16
        && offsetof(BoundaryLayerConfigV1,wallCount)==20
        && offsetof(BoundaryLayerConfigV1,quadratureCount)==24
        && offsetof(BoundaryLayerConfigV1,matchingCount)==28
        && offsetof(BoundaryLayerConfigV1,workspaceSlots)==32
        && offsetof(BoundaryLayerConfigV1,reservedAudit)==36
        && offsetof(BoundaryLayerConfigV1,relativeTolerance)==40
        && offsetof(BoundaryLayerConfigV1,absoluteTolerance)==48
        && offsetof(BoundaryLayerConfigV1,stretch)==56
        && offsetof(BoundaryLayerConfigV1,turbulentPrandtl)==64
        && offsetof(BoundaryLayerConfigV1,turbulentSchmidt)==72;
    static void encode(unsigned char* b,const BoundaryLayerConfigV1& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.version);
        encodeWireScalar(b+4,v.speciesCount);
        encodeWireScalar(b+8,v.model);
        encodeWireScalar(b+12,v.nodes);
        encodeWireScalar(b+16,v.maxIterations);
        encodeWireScalar(b+20,v.wallCount);
        encodeWireScalar(b+24,v.quadratureCount);
        encodeWireScalar(b+28,v.matchingCount);
        encodeWireScalar(b+32,v.workspaceSlots);
        encodeWireScalar(b+36,std::uint32_t(0));
        encodeWireScalar(b+40,v.relativeTolerance);
        encodeWireScalar(b+48,v.absoluteTolerance);
        encodeWireScalar(b+56,v.stretch);
        encodeWireScalar(b+64,v.turbulentPrandtl);
        encodeWireScalar(b+72,v.turbulentSchmidt);
    }
    static void decode(const unsigned char* b,BoundaryLayerConfigV1& v)
    {
        decodeWireScalar(b+0,v.version);
        decodeWireScalar(b+4,v.speciesCount);
        decodeWireScalar(b+8,v.model);
        decodeWireScalar(b+12,v.nodes);
        decodeWireScalar(b+16,v.maxIterations);
        decodeWireScalar(b+20,v.wallCount);
        decodeWireScalar(b+24,v.quadratureCount);
        decodeWireScalar(b+28,v.matchingCount);
        decodeWireScalar(b+32,v.workspaceSlots);
        v.reservedAudit=0;
        decodeWireScalar(b+40,v.relativeTolerance);
        decodeWireScalar(b+48,v.absoluteTolerance);
        decodeWireScalar(b+56,v.stretch);
        decodeWireScalar(b+64,v.turbulentPrandtl);
        decodeWireScalar(b+72,v.turbulentSchmidt);
    }
};
template<class T> constexpr std::size_t wireBytes(const T&) { return WireCodec<T>::size; }
template<class T> inline std::array<unsigned char,WireCodec<T>::size> encodeWire(const T& value)
{ std::array<unsigned char,WireCodec<T>::size> b{}; WireCodec<T>::encode(b.data(),value); return b; }


inline bool validCount(const std::int32_t count, const std::int32_t limit)
{
    (void)limit;
    return count >= 0;
}

inline bool addBytes(std::uint64_t& total, const std::uint64_t count,
                     const std::uint64_t elementBytes)
{
    if (count != 0 && elementBytes > std::numeric_limits<std::uint64_t>::max()/count)
    {
        return false;
    }
    const std::uint64_t bytes = count*elementBytes;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
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
    (void)compiledSpecies;
    bytes = WireCodec<GasModelConfigureArgsV1>::size;
    return addBytes(bytes, args.modelBytes, 1) && addBytes(bytes, args.mechanismBytes, 1);
}
inline bool gasSpeciesPayloadBytes(const GasSpeciesIdentityV1& identity,
                                   std::uint64_t cells, std::uint64_t& bytes)
{
    if (identity.speciesCount && cells > std::numeric_limits<std::uint64_t>::max()/identity.speciesCount) return false;
    bytes = WireCodec<GasSpeciesIdentityV1>::size;
    return addArrayBytes<double>(bytes, cells*identity.speciesCount);
}
// Optional wall-family extension; every pre-existing wire layout is unchanged.


static constexpr std::uint32_t boundaryLayerDiagnosticScalars=25;
inline bool boundaryLayerDiagnosticBytes(const std::uint32_t walls,
    const std::uint32_t species,std::uint64_t& bytes)
{
    bytes=0;
    return addBytes(bytes,walls,std::uint64_t(boundaryLayerDiagnosticScalars)*sizeof(double))
        && addBytes(bytes,walls,2ULL*species*sizeof(double));
}
// Seven physical scalars per real solver node, followed by all mechanism species.
inline bool boundaryLayerProfileBytes(std::uint32_t walls,std::uint32_t nodes,
    std::uint32_t species,std::uint64_t& bytes)
{
    bytes=0;
    return addBytes(bytes,std::uint64_t(walls)*nodes,(7ULL+species)*sizeof(double));
}
inline bool boundaryLayerPayloadBytes(const BoundaryLayerConfigV1& a,
    std::uint32_t faces,std::uint32_t species,std::uint64_t& bytes)
{
    (void)faces; (void)species;
    bytes=WireCodec<BoundaryLayerConfigV1>::size;
    if (!a.wallCount) return true;
    return addArrayBytes<int>(bytes,std::uint64_t(a.wallCount)+2ULL*(a.wallCount+1ULL)+a.matchingCount)
        && addArrayBytes<double>(bytes,7ULL*a.wallCount)
        && addArrayBytes<double>(bytes,2ULL*a.quadratureCount)
        && addArrayBytes<double>(bytes,a.matchingCount)
        && addArrayBytes<double>(bytes,std::uint64_t(a.wallCount)*a.speciesCount);
}


}                         

#endif
