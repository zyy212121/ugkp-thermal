#ifndef UGKWP_GPU_BACKEND_PROTOCOL_H
#define UGKWP_GPU_BACKEND_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <array>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace ugkwpGpuIpc
{

static constexpr std::uint32_t magic = 0x46534842U;          
static constexpr std::uint16_t protocolMajor = 7;
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
    downloadSst = 23,
    uploadParticleSavedVelocity = 24,
    downloadParticleSavedVelocity = 25
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
    static constexpr std::size_t size=352;
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
        && offsetof(CreateArgs,gasMu)==72
        && offsetof(CreateArgs,gasPr)==80
        && offsetof(CreateArgs,dragModelId)==88
        && offsetof(CreateArgs,particleGasHeatTransferModelId)==92
        && offsetof(CreateArgs,dragResidualRe)==96
        && offsetof(CreateArgs,gravityX)==104
        && offsetof(CreateArgs,gravityY)==112
        && offsetof(CreateArgs,gravityZ)==120
        && offsetof(CreateArgs,particleDiameterFallback)==128
        && offsetof(CreateArgs,particleDiameterMin)==136
        && offsetof(CreateArgs,particleDiameterMax)==144
        && offsetof(CreateArgs,particleDiameterSigma)==152
        && offsetof(CreateArgs,injectionTheta)==160
        && offsetof(CreateArgs,rhoMin)==168
        && offsetof(CreateArgs,TgasMin)==176
        && offsetof(CreateArgs,epsSMin)==184
        && offsetof(CreateArgs,thetaMin)==192
        && offsetof(CreateArgs,TpMin)==200
        && offsetof(CreateArgs,TpMax)==208
        && offsetof(CreateArgs,collisionalPressureEnabled)==216
        && offsetof(CreateArgs,collisionalRestitution)==224
        && offsetof(CreateArgs,pressureKickFraction)==232
        && offsetof(CreateArgs,jammingPressureEnabled)==240
        && offsetof(CreateArgs,packingFraction)==248
        && offsetof(CreateArgs,packingProjectionIterations)==256
        && offsetof(CreateArgs,gasFluxScheme)==260
        && offsetof(CreateArgs,gasReconstruction)==264
        && offsetof(CreateArgs,gasLimiter)==268
        && offsetof(CreateArgs,gasTimeIntegrator)==272
        && offsetof(CreateArgs,gasRobustFallback)==276
        && offsetof(CreateArgs,turbulenceModel)==280
        && offsetof(CreateArgs,lesDeltaCoeff)==288
        && offsetof(CreateArgs,turbulentPrandtl)==296
        && offsetof(CreateArgs,waleCw)==304
        && offsetof(CreateArgs,smagorinskyCs)==312
        && offsetof(CreateArgs,maxDiffusionNumber)==320
        && offsetof(CreateArgs,csrCellLocalPathEnabled)==328
        && offsetof(CreateArgs,csrHeavyReductionMode)==332
        && offsetof(CreateArgs,csrHeavyAutoInterval)==336
        && offsetof(CreateArgs,particleBlockThreads)==340
        && offsetof(CreateArgs,reductionBlockThreads)==344
        && offsetof(CreateArgs,csrWarpAggregatedBinning)==348;
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
        encodeWireScalar(b+72,v.gasMu);
        encodeWireScalar(b+80,v.gasPr);
        encodeWireScalar(b+88,v.dragModelId);
        encodeWireScalar(b+92,v.particleGasHeatTransferModelId);
        encodeWireScalar(b+96,v.dragResidualRe);
        encodeWireScalar(b+104,v.gravityX);
        encodeWireScalar(b+112,v.gravityY);
        encodeWireScalar(b+120,v.gravityZ);
        encodeWireScalar(b+128,v.particleDiameterFallback);
        encodeWireScalar(b+136,v.particleDiameterMin);
        encodeWireScalar(b+144,v.particleDiameterMax);
        encodeWireScalar(b+152,v.particleDiameterSigma);
        encodeWireScalar(b+160,v.injectionTheta);
        encodeWireScalar(b+168,v.rhoMin);
        encodeWireScalar(b+176,v.TgasMin);
        encodeWireScalar(b+184,v.epsSMin);
        encodeWireScalar(b+192,v.thetaMin);
        encodeWireScalar(b+200,v.TpMin);
        encodeWireScalar(b+208,v.TpMax);
        encodeWireScalar(b+216,v.collisionalPressureEnabled);
        encodeWireScalar(b+224,v.collisionalRestitution);
        encodeWireScalar(b+232,v.pressureKickFraction);
        encodeWireScalar(b+240,v.jammingPressureEnabled);
        encodeWireScalar(b+248,v.packingFraction);
        encodeWireScalar(b+256,v.packingProjectionIterations);
        encodeWireScalar(b+260,v.gasFluxScheme);
        encodeWireScalar(b+264,v.gasReconstruction);
        encodeWireScalar(b+268,v.gasLimiter);
        encodeWireScalar(b+272,v.gasTimeIntegrator);
        encodeWireScalar(b+276,v.gasRobustFallback);
        encodeWireScalar(b+280,v.turbulenceModel);
        encodeWireScalar(b+288,v.lesDeltaCoeff);
        encodeWireScalar(b+296,v.turbulentPrandtl);
        encodeWireScalar(b+304,v.waleCw);
        encodeWireScalar(b+312,v.smagorinskyCs);
        encodeWireScalar(b+320,v.maxDiffusionNumber);
        encodeWireScalar(b+328,v.csrCellLocalPathEnabled);
        encodeWireScalar(b+332,v.csrHeavyReductionMode);
        encodeWireScalar(b+336,v.csrHeavyAutoInterval);
        encodeWireScalar(b+340,v.particleBlockThreads);
        encodeWireScalar(b+344,v.reductionBlockThreads);
        encodeWireScalar(b+348,v.csrWarpAggregatedBinning);
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
        decodeWireScalar(b+72,v.gasMu);
        decodeWireScalar(b+80,v.gasPr);
        decodeWireScalar(b+88,v.dragModelId);
        decodeWireScalar(b+92,v.particleGasHeatTransferModelId);
        decodeWireScalar(b+96,v.dragResidualRe);
        decodeWireScalar(b+104,v.gravityX);
        decodeWireScalar(b+112,v.gravityY);
        decodeWireScalar(b+120,v.gravityZ);
        decodeWireScalar(b+128,v.particleDiameterFallback);
        decodeWireScalar(b+136,v.particleDiameterMin);
        decodeWireScalar(b+144,v.particleDiameterMax);
        decodeWireScalar(b+152,v.particleDiameterSigma);
        decodeWireScalar(b+160,v.injectionTheta);
        decodeWireScalar(b+168,v.rhoMin);
        decodeWireScalar(b+176,v.TgasMin);
        decodeWireScalar(b+184,v.epsSMin);
        decodeWireScalar(b+192,v.thetaMin);
        decodeWireScalar(b+200,v.TpMin);
        decodeWireScalar(b+208,v.TpMax);
        decodeWireScalar(b+216,v.collisionalPressureEnabled);
        decodeWireScalar(b+224,v.collisionalRestitution);
        decodeWireScalar(b+232,v.pressureKickFraction);
        decodeWireScalar(b+240,v.jammingPressureEnabled);
        decodeWireScalar(b+248,v.packingFraction);
        decodeWireScalar(b+256,v.packingProjectionIterations);
        decodeWireScalar(b+260,v.gasFluxScheme);
        decodeWireScalar(b+264,v.gasReconstruction);
        decodeWireScalar(b+268,v.gasLimiter);
        decodeWireScalar(b+272,v.gasTimeIntegrator);
        decodeWireScalar(b+276,v.gasRobustFallback);
        decodeWireScalar(b+280,v.turbulenceModel);
        decodeWireScalar(b+288,v.lesDeltaCoeff);
        decodeWireScalar(b+296,v.turbulentPrandtl);
        decodeWireScalar(b+304,v.waleCw);
        decodeWireScalar(b+312,v.smagorinskyCs);
        decodeWireScalar(b+320,v.maxDiffusionNumber);
        decodeWireScalar(b+328,v.csrCellLocalPathEnabled);
        decodeWireScalar(b+332,v.csrHeavyReductionMode);
        decodeWireScalar(b+336,v.csrHeavyAutoInterval);
        decodeWireScalar(b+340,v.particleBlockThreads);
        decodeWireScalar(b+344,v.reductionBlockThreads);
        decodeWireScalar(b+348,v.csrWarpAggregatedBinning);
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
template<> struct WireCodec<ParticleStuckConfigArgs>
{
    static constexpr std::size_t size=152;
    static constexpr bool nativeLayout=sizeof(ParticleStuckConfigArgs)==size
        && offsetof(ParticleStuckConfigArgs,nFaces)==0
        && offsetof(ParticleStuckConfigArgs,heatTransferEnabled)==4
        && offsetof(ParticleStuckConfigArgs,sommerfeldThreshold)==8
        && offsetof(ParticleStuckConfigArgs,maximumCoverage)==16
        && offsetof(ParticleStuckConfigArgs,depositionHeatTransferEfficiency)==24
        && offsetof(ParticleStuckConfigArgs,reflectionHeatTransferEfficiency)==32
        && offsetof(ParticleStuckConfigArgs,adhesionEnergyScale)==40
        && offsetof(ParticleStuckConfigArgs,wallDensityKgM3)==48
        && offsetof(ParticleStuckConfigArgs,wallSpecificHeatJkgK)==56
        && offsetof(ParticleStuckConfigArgs,wallConductivityWmK)==64
        && offsetof(ParticleStuckConfigArgs,contactAngleDegree)==72
        && offsetof(ParticleStuckConfigArgs,wallTransientResistance)==80
        && offsetof(ParticleStuckConfigArgs,nonlinearIterations)==84
        && offsetof(ParticleStuckConfigArgs,meltingTemperatureK)==88
        && offsetof(ParticleStuckConfigArgs,mushyRangeK)==96
        && offsetof(ParticleStuckConfigArgs,latentHeatJkg)==104
        && offsetof(ParticleStuckConfigArgs,solidDensityKgM3)==112
        && offsetof(ParticleStuckConfigArgs,solidSpecificHeatJkgK)==120
        && offsetof(ParticleStuckConfigArgs,solidThermalConductivityWmK)==128
        && offsetof(ParticleStuckConfigArgs,pinningThicknessFraction)==136
        && offsetof(ParticleStuckConfigArgs,interfaceResistanceM2KW)==144;
    static void encode(unsigned char* b,const ParticleStuckConfigArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.nFaces);
        encodeWireScalar(b+4,v.heatTransferEnabled);
        encodeWireScalar(b+8,v.sommerfeldThreshold);
        encodeWireScalar(b+16,v.maximumCoverage);
        encodeWireScalar(b+24,v.depositionHeatTransferEfficiency);
        encodeWireScalar(b+32,v.reflectionHeatTransferEfficiency);
        encodeWireScalar(b+40,v.adhesionEnergyScale);
        encodeWireScalar(b+48,v.wallDensityKgM3);
        encodeWireScalar(b+56,v.wallSpecificHeatJkgK);
        encodeWireScalar(b+64,v.wallConductivityWmK);
        encodeWireScalar(b+72,v.contactAngleDegree);
        encodeWireScalar(b+80,v.wallTransientResistance);
        encodeWireScalar(b+84,v.nonlinearIterations);
        encodeWireScalar(b+88,v.meltingTemperatureK);
        encodeWireScalar(b+96,v.mushyRangeK);
        encodeWireScalar(b+104,v.latentHeatJkg);
        encodeWireScalar(b+112,v.solidDensityKgM3);
        encodeWireScalar(b+120,v.solidSpecificHeatJkgK);
        encodeWireScalar(b+128,v.solidThermalConductivityWmK);
        encodeWireScalar(b+136,v.pinningThicknessFraction);
        encodeWireScalar(b+144,v.interfaceResistanceM2KW);
    }
    static void decode(const unsigned char* b,ParticleStuckConfigArgs& v)
    {
        decodeWireScalar(b+0,v.nFaces);
        decodeWireScalar(b+4,v.heatTransferEnabled);
        decodeWireScalar(b+8,v.sommerfeldThreshold);
        decodeWireScalar(b+16,v.maximumCoverage);
        decodeWireScalar(b+24,v.depositionHeatTransferEfficiency);
        decodeWireScalar(b+32,v.reflectionHeatTransferEfficiency);
        decodeWireScalar(b+40,v.adhesionEnergyScale);
        decodeWireScalar(b+48,v.wallDensityKgM3);
        decodeWireScalar(b+56,v.wallSpecificHeatJkgK);
        decodeWireScalar(b+64,v.wallConductivityWmK);
        decodeWireScalar(b+72,v.contactAngleDegree);
        decodeWireScalar(b+80,v.wallTransientResistance);
        decodeWireScalar(b+84,v.nonlinearIterations);
        decodeWireScalar(b+88,v.meltingTemperatureK);
        decodeWireScalar(b+96,v.mushyRangeK);
        decodeWireScalar(b+104,v.latentHeatJkg);
        decodeWireScalar(b+112,v.solidDensityKgM3);
        decodeWireScalar(b+120,v.solidSpecificHeatJkgK);
        decodeWireScalar(b+128,v.solidThermalConductivityWmK);
        decodeWireScalar(b+136,v.pinningThicknessFraction);
        decodeWireScalar(b+144,v.interfaceResistanceM2KW);
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
    static constexpr std::size_t size=24;
    static constexpr bool nativeLayout=sizeof(CourantArgs)==size
        && offsetof(CourantArgs,dt)==0
        && offsetof(CourantArgs,targetMaxCo)==8
        && offsetof(CourantArgs,scheduleTime)==16;
    static void encode(unsigned char* b,const CourantArgs& v)
    {
        std::fill(b,b+size,0);
        encodeWireScalar(b+0,v.dt);
        encodeWireScalar(b+8,v.targetMaxCo);
        encodeWireScalar(b+16,v.scheduleTime);
    }
    static void decode(const unsigned char* b,CourantArgs& v)
    {
        decodeWireScalar(b+0,v.dt);
        decodeWireScalar(b+8,v.targetMaxCo);
        decodeWireScalar(b+16,v.scheduleTime);
    }
};
template<> struct WireCodec<SstConfigArgs>
{
    static constexpr std::size_t size=152;
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
        && offsetof(SstConfigArgs,maxSourceNumber)==112
        && offsetof(SstConfigArgs,wallTreatment)==120
        && offsetof(SstConfigArgs,wallKappa)==128
        && offsetof(SstConfigArgs,wallE)==136
        && offsetof(SstConfigArgs,wallCmu)==144;
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
        encodeWireScalar(b+112,v.maxSourceNumber);
        encodeWireScalar(b+120,v.wallTreatment);
        encodeWireScalar(b+128,v.wallKappa);
        encodeWireScalar(b+136,v.wallE);
        encodeWireScalar(b+144,v.wallCmu);
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
        decodeWireScalar(b+112,v.maxSourceNumber);
        decodeWireScalar(b+120,v.wallTreatment);
        decodeWireScalar(b+128,v.wallKappa);
        decodeWireScalar(b+136,v.wallE);
        decodeWireScalar(b+144,v.wallCmu);
    }
};
template<class T> constexpr std::size_t wireBytes(const T&) { return WireCodec<T>::size; }
template<class T> inline std::array<unsigned char,WireCodec<T>::size> encodeWire(const T& value)
{ std::array<unsigned char,WireCodec<T>::size> b{}; WireCodec<T>::encode(b.data(),value); return b; }



inline void addBytes(std::uint64_t& total, const std::uint64_t count,
                     const std::uint64_t elementBytes)
{
    const std::uint64_t bytes = count*elementBytes;
    total += bytes;
}

template<class T>
inline void addArrayBytes(std::uint64_t& total, const std::uint64_t count)
{
    addBytes(total, count, sizeof(T));
}


}                         

#endif
