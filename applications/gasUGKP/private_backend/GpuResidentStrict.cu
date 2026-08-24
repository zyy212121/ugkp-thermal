#include <cuda_runtime.h>

#include "GpuBackendApi.H"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuDragModels.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>

#include "GpuCouplingMath.H"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef UGKP_DEVELOPMENT_PROBES
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#endif

namespace
{

char lastError[2048] = "no GPU resident strict error";
static constexpr double OfSmall = 2.22044604925031308085e-16;
static constexpr double OfVSmall = 2.22507385850720138309e-308;
static constexpr double OfGreat = 1.0/OfSmall;
static constexpr double OfPi = 3.141592653589793238462643383279502884;

enum class CsrReductionTaskSource : int
{
    fullIndexed = 0,
    splitBaseDirect = 1,
    splitInjectionIndexed = 2
};

struct CsrReductionTask
{
    int cell;
    int begin;
    int end;
    int source;
};

struct DeviceState
{
    DeviceState* deviceState = nullptr;

    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nCellPlanes = 0;
    int particleCapacity = 0;
    int particleWorkGrid = 0;
    bool particlesMayBePresent = false;
    int maxFaceWalkHops = 8;
    int* particleCountDevice = nullptr;
#ifdef UGKP_DEVELOPMENT_PROBES
    int* diagnosticPreTransportParticleCount = nullptr;
#endif
    double injectionParcelMass = 0.0;
    unsigned long long rngSeed = 0;
    double gammaGas = 1.4;
    double Rgas = 287.0;
    double gasCp = 0.0;
    double rhoSolid = 2500.0;
    double invRhoSolid = 0.0;
    int solveParticleTemperature = 0;
    int particleGasHeatTransferModelId = 1;
    double particleThermalRho = 0.0;
    double particleCp = 0.0;
    double particleThermalCapacity = 0.0;
    double gasMu = 0.0;
    double gasPr = 0.4;
    double gasPrClamped = 0.4;
    double gasPrOneThird = 0.0;
    int dragModel = 0;
    double dragParameter0 = 0.0;
    double dragParameter1 = 0.0;
    double dragParameter2 = 0.0;
    double dragParameter3 = 0.0;
    double gravityX = 0.0;
    double gravityY = 0.0;
    double gravityZ = 0.0;
    int gasFluxScheme = 2;
    int gasReconstruction = 0;
    int gasLimiter = 1;
    int gasTimeIntegrator = 1;
    int gasRobustFallback = 1;
    int turbulenceModel = 0;
                                                                             
                                                                             
    int hostGasFluxScheme = 2;
    int hostGasTimeIntegrator = 1;
    int hostTurbulenceModel = 0;
    int hostDragModel = 0;
    bool hostGravityActive = false;
    double lesDeltaCoeff = 1.0;
    double turbulentPrandtl = 0.9;
    double waleCw = 0.325;
    double smagorinskyCs = 0.17;
    double maxDiffusionNumber = 0.25;
    int sstConfigured = 0;
    ugkwp::SstCoefficients sstCoefficients =
        ugkwp::defaultSstCoefficients();
    double sstKMin = 1.0e-12;
    double sstOmegaMin = 1.0e-6;
    double sstMaxSourceNumber = 0.25;
    int sstWallTreatment = 0;
    double sstWallKappa = 0.41;
    double sstWallE = 9.8;
    double sstWallCmu = 0.09;
    double sstJayatillekeP = 0.0;
    double sstThermalYPlus = 0.0;
    double particleDiameterFallback = 0.0;
    double particleDiameterMin = 0.0;
    double particleDiameterMax = 0.0;
    double particleDiameterSigma = 0.0;
    double injectionTheta = 0.0;
    double rhoMin = 1.0e-12;
    double TgasMin = 1.0e-12;
    double epsSMin = 1.0e-12;
    double thetaMin = 1.0e-12;
    double TpMin = 1.0e-12;
    double TpMax = 1.0e30;
    int collisionalPressureEnabled = 0;
    double collisionalRestitution = 0.9;
    double pressureKickFraction = 0.25;
    int jammingPressureEnabled = 0;
    double packingFraction = 0.63;
    int packingProjectionIterations = 20;
    int csrCellLocalPathEnabled = 1;
    int csrHeavyReductionEnabled = 0;
    int csrHeavyReductionMode = 0;
    int csrHeavyAutoInterval = 100;
    int csrHeavyReductionActive = 0;
    unsigned long long schedulingAdvanceCount = 0;
    int fixedCellBlockThreads = 128;
    int fixedFaceBlockThreads = 128;
    int fixedWorkBlockTuned = 0;
    int particleBlockThreads = 128;
    int reductionBlockThreads = 128;
    int multiprocessorCount = 1;
    int hardwareMaxThreadsPerBlock = 0;
    int hardwareMaxBlocksPerSm = 0;
    int particleBlocksPerSm = 1;
    int lightBlocksPerSm = 1;
    int heavyBlocksPerSm = 1;
    int csrHeavyCellThreshold = 0;
    int csrHeavyTileParticles = 0;
    int csrWarpAggregatedBinning = 0;
    int csrSplitPreDirectoryEnabled = 1;
    int csrHeavyTaskCapacity = 0;
    int csrHeavyWorkerGrid = 0;
    int* faceOwner = nullptr;
    int* faceNeighbour = nullptr;
    int* facePeriodicPair = nullptr;
    int hasPeriodicFaces = 0;
    double* facePeriodicDx = nullptr;
    double* facePeriodicDy = nullptr;
    double* facePeriodicDz = nullptr;
    double* V = nullptr;
    double* Cx = nullptr;
    double* Cy = nullptr;
    double* Cz = nullptr;
    double* faceCx = nullptr;
    double* faceCy = nullptr;
    double* faceCz = nullptr;
    double* Sfx = nullptr;
    double* Sfy = nullptr;
    double* Sfz = nullptr;
    double* magSf = nullptr;
    double* deltaCoeffs = nullptr;
    double* faceWeight = nullptr;
    double* cellLength = nullptr;
    int* cellPlaneStart = nullptr;
    int* cellPlaneCount = nullptr;
    int* cellFaceId = nullptr;
    int* cellFaceNeighbor = nullptr;
    int* cellFaceKind = nullptr;
    double* cellFaceRestitution = nullptr;
    double* cellFaceTangential = nullptr;
    double* planeNx = nullptr;
    double* planeNy = nullptr;
    double* planeNz = nullptr;
    double* planeD = nullptr;

    double* rho = nullptr;
    double* rhoUx = nullptr;
    double* rhoUy = nullptr;
    double* rhoUz = nullptr;
    double* rhoE = nullptr;
    double* rhoNext = nullptr;
    double* rhoUxNext = nullptr;
    double* rhoUyNext = nullptr;
    double* rhoUzNext = nullptr;
    double* rhoENext = nullptr;
    double* gasPhiRho = nullptr;
    double* gasPhiRhoUx = nullptr;
    double* gasPhiRhoUy = nullptr;
    double* gasPhiRhoUz = nullptr;
    double* gasPhiRhoE = nullptr;
    double* gasFluxPositivityScale = nullptr;
    double* gasHllcAdcSensor = nullptr;
    int* gasBoundaryKind = nullptr;
    int* gasBoundaryRhoFix = nullptr;
    int* gasBoundaryUFix = nullptr;
    int* gasBoundaryPFix = nullptr;
    int* gasBoundaryTFix = nullptr;
    int* gasBoundaryPWave = nullptr;
    double* gasBoundaryPWaveGamma = nullptr;
    double* gasBoundaryPWaveFieldInf = nullptr;
    double* gasBoundaryPWaveLInf = nullptr;
    double* gasBoundaryRho = nullptr;
    double* gasBoundaryUx = nullptr;
    double* gasBoundaryUy = nullptr;
    double* gasBoundaryUz = nullptr;
    double* gasBoundaryP = nullptr;
    double* gasBoundaryT = nullptr;
                                                                           
                                                                              
                                                                            
    int* riemannBoundaryKind = nullptr;
    int* riemannBoundaryRhoFix = nullptr;
    int* riemannBoundaryUFix = nullptr;
    int* riemannBoundaryPFix = nullptr;
    int* riemannBoundaryTFix = nullptr;
    int* riemannBoundaryPWave = nullptr;
    double* riemannBoundaryPWaveGamma = nullptr;
    double* riemannBoundaryPWaveFieldInf = nullptr;
    double* riemannBoundaryPWaveLInf = nullptr;
    double* riemannBoundaryRho = nullptr;
    double* riemannBoundaryUx = nullptr;
    double* riemannBoundaryUy = nullptr;
    double* riemannBoundaryUz = nullptr;
    double* riemannBoundaryP = nullptr;
    double* riemannBoundaryT = nullptr;
    int nScheduledInletFaces = 0;
    int* scheduledInletFaceMask = nullptr;
    double scheduledInletTemperature = 0.0;
    int nPressureScheduleRows = 0;
    double* pressureScheduleTimes = nullptr;
    double* pressureScheduleValues = nullptr;
    int nVolumeFractionScheduleRows = 0;
    double* volumeFractionScheduleTimes = nullptr;
    double* volumeFractionScheduleValues = nullptr;
    double* gradPx = nullptr;
    double* gradPy = nullptr;
    double* gradPz = nullptr;
    double* gradRhoX = nullptr;
    double* gradRhoY = nullptr;
    double* gradRhoZ = nullptr;
    double* gradUxX = nullptr;
    double* gradUxY = nullptr;
    double* gradUxZ = nullptr;
    double* gradUyX = nullptr;
    double* gradUyY = nullptr;
    double* gradUyZ = nullptr;
    double* gradUzX = nullptr;
    double* gradUzY = nullptr;
    double* gradUzZ = nullptr;
    double* gradTX = nullptr;
    double* gradTY = nullptr;
    double* gradTZ = nullptr;
                                                                     
                                                                   
                                                                           
                                                                  
    double* gasGradientLimiterRho = nullptr;
    double* gasGradientLimiterUx = nullptr;
    double* gasGradientLimiterUy = nullptr;
    double* gasGradientLimiterUz = nullptr;
    double* gasGradientLimiterP = nullptr;
    double* gasGradientLimiterT = nullptr;
    double* nut = nullptr;
    double* gasDiffusionNumber = nullptr;
    double* rhoK = nullptr;
    double* rhoOmega = nullptr;
    double* rhoKInitial = nullptr;
    double* rhoOmegaInitial = nullptr;
    double* k = nullptr;
    double* omega = nullptr;
    double* sstWallDistance = nullptr;
    double* sstF1 = nullptr;
    double* sstF2 = nullptr;
    double* gradKX = nullptr;
    double* gradKY = nullptr;
    double* gradKZ = nullptr;
    double* gradOmegaX = nullptr;
    double* gradOmegaY = nullptr;
    double* gradOmegaZ = nullptr;
    double* sstPhiRhoK = nullptr;
    double* sstPhiRhoOmega = nullptr;
    int* sstBoundaryKMode = nullptr;
    int* sstBoundaryOmegaMode = nullptr;
    double* sstBoundaryK = nullptr;
    double* sstBoundaryOmega = nullptr;
    double* sstSourceNumber = nullptr;
    double* Ux = nullptr;
    double* Uy = nullptr;
    double* Uz = nullptr;
    double* p = nullptr;
    double* Tgas = nullptr;
    double* couplingRhoOld = nullptr;
    double* couplingUxOld = nullptr;
    double* couplingUyOld = nullptr;
    double* couplingUzOld = nullptr;
    double* couplingTgasOld = nullptr;

    double* epsS = nullptr;
    double* rhoUsx = nullptr;
    double* rhoUsy = nullptr;
    double* rhoUsz = nullptr;
    double* rhoEs = nullptr;
    double* rhoDs = nullptr;
    double* rhoHp = nullptr;
    double* Usx = nullptr;
    double* Usy = nullptr;
    double* Usz = nullptr;
    double* theta = nullptr;
    double* Tp = nullptr;
    double* dMeanCell = nullptr;
    double* epsGPrev = nullptr;
    double* collisionalPressure = nullptr;
    double* pressureKickScale = nullptr;
    double* pressureDeltaMomX = nullptr;
    double* pressureDeltaMomY = nullptr;
    double* pressureDeltaMomZ = nullptr;
    double* pressureDeltaEnergy = nullptr;
    double* solidPressurePhiMomX = nullptr;
    double* solidPressurePhiMomY = nullptr;
    double* solidPressurePhiMomZ = nullptr;
    double* solidPressurePhiEnergy = nullptr;
                                                                             
    double* mobilePackingRho = nullptr;
    double* mobilePackingMomX = nullptr;
    double* mobilePackingMomY = nullptr;
    double* mobilePackingMomZ = nullptr;
    int* mobilePackingActiveCellMask = nullptr;
    int* mobilePackingCorrectionCellMask = nullptr;
    int* mobilePackingActiveCellList = nullptr;
    int* mobilePackingCorrectionCellList = nullptr;
    int* mobilePackingFrontierCurrent = nullptr;
    int* mobilePackingFrontierNext = nullptr;
    int* mobilePackingActiveCellCount = nullptr;
    int* mobilePackingCorrectionCellCount = nullptr;
    int* mobilePackingFrontierCurrentCount = nullptr;
    int* mobilePackingFrontierNextCount = nullptr;
    
                                                                     
                                                                                
                                                            
    double* thetaDragAlpha = nullptr;

    double* momRhoP = nullptr;
    double* momRhoUPx = nullptr;
    double* momRhoUPy = nullptr;
    double* momRhoUPz = nullptr;
    double* momRhoEP = nullptr;
    double* momRhoPD = nullptr;
    double* momRhoHpP = nullptr;
    int* poolThermalCount = nullptr;
    double* poolThermalSumUx = nullptr;
    double* poolThermalSumUy = nullptr;
    double* poolThermalSumUz = nullptr;
    double* poolThermalSumU2 = nullptr;
    int* poissonPoolSampleTargetCount = nullptr;
    double* poissonPoolMass = nullptr;
    double* poissonPoolMomX = nullptr;
    double* poissonPoolMomY = nullptr;
    double* poissonPoolMomZ = nullptr;
    double* poissonPoolEnergy = nullptr;
    double* poissonPoolDiameter = nullptr;
    double* poissonPoolDiameter2 = nullptr;

    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    double* pux = nullptr;
    double* puy = nullptr;
    double* puz = nullptr;
    double* puxOld = nullptr;
    double* puyOld = nullptr;
    double* puzOld = nullptr;
    double* pT = nullptr;
    double* pTheta = nullptr;
    double* pd = nullptr;
    double* pm = nullptr;
    int* pCellId = nullptr;
    int* pStatus = nullptr;
    unsigned long long* pRng = nullptr;
    unsigned long long* pOrigId = nullptr;
    int* sortedParticleIndex = nullptr;
    int* cellParticleCount = nullptr;
    int* cellParticleOffset = nullptr;
                                                                           
                                                                              
                                                                           
    int* preBaseCellOffset = nullptr;
    int* preBaseParticleCountDevice = nullptr;
    int preBaseDirectoryReady = 0;
    int useSplitPreDirectory = 0;
    int preInjectionSegmentActive = 0;
    int* compactCellOffset = nullptr;
    int* cellParticleWrite = nullptr;
    unsigned char* cellScanTempStorage = nullptr;
    size_t cellScanTempBytes = 0;
    int* compactCountDevice = nullptr;
    unsigned char* compactSelectTempStorage = nullptr;
    size_t compactSelectTempBytes = 0;
    int* csrCellTaskCount = nullptr;
    int* csrCellTaskOffset = nullptr;
    CsrReductionTask* csrReductionTasks = nullptr;
    int* csrMultiTaskCellList = nullptr;
    int* csrHeavyTaskCount = nullptr;
    int* csrHeavyTaskCursor = nullptr;
    int* csrHeavyCellCount = nullptr;
    int* csrHeavyCellList = nullptr;
    int* csrHeavyTaskCell = nullptr;
    int* csrHeavyTaskBegin = nullptr;
    int* csrHeavyTaskEnd = nullptr;
    int* csrHeavyCellTaskStart = nullptr;
    int* csrHeavyCellTaskCount = nullptr;
    double* csrHeavyPartials = nullptr;
                                                                        
                                                                     
                                                                        
                                                                          
    int* preInjectionHeavyTaskCount = nullptr;
    int* preInjectionHeavyTaskCell = nullptr;
    int* preInjectionHeavyTaskBegin = nullptr;
    int* preInjectionHeavyTaskEnd = nullptr;
    int* preInjectionHeavyCellTaskStart = nullptr;
    int* preInjectionHeavyCellTaskCount = nullptr;
    double* preInjectionHeavyPartials = nullptr;

    int nBoundarySources = 0;
    int* sourceCell = nullptr;
    int* sourceFace = nullptr;
    double* sourcePx = nullptr;
    double* sourcePy = nullptr;
    double* sourcePz = nullptr;
    double* sourceUx = nullptr;
    double* sourceUy = nullptr;
    double* sourceUz = nullptr;
    double* sourceT = nullptr;
    double* sourceTheta = nullptr;
    double* sourceD = nullptr;
    double* sourceMassRate = nullptr;
    double* sourceResidualMass = nullptr;
#ifdef UGKP_DEVELOPMENT_PROBES
    int* sourceInjectedCount = nullptr;
#endif

    double* compactPx = nullptr;
    double* compactPy = nullptr;
    double* compactPz = nullptr;
    double* compactPux = nullptr;
    double* compactPuy = nullptr;
    double* compactPuz = nullptr;
    double* compactPT = nullptr;
    double* compactPTheta = nullptr;
    double* compactPd = nullptr;
    double* compactPm = nullptr;
    int* compactPCellId = nullptr;
    int* compactPStatus = nullptr;
    unsigned long long* compactPRng = nullptr;
    unsigned long long* compactPOrigId = nullptr;
};

struct ActiveParticleIndexPredicate
{
    DeviceState* state;

    __device__ bool operator()(const int i) const
    {
        const DeviceState& s = *state;
        const int activeCount = *s.particleCountDevice;
        if
        (
            i < 0
         || i >= activeCount
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
        )
        {
            return false;
        }
        const int c = s.pCellId[i];
        return c >= 0 && c < s.nCells;
    }
};

#ifdef UGKP_DEVELOPMENT_PROBES

enum class DevelopmentProbeMode
{
    off,
    timing,
    full
};

enum DevelopmentProbeStage
{
    ProbeGasFlux = 0,
    ProbeEulerianCoupling,
    ProbeInjection,
    ProbeBinPre,
    ProbePressurePre,
    ProbeCollisionPool,
    ProbeRelax,
    ProbeTrack,
    ProbeBinPost,
    ProbeMoments,
    ProbePressurePost,
    ProbeCompaction,
    ProbeBoundary,
    ProbeStageCount
};

static constexpr int ProbeMaxOccurrences = 2;

const char* const developmentProbeStageNames[ProbeStageCount] =
{
    "gas_flux",
    "eulerian_coupling",
    "injection",
    "bin_pre",
    "pressure_pre",
    "collision_pool",
    "relax",
    "track",
    "bin_post",
    "moments",
    "pressure_post",
    "compaction",
    "boundary"
};

                                                                               
                                                 
enum DevelopmentProbeBadField : unsigned long long
{
    ProbeBadGasConserved = 1ull << 0,
    ProbeBadGasPrimitive = 1ull << 1,
    ProbeBadSolidConserved = 1ull << 2,
    ProbeBadSolidPrimitive = 1ull << 3,
    ProbeBadSolidMoments = 1ull << 4,
    ProbeBadCellRange = 1ull << 5,
    ProbeBadParticlePosition = 1ull << 16,
    ProbeBadParticleVelocity = 1ull << 17,
    ProbeBadParticleThermal = 1ull << 18,
    ProbeBadParticleMetadata = 1ull << 19,
    ProbeBadParticleCount = 1ull << 30,
    ProbeBadOccupancy = 1ull << 31
};

struct DevelopmentProbeDeviceSummary
{
    unsigned long long badCells = 0;
    unsigned long long badParticles = 0;
    unsigned long long badFieldMask = 0;
    int firstBadCellPlusOne = 0;
    int firstBadParticlePlusOne = 0;
};

struct DevelopmentProbeSample
{
    unsigned long long step = 0;
    double simulationTime = 0.0;
    double dt = 0.0;
    const char* status = "ok";
    const char* errorStage = "";
    const char* errorMessage = "";
    int timingValid = 0;
    int nCells = 0;
    int blockExponent = 0;
    int blockThreads = 0;
    int smCount = 0;
    int particleBlocksPerSm = 0;
    int lightBlocksPerSm = 0;
    int heavyBlocksPerSm = 0;
    int heavyReductionEnabled = 0;
    int particlePath = 0;
    int particleCount = -1;
    int particleCapacity = 0;
    double particleUtilisation = 0.0;
    int preTransportParticleCount = 0;
    int baseParticleCount = 0;
    int injectedParticleCount = 0;
    int removedParticleCount = 0;
    double injectionFraction = 0.0;
    double sourceResidualMass = 0.0;
    long long occupancySum = 0;
    int occupancyNonEmpty = 0;
    int occupancyMin = 0;
    double occupancyMean = 0.0;
    double occupancyStddev = 0.0;
    double occupancyCv = 0.0;
    int occupancyP50 = 0;
    int occupancyP95 = 0;
    int occupancyP99 = 0;
    int occupancyMax = 0;
    double occupancyI2 = 0.0;
    double occupancyImax = 0.0;
    int heavyThreshold = 0;
    int heavyTileParticles = 0;
    int heavyCellCount = 0;
    long long heavyParticleCount = 0;
    double heavyCellFraction = 0.0;
    double heavyParticleFraction = 0.0;
    long long heavyTaskCountEstimate = 0;
    int occupancyMatchesCount = 1;
    unsigned long long badCells = 0;
    unsigned long long badParticles = 0;
    unsigned long long badFieldMask = 0;
    int firstBadCell = -1;
    int firstBadParticle = -1;
    float totalMs = -1.0f;
    float stageMs[ProbeStageCount]{};

    DevelopmentProbeSample()
    {
        for (int i = 0; i < ProbeStageCount; ++i)
        {
            stageMs[i] = -1.0f;
        }
    }
};

struct DevelopmentProbeState
{
    DeviceState* owner = nullptr;
    DevelopmentProbeMode mode = DevelopmentProbeMode::off;
    FILE* log = nullptr;
    unsigned long long interval = 1;
    unsigned long long advanceIndex = 0;
    bool failOnNonFinite = false;
    int pid = 0;
    std::string modeName = "off";
    std::string runId;
    std::string variant;
    std::string logPath;
    cudaEvent_t totalStartEvent = nullptr;
    cudaEvent_t totalStopEvent = nullptr;
    cudaEvent_t
        stageStartEvents[ProbeStageCount][ProbeMaxOccurrences]{};
    cudaEvent_t
        stageStopEvents[ProbeStageCount][ProbeMaxOccurrences]{};
    int stageOccurrenceCount[ProbeStageCount]{};
    bool stageOccurrenceExecuted[ProbeStageCount][ProbeMaxOccurrences]{};
    DevelopmentProbeDeviceSummary* deviceSummary = nullptr;
    std::vector<int> occupancy;
    std::vector<int> injectedBySource;
    std::vector<double> sourceResidualMass;
};

DevelopmentProbeState developmentProbe;

#endif

void setLastError(const char* api, const cudaError_t err)
{
    std::snprintf
    (
        lastError,
        sizeof(lastError),
        "%s failed: %s",
        api,
        cudaGetErrorString(err)
    );
}

void setLastErrorText(const char* text)
{
    std::snprintf(lastError, sizeof(lastError), "%s", text);
}

template<class T>
int allocate(T*& ptr, const size_t n, const char* name)
{
    ptr = nullptr;
    if (n == 0)
    {
        return 0;
    }

    const cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(T));
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

template<class T>
void release(T*& ptr)
{
    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template<class T>
int copyToDevice(T* dst, const T* src, const size_t n, const char* name)
{
    if (n == 0)
    {
        return 0;
    }
    const cudaError_t err =
        cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

template<class T>
int copyToHost(T* dst, const T* src, const size_t n, const char* name)
{
    if (n == 0)
    {
        return 0;
    }
    const cudaError_t err =
        cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

int syncDeviceState(DeviceState* s, const char* name)
{
    if (s == nullptr || s->deviceState == nullptr)
    {
        setLastErrorText("cannot sync null GPU resident strict device state");
        return 1;
    }

    const cudaError_t err =
        cudaMemcpy(s->deviceState, s, sizeof(DeviceState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

int syncSstConfiguration(DeviceState* s, const char* name)
{
    const size_t first = offsetof(DeviceState, sstConfigured);
    const size_t last =
        offsetof(DeviceState, sstThermalYPlus)
      + sizeof(s->sstThermalYPlus);
    const cudaError_t err = cudaMemcpy
    (
        reinterpret_cast<unsigned char*>(s->deviceState) + first,
        reinterpret_cast<const unsigned char*>(s) + first,
        last - first,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

DeviceState* asState(void* handle)
{
    return reinterpret_cast<DeviceState*>(handle);
}

int validateState(DeviceState* s, const char* action)
{
    if (s == nullptr)
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "%s requested before GPU resident strict state creation",
            action
        );
        return 1;
    }
    return 0;
}

void scrubHostCalculationScalars(DeviceState* s)
{
    if (s == nullptr)
    {
        return;
    }

    s->injectionParcelMass = 0.0;
    s->rngSeed = 0;
    s->gammaGas = 0.0;
    s->Rgas = 0.0;
    s->gasCp = 0.0;
    s->rhoSolid = 0.0;
    s->invRhoSolid = 0.0;
    s->solveParticleTemperature = 0;
    s->particleThermalRho = 0.0;
    s->particleCp = 0.0;
    s->particleThermalCapacity = 0.0;
    s->gasMu = 0.0;
    s->gasPr = 0.0;
    s->gasPrClamped = 0.0;
    s->gasPrOneThird = 0.0;
    s->dragModel = 0;
    s->dragParameter0 = 0.0;
    s->dragParameter1 = 0.0;
    s->dragParameter2 = 0.0;
    s->dragParameter3 = 0.0;
    s->gravityX = 0.0;
    s->gravityY = 0.0;
    s->gravityZ = 0.0;
    s->gasFluxScheme = 0;
    s->gasReconstruction = 0;
    s->gasLimiter = 0;
    s->gasTimeIntegrator = 0;
    s->gasRobustFallback = 0;
    s->turbulenceModel = 0;
    s->lesDeltaCoeff = 0.0;
    s->turbulentPrandtl = 0.0;
    s->waleCw = 0.0;
    s->smagorinskyCs = 0.0;
    s->maxDiffusionNumber = 0.0;
    s->sstCoefficients = ugkwp::defaultSstCoefficients();
    s->sstKMin = 0.0;
    s->sstOmegaMin = 0.0;
    s->sstMaxSourceNumber = 0.0;
    s->sstWallTreatment = 0;
    s->sstWallKappa = 0.0;
    s->sstWallE = 0.0;
    s->sstWallCmu = 0.0;
    s->particleDiameterFallback = 0.0;
    s->particleDiameterMin = 0.0;
    s->particleDiameterMax = 0.0;
    s->particleDiameterSigma = 0.0;
    s->injectionTheta = 0.0;
    s->rhoMin = 0.0;
    s->TgasMin = 0.0;
    s->epsSMin = 0.0;
    s->thetaMin = 0.0;
    s->TpMin = 0.0;
    s->TpMax = 0.0;
}

void releaseState(DeviceState* s)
{
    if (s == nullptr)
    {
        return;
    }

    release(s->deviceState);
    release(s->faceOwner);
    release(s->faceNeighbour);
    release(s->facePeriodicPair);
    release(s->facePeriodicDx);
    release(s->facePeriodicDy);
    release(s->facePeriodicDz);
    release(s->V);
    release(s->Cx);
    release(s->Cy);
    release(s->Cz);
    release(s->faceCx);
    release(s->faceCy);
    release(s->faceCz);
    release(s->Sfx);
    release(s->Sfy);
    release(s->Sfz);
    release(s->magSf);
    release(s->deltaCoeffs);
    release(s->faceWeight);
    release(s->cellLength);
    release(s->cellPlaneStart);
    release(s->cellPlaneCount);
    release(s->cellFaceId);
    release(s->cellFaceNeighbor);
    release(s->cellFaceKind);
    release(s->cellFaceRestitution);
    release(s->cellFaceTangential);
    release(s->planeNx);
    release(s->planeNy);
    release(s->planeNz);
    release(s->planeD);
    release(s->particleCountDevice);
#ifdef UGKP_DEVELOPMENT_PROBES
    release(s->diagnosticPreTransportParticleCount);
#endif
    release(s->rho);
    release(s->rhoUx);
    release(s->rhoUy);
    release(s->rhoUz);
    release(s->rhoE);
    release(s->rhoNext);
    release(s->rhoUxNext);
    release(s->rhoUyNext);
    release(s->rhoUzNext);
    release(s->rhoENext);
    release(s->gasPhiRho);
    release(s->gasPhiRhoUx);
    release(s->gasPhiRhoUy);
    release(s->gasPhiRhoUz);
    release(s->gasPhiRhoE);
    release(s->gasFluxPositivityScale);
    release(s->gasHllcAdcSensor);
    release(s->gasBoundaryKind);
    release(s->gasBoundaryRhoFix);
    release(s->gasBoundaryUFix);
    release(s->gasBoundaryPFix);
    release(s->gasBoundaryTFix);
    release(s->gasBoundaryPWave);
    release(s->gasBoundaryPWaveGamma);
    release(s->gasBoundaryPWaveFieldInf);
    release(s->gasBoundaryPWaveLInf);
    release(s->gasBoundaryRho);
    release(s->gasBoundaryUx);
    release(s->gasBoundaryUy);
    release(s->gasBoundaryUz);
    release(s->gasBoundaryP);
    release(s->gasBoundaryT);
    release(s->riemannBoundaryKind);
    release(s->riemannBoundaryRhoFix);
    release(s->riemannBoundaryUFix);
    release(s->riemannBoundaryPFix);
    release(s->riemannBoundaryTFix);
    release(s->riemannBoundaryPWave);
    release(s->riemannBoundaryPWaveGamma);
    release(s->riemannBoundaryPWaveFieldInf);
    release(s->riemannBoundaryPWaveLInf);
    release(s->riemannBoundaryRho);
    release(s->riemannBoundaryUx);
    release(s->riemannBoundaryUy);
    release(s->riemannBoundaryUz);
    release(s->riemannBoundaryP);
    release(s->riemannBoundaryT);
    release(s->scheduledInletFaceMask);
    release(s->pressureScheduleTimes);
    release(s->pressureScheduleValues);
    release(s->volumeFractionScheduleTimes);
    release(s->volumeFractionScheduleValues);
    release(s->gradPx);
    release(s->gradPy);
    release(s->gradPz);
    release(s->gradRhoX);
    release(s->gradRhoY);
    release(s->gradRhoZ);
    release(s->gradUxX);
    release(s->gradUxY);
    release(s->gradUxZ);
    release(s->gradUyX);
    release(s->gradUyY);
    release(s->gradUyZ);
    release(s->gradUzX);
    release(s->gradUzY);
    release(s->gradUzZ);
    release(s->gradTX);
    release(s->gradTY);
    release(s->gradTZ);
    release(s->gasGradientLimiterRho);
    release(s->gasGradientLimiterUx);
    release(s->gasGradientLimiterUy);
    release(s->gasGradientLimiterUz);
    release(s->gasGradientLimiterP);
    release(s->gasGradientLimiterT);
    release(s->nut);
    release(s->gasDiffusionNumber);
    release(s->rhoK);
    release(s->rhoOmega);
    release(s->rhoKInitial);
    release(s->rhoOmegaInitial);
    release(s->k);
    release(s->omega);
    release(s->sstWallDistance);
    release(s->sstF1);
    release(s->sstF2);
    release(s->gradKX);
    release(s->gradKY);
    release(s->gradKZ);
    release(s->gradOmegaX);
    release(s->gradOmegaY);
    release(s->gradOmegaZ);
    release(s->sstPhiRhoK);
    release(s->sstPhiRhoOmega);
    release(s->sstBoundaryKMode);
    release(s->sstBoundaryOmegaMode);
    release(s->sstBoundaryK);
    release(s->sstBoundaryOmega);
    release(s->sstSourceNumber);
    release(s->Ux);
    release(s->Uy);
    release(s->Uz);
    release(s->p);
    release(s->Tgas);
    release(s->couplingRhoOld);
    release(s->couplingUxOld);
    release(s->couplingUyOld);
    release(s->couplingUzOld);
    release(s->couplingTgasOld);

    release(s->epsS);
    release(s->rhoUsx);
    release(s->rhoUsy);
    release(s->rhoUsz);
    release(s->rhoEs);
    release(s->rhoDs);
    release(s->rhoHp);
    release(s->Usx);
    release(s->Usy);
    release(s->Usz);
    release(s->theta);
    release(s->Tp);
    release(s->dMeanCell);
    release(s->epsGPrev);
    release(s->collisionalPressure);
    release(s->pressureKickScale);
    release(s->pressureDeltaMomX);
    release(s->pressureDeltaMomY);
    release(s->pressureDeltaMomZ);
    release(s->pressureDeltaEnergy);
    release(s->solidPressurePhiMomX);
    release(s->solidPressurePhiMomY);
    release(s->solidPressurePhiMomZ);
    release(s->solidPressurePhiEnergy);
    release(s->mobilePackingRho);
    release(s->mobilePackingMomX);
    release(s->mobilePackingMomY);
    release(s->mobilePackingMomZ);
    release(s->mobilePackingActiveCellMask);
    release(s->mobilePackingCorrectionCellMask);
    release(s->mobilePackingActiveCellList);
    release(s->mobilePackingCorrectionCellList);
    release(s->mobilePackingFrontierCurrent);
    release(s->mobilePackingFrontierNext);
    release(s->mobilePackingActiveCellCount);
    release(s->mobilePackingCorrectionCellCount);
    release(s->mobilePackingFrontierCurrentCount);
    release(s->mobilePackingFrontierNextCount);
    release(s->thetaDragAlpha);
    release(s->momRhoP);
    release(s->momRhoUPx);
    release(s->momRhoUPy);
    release(s->momRhoUPz);
    release(s->momRhoEP);
    release(s->momRhoPD);
    release(s->momRhoHpP);
    release(s->poolThermalCount);
    release(s->poolThermalSumUx);
    release(s->poolThermalSumUy);
    release(s->poolThermalSumUz);
    release(s->poolThermalSumU2);
    release(s->poissonPoolSampleTargetCount);
    release(s->poissonPoolMass);
    release(s->poissonPoolMomX);
    release(s->poissonPoolMomY);
    release(s->poissonPoolMomZ);
    release(s->poissonPoolEnergy);
    release(s->poissonPoolDiameter);
    release(s->poissonPoolDiameter2);

    release(s->px);
    release(s->py);
    release(s->pz);
    release(s->pux);
    release(s->puy);
    release(s->puz);
    release(s->puxOld);
    release(s->puyOld);
    release(s->puzOld);
    release(s->pT);
    release(s->pTheta);
    release(s->pd);
    release(s->pm);
    release(s->pCellId);
    release(s->pStatus);
    release(s->pRng);
    release(s->pOrigId);
    release(s->sortedParticleIndex);
    release(s->cellParticleCount);
    release(s->cellParticleOffset);
    release(s->preBaseCellOffset);
    release(s->preBaseParticleCountDevice);
    release(s->compactCellOffset);
    release(s->cellParticleWrite);
    release(s->cellScanTempStorage);
    release(s->compactCountDevice);
    release(s->compactSelectTempStorage);
    release(s->csrCellTaskCount);
    release(s->csrCellTaskOffset);
    release(s->csrReductionTasks);
    release(s->csrMultiTaskCellList);
    release(s->csrHeavyTaskCount);
    release(s->csrHeavyTaskCursor);
    release(s->csrHeavyCellCount);
    release(s->csrHeavyCellList);
    release(s->csrHeavyTaskCell);
    release(s->csrHeavyTaskBegin);
    release(s->csrHeavyTaskEnd);
    release(s->csrHeavyCellTaskStart);
    release(s->csrHeavyCellTaskCount);
    release(s->csrHeavyPartials);
    release(s->preInjectionHeavyTaskCount);
    release(s->preInjectionHeavyTaskCell);
    release(s->preInjectionHeavyTaskBegin);
    release(s->preInjectionHeavyTaskEnd);
    release(s->preInjectionHeavyCellTaskStart);
    release(s->preInjectionHeavyCellTaskCount);
    release(s->preInjectionHeavyPartials);

    release(s->sourceCell);
    release(s->sourceFace);
    release(s->sourcePx);
    release(s->sourcePy);
    release(s->sourcePz);
    release(s->sourceUx);
    release(s->sourceUy);
    release(s->sourceUz);
    release(s->sourceT);
    release(s->sourceTheta);
    release(s->sourceD);
    release(s->sourceMassRate);
    release(s->sourceResidualMass);
#ifdef UGKP_DEVELOPMENT_PROBES
    release(s->sourceInjectedCount);
#endif

    release(s->compactPx);
    release(s->compactPy);
    release(s->compactPz);
    release(s->compactPux);
    release(s->compactPuy);
    release(s->compactPuz);
    release(s->compactPT);
    release(s->compactPTheta);
    release(s->compactPd);
    release(s->compactPm);
    release(s->compactPCellId);
    release(s->compactPStatus);
    release(s->compactPRng);
    release(s->compactPOrigId);

    delete s;
}

int allocateFields(DeviceState* s)
{
    const size_t nc = static_cast<size_t>(s->nCells);
    const size_t nf = static_cast<size_t>(s->nFaces);
    const size_t nPlanes = static_cast<size_t>(s->nCellPlanes);
    const size_t np = static_cast<size_t>(s->particleCapacity);
    int rc = 0;

    rc |= allocate(s->deviceState, 1, "cudaMalloc strict deviceState");
    rc |= allocate(s->faceOwner, nf, "cudaMalloc strict faceOwner");
    rc |= allocate(s->faceNeighbour, nf, "cudaMalloc strict faceNeighbour");
    rc |= allocate(s->facePeriodicPair, nf, "cudaMalloc strict facePeriodicPair");
    rc |= allocate(s->facePeriodicDx, nf, "cudaMalloc strict facePeriodicDx");
    rc |= allocate(s->facePeriodicDy, nf, "cudaMalloc strict facePeriodicDy");
    rc |= allocate(s->facePeriodicDz, nf, "cudaMalloc strict facePeriodicDz");
    rc |= allocate(s->V, nc, "cudaMalloc strict V");
    rc |= allocate(s->Cx, nc, "cudaMalloc strict Cx");
    rc |= allocate(s->Cy, nc, "cudaMalloc strict Cy");
    rc |= allocate(s->Cz, nc, "cudaMalloc strict Cz");
    rc |= allocate(s->faceCx, nf, "cudaMalloc strict faceCx");
    rc |= allocate(s->faceCy, nf, "cudaMalloc strict faceCy");
    rc |= allocate(s->faceCz, nf, "cudaMalloc strict faceCz");
    rc |= allocate(s->Sfx, nf, "cudaMalloc strict Sfx");
    rc |= allocate(s->Sfy, nf, "cudaMalloc strict Sfy");
    rc |= allocate(s->Sfz, nf, "cudaMalloc strict Sfz");
    rc |= allocate(s->magSf, nf, "cudaMalloc strict magSf");
    rc |= allocate(s->deltaCoeffs, nf, "cudaMalloc strict deltaCoeffs");
    rc |= allocate(s->faceWeight, nf, "cudaMalloc strict faceWeight");
    rc |= allocate(s->cellLength, nc, "cudaMalloc strict cellLength");
    rc |= allocate(s->cellPlaneStart, nc, "cudaMalloc strict cellPlaneStart");
    rc |= allocate(s->cellPlaneCount, nc, "cudaMalloc strict cellPlaneCount");
    rc |= allocate(s->cellFaceId, nPlanes, "cudaMalloc strict cellFaceId");
    rc |= allocate(s->cellFaceNeighbor, nPlanes, "cudaMalloc strict cellFaceNeighbor");
    rc |= allocate(s->cellFaceKind, nPlanes, "cudaMalloc strict cellFaceKind");
    rc |= allocate(s->cellFaceRestitution, nPlanes, "cudaMalloc strict cellFaceRestitution");
    rc |= allocate(s->cellFaceTangential, nPlanes, "cudaMalloc strict cellFaceTangential");
    rc |= allocate(s->planeNx, nPlanes, "cudaMalloc strict planeNx");
    rc |= allocate(s->planeNy, nPlanes, "cudaMalloc strict planeNy");
    rc |= allocate(s->planeNz, nPlanes, "cudaMalloc strict planeNz");
    rc |= allocate(s->planeD, nPlanes, "cudaMalloc strict planeD");
    rc |= allocate(s->particleCountDevice, 1, "cudaMalloc strict particleCountDevice");
#ifdef UGKP_DEVELOPMENT_PROBES
    rc |= allocate
    (
        s->diagnosticPreTransportParticleCount,
        1,
        "cudaMalloc diagnostic pre-transport particle count"
    );
#endif

    rc |= allocate(s->rho, nc, "cudaMalloc strict rho");
    rc |= allocate(s->rhoUx, nc, "cudaMalloc strict rhoUx");
    rc |= allocate(s->rhoUy, nc, "cudaMalloc strict rhoUy");
    rc |= allocate(s->rhoUz, nc, "cudaMalloc strict rhoUz");
    rc |= allocate(s->rhoE, nc, "cudaMalloc strict rhoE");
    rc |= allocate(s->rhoNext, nc, "cudaMalloc strict rhoNext");
    rc |= allocate(s->rhoUxNext, nc, "cudaMalloc strict rhoUxNext");
    rc |= allocate(s->rhoUyNext, nc, "cudaMalloc strict rhoUyNext");
    rc |= allocate(s->rhoUzNext, nc, "cudaMalloc strict rhoUzNext");
    rc |= allocate(s->rhoENext, nc, "cudaMalloc strict rhoENext");
    rc |= allocate(s->gasPhiRho, nf, "cudaMalloc strict gasPhiRho");
    rc |= allocate(s->gasPhiRhoUx, nf, "cudaMalloc strict gasPhiRhoUx");
    rc |= allocate(s->gasPhiRhoUy, nf, "cudaMalloc strict gasPhiRhoUy");
    rc |= allocate(s->gasPhiRhoUz, nf, "cudaMalloc strict gasPhiRhoUz");
    rc |= allocate(s->gasPhiRhoE, nf, "cudaMalloc strict gasPhiRhoE");
    rc |= allocate(s->gasFluxPositivityScale, nc, "cudaMalloc strict gasFluxPositivityScale");
    rc |= allocate(s->gasHllcAdcSensor, nc, "cudaMalloc strict gasHllcAdcSensor");
    rc |= allocate(s->gasBoundaryKind, nf, "cudaMalloc strict gasBoundaryKind");
    rc |= allocate(s->gasBoundaryRhoFix, nf, "cudaMalloc strict gasBoundaryRhoFix");
    rc |= allocate(s->gasBoundaryUFix, nf, "cudaMalloc strict gasBoundaryUFix");
    rc |= allocate(s->gasBoundaryPFix, nf, "cudaMalloc strict gasBoundaryPFix");
    rc |= allocate(s->gasBoundaryTFix, nf, "cudaMalloc strict gasBoundaryTFix");
    rc |= allocate(s->gasBoundaryPWave, nf, "cudaMalloc strict gasBoundaryPWave");
    rc |= allocate(s->gasBoundaryPWaveGamma, nf, "cudaMalloc strict gasBoundaryPWaveGamma");
    rc |= allocate(s->gasBoundaryPWaveFieldInf, nf, "cudaMalloc strict gasBoundaryPWaveFieldInf");
    rc |= allocate(s->gasBoundaryPWaveLInf, nf, "cudaMalloc strict gasBoundaryPWaveLInf");
    rc |= allocate(s->gasBoundaryRho, nf, "cudaMalloc strict gasBoundaryRho");
    rc |= allocate(s->gasBoundaryUx, nf, "cudaMalloc strict gasBoundaryUx");
    rc |= allocate(s->gasBoundaryUy, nf, "cudaMalloc strict gasBoundaryUy");
    rc |= allocate(s->gasBoundaryUz, nf, "cudaMalloc strict gasBoundaryUz");
    rc |= allocate(s->gasBoundaryP, nf, "cudaMalloc strict gasBoundaryP");
    rc |= allocate(s->gasBoundaryT, nf, "cudaMalloc strict gasBoundaryT");
    rc |= allocate(s->riemannBoundaryKind, nf, "cudaMalloc strict riemannBoundaryKind");
    rc |= allocate(s->riemannBoundaryRhoFix, nf, "cudaMalloc strict riemannBoundaryRhoFix");
    rc |= allocate(s->riemannBoundaryUFix, nf, "cudaMalloc strict riemannBoundaryUFix");
    rc |= allocate(s->riemannBoundaryPFix, nf, "cudaMalloc strict riemannBoundaryPFix");
    rc |= allocate(s->riemannBoundaryTFix, nf, "cudaMalloc strict riemannBoundaryTFix");
    rc |= allocate(s->riemannBoundaryPWave, nf, "cudaMalloc strict riemannBoundaryPWave");
    rc |= allocate(s->riemannBoundaryPWaveGamma, nf, "cudaMalloc strict riemannBoundaryPWaveGamma");
    rc |= allocate(s->riemannBoundaryPWaveFieldInf, nf, "cudaMalloc strict riemannBoundaryPWaveFieldInf");
    rc |= allocate(s->riemannBoundaryPWaveLInf, nf, "cudaMalloc strict riemannBoundaryPWaveLInf");
    rc |= allocate(s->riemannBoundaryRho, nf, "cudaMalloc strict riemannBoundaryRho");
    rc |= allocate(s->riemannBoundaryUx, nf, "cudaMalloc strict riemannBoundaryUx");
    rc |= allocate(s->riemannBoundaryUy, nf, "cudaMalloc strict riemannBoundaryUy");
    rc |= allocate(s->riemannBoundaryUz, nf, "cudaMalloc strict riemannBoundaryUz");
    rc |= allocate(s->riemannBoundaryP, nf, "cudaMalloc strict riemannBoundaryP");
    rc |= allocate(s->riemannBoundaryT, nf, "cudaMalloc strict riemannBoundaryT");
    rc |= allocate(s->gradPx, nc, "cudaMalloc strict gradPx");
    rc |= allocate(s->gradPy, nc, "cudaMalloc strict gradPy");
    rc |= allocate(s->gradPz, nc, "cudaMalloc strict gradPz");
    rc |= allocate(s->gradRhoX, nc, "cudaMalloc strict gradRhoX");
    rc |= allocate(s->gradRhoY, nc, "cudaMalloc strict gradRhoY");
    rc |= allocate(s->gradRhoZ, nc, "cudaMalloc strict gradRhoZ");
    rc |= allocate(s->gradUxX, nc, "cudaMalloc strict gradUxX");
    rc |= allocate(s->gradUxY, nc, "cudaMalloc strict gradUxY");
    rc |= allocate(s->gradUxZ, nc, "cudaMalloc strict gradUxZ");
    rc |= allocate(s->gradUyX, nc, "cudaMalloc strict gradUyX");
    rc |= allocate(s->gradUyY, nc, "cudaMalloc strict gradUyY");
    rc |= allocate(s->gradUyZ, nc, "cudaMalloc strict gradUyZ");
    rc |= allocate(s->gradUzX, nc, "cudaMalloc strict gradUzX");
    rc |= allocate(s->gradUzY, nc, "cudaMalloc strict gradUzY");
    rc |= allocate(s->gradUzZ, nc, "cudaMalloc strict gradUzZ");
    rc |= allocate(s->gradTX, nc, "cudaMalloc strict gradTX");
    rc |= allocate(s->gradTY, nc, "cudaMalloc strict gradTY");
    rc |= allocate(s->gradTZ, nc, "cudaMalloc strict gradTZ");
    rc |= allocate(s->gasGradientLimiterRho, nc, "cudaMalloc strict gasGradientLimiterRho");
    rc |= allocate(s->gasGradientLimiterUx, nc, "cudaMalloc strict gasGradientLimiterUx");
    rc |= allocate(s->gasGradientLimiterUy, nc, "cudaMalloc strict gasGradientLimiterUy");
    rc |= allocate(s->gasGradientLimiterUz, nc, "cudaMalloc strict gasGradientLimiterUz");
    rc |= allocate(s->gasGradientLimiterP, nc, "cudaMalloc strict gasGradientLimiterP");
    rc |= allocate(s->gasGradientLimiterT, nc, "cudaMalloc strict gasGradientLimiterT");
    rc |= allocate(s->nut, nc, "cudaMalloc strict nut");
    rc |= allocate(s->gasDiffusionNumber, nc, "cudaMalloc strict gasDiffusionNumber");
    if (s->hostTurbulenceModel == 3)
    {
        rc |= allocate(s->rhoK, nc, "cudaMalloc SST rhoK");
        rc |= allocate(s->rhoOmega, nc, "cudaMalloc SST rhoOmega");
        rc |= allocate(s->rhoKInitial, nc, "cudaMalloc SST rhoKInitial");
        rc |= allocate(s->rhoOmegaInitial, nc, "cudaMalloc SST rhoOmegaInitial");
        rc |= allocate(s->k, nc, "cudaMalloc SST k");
        rc |= allocate(s->omega, nc, "cudaMalloc SST omega");
        rc |= allocate(s->sstWallDistance, nc, "cudaMalloc SST wallDistance");
        rc |= allocate(s->sstF1, nc, "cudaMalloc SST F1");
        rc |= allocate(s->sstF2, nc, "cudaMalloc SST F2");
        rc |= allocate(s->gradKX, nc, "cudaMalloc SST gradKX");
        rc |= allocate(s->gradKY, nc, "cudaMalloc SST gradKY");
        rc |= allocate(s->gradKZ, nc, "cudaMalloc SST gradKZ");
        rc |= allocate(s->gradOmegaX, nc, "cudaMalloc SST gradOmegaX");
        rc |= allocate(s->gradOmegaY, nc, "cudaMalloc SST gradOmegaY");
        rc |= allocate(s->gradOmegaZ, nc, "cudaMalloc SST gradOmegaZ");
        rc |= allocate(s->sstPhiRhoK, nf, "cudaMalloc SST phiRhoK");
        rc |= allocate(s->sstPhiRhoOmega, nf, "cudaMalloc SST phiRhoOmega");
        rc |= allocate(s->sstBoundaryKMode, nf, "cudaMalloc SST boundaryKMode");
        rc |= allocate(s->sstBoundaryOmegaMode, nf, "cudaMalloc SST boundaryOmegaMode");
        rc |= allocate(s->sstBoundaryK, nf, "cudaMalloc SST boundaryK");
        rc |= allocate(s->sstBoundaryOmega, nf, "cudaMalloc SST boundaryOmega");
        rc |= allocate(s->sstSourceNumber, nc, "cudaMalloc SST sourceNumber");
    }
    rc |= allocate(s->Ux, nc, "cudaMalloc strict Ux");
    rc |= allocate(s->Uy, nc, "cudaMalloc strict Uy");
    rc |= allocate(s->Uz, nc, "cudaMalloc strict Uz");
    rc |= allocate(s->p, nc, "cudaMalloc strict p");
    rc |= allocate(s->Tgas, nc, "cudaMalloc strict Tgas");
    rc |= allocate(s->couplingRhoOld, nc, "cudaMalloc strict couplingRhoOld");
    rc |= allocate(s->couplingUxOld, nc, "cudaMalloc strict couplingUxOld");
    rc |= allocate(s->couplingUyOld, nc, "cudaMalloc strict couplingUyOld");
    rc |= allocate(s->couplingUzOld, nc, "cudaMalloc strict couplingUzOld");
    rc |= allocate(s->couplingTgasOld, nc, "cudaMalloc strict couplingTgasOld");

    rc |= allocate(s->epsS, nc, "cudaMalloc strict epsS");
    rc |= allocate(s->rhoUsx, nc, "cudaMalloc strict rhoUsx");
    rc |= allocate(s->rhoUsy, nc, "cudaMalloc strict rhoUsy");
    rc |= allocate(s->rhoUsz, nc, "cudaMalloc strict rhoUsz");
    rc |= allocate(s->rhoEs, nc, "cudaMalloc strict rhoEs");
    rc |= allocate(s->rhoDs, nc, "cudaMalloc strict rhoDs");
    rc |= allocate(s->rhoHp, nc, "cudaMalloc strict rhoHp");
    rc |= allocate(s->Usx, nc, "cudaMalloc strict Usx");
    rc |= allocate(s->Usy, nc, "cudaMalloc strict Usy");
    rc |= allocate(s->Usz, nc, "cudaMalloc strict Usz");
    rc |= allocate(s->theta, nc, "cudaMalloc strict theta");
    rc |= allocate(s->Tp, nc, "cudaMalloc strict Tp");
    rc |= allocate(s->dMeanCell, nc, "cudaMalloc strict dMeanCell");
    rc |= allocate(s->epsGPrev, nc, "cudaMalloc strict epsGPrev");
    rc |= allocate(s->collisionalPressure, nc, "cudaMalloc strict collisionalPressure");
    rc |= allocate(s->pressureKickScale, nc, "cudaMalloc strict pressureKickScale");
    rc |= allocate(s->pressureDeltaMomX, nc, "cudaMalloc strict pressureDeltaMomX");
    rc |= allocate(s->pressureDeltaMomY, nc, "cudaMalloc strict pressureDeltaMomY");
    rc |= allocate(s->pressureDeltaMomZ, nc, "cudaMalloc strict pressureDeltaMomZ");
    rc |= allocate(s->pressureDeltaEnergy, nc, "cudaMalloc strict pressureDeltaEnergy");
    rc |= allocate(s->solidPressurePhiMomX, nf, "cudaMalloc strict solidPressurePhiMomX");
    rc |= allocate(s->solidPressurePhiMomY, nf, "cudaMalloc strict solidPressurePhiMomY");
    rc |= allocate(s->solidPressurePhiMomZ, nf, "cudaMalloc strict solidPressurePhiMomZ");
    rc |= allocate(s->solidPressurePhiEnergy, nf, "cudaMalloc strict solidPressurePhiEnergy");
    rc |= allocate(s->mobilePackingRho, nc, "cudaMalloc mobile packing density");
    rc |= allocate(s->mobilePackingMomX, nc, "cudaMalloc mobile packing momentum x");
    rc |= allocate(s->mobilePackingMomY, nc, "cudaMalloc mobile packing momentum y");
    rc |= allocate(s->mobilePackingMomZ, nc, "cudaMalloc mobile packing momentum z");
    rc |= allocate(s->mobilePackingActiveCellMask, nc, "cudaMalloc mobile packing active-cell mask");
    rc |= allocate(s->mobilePackingCorrectionCellMask, nc, "cudaMalloc mobile packing correction-cell mask");
    rc |= allocate(s->mobilePackingActiveCellList, nc, "cudaMalloc mobile packing active-cell list");
    rc |= allocate(s->mobilePackingCorrectionCellList, nc, "cudaMalloc mobile packing correction-cell list");
    rc |= allocate(s->mobilePackingFrontierCurrent, nc, "cudaMalloc mobile packing current frontier");
    rc |= allocate(s->mobilePackingFrontierNext, nc, "cudaMalloc mobile packing next frontier");
    rc |= allocate(s->mobilePackingActiveCellCount, 1, "cudaMalloc mobile packing active-cell count");
    rc |= allocate(s->mobilePackingCorrectionCellCount, 1, "cudaMalloc mobile packing correction-cell count");
    rc |= allocate(s->mobilePackingFrontierCurrentCount, 1, "cudaMalloc mobile packing current-frontier count");
    rc |= allocate(s->mobilePackingFrontierNextCount, 1, "cudaMalloc mobile packing next-frontier count");
    rc |= allocate(s->thetaDragAlpha, nc, "cudaMalloc strict thetaDragAlpha");
    rc |= allocate(s->momRhoP, nc, "cudaMalloc strict momRhoP");
    rc |= allocate(s->momRhoUPx, nc, "cudaMalloc strict momRhoUPx");
    rc |= allocate(s->momRhoUPy, nc, "cudaMalloc strict momRhoUPy");
    rc |= allocate(s->momRhoUPz, nc, "cudaMalloc strict momRhoUPz");
    rc |= allocate(s->momRhoEP, nc, "cudaMalloc strict momRhoEP");
    rc |= allocate(s->momRhoPD, nc, "cudaMalloc strict momRhoPD");
    rc |= allocate(s->momRhoHpP, nc, "cudaMalloc strict momRhoHpP");
    rc |= allocate(s->poolThermalCount, nc, "cudaMalloc strict poolThermalCount");
    rc |= allocate(s->poolThermalSumUx, nc, "cudaMalloc strict poolThermalSumUx");
    rc |= allocate(s->poolThermalSumUy, nc, "cudaMalloc strict poolThermalSumUy");
    rc |= allocate(s->poolThermalSumUz, nc, "cudaMalloc strict poolThermalSumUz");
    rc |= allocate(s->poolThermalSumU2, nc, "cudaMalloc strict poolThermalSumU2");
    rc |= allocate(s->poissonPoolSampleTargetCount, nc, "cudaMalloc strict poissonPoolSampleTargetCount");
    rc |= allocate(s->poissonPoolMass, nc, "cudaMalloc strict poissonPoolMass");
    rc |= allocate(s->poissonPoolMomX, nc, "cudaMalloc strict poissonPoolMomX");
    rc |= allocate(s->poissonPoolMomY, nc, "cudaMalloc strict poissonPoolMomY");
    rc |= allocate(s->poissonPoolMomZ, nc, "cudaMalloc strict poissonPoolMomZ");
    rc |= allocate(s->poissonPoolEnergy, nc, "cudaMalloc strict poissonPoolEnergy");
    rc |= allocate(s->poissonPoolDiameter, nc, "cudaMalloc strict poissonPoolDiameter");
    rc |= allocate(s->poissonPoolDiameter2, nc, "cudaMalloc strict poissonPoolDiameter2");

    rc |= allocate(s->px, np, "cudaMalloc strict particle x");
    rc |= allocate(s->py, np, "cudaMalloc strict particle y");
    rc |= allocate(s->pz, np, "cudaMalloc strict particle z");
    rc |= allocate(s->pux, np, "cudaMalloc strict particle ux");
    rc |= allocate(s->puy, np, "cudaMalloc strict particle uy");
    rc |= allocate(s->puz, np, "cudaMalloc strict particle uz");
    rc |= allocate(s->puxOld, np, "cudaMalloc strict particle old ux");
    rc |= allocate(s->puyOld, np, "cudaMalloc strict particle old uy");
    rc |= allocate(s->puzOld, np, "cudaMalloc strict particle old uz");
    rc |= allocate(s->pT, np, "cudaMalloc strict particle T");
    rc |= allocate(s->pTheta, np, "cudaMalloc strict particle theta");
    rc |= allocate(s->pd, np, "cudaMalloc strict particle d");
    rc |= allocate(s->pm, np, "cudaMalloc strict particle m");
    rc |= allocate(s->pCellId, np, "cudaMalloc strict particle cellId");
    rc |= allocate(s->pStatus, np, "cudaMalloc strict particle status");
    rc |= allocate(s->pRng, np, "cudaMalloc strict particle rng");
    rc |= allocate(s->pOrigId, np, "cudaMalloc strict particle origId");
    rc |= allocate(s->sortedParticleIndex, np, "cudaMalloc strict sorted particle index");
    rc |= allocate(s->cellParticleCount, nc + 1, "cudaMalloc strict cellParticleCount");
    rc |= allocate(s->cellParticleOffset, nc + 1, "cudaMalloc strict cellParticleOffset");
    rc |= allocate(s->preBaseCellOffset, nc + 1, "cudaMalloc strict preBaseCellOffset");
    rc |= allocate(s->preBaseParticleCountDevice, 1, "cudaMalloc strict preBaseParticleCountDevice");
    rc |= allocate(s->compactCellOffset, nc + 1, "cudaMalloc strict compactCellOffset");
    rc |= allocate(s->cellParticleWrite, nc, "cudaMalloc strict cellParticleWrite");
    if (s->csrHeavyReductionEnabled != 0 && s->particleCapacity > 0)
    {
        const size_t segmentedTaskCapacity =
            (np + static_cast<size_t>(s->reductionBlockThreads) - 1)
           /static_cast<size_t>(s->reductionBlockThreads)
          + 2u*nc
          + 1u;
        if (segmentedTaskCapacity > static_cast<size_t>(2147483647))
        {
            setLastErrorText("segmented task capacity exceeds 32-bit indexing");
            releaseState(s);
            return 1;
        }
        s->csrHeavyTaskCapacity = static_cast<int>(segmentedTaskCapacity);
        rc |= allocate(s->csrCellTaskCount, nc + 1u, "cudaMalloc CSR cell task count");
        rc |= allocate(s->csrCellTaskOffset, nc + 1u, "cudaMalloc CSR cell task offset");
        rc |= allocate(s->csrReductionTasks, segmentedTaskCapacity, "cudaMalloc CSR reduction tasks");
        rc |= allocate(s->csrMultiTaskCellList, nc, "cudaMalloc CSR multi-task cell list");
        rc |= allocate(s->csrHeavyTaskCount, 1, "cudaMalloc CSR heavy task count");
        rc |= allocate(s->csrHeavyTaskCursor, 1, "cudaMalloc CSR heavy task cursor");
        rc |= allocate(s->csrHeavyCellCount, 1, "cudaMalloc CSR heavy cell count");
        rc |= allocate(s->csrHeavyPartials, 8u*segmentedTaskCapacity, "cudaMalloc CSR heavy partials");
    }
    rc |= allocate(s->compactPx, np, "cudaMalloc strict compact particle x");
    rc |= allocate(s->compactPy, np, "cudaMalloc strict compact particle y");
    rc |= allocate(s->compactPz, np, "cudaMalloc strict compact particle z");
    rc |= allocate(s->compactPux, np, "cudaMalloc strict compact particle ux");
    rc |= allocate(s->compactPuy, np, "cudaMalloc strict compact particle uy");
    rc |= allocate(s->compactPuz, np, "cudaMalloc strict compact particle uz");
    rc |= allocate(s->compactPT, np, "cudaMalloc strict compact particle T");
    rc |= allocate(s->compactPTheta, np, "cudaMalloc strict compact particle theta");
    rc |= allocate(s->compactPd, np, "cudaMalloc strict compact particle d");
    rc |= allocate(s->compactPm, np, "cudaMalloc strict compact particle m");
    rc |= allocate(s->compactPCellId, np, "cudaMalloc strict compact particle cellId");
    rc |= allocate(s->compactPStatus, np, "cudaMalloc strict compact particle status");
    rc |= allocate(s->compactPRng, np, "cudaMalloc strict compact particle rng");
    rc |= allocate(s->compactPOrigId, np, "cudaMalloc strict compact particle origId");
    rc |= allocate(s->compactCountDevice, 1, "cudaMalloc strict selected particle count");

    if (rc != 0)
    {
        releaseState(s);
        return 1;
    }

    int multiprocessorCount = 1;
    cudaError_t attrErr = cudaDeviceGetAttribute
    (
        &multiprocessorCount,
        cudaDevAttrMultiProcessorCount,
        0
    );
    if (attrErr != cudaSuccess)
    {
        setLastError("cudaDeviceGetAttribute multiprocessor count", attrErr);
        releaseState(s);
        return 1;
    }
    s->multiprocessorCount = multiprocessorCount;
    const int capacityGrid =
        (s->particleCapacity + s->particleBlockThreads - 1)
       /s->particleBlockThreads;
    const int saturatedGrid = multiprocessorCount;
    s->particleWorkGrid = capacityGrid < saturatedGrid
      ? capacityGrid
      : saturatedGrid;
    s->csrHeavyWorkerGrid = multiprocessorCount;

    cudaError_t scanErr = cub::DeviceScan::ExclusiveSum
    (
        nullptr,
        s->cellScanTempBytes,
        s->cellParticleCount,
        s->cellParticleOffset,
        s->nCells + 1
    );
    if (scanErr != cudaSuccess)
    {
        setLastError("query cell scan temporary storage", scanErr);
        releaseState(s);
        return 1;
    }
    if
    (
        allocate
        (
            s->cellScanTempStorage,
            s->cellScanTempBytes,
            "cudaMalloc strict cellScanTempStorage"
        ) != 0
    )
    {
        releaseState(s);
        return 1;
    }
    if (s->particleCapacity > 0)
    {
        const thrust::counting_iterator<int> particleIndices(0);
        const ActiveParticleIndexPredicate predicate{s->deviceState};
        cudaError_t selectErr = cub::DeviceSelect::If
        (
            nullptr,
            s->compactSelectTempBytes,
            particleIndices,
            s->sortedParticleIndex,
            s->compactCountDevice,
            s->particleCapacity,
            predicate
        );
        if (selectErr != cudaSuccess)
        {
            setLastError("query particle DeviceSelect temporary storage", selectErr);
            releaseState(s);
            return 1;
        }
        if
        (
            allocate
            (
                s->compactSelectTempStorage,
                s->compactSelectTempBytes,
                "cudaMalloc strict compactSelectTempStorage"
            ) != 0
        )
        {
            releaseState(s);
            return 1;
        }
    }
    cudaError_t err =
        cudaMemset(s->particleCountDevice, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict particleCountDevice", err);
        releaseState(s);
        return 1;
    }
#ifdef UGKP_DEVELOPMENT_PROBES
    err = cudaMemset
    (
        s->diagnosticPreTransportParticleCount,
        0,
        sizeof(int)
    );
    if (err != cudaSuccess)
    {
        setLastError
        (
            "cudaMemset diagnostic pre-transport particle count",
            err
        );
        releaseState(s);
        return 1;
    }
#endif
    if (syncDeviceState(s, "cudaMemcpy strict deviceState") != 0)
    {
        releaseState(s);
        return 1;
    }
    return 0;
}

__device__ double clampMin(const double x, const double lo)
{
    return x < lo ? lo : x;
}

__device__ bool finiteDevice(const double x)
{
    return (x == x) && (fabs(x) < 1.0e300);
}

__device__ bool nonFiniteDevice(const double x)
{
    return !finiteDevice(x);
}

__device__ double finiteOr(const double x, const double fallback)
{
    return finiteDevice(x) ? x : fallback;
}

#ifdef UGKP_DEVELOPMENT_PROBES

__device__ void recordDevelopmentProbeCellFailure
(
    DevelopmentProbeDeviceSummary* summary,
    const int cell,
    const unsigned long long mask
)
{
    if (mask == 0)
    {
        return;
    }

    atomicAdd(&summary->badCells, 1ull);
    atomicOr(&summary->badFieldMask, mask);
    atomicCAS(&summary->firstBadCellPlusOne, 0, cell + 1);
}

__device__ void recordDevelopmentProbeParticleFailure
(
    DevelopmentProbeDeviceSummary* summary,
    const int particle,
    const unsigned long long mask
)
{
    if (mask == 0)
    {
        return;
    }

    atomicAdd(&summary->badParticles, 1ull);
    atomicOr(&summary->badFieldMask, mask);
    atomicCAS(&summary->firstBadParticlePlusOne, 0, particle + 1);
}

__global__ void validateDevelopmentProbeCellsKernel
(
    DeviceState* sp,
    DevelopmentProbeDeviceSummary* summary,
    const int validateSolid
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        unsigned long long mask = 0;

        if
        (
            !finiteDevice(s.rho[c])
         || !finiteDevice(s.rhoUx[c])
         || !finiteDevice(s.rhoUy[c])
         || !finiteDevice(s.rhoUz[c])
         || !finiteDevice(s.rhoE[c])
        )
        {
            mask |= ProbeBadGasConserved;
        }

        if
        (
            !finiteDevice(s.Ux[c])
         || !finiteDevice(s.Uy[c])
         || !finiteDevice(s.Uz[c])
         || !finiteDevice(s.p[c])
         || !finiteDevice(s.Tgas[c])
        )
        {
            mask |= ProbeBadGasPrimitive;
        }

        if
        (
            !(s.rho[c] > 0.0)
         || !(s.p[c] > 0.0)
         || !(s.Tgas[c] > 0.0)
        )
        {
            mask |= ProbeBadCellRange;
        }

        if (validateSolid != 0)
        {
            if
            (
                !finiteDevice(s.epsS[c])
             || !finiteDevice(s.rhoUsx[c])
             || !finiteDevice(s.rhoUsy[c])
             || !finiteDevice(s.rhoUsz[c])
             || !finiteDevice(s.rhoEs[c])
             || !finiteDevice(s.rhoDs[c])
             || !finiteDevice(s.rhoHp[c])
            )
            {
                mask |= ProbeBadSolidConserved;
            }

            if
            (
                !finiteDevice(s.Usx[c])
             || !finiteDevice(s.Usy[c])
             || !finiteDevice(s.Usz[c])
             || !finiteDevice(s.theta[c])
             || !finiteDevice(s.Tp[c])
             || !finiteDevice(s.dMeanCell[c])
            )
            {
                mask |= ProbeBadSolidPrimitive;
            }

            if
            (
                !finiteDevice(s.momRhoP[c])
             || !finiteDevice(s.momRhoUPx[c])
             || !finiteDevice(s.momRhoUPy[c])
             || !finiteDevice(s.momRhoUPz[c])
             || !finiteDevice(s.momRhoEP[c])
             || !finiteDevice(s.momRhoPD[c])
             || !finiteDevice(s.momRhoHpP[c])
            )
            {
                mask |= ProbeBadSolidMoments;
            }

            if
            (
                s.epsS[c] < 0.0
             || s.epsS[c] > 1.0
             || s.theta[c] < 0.0
             || (s.solveParticleTemperature != 0 && !(s.Tp[c] > 0.0))
             || (s.epsS[c] > s.epsSMin && !(s.dMeanCell[c] > 0.0))
            )
            {
                mask |= ProbeBadCellRange;
            }
        }

        recordDevelopmentProbeCellFailure(summary, c, mask);
    }
}

__global__ void validateDevelopmentProbeParticlesKernel
(
    DeviceState* sp,
    DevelopmentProbeDeviceSummary* summary
)
{
    DeviceState& s = *sp;
    int nParticles = *s.particleCountDevice;
    nParticles = nParticles < 0 ? 0 : nParticles;
    nParticles = nParticles > s.particleCapacity ? s.particleCapacity : nParticles;

    const int stride = blockDim.x*gridDim.x;
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += stride
    )
    {
        unsigned long long mask = 0;
        if
        (
            !finiteDevice(s.px[i])
         || !finiteDevice(s.py[i])
         || !finiteDevice(s.pz[i])
        )
        {
            mask |= ProbeBadParticlePosition;
        }
        if
        (
            !finiteDevice(s.pux[i])
         || !finiteDevice(s.puy[i])
         || !finiteDevice(s.puz[i])
        )
        {
            mask |= ProbeBadParticleVelocity;
        }
        if
        (
            !finiteDevice(s.pT[i])
         || !finiteDevice(s.pTheta[i])
         || !finiteDevice(s.pd[i])
         || !finiteDevice(s.pm[i])
         || s.pTheta[i] < 0.0
         || !(s.pd[i] > 0.0)
         || (s.pStatus[i] != 0 && !(s.pm[i] > 0.0))
         || (s.solveParticleTemperature != 0 && !(s.pT[i] > 0.0))
        )
        {
            mask |= ProbeBadParticleThermal;
        }
        if
        (
            s.pCellId[i] < 0
         || s.pCellId[i] >= s.nCells
         || s.pStatus[i] != 1
        )
        {
            mask |= ProbeBadParticleMetadata;
        }

        recordDevelopmentProbeParticleFailure(summary, i, mask);
    }
}

#endif

__device__ double particleHeatFactorDevice(const DeviceState& s)
{
    if
    (
        s.solveParticleTemperature == 0
     || s.particleThermalRho <= 0.0
     || s.particleCp <= 0.0
    )
    {
        return 0.0;
    }

    return s.particleThermalRho*s.particleCp
       /clampMin(s.rhoSolid, 1.0e-300);
}

__device__ void clearSolidCell(DeviceState& s, const int c)
{
    s.epsS[c] = 0.0;
    s.rhoUsx[c] = 0.0;
    s.rhoUsy[c] = 0.0;
    s.rhoUsz[c] = 0.0;
    s.rhoEs[c] = 0.0;
    s.rhoDs[c] = 0.0;
    s.rhoHp[c] = 0.0;
    s.Usx[c] = 0.0;
    s.Usy[c] = 0.0;
    s.Usz[c] = 0.0;
    s.theta[c] = 0.0;
    s.Tp[c] = s.TpMin;
    s.dMeanCell[c] =
        clampMin(finiteOr(s.particleDiameterFallback, 1.0e-12), 1.0e-12);
}

__device__ double clampRange(const double x, const double lo, const double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ double linearScheduledValueDevice
(
    const double* times,
    const double* values,
    const int count,
    const double simulationTime
)
{
    if (count <= 0 || times == nullptr || values == nullptr)
    {
        return 0.0;
    }
    if (simulationTime <= times[0])
    {
        return values[0];
    }
    if (simulationTime >= times[count - 1])
    {
        return values[count - 1];
    }
    int low = 0;
    int high = count - 1;
    while (low + 1 < high)
    {
        const int middle = low + (high - low)/2;
        if (times[middle] <= simulationTime)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    const double t0 = times[low];
    const double t1 = times[high];
    const double fraction = clampRange
    (
        (simulationTime - t0)/clampMin(t1 - t0, OfSmall),
        0.0,
        1.0
    );
    return values[low] + fraction*(values[high] - values[low]);
}

__device__ bool scheduledInletFaceDevice
(
    const DeviceState& s,
    const int face
)
{
    return
        s.nScheduledInletFaces > 0
     && s.scheduledInletFaceMask != nullptr
     && face >= s.nInternalFaces
     && face < s.nFaces
     && s.scheduledInletFaceMask[face] != 0;
}

__device__ double scheduledPressureDevice
(
    const DeviceState& s,
    const double simulationTime
)
{
    return linearScheduledValueDevice
    (
        s.pressureScheduleTimes,
        s.pressureScheduleValues,
        s.nPressureScheduleRows,
        simulationTime
    );
}

__device__ double scheduledSolidVolumeFractionDevice
(
    const DeviceState& s,
    const double simulationTime
)
{
    return clampRange
    (
        linearScheduledValueDevice
        (
            s.volumeFractionScheduleTimes,
            s.volumeFractionScheduleValues,
            s.nVolumeFractionScheduleRows,
            simulationTime
        ),
        0.0,
        1.0 - OfSmall
    );
}

__global__ void publishScheduledInletConfigurationKernel
(
    DeviceState* sp,
    const int nFaces,
    int* faceMask,
    const double inletTemperature,
    const int nPressureRows,
    double* pressureTimes,
    double* pressureValues,
    const int nVolumeFractionRows,
    double* volumeFractionTimes,
    double* volumeFractionValues,
    const int particlesMayBePresent
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    DeviceState& s = *sp;
    s.nScheduledInletFaces = nFaces;
    s.scheduledInletFaceMask = faceMask;
    s.scheduledInletTemperature = inletTemperature;
    s.nPressureScheduleRows = nPressureRows;
    s.pressureScheduleTimes = pressureTimes;
    s.pressureScheduleValues = pressureValues;
    s.nVolumeFractionScheduleRows = nVolumeFractionRows;
    s.volumeFractionScheduleTimes = volumeFractionTimes;
    s.volumeFractionScheduleValues = volumeFractionValues;
    if (particlesMayBePresent != 0)
    {
        s.particlesMayBePresent = true;
    }
}

__device__ double sqr3(const double x, const double y, const double z)
{
    return x*x + y*y + z*z;
}

struct GasPrimDevice
{
    double rho;
    double ux;
    double uy;
    double uz;
    double p;
    double T;
};

__device__ GasPrimDevice makeGasPrimDevice
(
    const double rho,
    const double ux,
    const double uy,
    const double uz,
    const double p,
    const double Rgas,
    const double rhoMin,
    const double Tmin
)
{
    GasPrimDevice g;
    g.rho = clampMin(finiteOr(rho, rhoMin), rhoMin);
    g.ux = finiteOr(ux, 0.0);
    g.uy = finiteOr(uy, 0.0);
    g.uz = finiteOr(uz, 0.0);
    const double pMinimum =
        g.rho*clampMin(Rgas, OfSmall)*clampMin(Tmin, OfSmall);
    g.p = clampMin(finiteOr(p, pMinimum), pMinimum);
    g.T = g.p/clampMin(g.rho*Rgas, OfSmall);
    return g;
}

__device__ bool useRiemannBoundaryVelocity
(
    const DeviceState& s,
    const int f,
    const GasPrimDevice& ownerState
)
{
    const int mode = s.riemannBoundaryUFix[f];
    if (mode == 1)
    {
        return true;
    }
    if (mode != 2)
    {
        return false;
    }

                                                                           
    const double area = clampMin(s.magSf[f], OfSmall);
    const double outwardVelocity =
        (ownerState.ux*s.Sfx[f]
       + ownerState.uy*s.Sfy[f]
       + ownerState.uz*s.Sfz[f])/area;
    return outwardVelocity < 0.0;
}

__device__ GasPrimDevice riemannBoundaryState
(
    const DeviceState& s,
    const int f,
    const GasPrimDevice& ownerState
)
{
    if (s.riemannBoundaryUFix[f] == 3)
    {
        const double totalTemperature = clampMin
        (
            finiteOr(s.riemannBoundaryT[f], ownerState.T),
            s.TgasMin
        );
        const double totalPressure = clampMin
        (
            finiteOr(s.riemannBoundaryP[f], ownerState.p),
            s.rhoMin*s.Rgas*totalTemperature
        );
        const double area = clampMin(s.magSf[f], OfSmall);
        const ugkpboundary::Primitive boundary =
            ugkpboundary::totalConditionInletState
            (
                ugkpboundary::Primitive
                {
                    ownerState.rho,
                    ownerState.ux,
                    ownerState.uy,
                    ownerState.uz,
                    ownerState.p,
                    ownerState.T
                },
                s.Sfx[f]/area,
                s.Sfy[f]/area,
                s.Sfz[f]/area,
                totalPressure,
                totalTemperature,
                s.gammaGas,
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
        return makeGasPrimDevice
        (
            boundary.rho,
            boundary.ux,
            boundary.uy,
            boundary.uz,
            boundary.p,
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }

    const bool velocityFixed =
        useRiemannBoundaryVelocity(s, f, ownerState);
    const bool rhoFixed = s.riemannBoundaryRhoFix[f] != 0;
    const bool pFixed =
        s.riemannBoundaryPFix[f] != 0
     || s.riemannBoundaryPWave[f] != 0;
    const bool TFixed = s.riemannBoundaryTFix[f] != 0;

    double rho = ownerState.rho;
    double p = ownerState.p;
    double T = ownerState.T;
    if (rhoFixed)
    {
        rho = clampMin
        (
            finiteOr(s.riemannBoundaryRho[f], ownerState.rho),
            s.rhoMin
        );
    }
    if (pFixed)
    {
        p = clampMin
        (
            finiteOr(s.riemannBoundaryP[f], ownerState.p),
            rho*s.Rgas*s.TgasMin
        );
    }
    if (TFixed)
    {
        T = clampMin
        (
            finiteOr(s.riemannBoundaryT[f], ownerState.T),
            s.TgasMin
        );
    }

                                                                             
                                                                               
                                                         
    if (pFixed && TFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }
    else if (rhoFixed && TFixed)
    {
        p = rho*s.Rgas*T;
    }
    else if (rhoFixed && pFixed)
    {
        T = p/clampMin(rho*s.Rgas, OfSmall);
    }
    else if (pFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }
    else if (rhoFixed)
    {
        p = rho*s.Rgas*T;
    }
    else if (TFixed)
    {
        rho = p/clampMin(s.Rgas*T, OfSmall);
    }

    const double ux = velocityFixed
      ? s.riemannBoundaryUx[f] : ownerState.ux;
    const double uy = velocityFixed
      ? s.riemannBoundaryUy[f] : ownerState.uy;
    const double uz = velocityFixed
      ? s.riemannBoundaryUz[f] : ownerState.uz;
    return makeGasPrimDevice
    (
        rho, ux, uy, uz, p, s.Rgas, s.rhoMin, s.TgasMin
    );
}

__device__ bool isPeriodicFace(const DeviceState& s, const int f)
{
    return
        f >= s.nInternalFaces
     && f < s.nFaces
     && s.facePeriodicPair[f] >= s.nInternalFaces
     && s.facePeriodicPair[f] < s.nFaces;
}

__device__ int coupledFaceNeighbour(const DeviceState& s, const int f)
{
    return (f < s.nInternalFaces || isPeriodicFace(s, f))
      ? s.faceNeighbour[f] : -1;
}

__device__ void periodicMappedCellCentre
(
    const DeviceState& s,
    const int f,
    const int c,
    double& x,
    double& y,
    double& z
)
{
    x = s.Cx[c];
    y = s.Cy[c];
    z = s.Cz[c];
    if (isPeriodicFace(s, f) && c == s.faceNeighbour[f])
    {
        x += s.facePeriodicDx[f];
        y += s.facePeriodicDy[f];
        z += s.facePeriodicDz[f];
    }
}

__device__ GasPrimDevice riemannExteriorStateForFace
(
    const DeviceState& s,
    const int f,
    const GasPrimDevice& ownerFaceState
)
{
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        return makeGasPrimDevice
        (
            s.rho[nei],
            s.Ux[nei],
            s.Uy[nei],
            s.Uz[nei],
            s.p[nei],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }
    return riemannBoundaryState(s, f, ownerFaceState);
}

__device__ GasPrimDevice riemannFacePrimitiveForGradient
(
    const DeviceState& s,
    const int c,
    const int f
)
{
    GasPrimDevice centre = makeGasPrimDevice
    (
        s.rho[c], s.Ux[c], s.Uy[c], s.Uz[c], s.p[c],
        s.Rgas, s.rhoMin, s.TgasMin
    );
    centre.T = clampMin(finiteOr(s.Tgas[c], centre.T), s.TgasMin);
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const int other = c == own ? nei : own;
        if (other < 0 || other >= s.nCells)
        {
            return centre;
        }
        GasPrimDevice adjacent = makeGasPrimDevice
        (
            s.rho[other], s.Ux[other], s.Uy[other], s.Uz[other], s.p[other],
            s.Rgas, s.rhoMin, s.TgasMin
        );
        adjacent.T = clampMin
        (
            finiteOr(s.Tgas[other], adjacent.T),
            s.TgasMin
        );
        const double ownerWeight = clampRange(s.faceWeight[f], 0.0, 1.0);
        const double wc = c == own ? ownerWeight : 1.0 - ownerWeight;
        GasPrimDevice face = makeGasPrimDevice
        (
            wc*centre.rho + (1.0 - wc)*adjacent.rho,
            wc*centre.ux + (1.0 - wc)*adjacent.ux,
            wc*centre.uy + (1.0 - wc)*adjacent.uy,
            wc*centre.uz + (1.0 - wc)*adjacent.uz,
            wc*centre.p + (1.0 - wc)*adjacent.p,
            s.Rgas, s.rhoMin, s.TgasMin
        );
        face.T = wc*centre.T + (1.0 - wc)*adjacent.T;
        return face;
    }

    const int kind = s.riemannBoundaryKind[f];
    if (kind == 4 || kind == 3)
    {
        return centre;
    }
    if (kind == 1)
    {
        const double area = clampMin(s.magSf[f], OfSmall);
        const double nx = s.Sfx[f]/area;
        const double ny = s.Sfy[f]/area;
        const double nz = s.Sfz[f]/area;
        const double un = centre.ux*nx + centre.uy*ny + centre.uz*nz;
        GasPrimDevice face = centre;
        face.ux -= un*nx;
        face.uy -= un*ny;
        face.uz -= un*nz;
        return face;
    }

    if (kind == 2)
    {
        GasPrimDevice wall = centre;
        wall.ux = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
        wall.uy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
        wall.uz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
        if (s.riemannBoundaryTFix[f] != 0)
        {
            wall.T = clampMin
            (
                finiteOr(s.riemannBoundaryT[f], centre.T),
                s.TgasMin
            );
            wall.rho = wall.p/clampMin(s.Rgas*wall.T, OfSmall);
        }
        return wall;
    }

    return riemannBoundaryState(s, f, centre);
}

__global__ void computeGasPrimitiveGradientsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double grx = 0.0, gry = 0.0, grz = 0.0;
    double guxx = 0.0, guxy = 0.0, guxz = 0.0;
    double guyx = 0.0, guyy = 0.0, guyz = 0.0;
    double guzx = 0.0, guzy = 0.0, guzz = 0.0;
    double gpx = 0.0, gpy = 0.0, gpz = 0.0;
    double gtx = 0.0, gty = 0.0, gtz = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double sign = s.faceOwner[f] == c ? 1.0 : -1.0;
        const double sx = sign*s.Sfx[f];
        const double sy = sign*s.Sfy[f];
        const double sz = sign*s.Sfz[f];
        const GasPrimDevice qf =
            riemannFacePrimitiveForGradient(s, c, f);
        grx += qf.rho*sx; gry += qf.rho*sy; grz += qf.rho*sz;
        guxx += qf.ux*sx; guxy += qf.ux*sy; guxz += qf.ux*sz;
        guyx += qf.uy*sx; guyy += qf.uy*sy; guyz += qf.uy*sz;
        guzx += qf.uz*sx; guzy += qf.uz*sy; guzz += qf.uz*sz;
        gpx += qf.p*sx; gpy += qf.p*sy; gpz += qf.p*sz;
        gtx += qf.T*sx; gty += qf.T*sy; gtz += qf.T*sz;
    }
    const double invV = 1.0/clampMin(s.V[c], OfSmall);
    s.gradRhoX[c] = grx*invV; s.gradRhoY[c] = gry*invV; s.gradRhoZ[c] = grz*invV;
    s.gradUxX[c] = guxx*invV; s.gradUxY[c] = guxy*invV; s.gradUxZ[c] = guxz*invV;
    s.gradUyX[c] = guyx*invV; s.gradUyY[c] = guyy*invV; s.gradUyZ[c] = guyz*invV;
    s.gradUzX[c] = guzx*invV; s.gradUzY[c] = guzy*invV; s.gradUzZ[c] = guzz*invV;
    s.gradPx[c] = gpx*invV; s.gradPy[c] = gpy*invV; s.gradPz[c] = gpz*invV;
    s.gradTX[c] = gtx*invV; s.gradTY[c] = gty*invV; s.gradTZ[c] = gtz*invV;
}

__device__ double sstDynamicOmegaWallValue
(
    const DeviceState& s,
    const int f,
    const int owner
)
{
    const double rhoSafe = clampMin(s.rho[owner], s.rhoMin);
    const double nu = s.gasMu/rhoSafe;
    const double wallUx = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
    const double wallUy = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
    const double wallUz = s.riemannBoundaryUFix[f] != 0
      ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
    const double dux = s.Ux[owner] - wallUx;
    const double duy = s.Uy[owner] - wallUy;
    const double duz = s.Uz[owner] - wallUz;
    const double y = clampMin(s.sstWallDistance[owner], OfVSmall);
    const double magGradU = sqrt(dux*dux + duy*duy + duz*duz)/y;
    return ugkpwall::omegaWallFunctionState
    (
        s.k[owner],
        magGradU,
        y,
        nu,
        s.sstCoefficients.beta1,
        s.sstWallCmu,
        s.sstWallKappa,
        s.sstWallE
    ).omega;
}

__device__ double sstBoundaryValue
(
    const DeviceState& s,
    const int f,
    const int owner,
    const bool omegaField
)
{
    const double centre = omegaField ? s.omega[owner] : s.k[owner];
    const int boundaryKind = s.riemannBoundaryKind[f];
    if (boundaryKind == 2)
    {
        if (!omegaField)
        {
            return s.sstWallTreatment == 0 ? 0.0 : centre;
        }
        if (s.sstWallTreatment == 0)
        {
            const double rhoSafe = clampMin(s.rho[owner], s.rhoMin);
            return ugkwp::sstLowReWallOmega
            (
                s.gasMu/rhoSafe,
                s.sstWallDistance[owner],
                s.sstCoefficients
            );
        }
        return centre;
    }
    if (boundaryKind == 1 || boundaryKind == 3 || boundaryKind == 4)
    {
        return centre;
    }

    const int mode = omegaField
      ? s.sstBoundaryOmegaMode[f] : s.sstBoundaryKMode[f];
    const double prescribed = omegaField
      ? s.sstBoundaryOmega[f] : s.sstBoundaryK[f];
    if (mode == 1)
    {
        return prescribed;
    }
    if (mode == 2)
    {
        const double outwardMassDirection =
            s.Ux[owner]*s.Sfx[f]
          + s.Uy[owner]*s.Sfy[f]
          + s.Uz[owner]*s.Sfz[f];
        return outwardMassDirection >= 0.0 ? centre : prescribed;
    }
    return centre;
}

__global__ void applySstWallFunctionStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if
    (
        c >= s.nCells
     || s.sstConfigured == 0
     || s.sstWallTreatment != 1
    )
    {
        return;
    }

    double omegaSum = 0.0;
    int wallCount = 0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f >= s.nInternalFaces
         && f < s.nFaces
         && s.riemannBoundaryKind[f] == 2
        )
        {
            omegaSum += sstDynamicOmegaWallValue(s, f, c);
            ++wallCount;
        }
    }
    if (wallCount > 0)
    {
        const double rhoSafe = clampMin(s.rho[c], s.rhoMin);
        const double omegaTarget = clampMin
        (
            finiteOr(omegaSum/double(wallCount), s.sstOmegaMin),
            s.sstOmegaMin
        );
        s.omega[c] = omegaTarget;
        s.rhoOmega[c] = rhoSafe*omegaTarget;
    }
}

__global__ void initialiseSstConservativeStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c == 0 && s.sstConfigured != 0)
    {
        const double PrRatio =
            s.gasPrClamped/clampMin(s.turbulentPrandtl, OfSmall);
        s.sstJayatillekeP = ugkpwall::jayatillekeSmoothP(PrRatio);
        s.sstThermalYPlus = ugkpwall::jayatillekeThermalYPlus
        (
            PrRatio,
            s.sstWallKappa,
            s.sstWallE
        );
    }
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const double rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const double kSafe = clampMin(finiteOr(s.k[c], s.sstKMin), s.sstKMin);
    const double omegaSafe =
        clampMin(finiteOr(s.omega[c], s.sstOmegaMin), s.sstOmegaMin);
    s.k[c] = kSafe;
    s.omega[c] = omegaSafe;
    s.rhoK[c] = rhoSafe*kSafe;
    s.rhoOmega[c] = rhoSafe*omegaSafe;
    s.rhoKInitial[c] = s.rhoK[c];
    s.rhoOmegaInitial[c] = s.rhoOmega[c];
    s.nut[c] = 0.0;
}

__global__ void recoverSstPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }
    const double rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const double rhoKFloor = rhoSafe*s.sstKMin;
    const double rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.rhoK[c] = clampMin(finiteOr(s.rhoK[c], rhoKFloor), rhoKFloor);
    s.rhoOmega[c] =
        clampMin(finiteOr(s.rhoOmega[c], rhoOmegaFloor), rhoOmegaFloor);
    s.k[c] = s.rhoK[c]/rhoSafe;
    s.omega[c] = s.rhoOmega[c]/rhoSafe;
}

__global__ void computeSstGradientsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    double gkx = 0.0, gky = 0.0, gkz = 0.0;
    double gox = 0.0, goy = 0.0, goz = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double sign = s.faceOwner[f] == c ? 1.0 : -1.0;
        double kFace = s.k[c];
        double omegaFace = s.omega[c];
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = c == own ? nei : own;
            if (other >= 0 && other < s.nCells)
            {
                const double ownerWeight =
                    clampRange(s.faceWeight[f], 0.0, 1.0);
                const double cellWeight =
                    c == own ? ownerWeight : 1.0 - ownerWeight;
                kFace =
                    cellWeight*s.k[c] + (1.0 - cellWeight)*s.k[other];
                omegaFace =
                    cellWeight*s.omega[c]
                  + (1.0 - cellWeight)*s.omega[other];
            }
        }
        else
        {
            kFace = sstBoundaryValue(s, f, c, false);
            omegaFace = sstBoundaryValue(s, f, c, true);
        }
        const double sx = sign*s.Sfx[f];
        const double sy = sign*s.Sfy[f];
        const double sz = sign*s.Sfz[f];
        gkx += kFace*sx;
        gky += kFace*sy;
        gkz += kFace*sz;
        gox += omegaFace*sx;
        goy += omegaFace*sy;
        goz += omegaFace*sz;
    }
    const double invV = 1.0/clampMin(s.V[c], OfSmall);
    s.gradKX[c] = gkx*invV;
    s.gradKY[c] = gky*invV;
    s.gradKZ[c] = gkz*invV;
    s.gradOmegaX[c] = gox*invV;
    s.gradOmegaY[c] = goy*invV;
    s.gradOmegaZ[c] = goz*invV;
}

__device__ void sstVelocityInvariants
(
    const DeviceState& s,
    const int c,
    double& divU,
    double& s2,
    double& gByNu
)
{
    const double g[3][3] =
    {
        {s.gradUxX[c], s.gradUxY[c], s.gradUxZ[c]},
        {s.gradUyX[c], s.gradUyY[c], s.gradUyZ[c]},
        {s.gradUzX[c], s.gradUzY[c], s.gradUzZ[c]}
    };
    divU = g[0][0] + g[1][1] + g[2][2];
    double symmSquared = 0.0;
    gByNu = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double twoSymm = g[i][j] + g[j][i];
            const double devTwoSymm =
                twoSymm - (i == j ? (2.0/3.0)*divU : 0.0);
            const double symm = 0.5*twoSymm;
            symmSquared += symm*symm;
            gByNu += devTwoSymm*g[i][j];
        }
    }
    s2 = 2.0*symmSquared;
}

  
                                                              
  
                                                     
                                                                          
                                                                           
                                                                           
                                                                       
                                                                     
                                                                          
                                             
   
__global__ void computeGasHllcAdcSensorKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double centrePressure = clampMin(s.p[c], OfSmall);
    double omega = 1.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        double otherPressure = centrePressure;
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = own == c ? nei : own;
            if (other < 0 || other >= s.nCells)
            {
                continue;
            }
            otherPressure = clampMin(s.p[other], OfSmall);
        }
        else if (s.riemannBoundaryKind[f] == 0)
        {
            const GasPrimDevice centre = makeGasPrimDevice
            (
                s.rho[c],
                s.Ux[c],
                s.Uy[c],
                s.Uz[c],
                centrePressure,
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
            otherPressure = clampMin
            (
                riemannBoundaryState(s, f, centre).p,
                OfSmall
            );
        }

        const double ratio = clampRange
        (
            fmin
            (
                otherPressure/centrePressure,
                centrePressure/otherPressure
            ),
            0.0,
            1.0
        );
        const double faceSensor = ratio*ratio*ratio;
        omega = fmin(omega, faceSensor);
    }
    s.gasHllcAdcSensor[c] = finiteDevice(omega)
      ? clampRange(omega, 0.0, 1.0) : 0.0;
}

__device__ void updateBarthLimiter
(
    const double centre,
    const double predicted,
    const double minimum,
    const double maximum,
    double& limiter
)
{
    const double delta = predicted - centre;
    if (delta > OfSmall)
    {
        limiter = fmin(limiter, (maximum - centre)/delta);
    }
    else if (delta < -OfSmall)
    {
        limiter = fmin(limiter, (minimum - centre)/delta);
    }
}

__device__ void updateVenkatakrishnanLimiter
(
    const double centre,
    const double predicted,
    const double minimum,
    const double maximum,
    double& limiter
)
{
    const double delta = predicted - centre;
    if (fabs(delta) <= OfSmall)
    {
        return;
    }
    const double admissible =
        delta > 0.0 ? maximum - centre : minimum - centre;
    if (admissible*delta <= 0.0)
    {
        limiter = 0.0;
        return;
    }

                                                                          
                                                                    
    const double ratio = fmax(admissible/delta, 0.0);
    const double numerator = ratio*ratio + 2.0*ratio;
    const double denominator = ratio*ratio + ratio + 2.0;
    limiter = fmin
    (
        limiter,
        denominator > OfSmall ? numerator/denominator : 0.0
    );
}

__device__ void updateConfiguredGasLimiter
(
    const int limiterScheme,
    const double centre,
    const double predicted,
    const double minimum,
    const double maximum,
    double& limiter
)
{
    if (limiterScheme == 2)
    {
        updateVenkatakrishnanLimiter
        (
            centre, predicted, minimum, maximum, limiter
        );
    }
    else
    {
        updateBarthLimiter
        (
            centre, predicted, minimum, maximum, limiter
        );
    }
}

__global__ void computeGasGradientLimiterKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if (s.gasLimiter == 0)
    {
        s.gasGradientLimiterRho[c] = 1.0;
        s.gasGradientLimiterUx[c] = 1.0;
        s.gasGradientLimiterUy[c] = 1.0;
        s.gasGradientLimiterUz[c] = 1.0;
        s.gasGradientLimiterP[c] = 1.0;
        s.gasGradientLimiterT[c] = 1.0;
        return;
    }
    const GasPrimDevice qc = makeGasPrimDevice
    (
        s.rho[c], s.Ux[c], s.Uy[c], s.Uz[c], s.p[c],
        s.Rgas, s.rhoMin, s.TgasMin
    );
    double rMin = qc.rho, rMax = qc.rho;
    double uxMin = qc.ux, uxMax = qc.ux;
    double uyMin = qc.uy, uyMax = qc.uy;
    double uzMin = qc.uz, uzMax = qc.uz;
    double pMin = qc.p, pMax = qc.p;
    double tMin = qc.T, tMax = qc.T;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        GasPrimDevice qf;
        if (f < s.nInternalFaces || isPeriodicFace(s, f))
        {
            const int own = s.faceOwner[f];
            const int nei = s.faceNeighbour[f];
            const int other = c == own ? nei : own;
            qf = makeGasPrimDevice
            (
                s.rho[other],
                s.Ux[other],
                s.Uy[other],
                s.Uz[other],
                s.p[other],
                s.Rgas,
                s.rhoMin,
                s.TgasMin
            );
            qf.T = clampMin
            (
                finiteOr(s.Tgas[other], qf.T),
                s.TgasMin
            );
        }
        else
        {
            qf = riemannFacePrimitiveForGradient(s, c, f);
        }
        rMin = fmin(rMin, qf.rho); rMax = fmax(rMax, qf.rho);
        uxMin = fmin(uxMin, qf.ux); uxMax = fmax(uxMax, qf.ux);
        uyMin = fmin(uyMin, qf.uy); uyMax = fmax(uyMax, qf.uy);
        uzMin = fmin(uzMin, qf.uz); uzMax = fmax(uzMax, qf.uz);
        pMin = fmin(pMin, qf.p); pMax = fmax(pMax, qf.p);
        tMin = fmin(tMin, qf.T); tMax = fmax(tMax, qf.T);
    }
    rMin = fmax(rMin, s.rhoMin);
    pMin = fmax(pMin, s.rhoMin);
    tMin = fmax(tMin, s.TgasMin);
    double rhoLimiter = 1.0;
    double uxLimiter = 1.0;
    double uyLimiter = 1.0;
    double uzLimiter = 1.0;
    double pLimiter = 1.0;
    double tLimiter = 1.0;
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        const double dx = s.faceCx[f] - s.Cx[c];
        const double dy = s.faceCy[f] - s.Cy[c];
        const double dz = s.faceCz[f] - s.Cz[c];
        updateConfiguredGasLimiter(s.gasLimiter, qc.rho, qc.rho + s.gradRhoX[c]*dx + s.gradRhoY[c]*dy + s.gradRhoZ[c]*dz, rMin, rMax, rhoLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.ux, qc.ux + s.gradUxX[c]*dx + s.gradUxY[c]*dy + s.gradUxZ[c]*dz, uxMin, uxMax, uxLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.uy, qc.uy + s.gradUyX[c]*dx + s.gradUyY[c]*dy + s.gradUyZ[c]*dz, uyMin, uyMax, uyLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.uz, qc.uz + s.gradUzX[c]*dx + s.gradUzY[c]*dy + s.gradUzZ[c]*dz, uzMin, uzMax, uzLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.p, qc.p + s.gradPx[c]*dx + s.gradPy[c]*dy + s.gradPz[c]*dz, pMin, pMax, pLimiter);
        updateConfiguredGasLimiter(s.gasLimiter, qc.T, qc.T + s.gradTX[c]*dx + s.gradTY[c]*dy + s.gradTZ[c]*dz, tMin, tMax, tLimiter);
    }
    s.gasGradientLimiterRho[c] =
        clampRange(finiteOr(rhoLimiter, 0.0), 0.0, 1.0);
    s.gasGradientLimiterUx[c] =
        clampRange(finiteOr(uxLimiter, 0.0), 0.0, 1.0);
    s.gasGradientLimiterUy[c] =
        clampRange(finiteOr(uyLimiter, 0.0), 0.0, 1.0);
    s.gasGradientLimiterUz[c] =
        clampRange(finiteOr(uzLimiter, 0.0), 0.0, 1.0);
    s.gasGradientLimiterP[c] =
        clampRange(finiteOr(pLimiter, 0.0), 0.0, 1.0);
    s.gasGradientLimiterT[c] =
        clampRange(finiteOr(tLimiter, 0.0), 0.0, 1.0);
}

__global__ void computeGasEddyViscosityKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if (s.turbulenceModel == 0)
    {
        s.nut[c] = 0.0;
        return;
    }
    if (s.turbulenceModel == 3)
    {
        if (s.sstConfigured == 0)
        {
            asm("trap;");
            return;
        }
        double divU = 0.0;
        double s2 = 0.0;
        double gByNu = 0.0;
        sstVelocityInvariants(s, c, divU, s2, gByNu);
        (void)divU;
        (void)gByNu;
        const double rhoSafe = clampMin(s.rho[c], s.rhoMin);
        const double nu = s.gasMu/rhoSafe;
        const double gradDot =
            s.gradKX[c]*s.gradOmegaX[c]
          + s.gradKY[c]*s.gradOmegaY[c]
          + s.gradKZ[c]*s.gradOmegaZ[c];
        const double cd = ugkwp::sstCrossDiffusion
        (
            s.omega[c],
            gradDot,
            s.sstCoefficients
        );
        const double f1 = ugkwp::sstF1
        (
            s.k[c],
            s.omega[c],
            nu,
            s.sstWallDistance[c],
            cd,
            s.sstCoefficients
        );
        const double f2 = ugkwp::sstF2
        (
            s.k[c],
            s.omega[c],
            nu,
            s.sstWallDistance[c],
            s.sstCoefficients
        );
        const double nuT = ugkwp::sstNut
        (
            s.k[c],
            s.omega[c],
            s2,
            f2,
            s.sstCoefficients
        );
        s.sstF1[c] = clampRange(finiteOr(f1, 1.0), 0.0, 1.0);
        s.sstF2[c] = clampRange(finiteOr(f2, 1.0), 0.0, 1.0);
        s.nut[c] = finiteDevice(nuT) && nuT > 0.0 ? nuT : 0.0;
        return;
    }
    double g[3][3] =
    {
        {s.gradUxX[c], s.gradUxY[c], s.gradUxZ[c]},
        {s.gradUyX[c], s.gradUyY[c], s.gradUyZ[c]},
        {s.gradUzX[c], s.gradUzY[c], s.gradUzZ[c]}
    };
    const double tr = g[0][0] + g[1][1] + g[2][2];
    double ss = 0.0;
    double symmetricGradientSquared = 0.0;
    double strain[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double symmetricGradient =
                0.5*(g[i][j] + g[j][i]);
            symmetricGradientSquared +=
                symmetricGradient*symmetricGradient;
            strain[i][j] = symmetricGradient
              - (i == j ? tr/3.0 : 0.0);
            ss += strain[i][j]*strain[i][j];
        }
    }
    const double delta = s.lesDeltaCoeff*clampMin(s.cellLength[c], OfSmall);
    double nuT = 0.0;
    if (s.turbulenceModel == 2)
    {
        nuT = s.smagorinskyCs*s.smagorinskyCs*delta*delta
          *sqrt(fmax(2.0*ss, 0.0));
    }
    else
    {
        double g2[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                g2[i][j] = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    g2[i][j] += g[i][k]*g[k][j];
                }
            }
        }
        const double trG2 = g2[0][0] + g2[1][1] + g2[2][2];
        double sd2 = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                const double sd = 0.5*(g2[i][j] + g2[j][i])
                  - (i == j ? trG2/3.0 : 0.0);
                sd2 += sd*sd;
            }
        }
        const double numerator = pow(fmax(sd2, 0.0), 1.5);
                                                                 
                                                                           
                                                                             
                                   
        const double denominator =
            pow(fmax(symmetricGradientSquared, 0.0), 2.5)
          + pow(fmax(sd2, 0.0), 1.25);
        nuT = denominator > OfSmall
          ? s.waleCw*s.waleCw*delta*delta*numerator/denominator
          : 0.0;
    }
    s.nut[c] = finiteDevice(nuT) && nuT > 0.0 ? nuT : 0.0;
}

__global__ void updateWaveTransmissivePressureBoundaryKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces || s.gasBoundaryPWave[f] == 0)
    {
        return;
    }

    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        return;
    }

    const double area = clampMin(s.magSf[f], 1.0e-300);
    double boundaryUx = s.gasBoundaryUx[f];
    double boundaryUy = s.gasBoundaryUy[f];
    double boundaryUz = s.gasBoundaryUz[f];
    if (s.gasBoundaryUFix[f] == 2)
    {
        const double ownerOutwardVelocity =
            s.Ux[own]*s.Sfx[f]
          + s.Uy[own]*s.Sfy[f]
          + s.Uz[own]*s.Sfz[f];
        if (ownerOutwardVelocity >= 0.0)
        {
            boundaryUx = s.Ux[own];
            boundaryUy = s.Uy[own];
            boundaryUz = s.Uz[own];
        }
    }
    const double phip =
        boundaryUx*s.Sfx[f]
      + boundaryUy*s.Sfy[f]
      + boundaryUz*s.Sfz[f];
    const double psi =
        s.gasBoundaryRho[f]
       /clampMin(s.gasBoundaryP[f], OfSmall);
    double w =
        phip/area
      + sqrt(clampMin(s.gasBoundaryPWaveGamma[f]/clampMin(psi, OfSmall), 0.0));
    w = w > 0.0 ? w : 0.0;

    const double alpha = w*dt*s.deltaCoeffs[f];
    const double lInf = s.gasBoundaryPWaveLInf[f];
    const bool hasRelaxation = lInf > 1.0e-300;
    const double k = hasRelaxation ? w*dt/lInf : 0.0;
    const double oldBoundaryP = s.gasBoundaryP[f];
    const double refValue = hasRelaxation
      ? (oldBoundaryP + k*s.gasBoundaryPWaveFieldInf[f])/(1.0 + k)
      : oldBoundaryP;
    const double valueFraction = hasRelaxation
      ? (1.0 + k)/(1.0 + alpha + k)
      : 1.0/(1.0 + alpha);
    const double boundaryP =
        valueFraction*refValue
      + (1.0 - valueFraction)*s.p[own];

    const double updatedPressure =
        clampMin(finiteOr(boundaryP, s.p[own]), OfVSmall);
    s.gasBoundaryP[f] = updatedPressure;
    s.riemannBoundaryP[f] = updatedPressure;
}

__global__ void updateLegacyGasBoundaryMirrorKernel
(
    DeviceState* sp,
    const double simulationTime
)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces)
    {
        return;
    }

    if (scheduledInletFaceDevice(s, f))
    {
        const double temperature = clampMin
        (
            finiteOr(s.scheduledInletTemperature, s.TgasMin),
            s.TgasMin
        );
        const double pressure = clampMin
        (
            finiteOr(scheduledPressureDevice(s, simulationTime), OfSmall),
            s.rhoMin*s.Rgas*temperature
        );
        const double density = pressure/clampMin(s.Rgas*temperature, OfSmall);
        s.gasBoundaryP[f] = pressure;
        s.gasBoundaryRho[f] = density;
        s.gasBoundaryT[f] = temperature;
        s.riemannBoundaryP[f] = pressure;
        s.riemannBoundaryRho[f] = density;
        s.riemannBoundaryT[f] = temperature;
        s.riemannBoundaryUFix[f] = 3;
        return;
    }

    const int kind = s.gasBoundaryKind[f];
    if (kind == 3 || kind == 4)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int mirrorCell = kind == 5 ? coupledFaceNeighbour(s, f) : own;
    if (mirrorCell < 0 || mirrorCell >= s.nCells)
    {
        return;
    }

    if (s.gasBoundaryRhoFix[f] == 0)
    {
        s.gasBoundaryRho[f] = s.rho[mirrorCell];
    }
    if (s.gasBoundaryUFix[f] == 0)
    {
        s.gasBoundaryUx[f] = s.Ux[mirrorCell];
        s.gasBoundaryUy[f] = s.Uy[mirrorCell];
        s.gasBoundaryUz[f] = s.Uz[mirrorCell];
    }
    if (s.gasBoundaryTFix[f] == 0)
    {
        s.gasBoundaryT[f] = s.Tgas[mirrorCell];
    }
    if (s.gasBoundaryPFix[f] == 0 && s.gasBoundaryPWave[f] == 0)
    {
        s.gasBoundaryP[f] = s.p[mirrorCell];
    }
}

__global__ void updateRiemannBoundaryMirrorKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < s.nInternalFaces || f >= s.nFaces)
    {
        return;
    }

    const int kind = s.riemannBoundaryKind[f];
    if (kind == 3 || kind == 4)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int mirrorCell = kind == 5 ? coupledFaceNeighbour(s, f) : own;
    if (mirrorCell < 0 || mirrorCell >= s.nCells)
    {
        return;
    }

    if (s.riemannBoundaryRhoFix[f] == 0)
    {
        s.riemannBoundaryRho[f] = s.rho[mirrorCell];
    }
    if (s.riemannBoundaryUFix[f] == 0)
    {
        s.riemannBoundaryUx[f] = s.Ux[mirrorCell];
        s.riemannBoundaryUy[f] = s.Uy[mirrorCell];
        s.riemannBoundaryUz[f] = s.Uz[mirrorCell];
    }
    if (s.riemannBoundaryTFix[f] == 0)
    {
        s.riemannBoundaryT[f] = s.Tgas[mirrorCell];
    }
    if
    (
        s.riemannBoundaryPFix[f] == 0
     && s.riemannBoundaryPWave[f] == 0
    )
    {
        s.riemannBoundaryP[f] = s.p[mirrorCell];
    }
}

__device__ GasPrimDevice reconstructGasCellToFace
(
    const DeviceState& s,
    const int c,
    const int f
)
{
    if (s.gasReconstruction != 1)
    {
        return makeGasPrimDevice
        (
            s.rho[c],
            s.Ux[c],
            s.Uy[c],
            s.Uz[c],
            s.p[c],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }

    double mappedCx = 0.0;
    double mappedCy = 0.0;
    double mappedCz = 0.0;
    periodicMappedCellCentre(s, f, c, mappedCx, mappedCy, mappedCz);
    const double dx = s.faceCx[f] - mappedCx;
    const double dy = s.faceCy[f] - mappedCy;
    const double dz = s.faceCz[f] - mappedCz;
    double rhoIncrement = s.gasGradientLimiterRho[c]
      *(s.gradRhoX[c]*dx + s.gradRhoY[c]*dy + s.gradRhoZ[c]*dz);
    double pressureIncrement = s.gasGradientLimiterP[c]
      *(s.gradPx[c]*dx + s.gradPy[c]*dy + s.gradPz[c]*dz);
    double uxIncrement = s.gasGradientLimiterUx[c]
      *(s.gradUxX[c]*dx + s.gradUxY[c]*dy + s.gradUxZ[c]*dz);
    double uyIncrement = s.gasGradientLimiterUy[c]
      *(s.gradUyX[c]*dx + s.gradUyY[c]*dy + s.gradUyZ[c]*dz);
    double uzIncrement = s.gasGradientLimiterUz[c]
      *(s.gradUzX[c]*dx + s.gradUzY[c]*dy + s.gradUzZ[c]*dz);

                                                                             
                                                                       
                                                                           
                                                                        
    const double rho = s.rho[c] + rhoIncrement;
    const double ux = s.Ux[c] + uxIncrement;
    const double uy = s.Uy[c] + uyIncrement;
    const double uz = s.Uz[c] + uzIncrement;
    const double p = s.p[c] + pressureIncrement;
    return makeGasPrimDevice
    (
        rho, ux, uy, uz, p, s.Rgas, s.rhoMin, s.TgasMin
    );
}

__device__ double molecularGasConductivity(const DeviceState& s)
{
    return s.gasMu*s.gasCp/s.gasPrClamped;
}

__device__ void gasFaceSubgridTransportProperties
(
    const DeviceState& s,
    const int f,
    const int own,
    const int nei,
    const int boundaryKind,
    const double rhoFace,
    double& muTurbulent,
    double& kTurbulent,
    double& directWallHeatFlux,
    int& directWallHeatFluxActive
)
{
    directWallHeatFlux = 0.0;
    directWallHeatFluxActive = 0;
    double nutFace = nei >= 0
      ? 0.5*(s.nut[own] + s.nut[nei])
      : s.nut[own];

    if (nei < 0 && boundaryKind == 2)
    {
        if (s.turbulenceModel == 3 && s.sstWallTreatment == 0)
        {
            muTurbulent = 0.0;
            kTurbulent = 0.0;
            return;
        }
        const double wallUx = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
        const double wallUy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
        const double wallUz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
        const double dux = s.Ux[own] - wallUx;
        const double duy = s.Uy[own] - wallUy;
        const double duz = s.Uz[own] - wallUz;
        const double velocityDifference = sqrt
        (
            dux*dux + duy*duy + duz*duz
        );
        const double wallDistance = s.turbulenceModel == 3
          ? clampMin(s.sstWallDistance[own], OfVSmall)
          : 1.0/clampMin(s.deltaCoeffs[f], OfVSmall);
        const double rhoSafe = clampMin(rhoFace, s.rhoMin);
        const ugkpwall::SpaldingWallState wallState =
            ugkpwall::spaldingWallState
            (
                velocityDifference,
                wallDistance,
                s.gasMu/rhoSafe,
                s.turbulenceModel == 3 ? s.sstWallKappa : 0.41,
                s.turbulenceModel == 3 ? s.sstWallE : 9.8
            );
        if (s.turbulenceModel == 3)
        {
            const double wallTemperature = s.riemannBoundaryTFix[f] != 0
              ? s.riemannBoundaryT[f] : s.Tgas[own];
            const ugkpwall::JayatillekeWallHeatState heatState =
                ugkpwall::jayatillekeWallHeatFluxPrecomputed
                (
                    rhoSafe,
                    s.gasCp,
                    s.gasPrClamped,
                    s.turbulentPrandtl,
                    s.sstWallKappa,
                    s.sstWallE,
                    s.sstJayatillekeP,
                    s.sstThermalYPlus,
                    wallState.uTau,
                    wallState.yPlus,
                    s.Tgas[own],
                    wallTemperature
                );
            muTurbulent = rhoSafe*wallState.nut;
            directWallHeatFlux = heatState.heatFlux;
            directWallHeatFluxActive =
                s.riemannBoundaryTFix[f] != 0 && heatState.valid != 0;
            const double equivalentConductivity = heatState.valid != 0
              ? rhoSafe*s.gasCp*wallState.uTau
               /(clampMin(heatState.temperaturePlus, OfSmall)
                *clampMin(s.deltaCoeffs[f], OfSmall))
              : molecularGasConductivity(s);
            kTurbulent = fmax
            (
                equivalentConductivity - molecularGasConductivity(s),
                0.0
            );
            return;
        }
        const ugkpwall::WallSubgridTransport wallTransport =
            ugkpwall::wallSubgridTransport
            (
                rhoSafe,
                s.gasCp,
                s.turbulentPrandtl,
                wallState.nut
            );
        muTurbulent = wallTransport.dynamicViscosity;
        kTurbulent = wallTransport.thermalConductivity;
        return;
    }

    const double muT = clampMin(rhoFace, s.rhoMin)*fmax(nutFace, 0.0);
    muTurbulent = muT;
    kTurbulent =
        s.gasCp*muT/clampMin(s.turbulentPrandtl, OfSmall);
}

template<bool IncludeTurbulence>
__device__ bool computeRiemannGasFaceFluxDevice
(
    const DeviceState& s,
    const int f,
    double& massFluxArea,
    double& momFluxXArea,
    double& momFluxYArea,
    double& momFluxZArea,
    double& energyFluxArea
)
{
    massFluxArea = 0.0;
    momFluxXArea = 0.0;
    momFluxYArea = 0.0;
    momFluxZArea = 0.0;
    energyFluxArea = 0.0;

    if (f < 0 || f >= s.nFaces)
    {
        return false;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        return false;
    }

    const int nei = coupledFaceNeighbour(s, f);
    const int boundaryKind =
        nei >= 0 ? 0 : s.riemannBoundaryKind[f];
    double mappedNeiCx = 0.0;
    double mappedNeiCy = 0.0;
    double mappedNeiCz = 0.0;
    if (nei >= 0)
    {
        periodicMappedCellCentre
        (
            s, f, nei, mappedNeiCx, mappedNeiCy, mappedNeiCz
        );
    }
    if (boundaryKind == 3 || boundaryKind == 4)
    {
        return false;
    }

    const double area = clampMin(s.magSf[f], OfSmall);
    const double nx = s.Sfx[f]/area;
    const double ny = s.Sfy[f]/area;
    const double nz = s.Sfz[f]/area;
    GasPrimDevice left = reconstructGasCellToFace(s, own, f);
    GasPrimDevice right = left;

    double massFlux = 0.0;
    double momentumFluxX = 0.0;
    double momentumFluxY = 0.0;
    double momentumFluxZ = 0.0;
    double energyFlux = 0.0;

    if (boundaryKind == 1)
    {
                                                                          
                                                                            
        momentumFluxX = left.p*nx;
        momentumFluxY = left.p*ny;
        momentumFluxZ = left.p*nz;
    }
    else if (boundaryKind == 2)
    {
        right = riemannFacePrimitiveForGradient(s, own, f);
        const double wallUx = right.ux;
        const double wallUy = right.uy;
        const double wallUz = right.uz;
        momentumFluxX = left.p*nx;
        momentumFluxY = left.p*ny;
        momentumFluxZ = left.p*nz;
        energyFlux =
            momentumFluxX*wallUx
          + momentumFluxY*wallUy
          + momentumFluxZ*wallUz;
    }
    else
    {
        right = nei >= 0
          ? reconstructGasCellToFace(s, nei, f)
          : riemannExteriorStateForFace(s, f, left);
        if (s.gasReconstruction == 1 && nei >= 0)
        {
            const ugkpriemann::Primitive ownerCentre
            {
                s.rho[own], s.Ux[own], s.Uy[own], s.Uz[own], s.p[own]
            };
            const ugkpriemann::Primitive neighbourCentre
            {
                s.rho[nei], s.Ux[nei], s.Uy[nei], s.Uz[nei], s.p[nei]
            };
            const ugkpriemann::RoeAverage roe =
                ugkpriemann::makeRoeAverage
                (
                    ownerCentre,
                    neighbourCentre,
                    nx,
                    ny,
                    nz,
                    s.gammaGas
                );
            if (roe.valid)
            {
                ugkpcharacteristic::Increment leftIncrement
                {
                    left.rho - ownerCentre.rho,
                    left.ux - ownerCentre.ux,
                    left.uy - ownerCentre.uy,
                    left.uz - ownerCentre.uz,
                    left.p - ownerCentre.p
                };
                ugkpcharacteristic::Increment rightIncrement
                {
                    right.rho - neighbourCentre.rho,
                    right.ux - neighbourCentre.ux,
                    right.uy - neighbourCentre.uy,
                    right.uz - neighbourCentre.uz,
                    right.p - neighbourCentre.p
                };
                const ugkpcharacteristic::Increment centreDifference
                {
                    neighbourCentre.rho - ownerCentre.rho,
                    neighbourCentre.ux - ownerCentre.ux,
                    neighbourCentre.uy - ownerCentre.uy,
                    neighbourCentre.uz - ownerCentre.uz,
                    neighbourCentre.p - ownerCentre.p
                };
                ugkpcharacteristic::limitFacePair
                (
                    leftIncrement,
                    rightIncrement,
                    centreDifference,
                    nx,
                    ny,
                    nz,
                    roe.density,
                    roe.soundSpeed,
                    s.faceWeight[f]
                );
                left = makeGasPrimDevice
                (
                    ownerCentre.rho + leftIncrement.rho,
                    ownerCentre.ux + leftIncrement.ux,
                    ownerCentre.uy + leftIncrement.uy,
                    ownerCentre.uz + leftIncrement.uz,
                    ownerCentre.p + leftIncrement.p,
                    s.Rgas,
                    s.rhoMin,
                    s.TgasMin
                );
                right = makeGasPrimDevice
                (
                    neighbourCentre.rho + rightIncrement.rho,
                    neighbourCentre.ux + rightIncrement.ux,
                    neighbourCentre.uy + rightIncrement.uy,
                    neighbourCentre.uz + rightIncrement.uz,
                    neighbourCentre.p + rightIncrement.p,
                    s.Rgas,
                    s.rhoMin,
                    s.TgasMin
                );
            }
        }
        const ugkpriemann::Primitive leftRiemann
        {
            left.rho, left.ux, left.uy, left.uz, left.p
        };
        const ugkpriemann::Primitive rightRiemann
        {
            right.rho, right.ux, right.uy, right.uz, right.p
        };
        ugkpriemann::Scheme scheme;
        if (!ugkpriemann::schemeFromCreateCode(s.gasFluxScheme, scheme))
        {
            asm("trap;");
            return false;
        }
        double hllcAdcOmega = 1.0;
        if (scheme == ugkpriemann::Scheme::HLLC_ADC)
        {
            hllcAdcOmega = clampRange
            (
                finiteOr(s.gasHllcAdcSensor[own], 0.0),
                0.0,
                1.0
            );
            if (nei >= 0)
            {
                hllcAdcOmega = fmin
                (
                    hllcAdcOmega,
                    clampRange
                    (
                        finiteOr(s.gasHllcAdcSensor[nei], 0.0),
                        0.0,
                        1.0
                    )
                );
            }
        }
        ugkpriemann::FluxResult result;
        if (scheme == ugkpriemann::Scheme::SLAU2_2)
        {
            const int gradientNeighbour = nei >= 0 ? nei : own;
            result = ugkpriemann::slau22FluxUnitNormal
            (
                leftRiemann,
                rightRiemann,
                ugkpriemann::DensityGradient
                {
                    s.gradRhoX[own],
                    s.gradRhoY[own],
                    s.gradRhoZ[own]
                },
                ugkpriemann::DensityGradient
                {
                    s.gradRhoX[gradientNeighbour],
                    s.gradRhoY[gradientNeighbour],
                    s.gradRhoZ[gradientNeighbour]
                },
                nx,
                ny,
                nz,
                s.gammaGas,
                s.rhoMin,
                s.rhoMin*s.Rgas*s.TgasMin
            );
        }
        else
        {
            result = ugkpriemann::fluxUnitArea
            (
                leftRiemann,
                rightRiemann,
                nx,
                ny,
                nz,
                s.gammaGas,
                scheme,
                s.rhoMin,
                s.rhoMin*s.Rgas*s.TgasMin,
                hllcAdcOmega
            );
        }
        if (!result.valid)
        {
            asm("trap;");
            return false;
        }
        massFlux = result.flux[0];
        momentumFluxX = result.flux[1];
        momentumFluxY = result.flux[2];
        momentumFluxZ = result.flux[3];
        energyFlux = result.flux[4];
        if (s.gasReconstruction == 2 && nei >= 0)
        {
                                                                      
                                                                           
                                                                         
                                                                        
                                                    
                                                                           
                                                                       
                                                                          
                                                                           
                                                                     
            const double ownerVelocitySquared =
                s.Ux[own]*s.Ux[own]
              + s.Uy[own]*s.Uy[own]
              + s.Uz[own]*s.Uz[own];
            const double neighbourVelocitySquared =
                s.Ux[nei]*s.Ux[nei]
              + s.Uy[nei]*s.Uy[nei]
              + s.Uz[nei]*s.Uz[nei];
            const double ownerThermalEnthalpy = s.gasCp*s.Tgas[own];
            const double neighbourThermalEnthalpy = s.gasCp*s.Tgas[nei];
            const double ownerKineticEnergy = 0.5*ownerVelocitySquared;
            const double neighbourKineticEnergy =
                0.5*neighbourVelocitySquared;
            const ugkpinterpolation::Vector3
                ownerThermalEnthalpyGradient
            {
                s.gasCp*s.gradTX[own],
                s.gasCp*s.gradTY[own],
                s.gasCp*s.gradTZ[own]
            };
            const ugkpinterpolation::Vector3
                neighbourThermalEnthalpyGradient
            {
                s.gasCp*s.gradTX[nei],
                s.gasCp*s.gradTY[nei],
                s.gasCp*s.gradTZ[nei]
            };
            const ugkpinterpolation::Vector3 ownerKineticEnergyGradient
            {
                s.Ux[own]*s.gradUxX[own]
              + s.Uy[own]*s.gradUyX[own]
              + s.Uz[own]*s.gradUzX[own],
                s.Ux[own]*s.gradUxY[own]
              + s.Uy[own]*s.gradUyY[own]
              + s.Uz[own]*s.gradUzY[own],
                s.Ux[own]*s.gradUxZ[own]
              + s.Uy[own]*s.gradUyZ[own]
              + s.Uz[own]*s.gradUzZ[own]
            };
            const ugkpinterpolation::Vector3
                neighbourKineticEnergyGradient
            {
                s.Ux[nei]*s.gradUxX[nei]
              + s.Uy[nei]*s.gradUyX[nei]
              + s.Uz[nei]*s.gradUzX[nei],
                s.Ux[nei]*s.gradUxY[nei]
              + s.Uy[nei]*s.gradUyY[nei]
              + s.Uz[nei]*s.gradUzY[nei],
                s.Ux[nei]*s.gradUxZ[nei]
              + s.Uy[nei]*s.gradUyZ[nei]
              + s.Uz[nei]*s.gradUzZ[nei]
            };
            const ugkpinterpolation::Vector3 centreToCentre
            {
                mappedNeiCx - s.Cx[own],
                mappedNeiCy - s.Cy[own],
                mappedNeiCz - s.Cz[own]
            };
            const double ownerWeight =
                clampRange(s.faceWeight[f], 0.0, 1.0);
            const double faceThermalEnthalpy =
                ugkpinterpolation::limitedLinearFaceValue
                (
                    ownerThermalEnthalpy,
                    neighbourThermalEnthalpy,
                    ownerThermalEnthalpyGradient,
                    neighbourThermalEnthalpyGradient,
                    centreToCentre,
                    ownerWeight,
                    massFlux,
                    1.0
                );
            const double faceKineticEnergy =
                ugkpinterpolation::limitedLinearFaceValue
                (
                    ownerKineticEnergy,
                    neighbourKineticEnergy,
                    ownerKineticEnergyGradient,
                    neighbourKineticEnergyGradient,
                    centreToCentre,
                    ownerWeight,
                    massFlux,
                    1.0
                );
            const bool ownerIsUpwind = massFlux >= 0.0;
            const double upwindSpecificEnergy =
                (ownerIsUpwind
                  ? ownerThermalEnthalpy
                  : neighbourThermalEnthalpy)
              + (ownerIsUpwind
                  ? ownerKineticEnergy
                  : neighbourKineticEnergy);
            const double limitedSpecificEnergy =
                faceThermalEnthalpy + faceKineticEnergy;
            energyFlux =
                ugkpinterpolation::limitedLinearRiemannEnergyFlux
                (
                    energyFlux,
                    massFlux,
                    upwindSpecificEnergy,
                    limitedSpecificEnergy
                );
        }
    }

                                                                       
    if (boundaryKind != 1)
    {
                                                                
                                                                          
                                                                     
                                                    
        double gradUxX = s.gradUxX[own];
        double gradUxY = s.gradUxY[own];
        double gradUxZ = s.gradUxZ[own];
        double gradUyX = s.gradUyX[own];
        double gradUyY = s.gradUyY[own];
        double gradUyZ = s.gradUyZ[own];
        double gradUzX = s.gradUzX[own];
        double gradUzY = s.gradUzY[own];
        double gradUzZ = s.gradUzZ[own];
        double gradTX = s.gradTX[own];
        double gradTY = s.gradTY[own];
        double gradTZ = s.gradTZ[own];
        const ugkptransport::Vector3 unitNormal{nx, ny, nz};
        ugkptransport::Vector3 compactSnGradU
        {
            gradUxX*nx + gradUxY*ny + gradUxZ*nz,
            gradUyX*nx + gradUyY*ny + gradUyZ*nz,
            gradUzX*nx + gradUzY*ny + gradUzZ*nz
        };
        double normalTemperatureGradient =
            gradTX*nx + gradTY*ny + gradTZ*nz;

        if (nei >= 0)
        {
            const double ownerWeight =
                clampRange(s.faceWeight[f], 0.0, 1.0);
            const ugkptransport::SnGradGeometry snGradGeometry =
                ugkptransport::makeInternalSnGradGeometry
                (
                    ugkptransport::Vector3
                    {
                        s.Cx[own], s.Cy[own], s.Cz[own]
                    },
                    ugkptransport::Vector3
                    {
                        mappedNeiCx, mappedNeiCy, mappedNeiCz
                    },
                    unitNormal
                );
            const ugkptransport::Vector3 gradUx =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUxX[nei],
                        s.gradUxY[nei],
                        s.gradUxZ[nei]
                    },
                    ownerWeight
                );
            const ugkptransport::Vector3 gradUy =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUyX[nei],
                        s.gradUyY[nei],
                        s.gradUyZ[nei]
                    },
                    ownerWeight
                );
            const ugkptransport::Vector3 gradUz =
                ugkptransport::linearInterpolate
                (
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUzX[nei],
                        s.gradUzY[nei],
                        s.gradUzZ[nei]
                    },
                    ownerWeight
                );
            compactSnGradU.x =
                ugkptransport::correctedSnGrad
                (
                    s.Ux[own],
                    s.Ux[nei],
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUxX[nei],
                        s.gradUxY[nei],
                        s.gradUxZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            compactSnGradU.y =
                ugkptransport::correctedSnGrad
                (
                    s.Uy[own],
                    s.Uy[nei],
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUyX[nei],
                        s.gradUyY[nei],
                        s.gradUyZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            compactSnGradU.z =
                ugkptransport::correctedSnGrad
                (
                    s.Uz[own],
                    s.Uz[nei],
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradUzX[nei],
                        s.gradUzY[nei],
                        s.gradUzZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            normalTemperatureGradient =
                ugkptransport::correctedSnGrad
                (
                    s.Tgas[own],
                    s.Tgas[nei],
                    ugkptransport::Vector3
                    {
                        gradTX, gradTY, gradTZ
                    },
                    ugkptransport::Vector3
                    {
                        s.gradTX[nei],
                        s.gradTY[nei],
                        s.gradTZ[nei]
                    },
                    ownerWeight,
                    snGradGeometry
                );
            gradUxX = gradUx.x;
            gradUxY = gradUx.y;
            gradUxZ = gradUx.z;
            gradUyX = gradUy.x;
            gradUyY = gradUy.y;
            gradUyZ = gradUy.z;
            gradUzX = gradUz.x;
            gradUzY = gradUz.y;
            gradUzZ = gradUz.z;
        }
        else
        {
            const bool velocityFixed = boundaryKind == 2
              || useRiemannBoundaryVelocity(s, f, left);
            const double boundaryUx = boundaryKind == 2
              ? right.ux : s.riemannBoundaryUx[f];
            const double boundaryUy = boundaryKind == 2
              ? right.uy : s.riemannBoundaryUy[f];
            const double boundaryUz = boundaryKind == 2
              ? right.uz : s.riemannBoundaryUz[f];
            const double currentNormalGradUx =
                gradUxX*nx + gradUxY*ny + gradUxZ*nz;
            const double currentNormalGradUy =
                gradUyX*nx + gradUyY*ny + gradUyZ*nz;
            const double currentNormalGradUz =
                gradUzX*nx + gradUzY*ny + gradUzZ*nz;
            const double targetNormalGradUx = velocityFixed
              ? (boundaryUx - s.Ux[own])*s.deltaCoeffs[f] : 0.0;
            const double targetNormalGradUy = velocityFixed
              ? (boundaryUy - s.Uy[own])*s.deltaCoeffs[f] : 0.0;
            const double targetNormalGradUz = velocityFixed
              ? (boundaryUz - s.Uz[own])*s.deltaCoeffs[f] : 0.0;
            (void)currentNormalGradUx;
            (void)currentNormalGradUy;
            (void)currentNormalGradUz;
            compactSnGradU =
                ugkptransport::Vector3
                {
                    targetNormalGradUx,
                    targetNormalGradUy,
                    targetNormalGradUz
                };
            const double targetNormalGradT =
                s.riemannBoundaryTFix[f] != 0
              ? (s.riemannBoundaryT[f] - s.Tgas[own])
               *s.deltaCoeffs[f]
              : 0.0;
            normalTemperatureGradient = targetNormalGradT;
        }

        const double rhoFace =
            0.5*(left.rho + right.rho);
        double muTurbulent = 0.0;
        double kTurbulent = 0.0;
        double directWallHeatFlux = 0.0;
        int directWallHeatFluxActive = 0;
        if constexpr (IncludeTurbulence)
        {
            gasFaceSubgridTransportProperties
            (
                s,
                f,
                own,
                nei,
                boundaryKind,
                rhoFace,
                muTurbulent,
                kTurbulent,
                directWallHeatFlux,
                directWallHeatFluxActive
            );
        }
        const double muEffective = s.gasMu + muTurbulent;
        const double kEffective =
            molecularGasConductivity(s) + kTurbulent;
        if (muEffective > 0.0 || kEffective > 0.0)
        {
            const ugkptransport::Vector3 traction =
                ugkptransport::openFoamNewtonianTraction
                (
                    muEffective,
                    unitNormal,
                    compactSnGradU,
                    ugkptransport::Vector3
                    {
                        gradUxX, gradUxY, gradUxZ
                    },
                    ugkptransport::Vector3
                    {
                        gradUyX, gradUyY, gradUyZ
                    },
                    ugkptransport::Vector3
                    {
                        gradUzX, gradUzY, gradUzZ
                    }
                );

            double faceUx = 0.5*(left.ux + right.ux);
            double faceUy = 0.5*(left.uy + right.uy);
            double faceUz = 0.5*(left.uz + right.uz);
            if (nei >= 0)
            {
                const double ownerWeight =
                    clampRange(s.faceWeight[f], 0.0, 1.0);
                faceUx =
                    ownerWeight*left.ux + (1.0 - ownerWeight)*right.ux;
                faceUy =
                    ownerWeight*left.uy + (1.0 - ownerWeight)*right.uy;
                faceUz =
                    ownerWeight*left.uz + (1.0 - ownerWeight)*right.uz;
            }
            if
            (
                boundaryKind == 2
             || (nei < 0 && useRiemannBoundaryVelocity(s, f, left))
            )
            {
                faceUx = right.ux;
                faceUy = right.uy;
                faceUz = right.uz;
            }
            momentumFluxX -= traction.x;
            momentumFluxY -= traction.y;
            momentumFluxZ -= traction.z;
            energyFlux -=
                traction.x*faceUx
              + traction.y*faceUy
              + traction.z*faceUz;
            if (directWallHeatFluxActive != 0)
            {
                energyFlux += directWallHeatFlux;
            }
            else
            {
                energyFlux -= kEffective*normalTemperatureGradient;
            }
        }
    }

    massFluxArea = massFlux*area;
    momFluxXArea = momentumFluxX*area;
    momFluxYArea = momentumFluxY*area;
    momFluxZArea = momentumFluxZ*area;
    energyFluxArea = energyFlux*area;
    return true;
}

template<bool IncludeTurbulence>
__global__ void computeGasInternalFaceFluxKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    s.gasPhiRho[f] = 0.0;
    s.gasPhiRhoUx[f] = 0.0;
    s.gasPhiRhoUy[f] = 0.0;
    s.gasPhiRhoUz[f] = 0.0;
    s.gasPhiRhoE[f] = 0.0;

    (void)dt;
    computeRiemannGasFaceFluxDevice<IncludeTurbulence>
    (
        s,
        f,
        s.gasPhiRho[f],
        s.gasPhiRhoUx[f],
        s.gasPhiRhoUy[f],
        s.gasPhiRhoUz[f],
        s.gasPhiRhoE[f]
    );
}

__global__ void enforcePeriodicGasFluxAntisymmetryKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (!isPeriodicFace(s, f))
    {
        return;
    }
    const int pair = s.facePeriodicPair[f];
    if (f > pair)
    {
        return;
    }

    const double rho = 0.5*(s.gasPhiRho[f] - s.gasPhiRho[pair]);
    const double rhoUx = 0.5*(s.gasPhiRhoUx[f] - s.gasPhiRhoUx[pair]);
    const double rhoUy = 0.5*(s.gasPhiRhoUy[f] - s.gasPhiRhoUy[pair]);
    const double rhoUz = 0.5*(s.gasPhiRhoUz[f] - s.gasPhiRhoUz[pair]);
    const double rhoE = 0.5*(s.gasPhiRhoE[f] - s.gasPhiRhoE[pair]);
    s.gasPhiRho[f] = rho;
    s.gasPhiRhoUx[f] = rhoUx;
    s.gasPhiRhoUy[f] = rhoUy;
    s.gasPhiRhoUz[f] = rhoUz;
    s.gasPhiRhoE[f] = rhoE;
    s.gasPhiRho[pair] = -rho;
    s.gasPhiRhoUx[pair] = -rhoUx;
    s.gasPhiRhoUy[pair] = -rhoUy;
    s.gasPhiRhoUz[pair] = -rhoUz;
    s.gasPhiRhoE[pair] = -rhoE;
}


__global__ void computeGasFluxPositivityScaleKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double outgoingMassFlux = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const double phi = finiteOr(s.gasPhiRho[f], 0.0);
        if (s.faceOwner[f] == c && phi > 0.0)
        {
            outgoingMassFlux += phi;
        }
        else if (s.faceNeighbour[f] == c && phi < 0.0)
        {
            outgoingMassFlux -= phi;
        }
    }

    double scale = 1.0;
    if (outgoingMassFlux > 0.0 && dt > 0.0)
    {
        const double availableMass =
            clampMin(s.rho[c] - s.rhoMin, 0.0)*s.V[c];
        const double requestedOutflowMass = dt*outgoingMassFlux;
        if (requestedOutflowMass > availableMass)
        {
            scale = clampRange
            (
                0.999*availableMass
               /(requestedOutflowMass + 1.0e-300),
                0.0,
                1.0
            );
        }
    }
    s.gasFluxPositivityScale[c] = scale;
}

__global__ void applyGasFluxPositivityScaleKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const double phi = finiteOr(s.gasPhiRho[f], 0.0);
    double scale = 1.0;
    if (phi > 0.0)
    {
        const int own = s.faceOwner[f];
        if (own >= 0 && own < s.nCells)
        {
            scale = s.gasFluxPositivityScale[own];
        }
    }
    else if (phi < 0.0 && coupledFaceNeighbour(s, f) >= 0)
    {
        const int nei = s.faceNeighbour[f];
        if (nei >= 0 && nei < s.nCells)
        {
            scale = s.gasFluxPositivityScale[nei];
        }
    }

    scale = clampRange(finiteOr(scale, 0.0), 0.0, 1.0);
    s.gasPhiRho[f] *= scale;
    s.gasPhiRhoUx[f] *= scale;
    s.gasPhiRhoUy[f] *= scale;
    s.gasPhiRhoUz[f] *= scale;
    s.gasPhiRhoE[f] *= scale;
}

__global__ void computeSstFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces || s.sstConfigured == 0)
    {
        return;
    }
    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        s.sstPhiRhoK[f] = 0.0;
        s.sstPhiRhoOmega[f] = 0.0;
        return;
    }
    const int nei = coupledFaceNeighbour(s, f);
    const int boundaryKind =
        nei >= 0 ? 0 : s.riemannBoundaryKind[f];
    double sstMappedNeiCx = 0.0;
    double sstMappedNeiCy = 0.0;
    double sstMappedNeiCz = 0.0;
    if (nei >= 0)
    {
        periodicMappedCellCentre
        (
            s, f, nei,
            sstMappedNeiCx, sstMappedNeiCy, sstMappedNeiCz
        );
    }
    if (boundaryKind == 1 || boundaryKind == 3 || boundaryKind == 4)
    {
        s.sstPhiRhoK[f] = 0.0;
        s.sstPhiRhoOmega[f] = 0.0;
        return;
    }

    const double massFlux = s.gasPhiRho[f];
    const double ownerWeight = clampRange(s.faceWeight[f], 0.0, 1.0);
    const double kExterior = nei >= 0
      ? s.k[nei] : sstBoundaryValue(s, f, own, false);
    const double omegaExterior = nei >= 0
      ? s.omega[nei] : sstBoundaryValue(s, f, own, true);
    const double kUpwind = massFlux >= 0.0 ? s.k[own] : kExterior;
    const double omegaUpwind =
        massFlux >= 0.0 ? s.omega[own] : omegaExterior;

    const double rhoFace = nei >= 0
      ? ownerWeight*s.rho[own] + (1.0 - ownerWeight)*s.rho[nei]
      : s.rho[own];
    const double f1Face = nei >= 0
      ? ownerWeight*s.sstF1[own] + (1.0 - ownerWeight)*s.sstF1[nei]
      : s.sstF1[own];
    const double nuFace = s.gasMu/clampMin(rhoFace, s.rhoMin);
    double nutFace = nei >= 0
      ? ownerWeight*s.nut[own] + (1.0 - ownerWeight)*s.nut[nei]
      : s.nut[own];
    if (boundaryKind == 2)
    {
        nutFace = 0.0;
        if (s.sstWallTreatment == 1)
        {
            const double wallUx = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
            const double wallUy = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
            const double wallUz = s.riemannBoundaryUFix[f] != 0
              ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
            const double dux = s.Ux[own] - wallUx;
            const double duy = s.Uy[own] - wallUy;
            const double duz = s.Uz[own] - wallUz;
            nutFace = ugkpwall::spaldingWallState
            (
                sqrt(dux*dux + duy*duy + duz*duz),
                s.sstWallDistance[own],
                nuFace,
                s.sstWallKappa,
                s.sstWallE
            ).nut;
        }
    }
    const double dk =
        nuFace + ugkwp::sstAlphaK(f1Face, s.sstCoefficients)*nutFace;
    const double domega =
        nuFace
      + ugkwp::sstAlphaOmega(f1Face, s.sstCoefficients)*nutFace;

    double snGradK = 0.0;
    double snGradOmega = 0.0;
    if (nei >= 0)
    {
        const double area = clampMin(s.magSf[f], OfSmall);
        const ugkptransport::SnGradGeometry geometry =
            ugkptransport::makeInternalSnGradGeometry
            (
                ugkptransport::Vector3{s.Cx[own], s.Cy[own], s.Cz[own]},
                ugkptransport::Vector3
                {
                    sstMappedNeiCx, sstMappedNeiCy, sstMappedNeiCz
                },
                ugkptransport::Vector3
                {
                    s.Sfx[f]/area,
                    s.Sfy[f]/area,
                    s.Sfz[f]/area
                }
            );
        snGradK = ugkptransport::correctedSnGrad
        (
            s.k[own],
            s.k[nei],
            ugkptransport::Vector3
            {
                s.gradKX[own], s.gradKY[own], s.gradKZ[own]
            },
            ugkptransport::Vector3
            {
                s.gradKX[nei], s.gradKY[nei], s.gradKZ[nei]
            },
            ownerWeight,
            geometry
        );
        snGradOmega = ugkptransport::correctedSnGrad
        (
            s.omega[own],
            s.omega[nei],
            ugkptransport::Vector3
            {
                s.gradOmegaX[own], s.gradOmegaY[own], s.gradOmegaZ[own]
            },
            ugkptransport::Vector3
            {
                s.gradOmegaX[nei], s.gradOmegaY[nei], s.gradOmegaZ[nei]
            },
            ownerWeight,
            geometry
        );
    }
    else
    {
        const int kMode = s.sstBoundaryKMode[f];
        const int omegaMode = s.sstBoundaryOmegaMode[f];
        const bool kFixed =
            (boundaryKind == 2 && s.sstWallTreatment == 0)
          || kMode == 1
          || (kMode == 2 && massFlux < 0.0);
        const bool omegaFixed = boundaryKind == 2 || omegaMode == 1
          || (omegaMode == 2 && massFlux < 0.0);
        if (kFixed)
        {
            snGradK = s.deltaCoeffs[f]*(kExterior - s.k[own]);
        }
        if (omegaFixed)
        {
            snGradOmega =
                s.deltaCoeffs[f]*(omegaExterior - s.omega[own]);
        }
    }

    const double area = s.magSf[f];
    s.sstPhiRhoK[f] =
        massFlux*kUpwind - rhoFace*dk*snGradK*area;
    s.sstPhiRhoOmega[f] =
        massFlux*omegaUpwind - rhoFace*domega*snGradOmega*area;
}

__global__ void enforcePeriodicSstFluxAntisymmetryKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (!isPeriodicFace(s, f))
    {
        return;
    }
    const int pair = s.facePeriodicPair[f];
    if (f > pair)
    {
        return;
    }
    const double rhoK = 0.5*(s.sstPhiRhoK[f] - s.sstPhiRhoK[pair]);
    const double rhoOmega =
        0.5*(s.sstPhiRhoOmega[f] - s.sstPhiRhoOmega[pair]);
    s.sstPhiRhoK[f] = rhoK;
    s.sstPhiRhoOmega[f] = rhoOmega;
    s.sstPhiRhoK[pair] = -rhoK;
    s.sstPhiRhoOmega[pair] = -rhoOmega;
}

__device__ double sstKProductionForCell
(
    const DeviceState& s,
    const int c,
    const double gByNu
)
{
    double production = ugkwp::sstKProduction
    (
        s.k[c],
        s.omega[c],
        s.nut[c],
        gByNu,
        s.sstCoefficients
    );
    if (s.sstWallTreatment != 1)
    {
        return production;
    }

    double wallProductionSum = 0.0;
    int wallCount = 0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < s.nInternalFaces
         || f >= s.nFaces
         || s.riemannBoundaryKind[f] != 2
        )
        {
            continue;
        }
        const double wallUx = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
        const double wallUy = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
        const double wallUz = s.riemannBoundaryUFix[f] != 0
          ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
        const double dux = s.Ux[c] - wallUx;
        const double duy = s.Uy[c] - wallUy;
        const double duz = s.Uz[c] - wallUz;
        const double y = clampMin(s.sstWallDistance[c], OfVSmall);
        const double magGradU = sqrt(dux*dux + duy*duy + duz*duz)/y;
        wallProductionSum += ugkpwall::omegaWallFunctionState
        (
            s.k[c],
            magGradU,
            y,
            s.gasMu/clampMin(s.rho[c], s.rhoMin),
            s.sstCoefficients.beta1,
            s.sstWallCmu,
            s.sstWallKappa,
            s.sstWallE,
            production
        ).production;
        ++wallCount;
    }
    return wallCount > 0
      ? wallProductionSum/double(wallCount)
      : production;
}

__global__ void applySstFluxAndSourceKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    double fluxK = 0.0;
    double fluxOmega = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double sign = s.faceOwner[f] == c ? -1.0 : 1.0;
        fluxK += sign*s.sstPhiRhoK[f];
        fluxOmega += sign*s.sstPhiRhoOmega[f];
    }

    double divU = 0.0;
    double s2 = 0.0;
    double gByNu = 0.0;
    sstVelocityInvariants(s, c, divU, s2, gByNu);
    const double gradDot =
        s.gradKX[c]*s.gradOmegaX[c]
      + s.gradKY[c]*s.gradOmegaY[c]
      + s.gradKZ[c]*s.gradOmegaZ[c];
    const double cd = ugkwp::sstCrossDiffusion
    (
        s.omega[c],
        gradDot,
        s.sstCoefficients
    );
    const double kProduction = sstKProductionForCell(s, c, gByNu);
    const double sourceK = s.rho[c]*
    (
        kProduction
      - (2.0/3.0)*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    const double sourceOmega = ugkwp::sstOmegaSource
    (
        s.rho[c],
        s.k[c],
        s.omega[c],
        divU,
        gByNu,
        s2,
        s.sstF1[c],
        s.sstF2[c],
        cd,
        s.sstCoefficients
    );
    const double invV = 1.0/clampMin(s.V[c], OfSmall);
    const double deltaRhoK = dt*(fluxK*invV + sourceK);
    const double deltaRhoOmega = dt*(fluxOmega*invV + sourceOmega);
    const double rhoSafe = clampMin(s.rho[c], s.rhoMin);
    const double rhoKFloor = rhoSafe*s.sstKMin;
    const double rhoOmegaFloor = rhoSafe*s.sstOmegaMin;
    s.sstSourceNumber[c] = fmax
    (
        fabs(dt*sourceK)/clampMin(s.rhoK[c], rhoKFloor),
        fabs(dt*sourceOmega)/clampMin(s.rhoOmega[c], rhoOmegaFloor)
    );
    s.rhoK[c] =
        clampMin(finiteOr(s.rhoK[c] + deltaRhoK, rhoKFloor), rhoKFloor);
    s.rhoOmega[c] = clampMin
    (
        finiteOr(s.rhoOmega[c] + deltaRhoOmega, rhoOmegaFloor),
        rhoOmegaFloor
    );
}
__global__ void computeGasCourantFieldKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    (void)dt;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const int own = s.faceOwner[f];
    if (own < 0 || own >= s.nCells)
    {
        s.gasPhiRho[f] = OfGreat;
        return;
    }
    if
    (
        f >= s.nInternalFaces
     &&
        (
            s.riemannBoundaryKind[f] == 1
         || s.riemannBoundaryKind[f] == 3
         || s.riemannBoundaryKind[f] == 4
        )
    )
    {
        s.gasPhiRho[f] = 0.0;
        return;
    }

    const double area = clampMin(s.magSf[f], OfSmall);
    const double nx = s.Sfx[f]/area;
    const double ny = s.Sfy[f]/area;
    const double nz = s.Sfz[f]/area;
    const GasPrimDevice left = makeGasPrimDevice
    (
        s.rho[own],
        s.Ux[own],
        s.Uy[own],
        s.Uz[own],
        s.p[own],
        s.Rgas,
        s.rhoMin,
        s.TgasMin
    );
    GasPrimDevice right = left;
    if (f < s.nInternalFaces || isPeriodicFace(s, f))
    {
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            s.gasPhiRho[f] = OfGreat;
            return;
        }
        right = makeGasPrimDevice
        (
            s.rho[nei],
            s.Ux[nei],
            s.Uy[nei],
            s.Uz[nei],
            s.p[nei],
            s.Rgas,
            s.rhoMin,
            s.TgasMin
        );
    }
    else if
    (
        s.riemannBoundaryKind[f] != 1
     && s.riemannBoundaryKind[f] != 2
    )
    {
        right = riemannBoundaryState(s, f, left);
    }
    const double unLeft =
        left.ux*nx + left.uy*ny + left.uz*nz;
    const double unRight =
        right.ux*nx + right.uy*ny + right.uz*nz;
    const double aLeft =
        sqrt(clampMin(s.gammaGas*left.p/left.rho, OfSmall));
    const double aRight =
        sqrt(clampMin(s.gammaGas*right.p/right.rho, OfSmall));
    const double spectralRadius = fmax
    (
        fabs(unLeft) + aLeft,
        fabs(unRight) + aRight
    );
    const double amaxSf = spectralRadius*area;

    s.gasPhiRho[f] = finiteDevice(amaxSf) ? amaxSf : OfGreat;
}

__global__ void computeGasConvectiveCourantByCellKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double sumAmaxSf = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        sumAmaxSf += finiteOr(s.gasPhiRho[f], OfGreat);
    }

                                                     
                                              
    const double co =
        0.5*dt*sumAmaxSf/clampMin(s.V[c], OfSmall);
    s.gasFluxPositivityScale[c] = finiteDevice(co) ? co : OfGreat;
}

__global__ void computeGasDiffusionNumberKernel
(
    DeviceState* sp,
    const double dt,
    const double targetMaxCo
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    double sum = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < 0
         || f >= s.nFaces
         || (f >= s.nInternalFaces && s.riemannBoundaryKind[f] == 4)
        )
        {
            continue;
        }
        const int other = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? (s.faceOwner[f] == c ? s.faceNeighbour[f] : s.faceOwner[f])
          : -1;
        const double rhoFace = other >= 0
          ? 0.5*(s.rho[c] + s.rho[other]) : s.rho[c];
        double muTurbulent = 0.0;
        double kTurbulent = 0.0;
        double directWallHeatFlux = 0.0;
        int directWallHeatFluxActive = 0;
        const int boundaryKind =
            (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? 0 : s.riemannBoundaryKind[f];
        gasFaceSubgridTransportProperties
        (
            s, f, c, other, boundaryKind, rhoFace,
            muTurbulent, kTurbulent,
            directWallHeatFlux, directWallHeatFluxActive
        );
        (void)directWallHeatFlux;
        (void)directWallHeatFluxActive;
                                                                         
        const double muEffective = s.gasMu + muTurbulent;
        const double kEffective = molecularGasConductivity(s) + kTurbulent;
        const double rhoSafe = clampMin(rhoFace, s.rhoMin);
        const double nu = muEffective/rhoSafe;
        const double thermalAlpha = kEffective/(rhoSafe*s.gasCp + OfSmall);
        sum += fmax(nu, thermalAlpha)*s.magSf[f]*s.deltaCoeffs[f];
    }
    const double d = dt*sum/clampMin(s.V[c], OfSmall);
    const double equivalentCo =
        targetMaxCo*d/clampMin(s.maxDiffusionNumber, OfSmall);
    s.gasDiffusionNumber[c] = finiteDevice(equivalentCo)
      ? equivalentCo : OfGreat;
}

__global__ void computeSstStabilityNumberKernel
(
    DeviceState* sp,
    const double dt,
    const double targetMaxCo
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells || s.sstConfigured == 0)
    {
        return;
    }

    double diffusionRate = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int f = s.cellFaceId[start + i];
        if
        (
            f < 0
         || f >= s.nFaces
         || (f >= s.nInternalFaces
          && (s.riemannBoundaryKind[f] == 3
           || s.riemannBoundaryKind[f] == 4))
        )
        {
            continue;
        }
        const int other = (f < s.nInternalFaces || isPeriodicFace(s, f))
          ? (s.faceOwner[f] == c ? s.faceNeighbour[f] : s.faceOwner[f])
          : -1;
        const double rhoFace = other >= 0
          ? 0.5*(s.rho[c] + s.rho[other]) : s.rho[c];
        const double f1Face = other >= 0
          ? 0.5*(s.sstF1[c] + s.sstF1[other]) : s.sstF1[c];
        const double nu = s.gasMu/clampMin(rhoFace, s.rhoMin);
        const bool physicalWall =
            f >= s.nInternalFaces
         && !isPeriodicFace(s, f)
         && s.riemannBoundaryKind[f] == 2;
        double nutFace = other >= 0
          ? 0.5*(s.nut[c] + s.nut[other]) : s.nut[c];
        if (physicalWall)
        {
            nutFace = 0.0;
            if (s.sstWallTreatment == 1)
            {
                const double wallUx = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUx[f], 0.0) : 0.0;
                const double wallUy = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUy[f], 0.0) : 0.0;
                const double wallUz = s.riemannBoundaryUFix[f] != 0
                  ? finiteOr(s.riemannBoundaryUz[f], 0.0) : 0.0;
                const double dux = s.Ux[c] - wallUx;
                const double duy = s.Uy[c] - wallUy;
                const double duz = s.Uz[c] - wallUz;
                nutFace = ugkpwall::spaldingWallState
                (
                    sqrt(dux*dux + duy*duy + duz*duz),
                    s.sstWallDistance[c],
                    nu,
                    s.sstWallKappa,
                    s.sstWallE
                ).nut;
            }
        }
        const double maximumDiffusivity = fmax
        (
            nu + ugkwp::sstAlphaK(f1Face, s.sstCoefficients)*nutFace,
            nu + ugkwp::sstAlphaOmega(f1Face, s.sstCoefficients)*nutFace
        );
        diffusionRate +=
            maximumDiffusivity*s.magSf[f]*s.deltaCoeffs[f];
    }
    const double diffusionNumber =
        dt*diffusionRate/clampMin(s.V[c], OfSmall);

    double divU = 0.0;
    double s2 = 0.0;
    double gByNu = 0.0;
    sstVelocityInvariants(s, c, divU, s2, gByNu);
    const double gradDot =
        s.gradKX[c]*s.gradOmegaX[c]
      + s.gradKY[c]*s.gradOmegaY[c]
      + s.gradKZ[c]*s.gradOmegaZ[c];
    const double cd = ugkwp::sstCrossDiffusion
    (
        s.omega[c],
        gradDot,
        s.sstCoefficients
    );
    const double sourceK = s.rho[c]*
    (
        sstKProductionForCell(s, c, gByNu)
      - (2.0/3.0)*divU*s.k[c]
      - s.sstCoefficients.betaStar*s.k[c]*s.omega[c]
    );
    const double sourceOmega = ugkwp::sstOmegaSource
    (
        s.rho[c], s.k[c], s.omega[c], divU, gByNu, s2,
        s.sstF1[c], s.sstF2[c], cd, s.sstCoefficients
    );
    const double sourceNumber = fmax
    (
        fabs(dt*sourceK)/clampMin(s.rhoK[c], s.rho[c]*s.sstKMin),
        fabs(dt*sourceOmega)
       /clampMin(s.rhoOmega[c], s.rho[c]*s.sstOmegaMin)
    );
    const double equivalentCo = targetMaxCo*fmax
    (
        diffusionNumber/clampMin(s.maxDiffusionNumber, OfSmall),
        sourceNumber/clampMin(s.sstMaxSourceNumber, OfSmall)
    );
    s.sstSourceNumber[c] = finiteDevice(equivalentCo)
      ? equivalentCo : OfGreat;
}

__global__ void applyGasFluxDivergenceByCellKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double dRho = 0.0;
    double dRhoUx = 0.0;
    double dRhoUy = 0.0;
    double dRhoUz = 0.0;
    double dRhoE = 0.0;

    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int faceI = s.cellFaceId[start + i];
        if (faceI < 0 || faceI >= s.nFaces)
        {
            continue;
        }

        double sign = 0.0;
        if (s.faceOwner[faceI] == c)
        {
            sign = -1.0;
        }
        else if (s.faceNeighbour[faceI] == c)
        {
            sign = 1.0;
        }
        else
        {
            continue;
        }

        dRho += sign*s.gasPhiRho[faceI];
        dRhoUx += sign*s.gasPhiRhoUx[faceI];
        dRhoUy += sign*s.gasPhiRhoUy[faceI];
        dRhoUz += sign*s.gasPhiRhoUz[faceI];
        dRhoE += sign*s.gasPhiRhoE[faceI];
    }

    const double scale = dt/clampMin(s.V[c], s.rhoMin);
    const double rhoBefore = s.rho[c];
    const double rhoAfter = rhoBefore + scale*dRho;
    if (!finiteDevice(rhoAfter) || rhoAfter <= 0.0)
    {
        asm("trap;");
    }
    s.rho[c] += scale*dRho;
    s.rhoUx[c] += scale*dRhoUx;
    s.rhoUy[c] += scale*dRhoUy;
    s.rhoUz[c] += scale*dRhoUz;
    s.rhoE[c] += scale*dRhoE;
}

__global__ void saveGasConservativeStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.rhoNext[c] = s.rho[c];
    s.rhoUxNext[c] = s.rhoUx[c];
    s.rhoUyNext[c] = s.rhoUy[c];
    s.rhoUzNext[c] = s.rhoUz[c];
    s.rhoENext[c] = s.rhoE[c];
    if (s.sstConfigured != 0)
    {
        s.rhoKInitial[c] = s.rhoK[c];
        s.rhoOmegaInitial[c] = s.rhoOmega[c];
    }
}

__global__ void blendGasConservativeStateKernel
(
    DeviceState* sp,
    const double initialWeight,
    const double stageWeight
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.rho[c] =
        initialWeight*s.rhoNext[c] + stageWeight*s.rho[c];
    s.rhoUx[c] =
        initialWeight*s.rhoUxNext[c] + stageWeight*s.rhoUx[c];
    s.rhoUy[c] =
        initialWeight*s.rhoUyNext[c] + stageWeight*s.rhoUy[c];
    s.rhoUz[c] =
        initialWeight*s.rhoUzNext[c] + stageWeight*s.rhoUz[c];
    s.rhoE[c] =
        initialWeight*s.rhoENext[c] + stageWeight*s.rhoE[c];
    if (s.sstConfigured != 0)
    {
        s.rhoK[c] =
            initialWeight*s.rhoKInitial[c] + stageWeight*s.rhoK[c];
        s.rhoOmega[c] =
            initialWeight*s.rhoOmegaInitial[c]
          + stageWeight*s.rhoOmega[c];
    }
}

__device__ void recoverGasPrimitiveCell(DeviceState& s, const int c)
{
    const double rhoSafe =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const double ux = finiteOr(s.rhoUx[c]/rhoSafe, 0.0);
    const double uy = finiteOr(s.rhoUy[c]/rhoSafe, 0.0);
    const double uz = finiteOr(s.rhoUz[c]/rhoSafe, 0.0);
    const double kinetic = 0.5*rhoSafe*(ux*ux + uy*uy + uz*uz);
    const double minimumInternalEnergy = fmax
    (
        s.rhoMin,
        rhoSafe*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - 1.0, OfSmall)
    );
    const double internalE = clampMin
    (
        finiteOr(s.rhoE[c] - kinetic, minimumInternalEnergy),
        minimumInternalEnergy
    );
    const double p = (s.gammaGas - 1.0)*internalE;
    const double T = p/(rhoSafe*s.Rgas);

    s.rho[c] = rhoSafe;
    s.rhoUx[c] = rhoSafe*ux;
    s.rhoUy[c] = rhoSafe*uy;
    s.rhoUz[c] = rhoSafe*uz;
    s.rhoE[c] = kinetic + internalE;
    s.Ux[c] = ux;
    s.Uy[c] = uy;
    s.Uz[c] = uz;
    s.p[c] = p;
    s.Tgas[c] = T;
}

__global__ void recoverGasPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    recoverGasPrimitiveCell(s, c);
}

__global__ void applyGasGravitySourceKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double rho = s.rho[c];
    const double momX = s.rhoUx[c];
    const double momY = s.rhoUy[c];
    const double momZ = s.rhoUz[c];
    const double gx = s.gravityX;
    const double gy = s.gravityY;
    const double gz = s.gravityZ;
    const double gravitySquared = gx*gx + gy*gy + gz*gz;

                                                                           
                                                                         
    s.rhoE[c] +=
        dt*(momX*gx + momY*gy + momZ*gz)
      + 0.5*rho*dt*dt*gravitySquared;
    s.rhoUx[c] = momX + rho*dt*gx;
    s.rhoUy[c] = momY + rho*dt*gy;
    s.rhoUz[c] = momZ + rho*dt*gz;
}

int applyGasGravitySource
(
    DeviceState* s,
    const int grid,
    const int block,
    const double dt
)
{
    if (!s->hostGravityActive)
    {
        return 0;
    }

    applyGasGravitySourceKernel<<<grid, block>>>(s->deviceState, dt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasGravitySourceKernel launch", err);
        return 1;
    }

                                                                             
                                                                       
    recoverGasPrimitivesKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel post-gravity launch", err);
        return 1;
    }
    return 0;
}

__global__ void recoverPrimitivesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    recoverGasPrimitiveCell(s, c);

    const double eps = clampMin(finiteOr(s.epsS[c], 0.0), 0.0);
    const double solidMass = eps*s.rhoSolid;
    if (eps <= s.epsSMin || solidMass <= s.epsSMin*s.rhoSolid)
    {
        clearSolidCell(s, c);
        return;
    }

    const double usx = finiteOr(s.rhoUsx[c]/solidMass, 0.0);
    const double usy = finiteOr(s.rhoUsy[c]/solidMass, 0.0);
    const double usz = finiteOr(s.rhoUsz[c]/solidMass, 0.0);
    const double solidKinetic = 0.5*sqr3(usx, usy, usz);
    double theta =
        (finiteOr(s.rhoEs[c], 0.0)/solidMass - solidKinetic)/1.5;
    if (!finiteDevice(theta) || theta < 0.0)
    {
        theta = 0.0;
        s.rhoEs[c] = solidMass*(solidKinetic + 1.5*theta);
    }

    double dMean = finiteOr(s.rhoDs[c]/solidMass, s.particleDiameterFallback);
    if (!finiteDevice(dMean) || dMean <= 0.0)
    {
        dMean = s.particleDiameterFallback;
        s.rhoDs[c] = solidMass*dMean;
    }
    dMean = clampMin(dMean, 1.0e-12);

    const double heatFactor = particleHeatFactorDevice(s);
    if (heatFactor > 0.0)
    {
        double Tp = finiteOr(s.rhoHp[c]/(solidMass*heatFactor), s.TpMin);
        Tp = clampRange(Tp, s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        s.rhoHp[c] = solidMass*heatFactor*Tp;
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = 0.0;
    }

    s.epsS[c] = eps;
    s.Usx[c] = usx;
    s.Usy[c] = usy;
    s.Usz[c] = usz;
    s.theta[c] = theta;
    s.dMeanCell[c] = dMean;
}

__global__ void initialiseParticleMaterialEnthalpyKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double eps = clampMin(s.epsS[c], 0.0);
    const double cap = eps*s.particleThermalRho*s.particleCp;
    if (eps <= s.epsSMin || cap <= s.rhoMin)
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = 0.0;
        return;
    }

    s.Tp[c] = clampRange(s.Tp[c], s.TpMin, s.TpMax);
    s.rhoHp[c] = cap*s.Tp[c];
}

__global__ void clearParticleMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    s.momRhoP[c] = 0.0;
    s.momRhoUPx[c] = 0.0;
    s.momRhoUPy[c] = 0.0;
    s.momRhoUPz[c] = 0.0;
    s.momRhoEP[c] = 0.0;
    s.momRhoPD[c] = 0.0;
    s.momRhoHpP[c] = 0.0;
}

__global__ void initialiseEpsGPrevKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    if (s.dragModel == 2)
    {
        s.epsGPrev[c] = 1.0;
    }
    else
    {
        const double eps = clampRange(finiteOr(s.epsS[c], 0.0), 0.0, 1.0);
        s.epsGPrev[c] = 1.0 - eps;
    }
}

__global__ void initialiseThetaDragAlphaKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    s.thetaDragAlpha[c] = 1.0;
}

__global__ void computePressureGradientKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const int f = s.cellFaceId[p];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const int kind = s.cellFaceKind[p];
        const int own = s.faceOwner[f];
        const int neiFace = s.faceNeighbour[f];
        const double lambda = finiteOr(s.faceWeight[f], 0.5);
        const double pf =
            (own >= 0 && own < s.nCells && neiFace >= 0 && neiFace < s.nCells)
          ? lambda*s.p[own] + (1.0 - lambda)*s.p[neiFace]
          : ((kind != 4 && f >= 0 && f < s.nFaces)
              ? s.gasBoundaryP[f]
              : s.p[c]);
        const double sign = (own == c) ? 1.0 : -1.0;
        gx += pf*sign*s.Sfx[f];
        gy += pf*sign*s.Sfy[f];
        gz += pf*sign*s.Sfz[f];
    }

    const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
    s.gradPx[c] = gx*invV;
    s.gradPy[c] = gy*invV;
    s.gradPz[c] = gz*invV;
}

__device__ double solidEpsFromMomentDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return 0.0;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    return clampRange
    (
        rhoP/clampMin(s.rhoSolid, 1.0e-300),
        0.0,
        1.0
    );
}

template<class DragModel>
__global__ void applyEulerianGasSolidCouplingKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

                                                                         
                                                                   
    s.couplingRhoOld[c] =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    s.couplingUxOld[c] = finiteOr(s.Ux[c], 0.0);
    s.couplingUyOld[c] = finiteOr(s.Uy[c], 0.0);
    s.couplingUzOld[c] = finiteOr(s.Uz[c], 0.0);
    s.couplingTgasOld[c] =
        clampMin(finiteOr(s.Tgas[c], s.TgasMin), s.TgasMin);

    s.thetaDragAlpha[c] = 1.0;
    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    const double eps = solidEpsFromMomentDevice(s, c);
    const double epsG = 1.0 - eps;
    const double epsGsafe = clampMin(epsG, OfSmall);
    const double epsGOld = finiteOr(s.epsGPrev[c], epsG);

    double gradEx = 0.0;
    double gradEy = 0.0;
    double gradEz = 0.0;

    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];

    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const int f = s.cellFaceId[p];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }

        const int own = s.faceOwner[f];
        const int neiFace = s.faceNeighbour[f];

        const double epsOwn =
            (own >= 0 && own < s.nCells)
          ? 1.0 - solidEpsFromMomentDevice(s, own)
          : epsG;

        const double epsNei =
            (neiFace >= 0 && neiFace < s.nCells)
          ? 1.0 - solidEpsFromMomentDevice(s, neiFace)
          : epsG;

        const double lambda = finiteOr(s.faceWeight[f], 0.5);
        const double epsFace = lambda*epsOwn + (1.0 - lambda)*epsNei;
        const double sign = (own == c) ? 1.0 : -1.0;

        gradEx += epsFace*sign*s.Sfx[f];
        gradEy += epsFace*sign*s.Sfy[f];
        gradEz += epsFace*sign*s.Sfz[f];
    }

    const double invV = 1.0/clampMin(s.V[c], OfSmall);
    gradEx *= invV;
    gradEy *= invV;
    gradEz *= invV;

    const double ugx0 = s.couplingUxOld[c];
    const double ugy0 = s.couplingUyOld[c];
    const double ugz0 = s.couplingUzOld[c];

    const double cepsG =
        -((epsG - epsGOld)/(dt + OfSmall)
          + ugx0*gradEx + ugy0*gradEy + ugz0*gradEz)/epsGsafe;

    const double mgOld = s.couplingRhoOld[c];
    const double massScale = 1.0 + dt*cepsG;
    const double mgCandidate = mgOld*massScale;
    const double momGXCandidate = s.rhoUx[c]*massScale;
    const double momGYCandidate = s.rhoUy[c]*massScale;
    const double momGZCandidate = s.rhoUz[c]*massScale;
    const double enerGCandidate = s.rhoE[c]*massScale;
    const double kineticCandidate =
        0.5
       *sqr3(momGXCandidate, momGYCandidate, momGZCandidate)
       /clampMin(mgCandidate, s.rhoMin);
    const double internalEnergyFloorCandidate =
        mgCandidate*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - 1.0, OfSmall);
    if
    (
        !finiteDevice(cepsG)
     || !finiteDevice(massScale)
     || !finiteDevice(mgCandidate)
     || !finiteDevice(momGXCandidate)
     || !finiteDevice(momGYCandidate)
     || !finiteDevice(momGZCandidate)
     || !finiteDevice(enerGCandidate)
     || !finiteDevice(kineticCandidate)
     || !finiteDevice(internalEnergyFloorCandidate)
     || mgCandidate < s.rhoMin
     || enerGCandidate < kineticCandidate + internalEnergyFloorCandidate
    )
    {
                                                                              
                                                                           
                                                                             
        asm("trap;");
        return;
    }

                                                                             
                                                                            
                                                                              
                                                                           
    s.rho[c] = mgCandidate;
    s.rhoUx[c] = momGXCandidate;
    s.rhoUy[c] = momGYCandidate;
    s.rhoUz[c] = momGZCandidate;
    s.rhoE[c] = enerGCandidate;
    s.epsGPrev[c] = epsG;

    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    double mg = mgCandidate;

    const double momSX = finiteOr(s.momRhoUPx[c], 0.0);
    const double momSY = finiteOr(s.momRhoUPy[c], 0.0);
    const double momSZ = finiteOr(s.momRhoUPz[c], 0.0);
    const double enerS0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);

    const double ms = rhoP;
    if (!finiteDevice(ms) || ms < s.rhoMin)
    {
        return;
    }

    const double usx0 = momSX/ms;
    const double usy0 = momSY/ms;
    const double usz0 = momSZ/ms;

    const double dLocal =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/ms, s.particleDiameterFallback),
            1.0e-12
        );

    const double urx = ugx0 - usx0;
    const double ury = ugy0 - usy0;
    const double urz = ugz0 - usz0;

    const double urMag = sqrt(sqr3(urx, ury, urz));
    const ugkwpGpuDrag::DragInput dragInput =
    {
        mg,
        s.gasMu,
        epsG,
        s.rhoSolid,
        dLocal,
        urMag,
        OfSmall,
        s.dragParameter0,
        s.dragParameter1,
        s.dragParameter2,
        s.dragParameter3
    };
    const double beta = DragModel::inverseResponseTime(dragInput);

    if (!finiteDevice(beta) || beta <= OfSmall)
    {
        return;
    }

    const double tauDragCell = clampMin(1.0/beta, OfSmall);

    const double momGX0 = momGXCandidate;
    const double momGY0 = momGYCandidate;
    const double momGZ0 = momGZCandidate;
    const double enerG0 = s.rhoE[c];

    double momGX = momGX0;
    double momGY = momGY0;
    double momGZ = momGZ0;

                                                                              
                                                                                
                                                                              
                                                                            
    double enerG = enerG0;
    double enerS = enerS0;

    if (!finiteDevice(mg) || mg < s.rhoMin)
    {
        return;
    }

    const double ugx = momGX/mg;
    const double ugy = momGY/mg;
    const double ugz = momGZ/mg;

    const double usx = momSX/ms;
    const double usy = momSY/ms;
    const double usz = momSZ/ms;

    const double gm1 = clampMin(s.gammaGas - 1.0, OfSmall);
    const double eMinGas = s.Rgas*s.TgasMin/gm1;

    const double kgOld = 0.5*mg*sqr3(ugx, ugy, ugz);
    const double ksOld = 0.5*ms*sqr3(usx, usy, usz);

    double igOld = enerG - kgOld;
    double isOld = enerS - ksOld;

    const double igMin = mg*eMinGas;
    const double isMin = 0.0;

    if (!finiteDevice(igOld) || igOld < igMin)
    {
        igOld = igMin;
    }

    if (!finiteDevice(isOld) || isOld < isMin)
    {
        isOld = isMin;
    }

    const double invTauDragCell = 1.0/clampMin(tauDragCell, OfSmall);

    const double alphaI =
        clampRange
        (
            finiteOr(exp(-2.0*dt*invTauDragCell), 1.0),
            0.0,
            1.0
        );

                                                                   
                                                                    
                                                      
    s.thetaDragAlpha[c] = alphaI;

    double isAfter = isOld*alphaI;
    if (isAfter < isMin)
    {
        isAfter = isMin;
    }

    const double dI = isOld - isAfter;
    const double igAfter = igOld + dI;

    const double wx = ugx - usx;
    const double wy = ugy - usy;
    const double wz = ugz - usz;

    const double kW = dt*invTauDragCell*(1.0 + ms/(mg + OfSmall));
    const double alphaW = exp(-kW);

    const double wxNew = wx*alphaW;
    const double wyNew = wy*alphaW;
    const double wzNew = wz*alphaW;

    const double pX = momGX + momSX;
    const double pY = momGY + momSY;
    const double pZ = momGZ + momSZ;

    const double mTot = mg + ms;

    const double usNewX = (pX - mg*wxNew)/mTot;
    const double usNewY = (pY - mg*wyNew)/mTot;
    const double usNewZ = (pZ - mg*wzNew)/mTot;

    const double ugNewX = usNewX + wxNew;
    const double ugNewY = usNewY + wyNew;
    const double ugNewZ = usNewZ + wzNew;

    const double kgNew = 0.5*mg*sqr3(ugNewX, ugNewY, ugNewZ);
    const double ksNew = 0.5*ms*sqr3(usNewX, usNewY, usNewZ);

    double diss = (kgOld + ksOld) - (kgNew + ksNew);
    if (!finiteDevice(diss) || diss < 0.0)
    {
        diss = 0.0;
    }

    s.rho[c] = mg;
    s.rhoUx[c] = mg*ugNewX;
    s.rhoUy[c] = mg*ugNewY;
    s.rhoUz[c] = mg*ugNewZ;
    s.rhoE[c] = kgNew + igAfter + diss;
}
}

int launchEulerianGasSolidCoupling
(
    DeviceState* s,
    const int grid,
    const int block,
    const double dt
)
{
    switch (s->hostDragModel)
    {
        case 0:
            applyEulerianGasSolidCouplingKernel
                <ugkwpGpuDrag::SchillerNaumannDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        case 1:
            applyEulerianGasSolidCouplingKernel
                <ugkwpGpuDrag::GidaspowErgunWenYuDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        default:
            setLastErrorText
            (
                "unsupported drag model in Eulerian coupling launch"
            );
            return 1;
    }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyEulerianGasSolidCouplingKernel launch", err);
        return 1;
    }
    return 0;
}

__global__ void applyEulerianParticleMaterialHeatKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    if
    (
        c >= s.nCells
     || s.solveParticleTemperature == 0
     || s.particleGasHeatTransferModelId == 0
    )
    {
        return;
    }

    const double heatFactor = particleHeatFactorDevice(s);
    if (heatFactor <= 0.0)
    {
        return;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const double hp = clampMin(finiteOr(s.momRhoHpP[c], 0.0), 0.0);
    if (hp <= 0.0)
    {
        return;
    }

    const double tpAvg =
        clampRange
        (
            finiteOr(hp/(rhoP*heatFactor + OfSmall), s.TpMin),
            s.TpMin,
            s.TpMax
        );

    const double rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], 0.0), s.rhoMin);
    const double tg =
        clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            1.0e30
        );

    const double dLocal =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/rhoP, s.particleDiameterFallback),
            1.0e-12
        );

    const double usx = finiteOr(s.momRhoUPx[c], 0.0)/(rhoP + OfSmall);
    const double usy = finiteOr(s.momRhoUPy[c], 0.0)/(rhoP + OfSmall);
    const double usz = finiteOr(s.momRhoUPz[c], 0.0)/(rhoP + OfSmall);

    const double urx = finiteOr(s.couplingUxOld[c], 0.0) - usx;
    const double ury = finiteOr(s.couplingUyOld[c], 0.0) - usy;
    const double urz = finiteOr(s.couplingUzOld[c], 0.0) - usz;
    const double urMag = sqrt(sqr3(urx, ury, urz));

    const double mu = clampMin(s.gasMu, 1.0e-30);
    const double re = rhoG*dLocal*urMag/mu;

    const double pr = s.gasPrClamped;
    const double gasConductivity = molecularGasConductivity(s);

    const double nu =
        2.0
      + 0.6*sqrt(clampMin(re, 0.0))
         *pow(clampMin(pr, 1.0e-12), 1.0/3.0);

    const double rate =
        6.0*nu*gasConductivity
       /(s.particleThermalRho*s.particleCp*dLocal*dLocal + 1.0e-300);

    const double gasCv =
        s.Rgas/clampMin(s.gammaGas - 1.0, OfSmall);
    const double gasCapacity = rhoG*gasCv;
    const double solidCapacity = rhoP*heatFactor;

                                                                           
                                                                             
                                                                        
    const double dHp =
        gpuFiniteCapacityHeatExchange
        (
            gasCapacity,
            solidCapacity,
            tg,
            tpAvg,
            clampMin(rate, 0.0),
            dt
        );

    const double rhoEold = finiteOr(s.rhoE[c], 0.0);
    double rhoEnew = rhoEold - dHp;

    const double rhoGCurrent =
        clampMin(finiteOr(s.rho[c], rhoG), s.rhoMin);
    const double kinetic =
        0.5
       *sqr3
        (
            finiteOr(s.rhoUx[c], 0.0),
            finiteOr(s.rhoUy[c], 0.0),
            finiteOr(s.rhoUz[c], 0.0)
        )
       /rhoGCurrent;

    const double eMinGas =
        rhoGCurrent*s.Rgas*s.TgasMin
       /clampMin(s.gammaGas - 1.0, OfSmall);

    rhoEnew = clampMin(finiteOr(rhoEnew, rhoEold), kinetic + eMinGas);
    s.rhoE[c] = rhoEnew;
}

__global__ void snapshotParticleGasCouplingStateKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.couplingRhoOld[c] =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    s.couplingUxOld[c] = finiteOr(s.Ux[c], 0.0);
    s.couplingUyOld[c] = finiteOr(s.Uy[c], 0.0);
    s.couplingUzOld[c] = finiteOr(s.Uz[c], 0.0);
    s.couplingTgasOld[c] =
        clampMin(finiteOr(s.Tgas[c], s.TgasMin), s.TgasMin);
}

__device__ unsigned long long mixSeed(unsigned long long x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ double uniform01Device(unsigned long long& state)
{
    state = mixSeed(state + 0x9e3779b97f4a7c15ULL);
    return static_cast<double>(state >> 11)*(1.0/9007199254740992.0);
}

__device__ double normalDevice(unsigned long long& state)
{
    const double u1 = clampMin(uniform01Device(state), 1.0e-12);
    const double u2 = uniform01Device(state);
    return sqrt(-2.0*log(u1))*cos(6.28318530717958647692*u2);
}
__device__ void normalPairDevice(
    unsigned long long& state,
    double& z0,
    double& z1
)
{
    const double u1 = clampMin(uniform01Device(state), 1.0e-12);
    const double u2 = uniform01Device(state);

    const double r = sqrt(-2.0*log(u1));
    double sVal = 0.0;
    double cVal = 0.0;

    sincos(6.28318530717958647692*u2, &sVal, &cVal);

    z0 = r*cVal;
    z1 = r*sVal;

}

__device__ double sampleDiameterAroundDevice
(
    const DeviceState& s,
    const double dCenter,
    unsigned long long& rng
)
{
    const double dMin = clampMin(s.particleDiameterMin, 1.0e-12);
    const double dMax =
        clampMin(s.particleDiameterMax, clampMin(dMin, 1.0e-12));
    const double dLocal = clampRange
    (
        finiteOr(dCenter, s.particleDiameterFallback),
        dMin,
        dMax
    );

    if (dMax <= dMin || s.particleDiameterSigma <= OfSmall)
    {
        return dLocal;
    }

    const double sigma = s.particleDiameterSigma;
    const double lnMin = log(dMin);
    const double lnMax = log(dMax);
    const double lnD =
        log(clampMin(dLocal, 1.0e-12))
      - 0.5*sigma*sigma
      + sigma*normalDevice(rng);

    if (!finiteDevice(lnD))
    {
        return lnD < 0.0 ? dMin : dMax;
    }
    if (lnD <= lnMin)
    {
        return dMin;
    }
    if (lnD >= lnMax)
    {
        return dMax;
    }
    return exp(lnD);
}

__device__ double sampleDiameterFromPoolMomentDevice
(
    const DeviceState& s,
    const double meanD,
    const double secondD,
    unsigned long long& rng
)
{
    const double dMin = clampMin(s.particleDiameterMin, 1.0e-12);
    const double dMax = clampMin(s.particleDiameterMax, dMin);

    const double mean =
        clampRange(finiteOr(meanD, s.particleDiameterFallback), dMin, dMax);
    const double second = clampMin(finiteOr(secondD, mean*mean), 0.0);
    const double varD = second - mean*mean;

    if (!finiteDevice(varD) || varD <= 0.0 || dMax <= dMin)
    {
        return mean;
    }

    const double sampledD = mean + sqrt(varD)*normalDevice(rng);
    if (!finiteDevice(sampledD))
    {
        return mean;
    }

    return clampRange(sampledD, dMin, dMax);
}

__device__ double radialDistributionG0Device(const double eps)
{
    double cRatio = clampRange(eps/(0.63 + 1.0e-6), 0.0, 0.99);
    return (2.0 - cRatio)/(2.0*pow(1.0 - cRatio, 3.0) + 1.0e-5);
}

__device__ double solidPressureFromMomentsDevice
(
    const DeviceState& s,
    const int c
)
{
    if (c < 0 || c >= s.nCells)
    {
        return 0.0;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return 0.0;
    }

    const double eps =
        clampRange(rhoP/clampMin(s.rhoSolid, 1.0e-300), 0.0, 1.0);
    const double ux = finiteOr(s.momRhoUPx[c], 0.0)/rhoP;
    const double uy = finiteOr(s.momRhoUPy[c], 0.0)/rhoP;
    const double uz = finiteOr(s.momRhoUPz[c], 0.0)/rhoP;
    const double kinetic = 0.5*rhoP*sqr3(ux, uy, uz);
    const double theta =
        clampMin
        (
            (finiteOr(s.momRhoEP[c], kinetic) - kinetic)/(1.5*rhoP),
            0.0
        );
    double pColl = 0.0;
    if (s.collisionalPressureEnabled)
    {
        const double g0 = radialDistributionG0Device(eps);
        pColl =
            2.0*(1.0 + s.collisionalRestitution)
           *s.rhoSolid*eps*eps*g0*theta;
        pColl = clampRange(finiteOr(pColl, 0.0), 0.0, OfGreat);
    }

    return clampRange(finiteOr(pColl, OfGreat), 0.0, OfGreat);
}

__global__ void computeCollisionalPressureKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.collisionalPressure[c] = solidPressureFromMomentsDevice(s, c);
}

__global__ void computeCollisionalPressureFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    double pFace = 0.0;
    double ufx = 0.0;
    double ufy = 0.0;
    double ufz = 0.0;

    if (own >= 0 && own < s.nCells && nei >= 0 && nei < s.nCells)
    {
        const double w = clampRange(finiteOr(s.faceWeight[f], 0.5), 0.0, 1.0);
        const double rhoOwn =
            clampMin(finiteOr(s.momRhoP[own], 0.0), s.epsSMin*s.rhoSolid);
        const double rhoNei =
            clampMin(finiteOr(s.momRhoP[nei], 0.0), s.epsSMin*s.rhoSolid);
        pFace =
            w*solidPressureFromMomentsDevice(s, own)
          + (1.0 - w)*solidPressureFromMomentsDevice(s, nei);
        ufx =
            w*finiteOr(s.momRhoUPx[own], 0.0)/rhoOwn
          + (1.0 - w)*finiteOr(s.momRhoUPx[nei], 0.0)/rhoNei;
        ufy =
            w*finiteOr(s.momRhoUPy[own], 0.0)/rhoOwn
          + (1.0 - w)*finiteOr(s.momRhoUPy[nei], 0.0)/rhoNei;
        ufz =
            w*finiteOr(s.momRhoUPz[own], 0.0)/rhoOwn
          + (1.0 - w)*finiteOr(s.momRhoUPz[nei], 0.0)/rhoNei;
    }
    else if
    (
        own >= 0
     && own < s.nCells
     && (s.gasBoundaryKind[f] == 1 || s.gasBoundaryKind[f] == 2)
    )
    {
        pFace = solidPressureFromMomentsDevice(s, own);
                                                                                 
        ufx = 0.0;
        ufy = 0.0;
        ufz = 0.0;
    }

    const double fx = pFace*s.Sfx[f];
    const double fy = pFace*s.Sfy[f];
    const double fz = pFace*s.Sfz[f];
    s.solidPressurePhiMomX[f] = finiteOr(fx, 0.0);
    s.solidPressurePhiMomY[f] = finiteOr(fy, 0.0);
    s.solidPressurePhiMomZ[f] = finiteOr(fz, 0.0);
    s.solidPressurePhiEnergy[f] =
        finiteOr(ufx*fx + ufy*fy + ufz*fz, 0.0);
}

__global__ void accumulateCollisionalPressureKickByCellKernel
(
    DeviceState* sp,
    const double kickDt,
    const int computeScale
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double dpx = 0.0;
    double dpy = 0.0;
    double dpz = 0.0;
    double de = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int plane = start + j;
        const int f = s.cellFaceId[plane];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double sign = s.faceOwner[f] == c ? 1.0 : -1.0;
        dpx -= sign*s.solidPressurePhiMomX[f];
        dpy -= sign*s.solidPressurePhiMomY[f];
        dpz -= sign*s.solidPressurePhiMomZ[f];
        de -= sign*s.solidPressurePhiEnergy[f];
    }

    const double factor = kickDt/clampMin(s.V[c], OfVSmall);
    const double deltaPx = finiteOr(factor*dpx, 0.0);
    const double deltaPy = finiteOr(factor*dpy, 0.0);
    const double deltaPz = finiteOr(factor*dpz, 0.0);
    const double deltaE = finiteOr(factor*de, 0.0);
    s.pressureDeltaMomX[c] = deltaPx;
    s.pressureDeltaMomY[c] = deltaPy;
    s.pressureDeltaMomZ[c] = deltaPz;
    s.pressureDeltaEnergy[c] = deltaE;

    if (computeScale == 0)
    {
        return;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        s.pressureKickScale[c] = 0.0;
        return;
    }

    double lambda = 1.0;
    const double dUmag = sqrt(sqr3(deltaPx, deltaPy, deltaPz))/rhoP;
    const double maxDU =
        s.pressureKickFraction*clampMin(s.cellLength[c], 1.0e-12)
       /clampMin(kickDt, OfSmall);
    if (dUmag > maxDU)
    {
        lambda = clampRange(maxDU/dUmag, 0.0, 1.0);
    }

    const double px0 = finiteOr(s.momRhoUPx[c], 0.0);
    const double py0 = finiteOr(s.momRhoUPy[c], 0.0);
    const double pz0 = finiteOr(s.momRhoUPz[c], 0.0);
    const double e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double minInternal = 1.5*rhoP*clampMin(s.thetaMin, 0.0);
    const double trialInternal =
        e0 + lambda*deltaE
      - 0.5*sqr3
        (
            px0 + lambda*deltaPx,
            py0 + lambda*deltaPy,
            pz0 + lambda*deltaPz
        )/rhoP;
    if (!finiteDevice(trialInternal) || trialInternal < minInternal)
    {
        double lo = 0.0;
        double hi = lambda;
        for (int iter = 0; iter < 20; ++iter)
        {
            const double mid = 0.5*(lo + hi);
            const double internal =
                e0 + mid*deltaE
              - 0.5*sqr3
                (
                    px0 + mid*deltaPx,
                    py0 + mid*deltaPy,
                    pz0 + mid*deltaPz
                )/rhoP;
            if (finiteDevice(internal) && internal >= minInternal)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        lambda = lo;
    }
    s.pressureKickScale[c] = clampRange(finiteOr(lambda, 0.0), 0.0, 1.0);
}

__device__ double pressureKickInternalEnergy
(
    const double rhoP,
    const double px,
    const double py,
    const double pz,
    const double energy
)
{
    return energy - 0.5*sqr3(px, py, pz)/clampMin(rhoP, OfVSmall);
}

__global__ void computeCollisionalPressureKickScaleKernel
(
    DeviceState* sp,
    const double kickDt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        s.pressureKickScale[c] = 0.0;
        return;
    }

    const double dpx = finiteOr(s.pressureDeltaMomX[c], 0.0);
    const double dpy = finiteOr(s.pressureDeltaMomY[c], 0.0);
    const double dpz = finiteOr(s.pressureDeltaMomZ[c], 0.0);
    const double de = finiteOr(s.pressureDeltaEnergy[c], 0.0);
    double lambda = 1.0;

    const double dUmag = sqrt(sqr3(dpx, dpy, dpz))/rhoP;
    const double maxDU =
        s.pressureKickFraction*clampMin(s.cellLength[c], 1.0e-12)
       /clampMin(kickDt, OfSmall);
    if (dUmag > maxDU)
    {
        lambda = clampRange(maxDU/dUmag, 0.0, 1.0);
    }

    const double px0 = finiteOr(s.momRhoUPx[c], 0.0);
    const double py0 = finiteOr(s.momRhoUPy[c], 0.0);
    const double pz0 = finiteOr(s.momRhoUPz[c], 0.0);
    const double e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double minInternal = 1.5*rhoP*clampMin(s.thetaMin, 0.0);

    const double trialInternal = pressureKickInternalEnergy
    (
        rhoP,
        px0 + lambda*dpx,
        py0 + lambda*dpy,
        pz0 + lambda*dpz,
        e0 + lambda*de
    );
    if (!finiteDevice(trialInternal) || trialInternal < minInternal)
    {
        double lo = 0.0;
        double hi = lambda;
        for (int iter = 0; iter < 20; ++iter)
        {
            const double mid = 0.5*(lo + hi);
            const double internal = pressureKickInternalEnergy
            (
                rhoP,
                px0 + mid*dpx,
                py0 + mid*dpy,
                pz0 + mid*dpz,
                e0 + mid*de
            );
            if (finiteDevice(internal) && internal >= minInternal)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        lambda = lo;
    }
    s.pressureKickScale[c] = clampRange(finiteOr(lambda, 0.0), 0.0, 1.0);
}

__global__ void scaleCollisionalPressureFaceFluxKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }
    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    double scale = own >= 0 && own < s.nCells ? s.pressureKickScale[own] : 0.0;
    if (nei >= 0 && nei < s.nCells)
    {
        scale = fmin(scale, s.pressureKickScale[nei]);
    }
    scale = clampRange(finiteOr(scale, 0.0), 0.0, 1.0);
    s.solidPressurePhiMomX[f] *= scale;
    s.solidPressurePhiMomY[f] *= scale;
    s.solidPressurePhiMomZ[f] *= scale;
    s.solidPressurePhiEnergy[f] *= scale;
}

__device__ inline void applyCollisionalPressureProjectionOneParticle
(
    DeviceState& s,
    const int i,
    const int c,
    const double ux0,
    const double uy0,
    const double uz0,
    const double ux1,
    const double uy1,
    const double uz1,
    const double theta1,
    const bool resolved,
    const double thermalScale,
    const double thetaScale
)
{
    if
    (
        i < 0
     || i >= s.particleCapacity
     || s.pStatus[i] == 0
     || s.pCellId[i] != c
    )
    {
        return;
    }
    const double dux = finiteOr(s.pux[i], ux0) - ux0;
    const double duy = finiteOr(s.puy[i], uy0) - uy0;
    const double duz = finiteOr(s.puz[i], uz0) - uz0;
    s.pux[i] = resolved ? ux1 + thermalScale*dux : ux1;
    s.puy[i] = resolved ? uy1 + thermalScale*duy : uy1;
    s.puz[i] = resolved ? uz1 + thermalScale*duz : uz1;
    s.pTheta[i] = resolved
      ? clampMin(finiteOr(s.pTheta[i], 0.0)*thetaScale, 0.0)
      : theta1;
}

template<bool SplitDirectory>
__global__ void applyCollisionalPressureProjectionKernel
(
    DeviceState* sp,
    const double kickDt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    __shared__ double scaledDelta[4];
    if (threadIdx.x == 0)
    {
        double dpx = 0.0;
        double dpy = 0.0;
        double dpz = 0.0;
        double de = 0.0;
        const int startFace = s.cellPlaneStart[c];
        const int faceCount = s.cellPlaneCount[c];
        for (int j = 0; j < faceCount; ++j)
        {
            const int f = s.cellFaceId[startFace + j];
            if (f < 0 || f >= s.nFaces)
            {
                continue;
            }
            const double sign = s.faceOwner[f] == c ? 1.0 : -1.0;
            dpx -= sign*s.solidPressurePhiMomX[f];
            dpy -= sign*s.solidPressurePhiMomY[f];
            dpz -= sign*s.solidPressurePhiMomZ[f];
            de -= sign*s.solidPressurePhiEnergy[f];
        }
        const double factor = kickDt/clampMin(s.V[c], OfVSmall);
        scaledDelta[0] = finiteOr(factor*dpx, 0.0);
        scaledDelta[1] = finiteOr(factor*dpy, 0.0);
        scaledDelta[2] = finiteOr(factor*dpz, 0.0);
        scaledDelta[3] = finiteOr(factor*de, 0.0);
        s.pressureDeltaMomX[c] = scaledDelta[0];
        s.pressureDeltaMomY[c] = scaledDelta[1];
        s.pressureDeltaMomZ[c] = scaledDelta[2];
        s.pressureDeltaEnergy[c] = scaledDelta[3];
    }
    __syncthreads();

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }
    const double px0 = finiteOr(s.momRhoUPx[c], 0.0);
    const double py0 = finiteOr(s.momRhoUPy[c], 0.0);
    const double pz0 = finiteOr(s.momRhoUPz[c], 0.0);
    const double e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double px1 = px0 + scaledDelta[0];
    const double py1 = py0 + scaledDelta[1];
    const double pz1 = pz0 + scaledDelta[2];
    const double e1 = e0 + scaledDelta[3];
    const double ux0 = px0/rhoP;
    const double uy0 = py0/rhoP;
    const double uz0 = pz0/rhoP;
    const double ux1 = px1/rhoP;
    const double uy1 = py1/rhoP;
    const double uz1 = pz1/rhoP;
    const double theta0 =
        clampMin(pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)/(1.5*rhoP), 0.0);
    const double theta1 =
        clampMin(pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(1.5*rhoP), 0.0);
    const bool resolved = theta0 > 10.0*s.thetaMin;
    const double thermalScale =
        resolved ? sqrt(clampMin(theta1/theta0, 0.0)) : 0.0;
    const double thetaScale = resolved ? thermalScale*thermalScale : 0.0;

    if constexpr (SplitDirectory)
    {
        const int baseStart = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        for (int i = baseStart + threadIdx.x; i < baseEnd; i += blockDim.x)
        {
            applyCollisionalPressureProjectionOneParticle
            (
                s, i, c, ux0, uy0, uz0, ux1, uy1, uz1, theta1,
                resolved, thermalScale, thetaScale
            );
        }
    }

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        applyCollisionalPressureProjectionOneParticle
        (
            s, i, c, ux0, uy0, uz0, ux1, uy1, uz1, theta1,
            resolved, thermalScale, thetaScale
        );
    }

    if (threadIdx.x == 0)
    {
        s.momRhoUPx[c] = px1;
        s.momRhoUPy[c] = py1;
        s.momRhoUPz[c] = pz1;
        s.momRhoEP[c] = e1;
        s.rhoUsx[c] = px1;
        s.rhoUsy[c] = py1;
        s.rhoUsz[c] = pz1;
        s.rhoEs[c] = e1;
        s.Usx[c] = ux1;
        s.Usy[c] = uy1;
        s.Usz[c] = uz1;
        s.theta[c] = theta1;
    }
}

                                                                               
                                                                            
                                                                    
__global__ void applyCollisionalPressureProjectionCellAtomicKernel
(
    DeviceState* sp,
    const double kickDt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double dpx = 0.0;
    double dpy = 0.0;
    double dpz = 0.0;
    double de = 0.0;
    const int startFace = s.cellPlaneStart[c];
    const int faceCount = s.cellPlaneCount[c];
    for (int j = 0; j < faceCount; ++j)
    {
        const int f = s.cellFaceId[startFace + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double sign = s.faceOwner[f] == c ? 1.0 : -1.0;
        dpx -= sign*s.solidPressurePhiMomX[f];
        dpy -= sign*s.solidPressurePhiMomY[f];
        dpz -= sign*s.solidPressurePhiMomZ[f];
        de -= sign*s.solidPressurePhiEnergy[f];
    }

    const double factor = kickDt/clampMin(s.V[c], OfVSmall);
    dpx = finiteOr(factor*dpx, 0.0);
    dpy = finiteOr(factor*dpy, 0.0);
    dpz = finiteOr(factor*dpz, 0.0);
    de = finiteOr(factor*de, 0.0);
    s.pressureDeltaMomX[c] = dpx;
    s.pressureDeltaMomY[c] = dpy;
    s.pressureDeltaMomZ[c] = dpz;
    s.pressureDeltaEnergy[c] = de;

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const double px1 = finiteOr(s.momRhoUPx[c], 0.0) + dpx;
    const double py1 = finiteOr(s.momRhoUPy[c], 0.0) + dpy;
    const double pz1 = finiteOr(s.momRhoUPz[c], 0.0) + dpz;
    const double e1 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0) + de;
    const double theta1 =
        clampMin
        (
            pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)/(1.5*rhoP),
            0.0
        );

    s.momRhoUPx[c] = px1;
    s.momRhoUPy[c] = py1;
    s.momRhoUPz[c] = pz1;
    s.momRhoEP[c] = e1;
    s.rhoUsx[c] = px1;
    s.rhoUsy[c] = py1;
    s.rhoUsz[c] = pz1;
    s.rhoEs[c] = e1;
    s.Usx[c] = px1/rhoP;
    s.Usy[c] = py1/rhoP;
    s.Usz[c] = pz1/rhoP;
    s.theta[c] = theta1;
}

__global__ void applyCollisionalPressureProjectionParticlesAtomicKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] == 0)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
        if (rhoP <= s.epsSMin*s.rhoSolid)
        {
            continue;
        }
        const double dpx = finiteOr(s.pressureDeltaMomX[c], 0.0);
        const double dpy = finiteOr(s.pressureDeltaMomY[c], 0.0);
        const double dpz = finiteOr(s.pressureDeltaMomZ[c], 0.0);
        const double de = finiteOr(s.pressureDeltaEnergy[c], 0.0);
        const double px1 = finiteOr(s.momRhoUPx[c], 0.0);
        const double py1 = finiteOr(s.momRhoUPy[c], 0.0);
        const double pz1 = finiteOr(s.momRhoUPz[c], 0.0);
        const double e1 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
        const double px0 = px1 - dpx;
        const double py0 = py1 - dpy;
        const double pz0 = pz1 - dpz;
        const double e0 = e1 - de;
        const double ux0 = px0/rhoP;
        const double uy0 = py0/rhoP;
        const double uz0 = pz0/rhoP;
        const double ux1 = px1/rhoP;
        const double uy1 = py1/rhoP;
        const double uz1 = pz1/rhoP;
        const double theta0 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px0, py0, pz0, e0)
               /(1.5*rhoP),
                0.0
            );
        const double theta1 =
            clampMin
            (
                pressureKickInternalEnergy(rhoP, px1, py1, pz1, e1)
               /(1.5*rhoP),
                0.0
            );
        const bool resolved = theta0 > 10.0*s.thetaMin;
        const double thermalScale =
            resolved ? sqrt(clampMin(theta1/theta0, 0.0)) : 0.0;
        const double thetaScale = resolved ? thermalScale*thermalScale : 0.0;
        const double dux = finiteOr(s.pux[i], ux0) - ux0;
        const double duy = finiteOr(s.puy[i], uy0) - uy0;
        const double duz = finiteOr(s.puz[i], uz0) - uz0;
        s.pux[i] = resolved ? ux1 + thermalScale*dux : ux1;
        s.puy[i] = resolved ? uy1 + thermalScale*duy : uy1;
        s.puz[i] = resolved ? uz1 + thermalScale*duz : uz1;
        s.pTheta[i] = resolved
          ? clampMin(finiteOr(s.pTheta[i], 0.0)*thetaScale, 0.0)
          : theta1;
    }
}

int applyCollisionalPressureKick
(
    DeviceState* s,
    const double kickDt,
    const int block,
    const int useSplitPreDirectory
)
{
    if (!s->collisionalPressureEnabled)
    {
        return 0;
    }
    if (!(kickDt > 0.0) || !std::isfinite(kickDt))
    {
        setLastErrorText("invalid collisional-pressure kick dt");
        return 1;
    }

    const int cellGrid = (s->nCells + block - 1)/block;
    const int faceGrid = (s->nFaces + block - 1)/block;
    cudaError_t err = cudaSuccess;

#define PRESSURE_LAUNCH(CALL, NAME) \
    CALL; \
    err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        setLastError(NAME, err); \
        return 1; \
    }

    PRESSURE_LAUNCH
    (
        (computeCollisionalPressureFaceFluxKernel<<<faceGrid, block>>>(s->deviceState)),
        "computeCollisionalPressureFaceFluxKernel launch"
    );
    PRESSURE_LAUNCH
    (
        (accumulateCollisionalPressureKickByCellKernel<<<cellGrid, block>>>(s->deviceState, kickDt, 1)),
        "accumulate and limit collisional pressure kick launch"
    );
    PRESSURE_LAUNCH
    (
        (scaleCollisionalPressureFaceFluxKernel<<<faceGrid, block>>>(s->deviceState)),
        "scaleCollisionalPressureFaceFluxKernel launch"
    );
    if (s->csrCellLocalPathEnabled != 0)
    {
        if (useSplitPreDirectory != 0)
        {
            PRESSURE_LAUNCH
            (
                (applyCollisionalPressureProjectionKernel<true>
                <<<s->nCells, block>>>(s->deviceState, kickDt)),
                "applyCollisionalPressureProjectionKernel<split> launch"
            );
        }
        else
        {
            PRESSURE_LAUNCH
            (
                (applyCollisionalPressureProjectionKernel<false>
                <<<s->nCells, block>>>(s->deviceState, kickDt)),
                "applyCollisionalPressureProjectionKernel<full> launch"
            );
        }
    }
    else
    {
        PRESSURE_LAUNCH
        (
            (applyCollisionalPressureProjectionCellAtomicKernel<<<cellGrid, block>>>(s->deviceState, kickDt)),
            "applyCollisionalPressureProjectionCellAtomicKernel launch"
        );
        PRESSURE_LAUNCH
        (
            (applyCollisionalPressureProjectionParticlesAtomicKernel<<<s->particleWorkGrid, block>>>(s->deviceState)),
            "applyCollisionalPressureProjectionParticlesAtomicKernel launch"
        );
    }

#undef PRESSURE_LAUNCH
    return 0;
}

                                                                           
                                                                           
                                                                       
constexpr double mobilePackingJacobiOmega = 0.8;

__global__ void clearMobilePackingActivityCountsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *s.mobilePackingActiveCellCount = 0;
        *s.mobilePackingCorrectionCellCount = 0;
        *s.mobilePackingFrontierCurrentCount = 0;
        *s.mobilePackingFrontierNextCount = 0;
    }
}

__global__ void clearMobilePackingFrontierCountKernel(int* count)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *count = 0;
    }
}

__global__ void clearMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.mobilePackingRho[c] = 0.0;
    s.mobilePackingMomX[c] = 0.0;
    s.mobilePackingMomY[c] = 0.0;
    s.mobilePackingMomZ[c] = 0.0;
    s.mobilePackingActiveCellMask[c] = 0;
    s.mobilePackingCorrectionCellMask[c] = 0;
    s.collisionalPressure[c] = 0.0;
    s.pressureKickScale[c] = 0.0;
    s.pressureDeltaMomX[c] = 0.0;
    s.pressureDeltaMomY[c] = 0.0;
    s.pressureDeltaMomZ[c] = 0.0;
}

__device__ void seedMobilePackingActivity(DeviceState& s, const int c)
{
    if (atomicCAS(&s.mobilePackingActiveCellMask[c], 0, 1) != 0)
    {
        return;
    }
    const int activeSlot = atomicAdd(s.mobilePackingActiveCellCount, 1);
    const int frontierSlot =
        atomicAdd(s.mobilePackingFrontierCurrentCount, 1);
    if (activeSlot < s.nCells)
    {
        s.mobilePackingActiveCellList[activeSlot] = c;
    }
    if (frontierSlot < s.nCells)
    {
        s.mobilePackingFrontierCurrent[frontierSlot] = c;
    }
}

__global__ void accumulateMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double ux = 0.5*
        (
            finiteOr(s.puxOld[i], s.pux[i]) + finiteOr(s.pux[i], 0.0)
        );
        const double uy = 0.5*
        (
            finiteOr(s.puyOld[i], s.puy[i]) + finiteOr(s.puy[i], 0.0)
        );
        const double uz = 0.5*
        (
            finiteOr(s.puzOld[i], s.puz[i]) + finiteOr(s.puz[i], 0.0)
        );
        atomicAdd(&s.mobilePackingRho[c], m);
        atomicAdd(&s.mobilePackingMomX[c], m*ux);
        atomicAdd(&s.mobilePackingMomY[c], m*uy);
        atomicAdd(&s.mobilePackingMomZ[c], m*uz);
    }
}

__global__ void normalizeMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const double invV = 1.0/clampMin(s.V[c], OfVSmall);
    s.mobilePackingRho[c] *= invV;
    s.mobilePackingMomX[c] *= invV;
    s.mobilePackingMomY[c] *= invV;
    s.mobilePackingMomZ[c] *= invV;
}

__device__ void mobilePackingPrimitive
(
    const DeviceState& s,
    const int c,
    double& eps,
    double& ux,
    double& uy,
    double& uz
)
{
    if (c < 0 || c >= s.nCells)
    {
        eps = 0.0;
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        return;
    }
    const double rho = clampMin(finiteOr(s.mobilePackingRho[c], 0.0), 0.0);
    eps = rho/clampMin(s.rhoSolid, OfVSmall);
    if (rho <= s.epsSMin*s.rhoSolid)
    {
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        return;
    }
    ux = finiteOr(s.mobilePackingMomX[c], 0.0)/rho;
    uy = finiteOr(s.mobilePackingMomY[c], 0.0)/rho;
    uz = finiteOr(s.mobilePackingMomZ[c], 0.0)/rho;
}

__global__ void prepareMobilePackingProjectionKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    double epsC = 0.0;
    double uxC = 0.0;
    double uyC = 0.0;
    double uzC = 0.0;
    mobilePackingPrimitive(s, c, epsC, uxC, uyC, uzC);
    double volumeFluxSum = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const double sign = own == c ? 1.0 : -1.0;
        if (nei >= 0 && nei < s.nCells)
        {
            double epsOwn = 0.0;
            double uxOwn = 0.0;
            double uyOwn = 0.0;
            double uzOwn = 0.0;
            double epsNei = 0.0;
            double uxNei = 0.0;
            double uyNei = 0.0;
            double uzNei = 0.0;
            mobilePackingPrimitive
            (
                s, own, epsOwn, uxOwn, uyOwn, uzOwn
            );
            mobilePackingPrimitive
            (
                s, nei, epsNei, uxNei, uyNei, uzNei
            );
            const double w =
                clampRange(finiteOr(s.faceWeight[f], 0.5), 0.0, 1.0);
            const double ufx = w*uxOwn + (1.0 - w)*uxNei;
            const double ufy = w*uyOwn + (1.0 - w)*uyNei;
            const double ufz = w*uzOwn + (1.0 - w)*uzNei;
            const double un =
                ufx*s.Sfx[f] + ufy*s.Sfy[f] + ufz*s.Sfz[f];
            const double epsUpwind = un >= 0.0 ? epsOwn : epsNei;
            volumeFluxSum += sign*epsUpwind*un;
        }
        else if (own == c && s.gasBoundaryKind[f] == 0)
        {
                                                                            
                                                                             
            const double un =
                uxC*s.Sfx[f] + uyC*s.Sfy[f] + uzC*s.Sfz[f];
            volumeFluxSum += epsC*fmax(un, 0.0);
        }
    }

    const double epsPred = clampMin
    (
        epsC - dt*volumeFluxSum/clampMin(s.V[c], OfVSmall),
        0.0
    );
                                                                            
                                                                            
                                                                            
    const double signedSlack = epsPred - s.packingFraction;
    s.pressureDeltaEnergy[c] = clampRange
    (
        finiteOr
        (
            s.rhoSolid*s.V[c]*signedSlack/(dt*dt),
            0.0
        ),
        -OfGreat,
        OfGreat
    );
    if (signedSlack > 0.0)
    {
        seedMobilePackingActivity(s, c);
    }
}

__global__ void expandMobilePackingActivityFrontierKernel
(
    DeviceState* sp,
    const int* frontier,
    const int* frontierCount,
    int* nextFrontier,
    int* nextFrontierCount,
    int* accumulatedCells,
    int* accumulatedCount,
    int* activityMask
)
{
    DeviceState& s = *sp;
    const int count = clampRange(*frontierCount, 0, s.nCells);
    const int frontierI = blockIdx.x*blockDim.x + threadIdx.x;
    if (frontierI >= count)
    {
        return;
    }
    const int c = frontier[frontierI];
    if (c < 0 || c >= s.nCells)
    {
        return;
    }
    const int start = s.cellPlaneStart[c];
    const int nCellFaces = s.cellPlaneCount[c];
    for (int j = 0; j < nCellFaces; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        if (nei < 0 || nei >= s.nCells)
        {
            continue;
        }
        const int other = own == c ? nei : own;
        if (other < 0 || other >= s.nCells)
        {
            continue;
        }
        if (atomicCAS(&activityMask[other], 0, 1) != 0)
        {
            continue;
        }
        const int nextSlot = atomicAdd(nextFrontierCount, 1);
        const int accumulatedSlot = atomicAdd(accumulatedCount, 1);
        if (nextSlot < s.nCells)
        {
            nextFrontier[nextSlot] = other;
        }
        if (accumulatedSlot < s.nCells)
        {
            accumulatedCells[accumulatedSlot] = other;
        }
    }
}

__global__ void initialiseMobilePackingCorrectionRegionKernel
(
    DeviceState* sp,
    const int pressureActiveCount
)
{
    DeviceState& s = *sp;
    const int activeI = blockIdx.x*blockDim.x + threadIdx.x;
    if (activeI >= pressureActiveCount)
    {
        return;
    }
    const int c = s.mobilePackingActiveCellList[activeI];
    if (c < 0 || c >= s.nCells)
    {
        return;
    }
    if (activeI == 0)
    {
        *s.mobilePackingCorrectionCellCount = pressureActiveCount;
    }
    s.mobilePackingCorrectionCellMask[c] = 1;
    s.mobilePackingCorrectionCellList[activeI] = c;
}

__device__ double mobilePackingProjectedJacobiValue
(
    const DeviceState& s,
    const int c,
    const double* oldPressure
)
{
    double diagonal = 0.0;
    double neighbourSum = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const int own = s.faceOwner[f];
        const int nei = s.faceNeighbour[f];
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        if (nei >= 0 && nei < s.nCells)
        {
            const int other = own == c ? nei : own;
            diagonal += a;
            neighbourSum +=
                a*clampMin(finiteOr(oldPressure[other], 0.0), 0.0);
        }
        else if (own == c && s.gasBoundaryKind[f] == 0)
        {
                                                                       
            diagonal += a;
        }
    }
    if (diagonal <= OfVSmall)
    {
        return 0.0;
    }

    const double rhs = clampRange
    (
        finiteOr(s.pressureDeltaEnergy[c], 0.0),
        -OfGreat,
        OfGreat
    );
    const double candidate =
        clampMin((rhs + neighbourSum)/diagonal, 0.0);
    const double relaxed =
        (1.0 - mobilePackingJacobiOmega)
       *clampMin(finiteOr(oldPressure[c], 0.0), 0.0)
      + mobilePackingJacobiOmega*candidate;
    return clampRange(finiteOr(relaxed, 0.0), 0.0, OfGreat);
}

__global__ void solveActiveMobilePackingPressureJacobiKernel
(
    DeviceState* sp,
    const int* activeCells,
    const int activeCount,
    const double* oldPressure,
    double* newPressure
)
{
    DeviceState& s = *sp;
    const int activeI = blockIdx.x*blockDim.x + threadIdx.x;
    if (activeI >= activeCount)
    {
        return;
    }
    const int c = activeCells[activeI];
    newPressure[c] =
        mobilePackingProjectedJacobiValue(s, c, oldPressure);
}

__global__ void computeMobilePackingFaceCorrectionFluxKernel
(
    DeviceState* sp,
    const double* pressure,
    const double dt
)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    const int own = s.faceOwner[f];
    const int nei = s.faceNeighbour[f];
    const bool ownerPressureActive =
        own >= 0 && own < s.nCells
     && s.mobilePackingActiveCellMask[own] != 0;
    const bool neighbourPressureActive =
        nei >= 0 && nei < s.nCells
     && s.mobilePackingActiveCellMask[nei] != 0;
    if (!ownerPressureActive && !neighbourPressureActive)
    {
        s.solidPressurePhiMomX[f] = 0.0;
        s.solidPressurePhiMomY[f] = 0.0;
        s.solidPressurePhiMomZ[f] = 0.0;
        s.solidPressurePhiEnergy[f] = 0.0;
        return;
    }
    const double invRhoSolid = 1.0/clampMin(s.rhoSolid, OfVSmall);
    double solidVolumeFlux = 0.0;
    double faceMobileFraction = 0.0;
    double velocityCorrectionFlux = 0.0;

    if (own >= 0 && own < s.nCells && nei >= 0 && nei < s.nCells)
    {
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        const double pressureOwn =
            clampMin(finiteOr(pressure[own], 0.0), 0.0);
        const double pressureNei =
            clampMin(finiteOr(pressure[nei], 0.0), 0.0);
        const double epsOwn = clampMin
        (
            finiteOr(s.mobilePackingRho[own], 0.0)*invRhoSolid,
            0.0
        );
        const double epsNei = clampMin
        (
            finiteOr(s.mobilePackingRho[nei], 0.0)*invRhoSolid,
            0.0
        );
        const double w =
            clampRange(finiteOr(s.faceWeight[f], 0.5), 0.0, 1.0);
        faceMobileFraction = clampMin
        (
            w*epsOwn + (1.0 - w)*epsNei,
            s.epsSMin
        );
                                                                              
                                                                            
                                                             
        solidVolumeFlux = finiteOr
        (
            dt*invRhoSolid*a*(pressureOwn - pressureNei),
            0.0
        );
        velocityCorrectionFlux = finiteOr
        (
            solidVolumeFlux/faceMobileFraction,
            0.0
        );
    }
    else if
    (
        own >= 0
     && own < s.nCells
     && s.gasBoundaryKind[f] == 0
    )
    {
        const double a = clampMin
        (
            finiteOr(s.magSf[f]*s.deltaCoeffs[f], 0.0),
            0.0
        );
        const double pressureOwn =
            clampMin(finiteOr(pressure[own], 0.0), 0.0);
        faceMobileFraction = clampMin
        (
            finiteOr(s.mobilePackingRho[own], 0.0)*invRhoSolid,
            s.epsSMin
        );
        solidVolumeFlux = finiteOr
        (
            dt*invRhoSolid*a*pressureOwn,
            0.0
        );
        velocityCorrectionFlux = finiteOr
        (
            solidVolumeFlux/faceMobileFraction,
            0.0
        );
    }

                                                                          
                                                                              
    s.solidPressurePhiMomX[f] = solidVolumeFlux;
    s.solidPressurePhiMomY[f] = faceMobileFraction;
    s.solidPressurePhiMomZ[f] = 0.0;
    s.solidPressurePhiEnergy[f] = velocityCorrectionFlux;
}

__global__ void reconstructMobilePackingVelocityCorrectionKernel
(
    DeviceState* sp,
    const int* correctionCells,
    const int correctionActiveCount
)
{
    DeviceState& s = *sp;
    const int activeI = blockIdx.x*blockDim.x + threadIdx.x;
    if (activeI >= correctionActiveCount)
    {
        return;
    }
    const int c = correctionCells[activeI];

                                                                  
                                                                            
                                                                              
                                               
    double m00 = 0.0;
    double m01 = 0.0;
    double m02 = 0.0;
    double m11 = 0.0;
    double m12 = 0.0;
    double m22 = 0.0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int f = s.cellFaceId[start + j];
        if (f < 0 || f >= s.nFaces)
        {
            continue;
        }
        const double magSf = clampMin(finiteOr(s.magSf[f], 0.0), 0.0);
        if (magSf <= OfVSmall)
        {
            continue;
        }
        const double nx = s.Sfx[f]/magSf;
        const double ny = s.Sfy[f]/magSf;
        const double nz = s.Sfz[f]/magSf;
        const double velocityCorrectionFlux =
            finiteOr(s.solidPressurePhiEnergy[f], 0.0);
        m00 += nx*s.Sfx[f];
        m01 += nx*s.Sfy[f];
        m02 += nx*s.Sfz[f];
        m11 += ny*s.Sfy[f];
        m12 += ny*s.Sfz[f];
        m22 += nz*s.Sfz[f];
        b0 += nx*velocityCorrectionFlux;
        b1 += ny*velocityCorrectionFlux;
        b2 += nz*velocityCorrectionFlux;
    }

    const double trace = m00 + m11 + m22;
    if (!(trace > OfVSmall) || !finiteDevice(trace))
    {
        s.pressureDeltaMomX[c] = 0.0;
        s.pressureDeltaMomY[c] = 0.0;
        s.pressureDeltaMomZ[c] = 0.0;
        return;
    }
    const double regularisation = 1.0e-12*trace;
    const double l00 = sqrt(fmax(m00 + regularisation, regularisation));
    const double l10 = m01/l00;
    const double l20 = m02/l00;
    const double l11 = sqrt
    (
        fmax(m11 + regularisation - l10*l10, regularisation)
    );
    const double l21 = (m12 - l20*l10)/l11;
    const double l22 = sqrt
    (
        fmax
        (
            m22 + regularisation - l20*l20 - l21*l21,
            regularisation
        )
    );

    const double y0 = b0/l00;
    const double y1 = (b1 - l10*y0)/l11;
    const double y2 = (b2 - l20*y0 - l21*y1)/l22;
    const double uz = y2/l22;
    const double uy = (y1 - l21*uz)/l11;
    const double ux = (y0 - l10*uy - l20*uz)/l00;
    s.pressureDeltaMomX[c] = finiteOr(ux, 0.0);
    s.pressureDeltaMomY[c] = finiteOr(uy, 0.0);
    s.pressureDeltaMomZ[c] = finiteOr(uz, 0.0);
}

__device__ void mobilePackingParticleVelocityCorrection
(
    const DeviceState& s,
    const int particleI,
    double& dux,
    double& duy,
    double& duz
)
{
    const int c = s.pCellId[particleI];
    dux = finiteOr(s.pressureDeltaMomX[c], 0.0);
    duy = finiteOr(s.pressureDeltaMomY[c], 0.0);
    duz = finiteOr(s.pressureDeltaMomZ[c], 0.0);

    int closestPlane = -1;
    double closestCoordinate = 1.0;
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    for (int j = 0; j < count; ++j)
    {
        const int plane = start + j;
        const double nx = s.planeNx[plane];
        const double ny = s.planeNy[plane];
        const double nz = s.planeNz[plane];
        const double centreToFaceDistance =
            s.planeD[plane]
          - (nx*s.Cx[c] + ny*s.Cy[c] + nz*s.Cz[c]);
        if (!(centreToFaceDistance > OfVSmall))
        {
            continue;
        }
        const double particleToFaceDistance =
            s.planeD[plane]
          - (nx*s.px[particleI] + ny*s.py[particleI] + nz*s.pz[particleI]);
        const double coordinate = clampRange
        (
            finiteOr(particleToFaceDistance/centreToFaceDistance, 1.0),
            0.0,
            1.0
        );
        if (coordinate < closestCoordinate)
        {
            closestCoordinate = coordinate;
            closestPlane = plane;
        }
    }
    if (closestPlane < 0)
    {
        return;
    }

    const int faceI = s.cellFaceId[closestPlane];
    if (faceI < 0 || faceI >= s.nFaces)
    {
        return;
    }
    const double magSf = clampMin(finiteOr(s.magSf[faceI], 0.0), 0.0);
    if (magSf <= OfVSmall)
    {
        return;
    }
    const double nx = s.planeNx[closestPlane];
    const double ny = s.planeNy[closestPlane];
    const double nz = s.planeNz[closestPlane];
    const double sign = s.faceOwner[faceI] == c ? 1.0 : -1.0;
    const double faceVelocityCorrection =
        sign*finiteOr(s.solidPressurePhiEnergy[faceI], 0.0)/magSf;
    const double cellNormalCorrection = dux*nx + duy*ny + duz*nz;
    const double faceBlend = 1.0 - closestCoordinate;
    const double normalAdjustment =
        faceBlend*(faceVelocityCorrection - cellNormalCorrection);
    dux += normalAdjustment*nx;
    duy += normalAdjustment*ny;
    duz += normalAdjustment*nz;
}

__global__ void applyMobilePackingCorrectionToParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        if (s.mobilePackingCorrectionCellMask[c] == 0)
        {
            continue;
        }
        double dux = 0.0;
        double duy = 0.0;
        double duz = 0.0;
        mobilePackingParticleVelocityCorrection(s, i, dux, duy, duz);
        s.pux[i] = finiteOr(s.pux[i], 0.0) + dux;
        s.puy[i] = finiteOr(s.puy[i], 0.0) + duy;
        s.puz[i] = finiteOr(s.puz[i], 0.0) + duz;
        s.puxOld[i] = finiteOr(s.puxOld[i], 0.0) + dux;
        s.puyOld[i] = finiteOr(s.puyOld[i], 0.0) + duy;
        s.puzOld[i] = finiteOr(s.puzOld[i], 0.0) + duz;
    }
}

int applyMobilePackingProjection
(
    DeviceState* s,
    const double dt,
    const int block
)
{
    if (!s->jammingPressureEnabled || s->particleWorkGrid <= 0)
    {
        return 0;
    }
    if (!(dt > 0.0) || !std::isfinite(dt))
    {
        setLastErrorText("invalid mobile packing-projection dt");
        return 1;
    }
    const int cellGrid = (s->nCells + block - 1)/block;
    const int faceGrid = (s->nFaces + block - 1)/block;
    cudaError_t err = cudaSuccess;

#define PACKING_LAUNCH(CALL, NAME) \
    CALL; \
    err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        setLastError(NAME, err); \
        return 1; \
    }

    PACKING_LAUNCH
    (
        (clearMobilePackingMomentsKernel<<<cellGrid, block>>>(s->deviceState)),
        "clear mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (clearMobilePackingActivityCountsKernel<<<1, 1>>>(s->deviceState)),
        "clear mobile packing activity counts launch"
    );
    PACKING_LAUNCH
    (
        (accumulateMobilePackingMomentsKernel<<<s->particleWorkGrid, block>>>
        (s->deviceState)),
        "accumulate mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (normalizeMobilePackingMomentsKernel<<<cellGrid, block>>>(s->deviceState)),
        "normalize mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (prepareMobilePackingProjectionKernel<<<cellGrid, block>>>
        (s->deviceState, dt)),
        "prepare mobile packing projection launch"
    );

    int seedCount = 0;
    if
    (
        copyToHost
        (
            &seedCount,
            s->mobilePackingFrontierCurrentCount,
            1,
            "cudaMemcpy mobile packing seed count"
        ) != 0
    )
    {
        return 1;
    }
    if (seedCount == 0)
    {
        return 0;
    }

    int* currentFrontier = s->mobilePackingFrontierCurrent;
    int* nextFrontier = s->mobilePackingFrontierNext;
    int* currentFrontierCount = s->mobilePackingFrontierCurrentCount;
    int* nextFrontierCount = s->mobilePackingFrontierNextCount;
    for (int layer = 0; layer < s->packingProjectionIterations; ++layer)
    {
        PACKING_LAUNCH
        (
            (clearMobilePackingFrontierCountKernel<<<1, 1>>>
            (nextFrontierCount)),
            "clear mobile packing next-frontier count launch"
        );
        PACKING_LAUNCH
        (
            (expandMobilePackingActivityFrontierKernel<<<cellGrid, block>>>
            (
                s->deviceState,
                currentFrontier,
                currentFrontierCount,
                nextFrontier,
                nextFrontierCount,
                s->mobilePackingActiveCellList,
                s->mobilePackingActiveCellCount,
                s->mobilePackingActiveCellMask
            )),
            "expand mobile packing pressure activity launch"
        );
        int* frontierSwap = currentFrontier;
        currentFrontier = nextFrontier;
        nextFrontier = frontierSwap;
        int* countSwap = currentFrontierCount;
        currentFrontierCount = nextFrontierCount;
        nextFrontierCount = countSwap;
    }

    int pressureActiveCount = 0;
    if
    (
        copyToHost
        (
            &pressureActiveCount,
            s->mobilePackingActiveCellCount,
            1,
            "cudaMemcpy mobile packing pressure-active count"
        ) != 0
    )
    {
        return 1;
    }
    if (pressureActiveCount < 1 || pressureActiveCount > s->nCells)
    {
        setLastErrorText("invalid mobile packing pressure-active count");
        return 1;
    }
    const int activeGrid = (pressureActiveCount + block - 1)/block;

    double* oldPressure = s->collisionalPressure;
    double* newPressure = s->pressureKickScale;
    for (int iter = 0; iter < s->packingProjectionIterations; ++iter)
    {
        PACKING_LAUNCH
        (
            (solveActiveMobilePackingPressureJacobiKernel
            <<<activeGrid, block>>>
            (
                s->deviceState,
                s->mobilePackingActiveCellList,
                pressureActiveCount,
                oldPressure,
                newPressure
            )),
            "mobile packing projected Jacobi launch"
        );
        double* swap = oldPressure;
        oldPressure = newPressure;
        newPressure = swap;
    }

    PACKING_LAUNCH
    (
        (initialiseMobilePackingCorrectionRegionKernel
        <<<activeGrid, block>>>
        (s->deviceState, pressureActiveCount)),
        "initialise mobile packing correction region launch"
    );
    PACKING_LAUNCH
    (
        (clearMobilePackingFrontierCountKernel<<<1, 1>>>
        (s->mobilePackingFrontierNextCount)),
        "clear mobile packing correction-halo frontier count launch"
    );
    PACKING_LAUNCH
    (
        (expandMobilePackingActivityFrontierKernel<<<activeGrid, block>>>
        (
            s->deviceState,
            s->mobilePackingActiveCellList,
            s->mobilePackingActiveCellCount,
            s->mobilePackingFrontierNext,
            s->mobilePackingFrontierNextCount,
            s->mobilePackingCorrectionCellList,
            s->mobilePackingCorrectionCellCount,
            s->mobilePackingCorrectionCellMask
        )),
        "expand mobile packing correction halo launch"
    );

    int correctionActiveCount = 0;
    if
    (
        copyToHost
        (
            &correctionActiveCount,
            s->mobilePackingCorrectionCellCount,
            1,
            "cudaMemcpy mobile packing correction-active count"
        ) != 0
    )
    {
        return 1;
    }
    if
    (
        correctionActiveCount < pressureActiveCount
     || correctionActiveCount > s->nCells
    )
    {
        setLastErrorText("invalid mobile packing correction-active count");
        return 1;
    }
    const int correctionGrid =
        (correctionActiveCount + block - 1)/block;
    PACKING_LAUNCH
    (
        (computeMobilePackingFaceCorrectionFluxKernel<<<faceGrid, block>>>
        (s->deviceState, oldPressure, dt)),
        "compute mobile packing face correction flux launch"
    );
    PACKING_LAUNCH
    (
        (reconstructMobilePackingVelocityCorrectionKernel
        <<<correctionGrid, block>>>
        (
            s->deviceState,
            s->mobilePackingCorrectionCellList,
            correctionActiveCount
        )),
        "reconstruct mobile packing velocity correction launch"
    );
    PACKING_LAUNCH
    (
        (applyMobilePackingCorrectionToParticlesKernel
        <<<s->particleWorkGrid, block>>>(s->deviceState)),
        "apply mobile packing particle correction launch"
    );

#undef PACKING_LAUNCH
    return 0;
}

__device__ double granularCollisionTauFromCellDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return OfGreat;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return OfGreat;
    }

    const double eps =
        clampRange(rhoP/clampMin(s.rhoSolid, 1.0e-300), 0.0, 1.0);

    const double usx = finiteOr(s.momRhoUPx[c], 0.0)/rhoP;
    const double usy = finiteOr(s.momRhoUPy[c], 0.0)/rhoP;
    const double usz = finiteOr(s.momRhoUPz[c], 0.0)/rhoP;

    const double e =
        clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0)/rhoP;

    const double theta =
        clampMin((e - 0.5*sqr3(usx, usy, usz))/1.5, 0.0);

    const double dPart =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/rhoP, s.particleDiameterFallback),
            1.0e-12
        );

    const double g0 = radialDistributionG0Device(eps);
    const double lmfp =
        sqrt(OfPi)*dPart/(12.0*eps*g0 + OfSmall);

    return clampMin(lmfp/(sqrt(theta) + OfSmall), OfSmall);
}

__global__ void injectBoundaryParticlesKernel
(
    DeviceState* sp,
    const double dt,
    const double simulationTime
)
{
    DeviceState& s = *sp;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= s.nBoundarySources)
    {
        return;
    }
#ifdef UGKP_DEVELOPMENT_PROBES
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = 0;
    }
#endif
    if (s.particleCapacity <= 0 || s.injectionParcelMass <= 0.0)
    {
        return;
    }

    const int sourceFace = s.sourceFace[j];
    if (sourceFace < 0 || sourceFace >= s.nFaces)
    {
        return;
    }

    const int sourceCell = s.sourceCell[j];
    if (sourceCell < 0 || sourceCell >= s.nCells)
    {
        return;
    }

    const bool scheduledInletActive =
        scheduledInletFaceDevice(s, sourceFace);
    const double sourceUx = scheduledInletActive
      ? finiteOr(s.gasBoundaryUx[sourceFace], finiteOr(s.sourceUx[j], 0.0))
      : finiteOr(s.sourceUx[j], 0.0);
    const double sourceUy = scheduledInletActive
      ? finiteOr(s.gasBoundaryUy[sourceFace], finiteOr(s.sourceUy[j], 0.0))
      : finiteOr(s.sourceUy[j], 0.0);
    const double sourceUz = scheduledInletActive
      ? finiteOr(s.gasBoundaryUz[sourceFace], finiteOr(s.sourceUz[j], 0.0))
      : finiteOr(s.sourceUz[j], 0.0);
    const double sourceTheta = scheduledInletActive
      ? clampMin(finiteOr(s.theta[sourceCell], 0.0), 0.0)
      : clampMin(finiteOr(s.sourceTheta[j], 0.0), 0.0);
    const double scheduledRate =
        scheduledSolidVolumeFractionDevice(s, simulationTime)
       *s.rhoSolid
       *clampMin
        (
           -(sourceUx*s.Sfx[sourceFace]
           + sourceUy*s.Sfy[sourceFace]
           + sourceUz*s.Sfz[sourceFace]),
            0.0
        );
    const double rate = scheduledInletActive
      ? scheduledRate
      : finiteOr(s.sourceMassRate[j], 0.0);
    if (rate <= 0.0)
    {
        return;
    }

    const double injectionParcelMass = s.injectionParcelMass;
    double available =
        clampMin(finiteOr(s.sourceResidualMass[j], 0.0), 0.0) + rate*dt;
    if (available < injectionParcelMass)
    {
        s.sourceResidualMass[j] = available;
        return;
    }

    const int nNew = static_cast<int>(floor(available/injectionParcelMass));
    double consumed = 0.0;
    int created = 0;

    for (int k = 0; k < nNew; ++k)
    {
        const int slot = atomicAdd(s.particleCountDevice, 1);
        if (slot >= s.particleCapacity)
        {
            atomicSub(s.particleCountDevice, 1);
            break;
        }

        unsigned long long rng =
            mixSeed
            (
                s.rngSeed
              ^ static_cast<unsigned long long>
                (
                    0xd1b54a32d192ed03ULL
                  + (static_cast<unsigned long long>(j) << 32)
                  + static_cast<unsigned long long>(slot)
                )
            );
        s.px[slot] = finiteOr(s.sourcePx[j], s.Cx[sourceCell]);
        s.py[slot] = finiteOr(s.sourcePy[j], s.Cy[sourceCell]);
        s.pz[slot] = finiteOr(s.sourcePz[j], s.Cz[sourceCell]);
        s.pux[slot] = sourceUx;
        s.puy[slot] = sourceUy;
        s.puz[slot] = sourceUz;
        s.pT[slot] =
            clampRange(finiteOr(s.sourceT[j], s.TpMin), s.TpMin, s.TpMax);
        s.pTheta[slot] = sourceTheta;
        s.pd[slot] = sampleDiameterAroundDevice(s, finiteOr(s.sourceD[j], s.particleDiameterFallback), rng);
        s.pm[slot] = injectionParcelMass;
        s.pCellId[slot] = sourceCell;
        s.pStatus[slot] = 1;
        s.pRng[slot] = rng;
        s.pOrigId[slot] =
            (static_cast<unsigned long long>(j) << 40)
          ^ static_cast<unsigned long long>(slot);
        consumed += injectionParcelMass;
        ++created;
    }

    s.sourceResidualMass[j] = available - consumed;
#ifdef UGKP_DEVELOPMENT_PROBES
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = created;
    }
#endif
}

__device__ bool pointInsideCell(const DeviceState& s, const int c, const double x, const double y, const double z)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const double tol = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const double dist =
            s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
        if (dist > tol)
        {
            return false;
        }
    }
    return true;
}

__device__ int mostViolatedPlane
(
    const DeviceState& s,
    const int c,
    const double x,
    const double y,
    const double z
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    int plane = -1;
    double maxDist = -1.0e300;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const double dist =
            s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
        if (dist > maxDist)
        {
            maxDist = dist;
            plane = p;
        }
    }
    return plane;
}

__device__ int firstSegmentIntersection
(
    const DeviceState& s,
    const int c,
    const double x0,
    const double y0,
    const double z0,
    const double x1,
    const double y1,
    const double z1,
    double& hitT
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const double tol = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
    int plane = -1;
    double bestT = 2.0;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const double d0 =
            s.planeNx[p]*x0 + s.planeNy[p]*y0 + s.planeNz[p]*z0 - s.planeD[p];
        const double d1 =
            s.planeNx[p]*x1 + s.planeNy[p]*y1 + s.planeNz[p]*z1 - s.planeD[p];
        if (d1 <= tol)
        {
            continue;
        }

        const double denom = d1 - d0;
        double t = 1.0;
        if (denom > 1.0e-300)
        {
            t = (tol - d0)/denom;
        }
        t = clampRange(t, 0.0, 1.0);
        if (t < bestT)
        {
            bestT = t;
            plane = p;
        }
    }

    hitT = bestT <= 1.0 ? bestT : 1.0;
    if (plane < 0)
    {
        plane = mostViolatedPlane(s, c, x1, y1, z1);
    }
    return plane;
}

__device__ void trackOneParticleLocalFaceWalk(DeviceState& s, const int i, const double dt)
{
    if (i >= s.particleCapacity || s.pStatus[i] == 0)
    {
        return;
    }

    int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    double x = s.px[i];
    double y = s.py[i];
    double z = s.pz[i];
    double vxStep = 0.5*(s.puxOld[i] + s.pux[i]);
    double vyStep = 0.5*(s.puyOld[i] + s.puy[i]);
    double vzStep = 0.5*(s.puzOld[i] + s.puz[i]);
    double remainingDt = dt;
    bool inside = pointInsideCell(s, c, x, y, z);
    for (int hop = 0; hop < s.maxFaceWalkHops && remainingDt > 0.0; ++hop)
    {
        const double x1 = x + vxStep*remainingDt;
        const double y1 = y + vyStep*remainingDt;
        const double z1 = z + vzStep*remainingDt;
        inside = pointInsideCell(s, c, x1, y1, z1);
        if (inside)
        {
            x = x1;
            y = y1;
            z = z1;
            remainingDt = 0.0;
            break;
        }

        double hitT = 1.0;
        const int plane = firstSegmentIntersection(s, c, x, y, z, x1, y1, z1, hitT);
        if (plane < 0)
        {
            s.pStatus[i] = 0;
            return;
        }

        const int kind = s.cellFaceKind[plane];
        const int next = s.cellFaceNeighbor[plane];
        const double xHit = x + hitT*(x1 - x);
        const double yHit = y + hitT*(y1 - y);
        const double zHit = z + hitT*(z1 - z);
        remainingDt *= clampRange(1.0 - hitT, 0.0, 1.0);
        if (kind == 0 && next >= 0 && next < s.nCells)
        {
            c = next;
            x = xHit;
            y = yHit;
            z = zHit;
        }
        else if (kind == 5 && next >= 0 && next < s.nCells)
        {
            const int faceI = s.cellFaceId[plane];
            if (!isPeriodicFace(s, faceI))
            {
                s.pStatus[i] = 0;
                return;
            }
            const double eps =
                1.0e-9*clampMin(s.cellLength[next], 1.0e-12);
            c = next;
            x = xHit - s.facePeriodicDx[faceI] + eps*s.planeNx[plane];
            y = yHit - s.facePeriodicDy[faceI] + eps*s.planeNy[plane];
            z = zHit - s.facePeriodicDz[faceI] + eps*s.planeNz[plane];
        }
        else if (kind == 1 || kind == 2)
        {
            const double nx = s.planeNx[plane];
            const double ny = s.planeNy[plane];
            const double nz = s.planeNz[plane];
            const double un = s.pux[i]*nx + s.puy[i]*ny + s.puz[i]*nz;
            const double unStep = vxStep*nx + vyStep*ny + vzStep*nz;
            const double restitution = s.cellFaceRestitution[plane];
            s.pux[i] -= (1.0 + restitution)*un*nx;
            s.puy[i] -= (1.0 + restitution)*un*ny;
            s.puz[i] -= (1.0 + restitution)*un*nz;
            vxStep -= (1.0 + restitution)*unStep*nx;
            vyStep -= (1.0 + restitution)*unStep*ny;
            vzStep -= (1.0 + restitution)*unStep*nz;

            const double eps = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
            x = xHit - eps*nx;
            y = yHit - eps*ny;
            z = zHit - eps*nz;
        }
        else
        {
            const double exitMass = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
            s.pStatus[i] = 0;
            return;
        }

        if (c < 0 || c >= s.nCells)
        {
            s.pStatus[i] = 0;
            return;
        }
    }

    inside = pointInsideCell(s, c, x, y, z);
    if (!inside || remainingDt > 0.0)
    {
        s.pStatus[i] = 0;
        return;
    }
    s.px[i] = x;
    s.py[i] = y;
    s.pz[i] = z;
    s.pCellId[i] = c;
}

__global__ void trackParticlesLocalFaceWalkKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        trackOneParticleLocalFaceWalk(s, i, dt);
    }
}

template<class DragModel>
__device__ void relaxOneParticleToResidentGas(DeviceState& s, const int i, const double dt)
{
    if (i >= s.particleCapacity || s.pStatus[i] == 0)
    {
        return;
    }

    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    const double rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], s.rhoMin), s.rhoMin);
    const double ugx = finiteOr(s.couplingUxOld[c], 0.0);
    const double ugy = finiteOr(s.couplingUyOld[c], 0.0);
    const double ugz = finiteOr(s.couplingUzOld[c], 0.0);
    const double ux = s.pux[i];
    const double uy = s.puy[i];
    const double uz = s.puz[i];
    s.puxOld[i] = ux;
    s.puyOld[i] = uy;
    s.puzOld[i] = uz;

    const double relX = ugx - ux;
    const double relY = ugy - uy;
    const double relZ = ugz - uz;
    const double relMag = sqrt(sqr3(relX, relY, relZ));
    const double dPart =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            1.0e-12
        );
    const double mu = clampMin(s.gasMu, 1.0e-30);
    const ugkwpGpuDrag::DragInput dragInput =
    {
        rhoG,
        mu,
        1.0 - solidEpsFromMomentDevice(s, c),
        s.rhoSolid,
        dPart,
        relMag,
        1.0e-300,
        s.dragParameter0,
        s.dragParameter1,
        s.dragParameter2,
        s.dragParameter3
    };
    const double re = DragModel::reynoldsNumber(dragInput);
    if (s.dragModel != 2)
    {
    const double invTauDrag = DragModel::inverseResponseTime(dragInput);

    const double xDrag = dt*clampMin(invTauDrag, 0.0);
        const double alpha = exp(-xDrag);
        const double ax =
            s.gravityX - finiteOr(s.gradPx[c], 0.0)*s.invRhoSolid;
        const double ay =
            s.gravityY - finiteOr(s.gradPy[c], 0.0)*s.invRhoSolid;
        const double az =
            s.gravityZ - finiteOr(s.gradPz[c], 0.0)*s.invRhoSolid;

        const double impulse =
            (xDrag < 1.0e-6)
          ? dt*(1.0 - 0.5*xDrag + xDrag*xDrag/6.0)
          : (1.0 - alpha)/(invTauDrag + 1.0e-300);
    const double uNewX = ugx + (ux - ugx)*alpha + ax*impulse;
    const double uNewY = ugy + (uy - ugy)*alpha + ay*impulse;
    const double uNewZ = ugz + (uz - ugz)*alpha + az*impulse;
    if
    (
        nonFiniteDevice(uNewX) || nonFiniteDevice(uNewY)
     || nonFiniteDevice(uNewZ)
    )
    {
        asm("trap;");
    }
        s.pux[i] = finiteOr(uNewX, ux);
        s.puy[i] = finiteOr(uNewY, uy);
    s.puz[i] = finiteOr(uNewZ, uz);

                                                                      
                                                                          
                                                                          
                                                                              
    const double alphaTheta =
        clampRange(finiteOr(s.thetaDragAlpha[c], 1.0), 0.0, 1.0);
    const double unresolvedTheta =
        clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
    s.pTheta[i] = unresolvedTheta*alphaTheta;
    }

    if
    (
        s.solveParticleTemperature != 0
     && s.particleGasHeatTransferModelId != 0
     && s.particleThermalRho > 0.0
     && s.particleCp > 0.0
    )
    {
        const double gasConductivity = molecularGasConductivity(s);
        const double nu =
            2.0 + 0.6*sqrt(clampMin(re, 0.0))*s.gasPrOneThird;
        const double rate =
            6.0*nu*gasConductivity
           /(s.particleThermalCapacity*dPart*dPart + 1.0e-300);
        const double decay = exp(-clampMin(rate, 0.0)*dt);
        const double tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const double tgOld =
            clampRange
            (
                finiteOr(s.couplingTgasOld[c], s.TgasMin),
                s.TgasMin,
                1.0e30
            );
        const double tpNew =
            clampRange(tgOld + (tpOld - tgOld)*decay, s.TpMin, s.TpMax);
        s.pT[i] = tpNew;
    }
}

template<class DragModel>
__global__ void relaxParticlesToResidentGasKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        relaxOneParticleToResidentGas<DragModel>(s, i, dt);
    }
}

int launchParticleDragRelaxation
(
    DeviceState* s,
    const int grid,
    const int block,
    const double dt
)
{
    switch (s->hostDragModel)
    {
        case 0:
            relaxParticlesToResidentGasKernel
                <ugkwpGpuDrag::SchillerNaumannDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        case 1:
            relaxParticlesToResidentGasKernel
                <ugkwpGpuDrag::GidaspowErgunWenYuDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        case 2:
            relaxParticlesToResidentGasKernel
                <ugkwpGpuDrag::SchillerNaumannDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        default:
            setLastErrorText
            (
                "unsupported drag model in particle relaxation launch"
            );
            return 1;
    }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("relaxParticlesToResidentGasKernel launch", err);
        return 1;
    }
    return 0;
}

__global__ void clearPoissonThermalPoolKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
#ifdef UGKP_DEVELOPMENT_PROBES
    if (c == 0 && s.diagnosticPreTransportParticleCount != nullptr)
    {
        *s.diagnosticPreTransportParticleCount =
            clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    }
#endif
    if (c >= s.nCells)
    {
        return;
    }

    s.poolThermalCount[c] = 0;
    s.poolThermalSumUx[c] = 0.0;
    s.poolThermalSumUy[c] = 0.0;
    s.poolThermalSumUz[c] = 0.0;
    s.poolThermalSumU2[c] = 0.0;
    s.poissonPoolSampleTargetCount[c] = 0;
    s.poissonPoolMass[c] = 0.0;
    s.poissonPoolMomX[c] = 0.0;
    s.poissonPoolMomY[c] = 0.0;
    s.poissonPoolMomZ[c] = 0.0;
    s.poissonPoolEnergy[c] = 0.0;
    s.poissonPoolDiameter[c] = 0.0;
    s.poissonPoolDiameter2[c] = 0.0;
}

template<int NumComponents>
__device__ void blockReduceComponentSums
(
    double (&sums)[NumComponents],
    double* warpPartials
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31)/32;
    constexpr unsigned int fullWarpMask = 0xffffffffu;

                                                                             
                                                                           
                                                                             
                                                          
    if ((blockDim.x & 31) != 0)
    {
        asm("trap;");
    }
    __syncwarp(fullWarpMask);

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            const double other =
                __shfl_down_sync(fullWarpMask, sums[component], offset);
            if (lane + offset < 32)
            {
                sums[component] += other;
            }
        }
    }

    if (lane == 0)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            warpPartials[component*warpCount + warp] = sums[component];
        }
    }

    __syncthreads();

    if (warp == 0)
    {
        __syncwarp(fullWarpMask);

        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            double value =
                lane < warpCount
              ? warpPartials[component*warpCount + lane]
              : 0.0;

            for (int offset = 16; offset > 0; offset >>= 1)
            {
                const double other =
                    __shfl_down_sync(fullWarpMask, value, offset);
                if (lane + offset < 32)
                {
                    value += other;
                }
            }

            if (lane == 0)
            {
                sums[component] = value;
            }
        }
    }
}

__device__ inline void accumulateOnePoissonPoolParticle
(
    DeviceState& s,
    const int i,
    const int c,
    const double prob,
    double& locMass,
    double& locMomX,
    double& locMomY,
    double& locMomZ,
    double& locEnergy,
    double& locDiameter,
    double& locDiameter2,
    double& locCount
)
{
    if
    (
        i < 0
     || i >= s.particleCapacity
     || s.pStatus[i] == 0
     || s.pCellId[i] != c
    )
    {
        return;
    }

    unsigned long long rng = s.pRng[i];
    if (uniform01Device(rng) >= prob)
    {
        s.pRng[i] = rng;
        return;
    }

    const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
    const double ux = finiteOr(s.pux[i], 0.0);
    const double uy = finiteOr(s.puy[i], 0.0);
    const double uz = finiteOr(s.puz[i], 0.0);
    const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
    const double d =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            1.0e-12
        );
    const double specificEnergy = 0.5*sqr3(ux, uy, uz) + 1.5*theta;

    if
    (
        nonFiniteDevice(m) || m < 0.0
     || nonFiniteDevice(ux) || nonFiniteDevice(uy) || nonFiniteDevice(uz)
     || nonFiniteDevice(theta) || theta < 0.0
     || nonFiniteDevice(specificEnergy)
    )
    {
        asm("trap;");
    }

    locMass += m;
    locMomX += m*ux;
    locMomY += m*uy;
    locMomZ += m*uz;
    locEnergy += m*specificEnergy;
    locDiameter += m*d;
    locDiameter2 += m*d*d;
    locCount += 1.0;
    s.pRng[i] = rng;
    s.pStatus[i] = 2;
}

template<bool HeavyReductionEnabled>
__global__ void accumulatePoissonPoolParticlesByCellKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;

    if (c >= s.nCells)
    {
        return;
    }

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    if constexpr (HeavyReductionEnabled)
    {
        return;
    }

    __shared__ double cellCollisionProbability;

    if (threadIdx.x == 0)
    {
        const double tauColl = granularCollisionTauFromCellDevice(s, c);

        if (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
        {
            cellCollisionProbability = 0.0;
        }
        else
        {
            cellCollisionProbability =
                clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
        }
    }

    __syncthreads();

    const double prob = cellCollisionProbability;

    if (prob <= 0.0)
    {
        return;
    }

    double locMass = 0.0;
    double locMomX = 0.0;
    double locMomY = 0.0;
    double locMomZ = 0.0;
    double locEnergy = 0.0;
    double locDiameter = 0.0;
    double locDiameter2 = 0.0;
    double locCount = 0.0;

    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        accumulateOnePoissonPoolParticle
        (
            s, i, c, prob, locMass, locMomX, locMomY, locMomZ,
            locEnergy, locDiameter, locDiameter2, locCount
        );
    }

    extern __shared__ double sh[];

    double* shMass = sh;
    double* shMomX = shMass + blockDim.x;
    double* shMomY = shMomX + blockDim.x;
    double* shMomZ = shMomY + blockDim.x;
    double* shEnergy = shMomZ + blockDim.x;
    double* shDiameter = shEnergy + blockDim.x;
    double* shDiameter2 = shDiameter + blockDim.x;
    double* shCount = shDiameter2 + blockDim.x;

    shMass[threadIdx.x] = locMass;
    shMomX[threadIdx.x] = locMomX;
    shMomY[threadIdx.x] = locMomY;
    shMomZ[threadIdx.x] = locMomZ;
    shEnergy[threadIdx.x] = locEnergy;
    shDiameter[threadIdx.x] = locDiameter;
    shDiameter2[threadIdx.x] = locDiameter2;
    shCount[threadIdx.x] = locCount;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shMass[threadIdx.x] += shMass[threadIdx.x + stride];
            shMomX[threadIdx.x] += shMomX[threadIdx.x + stride];
            shMomY[threadIdx.x] += shMomY[threadIdx.x + stride];
            shMomZ[threadIdx.x] += shMomZ[threadIdx.x + stride];
            shEnergy[threadIdx.x] += shEnergy[threadIdx.x + stride];
            shDiameter[threadIdx.x] += shDiameter[threadIdx.x + stride];
            shDiameter2[threadIdx.x] += shDiameter2[threadIdx.x + stride];
            shCount[threadIdx.x] += shCount[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        s.poissonPoolMass[c] = shMass[0];
        s.poissonPoolMomX[c] = shMomX[0];
        s.poissonPoolMomY[c] = shMomY[0];
        s.poissonPoolMomZ[c] = shMomZ[0];
        s.poissonPoolEnergy[c] = shEnergy[0];
        s.poissonPoolDiameter[c] = shDiameter[0];
        s.poissonPoolDiameter2[c] = shDiameter2[0];
        s.poolThermalCount[c] = static_cast<int>(shCount[0]);

    }
}

template
<
    bool DirectParticleIndex,
    bool AddToExistingPool,
    bool HeavyReductionEnabled
>
__global__ void accumulatePoissonPoolSplitSegmentByCellKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;

    if (c >= s.nCells)
    {
        return;
    }

    const int start = DirectParticleIndex
      ? s.preBaseCellOffset[c]
      : s.cellParticleOffset[c];
    const int end = DirectParticleIndex
      ? s.preBaseCellOffset[c + 1]
      : s.cellParticleOffset[c + 1];
    if (start >= end)
    {
        return;
    }

    if constexpr (HeavyReductionEnabled)
    {
        return;
    }

    __shared__ double cellCollisionProbability;

    if (threadIdx.x == 0)
    {
        const double tauColl = granularCollisionTauFromCellDevice(s, c);

        if (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
        {
            cellCollisionProbability = 0.0;
        }
        else
        {
            cellCollisionProbability =
                clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
        }
    }

    __syncthreads();

    const double prob = cellCollisionProbability;

    if (prob <= 0.0)
    {
        return;
    }

    double locMass = 0.0;
    double locMomX = 0.0;
    double locMomY = 0.0;
    double locMomZ = 0.0;
    double locEnergy = 0.0;
    double locDiameter = 0.0;
    double locDiameter2 = 0.0;
    double locCount = 0.0;

    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = DirectParticleIndex
          ? pos
          : s.sortedParticleIndex[pos];
        accumulateOnePoissonPoolParticle
        (
            s, i, c, prob, locMass, locMomX, locMomY, locMomZ,
            locEnergy, locDiameter, locDiameter2, locCount
        );
    }

    extern __shared__ double sh[];

    double* shMass = sh;
    double* shMomX = shMass + blockDim.x;
    double* shMomY = shMomX + blockDim.x;
    double* shMomZ = shMomY + blockDim.x;
    double* shEnergy = shMomZ + blockDim.x;
    double* shDiameter = shEnergy + blockDim.x;
    double* shDiameter2 = shDiameter + blockDim.x;
    double* shCount = shDiameter2 + blockDim.x;

    shMass[threadIdx.x] = locMass;
    shMomX[threadIdx.x] = locMomX;
    shMomY[threadIdx.x] = locMomY;
    shMomZ[threadIdx.x] = locMomZ;
    shEnergy[threadIdx.x] = locEnergy;
    shDiameter[threadIdx.x] = locDiameter;
    shDiameter2[threadIdx.x] = locDiameter2;
    shCount[threadIdx.x] = locCount;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shMass[threadIdx.x] += shMass[threadIdx.x + stride];
            shMomX[threadIdx.x] += shMomX[threadIdx.x + stride];
            shMomY[threadIdx.x] += shMomY[threadIdx.x + stride];
            shMomZ[threadIdx.x] += shMomZ[threadIdx.x + stride];
            shEnergy[threadIdx.x] += shEnergy[threadIdx.x + stride];
            shDiameter[threadIdx.x] += shDiameter[threadIdx.x + stride];
            shDiameter2[threadIdx.x] += shDiameter2[threadIdx.x + stride];
            shCount[threadIdx.x] += shCount[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if constexpr (AddToExistingPool)
        {
            s.poissonPoolMass[c] += shMass[0];
            s.poissonPoolMomX[c] += shMomX[0];
            s.poissonPoolMomY[c] += shMomY[0];
            s.poissonPoolMomZ[c] += shMomZ[0];
            s.poissonPoolEnergy[c] += shEnergy[0];
            s.poissonPoolDiameter[c] += shDiameter[0];
            s.poissonPoolDiameter2[c] += shDiameter2[0];
            s.poolThermalCount[c] += static_cast<int>(shCount[0]);
        }
        else
        {
            s.poissonPoolMass[c] = shMass[0];
            s.poissonPoolMomX[c] = shMomX[0];
            s.poissonPoolMomY[c] = shMomY[0];
            s.poissonPoolMomZ[c] = shMomZ[0];
            s.poissonPoolEnergy[c] = shEnergy[0];
            s.poissonPoolDiameter[c] = shDiameter[0];
            s.poissonPoolDiameter2[c] = shDiameter2[0];
            s.poolThermalCount[c] = static_cast<int>(shCount[0]);
        }
    }
}

int launchSplitPrePoissonPoolLightReduction
(
    DeviceState* s,
    const double dt,
    const int block,
    const size_t sharedBytes
)
{
    cudaError_t err = cudaSuccess;

    if (s->csrHeavyReductionEnabled != 0)
    {
        accumulatePoissonPoolSplitSegmentByCellKernel<true, false, true>
            <<<s->nCells, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulatePoissonPoolSplitSegmentByCellKernel<true, false, false>
            <<<s->nCells, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("split-Dpre Base collision-pool launch", err);
        return 1;
    }

    if (s->preInjectionSegmentActive != 0)
    {
        if (s->csrHeavyReductionEnabled != 0)
        {
            accumulatePoissonPoolSplitSegmentByCellKernel<false, true, true>
                <<<s->nCells, block, sharedBytes>>>(s->deviceState, dt);
        }
        else
        {
            accumulatePoissonPoolSplitSegmentByCellKernel<false, true, false>
                <<<s->nCells, block, sharedBytes>>>(s->deviceState, dt);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("split-Dpre Injection collision-pool launch", err);
            return 1;
        }
    }
    return 0;
}

template<bool PoissonMode>
__global__ void accumulateParticlePoolAtomicKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] == 0)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }

        const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
        if (!PoissonMode && theta <= 10.0*s.thetaMin)
        {
            continue;
        }
        if (PoissonMode)
        {
            const double tauColl = granularCollisionTauFromCellDevice(s, c);
            const double probability =
                (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
              ? 0.0
              : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
            if (probability <= 0.0)
            {
                continue;
            }
            unsigned long long rng = s.pRng[i];
            if (uniform01Device(rng) >= probability)
            {
                s.pRng[i] = rng;
                continue;
            }
            s.pRng[i] = rng;
        }

        const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double ux = finiteOr(s.pux[i], 0.0);
        const double uy = finiteOr(s.puy[i], 0.0);
        const double uz = finiteOr(s.puz[i], 0.0);
        const double d =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        const double specificEnergy =
            0.5*sqr3(ux, uy, uz) + 1.5*theta;

        if
        (
            nonFiniteDevice(m) || m < 0.0
         || nonFiniteDevice(ux) || nonFiniteDevice(uy)
         || nonFiniteDevice(uz)
         || nonFiniteDevice(theta) || theta < 0.0
         || nonFiniteDevice(specificEnergy)
        )
        {
            asm("trap;");
        }

        atomicAdd(&s.poissonPoolMass[c], m);
        atomicAdd(&s.poissonPoolMomX[c], m*ux);
        atomicAdd(&s.poissonPoolMomY[c], m*uy);
        atomicAdd(&s.poissonPoolMomZ[c], m*uz);
        atomicAdd(&s.poissonPoolEnergy[c], m*specificEnergy);
        atomicAdd(&s.poissonPoolDiameter[c], m*d);
        atomicAdd(&s.poissonPoolDiameter2[c], m*d*d);
        atomicAdd(&s.poolThermalCount[c], 1);
        s.pStatus[i] = 2;
    }
}

template<bool PoissonMode>
__device__ void accumulateCsrHeavyPoolTask
(
    DeviceState& s,
    const int c,
    const int begin,
    const int end,
    const bool directParticleIndex,
    const double collisionProbability,
    double (&sums)[8],
    double* warpPartials
)
{
    double mass = 0.0;
    double momX = 0.0;
    double momY = 0.0;
    double momZ = 0.0;
    double energy = 0.0;
    double diameter = 0.0;
    double diameter2 = 0.0;
    double count = 0.0;

    if (!PoissonMode || collisionProbability > 0.0)
    {
        for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x)
        {
            const int i = directParticleIndex
              ? pos
              : s.sortedParticleIndex[pos];
            if
            (
                i < 0
             || i >= s.particleCapacity
             || s.pStatus[i] == 0
             || s.pCellId[i] != c
            )
            {
                continue;
            }

            const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
            if (!PoissonMode && theta <= 10.0*s.thetaMin)
            {
                continue;
            }

            if (PoissonMode)
            {
                unsigned long long rng = s.pRng[i];
                if (uniform01Device(rng) >= collisionProbability)
                {
                    s.pRng[i] = rng;
                    continue;
                }
                s.pRng[i] = rng;
            }

            const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
            const double ux = finiteOr(s.pux[i], 0.0);
            const double uy = finiteOr(s.puy[i], 0.0);
            const double uz = finiteOr(s.puz[i], 0.0);
            const double d =
                clampMin
                (
                    finiteOr(s.pd[i], s.particleDiameterFallback),
                    1.0e-12
                );
            const double specificEnergy =
                0.5*sqr3(ux, uy, uz) + 1.5*theta;

            if
            (
                nonFiniteDevice(m) || m < 0.0
             || nonFiniteDevice(ux) || nonFiniteDevice(uy)
             || nonFiniteDevice(uz)
             || nonFiniteDevice(theta) || theta < 0.0
             || nonFiniteDevice(specificEnergy)
            )
            {
                asm("trap;");
            }

            mass += m;
            momX += m*ux;
            momY += m*uy;
            momZ += m*uz;
            energy += m*specificEnergy;
            diameter += m*d;
            diameter2 += m*d*d;
            count += 1.0;
            s.pStatus[i] = 2;
        }
    }

    sums[0] = mass;
    sums[1] = momX;
    sums[2] = momY;
    sums[3] = momZ;
    sums[4] = energy;
    sums[5] = diameter;
    sums[6] = diameter2;
    sums[7] = count;
    blockReduceComponentSums<8>(sums, warpPartials);
}

template
<
    bool PoissonMode,
    bool DirectParticleIndex,
    bool InjectionTasks
>
__global__ void accumulateCsrHeavyPoolTasksPersistentKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ double collisionProbability;
    extern __shared__ double warpPartials[];

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        const int taskCount = InjectionTasks
          ? *s.preInjectionHeavyTaskCount
          : *s.csrHeavyTaskCount;
        if (task >= taskCount)
        {
            return;
        }

        const int c = InjectionTasks
          ? s.preInjectionHeavyTaskCell[task]
          : s.csrHeavyTaskCell[task];
        if (threadIdx.x == 0)
        {
            if (PoissonMode)
            {
                const double tauColl = granularCollisionTauFromCellDevice(s, c);
                collisionProbability =
                    (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
                  ? 0.0
                  : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
            }
            else
            {
                collisionProbability = 1.0;
            }
        }
        __syncthreads();

        double sums[8];
        const int begin = InjectionTasks
          ? s.preInjectionHeavyTaskBegin[task]
          : s.csrHeavyTaskBegin[task];
        const int end = InjectionTasks
          ? s.preInjectionHeavyTaskEnd[task]
          : s.csrHeavyTaskEnd[task];
        accumulateCsrHeavyPoolTask<PoissonMode>
        (
            s,
            c,
            begin,
            end,
            DirectParticleIndex,
            collisionProbability,
            sums,
            warpPartials
        );
        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                double* partials = InjectionTasks
                  ? s.preInjectionHeavyPartials
                  : s.csrHeavyPartials;
                partials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ] = sums[component];
            }
        }
        __syncthreads();
    }
}

__global__ void finalizeSplitPreCsrHeavyPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int heavyCellIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            heavyCellIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (heavyCellIndex >= *s.csrHeavyCellCount)
        {
            return;
        }

        const int c = s.csrHeavyCellList[heavyCellIndex];
        double sums[8] =
        {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
        const int baseFirstTask = s.csrHeavyCellTaskStart[c];
        const int baseTaskCount = s.csrHeavyCellTaskCount[c];
        for
        (
            int localTask = threadIdx.x;
            localTask < baseTaskCount;
            localTask += blockDim.x
        )
        {
            const int task = baseFirstTask + localTask;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] +=
                    s.csrHeavyPartials
                    [
                        8u*static_cast<size_t>(task)
                      + static_cast<size_t>(component)
                    ];
            }
        }

        const int injectionFirstTask =
            s.preInjectionHeavyCellTaskStart[c];
        const int injectionTaskCount =
            s.preInjectionHeavyCellTaskCount[c];
        for
        (
            int localTask = threadIdx.x;
            localTask < injectionTaskCount;
            localTask += blockDim.x
        )
        {
            const int task = injectionFirstTask + localTask;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] +=
                    s.preInjectionHeavyPartials
                    [
                        8u*static_cast<size_t>(task)
                      + static_cast<size_t>(component)
                    ];
            }
        }

        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.poissonPoolMass[c] = sums[0];
            s.poissonPoolMomX[c] = sums[1];
            s.poissonPoolMomY[c] = sums[2];
            s.poissonPoolMomZ[c] = sums[3];
            s.poissonPoolEnergy[c] = sums[4];
            s.poissonPoolDiameter[c] = sums[5];
            s.poissonPoolDiameter2[c] = sums[6];
            s.poolThermalCount[c] = static_cast<int>(sums[7]);
        }
        __syncthreads();
    }
}

__global__ void finalizeCsrHeavyPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int heavyCellIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            heavyCellIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (heavyCellIndex >= *s.csrHeavyCellCount)
        {
            return;
        }

        const int c = s.csrHeavyCellList[heavyCellIndex];
        double sums[8] =
        {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        };
        const int firstTask = s.csrHeavyCellTaskStart[c];
        const int nTasks = s.csrHeavyCellTaskCount[c];
        for
        (
            int localTask = threadIdx.x;
            localTask < nTasks;
            localTask += blockDim.x
        )
        {
            const int task = firstTask + localTask;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] +=
                    s.csrHeavyPartials
                    [
                        8u*static_cast<size_t>(task)
                      + static_cast<size_t>(component)
                    ];
            }
        }

        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.poissonPoolMass[c] = sums[0];
            s.poissonPoolMomX[c] = sums[1];
            s.poissonPoolMomY[c] = sums[2];
            s.poissonPoolMomZ[c] = sums[3];
            s.poissonPoolEnergy[c] = sums[4];
            s.poissonPoolDiameter[c] = sums[5];
            s.poissonPoolDiameter2[c] = sums[6];
            s.poolThermalCount[c] = static_cast<int>(sums[7]);
        }
        __syncthreads();
    }
}

template<bool PoissonMode>
__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
    __shared__ double collisionProbability;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            taskCount = *s.csrHeavyTaskCount;
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            if (task < taskCount)
            {
                descriptor = s.csrReductionTasks[task];
                if (PoissonMode)
                {
                    const double tauColl =
                        granularCollisionTauFromCellDevice(s, descriptor.cell);
                    collisionProbability =
                        (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
                      ? 0.0
                      : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
                }
                else
                {
                    collisionProbability = 1.0;
                }
            }
        }
        __syncthreads();
        if (task >= taskCount)
        {
            return;
        }
        const int c = descriptor.cell;
            if (PoissonMode && collisionProbability <= 0.0)
            {
                if (threadIdx.x == 0 && s.csrCellTaskCount[c] > 1)
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = 0.0;
                    }
                }
                continue;
            }
            double sums[8];
            const bool directParticleIndex =
                descriptor.source == static_cast<int>
                (
                    CsrReductionTaskSource::splitBaseDirect
                );
            accumulateCsrHeavyPoolTask<PoissonMode>
            (
                s, c, descriptor.begin, descriptor.end, directParticleIndex,
                collisionProbability, sums, warpPartials
            );
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.poissonPoolMass[c] = sums[0];
                    s.poissonPoolMomX[c] = sums[1];
                    s.poissonPoolMomY[c] = sums[2];
                    s.poissonPoolMomZ[c] = sums[3];
                    s.poissonPoolEnergy[c] = sums[4];
                    s.poissonPoolDiameter[c] = sums[5];
                    s.poissonPoolDiameter2[c] = sums[6];
                    s.poolThermalCount[c] = static_cast<int>(sums[7]);
                }
                else
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = sums[component];
                    }
                }
            }
    }
}

__global__ void finalizeCsrSegmentedPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int multiIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            multiIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (multiIndex >= *s.csrHeavyCellCount)
        {
            return;
        }
        const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1))
        {
            asm("trap;");
        }
        double sums[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = s.csrCellTaskOffset[c + 1];
        for (int task = firstTask + threadIdx.x; task < endTask; task += blockDim.x)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ];
            }
        }
        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.poissonPoolMass[c] = sums[0];
            s.poissonPoolMomX[c] = sums[1];
            s.poissonPoolMomY[c] = sums[2];
            s.poissonPoolMomZ[c] = sums[3];
            s.poissonPoolEnergy[c] = sums[4];
            s.poissonPoolDiameter[c] = sums[5];
            s.poissonPoolDiameter2[c] = sums[6];
            s.poolThermalCount[c] = static_cast<int>(sums[7]);
        }
        __syncthreads();
    }
}

int launchCsrSegmentedPoolReduction
(
    DeviceState* s,
    const double dt,
    const bool poissonMode,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    if (poissonMode)
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented pool worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells
      ? s->multiprocessorCount
      : s->nCells;
    finalizeCsrSegmentedPoolCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrSegmentedPoolCellsKernel launch", err);
        return 1;
    }
    return 0;
}

int launchCsrHeavyPoolReduction
(
    DeviceState* s,
    const double dt,
    const bool poissonMode,
    const int block,
    const int useSplitPreDirectory
)
{
    return launchCsrSegmentedPoolReduction(s, dt, poissonMode, block);
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }

    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR heavy pool task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    if (poissonMode && useSplitPreDirectory != 0)
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<true, true, false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else if (poissonMode)
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<true, false, false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else if (useSplitPreDirectory != 0)
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<false, true, false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<false, false, false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR heavy pool persistent kernel launch", err);
        return 1;
    }

    if (useSplitPreDirectory != 0)
    {
        if (s->preInjectionSegmentActive != 0)
        {
            err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "reset split-pre injection heavy pool task cursor",
                    err
                );
                return 1;
            }
            if (poissonMode)
            {
                accumulateCsrHeavyPoolTasksPersistentKernel<true, false, true>
                    <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>
                    (s->deviceState, dt);
            }
            else
            {
                accumulateCsrHeavyPoolTasksPersistentKernel<false, false, true>
                    <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>
                    (s->deviceState, dt);
            }
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "split-pre injection CSR heavy pool worker kernel launch",
                    err
                );
                return 1;
            }
        }
    }

    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR heavy pool finalize cursor", err);
        return 1;
    }
    const int finalizeGridFromSm = s->multiprocessorCount;
    const int finalizeGrid =
        finalizeGridFromSm < s->nCells ? finalizeGridFromSm : s->nCells;
    if (useSplitPreDirectory != 0)
    {
        finalizeSplitPreCsrHeavyPoolCellsKernel
            <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    }
    else
    {
        finalizeCsrHeavyPoolCellsKernel
            <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrHeavyPoolCellsKernel launch", err);
        return 1;
    }
    return 0;
#endif
}

__global__ void preparePoissonPoolSamplingKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const int representatives = s.poolThermalCount[c];
    const double poolMass =
        clampMin(finiteOr(s.poissonPoolMass[c], 0.0), 0.0);

    int target = 0;
    if (poolMass > 0.0 && representatives > 0)
    {
        target = representatives;
    }

    s.poissonPoolSampleTargetCount[c] = target;
}

__device__ void sampleOnePoissonPoolParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    if (i >= s.particleCapacity || s.pStatus[i] != 2)
    {
        return;
    }

    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    const int n = s.poissonPoolSampleTargetCount[c];
    const double poolMass = clampMin(s.poissonPoolMass[c], 0.0);
    if (n <= 0 || poolMass <= 0.0)
    {
        s.pStatus[i] = 0;
        return;
    }

    const double invM = 1.0/poolMass;
    const double targetUx = s.poissonPoolMomX[c]*invM;
    const double targetUy = s.poissonPoolMomY[c]*invM;
    const double targetUz = s.poissonPoolMomZ[c]*invM;
    const double targetKinetic = 0.5*sqr3(targetUx, targetUy, targetUz);
    const double targetThetaRaw =
        clampMin(s.poissonPoolEnergy[c]*invM - targetKinetic, 0.0)/1.5;

    double targetTheta = targetThetaRaw;

    if (applyThetaDrag)
    {
        const double alphaTheta =
            clampRange(finiteOr(s.thetaDragAlpha[c], 1.0), 0.0, 1.0);

        targetTheta *= alphaTheta;
    }

    const double sigma = sqrt(clampRange(targetTheta, 0.0, OfGreat));
    unsigned long long rng = s.pRng[i];

    double z0 = 0.0;
    double z1 = 0.0;
    double z2 = 0.0;
    double z3 = 0.0;

    normalPairDevice(rng, z0, z1);
    normalPairDevice(rng, z2, z3);
    (void) z3;

    const double ux = targetUx + sigma*z0;
    const double uy = targetUy + sigma*z1;
    const double uz = targetUz + sigma*z2;
    if
    (
        n > 1
     && (
            nonFiniteDevice(ux) || nonFiniteDevice(uy) || nonFiniteDevice(uz)
        )
    )
    {
        asm("trap;");
    }

    s.pux[i] = finiteOr(ux, targetUx);
    s.puy[i] = finiteOr(uy, targetUy);
    s.puz[i] = finiteOr(uz, targetUz);
    s.pTheta[i] = 0.0;
    const double dMin = clampMin(s.particleDiameterMin, 1.0e-12);
    const double dMax = clampMin(s.particleDiameterMax, dMin);

    s.pd[i] =
        clampRange
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            dMin,
            dMax
        );

    const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
    if (!(m > 0.0))
    {
        asm("trap;");
        return;
    }
    s.pRng[i] = rng;

                                                                         
                                                                        
                                                                     
    const double sampleDeltaX = s.pux[i] - targetUx;
    const double sampleDeltaY = s.puy[i] - targetUy;
    const double sampleDeltaZ = s.puz[i] - targetUz;
    atomicAdd(&s.poolThermalSumUx[c], m*sampleDeltaX);
    atomicAdd(&s.poolThermalSumUy[c], m*sampleDeltaY);
    atomicAdd(&s.poolThermalSumUz[c], m*sampleDeltaZ);
    atomicAdd
    (
        &s.poolThermalSumU2[c],
        m*sqr3(sampleDeltaX, sampleDeltaY, sampleDeltaZ)
    );
}

__global__ void samplePoissonPoolParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        sampleOnePoissonPoolParticle(s, i, applyThetaDrag != 0);
    }
}

__device__ void correctOnePoissonThermalizedParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    if (i >= s.particleCapacity || s.pStatus[i] != 2)
    {
        return;
    }

    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    const int n = s.poissonPoolSampleTargetCount[c];
    const double poolMass = clampMin(s.poissonPoolMass[c], 0.0);
    if (n <= 0 || poolMass <= 0.0)
    {
        return;
    }
    const double meanParticleMass =
        poolMass/static_cast<double>(n);

    const double targetUx = s.poissonPoolMomX[c]/poolMass;
    const double targetUy = s.poissonPoolMomY[c]/poolMass;
    const double targetUz = s.poissonPoolMomZ[c]/poolMass;
    const double targetMean2 = sqr3(targetUx, targetUy, targetUz);
    const double targetThetaRaw =
        clampMin(s.poissonPoolEnergy[c]/poolMass - 0.5*targetMean2, 0.0)/1.5;

    const double sampleMeanDeltaX = s.poolThermalSumUx[c]/poolMass;
    const double sampleMeanDeltaY = s.poolThermalSumUy[c]/poolMass;
    const double sampleMeanDeltaZ = s.poolThermalSumUz[c]/poolMass;
    const double sampleMeanDelta2 =
        sqr3(sampleMeanDeltaX, sampleMeanDeltaY, sampleMeanDeltaZ);
    const double thermal =
        0.5*clampMin
        (
            s.poolThermalSumU2[c] - poolMass*sampleMeanDelta2,
            0.0
        );

    double targetTheta = targetThetaRaw;

    if (applyThetaDrag)
    {
        const double alphaTheta =
            clampRange(finiteOr(s.thetaDragAlpha[c], 1.0), 0.0, 1.0);

        targetTheta *= alphaTheta;
    }

    const double targetThermal = 1.5*poolMass*targetTheta;

    if (targetThermal <= 0.0)
    {
        s.pux[i] = targetUx;
        s.puy[i] = targetUy;
        s.puz[i] = targetUz;
        s.pTheta[i] = 0.0;
        s.pStatus[i] = 1;
        return;
    }

    if (n <= 1 || thermal <= OfSmall*meanParticleMass)
    {
                                                                            
                                                                       
                                                                            
        s.pux[i] = targetUx;
        s.puy[i] = targetUy;
        s.puz[i] = targetUz;
        s.pTheta[i] = targetTheta;
        s.pStatus[i] = 1;
        return;
    }

    double scale = 1.0;
    scale = sqrt(targetThermal/thermal);
    const double correctedUx =
        targetUx + scale*((s.pux[i] - targetUx) - sampleMeanDeltaX);
    const double correctedUy =
        targetUy + scale*((s.puy[i] - targetUy) - sampleMeanDeltaY);
    const double correctedUz =
        targetUz + scale*((s.puz[i] - targetUz) - sampleMeanDeltaZ);
    if
    (
        nonFiniteDevice(correctedUx) || nonFiniteDevice(correctedUy)
     || nonFiniteDevice(correctedUz)
    )
    {
        asm("trap;");
    }
    s.pux[i] = correctedUx;
    s.puy[i] = correctedUy;
    s.puz[i] = correctedUz;
    s.pTheta[i] = 0.0;
    s.pStatus[i] = 1;
}

__global__ void correctPoissonThermalizedParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        correctOnePoissonThermalizedParticle(s, i, applyThetaDrag != 0);
    }
}

template<int BlockThreads>
__global__ void gatherCellLocalParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells || blockDim.x != BlockThreads)
    {
        return;
    }

    using BlockScan = cub::BlockScan<int, BlockThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    const int outputStart = s.compactCellOffset[c];
    int tileOutputOffset = 0;

    for
    (
        int tileStart = start;
        tileStart < end;
        tileStart += BlockThreads
    )
    {
        const int pos = tileStart + threadIdx.x;
        const int i = pos < end ? s.sortedParticleIndex[pos] : -1;
        const int keep =
            i >= 0
         && i < s.particleCapacity
         && s.pStatus[i] == 1
         && s.pCellId[i] == c;

        int localOffset = 0;
        int tileCount = 0;
        BlockScan(scanStorage).ExclusiveSum(keep, localOffset, tileCount);

        if (keep != 0)
        {
            const int dst = outputStart + tileOutputOffset + localOffset;
            s.compactPx[dst] = s.px[i];
            s.compactPy[dst] = s.py[i];
            s.compactPz[dst] = s.pz[i];
            s.compactPux[dst] = s.pux[i];
            s.compactPuy[dst] = s.puy[i];
            s.compactPuz[dst] = s.puz[i];
            s.compactPT[dst] = s.pT[i];
            s.compactPTheta[dst] = s.pTheta[i];
            s.compactPd[dst] = s.pd[i];
            s.compactPm[dst] = s.pm[i];
            s.compactPCellId[dst] = c;
            s.compactPStatus[dst] = 1;
            s.compactPRng[dst] = s.pRng[i];
            s.compactPOrigId[dst] = s.pOrigId[i];
        }

        tileOutputOffset += tileCount;
        __syncthreads();
    }
}

int launchGatherCellLocalParticles(DeviceState* s)
{
    switch (s->reductionBlockThreads)
    {
        case 32:
            gatherCellLocalParticlesKernel<32>
                <<<s->nCells, 32>>>(s->deviceState);
            break;
        case 64:
            gatherCellLocalParticlesKernel<64>
                <<<s->nCells, 64>>>(s->deviceState);
            break;
        case 128:
            gatherCellLocalParticlesKernel<128>
                <<<s->nCells, 128>>>(s->deviceState);
            break;
        case 256:
            gatherCellLocalParticlesKernel<256>
                <<<s->nCells, 256>>>(s->deviceState);
            break;
        default:
            setLastErrorText("unsupported UGKP block size in cell gather");
            return 1;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("gatherCellLocalParticlesKernel launch", err);
        return 1;
    }
    return 0;
}

__global__ void gatherSelectedParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int selectedCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    for
    (
        int dst = blockIdx.x*blockDim.x + threadIdx.x;
        dst < selectedCount;
        dst += blockDim.x*gridDim.x
    )
    {
        const int i = s.sortedParticleIndex[dst];
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
        )
        {
            asm("trap;");
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            asm("trap;");
        }

        s.compactPx[dst] = s.px[i];
        s.compactPy[dst] = s.py[i];
        s.compactPz[dst] = s.pz[i];
        s.compactPux[dst] = s.pux[i];
        s.compactPuy[dst] = s.puy[i];
        s.compactPuz[dst] = s.puz[i];
        s.compactPT[dst] = s.pT[i];
        s.compactPTheta[dst] = s.pTheta[i];
        s.compactPd[dst] = s.pd[i];
        s.compactPm[dst] = s.pm[i];
        s.compactPCellId[dst] = c;
        s.compactPStatus[dst] = 1;
        s.compactPRng[dst] = s.pRng[i];
        s.compactPOrigId[dst] = s.pOrigId[i];
    }
}

template<class T>
__device__ void swapParticlePointerDevice(T*& active, T*& scratch)
{
    T* const oldActive = active;
    active = scratch;
    scratch = oldActive;
}

__global__ void commitCellLocalParticleBuffersKernel(DeviceState* sp)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    int compactCount = s.compactCellOffset[s.nCells];
    compactCount = compactCount < 0 ? 0 : compactCount;
    compactCount =
        compactCount > s.particleCapacity ? s.particleCapacity : compactCount;
    *s.particleCountDevice = compactCount;

    swapParticlePointerDevice(s.px, s.compactPx);
    swapParticlePointerDevice(s.py, s.compactPy);
    swapParticlePointerDevice(s.pz, s.compactPz);
    swapParticlePointerDevice(s.pux, s.compactPux);
    swapParticlePointerDevice(s.puy, s.compactPuy);
    swapParticlePointerDevice(s.puz, s.compactPuz);
    swapParticlePointerDevice(s.pT, s.compactPT);
    swapParticlePointerDevice(s.pTheta, s.compactPTheta);
    swapParticlePointerDevice(s.pd, s.compactPd);
    swapParticlePointerDevice(s.pm, s.compactPm);
    swapParticlePointerDevice(s.pCellId, s.compactPCellId);
    swapParticlePointerDevice(s.pStatus, s.compactPStatus);
    swapParticlePointerDevice(s.pRng, s.compactPRng);
    swapParticlePointerDevice(s.pOrigId, s.compactPOrigId);
}

__global__ void commitSelectedParticleBuffersKernel(DeviceState* sp)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    const int compactCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    *s.particleCountDevice = compactCount;

    swapParticlePointerDevice(s.px, s.compactPx);
    swapParticlePointerDevice(s.py, s.compactPy);
    swapParticlePointerDevice(s.pz, s.compactPz);
    swapParticlePointerDevice(s.pux, s.compactPux);
    swapParticlePointerDevice(s.puy, s.compactPuy);
    swapParticlePointerDevice(s.puz, s.compactPuz);
    swapParticlePointerDevice(s.pT, s.compactPT);
    swapParticlePointerDevice(s.pTheta, s.compactPTheta);
    swapParticlePointerDevice(s.pd, s.compactPd);
    swapParticlePointerDevice(s.pm, s.compactPm);
    swapParticlePointerDevice(s.pCellId, s.compactPCellId);
    swapParticlePointerDevice(s.pStatus, s.compactPStatus);
    swapParticlePointerDevice(s.pRng, s.compactPRng);
    swapParticlePointerDevice(s.pOrigId, s.compactPOrigId);
}

template<class T>
void swapParticlePointerHost(T*& active, T*& scratch)
{
    T* const oldActive = active;
    active = scratch;
    scratch = oldActive;
}

void swapParticleBufferPointersHost(DeviceState* s)
{
    swapParticlePointerHost(s->px, s->compactPx);
    swapParticlePointerHost(s->py, s->compactPy);
    swapParticlePointerHost(s->pz, s->compactPz);
    swapParticlePointerHost(s->pux, s->compactPux);
    swapParticlePointerHost(s->puy, s->compactPuy);
    swapParticlePointerHost(s->puz, s->compactPuz);
    swapParticlePointerHost(s->pT, s->compactPT);
    swapParticlePointerHost(s->pTheta, s->compactPTheta);
    swapParticlePointerHost(s->pd, s->compactPd);
    swapParticlePointerHost(s->pm, s->compactPm);
    swapParticlePointerHost(s->pCellId, s->compactPCellId);
    swapParticlePointerHost(s->pStatus, s->compactPStatus);
    swapParticlePointerHost(s->pRng, s->compactPRng);
    swapParticlePointerHost(s->pOrigId, s->compactPOrigId);
}

__global__ void clearParticleMomentsAndCountsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c > s.nCells)
    {
        return;
    }
    s.cellParticleCount[c] = 0;
    if (c == s.nCells)
    {
        return;
    }
    s.momRhoP[c] = 0.0;
    s.momRhoUPx[c] = 0.0;
    s.momRhoUPy[c] = 0.0;
    s.momRhoUPz[c] = 0.0;
    s.momRhoEP[c] = 0.0;
    s.momRhoPD[c] = 0.0;
    s.momRhoHpP[c] = 0.0;
}

__global__ void accumulateParticleMomentsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double ux = finiteOr(s.pux[i], 0.0);
        const double uy = finiteOr(s.puy[i], 0.0);
        const double uz = finiteOr(s.puz[i], 0.0);
        const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
        const double d =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        const double tp =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        atomicAdd(&s.momRhoP[c], m);
        atomicAdd(&s.momRhoUPx[c], m*ux);
        atomicAdd(&s.momRhoUPy[c], m*uy);
        atomicAdd(&s.momRhoUPz[c], m*uz);
        atomicAdd
        (
            &s.momRhoEP[c],
            m*(0.5*sqr3(ux, uy, uz) + 1.5*theta)
        );
        atomicAdd(&s.momRhoPD[c], m*d);
        atomicAdd(&s.momRhoHpP[c], m*particleHeatFactorDevice(s)*tp);
        atomicAdd(&s.cellParticleCount[c], 1);
    }
}

__global__ void normalizeParticleMomentsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
    s.momRhoP[c] *= invV;
    s.momRhoUPx[c] *= invV;
    s.momRhoUPy[c] *= invV;
    s.momRhoUPz[c] *= invV;
    s.momRhoEP[c] *= invV;
    s.momRhoPD[c] *= invV;
    s.momRhoHpP[c] *= invV;
}

template<bool HeavyReductionEnabled>
__global__ void accumulateParticleMomentsSegmentedKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    if constexpr (HeavyReductionEnabled)
    {
        return;
    }

    double rho = 0.0;
    double momX = 0.0;
    double momY = 0.0;
    double momZ = 0.0;
    double energy = 0.0;
    double diameter = 0.0;
    double heat = 0.0;
    int survivorCount = 0;

    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
         || s.pCellId[i] != c
        )
        {
            continue;
        }

        const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double ux = finiteOr(s.pux[i], 0.0);
        const double uy = finiteOr(s.puy[i], 0.0);
        const double uz = finiteOr(s.puz[i], 0.0);
        const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += m*(0.5*sqr3(ux, uy, uz) + 1.5*theta);
        diameter += m*clampMin(finiteOr(s.pd[i], s.particleDiameterFallback), 1.0e-12);
        heat += m*particleHeatFactorDevice(s)*clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        ++survivorCount;
    }

    double sums[8] =
    {
        rho,
        momX,
        momY,
        momZ,
        energy,
        diameter,
        heat,
        static_cast<double>(survivorCount)
    };
    extern __shared__ double warpPartials[];
    blockReduceComponentSums<8>(sums, warpPartials);

    if (threadIdx.x == 0)
    {
        s.cellParticleCount[c] = static_cast<int>(sums[7]);
        if (c == 0)
        {
            s.cellParticleCount[s.nCells] = 0;
        }
        const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
        s.momRhoP[c] = sums[0]*invV;
        s.momRhoUPx[c] = sums[1]*invV;
        s.momRhoUPy[c] = sums[2]*invV;
        s.momRhoUPz[c] = sums[3]*invV;
        s.momRhoEP[c] = sums[4]*invV;
        s.momRhoPD[c] = sums[5]*invV;
        s.momRhoHpP[c] = sums[6]*invV;
    }
}

__device__ void accumulateCsrHeavyMomentTask
(
    DeviceState& s,
    const int c,
    const int begin,
    const int end,
    double (&sums)[8],
    double* warpPartials
)
{
    double rho = 0.0;
    double momX = 0.0;
    double momY = 0.0;
    double momZ = 0.0;
    double energy = 0.0;
    double diameter = 0.0;
    double heat = 0.0;
    double count = 0.0;

    for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
         || s.pCellId[i] != c
        )
        {
            continue;
        }

        const double m = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double ux = finiteOr(s.pux[i], 0.0);
        const double uy = finiteOr(s.puy[i], 0.0);
        const double uz = finiteOr(s.puz[i], 0.0);
        const double theta = clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += m*(0.5*sqr3(ux, uy, uz) + 1.5*theta);
        diameter +=
            m*clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        heat +=
            m*particleHeatFactorDevice(s)
             *clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        count += 1.0;
    }

    sums[0] = rho;
    sums[1] = momX;
    sums[2] = momY;
    sums[3] = momZ;
    sums[4] = energy;
    sums[5] = diameter;
    sums[6] = heat;
    sums[7] = count;
    blockReduceComponentSums<8>(sums, warpPartials);
}

__global__ void accumulateCsrHeavyMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int task;
    extern __shared__ double warpPartials[];

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (task >= *s.csrHeavyTaskCount)
        {
            return;
        }

        const int c = s.csrHeavyTaskCell[task];
        double sums[8];
        accumulateCsrHeavyMomentTask
        (
            s,
            c,
            s.csrHeavyTaskBegin[task],
            s.csrHeavyTaskEnd[task],
            sums,
            warpPartials
        );
        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ] = sums[component];
            }
        }
        __syncthreads();
    }
}

__global__ void finalizeCsrHeavyMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int heavyCellIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            heavyCellIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (heavyCellIndex >= *s.csrHeavyCellCount)
        {
            return;
        }

        const int c = s.csrHeavyCellList[heavyCellIndex];
        double sums[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int firstTask = s.csrHeavyCellTaskStart[c];
        const int nTasks = s.csrHeavyCellTaskCount[c];
        for
        (
            int localTask = threadIdx.x;
            localTask < nTasks;
            localTask += blockDim.x
        )
        {
            const int task = firstTask + localTask;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] +=
                    s.csrHeavyPartials
                    [
                        8u*static_cast<size_t>(task)
                      + static_cast<size_t>(component)
                    ];
            }
        }

        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.cellParticleCount[c] = static_cast<int>(sums[7]);
            if (c == 0)
            {
                s.cellParticleCount[s.nCells] = 0;
            }
            const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
            s.momRhoP[c] = sums[0]*invV;
            s.momRhoUPx[c] = sums[1]*invV;
            s.momRhoUPy[c] = sums[2]*invV;
            s.momRhoUPz[c] = sums[3]*invV;
            s.momRhoEP[c] = sums[4]*invV;
            s.momRhoPD[c] = sums[5]*invV;
            s.momRhoHpP[c] = sums[6]*invV;
        }
        __syncthreads();
    }
}

__global__ void accumulateCsrSegmentedMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            taskCount = *s.csrHeavyTaskCount;
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            if (task < taskCount)
            {
                descriptor = s.csrReductionTasks[task];
            }
        }
        __syncthreads();
        if (task >= taskCount)
        {
            return;
        }
        const int c = descriptor.cell;
            double sums[8];
            accumulateCsrHeavyMomentTask
            (
                s, c, descriptor.begin, descriptor.end, sums, warpPartials
            );
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.cellParticleCount[c] = static_cast<int>(sums[7]);
                    if (c == 0)
                    {
                        s.cellParticleCount[s.nCells] = 0;
                    }
                    const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
                    s.momRhoP[c] = sums[0]*invV;
                    s.momRhoUPx[c] = sums[1]*invV;
                    s.momRhoUPy[c] = sums[2]*invV;
                    s.momRhoUPz[c] = sums[3]*invV;
                    s.momRhoEP[c] = sums[4]*invV;
                    s.momRhoPD[c] = sums[5]*invV;
                    s.momRhoHpP[c] = sums[6]*invV;
                }
                else
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = sums[component];
                    }
                }
            }
    }
}

__global__ void finalizeCsrSegmentedMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int multiIndex;
    extern __shared__ double warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            multiIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        }
        __syncthreads();
        if (multiIndex >= *s.csrHeavyCellCount)
        {
            return;
        }
        const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1))
        {
            asm("trap;");
        }
        double sums[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = s.csrCellTaskOffset[c + 1];
        for (int task = firstTask + threadIdx.x; task < endTask; task += blockDim.x)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ];
            }
        }
        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.cellParticleCount[c] = static_cast<int>(sums[7]);
            if (c == 0)
            {
                s.cellParticleCount[s.nCells] = 0;
            }
            const double invV = 1.0/clampMin(s.V[c], s.rhoMin);
            s.momRhoP[c] = sums[0]*invV;
            s.momRhoUPx[c] = sums[1]*invV;
            s.momRhoUPy[c] = sums[2]*invV;
            s.momRhoUPz[c] = sums[3]*invV;
            s.momRhoEP[c] = sums[4]*invV;
            s.momRhoPD[c] = sums[5]*invV;
            s.momRhoHpP[c] = sums[6]*invV;
        }
        __syncthreads();
    }
}

int launchCsrSegmentedMomentReduction(DeviceState* s, const int block)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    accumulateCsrSegmentedMomentTasksPersistentKernel
        <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented moment worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells
      ? s->multiprocessorCount
      : s->nCells;
    finalizeCsrSegmentedMomentCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrSegmentedMomentCellsKernel launch", err);
        return 1;
    }
    return 0;
}

int launchCsrHeavyMomentReduction(DeviceState* s, const int block)
{
    return launchCsrSegmentedMomentReduction(s, block);
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }

    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR heavy moment task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    accumulateCsrHeavyMomentTasksPersistentKernel
        <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR heavy moment persistent kernel launch", err);
        return 1;
    }

    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR heavy moment finalize cursor", err);
        return 1;
    }
    const int finalizeGridFromSm = s->multiprocessorCount;
    const int finalizeGrid =
        finalizeGridFromSm < s->nCells ? finalizeGridFromSm : s->nCells;
    finalizeCsrHeavyMomentCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrHeavyMomentCellsKernel launch", err);
        return 1;
    }
    return 0;
#endif
}

__global__ void solidRecoveryFromParticleMomentsKernel(DeviceState* sp);

__global__ void clearParticleCellBinsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c <= s.nCells; c += stride)
    {
        s.cellParticleCount[c] = 0;
    }
}

template<bool WarpAggregated>
__global__ void countParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        const int c = s.pCellId[i];
        const bool valid = s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                atomicAdd(&s.cellParticleCount[c], 1);
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            if (lane == leader)
            {
                atomicAdd(&s.cellParticleCount[c], __popc(groupMask));
            }
        }
#else
        if (valid)
        {
            atomicAdd(&s.cellParticleCount[c], 1);
        }
#endif
    }
}

__global__ void initialiseParticleCellWritesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.cellParticleWrite[c] = s.cellParticleOffset[c];
    }
}

template<bool WarpAggregated>
__global__ void scatterParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        const int c = s.pCellId[i];
        const bool valid = s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
                if
                (
                    pos >= s.cellParticleOffset[c]
                 && pos < s.cellParticleOffset[c + 1]
                )
                {
                    s.sortedParticleIndex[pos] = i;
                }
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            int groupBase = 0;
            if (lane == leader)
            {
                groupBase = atomicAdd(&s.cellParticleWrite[c], __popc(groupMask));
            }
            groupBase = __shfl_sync(groupMask, groupBase, leader);
            const unsigned int lowerLaneMask = lane == 0 ? 0u : ((1u << lane) - 1u);
            const int laneRank = __popc(groupMask & lowerLaneMask);
            const int pos = groupBase + laneRank;
            if
            (
                pos >= s.cellParticleOffset[c]
             && pos < s.cellParticleOffset[c + 1]
            )
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#else
        if (valid)
        {
            const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
            if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#endif
    }
}

                                                                         
                                                                           
template<bool WarpAggregated>
__global__ void countInjectedParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int baseCount = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int nParticles = clampRange
    (
        *s.particleCountDevice,
        baseCount,
        s.particleCapacity
    );
    for
    (
        int i = baseCount + blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        const int c = s.pCellId[i];
        const bool valid =
            s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                atomicAdd(&s.cellParticleCount[c], 1);
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            if (lane == leader)
            {
                atomicAdd(&s.cellParticleCount[c], __popc(groupMask));
            }
        }
#else
        if (valid)
        {
            atomicAdd(&s.cellParticleCount[c], 1);
        }
#endif
    }
}

template<bool WarpAggregated>
__global__ void scatterInjectedParticlesByCellKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int baseCount = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int nParticles = clampRange
    (
        *s.particleCountDevice,
        baseCount,
        s.particleCapacity
    );
    for
    (
        int i = baseCount + blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        const int c = s.pCellId[i];
        const bool valid =
            s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
                if
                (
                    pos >= s.cellParticleOffset[c]
                 && pos < s.cellParticleOffset[c + 1]
                )
                {
                    s.sortedParticleIndex[pos] = i;
                }
            }
            continue;
        }
#if __CUDA_ARCH__ >= 700
        const unsigned int activeMask = __activemask();
        const unsigned int validMask = __ballot_sync(activeMask, valid);
        if (valid)
        {
            const unsigned int groupMask = __match_any_sync(validMask, c);
            const int lane = threadIdx.x & 31;
            const int leader = __ffs(groupMask) - 1;
            int groupBase = 0;
            if (lane == leader)
            {
                groupBase = atomicAdd
                (
                    &s.cellParticleWrite[c],
                    __popc(groupMask)
                );
            }
            groupBase = __shfl_sync(groupMask, groupBase, leader);
            const unsigned int lowerLaneMask =
                lane == 0 ? 0u : ((1u << lane) - 1u);
            const int laneRank = __popc(groupMask & lowerLaneMask);
            const int pos = groupBase + laneRank;
            if
            (
                pos >= s.cellParticleOffset[c]
             && pos < s.cellParticleOffset[c + 1]
            )
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#else
        if (valid)
        {
            const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
            if
            (
                pos >= s.cellParticleOffset[c]
             && pos < s.cellParticleOffset[c + 1]
            )
            {
                s.sortedParticleIndex[pos] = i;
            }
        }
#endif
    }
}

__global__ void captureCompactedPreBaseOffsetsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for
    (
        int c = blockIdx.x*blockDim.x + threadIdx.x;
        c <= s.nCells;
        c += stride
    )
    {
        s.preBaseCellOffset[c] = s.compactCellOffset[c];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int compactCount = clampRange
        (
            s.compactCellOffset[s.nCells],
            0,
            s.particleCapacity
        );
        *s.preBaseParticleCountDevice = compactCount;
        s.preBaseDirectoryReady = 1;
    }
}

enum class HeavyDirectoryKind : int
{
    full = 0,
    splitBaseAndInjection = 1,
    baseOnly = 2
};

__global__ void configureDynamicCsrHeavyPolicyKernel
(
    DeviceState* sp,
    const int directoryKind
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    long long totalParticles = 0;
    if (directoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
    {
        totalParticles = s.preBaseCellOffset[s.nCells];
    }
    else if
    (
        directoryKind
     == static_cast<int>(HeavyDirectoryKind::splitBaseAndInjection)
    )
    {
        totalParticles =
            static_cast<long long>(s.preBaseCellOffset[s.nCells])
          + static_cast<long long>(s.cellParticleOffset[s.nCells]);
    }
    else
    {
        totalParticles = s.cellParticleOffset[s.nCells];
    }

    if (totalParticles < 0)
    {
        asm("trap;");
    }
    const long long denominator =
        static_cast<long long>(s.reductionBlockThreads)
       *static_cast<long long>(s.multiprocessorCount)
       *static_cast<long long>(s.lightBlocksPerSm);
    if (denominator <= 0)
    {
        asm("trap;");
    }
    long long rounds = (totalParticles + denominator - 1)/denominator;
    if (rounds < 1)
    {
        rounds = 1;
    }
    const long long threshold =
        static_cast<long long>(s.reductionBlockThreads)*rounds;
    if (threshold <= 0 || threshold > 2147483647LL)
    {
        asm("trap;");
    }
    s.csrHeavyCellThreshold = static_cast<int>(threshold);
    s.csrHeavyTileParticles = static_cast<int>(threshold);
}

int configureDynamicCsrHeavyPolicy
(
    DeviceState* s,
    const HeavyDirectoryKind directoryKind
)
{
    if (s->csrHeavyReductionMode == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    configureDynamicCsrHeavyPolicyKernel<<<1, 1>>>
    (
        s->deviceState,
        static_cast<int>(directoryKind)
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("configure dynamic CSR heavy policy launch", err);
        return 1;
    }
    return 0;
}

__global__ void maximumDirectoryOccupancyKernel
(
    DeviceState* sp,
    const int directoryKind,
    int* maximumOccupancy
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        int count = 0;
        if (directoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
        {
            count = s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
        }
        else if
        (
            directoryKind
         == static_cast<int>(HeavyDirectoryKind::splitBaseAndInjection)
        )
        {
            count =
                s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c]
              + s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        }
        else
        {
            count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        }
        atomicMax(maximumOccupancy, count);
    }
}

int runToolB3
(
    DeviceState* s,
    const int block,
    const HeavyDirectoryKind directoryKind
)
{
    if (s->csrHeavyReductionMode != 2 || s->particleCapacity <= 0)
    {
        return 0;
    }
    ++s->schedulingAdvanceCount;
    if
    (
        s->schedulingAdvanceCount != 1u
     && s->schedulingAdvanceCount
          % static_cast<unsigned long long>(s->csrHeavyAutoInterval) != 0u
    )
    {
        return 0;
    }
    if (configureDynamicCsrHeavyPolicy(s, directoryKind) != 0)
    {
        return 1;
    }
    cudaError_t err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 clear maximum occupancy", err);
        return 1;
    }
    const int grid = (s->nCells + block - 1)/block;
    maximumDirectoryOccupancyKernel<<<grid, block>>>
    (
        s->deviceState,
        static_cast<int>(directoryKind),
        s->csrHeavyCellCount
    );
    err = cudaGetLastError();
    int maximumOccupancy = 0;
    int threshold = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy
        (
            &maximumOccupancy,
            s->csrHeavyCellCount,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    }
    if (err == cudaSuccess)
    {
        err = cudaMemcpy
        (
            &threshold,
            reinterpret_cast<const unsigned char*>(s->deviceState)
              + offsetof(DeviceState, csrHeavyCellThreshold),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 occupancy decision", err);
        return 1;
    }
    const int active = maximumOccupancy > threshold ? 1 : 0;
    s->csrHeavyReductionActive = active;
    s->csrHeavyReductionEnabled = active;
    return syncDeviceState(s, "ToolB3 publish automatic L2 decision");
}

__global__ void countCsrReductionTasksKernel
(
    DeviceState* sp,
    const int directoryKind
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c > s.nCells)
    {
        return;
    }
    if (c == s.nCells)
    {
        s.csrCellTaskCount[c] = 0;
        return;
    }
    const int tile = s.csrHeavyTileParticles;
    if (tile <= 0)
    {
        asm("trap;");
    }
    if (directoryKind == static_cast<int>(HeavyDirectoryKind::full))
    {
        const int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        s.csrCellTaskCount[c] = count > 0 ? 1 + (count - 1)/tile : 0;
        return;
    }
    const int baseCount = s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
    const int baseTasks = baseCount > 0 ? 1 + (baseCount - 1)/tile : 0;
    if (directoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
    {
        s.csrCellTaskCount[c] = baseTasks;
        return;
    }
    const int injectionCount =
        s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
    const int injectionTasks =
        injectionCount > 0 ? 1 + (injectionCount - 1)/tile : 0;
    s.csrCellTaskCount[c] = baseTasks + injectionTasks;
}

__device__ void writeCsrReductionTask
(
    DeviceState& s,
    const int task,
    const int cell,
    const int begin,
    const int end,
    const CsrReductionTaskSource source
)
{
    if (task < 0 || task >= s.csrHeavyTaskCapacity || begin >= end)
    {
        asm("trap;");
    }
    const int sourceValue = static_cast<int>(source);
    CsrReductionTask descriptor;
    descriptor.cell = cell;
    descriptor.begin = begin;
    descriptor.end = end;
    descriptor.source = sourceValue;
    s.csrReductionTasks[task] = descriptor;
}

__global__ void materializeCsrReductionTasksKernel
(
    DeviceState* sp,
    const int directoryKind
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int taskStart = s.csrCellTaskOffset[c];
        const int nTasks = s.csrCellTaskCount[c];
        if (nTasks == 0)
        {
            continue;
        }
        if (taskStart < 0 || taskStart > s.csrHeavyTaskCapacity - nTasks)
        {
            asm("trap;");
        }
        if (nTasks > 1)
        {
            const int multiIndex = atomicAdd(s.csrHeavyCellCount, 1);
            if (multiIndex < 0 || multiIndex >= s.nCells)
            {
                asm("trap;");
            }
            s.csrMultiTaskCellList[multiIndex] = c;
        }

        const int tile = s.csrHeavyTileParticles;
        int localTask = 0;
        if (directoryKind == static_cast<int>(HeavyDirectoryKind::full))
        {
            const int begin = s.cellParticleOffset[c];
            const int end = s.cellParticleOffset[c + 1];
            for (; localTask < nTasks; ++localTask)
            {
                const int taskBegin = begin + localTask*tile;
                writeCsrReductionTask
                (
                    s, taskStart + localTask, c, taskBegin,
                    min(end, taskBegin + tile),
                    CsrReductionTaskSource::fullIndexed
                );
            }
            continue;
        }
        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        for (int taskBegin = baseBegin; taskBegin < baseEnd; taskBegin += tile)
        {
            writeCsrReductionTask
            (
                s, taskStart + localTask++, c, taskBegin,
                min(baseEnd, taskBegin + tile),
                CsrReductionTaskSource::splitBaseDirect
            );
        }
        if (directoryKind == static_cast<int>(HeavyDirectoryKind::baseOnly))
        {
            continue;
        }
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        for
        (
            int taskBegin = injectionBegin;
            taskBegin < injectionEnd;
            taskBegin += tile
        )
        {
            writeCsrReductionTask
            (
                s, taskStart + localTask++, c, taskBegin,
                min(injectionEnd, taskBegin + tile),
                CsrReductionTaskSource::splitInjectionIndexed
            );
        }
        if (localTask != nTasks)
        {
            asm("trap;");
        }
    }
}

__global__ void publishCsrReductionTaskCountKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    *s.csrHeavyTaskCount = s.csrCellTaskOffset[s.nCells];
}

int prepareCsrSegmentedReductionTasks
(
    DeviceState* s,
    const int block,
    const HeavyDirectoryKind directoryKind
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    if (configureDynamicCsrHeavyPolicy(s, directoryKind) != 0)
    {
        return 1;
    }
    cudaError_t err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("clear CSR multi-task cell count", err);
        return 1;
    }
    const int countGrid = (s->nCells + 1 + block - 1)/block;
    countCsrReductionTasksKernel<<<countGrid, block>>>
    (
        s->deviceState,
        static_cast<int>(directoryKind)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("countCsrReductionTasksKernel launch", err);
        return 1;
    }
    err = cub::DeviceScan::ExclusiveSum
    (
        s->cellScanTempStorage,
        s->cellScanTempBytes,
        s->csrCellTaskCount,
        s->csrCellTaskOffset,
        s->nCells + 1
    );
    if (err != cudaSuccess)
    {
        setLastError("CSR cell task count exclusive scan", err);
        return 1;
    }
    publishCsrReductionTaskCountKernel<<<1, 1>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("publish CSR segmented task count", err);
        return 1;
    }
    const int cellGrid = (s->nCells + block - 1)/block;
    materializeCsrReductionTasksKernel<<<cellGrid, block>>>
    (
        s->deviceState,
        static_cast<int>(directoryKind)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("materializeCsrReductionTasksKernel launch", err);
        return 1;
    }
    return 0;
}

__global__ void buildCsrHeavyReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int begin = s.cellParticleOffset[c];
        const int end = s.cellParticleOffset[c + 1];
        const int count = end - begin;
        if (count <= s.csrHeavyCellThreshold)
        {
            continue;
        }

        const int nTasks =
            1 + (count - 1)/s.csrHeavyTileParticles;
        const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
        if
        (
            taskStart < 0
         || nTasks <= 0
         || taskStart > s.csrHeavyTaskCapacity - nTasks
        )
        {
            asm("trap;");
        }

        s.csrHeavyCellTaskStart[c] = taskStart;
        s.csrHeavyCellTaskCount[c] = nTasks;
        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;
        for (int localTask = 0; localTask < nTasks; ++localTask)
        {
            const int task = taskStart + localTask;
            const int taskBegin = begin + localTask*s.csrHeavyTileParticles;
            const int remaining = end - taskBegin;
            const int taskLength =
                remaining < s.csrHeavyTileParticles
              ? remaining
              : s.csrHeavyTileParticles;
            const int taskEnd = taskBegin + taskLength;
            s.csrHeavyTaskCell[task] = c;
            s.csrHeavyTaskBegin[task] = taskBegin;
            s.csrHeavyTaskEnd[task] = taskEnd;
        }
    }
}

int prepareCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::full
    );
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }

    if (configureDynamicCsrHeavyPolicy(s, HeavyDirectoryKind::full) != 0)
    {
        return 1;
    }

    cudaError_t err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("clear CSR heavy task count", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("clear CSR heavy cell count", err);
        return 1;
    }

    const int cellGrid = (s->nCells + block - 1)/block;
    buildCsrHeavyReductionTasksKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("buildCsrHeavyReductionTasksKernel launch", err);
        return 1;
    }
    return 0;
#endif
}

__global__ void buildCsrHeavyBaseReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int begin = s.preBaseCellOffset[c];
        const int end = s.preBaseCellOffset[c + 1];
        const int count = end - begin;
        if (count <= s.csrHeavyCellThreshold)
        {
            continue;
        }

        const int nTasks = 1 + (count - 1)/s.csrHeavyTileParticles;
        const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
        if
        (
            taskStart < 0
         || nTasks <= 0
         || taskStart > s.csrHeavyTaskCapacity - nTasks
        )
        {
            asm("trap;");
        }

        s.csrHeavyCellTaskStart[c] = taskStart;
        s.csrHeavyCellTaskCount[c] = nTasks;
        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;
        for (int localTask = 0; localTask < nTasks; ++localTask)
        {
            const int task = taskStart + localTask;
            const int taskBegin = begin + localTask*s.csrHeavyTileParticles;
            const int remaining = end - taskBegin;
            const int taskLength = remaining < s.csrHeavyTileParticles
              ? remaining
              : s.csrHeavyTileParticles;
            s.csrHeavyTaskCell[task] = c;
            s.csrHeavyTaskBegin[task] = taskBegin;
            s.csrHeavyTaskEnd[task] = taskBegin + taskLength;
        }
    }
}

int prepareCsrHeavyBaseReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::baseOnly
    );
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }

    if (configureDynamicCsrHeavyPolicy(s, HeavyDirectoryKind::baseOnly) != 0)
    {
        return 1;
    }

    cudaError_t err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->csrHeavyCellTaskStart,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->csrHeavyCellTaskCount,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("reset split-pre base heavy task metadata", err);
        return 1;
    }

    const int cellGrid = (s->nCells + block - 1)/block;
    buildCsrHeavyBaseReductionTasksKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("build split-pre base heavy tasks launch", err);
        return 1;
    }
    return 0;
#endif
}

__global__ void buildSplitPreCsrHeavyReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        const int baseCount = baseEnd - baseBegin;
        const int injectionCount = injectionEnd - injectionBegin;
        if (baseCount + injectionCount <= s.csrHeavyCellThreshold)
        {
            continue;
        }

        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;

                                                                             
                                                                         
                                                                         
        if (baseCount > 0 && s.csrHeavyCellTaskCount[c] == 0)
        {
            const int nTasks =
                1 + (baseCount - 1)/s.csrHeavyTileParticles;
            const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
            if
            (
                taskStart < 0
             || nTasks <= 0
             || taskStart > s.csrHeavyTaskCapacity - nTasks
            )
            {
                asm("trap;");
            }
            s.csrHeavyCellTaskStart[c] = taskStart;
            s.csrHeavyCellTaskCount[c] = nTasks;
            for (int localTask = 0; localTask < nTasks; ++localTask)
            {
                const int task = taskStart + localTask;
                const int taskBegin =
                    baseBegin + localTask*s.csrHeavyTileParticles;
                const int remaining = baseEnd - taskBegin;
                const int taskLength = remaining < s.csrHeavyTileParticles
                  ? remaining
                  : s.csrHeavyTileParticles;
                s.csrHeavyTaskCell[task] = c;
                s.csrHeavyTaskBegin[task] = taskBegin;
                s.csrHeavyTaskEnd[task] = taskBegin + taskLength;
            }
        }

        if (injectionCount > 0)
        {
            const int nTasks =
                1 + (injectionCount - 1)/s.csrHeavyTileParticles;
            const int taskStart =
                atomicAdd(s.preInjectionHeavyTaskCount, nTasks);
            if
            (
                taskStart < 0
             || nTasks <= 0
             || taskStart > s.csrHeavyTaskCapacity - nTasks
            )
            {
                asm("trap;");
            }
            s.preInjectionHeavyCellTaskStart[c] = taskStart;
            s.preInjectionHeavyCellTaskCount[c] = nTasks;
            for (int localTask = 0; localTask < nTasks; ++localTask)
            {
                const int task = taskStart + localTask;
                const int taskBegin =
                    injectionBegin + localTask*s.csrHeavyTileParticles;
                const int remaining = injectionEnd - taskBegin;
                const int taskLength = remaining < s.csrHeavyTileParticles
                  ? remaining
                  : s.csrHeavyTileParticles;
                s.preInjectionHeavyTaskCell[task] = c;
                s.preInjectionHeavyTaskBegin[task] = taskBegin;
                s.preInjectionHeavyTaskEnd[task] = taskBegin + taskLength;
            }
        }
    }
}

int prepareSplitPreCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::splitBaseAndInjection
    );
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }

    if
    (
        configureDynamicCsrHeavyPolicy
        (
            s,
            HeavyDirectoryKind::splitBaseAndInjection
        ) != 0
    )
    {
        return 1;
    }

                                                                           
                                                                         
                                                          
    cudaError_t err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->preInjectionHeavyTaskCount, 0, sizeof(int));
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->csrHeavyCellTaskStart,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->csrHeavyCellTaskCount,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->preInjectionHeavyCellTaskStart,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->preInjectionHeavyCellTaskCount,
            0,
            static_cast<size_t>(s->nCells)*sizeof(int)
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("reset split-pre combined heavy task metadata", err);
        return 1;
    }

    const int cellGrid = (s->nCells + block - 1)/block;
    buildSplitPreCsrHeavyReductionTasksKernel<<<cellGrid, block>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("build split-pre combined heavy tasks launch", err);
        return 1;
    }
    return 0;
#endif
}

int binParticlesByCell(DeviceState* s, const int block)
{
    s->useSplitPreDirectory = 0;
    s->preInjectionSegmentActive = 0;
    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    clearParticleCellBinsKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clearParticleCellBinsKernel launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        countParticlesByCellKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        countParticlesByCellKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("countParticlesByCellKernel launch", err);
        return 1;
    }

    err = cub::DeviceScan::ExclusiveSum
    (
        s->cellScanTempStorage,
        s->cellScanTempBytes,
        s->cellParticleCount,
        s->cellParticleOffset,
        s->nCells + 1
    );
    if (err != cudaSuccess)
    {
        setLastError("cell particle count exclusive scan", err);
        return 1;
    }

    initialiseParticleCellWritesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("initialiseParticleCellWritesKernel launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        scatterParticlesByCellKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        scatterParticlesByCellKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatterParticlesByCellKernel launch", err);
        return 1;
    }
    return prepareCsrHeavyReductionTasks(s, block);
}

int prepareSourceFreeSplitPreDirectory(DeviceState* s)
{
    cudaError_t err = cudaMemset
    (
        s->cellParticleOffset,
        0,
        static_cast<size_t>(s->nCells + 1)*sizeof(int)
    );
    if (err != cudaSuccess)
    {
        setLastError("reset source-free split-Dpre injection offsets", err);
        return 1;
    }

    s->preInjectionSegmentActive = 0;
    s->useSplitPreDirectory = 1;
    return 0;
}

int preparePreTransportParticleDirectory(DeviceState* s, const int block)
{
    if (s->csrSplitPreDirectoryEnabled == 0)
    {
        return binParticlesByCell(s, block);
    }

                                                                         
                                                                             
                                                                           
    if (s->preBaseDirectoryReady == 0)
    {
        s->useSplitPreDirectory = 0;
        s->preInjectionSegmentActive = 0;
        return binParticlesByCell(s, block);
    }

    if (s->nBoundarySources == 0)
    {
        return prepareSourceFreeSplitPreDirectory(s);
    }

    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    clearParticleCellBinsKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clear injection cell bins launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        countInjectedParticlesByCellKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        countInjectedParticlesByCellKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("countInjectedParticlesByCellKernel launch", err);
        return 1;
    }

    err = cub::DeviceScan::ExclusiveSum
    (
        s->cellScanTempStorage,
        s->cellScanTempBytes,
        s->cellParticleCount,
        s->cellParticleOffset,
        s->nCells + 1
    );
    if (err != cudaSuccess)
    {
        setLastError("injection cell particle count exclusive scan", err);
        return 1;
    }

    initialiseParticleCellWritesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("initialise injection cell writes launch", err);
        return 1;
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        scatterInjectedParticlesByCellKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        scatterInjectedParticlesByCellKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatterInjectedParticlesByCellKernel launch", err);
        return 1;
    }

    if (prepareSplitPreCsrHeavyReductionTasks(s, block) != 0)
    {
        return 1;
    }

    s->preInjectionSegmentActive = 1;
    s->useSplitPreDirectory = 1;
    return 0;
}

int rebuildResidentParticleMomentsFromParticles
(
    DeviceState* s,
    const int nParticles
)
{
    const int block = s->reductionBlockThreads;
    const int warpCount = (block + 31)/32;
    const int grid = (s->nCells + block - 1)/block;
    cudaError_t err = cudaSuccess;

    if (s->csrCellLocalPathEnabled != 0)
    {
        clearParticleMomentsKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "clearParticleMomentsKernel restart/create recovery launch",
                err
            );
            return 1;
        }

        if (nParticles > 0 && s->particleCapacity > 0)
        {
            if (binParticlesByCell(s, block) != 0)
            {
                return 1;
            }

            const size_t momentSharedBytes =
                8u*static_cast<size_t>(warpCount)*sizeof(double);

            if (s->csrHeavyReductionEnabled != 0)
            {
                if (launchCsrHeavyMomentReduction(s, block) != 0)
                {
                    return 1;
                }
            }
            else
            {
                accumulateParticleMomentsSegmentedKernel<false>
                    <<<s->nCells, block, momentSharedBytes>>>
                    (s->deviceState);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    setLastError
                    (
                        "accumulateParticleMomentsSegmentedKernel restart/create recovery launch",
                        err
                    );
                    return 1;
                }
            }
        }
    }
    else
    {
        const int countGrid = (s->nCells + 1 + block - 1)/block;
        clearParticleMomentsAndCountsAtomicKernel<<<countGrid, block>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "clearParticleMomentsAndCountsAtomicKernel restart/create recovery launch",
                err
            );
            return 1;
        }

        if (nParticles > 0 && s->particleCapacity > 0)
        {
            accumulateParticleMomentsAtomicKernel<<<s->particleWorkGrid, block>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "accumulateParticleMomentsAtomicKernel restart/create recovery launch",
                    err
                );
                return 1;
            }

            normalizeParticleMomentsAtomicKernel<<<grid, block>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "normalizeParticleMomentsAtomicKernel restart/create recovery launch",
                    err
                );
                return 1;
            }
        }
    }

    solidRecoveryFromParticleMomentsKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "solidRecoveryFromParticleMomentsKernel restart/create recovery launch",
            err
        );
        return 1;
    }

                                          
                                                                           
                           
    initialiseEpsGPrevKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "initialiseEpsGPrevKernel restart/create recovery launch",
            err
        );
        return 1;
    }

    initialiseThetaDragAlphaKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "initialiseThetaDragAlphaKernel restart/create recovery launch",
            err
        );
        return 1;
    }

    return 0;
}

__global__ void solidRecoveryFromParticleMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        clearSolidCell(s, c);
        return;
    }

    const double totalMomX = finiteOr(s.momRhoUPx[c], 0.0);
    const double totalMomY = finiteOr(s.momRhoUPy[c], 0.0);
    const double totalMomZ = finiteOr(s.momRhoUPz[c], 0.0);
    double totalEnergy = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double totalDiameter = clampMin(finiteOr(s.momRhoPD[c], 0.0), 0.0);
    const double totalHeat = clampMin(finiteOr(s.momRhoHpP[c], 0.0), 0.0);

    s.epsS[c] = rhoP/s.rhoSolid;
    s.rhoUsx[c] = totalMomX;
    s.rhoUsy[c] = totalMomY;
    s.rhoUsz[c] = totalMomZ;
    s.rhoDs[c] = totalDiameter;
    s.rhoHp[c] = totalHeat;
    s.Usx[c] = totalMomX/rhoP;
    s.Usy[c] = totalMomY/rhoP;
    s.Usz[c] = totalMomZ/rhoP;

    const double invRhoP = 1.0/rhoP;
    const double kinetic =
        0.5*(sqr3(totalMomX*invRhoP, totalMomY*invRhoP, totalMomZ*invRhoP)*rhoP);
    if (totalEnergy < kinetic)
    {
        totalEnergy = kinetic;
    }
    s.rhoEs[c] = totalEnergy;
    s.theta[c] = clampMin((totalEnergy - kinetic)/(1.5*rhoP), 0.0);
    s.dMeanCell[c] =
        clampMin(finiteOr(totalDiameter/rhoP, s.particleDiameterFallback), 1.0e-12);

    const double heatFactor = particleHeatFactorDevice(s);
    if (heatFactor > 0.0)
    {
        s.Tp[c] =
            clampRange
            (
                finiteOr(totalHeat/(rhoP*heatFactor), s.TpMin),
                s.TpMin,
                s.TpMax
            );
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = 0.0;
    }
}

#ifdef UGKP_DEVELOPMENT_PROBES

bool developmentProbeEnabled()
{
    return developmentProbe.mode != DevelopmentProbeMode::off;
}

bool developmentProbeFullValidation()
{
    return developmentProbe.mode == DevelopmentProbeMode::full;
}

void shutdownDevelopmentProbe()
{
    if (developmentProbe.deviceSummary != nullptr)
    {
        cudaFree(developmentProbe.deviceSummary);
        developmentProbe.deviceSummary = nullptr;
    }
    if (developmentProbe.totalStartEvent != nullptr)
    {
        cudaEventDestroy(developmentProbe.totalStartEvent);
        developmentProbe.totalStartEvent = nullptr;
    }
    if (developmentProbe.totalStopEvent != nullptr)
    {
        cudaEventDestroy(developmentProbe.totalStopEvent);
        developmentProbe.totalStopEvent = nullptr;
    }
    for (int stage = 0; stage < ProbeStageCount; ++stage)
    {
        developmentProbe.stageOccurrenceCount[stage] = 0;
        for (int occurrence = 0; occurrence < ProbeMaxOccurrences; ++occurrence)
        {
            if (developmentProbe.stageStartEvents[stage][occurrence] != nullptr)
            {
                cudaEventDestroy
                (
                    developmentProbe.stageStartEvents[stage][occurrence]
                );
                developmentProbe.stageStartEvents[stage][occurrence] = nullptr;
            }
            if (developmentProbe.stageStopEvents[stage][occurrence] != nullptr)
            {
                cudaEventDestroy
                (
                    developmentProbe.stageStopEvents[stage][occurrence]
                );
                developmentProbe.stageStopEvents[stage][occurrence] = nullptr;
            }
        }
    }
    if (developmentProbe.log != nullptr)
    {
        std::fflush(developmentProbe.log);
        std::fclose(developmentProbe.log);
        developmentProbe.log = nullptr;
    }

    developmentProbe.mode = DevelopmentProbeMode::off;
    developmentProbe.owner = nullptr;
    developmentProbe.interval = 1;
    developmentProbe.advanceIndex = 0;
    developmentProbe.failOnNonFinite = false;
    developmentProbe.pid = 0;
    developmentProbe.modeName = "off";
    developmentProbe.runId.clear();
    developmentProbe.variant.clear();
    developmentProbe.logPath.clear();
    developmentProbe.occupancy.clear();
    developmentProbe.injectedBySource.clear();
    developmentProbe.sourceResidualMass.clear();
}

void writeDevelopmentProbeCsvField(FILE* file, const char* value)
{
    std::fputc('"', file);
    if (value != nullptr)
    {
        for (const char* p = value; *p != '\0'; ++p)
        {
            if (*p == '"')
            {
                std::fputc('"', file);
                std::fputc('"', file);
            }
            else if (*p == '\n' || *p == '\r')
            {
                std::fputc(' ', file);
            }
            else
            {
                std::fputc(*p, file);
            }
        }
    }
    std::fputc('"', file);
}

int writeDevelopmentProbeHeader()
{
    static const char header[] =
        "schema_version,run_id,variant,backend_pid,step,simulation_time,dt,"
        "mode,timing_valid,status,error_stage,error_message,n_cells,"
        "block_exponent,block_threads,sm_count,particle_blocks_per_sm,"
        "light_blocks_per_sm,heavy_blocks_per_sm,heavy_reduction_enabled,"
        "particle_path,particle_count,particle_capacity,particle_utilisation,"
        "pretransport_particle_count,base_particle_count,"
        "injected_particle_count,removed_particle_count,injection_fraction,"
        "source_residual_mass,"
        "occupancy_sum,occupancy_nonempty,occupancy_min,occupancy_mean,"
        "occupancy_stddev,occupancy_cv,occupancy_p50,occupancy_p95,"
        "occupancy_p99,occupancy_max,occupancy_i2,occupancy_imax,"
        "heavy_threshold,heavy_tile_particles,heavy_cell_count,"
        "heavy_particle_count,heavy_cell_fraction,heavy_particle_fraction,"
        "heavy_task_count_estimate,occupancy_matches_count,bad_cells,"
        "bad_particles,bad_field_mask,first_bad_cell,first_bad_particle,"
        "total_ms,gas_flux_ms,eulerian_coupling_ms,injection_ms,bin_pre_ms,"
        "pressure_pre_ms,collision_pool_ms,relax_ms,track_ms,bin_post_ms,"
        "moments_ms,pressure_post_ms,compaction_ms,boundary_ms\n";

    if (std::fputs(header, developmentProbe.log) == EOF)
    {
        setLastErrorText("cannot write UGKP development probe CSV header");
        return 1;
    }
    if (std::fflush(developmentProbe.log) != 0)
    {
        setLastErrorText("cannot flush UGKP development probe CSV header");
        return 1;
    }
    return 0;
}

int writeDevelopmentProbeSample(const DevelopmentProbeSample& sample)
{
    FILE* const file = developmentProbe.log;
    if (file == nullptr)
    {
        setLastErrorText("UGKP development probe CSV is not open");
        return 1;
    }

    std::fprintf(file, "3,");
    writeDevelopmentProbeCsvField(file, developmentProbe.runId.c_str());
    std::fputc(',', file);
    writeDevelopmentProbeCsvField(file, developmentProbe.variant.c_str());
    std::fprintf
    (
        file,
        ",%d,%llu,%.17g,%.17g,",
        developmentProbe.pid,
        sample.step,
        sample.simulationTime,
        sample.dt
    );
    writeDevelopmentProbeCsvField(file, developmentProbe.modeName.c_str());
    std::fprintf(file, ",%d,", sample.timingValid);
    writeDevelopmentProbeCsvField(file, sample.status);
    std::fputc(',', file);
    writeDevelopmentProbeCsvField(file, sample.errorStage);
    std::fputc(',', file);
    writeDevelopmentProbeCsvField(file, sample.errorMessage);
    std::fprintf
    (
        file,
        ",%d,%d,%d,%d,%d,%d,%d,%d,"
        "%d,%d,%d,%.17g,%d,%d,%d,%d,%.17g,%.17g,"
        "%lld,%d,%d,%.17g,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,"
        "%d,%d,%d,%lld,%.17g,%.17g,%lld,%d,"
        "%llu,%llu,0x%016llx,%d,%d,%.9g",
        sample.nCells,
        sample.blockExponent,
        sample.blockThreads,
        sample.smCount,
        sample.particleBlocksPerSm,
        sample.lightBlocksPerSm,
        sample.heavyBlocksPerSm,
        sample.heavyReductionEnabled,
        sample.particlePath,
        sample.particleCount,
        sample.particleCapacity,
        sample.particleUtilisation,
        sample.preTransportParticleCount,
        sample.baseParticleCount,
        sample.injectedParticleCount,
        sample.removedParticleCount,
        sample.injectionFraction,
        sample.sourceResidualMass,
        sample.occupancySum,
        sample.occupancyNonEmpty,
        sample.occupancyMin,
        sample.occupancyMean,
        sample.occupancyStddev,
        sample.occupancyCv,
        sample.occupancyP50,
        sample.occupancyP95,
        sample.occupancyP99,
        sample.occupancyMax,
        sample.occupancyI2,
        sample.occupancyImax,
        sample.heavyThreshold,
        sample.heavyTileParticles,
        sample.heavyCellCount,
        sample.heavyParticleCount,
        sample.heavyCellFraction,
        sample.heavyParticleFraction,
        sample.heavyTaskCountEstimate,
        sample.occupancyMatchesCount,
        sample.badCells,
        sample.badParticles,
        sample.badFieldMask,
        sample.firstBadCell,
        sample.firstBadParticle,
        static_cast<double>(sample.totalMs)
    );
    for (int i = 0; i < ProbeStageCount; ++i)
    {
        std::fprintf(file, ",%.9g", static_cast<double>(sample.stageMs[i]));
    }
    std::fputc('\n', file);

    if (std::fflush(file) != 0 || std::ferror(file) != 0)
    {
        setLastErrorText("cannot write UGKP development probe CSV sample");
        return 1;
    }
    return 0;
}

std::string developmentProbePathForPid(const char* configured)
{
    std::string path(configured == nullptr ? "" : configured);
    const std::string token("%p");
    const std::string pidText = std::to_string(static_cast<long long>(::getpid()));
    std::string::size_type pos = 0;
    while ((pos = path.find(token, pos)) != std::string::npos)
    {
        path.replace(pos, token.size(), pidText);
        pos += pidText.size();
    }
    return path;
}

bool developmentProbeBoolean(const char* value)
{
    return
        value != nullptr
     &&
        (
            std::strcmp(value, "1") == 0
         || std::strcmp(value, "true") == 0
         || std::strcmp(value, "on") == 0
         || std::strcmp(value, "yes") == 0
        );
}

int initialiseDevelopmentProbe(DeviceState* owner)
{
    if (developmentProbe.owner != nullptr)
    {
        setLastErrorText
        (
            "UGKP development probe already belongs to another backend handle"
        );
        return 1;
    }
    shutdownDevelopmentProbe();

    const char* const configuredMode = std::getenv("UGKP_DEV_PROBE_MODE");
    if
    (
        configuredMode == nullptr
     || *configuredMode == '\0'
     || std::strcmp(configuredMode, "off") == 0
     || std::strcmp(configuredMode, "0") == 0
    )
    {
        return 0;
    }

    if (std::strcmp(configuredMode, "timing") == 0)
    {
        developmentProbe.mode = DevelopmentProbeMode::timing;
        developmentProbe.modeName = "timing";
    }
    else if
    (
        std::strcmp(configuredMode, "full") == 0
     || std::strcmp(configuredMode, "1") == 0
    )
    {
        developmentProbe.mode = DevelopmentProbeMode::full;
        developmentProbe.modeName = "full";
    }
    else
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "invalid UGKP_DEV_PROBE_MODE '%s' (expected off, timing, or full)",
            configuredMode
        );
        return 1;
    }

    const char* const configuredLog = std::getenv("UGKP_DEV_PROBE_LOG");
    if (configuredLog == nullptr || *configuredLog == '\0')
    {
        setLastErrorText
        (
            "UGKP_DEV_PROBE_LOG is required when UGKP_DEV_PROBE_MODE is enabled"
        );
        shutdownDevelopmentProbe();
        return 1;
    }
    if (configuredLog[0] != '/')
    {
        setLastErrorText
        (
            "UGKP_DEV_PROBE_LOG must be an absolute path"
        );
        shutdownDevelopmentProbe();
        return 1;
    }

    if (const char* configuredInterval = std::getenv("UGKP_DEV_PROBE_INTERVAL"))
    {
        errno = 0;
        char* end = nullptr;
        const unsigned long long interval =
            std::strtoull(configuredInterval, &end, 10);
        if
        (
            errno != 0
         || end == configuredInterval
         || *end != '\0'
         || interval == 0
        )
        {
            std::snprintf
            (
                lastError,
                sizeof(lastError),
                "invalid UGKP_DEV_PROBE_INTERVAL '%s'",
                configuredInterval
            );
            shutdownDevelopmentProbe();
            return 1;
        }
        developmentProbe.interval = interval;
    }

    developmentProbe.failOnNonFinite = developmentProbeBoolean
    (
        std::getenv("UGKP_DEV_PROBE_FAIL_ON_NONFINITE")
    );
    developmentProbe.pid = static_cast<int>(::getpid());
    developmentProbe.logPath = developmentProbePathForPid(configuredLog);
    if (const char* value = std::getenv("UGKP_DEV_PROBE_RUN_ID"))
    {
        developmentProbe.runId = value;
    }
    if (const char* value = std::getenv("UGKP_DEV_PROBE_VARIANT"))
    {
        developmentProbe.variant = value;
    }

    developmentProbe.log = std::fopen(developmentProbe.logPath.c_str(), "a+");
    if (developmentProbe.log == nullptr)
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "cannot open UGKP development probe log '%s': %s",
            developmentProbe.logPath.c_str(),
            std::strerror(errno)
        );
        shutdownDevelopmentProbe();
        return 1;
    }
    std::setvbuf(developmentProbe.log, nullptr, _IOLBF, 0);
    if (std::fseek(developmentProbe.log, 0, SEEK_END) != 0)
    {
        setLastErrorText("cannot seek UGKP development probe log");
        shutdownDevelopmentProbe();
        return 1;
    }
    const long logBytes = std::ftell(developmentProbe.log);
    if (logBytes < 0)
    {
        setLastErrorText("cannot query UGKP development probe log length");
        shutdownDevelopmentProbe();
        return 1;
    }
    if (logBytes == 0 && writeDevelopmentProbeHeader() != 0)
    {
        shutdownDevelopmentProbe();
        return 1;
    }

    cudaError_t err = cudaEventCreate(&developmentProbe.totalStartEvent);
    if (err == cudaSuccess)
    {
        err = cudaEventCreate(&developmentProbe.totalStopEvent);
    }
    for
    (
        int stage = 0;
        err == cudaSuccess && stage < ProbeStageCount;
        ++stage
    )
    {
        for
        (
            int occurrence = 0;
            err == cudaSuccess && occurrence < ProbeMaxOccurrences;
            ++occurrence
        )
        {
            err = cudaEventCreate
            (
                &developmentProbe.stageStartEvents[stage][occurrence]
            );
            if (err == cudaSuccess)
            {
                err = cudaEventCreate
                (
                    &developmentProbe.stageStopEvents[stage][occurrence]
                );
            }
        }
    }
    if (err != cudaSuccess)
    {
        setLastError("cudaEventCreate UGKP development probe", err);
        shutdownDevelopmentProbe();
        return 1;
    }

    if (developmentProbeFullValidation())
    {
        const cudaError_t err = cudaMalloc
        (
            reinterpret_cast<void**>(&developmentProbe.deviceSummary),
            sizeof(DevelopmentProbeDeviceSummary)
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMalloc UGKP development probe summary", err);
            shutdownDevelopmentProbe();
            return 1;
        }
    }

    try
    {
        developmentProbe.occupancy.resize(static_cast<size_t>(owner->nCells));
    }
    catch (...)
    {
        setLastErrorText("cannot allocate UGKP development probe occupancy mirror");
        shutdownDevelopmentProbe();
        return 1;
    }
    developmentProbe.owner = owner;
    return 0;
}

int collectDevelopmentProbeSample
(
    DeviceState* s,
    const bool particlePath,
    DevelopmentProbeSample& sample
)
{
    cudaError_t err = cudaEventSynchronize(developmentProbe.totalStopEvent);
    if (err != cudaSuccess)
    {
        setLastError("UGKP development probe final event synchronization", err);
        return 1;
    }

    err = cudaEventElapsedTime
    (
        &sample.totalMs,
        developmentProbe.totalStartEvent,
        developmentProbe.totalStopEvent
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaEventElapsedTime UGKP development probe total", err);
        return 1;
    }
    for (int stage = 0; stage < ProbeStageCount; ++stage)
    {
        sample.stageMs[stage] = 0.0f;
        for
        (
            int occurrence = 0;
            occurrence < developmentProbe.stageOccurrenceCount[stage];
            ++occurrence
        )
        {
            if
            (
                !developmentProbe.stageOccurrenceExecuted[stage][occurrence]
            )
            {
                continue;
            }
            float elapsedMs = 0.0f;
            err = cudaEventElapsedTime
            (
                &elapsedMs,
                developmentProbe.stageStartEvents[stage][occurrence],
                developmentProbe.stageStopEvents[stage][occurrence]
            );
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "cudaEventElapsedTime UGKP development probe stage",
                    err
                );
                return 1;
            }
            sample.stageMs[stage] += elapsedMs;
        }
    }
    sample.timingValid = 1;

    int rawParticleCount = 0;
    err = cudaMemcpy
    (
        &rawParticleCount,
        s->particleCountDevice,
        sizeof(rawParticleCount),
        cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemcpy UGKP development probe particle count", err);
        return 1;
    }
    sample.particleCount = rawParticleCount;
    sample.particleUtilisation = s->particleCapacity > 0
      ? static_cast<double>(rawParticleCount)/static_cast<double>(s->particleCapacity)
      : 0.0;
    if (rawParticleCount < 0 || rawParticleCount > s->particleCapacity)
    {
        sample.badFieldMask |= ProbeBadParticleCount;
        ++sample.badParticles;
    }

    sample.preTransportParticleCount = rawParticleCount;
    if (particlePath && s->diagnosticPreTransportParticleCount != nullptr)
    {
        err = cudaMemcpy
        (
            &sample.preTransportParticleCount,
            s->diagnosticPreTransportParticleCount,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError
            (
                "cudaMemcpy diagnostic pre-transport particle count",
                err
            );
            return 1;
        }
    }

    if (s->nBoundarySources > 0)
    {
        try
        {
            developmentProbe.injectedBySource.resize
            (
                static_cast<size_t>(s->nBoundarySources)
            );
            developmentProbe.sourceResidualMass.resize
            (
                static_cast<size_t>(s->nBoundarySources)
            );
        }
        catch (...)
        {
            setLastErrorText
            (
                "cannot allocate development-probe injection mirrors"
            );
            return 1;
        }
        err = cudaMemcpy
        (
            developmentProbe.injectedBySource.data(),
            s->sourceInjectedCount,
            static_cast<size_t>(s->nBoundarySources)*sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (err == cudaSuccess)
        {
            err = cudaMemcpy
            (
                developmentProbe.sourceResidualMass.data(),
                s->sourceResidualMass,
                static_cast<size_t>(s->nBoundarySources)*sizeof(double),
                cudaMemcpyDeviceToHost
            );
        }
        if (err != cudaSuccess)
        {
            setLastError("cudaMemcpy development-probe injection counters", err);
            return 1;
        }
        for (const int count : developmentProbe.injectedBySource)
        {
            sample.injectedParticleCount += count;
        }
        for (const double residual : developmentProbe.sourceResidualMass)
        {
            sample.sourceResidualMass += residual;
        }
    }
    sample.baseParticleCount = std::max
    (
        sample.preTransportParticleCount - sample.injectedParticleCount,
        0
    );
    sample.removedParticleCount = std::max
    (
        sample.preTransportParticleCount - rawParticleCount,
        0
    );
    sample.injectionFraction = sample.preTransportParticleCount > 0
      ? static_cast<double>(sample.injectedParticleCount)
       /static_cast<double>(sample.preTransportParticleCount)
      : 0.0;

    if (particlePath)
    {
        err = cudaMemcpy
        (
            developmentProbe.occupancy.data(),
            s->cellParticleCount,
            static_cast<size_t>(s->nCells)*sizeof(int),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemcpy UGKP development probe cell occupancy", err);
            return 1;
        }
    }
    else
    {
        std::fill
        (
            developmentProbe.occupancy.begin(),
            developmentProbe.occupancy.end(),
            0
        );
    }

    long double sumSquares = 0.0L;
    int occupancyMin = INT_MAX;
    int occupancyMax = INT_MIN;
    if (s->csrHeavyReductionEnabled != 0)
    {
        int dynamicPolicy[2] = {0, 0};
        err = cudaMemcpy
        (
            dynamicPolicy,
            reinterpret_cast<const unsigned char*>(s->deviceState)
              + offsetof(DeviceState, csrHeavyCellThreshold),
            sizeof(dynamicPolicy),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError("copy dynamic UGKP heavy policy", err);
            return 1;
        }
        sample.heavyThreshold = dynamicPolicy[0];
        sample.heavyTileParticles = dynamicPolicy[1];
    }
    else
    {
        sample.heavyThreshold = 0;
        sample.heavyTileParticles = 0;
    }
    for (const int count : developmentProbe.occupancy)
    {
        sample.occupancySum += static_cast<long long>(count);
        sumSquares += static_cast<long double>(count)*count;
        occupancyMin = std::min(occupancyMin, count);
        occupancyMax = std::max(occupancyMax, count);
        sample.occupancyNonEmpty += count > 0 ? 1 : 0;
        if
        (
            s->csrHeavyReductionEnabled != 0
         && count > sample.heavyThreshold
        )
        {
            ++sample.heavyCellCount;
            sample.heavyParticleCount += static_cast<long long>(count);
            sample.heavyTaskCountEstimate +=
                (static_cast<long long>(count) + sample.heavyTileParticles - 1)
               /sample.heavyTileParticles;
        }
        if (count < 0)
        {
            sample.badFieldMask |= ProbeBadOccupancy;
        }
    }
    if (s->nCells > 0)
    {
        sample.occupancyMin = occupancyMin;
        sample.occupancyMax = occupancyMax;
        sample.occupancyMean =
            static_cast<double>(sample.occupancySum)/static_cast<double>(s->nCells);
        const long double mean = static_cast<long double>(sample.occupancyMean);
        long double variance = sumSquares/static_cast<long double>(s->nCells)
                             - mean*mean;
        variance = variance > 0.0L ? variance : 0.0L;
        sample.occupancyStddev = std::sqrt(static_cast<double>(variance));
        sample.occupancyCv = sample.occupancyMean > 0.0
          ? sample.occupancyStddev/sample.occupancyMean
          : 0.0;
        if (sample.occupancySum > 0)
        {
            const long double particleCount =
                static_cast<long double>(sample.occupancySum);
            sample.occupancyI2 = static_cast<double>
            (
                static_cast<long double>(s->nCells)*sumSquares
               /(particleCount*particleCount)
            );
            sample.occupancyImax =
                static_cast<double>(s->nCells)
               *static_cast<double>(occupancyMax)
               /static_cast<double>(sample.occupancySum);
            sample.heavyParticleFraction =
                static_cast<double>(sample.heavyParticleCount)
               /static_cast<double>(sample.occupancySum);
        }
        sample.heavyCellFraction =
            static_cast<double>(sample.heavyCellCount)
           /static_cast<double>(s->nCells);

        std::sort
        (
            developmentProbe.occupancy.begin(),
            developmentProbe.occupancy.end()
        );
        const size_t last = developmentProbe.occupancy.size() - 1;
        sample.occupancyP50 = developmentProbe.occupancy[(50u*last)/100u];
        sample.occupancyP95 = developmentProbe.occupancy[(95u*last)/100u];
        sample.occupancyP99 = developmentProbe.occupancy[(99u*last)/100u];
    }

    sample.occupancyMatchesCount =
        rawParticleCount >= 0
     && rawParticleCount <= s->particleCapacity
     && sample.occupancySum == static_cast<long long>(rawParticleCount);
    if (!sample.occupancyMatchesCount)
    {
        sample.badFieldMask |= ProbeBadOccupancy;
    }

    if (developmentProbeFullValidation())
    {
        err = cudaMemset
        (
            developmentProbe.deviceSummary,
            0,
            sizeof(DevelopmentProbeDeviceSummary)
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemset UGKP development probe summary", err);
            return 1;
        }

        const int block = s->reductionBlockThreads;
        const int cellGrid = (s->nCells + block - 1)/block;
        validateDevelopmentProbeCellsKernel<<<cellGrid, block>>>
        (
            s->deviceState,
            developmentProbe.deviceSummary,
            particlePath ? 1 : 0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("validateDevelopmentProbeCellsKernel launch", err);
            return 1;
        }

        if (particlePath && s->particleWorkGrid > 0)
        {
            validateDevelopmentProbeParticlesKernel<<<s->particleWorkGrid, block>>>
            (
                s->deviceState,
                developmentProbe.deviceSummary
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("validateDevelopmentProbeParticlesKernel launch", err);
                return 1;
            }
        }

        DevelopmentProbeDeviceSummary summary{};
        err = cudaMemcpy
        (
            &summary,
            developmentProbe.deviceSummary,
            sizeof(summary),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemcpy UGKP development probe summary", err);
            return 1;
        }
        sample.badCells += summary.badCells;
        sample.badParticles += summary.badParticles;
        sample.badFieldMask |= summary.badFieldMask;
        sample.firstBadCell = summary.firstBadCellPlusOne == 0
          ? -1
          : summary.firstBadCellPlusOne - 1;
        sample.firstBadParticle = summary.firstBadParticlePlusOne == 0
          ? -1
          : summary.firstBadParticlePlusOne - 1;
    }

    if (sample.badFieldMask != 0)
    {
        sample.status = "state_invalid";
    }
    return 0;
}

class DevelopmentAdvanceProbe
{
    DeviceState* state_ = nullptr;
    unsigned long long step_ = 0;
    double simulationTime_ = 0.0;
    double dt_ = 0.0;
    const char* currentStage_ = "advance_begin";
    bool enabled_ = false;
    bool sampled_ = false;
    bool failed_ = false;
    bool completed_ = false;
    bool rowWritten_ = false;

public:
    DevelopmentAdvanceProbe
    (
        DeviceState* state,
        const double dt,
        const double simulationTime
    )
    :
        state_(state),
        simulationTime_(simulationTime),
        dt_(dt),
        enabled_(developmentProbeEnabled())
    {
        if (!enabled_)
        {
            return;
        }

        step_ = ++developmentProbe.advanceIndex;
        sampled_ = (step_ % developmentProbe.interval) == 0;
        if (sampled_)
        {
            for (int stage = 0; stage < ProbeStageCount; ++stage)
            {
                developmentProbe.stageOccurrenceCount[stage] = 0;
                for
                (
                    int occurrence = 0;
                    occurrence < ProbeMaxOccurrences;
                    ++occurrence
                )
                {
                    developmentProbe.stageOccurrenceExecuted[stage][occurrence] =
                        false;
                }
            }
            const cudaError_t err = cudaEventRecord
            (
                developmentProbe.totalStartEvent,
                0
            );
            if (err != cudaSuccess)
            {
                setLastError("cudaEventRecord UGKP development probe start", err);
                failed_ = true;
            }
        }
    }

    ~DevelopmentAdvanceProbe()
    {
        if (!enabled_ || completed_ || rowWritten_)
        {
            return;
        }

        DevelopmentProbeSample sample;
        sample.step = step_;
        sample.simulationTime = simulationTime_;
        sample.dt = dt_;
        sample.status = "error";
        sample.errorStage = currentStage_;
        sample.errorMessage = lastError;
        if (state_ != nullptr)
        {
            sample.nCells = state_->nCells;
            sample.blockExponent = 0;
            sample.blockThreads = state_->reductionBlockThreads;
            sample.smCount = state_->multiprocessorCount;
            sample.particleBlocksPerSm = state_->particleBlocksPerSm;
            sample.lightBlocksPerSm = state_->lightBlocksPerSm;
            sample.heavyBlocksPerSm = state_->heavyBlocksPerSm;
            sample.heavyReductionEnabled =
                state_->csrHeavyReductionEnabled;
            sample.particlePath = state_->particlesMayBePresent ? 1 : 0;
            sample.particleCapacity = state_->particleCapacity;
        }
        char preservedError[sizeof(lastError)]{};
        std::snprintf
        (
            preservedError,
            sizeof(preservedError),
            "%s",
            lastError
        );
        (void)writeDevelopmentProbeSample(sample);
        std::snprintf(lastError, sizeof(lastError), "%s", preservedError);
        rowWritten_ = true;
    }

    bool failed() const
    {
        return failed_;
    }

    void enter(const DevelopmentProbeStage stage)
    {
        currentStage_ = developmentProbeStageNames[stage];
        if (!sampled_ || failed_)
        {
            return;
        }
        const int occurrence = developmentProbe.stageOccurrenceCount[stage];
        if (occurrence < 0 || occurrence >= ProbeMaxOccurrences)
        {
            setLastErrorText
            (
                "UGKP development probe stage occurrence capacity exceeded"
            );
            failed_ = true;
            return;
        }
        const cudaError_t err = cudaEventRecord
        (
            developmentProbe.stageStartEvents[stage][occurrence],
            0
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventRecord UGKP development probe stage start", err);
            failed_ = true;
        }
    }

    int leave
    (
        const DevelopmentProbeStage stage,
        const bool executed = true
    )
    {
        if (!sampled_)
        {
            return 0;
        }
        if (failed_)
        {
            return 1;
        }
        const int occurrence = developmentProbe.stageOccurrenceCount[stage];
        if (occurrence < 0 || occurrence >= ProbeMaxOccurrences)
        {
            setLastErrorText
            (
                "UGKP development probe stage occurrence capacity exceeded"
            );
            failed_ = true;
            return 1;
        }
        const cudaError_t err = cudaEventRecord
        (
            developmentProbe.stageStopEvents[stage][occurrence],
            0
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventRecord UGKP development probe stage", err);
            failed_ = true;
            return 1;
        }
        developmentProbe.stageOccurrenceExecuted[stage][occurrence] = executed;
        developmentProbe.stageOccurrenceCount[stage] = occurrence + 1;
        return 0;
    }

    int finish(const bool particlePath)
    {
        if (!enabled_ || !sampled_)
        {
            completed_ = true;
            return 0;
        }

        currentStage_ = "probe_collect";
        const cudaError_t stopError = cudaEventRecord
        (
            developmentProbe.totalStopEvent,
            0
        );
        if (stopError != cudaSuccess)
        {
            setLastError
            (
                "cudaEventRecord UGKP development probe total stop",
                stopError
            );
            return 1;
        }
        DevelopmentProbeSample sample;
        sample.step = step_;
        sample.simulationTime = simulationTime_;
        sample.dt = dt_;
        sample.nCells = state_->nCells;
        sample.blockExponent = 0;
        sample.blockThreads = state_->reductionBlockThreads;
        sample.smCount = state_->multiprocessorCount;
        sample.particleBlocksPerSm = state_->particleBlocksPerSm;
        sample.lightBlocksPerSm = state_->lightBlocksPerSm;
        sample.heavyBlocksPerSm = state_->heavyBlocksPerSm;
        sample.heavyReductionEnabled = state_->csrHeavyReductionEnabled;
        sample.particlePath = particlePath ? 1 : 0;
        sample.particleCapacity = state_->particleCapacity;
        if (collectDevelopmentProbeSample(state_, particlePath, sample) != 0)
        {
            return 1;
        }
        if (writeDevelopmentProbeSample(sample) != 0)
        {
            rowWritten_ = true;
            return 1;
        }
        rowWritten_ = true;

        if
        (
            developmentProbe.failOnNonFinite
         && sample.badFieldMask != 0
        )
        {
            setLastErrorText
            (
                "UGKP development probe found non-finite or invalid resident state"
            );
            return 1;
        }

        completed_ = true;
        return 0;
    }
};

#endif

extern "C" const char* ugkwpGpuResidentStrictLastError()
{
    return lastError;
}

int advanceGasEulerSubstage(DeviceState* s, const double dt)
{
    const int cellBlock = s->fixedCellBlockThreads;
    const int faceBlock = s->fixedFaceBlockThreads;
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    cudaError_t err = cudaSuccess;

    recoverGasPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel gas stage", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel gas stage", err);
            return 1;
        }
    }
    if (faceGrid > 0)
    {
        updateRiemannBoundaryMirrorKernel<<<faceGrid, faceBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("updateRiemannBoundaryMirrorKernel gas stage", err);
            return 1;
        }
    }
    if (s->hostTurbulenceModel == 3)
    {
        applySstWallFunctionStateKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstWallFunctionStateKernel gas stage", err);
            return 1;
        }
    }
    if (s->hostGasFluxScheme == 7)
    {
        computeGasHllcAdcSensorKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasHllcAdcSensorKernel launch", err);
            return 1;
        }
    }
    computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasPrimitiveGradientsKernel launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstGradientsKernel launch", err);
            return 1;
        }
    }
    computeGasGradientLimiterKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasGradientLimiterKernel launch", err);
        return 1;
    }
    computeGasEddyViscosityKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasEddyViscosityKernel launch", err);
        return 1;
    }
    if (faceGrid <= 0)
    {
        return 0;
    }
    if (s->hostTurbulenceModel == 0)
    {
        computeGasInternalFaceFluxKernel<false><<<faceGrid, faceBlock>>>
        (
            s->deviceState,
            dt
        );
    }
    else
    {
        computeGasInternalFaceFluxKernel<true><<<faceGrid, faceBlock>>>
        (
            s->deviceState,
            dt
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasInternalFaceFluxKernel launch", err);
        return 1;
    }
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<faceGrid, faceBlock>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "enforcePeriodicGasFluxAntisymmetryKernel launch",
                err
            );
            return 1;
        }
    }
    computeGasFluxPositivityScaleKernel<<<cellGrid, cellBlock>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasFluxPositivityScaleKernel launch", err);
        return 1;
    }
    applyGasFluxPositivityScaleKernel<<<faceGrid, faceBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasFluxPositivityScaleKernel launch", err);
        return 1;
    }
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<faceGrid, faceBlock>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "enforcePeriodicGasFluxAntisymmetryKernel post-scale launch",
                err
            );
            return 1;
        }
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstFaceFluxKernel<<<faceGrid, faceBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstFaceFluxKernel launch", err);
            return 1;
        }
        if (s->hasPeriodicFaces != 0)
        {
            enforcePeriodicSstFluxAntisymmetryKernel<<<faceGrid, faceBlock>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "enforcePeriodicSstFluxAntisymmetryKernel launch",
                    err
                );
                return 1;
            }
        }
        applySstFluxAndSourceKernel<<<cellGrid, cellBlock>>>
        (
            s->deviceState,
            dt
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstFluxAndSourceKernel launch", err);
            return 1;
        }
    }
    applyGasFluxDivergenceByCellKernel<<<cellGrid, cellBlock>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasFluxDivergenceByCellKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel post-gas launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel post-gas launch", err);
            return 1;
        }
    }
    return 0;
}

int blendGasRungeKuttaStage
(
    DeviceState* s,
    const double initialWeight,
    const double stageWeight
)
{
    const int block = s->fixedCellBlockThreads;
    const int cellGrid = (s->nCells + block - 1)/block;
    blendGasConservativeStateKernel<<<cellGrid, block>>>
    (
        s->deviceState,
        initialWeight,
        stageWeight
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("blendGasConservativeStateKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel after RK blend", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel after RK blend", err);
            return 1;
        }
    }
    return 0;
}

int advanceGasFluxStage
(
    DeviceState* s,
    const double dt,
    const double simulationTime
)
{
                                                                            
                                                                        
                                                                          
                                                                  
    const int preparationCellBlock = s->fixedCellBlockThreads;
    const int preparationFaceBlock = s->fixedFaceBlockThreads;
    const int preparationCellGrid =
        (s->nCells + preparationCellBlock - 1)/preparationCellBlock;
    const int preparationFaceGrid =
        (s->nFaces + preparationFaceBlock - 1)/preparationFaceBlock;
    recoverGasPrimitivesKernel<<<preparationCellGrid, preparationCellBlock>>>
    (
        s->deviceState
    );
    cudaError_t preparationError = cudaGetLastError();
    if (preparationError != cudaSuccess)
    {
        setLastError
        (
            "recoverGasPrimitivesKernel before gas advance",
            preparationError
        );
        return 1;
    }
    if (preparationFaceGrid > 0)
    {
        updateLegacyGasBoundaryMirrorKernel
            <<<preparationFaceGrid, preparationFaceBlock>>>
            (s->deviceState, simulationTime);
        preparationError = cudaGetLastError();
        if (preparationError != cudaSuccess)
        {
            setLastError
            (
                "updateLegacyGasBoundaryMirrorKernel before gas advance",
                preparationError
            );
            return 1;
        }
    }

    if (s->hostGasTimeIntegrator == 1)
    {
        return advanceGasEulerSubstage(s, dt);
    }

    const int block = s->fixedCellBlockThreads;
    const int cellGrid = (s->nCells + block - 1)/block;
    saveGasConservativeStateKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("saveGasConservativeStateKernel launch", err);
        return 1;
    }

    if (advanceGasEulerSubstage(s, dt) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt) != 0)
    {
        return 1;
    }

    if (s->hostGasTimeIntegrator == 2)
    {
                                                         
        return blendGasRungeKuttaStage(s, 0.5, 0.5);
    }
    if (s->hostGasTimeIntegrator != 3)
    {
        setLastErrorText("invalid resident gas time-integrator selector");
        return 1;
    }

                  
                                        
                                         
    if (blendGasRungeKuttaStage(s, 0.75, 0.25) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt) != 0)
    {
        return 1;
    }
    return blendGasRungeKuttaStage
    (
        s,
        1.0/3.0,
        2.0/3.0
    );
}

int finaliseGasBoundaryStage
(
    DeviceState* s,
    const double dt,
    const double simulationTime
)
{
    const int block = s->fixedFaceBlockThreads;
    const int allFaceGrid = (s->nFaces + block - 1)/block;
    if (allFaceGrid <= 0)
    {
        return 0;
    }

    updateLegacyGasBoundaryMirrorKernel<<<allFaceGrid, block>>>
    (
        s->deviceState,
        simulationTime
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateLegacyGasBoundaryMirrorKernel final gas boundary",
            err
        );
        return 1;
    }

    updateRiemannBoundaryMirrorKernel<<<allFaceGrid, block>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateRiemannBoundaryMirrorKernel final gas boundary",
            err
        );
        return 1;
    }

    updateWaveTransmissivePressureBoundaryKernel<<<allFaceGrid, block>>>
    (
        s->deviceState,
        dt
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateWaveTransmissivePressureBoundaryKernel final gas boundary",
            err
        );
        return 1;
    }
    return 0;
}

template<class Kernel>
int queryKernelBlocksPerSm
(
    int& blocksPerSm,
    const char* label,
    Kernel kernel,
    const int blockThreads,
    const size_t dynamicSharedBytes
)
{
    blocksPerSm = 0;
    const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
    (
        &blocksPerSm,
        kernel,
        blockThreads,
        dynamicSharedBytes
    );
    if (err != cudaSuccess)
    {
        setLastError(label, err);
        return 1;
    }
    if (blocksPerSm <= 0)
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "%s cannot launch with B=%d",
            label,
            blockThreads
        );
        return 1;
    }
    return 0;
}

static constexpr int toolB1WarmupRuns = 1;
static constexpr int toolB1MeasuredRuns = 5;

int launchToolB1CellBundle(DeviceState* s, const int block)
{
    const int grid = (s->nCells + block - 1)/block;
    recoverGasPrimitivesKernel<<<grid, block>>>(s->deviceState);
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<grid, block>>>(s->deviceState);
        applySstWallFunctionStateKernel<<<grid, block>>>(s->deviceState);
    }
    if (s->hostGasFluxScheme == 7)
    {
        computeGasHllcAdcSensorKernel<<<grid, block>>>(s->deviceState);
    }
    computeGasPrimitiveGradientsKernel<<<grid, block>>>(s->deviceState);
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<grid, block>>>(s->deviceState);
    }
    computeGasGradientLimiterKernel<<<grid, block>>>(s->deviceState);
    computeGasEddyViscosityKernel<<<grid, block>>>(s->deviceState);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("ToolB1 gas-cell bundle launch", err);
        return 1;
    }
    return 0;
}

int launchToolB1FaceBundle
(
    DeviceState* s,
    const int block,
    const double dt,
    const double simulationTime
)
{
    const int grid = (s->nFaces + block - 1)/block;
    if (grid <= 0)
    {
        return 0;
    }
    updateLegacyGasBoundaryMirrorKernel<<<grid, block>>>
    (
        s->deviceState,
        simulationTime
    );
    updateRiemannBoundaryMirrorKernel<<<grid, block>>>(s->deviceState);
    if (s->hostTurbulenceModel == 0)
    {
        computeGasInternalFaceFluxKernel<false><<<grid, block>>>
        (
            s->deviceState,
            dt
        );
    }
    else
    {
        computeGasInternalFaceFluxKernel<true><<<grid, block>>>
        (
            s->deviceState,
            dt
        );
    }
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<grid, block>>>
        (
            s->deviceState
        );
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstFaceFluxKernel<<<grid, block>>>(s->deviceState);
        if (s->hasPeriodicFaces != 0)
        {
            enforcePeriodicSstFluxAntisymmetryKernel<<<grid, block>>>
            (
                s->deviceState
            );
        }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("ToolB1 gas-face bundle launch", err);
        return 1;
    }
    return 0;
}

float toolB1Median(std::vector<float>& samples)
{
    const size_t middle = samples.size()/2;
    std::nth_element
    (
        samples.begin(),
        samples.begin() + middle,
        samples.end()
    );
    return samples[middle];
}

int tuneFixedWorkBlockThreads
(
    DeviceState* s,
    const double dt,
    const double simulationTime
)
{
    if (s->fixedWorkBlockTuned != 0)
    {
        return 0;
    }

    const int candidates[] = {32, 64, 96, 128, 160, 192, 224, 256};
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t err = cudaEventCreate(&start);
    if (err == cudaSuccess)
    {
        err = cudaEventCreate(&stop);
    }
    if (err != cudaSuccess)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        setLastError("ToolB1 CUDA event creation", err);
        return 1;
    }

    int bestCellBlock = 0;
    int bestFaceBlock = 0;
    float bestCellMs = 0.0f;
    float bestFaceMs = 0.0f;
    float measuredKernelMs = 0.0f;
    for (const int block : candidates)
    {
        if (block > s->hardwareMaxThreadsPerBlock)
        {
            continue;
        }

        int cellBlocksPerSm = 0;
        if
        (
            queryKernelBlocksPerSm
            (
                cellBlocksPerSm,
                "ToolB1 gas-cell occupancy query",
                recoverGasPrimitivesKernel,
                block,
                0
            ) != 0
        )
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return 1;
        }
        for (int run = 0; run < toolB1WarmupRuns; ++run)
        {
            if (launchToolB1CellBundle(s, block) != 0)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                return 1;
            }
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            setLastError("ToolB1 gas-cell warmup", err);
            return 1;
        }
        std::vector<float> samples;
        samples.reserve(toolB1MeasuredRuns);
        for (int run = 0; run < toolB1MeasuredRuns; ++run)
        {
            cudaEventRecord(start);
            if (launchToolB1CellBundle(s, block) != 0)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                return 1;
            }
            cudaEventRecord(stop);
            err = cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            if (err == cudaSuccess)
            {
                err = cudaEventElapsedTime(&elapsed, start, stop);
            }
            if (err != cudaSuccess)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                setLastError("ToolB1 gas-cell measurement", err);
                return 1;
            }
            samples.push_back(elapsed);
            measuredKernelMs += elapsed;
        }
        const float median = toolB1Median(samples);
        if (bestCellBlock == 0 || median < bestCellMs)
        {
            bestCellBlock = block;
            bestCellMs = median;
        }

        int faceBlocksPerSm = 0;
        int occupancyStatus = 0;
        if (s->hostTurbulenceModel == 0)
        {
            occupancyStatus = queryKernelBlocksPerSm
            (
                faceBlocksPerSm,
                "ToolB1 laminar gas-face occupancy query",
                computeGasInternalFaceFluxKernel<false>,
                block,
                0
            );
        }
        else
        {
            occupancyStatus = queryKernelBlocksPerSm
            (
                faceBlocksPerSm,
                "ToolB1 viscous gas-face occupancy query",
                computeGasInternalFaceFluxKernel<true>,
                block,
                0
            );
        }
        if (occupancyStatus != 0)
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return 1;
        }
        for (int run = 0; run < toolB1WarmupRuns; ++run)
        {
            if (launchToolB1FaceBundle(s, block, dt, simulationTime) != 0)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                return 1;
            }
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            setLastError("ToolB1 gas-face warmup", err);
            return 1;
        }
        samples.clear();
        for (int run = 0; run < toolB1MeasuredRuns; ++run)
        {
            cudaEventRecord(start);
            if (launchToolB1FaceBundle(s, block, dt, simulationTime) != 0)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                return 1;
            }
            cudaEventRecord(stop);
            err = cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            if (err == cudaSuccess)
            {
                err = cudaEventElapsedTime(&elapsed, start, stop);
            }
            if (err != cudaSuccess)
            {
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                setLastError("ToolB1 gas-face measurement", err);
                return 1;
            }
            samples.push_back(elapsed);
            measuredKernelMs += elapsed;
        }
        const float faceMedian = toolB1Median(samples);
        if (bestFaceBlock == 0 || faceMedian < bestFaceMs)
        {
            bestFaceBlock = block;
            bestFaceMs = faceMedian;
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (bestCellBlock == 0 || bestFaceBlock == 0)
    {
        setLastErrorText("ToolB1 found no valid measured launch size");
        return 1;
    }
    s->fixedCellBlockThreads = bestCellBlock;
    s->fixedFaceBlockThreads = bestFaceBlock;
    s->fixedWorkBlockTuned = 1;
    std::fprintf
    (
        stderr,
        "ToolB1: B1cell=%d B1face=%d cellMedianMs=%.6f "
        "faceMedianMs=%.6f measuredKernelMs=%.6f warmup=%d repeats=%d\n",
        bestCellBlock,
        bestFaceBlock,
        bestCellMs,
        bestFaceMs,
        measuredKernelMs,
        toolB1WarmupRuns,
        toolB1MeasuredRuns
    );
    return 0;
}

int configureParticleLaunchGeometry(DeviceState* s)
{
    const int particleBlock = s->particleBlockThreads;
    const int reductionBlock = s->reductionBlockThreads;
    cudaError_t err = cudaDeviceGetAttribute
    (
        &s->hardwareMaxThreadsPerBlock,
        cudaDevAttrMaxThreadsPerBlock,
        0
    );
    if (err == cudaSuccess)
    {
        err = cudaDeviceGetAttribute
        (
            &s->hardwareMaxBlocksPerSm,
            cudaDevAttrMaxBlocksPerMultiprocessor,
            0
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("query UGKP hardware launch limits", err);
        return 1;
    }
    if
    (
        particleBlock > s->hardwareMaxThreadsPerBlock
     || reductionBlock > s->hardwareMaxThreadsPerBlock
    )
    {
        setLastErrorText("B2 or B3 exceeds the device thread-block limit");
        return 1;
    }
    const int warpCount = (reductionBlock + 31)/32;
    const size_t poolSharedBytes =
        8u*static_cast<size_t>(reductionBlock)*sizeof(double);
    const size_t componentSharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);

    int gasBlocks = 0;
    int particleBlocks = 0;
    int binBlocks = 0;
    if
    (
        queryKernelBlocksPerSm
        (
            gasBlocks,
            "occupancy query gas internal-face kernel",
            computeGasInternalFaceFluxKernel<true>,
            s->fixedFaceBlockThreads,
            0
        ) != 0
     || queryKernelBlocksPerSm
        (
            particleBlocks,
            "occupancy query particle transport kernel",
            trackParticlesLocalFaceWalkKernel,
            particleBlock,
            0
        ) != 0
     || queryKernelBlocksPerSm
        (
            binBlocks,
            "occupancy query particle binning kernel",
            countParticlesByCellKernel<true>,
            particleBlock,
            0
        ) != 0
    )
    {
        return 1;
    }

    (void)gasBlocks;
    s->particleBlocksPerSm =
        particleBlocks < binBlocks ? particleBlocks : binBlocks;
    s->lightBlocksPerSm = 0;
    s->heavyBlocksPerSm = 0;
    s->csrHeavyWorkerGrid = 0;
    if (s->csrHeavyReductionEnabled != 0)
    {
        int heavyPoolBlocks = 0;
        int heavyMomentBlocks = 0;
        if
        (
            queryKernelBlocksPerSm
            (
                heavyPoolBlocks,
                "occupancy query segmented pool worker kernel",
                accumulateCsrSegmentedPoolTasksPersistentKernel<true>,
                reductionBlock,
                componentSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                heavyMomentBlocks,
                "occupancy query segmented moment worker kernel",
                accumulateCsrSegmentedMomentTasksPersistentKernel,
                reductionBlock,
                componentSharedBytes
            ) != 0
        )
        {
            return 1;
        }
        s->heavyBlocksPerSm =
            heavyPoolBlocks < heavyMomentBlocks
          ? heavyPoolBlocks
          : heavyMomentBlocks;
        s->lightBlocksPerSm = s->heavyBlocksPerSm;
        s->csrHeavyWorkerGrid =
            s->multiprocessorCount*s->heavyBlocksPerSm;
    }
    else
    {
        int lightPoolFull = 0;
        int lightPoolSplit = 0;
        int lightMoments = 0;
        if
        (
            queryKernelBlocksPerSm
            (
                lightPoolFull,
                "occupancy query full light pool kernel",
                accumulatePoissonPoolParticlesByCellKernel<false>,
                reductionBlock,
                poolSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                lightPoolSplit,
                "occupancy query split light pool kernel",
                accumulatePoissonPoolSplitSegmentByCellKernel<true, false, false>,
                reductionBlock,
                poolSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                lightMoments,
                "occupancy query light moment kernel",
                accumulateParticleMomentsSegmentedKernel<false>,
                reductionBlock,
                componentSharedBytes
            ) != 0
        )
        {
            return 1;
        }
        s->lightBlocksPerSm = lightPoolFull;
        if (lightPoolSplit < s->lightBlocksPerSm)
        {
            s->lightBlocksPerSm = lightPoolSplit;
        }
        if (lightMoments < s->lightBlocksPerSm)
        {
            s->lightBlocksPerSm = lightMoments;
        }
    }

    const int capacityGrid =
        (s->particleCapacity + particleBlock - 1)/particleBlock;
    const int saturatedGrid =
        s->multiprocessorCount*s->particleBlocksPerSm;
    s->particleWorkGrid = capacityGrid < saturatedGrid
      ? capacityGrid
      : saturatedGrid;

    std::fprintf
    (
        stderr,
        "Launch geometry: B1cell=%d B1face=%d (ToolB1 pending) "
        "B2=%d B3=%d SM=%d "
        "hardwareMaxThreadsPerBlock=%d hardwareMaxBlocksPerSM=%d "
        "particleBlocksPerSM=%d lightBlocksPerSM=%d "
        "heavyBlocksPerSM=%d\n",
        s->fixedCellBlockThreads,
        s->fixedFaceBlockThreads,
        s->particleBlockThreads,
        s->reductionBlockThreads,
        s->multiprocessorCount,
        s->hardwareMaxThreadsPerBlock,
        s->hardwareMaxBlocksPerSm,
        s->particleBlocksPerSm,
        s->lightBlocksPerSm,
        s->heavyBlocksPerSm
    );
    return syncDeviceState(s, "sync Particle launch geometry");
}

extern "C" int ugkwpGpuResidentStrictCreate
(
    int nCells,
    int nFaces,
    int nInternalFaces,
    int nCellPlanes,
    int particleCapacity,
    int maxFaceWalkHops,
    double injectionParcelMass,
    unsigned long long rngSeed,
    double gammaGas,
    double Rgas,
    double rhoSolid,
    int solveParticleTemperature,
    int particleGasHeatTransferModelId,
    double particleThermalRho,
    double particleCp,
    double gasMu,
    double gasPr,
    double particleDiameterFallback,
    double particleDiameterMin,
    double particleDiameterMax,
    double particleDiameterSigma,
    double injectionTheta,
    double rhoMin,
    double TgasMin,
    double epsSMin,
    double thetaMin,
    double TpMin,
    double TpMax,
    int collisionalPressureEnabled,
    double collisionalRestitution,
    double pressureKickFraction,
    int jammingPressureEnabled,
    double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme,
    int gasReconstruction,
    int gasLimiter,
    int gasTimeIntegrator,
    int gasRobustFallback,
    int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl,
    double waleCw,
    double smagorinskyCs,
    double maxDiffusionNumber,
    int csrCellLocalPathEnabled,
    int csrHeavyReductionMode,
    int csrHeavyAutoInterval,
    int particleBlockThreads,
    int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    int csrSplitPreDirectoryEnabled,
    int dragModel,
    double dragParameter0,
    double dragParameter1,
    double dragParameter2,
    double dragParameter3,
    double gravityX,
    double gravityY,
    double gravityZ,
    void** handle
)
{
    if (handle == nullptr)
    {
        setLastErrorText("null output handle");
        return 1;
    }
    *handle = nullptr;

    if
    (
        nCells <= 0
     || nFaces <= 0
     || nInternalFaces < 0
     || nInternalFaces > nFaces
     || nCellPlanes < 0
     || particleCapacity < 0
     || (particleCapacity > 0
      && (!std::isfinite(injectionParcelMass)
       || injectionParcelMass <= 0.0))
     || !std::isfinite(gammaGas) || gammaGas <= 1.0 || gammaGas > 5.0/3.0
     || !std::isfinite(Rgas) || Rgas <= 0.0
     || !std::isfinite(gasMu) || gasMu < 0.0
     || !std::isfinite(gasPr) || gasPr <= 0.0
     || maxFaceWalkHops <= 0
     || gasFluxScheme < 1
     || gasFluxScheme > 9
     || gasReconstruction < 0
     || gasReconstruction > 2
     || gasLimiter < 0
     || gasLimiter > 2
     || gasTimeIntegrator < 1
     || gasTimeIntegrator > 3
     || (gasRobustFallback != 0 && gasRobustFallback != 1)
     || ((gasFluxScheme == 4
       || gasFluxScheme == 5
       || gasFluxScheme == 6
       || gasFluxScheme == 7
       || gasFluxScheme == 8)
       && gasRobustFallback == 0)
     || turbulenceModel < 0
     || turbulenceModel > 3
     || !std::isfinite(lesDeltaCoeff)
     || lesDeltaCoeff <= 0.0
     || !std::isfinite(turbulentPrandtl)
     || turbulentPrandtl <= 0.0
     || !std::isfinite(waleCw)
     || waleCw < 0.0
     || !std::isfinite(smagorinskyCs)
     || smagorinskyCs < 0.0
     || !std::isfinite(maxDiffusionNumber)
     || maxDiffusionNumber <= 0.0
     || (csrCellLocalPathEnabled != 0 && csrCellLocalPathEnabled != 1)
     || csrHeavyReductionMode < 0
     || csrHeavyReductionMode > 2
     || csrHeavyAutoInterval < 1
     || (particleBlockThreads != 32 && particleBlockThreads != 64
      && particleBlockThreads != 128 && particleBlockThreads != 256)
     || (reductionBlockThreads != 32 && reductionBlockThreads != 64
      && reductionBlockThreads != 128 && reductionBlockThreads != 256)
     || (csrWarpAggregatedBinning != 0 && csrWarpAggregatedBinning != 1)
     || (csrSplitPreDirectoryEnabled != 0 && csrSplitPreDirectoryEnabled != 1)
     || dragModel < 0
     || dragModel > 2
     || particleGasHeatTransferModelId < 0
     || particleGasHeatTransferModelId > 1
     || !std::isfinite(dragParameter0)
     || !std::isfinite(dragParameter1)
     || !std::isfinite(dragParameter2)
     || !std::isfinite(dragParameter3)
     || (dragModel == 1 && dragParameter0 <= 0.0)
     || !std::isfinite(gravityX)
     || !std::isfinite(gravityY)
     || !std::isfinite(gravityZ)
     || (jammingPressureEnabled != 0 && jammingPressureEnabled != 1)
     || packingProjectionIterations < 1
     || (!csrCellLocalPathEnabled
      && (csrHeavyReductionMode != 0 || csrWarpAggregatedBinning))
    )
    {
        setLastErrorText("invalid GPU resident strict sizes");
        return 1;
    }

    if
    (
        jammingPressureEnabled != 0
     &&
        (
            !std::isfinite(packingFraction)
         || packingFraction <= 0.0
         || packingFraction >= 1.0
        )
    )
    {
        setLastErrorText("invalid GPU resident mobile packing-projection parameters");
        return 1;
    }

    cudaError_t err = cudaFree(nullptr);
    if (err != cudaSuccess)
    {
        setLastError("cudaFree(nullptr) strict warmup", err);
        return 1;
    }

    DeviceState* s = new DeviceState;
    s->nCells = nCells;
    s->nFaces = nFaces;
    s->nInternalFaces = nInternalFaces;
    s->nCellPlanes = nCellPlanes;
    s->particleCapacity = particleCapacity;
    s->maxFaceWalkHops = maxFaceWalkHops;
    s->injectionParcelMass = injectionParcelMass;
    s->rngSeed = rngSeed;
    s->gammaGas = gammaGas;
    s->Rgas = Rgas;
    const double gammaMinusOne = gammaGas - 1.0;
    const double gammaCpDenominator =
        gammaMinusOne < 1.0e-12 ? 1.0e-12 : gammaMinusOne;
    s->gasCp = gammaGas*Rgas/gammaCpDenominator;
    s->rhoSolid = rhoSolid;
    const double rhoSolidDenominator =
        rhoSolid < 1.0e-300 ? 1.0e-300 : rhoSolid;
    s->invRhoSolid = 1.0/rhoSolidDenominator;
    s->solveParticleTemperature = solveParticleTemperature;
    s->particleGasHeatTransferModelId = particleGasHeatTransferModelId;
    s->particleThermalRho = particleThermalRho;
    s->particleCp = particleCp;
    s->particleThermalCapacity = particleThermalRho*particleCp;
    s->gasMu = gasMu;
    s->gasPr = gasPr;
    s->gasPrClamped = gasPr < 1.0e-12 ? 1.0e-12 : gasPr;
    s->gasPrOneThird = std::pow(s->gasPrClamped, 1.0/3.0);
    s->dragModel = dragModel;
    s->dragParameter0 = dragParameter0;
    s->dragParameter1 = dragParameter1;
    s->dragParameter2 = dragParameter2;
    s->dragParameter3 = dragParameter3;
    s->gravityX = gravityX;
    s->gravityY = gravityY;
    s->gravityZ = gravityZ;
    s->gasFluxScheme = gasFluxScheme;
    s->gasReconstruction = gasReconstruction;
    s->gasLimiter = gasLimiter;
    s->gasTimeIntegrator = gasTimeIntegrator;
    s->gasRobustFallback = gasRobustFallback;
    s->turbulenceModel = turbulenceModel;
    s->hostGasFluxScheme = gasFluxScheme;
    s->hostGasTimeIntegrator = gasTimeIntegrator;
    s->hostTurbulenceModel = turbulenceModel;
    s->hostDragModel = dragModel;
    s->hostGravityActive =
        gravityX != 0.0 || gravityY != 0.0 || gravityZ != 0.0;
    s->lesDeltaCoeff = lesDeltaCoeff;
    s->turbulentPrandtl = turbulentPrandtl;
    s->waleCw = waleCw;
    s->smagorinskyCs = smagorinskyCs;
    s->maxDiffusionNumber = maxDiffusionNumber;
    s->csrCellLocalPathEnabled = csrCellLocalPathEnabled;
    s->csrHeavyReductionMode = csrHeavyReductionMode;
    s->csrHeavyAutoInterval = csrHeavyAutoInterval;
    s->csrHeavyReductionEnabled = csrHeavyReductionMode != 0 ? 1 : 0;
    s->csrHeavyReductionActive = csrHeavyReductionMode == 1 ? 1 : 0;
    s->particleBlockThreads = particleBlockThreads;
    s->reductionBlockThreads = reductionBlockThreads;
    s->csrWarpAggregatedBinning = csrWarpAggregatedBinning;
    s->csrSplitPreDirectoryEnabled = csrSplitPreDirectoryEnabled;
    s->particleDiameterFallback = particleDiameterFallback;
    s->particleDiameterMin = particleDiameterMin;
    s->particleDiameterMax = particleDiameterMax;
    s->particleDiameterSigma = particleDiameterSigma;
    s->injectionTheta = injectionTheta;
    s->rhoMin = rhoMin;
    s->TgasMin = TgasMin;
    s->epsSMin = epsSMin;
    s->thetaMin = thetaMin;
    s->TpMin = TpMin;
    s->TpMax = TpMax;
    s->collisionalPressureEnabled = collisionalPressureEnabled != 0 ? 1 : 0;
    s->collisionalRestitution =
        std::fmin(std::fmax(collisionalRestitution, 0.0), 1.0);
    s->pressureKickFraction =
        std::fmin(std::fmax(pressureKickFraction, OfSmall), 1.0);
    s->jammingPressureEnabled = jammingPressureEnabled != 0 ? 1 : 0;
    s->packingFraction = packingFraction;
    s->packingProjectionIterations = packingProjectionIterations;

    if (allocateFields(s) != 0)
    {
        return 1;
    }

    if (configureParticleLaunchGeometry(s) != 0)
    {
        releaseState(s);
        return 1;
    }

#ifdef UGKP_DEVELOPMENT_PROBES
    if (initialiseDevelopmentProbe(s) != 0)
    {
        releaseState(s);
        return 1;
    }
#endif

    *handle = s;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictUploadMesh
(
    void* handle,
    const int* faceOwner,
    const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx,
    const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V,
    const double* Cx,
    const double* Cy,
    const double* Cz,
    const double* faceCx,
    const double* faceCy,
    const double* faceCz,
    const double* Sfx,
    const double* Sfy,
    const double* Sfz,
    const double* magSf,
    const double* deltaCoeffs,
    const double* faceWeight,
    const double* cellLength,
    const int* cellPlaneStart,
    const int* cellPlaneCount,
    const int* cellFaceId,
    const int* cellFaceNeighbor,
    const int* cellFaceKind,
    const double* cellFaceRestitution,
    const double* cellFaceTangential,
    const double* planeNx,
    const double* planeNy,
    const double* planeNz,
    const double* planeD
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "mesh upload") != 0)
    {
        return 1;
    }

    int rc = 0;
    const size_t nc = static_cast<size_t>(s->nCells);
    const size_t nf = static_cast<size_t>(s->nFaces);
    const size_t nPlanes = static_cast<size_t>(s->nCellPlanes);

    if
    (
        cellFaceId == nullptr
     || cellFaceNeighbor == nullptr
     || cellFaceKind == nullptr
     || facePeriodicPair == nullptr
     || facePeriodicDx == nullptr
     || facePeriodicDy == nullptr
     || facePeriodicDz == nullptr
    )
    {
        setLastErrorText("null cell-face topology in mesh upload");
        return 1;
    }

    s->hasPeriodicFaces = 0;
    for (int f = 0; f < s->nFaces; ++f)
    {
        const int pair = facePeriodicPair[f];
        if (pair < 0)
        {
            continue;
        }
        if
        (
            f < s->nInternalFaces
         || pair < s->nInternalFaces
         || pair >= s->nFaces
         || pair == f
         || facePeriodicPair[pair] != f
         || faceNeighbour[f] < 0
         || faceNeighbour[f] >= s->nCells
         || !std::isfinite(facePeriodicDx[f])
         || !std::isfinite(facePeriodicDy[f])
         || !std::isfinite(facePeriodicDz[f])
        )
        {
            setLastErrorText("invalid serial translational cyclic mesh pair");
            return 1;
        }
        s->hasPeriodicFaces = 1;
    }

    rc |= copyToDevice(s->faceOwner, faceOwner, nf, "cudaMemcpy strict faceOwner");
    rc |= copyToDevice(s->faceNeighbour, faceNeighbour, nf, "cudaMemcpy strict faceNeighbour");
    rc |= copyToDevice(s->facePeriodicPair, facePeriodicPair, nf, "cudaMemcpy strict facePeriodicPair");
    rc |= copyToDevice(s->facePeriodicDx, facePeriodicDx, nf, "cudaMemcpy strict facePeriodicDx");
    rc |= copyToDevice(s->facePeriodicDy, facePeriodicDy, nf, "cudaMemcpy strict facePeriodicDy");
    rc |= copyToDevice(s->facePeriodicDz, facePeriodicDz, nf, "cudaMemcpy strict facePeriodicDz");
    rc |= copyToDevice(s->V, V, nc, "cudaMemcpy strict V");
    rc |= copyToDevice(s->Cx, Cx, nc, "cudaMemcpy strict Cx");
    rc |= copyToDevice(s->Cy, Cy, nc, "cudaMemcpy strict Cy");
    rc |= copyToDevice(s->Cz, Cz, nc, "cudaMemcpy strict Cz");
    rc |= copyToDevice(s->faceCx, faceCx, nf, "cudaMemcpy strict faceCx");
    rc |= copyToDevice(s->faceCy, faceCy, nf, "cudaMemcpy strict faceCy");
    rc |= copyToDevice(s->faceCz, faceCz, nf, "cudaMemcpy strict faceCz");
    rc |= copyToDevice(s->Sfx, Sfx, nf, "cudaMemcpy strict Sfx");
    rc |= copyToDevice(s->Sfy, Sfy, nf, "cudaMemcpy strict Sfy");
    rc |= copyToDevice(s->Sfz, Sfz, nf, "cudaMemcpy strict Sfz");
    rc |= copyToDevice(s->magSf, magSf, nf, "cudaMemcpy strict magSf");
    rc |= copyToDevice(s->deltaCoeffs, deltaCoeffs, nf, "cudaMemcpy strict deltaCoeffs");
    rc |= copyToDevice(s->faceWeight, faceWeight, nf, "cudaMemcpy strict faceWeight");
    rc |= copyToDevice(s->cellLength, cellLength, nc, "cudaMemcpy strict cellLength");
    rc |= copyToDevice(s->cellPlaneStart, cellPlaneStart, nc, "cudaMemcpy strict cellPlaneStart");
    rc |= copyToDevice(s->cellPlaneCount, cellPlaneCount, nc, "cudaMemcpy strict cellPlaneCount");
    rc |= copyToDevice(s->cellFaceId, cellFaceId, nPlanes, "cudaMemcpy strict cellFaceId");
    rc |= copyToDevice(s->cellFaceNeighbor, cellFaceNeighbor, nPlanes, "cudaMemcpy strict cellFaceNeighbor");
    rc |= copyToDevice(s->cellFaceKind, cellFaceKind, nPlanes, "cudaMemcpy strict cellFaceKind");
    rc |= copyToDevice(s->cellFaceRestitution, cellFaceRestitution, nPlanes, "cudaMemcpy strict cellFaceRestitution");
    rc |= copyToDevice(s->cellFaceTangential, cellFaceTangential, nPlanes, "cudaMemcpy strict cellFaceTangential");
    rc |= copyToDevice(s->planeNx, planeNx, nPlanes, "cudaMemcpy strict planeNx");
    rc |= copyToDevice(s->planeNy, planeNy, nPlanes, "cudaMemcpy strict planeNy");
    rc |= copyToDevice(s->planeNz, planeNz, nPlanes, "cudaMemcpy strict planeNz");
    rc |= copyToDevice(s->planeD, planeD, nPlanes, "cudaMemcpy strict planeD");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictUploadBoundarySources
(
    void* handle,
    int nBoundarySources,
    const int* sourceCell,
    const int* sourceFace,
    const double* sourcePx,
    const double* sourcePy,
    const double* sourcePz,
    const double* sourceUx,
    const double* sourceUy,
    const double* sourceUz,
    const double* sourceT,
    const double* sourceTheta,
    const double* sourceD,
    const double* sourceMassRate
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "boundary source upload") != 0)
    {
        return 1;
    }

    if (nBoundarySources < 0)
    {
        setLastErrorText("negative GPU resident boundary source count");
        return 1;
    }

    release(s->sourceCell);
    release(s->sourceFace);
    release(s->sourcePx);
    release(s->sourcePy);
    release(s->sourcePz);
    release(s->sourceUx);
    release(s->sourceUy);
    release(s->sourceUz);
    release(s->sourceT);
    release(s->sourceTheta);
    release(s->sourceD);
    release(s->sourceMassRate);
    release(s->sourceResidualMass);
#ifdef UGKP_DEVELOPMENT_PROBES
    release(s->sourceInjectedCount);
#endif
    s->nBoundarySources = 0;

    const size_t n = static_cast<size_t>(nBoundarySources);
    if (n == 0)
    {
        return syncDeviceState(s, "cudaMemcpy strict deviceState after empty boundary source upload");
    }

    if
    (
        sourceCell == nullptr
     || sourceFace == nullptr
     || sourcePx == nullptr
     || sourcePy == nullptr
     || sourcePz == nullptr
     || sourceUx == nullptr
     || sourceUy == nullptr
     || sourceUz == nullptr
     || sourceT == nullptr
     || sourceTheta == nullptr
     || sourceD == nullptr
     || sourceMassRate == nullptr
    )
    {
        setLastErrorText("null GPU resident boundary source array");
        return 1;
    }

    int rc = 0;
    rc |= allocate(s->sourceCell, n, "cudaMalloc strict sourceCell");
    rc |= allocate(s->sourceFace, n, "cudaMalloc strict sourceFace");
    rc |= allocate(s->sourcePx, n, "cudaMalloc strict sourcePx");
    rc |= allocate(s->sourcePy, n, "cudaMalloc strict sourcePy");
    rc |= allocate(s->sourcePz, n, "cudaMalloc strict sourcePz");
    rc |= allocate(s->sourceUx, n, "cudaMalloc strict sourceUx");
    rc |= allocate(s->sourceUy, n, "cudaMalloc strict sourceUy");
    rc |= allocate(s->sourceUz, n, "cudaMalloc strict sourceUz");
    rc |= allocate(s->sourceT, n, "cudaMalloc strict sourceT");
    rc |= allocate(s->sourceTheta, n, "cudaMalloc strict sourceTheta");
    rc |= allocate(s->sourceD, n, "cudaMalloc strict sourceD");
    rc |= allocate(s->sourceMassRate, n, "cudaMalloc strict sourceMassRate");
    rc |= allocate(s->sourceResidualMass, n, "cudaMalloc strict sourceResidualMass");
#ifdef UGKP_DEVELOPMENT_PROBES
    rc |= allocate
    (
        s->sourceInjectedCount,
        n,
        "cudaMalloc diagnostic sourceInjectedCount"
    );
#endif
    if (rc != 0)
    {
        return 1;
    }

    rc |= copyToDevice(s->sourceCell, sourceCell, n, "cudaMemcpy strict sourceCell");
    rc |= copyToDevice(s->sourceFace, sourceFace, n, "cudaMemcpy strict sourceFace");
    rc |= copyToDevice(s->sourcePx, sourcePx, n, "cudaMemcpy strict sourcePx");
    rc |= copyToDevice(s->sourcePy, sourcePy, n, "cudaMemcpy strict sourcePy");
    rc |= copyToDevice(s->sourcePz, sourcePz, n, "cudaMemcpy strict sourcePz");
    rc |= copyToDevice(s->sourceUx, sourceUx, n, "cudaMemcpy strict sourceUx");
    rc |= copyToDevice(s->sourceUy, sourceUy, n, "cudaMemcpy strict sourceUy");
    rc |= copyToDevice(s->sourceUz, sourceUz, n, "cudaMemcpy strict sourceUz");
    rc |= copyToDevice(s->sourceT, sourceT, n, "cudaMemcpy strict sourceT");
    rc |= copyToDevice(s->sourceTheta, sourceTheta, n, "cudaMemcpy strict sourceTheta");
    rc |= copyToDevice(s->sourceD, sourceD, n, "cudaMemcpy strict sourceD");
    rc |= copyToDevice(s->sourceMassRate, sourceMassRate, n, "cudaMemcpy strict sourceMassRate");
    if (rc != 0)
    {
        return 1;
    }
    for (int i = 0; i < nBoundarySources; ++i)
    {
        if (std::isfinite(sourceMassRate[i]) && sourceMassRate[i] > 0.0)
        {
            s->particlesMayBePresent = true;
            break;
        }
    }

    cudaError_t err = cudaMemset(s->sourceResidualMass, 0, n*sizeof(double));
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict sourceResidualMass", err);
        return 1;
    }
#ifdef UGKP_DEVELOPMENT_PROBES
    err = cudaMemset(s->sourceInjectedCount, 0, n*sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset diagnostic sourceInjectedCount", err);
        return 1;
    }
#endif

    s->nBoundarySources = nBoundarySources;
    return syncDeviceState(s, "cudaMemcpy strict deviceState after boundary source upload");
}

extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet
(
    void* handle,
    int nFaces,
    const int* faceIds,
    double inletTemperature,
    int nPressureRows,
    const double* pressureTimes,
    const double* pressureValues,
    int nVolumeFractionRows,
    const double* volumeFractionTimes,
    const double* volumeFractionValues
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "scheduled inlet configuration") != 0)
    {
        return 1;
    }
    if
    (
        nFaces <= 0
     || faceIds == nullptr
     || !std::isfinite(inletTemperature)
     || inletTemperature <= 0.0
     || nPressureRows <= 0
     || pressureTimes == nullptr
     || pressureValues == nullptr
     || nVolumeFractionRows <= 0
     || volumeFractionTimes == nullptr
     || volumeFractionValues == nullptr
    )
    {
        setLastErrorText("invalid scheduled inlet configuration arguments");
        return 1;
    }
    std::vector<int> faceMask(static_cast<size_t>(s->nFaces), 0);
    for (int i = 0; i < nFaces; ++i)
    {
        if (faceIds[i] < s->nInternalFaces || faceIds[i] >= s->nFaces)
        {
            setLastErrorText("scheduled inlet face is not a boundary face");
            return 1;
        }
        faceMask[static_cast<size_t>(faceIds[i])] = 1;
    }
    for (int i = 0; i < nPressureRows; ++i)
    {
        if
        (
            !std::isfinite(pressureTimes[i])
         || !std::isfinite(pressureValues[i])
         || pressureTimes[i] < 0.0
         || pressureValues[i] <= 0.0
         || (i > 0 && pressureTimes[i] <= pressureTimes[i - 1])
        )
        {
            setLastErrorText("invalid scheduled inlet pressure table");
            return 1;
        }
    }
    bool futureParticleInflow = false;
    for (int i = 0; i < nVolumeFractionRows; ++i)
    {
        if
        (
            !std::isfinite(volumeFractionTimes[i])
         || !std::isfinite(volumeFractionValues[i])
         || volumeFractionTimes[i] < 0.0
         || volumeFractionValues[i] < 0.0
         || volumeFractionValues[i] >= 1.0
         || (i > 0 && volumeFractionTimes[i] <= volumeFractionTimes[i - 1])
        )
        {
            setLastErrorText("invalid scheduled inlet volume-fraction table");
            return 1;
        }
        futureParticleInflow =
            futureParticleInflow || volumeFractionValues[i] > 0.0;
    }

    release(s->scheduledInletFaceMask);
    release(s->pressureScheduleTimes);
    release(s->pressureScheduleValues);
    release(s->volumeFractionScheduleTimes);
    release(s->volumeFractionScheduleValues);
    int rc = 0;
    rc |= allocate
    (
        s->scheduledInletFaceMask,
        static_cast<size_t>(s->nFaces),
        "cudaMalloc strict scheduled inlet face mask"
    );
    rc |= allocate
    (
        s->pressureScheduleTimes,
        static_cast<size_t>(nPressureRows),
        "cudaMalloc strict pressure schedule times"
    );
    rc |= allocate
    (
        s->pressureScheduleValues,
        static_cast<size_t>(nPressureRows),
        "cudaMalloc strict pressure schedule values"
    );
    rc |= allocate
    (
        s->volumeFractionScheduleTimes,
        static_cast<size_t>(nVolumeFractionRows),
        "cudaMalloc strict volume-fraction schedule times"
    );
    rc |= allocate
    (
        s->volumeFractionScheduleValues,
        static_cast<size_t>(nVolumeFractionRows),
        "cudaMalloc strict volume-fraction schedule values"
    );
    if (rc != 0)
    {
        return 1;
    }
    rc |= copyToDevice
    (
        s->scheduledInletFaceMask,
        faceMask.data(),
        static_cast<size_t>(s->nFaces),
        "cudaMemcpy strict scheduled inlet face mask"
    );
    rc |= copyToDevice
    (
        s->pressureScheduleTimes,
        pressureTimes,
        static_cast<size_t>(nPressureRows),
        "cudaMemcpy strict pressure schedule times"
    );
    rc |= copyToDevice
    (
        s->pressureScheduleValues,
        pressureValues,
        static_cast<size_t>(nPressureRows),
        "cudaMemcpy strict pressure schedule values"
    );
    rc |= copyToDevice
    (
        s->volumeFractionScheduleTimes,
        volumeFractionTimes,
        static_cast<size_t>(nVolumeFractionRows),
        "cudaMemcpy strict volume-fraction schedule times"
    );
    rc |= copyToDevice
    (
        s->volumeFractionScheduleValues,
        volumeFractionValues,
        static_cast<size_t>(nVolumeFractionRows),
        "cudaMemcpy strict volume-fraction schedule values"
    );
    if (rc != 0)
    {
        return 1;
    }
    s->nScheduledInletFaces = nFaces;
    s->scheduledInletTemperature = inletTemperature;
    s->nPressureScheduleRows = nPressureRows;
    s->nVolumeFractionScheduleRows = nVolumeFractionRows;
    if (futureParticleInflow && s->nBoundarySources > 0)
    {
        s->particlesMayBePresent = true;
    }
    publishScheduledInletConfigurationKernel<<<1, 1>>>
    (
        s->deviceState,
        nFaces,
        s->scheduledInletFaceMask,
        inletTemperature,
        nPressureRows,
        s->pressureScheduleTimes,
        s->pressureScheduleValues,
        nVolumeFractionRows,
        s->volumeFractionScheduleTimes,
        s->volumeFractionScheduleValues,
        futureParticleInflow && s->nBoundarySources > 0 ? 1 : 0
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("publish scheduled inlet configuration", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError("synchronize scheduled inlet configuration", err);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass
(
    void* handle,
    int* nSources,
    int maxSources,
    int* sourceFace,
    double* residualMass
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "source residual download") != 0 || nSources == nullptr)
    {
        return 1;
    }
    *nSources = s->nBoundarySources;
    if (maxSources == 0)
    {
        return 0;
    }
    if
    (
        maxSources < s->nBoundarySources
     || sourceFace == nullptr
     || residualMass == nullptr
    )
    {
        setLastErrorText("invalid source residual download buffers");
        return 1;
    }
    const size_t n = static_cast<size_t>(s->nBoundarySources);
    int rc = 0;
    rc |= copyToHost(sourceFace, s->sourceFace, n, "cudaMemcpy strict sourceFace restart");
    rc |= copyToHost(residualMass, s->sourceResidualMass, n, "cudaMemcpy strict source residual restart");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass
(
    void* handle,
    int nSources,
    const int* sourceFace,
    const double* residualMass
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "source residual upload") != 0)
    {
        return 1;
    }
    if
    (
        nSources != s->nBoundarySources
     || (nSources > 0 && (sourceFace == nullptr || residualMass == nullptr))
    )
    {
        setLastErrorText("source residual restart count/buffer mismatch");
        return 1;
    }

    std::vector<int> currentFaces(static_cast<size_t>(nSources));
    if
    (
        copyToHost
        (
            currentFaces.data(),
            s->sourceFace,
            currentFaces.size(),
            "cudaMemcpy strict sourceFace validation"
        ) != 0
    )
    {
        return 1;
    }
    for (int i = 0; i < nSources; ++i)
    {
        if
        (
            currentFaces[static_cast<size_t>(i)] != sourceFace[i]
         || !std::isfinite(residualMass[i])
         || residualMass[i] < 0.0
        )
        {
            setLastErrorText("source residual restart face/value mismatch");
            return 1;
        }
    }
    return copyToDevice
    (
        s->sourceResidualMass,
        residualMass,
        static_cast<size_t>(nSources),
        "cudaMemcpy strict source residual upload"
    );
}

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields
(
    void* handle,
    const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix,
    const int* gasBoundaryUFix,
    const int* gasBoundaryPFix,
    const int* gasBoundaryTFix,
    const int* gasBoundaryPWave,
    const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf,
    const double* gasBoundaryRho,
    const double* gasBoundaryUx,
    const double* gasBoundaryUy,
    const double* gasBoundaryUz,
    const double* gasBoundaryP,
    const double* gasBoundaryT
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "gas boundary upload") != 0)
    {
        return 1;
    }

    if
    (
        gasBoundaryKind == nullptr
     || gasBoundaryRhoFix == nullptr
     || gasBoundaryUFix == nullptr
     || gasBoundaryPFix == nullptr
     || gasBoundaryTFix == nullptr
     || gasBoundaryPWave == nullptr
     || gasBoundaryPWaveGamma == nullptr
     || gasBoundaryPWaveFieldInf == nullptr
     || gasBoundaryPWaveLInf == nullptr
     || gasBoundaryRho == nullptr
     || gasBoundaryUx == nullptr
     || gasBoundaryUy == nullptr
     || gasBoundaryUz == nullptr
     || gasBoundaryP == nullptr
     || gasBoundaryT == nullptr
    )
    {
        setLastErrorText("null GPU resident gas boundary array");
        return 1;
    }

    const size_t nf = static_cast<size_t>(s->nFaces);
    int rc = 0;
    rc |= copyToDevice(s->gasBoundaryKind, gasBoundaryKind, nf, "cudaMemcpy strict gasBoundaryKind");
    rc |= copyToDevice(s->gasBoundaryRhoFix, gasBoundaryRhoFix, nf, "cudaMemcpy strict gasBoundaryRhoFix");
    rc |= copyToDevice(s->gasBoundaryUFix, gasBoundaryUFix, nf, "cudaMemcpy strict gasBoundaryUFix");
    rc |= copyToDevice(s->gasBoundaryPFix, gasBoundaryPFix, nf, "cudaMemcpy strict gasBoundaryPFix");
    rc |= copyToDevice(s->gasBoundaryTFix, gasBoundaryTFix, nf, "cudaMemcpy strict gasBoundaryTFix");
    rc |= copyToDevice(s->gasBoundaryPWave, gasBoundaryPWave, nf, "cudaMemcpy strict gasBoundaryPWave");
    rc |= copyToDevice(s->gasBoundaryPWaveGamma, gasBoundaryPWaveGamma, nf, "cudaMemcpy strict gasBoundaryPWaveGamma");
    rc |= copyToDevice(s->gasBoundaryPWaveFieldInf, gasBoundaryPWaveFieldInf, nf, "cudaMemcpy strict gasBoundaryPWaveFieldInf");
    rc |= copyToDevice(s->gasBoundaryPWaveLInf, gasBoundaryPWaveLInf, nf, "cudaMemcpy strict gasBoundaryPWaveLInf");
    rc |= copyToDevice(s->gasBoundaryRho, gasBoundaryRho, nf, "cudaMemcpy strict gasBoundaryRho");
    rc |= copyToDevice(s->gasBoundaryUx, gasBoundaryUx, nf, "cudaMemcpy strict gasBoundaryUx");
    rc |= copyToDevice(s->gasBoundaryUy, gasBoundaryUy, nf, "cudaMemcpy strict gasBoundaryUy");
    rc |= copyToDevice(s->gasBoundaryUz, gasBoundaryUz, nf, "cudaMemcpy strict gasBoundaryUz");
    rc |= copyToDevice(s->gasBoundaryP, gasBoundaryP, nf, "cudaMemcpy strict gasBoundaryP");
    rc |= copyToDevice(s->gasBoundaryT, gasBoundaryT, nf, "cudaMemcpy strict gasBoundaryT");
    rc |= copyToDevice(s->riemannBoundaryKind, gasBoundaryKind, nf, "cudaMemcpy strict riemannBoundaryKind");
    rc |= copyToDevice(s->riemannBoundaryRhoFix, gasBoundaryRhoFix, nf, "cudaMemcpy strict riemannBoundaryRhoFix");
    rc |= copyToDevice(s->riemannBoundaryUFix, gasBoundaryUFix, nf, "cudaMemcpy strict riemannBoundaryUFix");
    rc |= copyToDevice(s->riemannBoundaryPFix, gasBoundaryPFix, nf, "cudaMemcpy strict riemannBoundaryPFix");
    rc |= copyToDevice(s->riemannBoundaryTFix, gasBoundaryTFix, nf, "cudaMemcpy strict riemannBoundaryTFix");
    rc |= copyToDevice(s->riemannBoundaryPWave, gasBoundaryPWave, nf, "cudaMemcpy strict riemannBoundaryPWave");
    rc |= copyToDevice(s->riemannBoundaryPWaveGamma, gasBoundaryPWaveGamma, nf, "cudaMemcpy strict riemannBoundaryPWaveGamma");
    rc |= copyToDevice(s->riemannBoundaryPWaveFieldInf, gasBoundaryPWaveFieldInf, nf, "cudaMemcpy strict riemannBoundaryPWaveFieldInf");
    rc |= copyToDevice(s->riemannBoundaryPWaveLInf, gasBoundaryPWaveLInf, nf, "cudaMemcpy strict riemannBoundaryPWaveLInf");
    rc |= copyToDevice(s->riemannBoundaryRho, gasBoundaryRho, nf, "cudaMemcpy strict riemannBoundaryRho");
    rc |= copyToDevice(s->riemannBoundaryUx, gasBoundaryUx, nf, "cudaMemcpy strict riemannBoundaryUx");
    rc |= copyToDevice(s->riemannBoundaryUy, gasBoundaryUy, nf, "cudaMemcpy strict riemannBoundaryUy");
    rc |= copyToDevice(s->riemannBoundaryUz, gasBoundaryUz, nf, "cudaMemcpy strict riemannBoundaryUz");
    rc |= copyToDevice(s->riemannBoundaryP, gasBoundaryP, nf, "cudaMemcpy strict riemannBoundaryP");
    rc |= copyToDevice(s->riemannBoundaryT, gasBoundaryT, nf, "cudaMemcpy strict riemannBoundaryT");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictUploadFields
(
    void* handle,
    const double* rho,
    const double* rhoUx,
    const double* rhoUy,
    const double* rhoUz,
    const double* rhoE,
    const double* Ux,
    const double* Uy,
    const double* Uz,
    const double* p,
    const double* Tgas
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "field upload") != 0)
    {
        return 1;
    }

    const size_t n = static_cast<size_t>(s->nCells);
    int rc = 0;

    rc |= copyToDevice(s->rho, rho, n, "cudaMemcpy strict rho");
    rc |= copyToDevice(s->rhoUx, rhoUx, n, "cudaMemcpy strict rhoUx");
    rc |= copyToDevice(s->rhoUy, rhoUy, n, "cudaMemcpy strict rhoUy");
    rc |= copyToDevice(s->rhoUz, rhoUz, n, "cudaMemcpy strict rhoUz");
    rc |= copyToDevice(s->rhoE, rhoE, n, "cudaMemcpy strict rhoE");
    rc |= copyToDevice(s->Ux, Ux, n, "cudaMemcpy strict Ux");
    rc |= copyToDevice(s->Uy, Uy, n, "cudaMemcpy strict Uy");
    rc |= copyToDevice(s->Uz, Uz, n, "cudaMemcpy strict Uz");
    rc |= copyToDevice(s->p, p, n, "cudaMemcpy strict p");
    rc |= copyToDevice(s->Tgas, Tgas, n, "cudaMemcpy strict Tgas");
    if (rc != 0)
    {
        return 1;
    }

    cudaError_t clearErr = cudaSuccess;

#define CLEAR_SOLID_ARRAY(ptr, name)                                      \
    clearErr = cudaMemset((ptr), 0, n*sizeof(double));                    \
    if (clearErr != cudaSuccess)                                          \
    {                                                                     \
        setLastError((name), clearErr);                                   \
        return 1;                                                         \
    }

    CLEAR_SOLID_ARRAY(s->epsS, "cudaMemset strict epsS");
    CLEAR_SOLID_ARRAY(s->rhoUsx, "cudaMemset strict rhoUsx");
    CLEAR_SOLID_ARRAY(s->rhoUsy, "cudaMemset strict rhoUsy");
    CLEAR_SOLID_ARRAY(s->rhoUsz, "cudaMemset strict rhoUsz");
    CLEAR_SOLID_ARRAY(s->rhoEs, "cudaMemset strict rhoEs");
    CLEAR_SOLID_ARRAY(s->rhoDs, "cudaMemset strict rhoDs");
    CLEAR_SOLID_ARRAY(s->rhoHp, "cudaMemset strict rhoHp");
    CLEAR_SOLID_ARRAY(s->Usx, "cudaMemset strict Usx");
    CLEAR_SOLID_ARRAY(s->Usy, "cudaMemset strict Usy");
    CLEAR_SOLID_ARRAY(s->Usz, "cudaMemset strict Usz");
    CLEAR_SOLID_ARRAY(s->theta, "cudaMemset strict theta");
    CLEAR_SOLID_ARRAY(s->Tp, "cudaMemset strict Tp");
    CLEAR_SOLID_ARRAY(s->dMeanCell, "cudaMemset strict dMeanCell");
    CLEAR_SOLID_ARRAY(s->momRhoP, "cudaMemset strict momRhoP");
    CLEAR_SOLID_ARRAY(s->momRhoUPx, "cudaMemset strict momRhoUPx");
    CLEAR_SOLID_ARRAY(s->momRhoUPy, "cudaMemset strict momRhoUPy");
    CLEAR_SOLID_ARRAY(s->momRhoUPz, "cudaMemset strict momRhoUPz");
    CLEAR_SOLID_ARRAY(s->momRhoEP, "cudaMemset strict momRhoEP");
    CLEAR_SOLID_ARRAY(s->momRhoPD, "cudaMemset strict momRhoPD");
    CLEAR_SOLID_ARRAY(s->momRhoHpP, "cudaMemset strict momRhoHpP");

#undef CLEAR_SOLID_ARRAY

    {
        const int block = s->fixedCellBlockThreads;
        const int grid = (s->nCells + block - 1)/block;
        initialiseEpsGPrevKernel<<<grid, block>>>(s->deviceState);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("initialiseEpsGPrevKernel launch", err);
            return 1;
        }
        initialiseThetaDragAlphaKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("initialiseThetaDragAlphaKernel launch", err);
            return 1;
        }
    }

    if (s->solveParticleTemperature != 0)
    {
        const int block = s->fixedCellBlockThreads;
        const int grid = (s->nCells + block - 1)/block;
        initialiseParticleMaterialEnthalpyKernel<<<grid, block>>>(s->deviceState);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("initialiseParticleMaterialEnthalpyKernel launch", err);
            return 1;
        }
    }

    scrubHostCalculationScalars(s);
    return 0;
}

extern "C" int ugkwpGpuResidentStrictConfigureSst
(
    void* handle,
    double alphaK1,
    double alphaK2,
    double alphaOmega1,
    double alphaOmega2,
    double beta1,
    double beta2,
    double betaStar,
    double gamma1,
    double gamma2,
    double a1,
    double b1,
    double c1,
    double kMin,
    double omegaMin,
    double maxSourceNumber,
    int wallTreatment,
    double wallKappa,
    double wallE,
    double wallCmu,
    const double* k,
    const double* omega,
    const double* wallDistance,
    const int* boundaryKMode,
    const int* boundaryOmegaMode,
    const double* boundaryK,
    const double* boundaryOmega
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "configure SST") != 0)
    {
        return 1;
    }
    if (s->hostTurbulenceModel != 3)
    {
        setLastErrorText("SST configuration requires turbulenceModel=3");
        return 1;
    }
    if
    (
        k == nullptr
     || omega == nullptr
     || wallDistance == nullptr
     || boundaryKMode == nullptr
     || boundaryOmegaMode == nullptr
     || boundaryK == nullptr
     || boundaryOmega == nullptr
    )
    {
        setLastErrorText("null SST configuration array");
        return 1;
    }
    const double values[] =
    {
        alphaK1, alphaK2, alphaOmega1, alphaOmega2,
        beta1, beta2, betaStar, gamma1, gamma2,
        a1, b1, c1, kMin, omegaMin, maxSourceNumber,
        wallKappa, wallE, wallCmu
    };
    for (const double value : values)
    {
        if (!std::isfinite(value) || value <= 0.0)
        {
            setLastErrorText("SST coefficients and limits must be positive");
            return 1;
        }
    }
    if (wallTreatment < 0 || wallTreatment > 1 || wallE <= 1.0)
    {
        setLastErrorText("invalid SST wall-function configuration");
        return 1;
    }
    for (int c = 0; c < s->nCells; ++c)
    {
        if
        (
            !std::isfinite(k[c])
         || !std::isfinite(omega[c])
         || !std::isfinite(wallDistance[c])
         || wallDistance[c] <= 0.0
        )
        {
            setLastErrorText("invalid SST cell state or wall distance");
            return 1;
        }
    }
    for (int f = 0; f < s->nFaces; ++f)
    {
        if
        (
            boundaryKMode[f] < 0 || boundaryKMode[f] > 2
         || boundaryOmegaMode[f] < 0 || boundaryOmegaMode[f] > 2
         || !std::isfinite(boundaryK[f])
         || !std::isfinite(boundaryOmega[f])
        )
        {
            setLastErrorText("invalid SST boundary mode or value");
            return 1;
        }
    }

    s->sstCoefficients = ugkwp::SstCoefficients
    {
        alphaK1,
        alphaK2,
        alphaOmega1,
        alphaOmega2,
        beta1,
        beta2,
        betaStar,
        gamma1,
        gamma2,
        a1,
        b1,
        c1
    };
    s->sstKMin = kMin;
    s->sstOmegaMin = omegaMin;
    s->sstMaxSourceNumber = maxSourceNumber;
    s->sstWallTreatment = wallTreatment;
    s->sstWallKappa = wallKappa;
    s->sstWallE = wallE;
    s->sstWallCmu = wallCmu;
    s->sstConfigured = 1;

    int rc = 0;
    const size_t nc = static_cast<size_t>(s->nCells);
    const size_t nf = static_cast<size_t>(s->nFaces);
    rc |= copyToDevice(s->k, k, nc, "cudaMemcpy SST k");
    rc |= copyToDevice(s->omega, omega, nc, "cudaMemcpy SST omega");
    rc |= copyToDevice
    (
        s->sstWallDistance,
        wallDistance,
        nc,
        "cudaMemcpy SST wallDistance"
    );
    rc |= copyToDevice
    (
        s->sstBoundaryKMode,
        boundaryKMode,
        nf,
        "cudaMemcpy SST boundaryKMode"
    );
    rc |= copyToDevice
    (
        s->sstBoundaryOmegaMode,
        boundaryOmegaMode,
        nf,
        "cudaMemcpy SST boundaryOmegaMode"
    );
    rc |= copyToDevice(s->sstBoundaryK, boundaryK, nf, "cudaMemcpy SST boundaryK");
    rc |= copyToDevice
    (
        s->sstBoundaryOmega,
        boundaryOmega,
        nf,
        "cudaMemcpy SST boundaryOmega"
    );
    if (rc != 0 || syncSstConfiguration(s, "cudaMemcpy SST configuration") != 0)
    {
        return 1;
    }

    const int block = s->fixedCellBlockThreads;
    const int grid = (s->nCells + block - 1)/block;
    initialiseSstConservativeStateKernel<<<grid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess)
    {
        setLastError("initialiseSstConservativeStateKernel", err);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(
    void* handle,
    double dt,
    double targetMaxCo,
    double scheduleTime,
    double* maxCo
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "compute gas Courant") != 0)
    {
        return 1;
    }
    if
    (
        maxCo == nullptr
     || !std::isfinite(dt)
     || dt <= 0.0
     || !std::isfinite(targetMaxCo)
     || targetMaxCo <= 0.0
     || !std::isfinite(scheduleTime)
     || scheduleTime < 0.0
    )
    {
        setLastErrorText
        (
            "compute gas Courant requires positive finite dt/targetMaxCo "
            "and a non-null output"
        );
        return 1;
    }
    *maxCo = 0.0;

    if (tuneFixedWorkBlockThreads(s, dt, scheduleTime) != 0)
    {
        return 1;
    }
    const int cellBlock = s->fixedCellBlockThreads;
    const int faceBlock = s->fixedFaceBlockThreads;
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    const int allFaceGrid = faceGrid;

    recoverGasPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel launch for gas Courant", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel for gas Courant", err);
            return 1;
        }
    }

    if (allFaceGrid > 0)
    {
        updateLegacyGasBoundaryMirrorKernel<<<allFaceGrid, faceBlock>>>
        (
            s->deviceState,
            scheduleTime
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "updateLegacyGasBoundaryMirrorKernel for gas Courant",
                err
            );
            return 1;
        }
        updateRiemannBoundaryMirrorKernel<<<allFaceGrid, faceBlock>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "updateRiemannBoundaryMirrorKernel for gas Courant",
                err
            );
            return 1;
        }
    }

    if (s->hostTurbulenceModel == 3)
    {
        applySstWallFunctionStateKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "applySstWallFunctionStateKernel for gas Courant",
                err
            );
            return 1;
        }
    }

    computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasPrimitiveGradientsKernel for gas Courant", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<cellGrid, cellBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstGradientsKernel for gas Courant", err);
            return 1;
        }
    }
    computeGasGradientLimiterKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasGradientLimiterKernel for gas Courant", err);
        return 1;
    }
    computeGasEddyViscosityKernel<<<cellGrid, cellBlock>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasEddyViscosityKernel for gas Courant", err);
        return 1;
    }

    if (faceGrid > 0)
    {
        computeGasCourantFieldKernel<<<faceGrid, faceBlock>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasCourantFieldKernel launch", err);
            return 1;
        }
    }
    computeGasConvectiveCourantByCellKernel<<<cellGrid, cellBlock>>>
    (
        s->deviceState,
        dt
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "computeGasConvectiveCourantByCellKernel launch",
            err
        );
        return 1;
    }

    computeGasDiffusionNumberKernel<<<cellGrid, cellBlock>>>
    (
        s->deviceState,
        dt,
        targetMaxCo
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasDiffusionNumberKernel launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstStabilityNumberKernel<<<cellGrid, cellBlock>>>
        (
            s->deviceState,
            dt,
            targetMaxCo
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstStabilityNumberKernel launch", err);
            return 1;
        }
    }

    thrust::device_ptr<double> convectiveBegin
    (
        s->gasFluxPositivityScale
    );
    const thrust::device_ptr<double> maxConvectiveIt = thrust::max_element
    (
        thrust::device,
        convectiveBegin,
        convectiveBegin + s->nCells
    );
    *maxCo = *maxConvectiveIt;
    const int cell = static_cast<int>(maxConvectiveIt - convectiveBegin);
    thrust::device_ptr<double> diffusionBegin(s->gasDiffusionNumber);
    const thrust::device_ptr<double> maxDiffusionIt = thrust::max_element
    (
        thrust::device,
        diffusionBegin,
        diffusionBegin + s->nCells
    );
    *maxCo = fmax(*maxCo, *maxDiffusionIt);
    if (s->hostTurbulenceModel == 3)
    {
        thrust::device_ptr<double> sstBegin(s->sstSourceNumber);
        const thrust::device_ptr<double> maxSstIt = thrust::max_element
        (
            thrust::device,
            sstBegin,
            sstBegin + s->nCells
        );
        *maxCo = fmax(*maxCo, *maxSstIt);
    }
    if (!std::isfinite(*maxCo) || *maxCo >= 0.5*OfGreat)
    {
        int owner = cell;
        double ownerUx = 0.0;
        double ownerT = 0.0;
        double ownerRho = 0.0;
        double ownerRhoUx = 0.0;
        double ownerRhoE = 0.0;
        if (owner >= 0 && owner < s->nCells)
        {
            copyToHost(&ownerUx, s->Ux + owner, 1, "diagnose Courant owner Ux");
            copyToHost(&ownerT, s->Tgas + owner, 1, "diagnose Courant owner T");
            copyToHost(&ownerRho, s->rho + owner, 1, "diagnose Courant owner rho");
            copyToHost(&ownerRhoUx, s->rhoUx + owner, 1, "diagnose Courant owner rhoUx");
            copyToHost(&ownerRhoE, s->rhoE + owner, 1, "diagnose Courant owner rhoE");
        }
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "non-finite/invalid gas Courant at cell=%d "
            "ownerRho=%.17g ownerRhoUx=%.17g ownerRhoE=%.17g "
            "ownerUx=%.17g ownerT=%.17g "
            "gamma=%.17g R=%.17g Co=%.17g",
            owner,
            ownerRho,
            ownerRhoUx,
            ownerRhoE,
            ownerUx,
            ownerT,
            s->gammaGas,
            s->Rgas,
            *maxCo
        );
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictAdvance
(
    void* handle,
    double dt,
    double simulationTime
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "advance") != 0)
    {
        return 1;
    }
    if (tuneFixedWorkBlockThreads(s, dt, simulationTime) != 0)
    {
        return 1;
    }

    const int block = s->reductionBlockThreads;
    const int warpCount = (block + 31)/32;
    const int grid = (s->nCells + block - 1)/block;
    cudaError_t err = cudaSuccess;

#ifdef UGKP_DEVELOPMENT_PROBES
    DevelopmentAdvanceProbe developmentAdvanceProbe(s, dt, simulationTime);
    if (developmentAdvanceProbe.failed())
    {
        return 1;
    }
#define UGKP_DEV_PROBE_ENTER(STAGE) developmentAdvanceProbe.enter(STAGE)
#define UGKP_DEV_PROBE_LEAVE_IF(STAGE, EXECUTED) \
    do \
    { \
        if (developmentAdvanceProbe.leave(STAGE, EXECUTED) != 0) \
        { \
            return 1; \
        } \
    } while (false)
#define UGKP_DEV_PROBE_LEAVE(STAGE) \
    UGKP_DEV_PROBE_LEAVE_IF(STAGE, true)
#else
#define UGKP_DEV_PROBE_ENTER(STAGE) ((void)0)
#define UGKP_DEV_PROBE_LEAVE(STAGE) ((void)0)
#define UGKP_DEV_PROBE_LEAVE_IF(STAGE, EXECUTED) ((void)0)
#endif

    UGKP_DEV_PROBE_ENTER(ProbeGasFlux);
    if (advanceGasFluxStage(s, dt, simulationTime) != 0)
    {
        return 1;
    }

    const int gasCellBlock = s->fixedCellBlockThreads;
    const int gasCellGrid = (s->nCells + gasCellBlock - 1)/gasCellBlock;
    if (applyGasGravitySource(s, gasCellGrid, gasCellBlock, dt) != 0)
    {
        return 1;
    }

    UGKP_DEV_PROBE_LEAVE(ProbeGasFlux);

    const bool skipParticlePath = !s->particlesMayBePresent;

    if (!skipParticlePath)
    {
    if
    (
        s->hostDragModel != 2
     || s->particleGasHeatTransferModelId != 0
    )
    {
    UGKP_DEV_PROBE_ENTER(ProbeEulerianCoupling);
    if (s->hostDragModel != 2)
    {
    computePressureGradientKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computePressureGradientKernel pre-coupling launch", err);
        return 1;
    }

    if (launchEulerianGasSolidCoupling(s, grid, block, dt) != 0)
    {
        return 1;
    }
    recoverPrimitivesKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverPrimitivesKernel post-Eulerian-coupling launch", err);
        return 1;
    }
    }
    else
    {
        snapshotParticleGasCouplingStateKernel<<<grid, block>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "snapshotParticleGasCouplingStateKernel launch",
                err
            );
            return 1;
        }
    }
    if (s->particleGasHeatTransferModelId != 0)
    {
    applyEulerianParticleMaterialHeatKernel<<<grid, block>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyEulerianParticleMaterialHeatKernel launch", err);
        return 1;
    }

    recoverPrimitivesKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverPrimitivesKernel post-material-heat launch", err);
        return 1;
    }
    }

    UGKP_DEV_PROBE_LEAVE(ProbeEulerianCoupling);
    }


    UGKP_DEV_PROBE_ENTER(ProbeInjection);
    const bool runInjection = s->nBoundarySources > 0;
    if (runInjection)
    {
        const int sourceGrid =
            (s->nBoundarySources + s->particleBlockThreads - 1)
           /s->particleBlockThreads;
        injectBoundaryParticlesKernel
            <<<sourceGrid, s->particleBlockThreads>>>
            (s->deviceState, dt, simulationTime);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("injectBoundaryParticlesKernel launch", err);
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE_IF(ProbeInjection, runInjection);


    UGKP_DEV_PROBE_ENTER(ProbeBinPre);
    const int particleGrid = s->particleWorkGrid;
    const bool runBinPre = s->csrCellLocalPathEnabled != 0;
    if
    (
        runBinPre && preparePreTransportParticleDirectory(s, block) != 0
    )
    {
        return 1;
    }
    if
    (
        runBinPre
     && runToolB3
        (
            s,
            block,
            s->useSplitPreDirectory != 0
              ? HeavyDirectoryKind::splitBaseAndInjection
              : HeavyDirectoryKind::full
        ) != 0
    )
    {
        return 1;
    }
    UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPre, runBinPre);

    UGKP_DEV_PROBE_ENTER(ProbePressurePre);
    if
    (
        applyCollisionalPressureKick
        (
            s,
            0.5*dt,
            block,
            s->useSplitPreDirectory
        ) != 0
    )
    {
        return 1;
    }
    UGKP_DEV_PROBE_LEAVE(ProbePressurePre);

                                                                
                                                                        
                                                     

    UGKP_DEV_PROBE_ENTER(ProbeCollisionPool);
    recoverPrimitivesKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverPrimitivesKernel post-macro-coupling launch", err);
        return 1;
    }

    clearPoissonThermalPoolKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clearPoissonThermalPoolKernel launch", err);
        return 1;
    }
    const size_t poolReduceSharedBytes =
        8u*static_cast<size_t>(block)*sizeof(double);

    if (particleGrid > 0)
    {
        if (s->csrCellLocalPathEnabled != 0)
        {
            if (s->csrHeavyReductionEnabled != 0)
            {
                if
                (
                    launchCsrHeavyPoolReduction
                    (
                        s,
                        dt,
                        true,
                        block,
                        s->useSplitPreDirectory
                    ) != 0
                )
                {
                    return 1;
                }
            }
            else if (s->useSplitPreDirectory != 0)
            {
                if
                (
                    launchSplitPrePoissonPoolLightReduction
                    (
                        s,
                        dt,
                        block,
                        poolReduceSharedBytes
                    ) != 0
                )
                {
                    return 1;
                }
            }
            else
            {
                accumulatePoissonPoolParticlesByCellKernel<false>
                    <<<s->nCells, block, poolReduceSharedBytes>>>
                    (s->deviceState, dt);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    setLastError
                    (
                        "accumulatePoissonPoolParticlesByCellKernel launch",
                        err
                    );
                    return 1;
                }
            }
        }
        else
        {
            accumulateParticlePoolAtomicKernel<true><<<particleGrid, block>>>
            (
                s->deviceState,
                dt
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "accumulateParticlePoolAtomicKernel<true> launch",
                    err
                );
                return 1;
            }
        }
        preparePoissonPoolSamplingKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("preparePoissonPoolSamplingKernel launch", err);
            return 1;
        }

        samplePoissonPoolParticlesKernel<<<particleGrid, block>>>
        (
            s->deviceState,
            0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("samplePoissonPoolParticlesKernel launch", err);
            return 1;
        }

        correctPoissonThermalizedParticlesKernel<<<particleGrid, block>>>
        (
            s->deviceState,
            0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("correctPoissonThermalizedParticlesKernel launch", err);
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeCollisionPool);

    UGKP_DEV_PROBE_ENTER(ProbeRelax);
    if
    (
        particleGrid > 0
     &&
        (
            s->hostDragModel != 2
         || s->particleGasHeatTransferModelId != 0
        )
    )
    {
        if
        (
            launchParticleDragRelaxation
            (
                s,
                particleGrid,
                block,
                dt
            ) != 0
        )
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeRelax);

    UGKP_DEV_PROBE_ENTER(ProbePressurePre);
    if (particleGrid > 0 && applyMobilePackingProjection(s, dt, block) != 0)
    {
        return 1;
    }
    UGKP_DEV_PROBE_LEAVE(ProbePressurePre);

    UGKP_DEV_PROBE_ENTER(ProbeTrack);
    if (particleGrid > 0)
    {
        trackParticlesLocalFaceWalkKernel
            <<<particleGrid, s->particleBlockThreads>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("trackParticlesLocalFaceWalkKernel launch", err);
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeTrack);

    UGKP_DEV_PROBE_ENTER(ProbeBinPost);
    const bool runBinPost =
        particleGrid > 0 && s->csrCellLocalPathEnabled != 0;
    if (runBinPost)
    {
        if (binParticlesByCell(s, block) != 0)
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPost, runBinPost);

    UGKP_DEV_PROBE_ENTER(ProbeMoments);
    if (particleGrid > 0)
    {
        if (s->csrCellLocalPathEnabled != 0)
        {
            clearParticleMomentsKernel<<<grid, block>>>(s->deviceState);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("clearParticleMomentsKernel launch", err);
                return 1;
            }

            const size_t momentSharedBytes =
                8u*static_cast<size_t>(warpCount)*sizeof(double);
            if (s->csrHeavyReductionEnabled != 0)
            {
                if (launchCsrHeavyMomentReduction(s, block) != 0)
                {
                    return 1;
                }
            }
            else
            {
                accumulateParticleMomentsSegmentedKernel<false>
                    <<<s->nCells, block, momentSharedBytes>>>
                    (s->deviceState);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    setLastError
                    (
                        "accumulateParticleMomentsSegmentedKernel launch",
                        err
                    );
                    return 1;
                }
            }
        }
        else
        {
            const int countGrid = (s->nCells + 1 + block - 1)/block;
            clearParticleMomentsAndCountsAtomicKernel<<<countGrid, block>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "clearParticleMomentsAndCountsAtomicKernel launch",
                    err
                );
                return 1;
            }

            accumulateParticleMomentsAtomicKernel<<<particleGrid, block>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("accumulateParticleMomentsAtomicKernel launch", err);
                return 1;
            }

            normalizeParticleMomentsAtomicKernel<<<grid, block>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("normalizeParticleMomentsAtomicKernel launch", err);
                return 1;
            }
        }
        solidRecoveryFromParticleMomentsKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("solidRecoveryFromParticleMomentsKernel launch", err);
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeMoments);

    UGKP_DEV_PROBE_ENTER(ProbePressurePost);
    if (particleGrid > 0)
    {
        if (applyCollisionalPressureKick(s, 0.5*dt, block, 0) != 0)
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbePressurePost);

    UGKP_DEV_PROBE_ENTER(ProbeCompaction);
    if (s->csrCellLocalPathEnabled != 0)
    {
        err = cub::DeviceScan::ExclusiveSum
        (
            s->cellScanTempStorage,
            s->cellScanTempBytes,
            s->cellParticleCount,
            s->compactCellOffset,
            s->nCells + 1
        );
        if (err != cudaSuccess)
        {
            setLastError("cell-local compact offset scan", err);
            return 1;
        }

        if (launchGatherCellLocalParticles(s) != 0)
        {
            return 1;
        }

        const int compactOffsetGrid = (s->nCells + 1 + block - 1)/block;
        captureCompactedPreBaseOffsetsKernel
            <<<compactOffsetGrid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("captureCompactedPreBaseOffsetsKernel launch", err);
            return 1;
        }

                                                                          
                                                                             
                                                                  
        if
        (
            s->nBoundarySources == 0
         && prepareCsrHeavyBaseReductionTasks(s, block) != 0
        )
        {
            return 1;
        }

        commitCellLocalParticleBuffersKernel<<<1, 1>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("commitCellLocalParticleBuffersKernel launch", err);
            return 1;
        }
    }
    else
    {
        const thrust::counting_iterator<int> particleIndices(0);
        const ActiveParticleIndexPredicate predicate{s->deviceState};
        err = cub::DeviceSelect::If
        (
            s->compactSelectTempStorage,
            s->compactSelectTempBytes,
            particleIndices,
            s->sortedParticleIndex,
            s->compactCountDevice,
            s->particleCapacity,
            predicate
        );
        if (err != cudaSuccess)
        {
            setLastError("CUB DeviceSelect active particle indices", err);
            return 1;
        }

        gatherSelectedParticlesKernel<<<particleGrid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("gatherSelectedParticlesKernel launch", err);
            return 1;
        }

        commitSelectedParticleBuffersKernel<<<1, 1>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("commitSelectedParticleBuffersKernel launch", err);
            return 1;
        }
    }
    swapParticleBufferPointersHost(s);
    s->preBaseDirectoryReady = s->csrCellLocalPathEnabled != 0 ? 1 : 0;
    s->useSplitPreDirectory = 0;
    UGKP_DEV_PROBE_LEAVE(ProbeCompaction);
    }
    else
    {
                                                                              
                                                             
        UGKP_DEV_PROBE_ENTER(ProbeEulerianCoupling);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeEulerianCoupling, false);
        UGKP_DEV_PROBE_ENTER(ProbeInjection);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeInjection, false);
        UGKP_DEV_PROBE_ENTER(ProbeBinPre);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPre, false);
        UGKP_DEV_PROBE_ENTER(ProbePressurePre);
        UGKP_DEV_PROBE_LEAVE_IF(ProbePressurePre, false);
        UGKP_DEV_PROBE_ENTER(ProbeCollisionPool);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeCollisionPool, false);
        UGKP_DEV_PROBE_ENTER(ProbeRelax);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeRelax, false);
        UGKP_DEV_PROBE_ENTER(ProbeTrack);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeTrack, false);
        UGKP_DEV_PROBE_ENTER(ProbeBinPost);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPost, false);
        UGKP_DEV_PROBE_ENTER(ProbeMoments);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeMoments, false);
        UGKP_DEV_PROBE_ENTER(ProbePressurePost);
        UGKP_DEV_PROBE_LEAVE_IF(ProbePressurePost, false);
        UGKP_DEV_PROBE_ENTER(ProbeCompaction);
        UGKP_DEV_PROBE_LEAVE_IF(ProbeCompaction, false);
    }

    UGKP_DEV_PROBE_ENTER(ProbeBoundary);
    if (finaliseGasBoundaryStage(s, dt, simulationTime) != 0)
    {
        return 1;
    }

    UGKP_DEV_PROBE_LEAVE(ProbeBoundary);

#ifdef UGKP_DEVELOPMENT_PROBES
    if (developmentAdvanceProbe.finish(!skipParticlePath) != 0)
    {
        return 1;
    }
#endif

#undef UGKP_DEV_PROBE_ENTER
#undef UGKP_DEV_PROBE_LEAVE

    return 0;
}


extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly
(
    void* handle,
    double dt,
    double simulationTime
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "pure-gas advance") != 0)
    {
        return 1;
    }
    if (tuneFixedWorkBlockThreads(s, dt, simulationTime) != 0)
    {
        return 1;
    }

    if (s->particleCapacity != 0 || s->nBoundarySources != 0)
    {
        setLastErrorText
        (
            "pure-gas advance requires zero particle capacity and zero "
            "boundary particle sources"
        );
        return 1;
    }

    if
    (
        !std::isfinite(simulationTime)
     || simulationTime < 0.0
     || advanceGasFluxStage(s, dt, simulationTime) != 0
    )
    {
        return 1;
    }

    const int block = s->fixedCellBlockThreads;
    const int grid = (s->nCells + block - 1)/block;
    if (applyGasGravitySource(s, grid, block, dt) != 0)
    {
        return 1;
    }
    return finaliseGasBoundaryStage(s, dt, simulationTime);
}

extern "C" int ugkwpGpuResidentStrictDownloadFields
(
    void* handle,
    double* rho,
    double* rhoUx,
    double* rhoUy,
    double* rhoUz,
    double* rhoE,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas,
    double* epsS,
    double* rhoUsx,
    double* rhoUsy,
    double* rhoUsz,
    double* rhoEs,
    double* rhoDs,
    double* rhoHp,
    double* Usx,
    double* Usy,
    double* Usz,
    double* theta,
    double* Tp,
    double* dMeanCell
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "field download") != 0)
    {
        return 1;
    }

    const size_t n = static_cast<size_t>(s->nCells);
    int rc = 0;
    rc |= copyToHost(rho, s->rho, n, "cudaMemcpy strict rho result");
    rc |= copyToHost(rhoUx, s->rhoUx, n, "cudaMemcpy strict rhoUx result");
    rc |= copyToHost(rhoUy, s->rhoUy, n, "cudaMemcpy strict rhoUy result");
    rc |= copyToHost(rhoUz, s->rhoUz, n, "cudaMemcpy strict rhoUz result");
    rc |= copyToHost(rhoE, s->rhoE, n, "cudaMemcpy strict rhoE result");
    rc |= copyToHost(Ux, s->Ux, n, "cudaMemcpy strict Ux result");
    rc |= copyToHost(Uy, s->Uy, n, "cudaMemcpy strict Uy result");
    rc |= copyToHost(Uz, s->Uz, n, "cudaMemcpy strict Uz result");
    rc |= copyToHost(p, s->p, n, "cudaMemcpy strict p result");
    rc |= copyToHost(Tgas, s->Tgas, n, "cudaMemcpy strict Tgas result");
    rc |= copyToHost(epsS, s->epsS, n, "cudaMemcpy strict epsS result");
    rc |= copyToHost(rhoUsx, s->rhoUsx, n, "cudaMemcpy strict rhoUsx result");
    rc |= copyToHost(rhoUsy, s->rhoUsy, n, "cudaMemcpy strict rhoUsy result");
    rc |= copyToHost(rhoUsz, s->rhoUsz, n, "cudaMemcpy strict rhoUsz result");
    rc |= copyToHost(rhoEs, s->rhoEs, n, "cudaMemcpy strict rhoEs result");
    rc |= copyToHost(rhoDs, s->rhoDs, n, "cudaMemcpy strict rhoDs result");
    rc |= copyToHost(rhoHp, s->rhoHp, n, "cudaMemcpy strict rhoHp result");
    rc |= copyToHost(Usx, s->Usx, n, "cudaMemcpy strict Usx result");
    rc |= copyToHost(Usy, s->Usy, n, "cudaMemcpy strict Usy result");
    rc |= copyToHost(Usz, s->Usz, n, "cudaMemcpy strict Usz result");
    rc |= copyToHost(theta, s->theta, n, "cudaMemcpy strict theta result");
    rc |= copyToHost(Tp, s->Tp, n, "cudaMemcpy strict Tp result");
    rc |= copyToHost(dMeanCell, s->dMeanCell, n, "cudaMemcpy strict dMeanCell result");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev
(
    void* handle,
    double* epsGPrev
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "epsGPrev download") != 0)
    {
        return 1;
    }
    if (epsGPrev == nullptr)
    {
        setLastErrorText("null epsGPrev download array");
        return 1;
    }

    const size_t n = static_cast<size_t>(s->nCells);
    return copyToHost
    (
        epsGPrev,
        s->epsGPrev,
        n,
        "cudaMemcpy strict epsGPrev download"
    );
}

extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev
(
    void* handle,
    const double* epsGPrev
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "epsGPrev upload") != 0)
    {
        return 1;
    }
    if (epsGPrev == nullptr)
    {
        setLastErrorText("null epsGPrev upload array");
        return 1;
    }

    const size_t n = static_cast<size_t>(s->nCells);
    return copyToDevice
    (
        s->epsGPrev,
        epsGPrev,
        n,
        "cudaMemcpy strict epsGPrev upload"
    );
}

extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields
(
    void* handle,
    double* rho,
    double* Ux,
    double* Uy,
    double* Uz,
    double* p,
    double* Tgas
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "gas boundary field download") != 0)
    {
        return 1;
    }

    const size_t n = static_cast<size_t>(s->nFaces);
    int rc = 0;
    rc |= copyToHost(rho, s->gasBoundaryRho, n, "cudaMemcpy strict gas boundary rho");
    rc |= copyToHost(Ux, s->gasBoundaryUx, n, "cudaMemcpy strict gas boundary Ux");
    rc |= copyToHost(Uy, s->gasBoundaryUy, n, "cudaMemcpy strict gas boundary Uy");
    rc |= copyToHost(Uz, s->gasBoundaryUz, n, "cudaMemcpy strict gas boundary Uz");
    rc |= copyToHost(p, s->gasBoundaryP, n, "cudaMemcpy strict gas boundary p");
    rc |= copyToHost(Tgas, s->gasBoundaryT, n, "cudaMemcpy strict gas boundary T");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictDownloadNut
(
    void* handle,
    double* nut
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "turbulent viscosity download") != 0)
    {
        return 1;
    }
    if (nut == nullptr)
    {
        setLastErrorText("null nut output array");
        return 1;
    }
    return copyToHost
    (
        nut,
        s->nut,
        static_cast<size_t>(s->nCells),
        "cudaMemcpy strict nut result"
    );
}

extern "C" int ugkwpGpuResidentStrictDownloadSst
(
    void* handle,
    double* k,
    double* omega,
    double* nut
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "SST field download") != 0)
    {
        return 1;
    }
    if
    (
        s->hostTurbulenceModel != 3
     || s->sstConfigured == 0
     || k == nullptr
     || omega == nullptr
     || nut == nullptr
    )
    {
        setLastErrorText("SST download requires configured SST and outputs");
        return 1;
    }
    const size_t n = static_cast<size_t>(s->nCells);
    int rc = 0;
    rc |= copyToHost(k, s->k, n, "cudaMemcpy SST k result");
    rc |= copyToHost(omega, s->omega, n, "cudaMemcpy SST omega result");
    rc |= copyToHost(nut, s->nut, n, "cudaMemcpy SST nut result");
    return rc == 0 ? 0 : 1;
}

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror
(
    void* handle,
    int nParticles,
    const double* px,
    const double* py,
    const double* pz,
    const double* pux,
    const double* puy,
    const double* puz,
    const double* pT,
    const double* pTheta,
    const double* pd,
    const double* pm,
    const int* pCellId,
    const int* pStatus,
    const unsigned long long* pRng,
    const unsigned long long* pOrigId
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "particle restart mirror upload") != 0)
    {
        return 1;
    }
    if (nParticles < 0 || nParticles > s->particleCapacity)
    {
        setLastErrorText("particle restart mirror count exceeds resident capacity");
        return 1;
    }

    cudaError_t err =
        cudaMemset(s->pStatus, 0, static_cast<size_t>(s->particleCapacity)*sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict particle restart status", err);
        return 1;
    }
    int rc = 0;
    rc |= copyToDevice(s->particleCountDevice, &nParticles, 1, "cudaMemcpy strict particle restart count");
    if (nParticles > 0)
    {
        s->particlesMayBePresent = true;
    }

    if (rc != 0)
    {
        return 1;
    }

    if (nParticles == 0)
    {
        return rebuildResidentParticleMomentsFromParticles(s, 0);
    }
    if
    (
        px == nullptr || py == nullptr || pz == nullptr
     || pux == nullptr || puy == nullptr || puz == nullptr
     || pT == nullptr || pTheta == nullptr || pd == nullptr || pm == nullptr
     || pCellId == nullptr || pStatus == nullptr
     || pRng == nullptr || pOrigId == nullptr
    )
    {
        setLastErrorText("null particle restart mirror upload array");
        return 1;
    }

    for (int i = 0; i < nParticles; ++i)
    {
        if (!std::isfinite(pm[i]) || pm[i] <= 0.0)
        {
            setLastErrorText("invalid non-positive particle restart mass");
            return 1;
        }
    }

    const size_t n = static_cast<size_t>(nParticles);
    rc |= copyToDevice(s->px, px, n, "cudaMemcpy strict restart px");
    rc |= copyToDevice(s->py, py, n, "cudaMemcpy strict restart py");
    rc |= copyToDevice(s->pz, pz, n, "cudaMemcpy strict restart pz");
    rc |= copyToDevice(s->pux, pux, n, "cudaMemcpy strict restart pux");
    rc |= copyToDevice(s->puy, puy, n, "cudaMemcpy strict restart puy");
    rc |= copyToDevice(s->puz, puz, n, "cudaMemcpy strict restart puz");
    rc |= copyToDevice(s->pT, pT, n, "cudaMemcpy strict restart pT");
    rc |= copyToDevice(s->pTheta, pTheta, n, "cudaMemcpy strict restart pTheta");
    rc |= copyToDevice(s->pd, pd, n, "cudaMemcpy strict restart pd");
    rc |= copyToDevice(s->pm, pm, n, "cudaMemcpy strict restart pm");
    rc |= copyToDevice(s->pCellId, pCellId, n, "cudaMemcpy strict restart pCellId");
    rc |= copyToDevice(s->pStatus, pStatus, n, "cudaMemcpy strict restart pStatus");
    rc |= copyToDevice(s->pRng, pRng, n, "cudaMemcpy strict restart pRng");
    rc |= copyToDevice(s->pOrigId, pOrigId, n, "cudaMemcpy strict restart pOrigId");

    if (rc != 0)
    {
        return 1;
    }

    return rebuildResidentParticleMomentsFromParticles(s, nParticles);
}

extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror
(
    void* handle,
    int* nParticles,
    int maxParticles,
    double* px,
    double* py,
    double* pz,
    double* pux,
    double* puy,
    double* puz,
    double* pT,
    double* pTheta,
    double* pd,
    double* pm,
    int* pCellId,
    int* pStatus,
    unsigned long long* pRng,
    unsigned long long* pOrigId
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "particle restart mirror download") != 0)
    {
        return 1;
    }
    if (nParticles == nullptr || maxParticles < 0)
    {
        setLastErrorText("invalid particle restart mirror download request");
        return 1;
    }

    int count = 0;
    int rc = copyToHost(&count, s->particleCountDevice, 1, "cudaMemcpy strict restart particle count");
    if (rc != 0)
    {
        return 1;
    }
    if (count < 0 || count > s->particleCapacity)
    {
        setLastErrorText("resident particle count is outside capacity");
        return 1;
    }
    *nParticles = count;

    if (maxParticles == 0)
    {
        return 0;
    }
    if (maxParticles < count)
    {
        setLastErrorText("particle restart mirror output arrays are too small");
        return 1;
    }
    if (count == 0)
    {
        return 0;
    }
    if
    (
        px == nullptr || py == nullptr || pz == nullptr
     || pux == nullptr || puy == nullptr || puz == nullptr
     || pT == nullptr || pTheta == nullptr || pd == nullptr || pm == nullptr
     || pCellId == nullptr || pStatus == nullptr
     || pRng == nullptr || pOrigId == nullptr
    )
    {
        setLastErrorText("null particle restart mirror download array");
        return 1;
    }

    const size_t n = static_cast<size_t>(count);
    rc |= copyToHost(px, s->px, n, "cudaMemcpy strict restart px");
    rc |= copyToHost(py, s->py, n, "cudaMemcpy strict restart py");
    rc |= copyToHost(pz, s->pz, n, "cudaMemcpy strict restart pz");
    rc |= copyToHost(pux, s->pux, n, "cudaMemcpy strict restart pux");
    rc |= copyToHost(puy, s->puy, n, "cudaMemcpy strict restart puy");
    rc |= copyToHost(puz, s->puz, n, "cudaMemcpy strict restart puz");
    rc |= copyToHost(pT, s->pT, n, "cudaMemcpy strict restart pT");
    rc |= copyToHost(pTheta, s->pTheta, n, "cudaMemcpy strict restart pTheta");
    rc |= copyToHost(pd, s->pd, n, "cudaMemcpy strict restart pd");
    rc |= copyToHost(pm, s->pm, n, "cudaMemcpy strict restart pm");
    rc |= copyToHost(pCellId, s->pCellId, n, "cudaMemcpy strict restart pCellId");
    rc |= copyToHost(pStatus, s->pStatus, n, "cudaMemcpy strict restart pStatus");
    rc |= copyToHost(pRng, s->pRng, n, "cudaMemcpy strict restart pRng");
    rc |= copyToHost(pOrigId, s->pOrigId, n, "cudaMemcpy strict restart pOrigId");
    return rc == 0 ? 0 : 1;
}

extern "C" void ugkwpGpuResidentStrictRelease(void* handle)
{
#ifdef UGKP_DEVELOPMENT_PROBES
    if (developmentProbe.owner == asState(handle))
    {
        shutdownDevelopmentProbe();
    }
#endif
    releaseState(asState(handle));
}
