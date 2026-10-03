#define GPU_POOL_THETA_AFTER_REJECTION 1
#define GPU_POOL_PARTICLE_THETA(s, i) clampMin(finiteOr(s.pTheta[i], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0))
#define GPU_POOL_MASS_FALLBACK GPU_OPERATOR_R(0.0)
#define GPU_POOL_SPLIT_REDUCTION_TOPOLOGY PoolReductionTopology::sharedTree
#define GPU_POOL_STATIC_DIRECTORY 1
#define GPU_POOL_CACHED_PROBABILITY 1
#define GPU_POOL_SPLIT_DIRECTORY (DirectoryKind == HeavyDirectoryKind::splitBaseAndInjection)
#define GPU_POOL_DIRECT_PARTICLE_INDEX (DirectoryKind == HeavyDirectoryKind::baseOnly)
#define GPU_POOL_SAMPLING_READY(s, c) preparePoissonPoolSamplingCell(s, c);
#define GPU_POOL_END_TASK(s, c, first) first + s.csrCellTaskCount[c]
#define GPU_DIRECTORY_PARAMETER_TYPE HeavyDirectoryKind
#define GPU_DIRECTORY_ARGUMENT static_cast<int>(directoryKind)
#define GPU_DIRECTORY_HAS_BASE_ONLY 1
#define GPU_DIRECTORY_OWNS_TILE_POLICY 1
#define GPU_DIRECTORY_SELECTOR directoryKind
#define GPU_DIRECTORY_FULL static_cast<int>(HeavyDirectoryKind::full)
#define GPU_OPERATOR_TINY(x) x
#define GPU_OPERATOR_THERMAL 0
#define GPU_RECOVERY_INLINE __forceinline__
#define GPU_PERIODIC_FACE_KIND 5
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) hit - shift + eps*normal
#define GPU_WALL_COORDINATE(hit, eps, normal) hit - eps*normal
#define GPU_RESET_CONTACT_AGE(s, i)
#define GPU_CONTACT_DURATION(value) static_cast<float>(value)
#define GPU_CONTACT_PEAK(value) static_cast<float>(value)
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) x
#include "../../../common/GpuOperatorContract.cuh"
#include <cuda_runtime.h>

#include "GpuBackendApi.H"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "../../../common/gasNumerics/GpuLesAlgebra.cuh"
#include "../../../common/gasNumerics/GpuParticlePhysicsAlgebra.cuh"
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
    splitInjectionIndexed = 2,
    splitLogical = 3
};

enum class HeavyDirectoryKind : int
{
    full = 0,
    splitBaseAndInjection = 1,
    baseOnly = 2
};

struct CsrReductionTask
{
    int cell;
    int begin;
    int end;
    int source;
};

struct PressureProjectionCell
{
    double ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale;
    bool active, resolved;
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
    cudaStream_t gasCaptureStream = nullptr;
    cudaGraph_t gasGraph = nullptr;
    cudaGraphExec_t gasGraphExec = nullptr;
    std::vector<cudaGraphNode_t> gasGraphTimeNodes;
    std::vector<cudaKernelNodeParams> gasGraphTimeParams;
    double gasGraphDt = -1;
    int gasGraphMode = -1;
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
    int csrReductionDirectoryKind = 0;
    int csrWarpAggregatedBinning = 0;
    int csrSplitPreDirectoryEnabled = 1;
    int csrHeavyTaskCapacity = 0;
    int csrHeavyWorkerGrid = 0;
    int mobilePackingCooperativeGrid = 0;
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
    double* packingStuckRho = nullptr;
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
    // All nonempty-cell tasks, including single-task cells.
    int* csrHeavyTaskCount = nullptr;
    int* csrHeavyTaskCursor = nullptr;
    // Only cells requiring multiple tasks and final reduction.
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
    double* poissonCellCollisionProbability = nullptr;
    PressureProjectionCell* pressureProjectionCache = nullptr;
    int trackingWorkGrid = 0;

};
#include "GpuContactAgeStorage.cuh"


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

    if (s->gasGraphExec) cudaGraphExecDestroy(s->gasGraphExec);
    if (s->gasGraph) cudaGraphDestroy(s->gasGraph);
    if (s->gasCaptureStream) cudaStreamDestroy(s->gasCaptureStream);
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
    release(s->packingStuckRho);
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
    release(s->poissonCellCollisionProbability);
    release(s->pressureProjectionCache);
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
    rc |= allocate(s->packingStuckRho, nc, "cudaMalloc stuck packing density");
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
    rc |= allocate(s->poissonCellCollisionProbability, nc, "cudaMalloc strict poissonCellCollisionProbability");
    rc |= allocate(s->pressureProjectionCache, nc, "cudaMalloc strict pressureProjectionCache");
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

#include "operators/clampMin.cuh"

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

#include "operators/validateDevelopmentProbeCellsKernel.cuh"

#define GPU_PROBE_REQUIRE_POSITIVE_ACTIVE_MASS 1
#include "operators/validateDevelopmentProbeParticlesKernel.cuh"
#undef GPU_PROBE_REQUIRE_POSITIVE_ACTIVE_MASS

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

#include "operators/clearSolidCell.cuh"

#include "operators/clampRange.cuh"

#include "operators/linearScheduledValueDevice.cuh"

#include "operators/scheduledSolidVolumeFractionDevice.cuh"

#include "operators/publishScheduledInletConfigurationKernel.cuh"

struct GasPrimDevice
{
    double rho;
    double ux;
    double uy;
    double uz;
    double p;
    double T;
};

#include "operators/makeGasPrimDevice.cuh"

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

#include "operators/computeGasPrimitiveGradientsKernel.cuh"

#include "operators/computeSstGradientsKernel.cuh"

#include "operators/sstVelocityInvariants.cuh"

  
                                                              
  
                                                     
                                                                          
                                                                           
                                                                           
                                                                       
                                                                     
                                                                          
                                             
   
#include "operators/computeGasHllcAdcSensorKernel.cuh"

#include "operators/updateBarthLimiter.cuh"

#include "operators/computeGasGradientLimiterKernel.cuh"

#include "operators/computeGasEddyViscosityKernel.cuh"

#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"

#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"

#include "operators/gasFaceSubgridTransportProperties.cuh"

#include "operators/computeRiemannGasFaceFluxDevice.cuh"

#include "operators/computeGasInternalFaceFluxKernel.cuh"


#include "operators/computeGasFluxPositivityScaleKernel.cuh"

#include "operators/computeSstFaceFluxKernel.cuh"

#include "GpuGravityUpdate.cuh"
__global__ void applyGasGravitySourceKernel(DeviceState* sp, const double dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<false>(*sp,c,dt);
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

#include "operators/recoverPrimitivesKernel.cuh"



#include "operators/clearParticleMomentsKernel.cuh"

#include "operators/computePressureGradientKernel.cuh"

#include "operators/solidEpsFromMomentDevice.cuh"

__host__ __device__ inline bool gasDragModelActive(const int modelId)
{
    return modelId != 2;
}

#include "operators/applyGasVolumeFractionSourceKernel.cuh"

#include "GpuCellDragRate.cuh"
template<class DragModel>
__global__ void applyEulerianGasSolidDragKernel
(
    DeviceState* sp,
    const double dt
)
{
    const DragModel dragModel{};
#include "GpuCellDragUpdate.inl"
}
}

int launchEulerianGasSolidDrag
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
            applyEulerianGasSolidDragKernel
                <ugkwpGpuDrag::SchillerNaumannDeviceDrag>
                <<<grid, block>>>(s->deviceState, dt);
            break;
        case 1:
            applyEulerianGasSolidDragKernel
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
        setLastError("applyEulerianGasSolidDragKernel launch", err);
        return 1;
    }
    return 0;
}

#include "operators/applyEulerianParticleMaterialHeatKernel.cuh"

#include "operators/snapshotParticleGasCouplingStateKernel.cuh"

#include "operators/uniform01Device.cuh"

#include "operators/normalDevice.cuh"

#include "operators/solidPressureFromMomentsDevice.cuh"

#include "operators/computeCollisionalPressureKernel.cuh"

using PressureReal = double;
using PressureTime = double;
#include "GpuPressureKickAccumulation.cuh"
struct AnalyticPressureScratch
{
    __device__ static __forceinline__ void initialise(DeviceState&,int) {}
};
__global__ void accumulateCollisionalPressureKickByCellKernel
(DeviceState* sp,const double kickDt,const int computeScale)
{
    runPressureKickAccumulation<AnalyticPressureScratch>(sp,kickDt,computeScale,0);
}

#include "GpuPressureLimiter.cuh"


#include "GpuPressureParticleUpdate.cuh"
template<bool CompactParticles = false>
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
     || (CompactParticles ? s.compactPStatus : s.pStatus)[i] == 0
     || (CompactParticles ? s.compactPCellId : s.pCellId)[i] != c
    )
    {
        return;
    }
    updateMobilePressureParticle<CompactParticles>
        (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
}

__device__ PressureProjectionCell preparePressureProjectionCell
(DeviceState& s, const int c, const double kickDt)
{
    PressureProjectionCell projection{};
    double scaledDelta[4];

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



    projection.active = false;
    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return projection;
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


    projection =
        {ux0, uy0, uz0, ux1, uy1, uz1, theta1, thermalScale, thetaScale, true, resolved};

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


    // The atomic path historically publishes the cell update, then reads it
    // back and reconstructs the old state. Preserve its rounding and clamps:
    // (e0 + de) - de is not equivalent to e0 near the resolved-theta threshold.
    if (!s.csrCellLocalPathEnabled)
    {
        const double ax1 = finiteOr(px1, 0.0);
        const double ay1 = finiteOr(py1, 0.0);
        const double az1 = finiteOr(pz1, 0.0);
        const double ae1 = clampMin(finiteOr(e1, 0.0), 0.0);
        const double ax0 = ax1 - scaledDelta[0];
        const double ay0 = ay1 - scaledDelta[1];
        const double az0 = az1 - scaledDelta[2];
        const double ae0 = ae1 - scaledDelta[3];
        const double at0 = clampMin
        (
            pressureKickInternalEnergy(rhoP, ax0, ay0, az0, ae0)/(1.5*rhoP), 0.0
        );
        const double at1 = clampMin
        (
            pressureKickInternalEnergy(rhoP, ax1, ay1, az1, ae1)/(1.5*rhoP), 0.0
        );
        const bool ar = at0 > 10.0*s.thetaMin;
        const double scale = ar ? sqrt(clampMin(at1/at0, 0.0)) : 0.0;
        projection =
        {
            ax0/rhoP, ay0/rhoP, az0/rhoP, ax1/rhoP, ay1/rhoP, az1/rhoP,
            at1, scale, ar ? scale*scale : 0.0, true, ar
        };
    }
    return projection;
}

#include "GpuPressureCellTraversal.cuh"
#include "GpuPressureAnalyticAdapter.cuh"
template<bool SplitDirectory, bool CompactParticles = false>
__global__ void applyCollisionalPressureProjectionKernel(DeviceState* sp,const double dt)
{
    runCellPressureProjection<AnalyticPressureOperation,
        SplitDirectory?PressureDirectory::Split:PressureDirectory::Full,CompactParticles>(sp,dt);
}

__global__ void preparePressureProjectionCacheKernel
(DeviceState* sp, const double kickDt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < s.nCells)
        s.pressureProjectionCache[c] = preparePressureProjectionCell(s, c, kickDt);
}

__global__ void applyCachedPressureProjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < nParticles; i += blockDim.x*gridDim.x)
    {
        if (s.pStatus[i] == 0) continue;
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells) continue;
        const PressureProjectionCell q = s.pressureProjectionCache[c];
        if (!q.active) continue;
        applyCollisionalPressureProjectionOneParticle(s, i, c,
            q.ux0, q.uy0, q.uz0, q.ux1, q.uy1, q.uz1, q.theta1,
            q.resolved, q.thermalScale, q.thetaScale);
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

#include "GpuPressureAnalyticLaunch.cuh"
#include "GpuPressurePipeline.cuh"
int applyCollisionalPressureKick(DeviceState* s,const double dt,const int block,
    const int split,const bool compact=false)
{
    return launchCommonPressureKick<AnalyticPressureLaunch>(s,dt,block,block,split!=0,compact);
}


                                                                           
                                                                           
                                                                       
constexpr double mobilePackingJacobiOmega = 0.8;

#include "operators/clearMobilePackingActivityCountsKernel.cuh"

#include "operators/accumulateMobilePackingMomentsKernel.cuh"

#include "operators/normalizeMobilePackingMomentsKernel.cuh"

#include "../../../common/gasNumerics/GpuPackingProjectionAlgebra.cuh"

#include "operators/prepareMobilePackingProjectionKernel.cuh"

#include "operators/mobilePackingParticleEligible.cuh"

#include "../../../common/gasNumerics/GpuPackingProjectionCooperative.cuh"

#include "../../../common/GpuGasHostPolicy.cuh"
using GasHostPolicy = GasHostWithoutWallEnergy<double, double>;
#include "../../../common/GpuMobilePackingHost.cuh"

#include "operators/granularCollisionTauFromCellDevice.cuh"

#include "operators/injectBoundaryParticlesKernel.cuh"

#include "operators/pointInsideCell.cuh"

#include "GpuParticleTransport.cuh"

#include "operators/trackParticlesLocalFaceWalkKernel.cuh"

#include "GpuParticleGasHeatRelaxation.cuh"
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
    if (gasDragModelActive(s.dragModel))
    {
    const double invTauDrag = DragModel::inverseResponseTime(dragInput);

    
#include "operators/ParticleDragMomentum.inl"


                                                                      
                                                                          
                                                                          
                                                                              
    decayParticleUnresolvedTheta(s, i, c);
    }

    if
    (
        s.solveParticleTemperature != 0
     && s.particleGasHeatTransferModelId != 0
     && s.particleThermalRho > 0.0
     && s.particleCp > 0.0
    )
    {
        const double tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const double tgOld = clampRange(finiteOr(s.couplingTgasOld[c], s.TgasMin), s.TgasMin, 1.0e30);
        s.pT[i] = particleTemperatureAfterGasRelaxation
            (s, tpOld, tgOld, re, dPart, s.particleThermalCapacity, dt);
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

#include "GpuCollisionProbability.cuh"

__global__ void clearPoissonThermalPoolKernel(DeviceState* sp, const double dt)
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

    // This producer runs after this step's pressure kick and primitive
    // recovery. Only multi-segment consumers need cross-block reuse.
    if (s.csrHeavyReductionEnabled != 0 && s.csrCellTaskCount[c] > 1)
    {
        s.poissonCellCollisionProbability[c] =
            poissonCollisionProbabilityForCell(s, c, dt);
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
            if (lane < offset)
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

            int firstOffset = 16;
            while (firstOffset >= warpCount) firstOffset >>= 1;
            for (int offset = firstOffset; offset > 0; offset >>= 1)
            {
                const double other =
                    __shfl_down_sync(fullWarpMask, value, offset);
                if (lane < offset && lane + offset < warpCount)
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

#include "GpuCollisionPoolParticle.cuh"

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
    accumulateOnePoolParticle<true>
    (
        s, c, i, prob, locMass, locMomX, locMomY, locMomZ,
        locEnergy, locDiameter, locDiameter2, locCount
    );
}

#include "../../../common/GpuCollisionPoolCell.cuh"

#include "../../../common/GpuCollisionPoolSplitCell.cuh"

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

#include "../../../common/GpuCollisionPoolAtomic.cuh"

#include "GpuCollisionPoolRange.cuh"

#include "GpuCollisionPoolSplitParticle.cuh"

#include "operators/accumulateCsrSplitLogicalPoolTask.cuh"










#include "GpuCollisionPoolTarget.cuh"

#include "CsrPersistentQueue.cuh"

#include "GpuSegmentedPoolWorkers.cuh"

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

}

__global__ void preparePoissonPoolSamplingKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    preparePoissonPoolSamplingCell(s, c);
}

#include "../../../common/GpuCollisionPoolSampling.cuh"

#include "operators/samplePoissonPoolParticlesKernel.cuh"

#include "../../../common/GpuCollisionPoolCorrection.cuh"

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

#include "GpuCellLocalPrimary.cuh"
#include "GpuParticlePayload.cuh"



#include "GpuCellLocalGather.cuh"

template<int BlockThreads>
struct CsrGatherOperation
{
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int task)
    {
        const CsrReductionTask descriptor = s.csrReductionTasks[task];
        const int c = descriptor.cell;
        const int start = s.cellParticleOffset[c];
        const int end = s.cellParticleOffset[c + 1];
        if (s.cellParticleCount[c] == 0) return;
        const bool allKept = s.cellParticleCount[c] == end - start;
        if (allKept)
            gatherCellLocalRange<BlockThreads>(s, c, descriptor.begin, descriptor.end,
                s.compactCellOffset[c] + descriptor.begin - start, true);
        else if (task == s.csrCellTaskOffset[c])
            gatherCellLocalRange<BlockThreads>(s, c, start, end,
                s.compactCellOffset[c], false);

    }
};

template<int BlockThreads>
__global__ void gatherCsrSegmentedParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    CsrGatherOperation<BlockThreads> operation;
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}


int launchGatherCellLocalParticles(DeviceState* s)
{
    if (s->csrHeavyReductionEnabled != 0)
    {
        const cudaError_t resetError = resetCsrPersistentQueue(s);
        if (resetError != cudaSuccess)
        {
            setLastError("reset CSR gather queue", resetError);
            return 1;
        }
    }
    switch (s->reductionBlockThreads)
    {
        case 32:
            if (s->csrHeavyReductionEnabled != 0)
                gatherCsrSegmentedParticlesKernel<32><<<s->csrHeavyWorkerGrid, 32>>>(s->deviceState);
            else
                gatherCellLocalParticlesKernel<32>
                <<<s->nCells, 32>>>(s->deviceState);
            break;
        case 64:
            if (s->csrHeavyReductionEnabled != 0)
                gatherCsrSegmentedParticlesKernel<64><<<s->csrHeavyWorkerGrid, 64>>>(s->deviceState);
            else
                gatherCellLocalParticlesKernel<64>
                <<<s->nCells, 64>>>(s->deviceState);
            break;
        case 128:
            if (s->csrHeavyReductionEnabled != 0)
                gatherCsrSegmentedParticlesKernel<128><<<s->csrHeavyWorkerGrid, 128>>>(s->deviceState);
            else
                gatherCellLocalParticlesKernel<128>
                <<<s->nCells, 128>>>(s->deviceState);
            break;
        case 256:
            if (s->csrHeavyReductionEnabled != 0)
                gatherCsrSegmentedParticlesKernel<256><<<s->csrHeavyWorkerGrid, 256>>>(s->deviceState);
            else
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

#include "operators/gatherSelectedParticlesKernel.cuh"

#include "operators/swapParticlePointerDevice.cuh"

#include "GpuParticleBufferCommit.cuh"





void swapParticleBufferPointersHost(DeviceState* s)
{
    swapParticleBuffersDevice(*s);
}

#include "operators/clearParticleMomentsAndCountsAtomicKernel.cuh"

#include "operators/accumulateParticleMomentsAtomicKernel.cuh"

#include "operators/normalizeParticleMomentsAtomicKernel.cuh"

constexpr bool postTransportFusePayload = true;
#define GPU_MOMENT_REAL double
#define GPU_MOMENT_R(x) (x)
#define GPU_MOMENT_THERMAL 0
#define GPU_MOMENT_GATHER 1
#include "GpuParticleMoments.cuh"



#define GPU_PIPELINE_REAL double
#include "GpuSegmentedMomentWorkers.cuh"

// Compatibility adapter: historical gas bool order is recovery, then payload.
int launchCsrSegmentedMomentReduction(DeviceState* s, const int block,
    const bool deferRecovery = false, const bool gatherSurvivors = false)
{
    return launchCommonSegmentedMomentReduction(s, block,
        {gatherSurvivors ? MomentPayload::gatherSurvivors : MomentPayload::momentsOnly,
         deferRecovery ? MomentRecovery::deferToAdvance : MomentRecovery::completeHere});
}

int launchCsrHeavyMomentReduction(DeviceState* s, const int block, const bool deferRecovery = false, const bool gatherSurvivors = false)
{
    return launchCsrSegmentedMomentReduction(s, block, deferRecovery, gatherSurvivors);

}

__global__ void solidRecoveryFromParticleMomentsKernel(DeviceState* sp);

#include "operators/clearParticleCellBinsKernel.cuh"

#include "operators/countParticlesByCellKernel.cuh"

                                                                         
                                                                           
#include "GpuInjectionDirectory.cuh"



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



#include "GpuHardwareReductionTile.cuh"
__device__ __forceinline__ int csrReductionTileParticles
(
    const DeviceState& s,
    const int directoryKind
)
{
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

    const int threshold = hardwareReductionTile(totalParticles, s.reductionBlockThreads,
        s.multiprocessorCount, s.lightBlocksPerSm);
    if (threshold == 0) asm("trap;");
    return threshold;
}

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
    const int threshold = csrReductionTileParticles(s, directoryKind);
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

#include "operators/publishHeavyReductionDecisionKernel.cuh"

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
    s->csrHeavyCellThreshold = threshold;
    s->csrHeavyTileParticles = threshold;
    s->csrHeavyReductionActive = active;
    s->csrHeavyReductionEnabled = active;
    publishHeavyReductionDecisionKernel<<<1, 1>>>(s->deviceState, active);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 publish automatic L2 decision launch", err);
        return 1;
    }
    return 0;
}

#include "GpuReductionTaskCount.cuh"

#include "GpuReductionTaskWrite.cuh"

#include "GpuReductionTaskMaterialize.cuh"



#include "GpuReductionTaskPipeline.cuh"



int prepareCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::full
    );
}



int prepareCsrHeavyBaseReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::baseOnly
    );
}



int prepareSplitPreCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks
    (
        s, block, HeavyDirectoryKind::splitBaseAndInjection
    );
}

int binParticlesByCell(DeviceState* s, const int block, const bool survivorsOnly = false)
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
        if (survivorsOnly)
            countParticlesByCellKernel<true, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            countParticlesByCellKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        if (survivorsOnly)
            countParticlesByCellKernel<false, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            countParticlesByCellKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
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
        if (survivorsOnly)
            scatterParticlesByCellKernel<true, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            scatterParticlesByCellKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        if (survivorsOnly)
            scatterParticlesByCellKernel<false, true><<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        else
            scatterParticlesByCellKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
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
        countSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        countSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("countSplitPreInjectionParticlesKernel launch", err);
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
        scatterSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    else
    {
        scatterSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatterSplitPreInjectionParticlesKernel launch", err);
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
            accumulateParticleMomentsAtomicKernel<<<s->particleWorkGrid, s->particleBlockThreads>>>
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

#include "GpuMechanicalRecovery.cuh"

#define GPU_PIPELINE_REAL double
#include "GpuMomentRecovery.cuh"
#define GPU_PIPELINE_REAL double
#include "GpuMomentPipeline.cuh"


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

#include "../../../common/GpuDevelopmentProbeInit.cuh"

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
            validateDevelopmentProbeParticlesKernel<<<s->particleWorkGrid, s->particleBlockThreads>>>
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

constexpr bool developmentProbeIncludesScheduling = true;
#include "../../../common/GpuDevelopmentAdvanceProbe.cuh"

#endif

extern "C" const char* ugkwpGpuResidentStrictLastError()
{
    return lastError;
}

#include "../../../common/GpuGasAdvance.cuh"

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

#include "../../../common/GpuToolB1Launch.cuh"

#include "../../../common/GpuToolB1.cuh"


#include "GpuParticleLaunchConfiguration.cuh"

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
    if (queryKernelBlocksPerSm(gasBlocks,
        "occupancy query gas internal-face kernel",
        computeGasInternalFaceFluxKernel<true>, s->fixedFaceBlockThreads, 0) != 0)
    {
        return 1;
    }

    (void)gasBlocks;
    s->lightBlocksPerSm = 0;
    s->heavyBlocksPerSm = 0;
    s->csrHeavyWorkerGrid = 0;
    if (s->csrHeavyReductionEnabled != 0)
    {
        int heavyPoolBlocks = 0;
        int heavyPoolBaseBlocks = 0;
        int heavyPoolSplitBlocks = 0;
        int heavyMomentBlocks = 0;
        if
        (
            queryKernelBlocksPerSm
            (
                heavyPoolBlocks,
                "occupancy query segmented pool worker kernel",
                accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::full>,
                reductionBlock,
                componentSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                heavyPoolBaseBlocks,
                "occupancy query segmented pool baseOnly worker kernel",
                accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::baseOnly>,
                reductionBlock,
                componentSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                heavyPoolSplitBlocks,
                "occupancy query segmented pool splitBaseAndInjection worker kernel",
                accumulateCsrSegmentedPoolTasksPersistentKernel<true, HeavyDirectoryKind::splitBaseAndInjection>,
                reductionBlock,
                componentSharedBytes
            ) != 0
         || queryKernelBlocksPerSm
            (
                heavyMomentBlocks,
                "occupancy query segmented moment worker kernel",
                accumulateCsrSegmentedMomentTasksPersistentKernel<postTransportFusePayload>,
                reductionBlock,
                componentSharedBytes
            ) != 0
        )
        {
            return 1;
        }
        heavyPoolBlocks = std::min(heavyPoolBlocks, std::min(heavyPoolBaseBlocks, heavyPoolSplitBlocks));
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

    s->mobilePackingCooperativeGrid = 0;
    if (s->jammingPressureEnabled != 0)
    {
        int device = 0;
        int cooperativeLaunch = 0;
        int residentBlocks = 0;
        err = cudaGetDevice(&device);
        if (err == cudaSuccess)
        {
            err = cudaDeviceGetAttribute
            (
                &cooperativeLaunch,
                cudaDevAttrCooperativeLaunch,
                device
            );
        }
        if (err == cudaSuccess && cooperativeLaunch != 0)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &residentBlocks,
                completeMobilePackingProjectionCooperativeKernel,
                particleBlock,
                0
            );
        }
        if (err != cudaSuccess || cooperativeLaunch == 0 || residentBlocks <= 0)
        {
            if (err != cudaSuccess)
            {
                setLastError("CUDA mobile-packing cooperative occupancy", err);
            }
            else
            {
                setLastErrorText("GPU does not support mobile-packing cooperative launch");
            }
            return 1;
        }
        s->mobilePackingCooperativeGrid =
            s->multiprocessorCount*residentBlocks;
    }

    if (queryParticleKernelResidency(s, countParticlesByCellKernel<true>, s->particleBlocksPerSm) != 0) return 1;
    setParticleWorkGridFromResidency(s, s->particleBlocksPerSm);
    if (configureTrackingWorkGrid(s) != 0) return 1;

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

#ifndef UGKP_DEVELOPMENT_PROBES
    if (s->gasGraphMode < 0)
    {
        const char* value = std::getenv("UGKP_GAS_GRAPH");
        s->gasGraphMode = !value || std::strcmp(value, "1") == 0;
    }
    if (s->gasGraphMode && !s->particlesMayBePresent && !s->hostGravityActive)
        return advancePureGasGraph(s, dt, simulationTime);
#endif

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
    const bool dragActive = gasDragModelActive(s->hostDragModel);

    if (!skipParticlePath)
    {
    UGKP_DEV_PROBE_ENTER(ProbeEulerianCoupling);
    applyGasVolumeFractionSourceKernel<<<grid, block>>>
    (
        s->deviceState,
        dt
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasVolumeFractionSourceKernel launch", err);
        return 1;
    }
    recoverPrimitivesKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverPrimitivesKernel post-volume-source launch", err);
        return 1;
    }

    if (dragActive)
    {
    computePressureGradientKernel<<<grid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computePressureGradientKernel pre-coupling launch", err);
        return 1;
    }

    if (launchEulerianGasSolidDrag(s, grid, block, dt) != 0)
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
    else if (s->particleGasHeatTransferModelId != 0)
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

    clearPoissonThermalPoolKernel<<<grid, block>>>(s->deviceState, dt);
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
            accumulateParticlePoolAtomicKernel<true><<<particleGrid, s->particleBlockThreads>>>
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
        if (s->csrHeavyReductionEnabled == 0)
        {
            preparePoissonPoolSamplingKernel<<<grid, block>>>(s->deviceState);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("preparePoissonPoolSamplingKernel launch", err);
                return 1;
            }
        }

        samplePoissonPoolParticlesKernel<<<particleGrid, s->particleBlockThreads>>>
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

        correctPoissonThermalizedParticlesKernel<<<particleGrid, s->particleBlockThreads>>>
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
            dragActive
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
                s->particleBlockThreads,
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
            <<<s->trackingWorkGrid, s->particleBlockThreads>>>(s->deviceState, dt);
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
        if (binParticlesByCell(s, block, true) != 0)
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE_IF(ProbeBinPost, runBinPost);

    UGKP_DEV_PROBE_ENTER(ProbeMoments);
    if (launchPostTransportMomentPipeline(s, particleGrid, block) != 0)
    {
        return 1;
    }
    UGKP_DEV_PROBE_LEAVE(ProbeMoments);

    UGKP_DEV_PROBE_ENTER(ProbePressurePost);
    if (particleGrid > 0)
    {
        if (applyCollisionalPressureKick(s, 0.5*dt, block, 0,
            s->csrCellLocalPathEnabled != 0 && postTransportFusePayload) != 0)
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

        if (launchDeferredSurvivorPayload<postTransportFusePayload>
            (s, particleGrid, block) != 0)
            return 1;

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

        gatherSelectedParticlesKernel<<<particleGrid, s->particleBlockThreads>>>(s->deviceState);
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

    if (!std::isfinite(simulationTime) || simulationTime < 0.0)
        return 1;
#ifndef UGKP_DEVELOPMENT_PROBES
    if (s->gasGraphMode < 0)
    {
        const char* value = std::getenv("UGKP_GAS_GRAPH");
        s->gasGraphMode = !value || std::strcmp(value, "1") == 0;
    }
    if (s->gasGraphMode && !s->particlesMayBePresent && !s->hostGravityActive)
        return advancePureGasGraph(s, dt, simulationTime);
#endif

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
