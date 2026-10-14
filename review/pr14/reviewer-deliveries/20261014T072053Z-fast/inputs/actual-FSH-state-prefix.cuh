#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/ContactCapabilityValidation.h"
#define GPU_CONTACT_SEPARATE_AGE 0
#define GPU_THERMAL_RELAX_NATIVE_ORDER 1
#define GPU_POOL_THETA_AFTER_REJECTION 1
#define GPU_POOL_PARTICLE_THETA(s, i) particleMomentThetaDevice(s, i)
#define GPU_POOL_MASS_FALLBACK s.injectionParcelMass
#define GPU_POOL_SPLIT_REDUCTION_TOPOLOGY PoolReductionTopology::warpComponents
#define GPU_POOL_STATIC_DIRECTORY 0
#define GPU_POOL_CACHED_PROBABILITY 0
#define GPU_POOL_SPLIT_DIRECTORY (descriptor.source == static_cast<int>(CsrReductionTaskSource::splitLogical))
#define GPU_POOL_DIRECT_PARTICLE_INDEX (descriptor.source == static_cast<int>(CsrReductionTaskSource::splitBaseDirect))
#define GPU_POOL_SAMPLING_READY(s, c)
#define GPU_POOL_END_TASK(s, c, first) s.csrCellTaskOffset[c + 1]
#define GPU_DIRECTORY_PARAMETER_TYPE bool
#define GPU_DIRECTORY_ARGUMENT (splitDirectory ? 1 : 0)
#define GPU_DIRECTORY_HAS_BASE_ONLY 0
#define GPU_DIRECTORY_OWNS_TILE_POLICY 0
#define GPU_DIRECTORY_SELECTOR splitDirectory
#define GPU_DIRECTORY_FULL 0
#define GPU_CONTACT_AGE(s, i) s.pTheta[i]
#define GPU_CONTACT_TIME_ZERO 0.0
#define GPU_OPERATOR_TINY(x) x
#define GPU_OPERATOR_THERMAL 1
#define GPU_RECOVERY_INLINE 
#define GPU_PERIODIC_FACE_KIND 6
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) hit - shift + eps*normal
#define GPU_WALL_COORDINATE(hit, eps, normal) hit - eps*normal
#define GPU_RESET_CONTACT_AGE(s, i)
#define GPU_CONTACT_DURATION(value) static_cast<double>(value)
#define GPU_CONTACT_PEAK(value) static_cast<double>(value)
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) x
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuOperatorContract.cuh"
#include <cuda_runtime.h>
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuGasOperatorDependencies.cuh"

#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/applications/FSH/gpu/GpuBackendApi.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/CharacteristicMuscl.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/OpenFoamLimitedLinear.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/OpenFoamViscousFlux.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/OpenFoamWallFunctions.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/RiemannBoundaryState.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/RiemannGasFlux.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/GpuSstAlgebra.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/GpuLesAlgebra.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gasNumerics/GpuDragModelsFSHCHT.cuh"

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

#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/gpu/GpuCouplingMath.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuParticleWallContactHeat.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuCapillaryDetachment.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuSommerfeldSticking.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuFiniteWallContact.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuColdWallSolidification.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuColdWall2DSolidification.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuWallContactDirectory.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <type_traits>

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

struct CsrReductionTask
{
    int cell;
    int begin;
    int end;
    int source;
};

struct DeviceState
{
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuAutomaticCsrScheduleFields.inl"
    DeviceState* deviceState = nullptr;

    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nCellPlanes = 0;
    int particleCapacity = 0;
    int particleWorkGrid = 0;
    bool particlesMayBePresent = false;
    unsigned short particleStuckInitialisationPending = 0;
    int maxFaceWalkHops = 8;
    int* particleCountDevice = nullptr;
    double injectionParcelMass = 0.0;
    unsigned long long rngSeed = 0;
    double gammaGas = 1.4;
    double Rgas = 287.0;
    double gasCp = 0.0;
    double rhoSolid = 2500.0;
    double invRhoSolid = 0.0;
    int solveParticleTemperature = 0;
    double gasMu = 0.0;
    double gasPr = 0.4;
    double gasPrClamped = 0.4;
    double gasPrOneThird = 0.0;
    int dragModelId = 1;
    int particleGasHeatTransferModelId = 1;
    double dragResidualRe = 1.0e-3;
    double gravityX = 0.0;
    double gravityY = 0.0;
    double gravityZ = 0.0;
    int gravityEnabled = 0;
    int gasFluxScheme = 2;
    int gasReconstruction = 0;
    int gasLimiter = 1;
    int gasTimeIntegrator = 1;
    int gasRobustFallback = 1;
    int turbulenceModel = 0;
                                                                         
                                                                       
                                                              
    int hostGasFluxScheme = 2;
    int hostGasTimeIntegrator = 1;
    int hostTurbulenceModel = 0;
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
    int csrWarpAggregatedBinning = 0;
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
    int lightResidentBlocksPerSm = 1;
    int heavyResidentBlocksPerSm = 1;
    int dynamicHeavyThreshold = 256;
    int csrHeavyTileParticles = 256;
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
    unsigned char* particleStuckCandidateMask = nullptr;
    double* particleWallDepositedEnergy = nullptr;
    double* particleWallReflectedEnergy = nullptr;
    double* particleWallRepresentedContactArea = nullptr;
    double* particleWallContactAreaScale = nullptr;
    double* particleWallEffusivityByFace = nullptr;
    int particleStuckModelConfigured = 0;
    int particleWallHeatTransferEnabled = 0;
    double sommerfeldThreshold = 20.0;
    double particleWallMaximumCoverage = 0.63;
    double particleWallDepositionHeatTransferEfficiency = 1.0;
    double particleWallReflectionHeatTransferEfficiency = 1.0;
    double particleWallAdhesionEnergyScale = 1.0;
    double particleWallContactAngleCosine =
        ::cos(145.0*M_PI/180.0);
    double particleWallDensityKgM3 = 1800.0;
    double particleWallSpecificHeatJkgK = 710.0;
    double particleWallConductivityWmK = 100.0;
    int coldWallSolidificationEnabled = 0;
    int coldWall2DEnabled = 0;
    Foam::gpuThermal::ColdWallSolidificationParameters
        coldWallSolidificationParameters{};
    double* finiteContactRateTable = nullptr;
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
    double* pressureParticleMoments = nullptr;
    int* pressureParticleCount = nullptr;
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
    unsigned char* pStuck = nullptr;
    int* pStuckFaceId = nullptr;
    double* pDepositionArea = nullptr;
    double* pContactDuration = nullptr;
    double* pContactMaximumArea = nullptr;
    double* pContactPeakFraction = nullptr;
    double* pColdNodeSpecificEnthalpy = nullptr;
    double* pColdRingSolidMass = nullptr;
    double* pColdFrozenArea = nullptr;
    double* pColdContactAge = nullptr;
    double* pCold2DNodeSpecificEnthalpy = nullptr;
    double* pCold2DRingContactAge = nullptr;
    double* pCold2DFrozenArea = nullptr;
    unsigned long long* pRng = nullptr;
    unsigned long long* pOrigId = nullptr;
    int* wallBoundParticleIndex = nullptr;
    int* wallBoundParticleCountDevice = nullptr;
    int* sortedParticleIndex = nullptr;
    int* cellParticleCount = nullptr;
    int* cellParticleOffset = nullptr;
    int* compactCellOffset = nullptr;
    int* cellParticleWrite = nullptr;
                                                                            
                                                                       
    int* preBaseCellOffset = nullptr;
    int* preBaseParticleCountDevice = nullptr;
    int preBaseDirectoryReady = 0;
    int splitPreDirectoryActive = 0;
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
    // Only cells requiring multiple tasks and final reduction.
    int* csrHeavyCellList = nullptr;
    int* csrHeavyTaskCell = nullptr;
    int* csrHeavyTaskBegin = nullptr;
    int* csrHeavyTaskEnd = nullptr;
    int* csrHeavyCellTaskStart = nullptr;
    int* csrHeavyCellTaskCount = nullptr;
    double* csrHeavyPartials = nullptr;
    int* csrHeavyInjectionTaskCount = nullptr;
    int* csrHeavyInjectionTaskCursor = nullptr;
    int* csrHeavyInjectionTaskCell = nullptr;
    int* csrHeavyInjectionTaskBegin = nullptr;
    int* csrHeavyInjectionTaskEnd = nullptr;
    int* csrHeavyInjectionCellTaskStart = nullptr;
    int* csrHeavyInjectionCellTaskCount = nullptr;
    double* csrHeavyInjectionPartials = nullptr;

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

    double* compactPx = nullptr;
    double* compactPy = nullptr;
    double* compactPz = nullptr;
    double* compactPux = nullptr;
    double* compactPuxOld = nullptr;
    double* compactPuy = nullptr;
    double* compactPuyOld = nullptr;
    double* compactPuz = nullptr;
    double* compactPuzOld = nullptr;
    double* compactPT = nullptr;
    double* compactPTheta = nullptr;
    double* compactPd = nullptr;
    double* compactPm = nullptr;
    int* compactPCellId = nullptr;
    int* compactPStatus = nullptr;
    unsigned char* compactPStuck = nullptr;
    int* compactPStuckFaceId = nullptr;
    double* compactPDepositionArea = nullptr;
    double* compactPContactDuration = nullptr;
    double* compactPContactMaximumArea = nullptr;
    double* compactPContactPeakFraction = nullptr;
    double* compactPColdNodeSpecificEnthalpy = nullptr;
    double* compactPColdRingSolidMass = nullptr;
    double* compactPColdFrozenArea = nullptr;
    double* compactPColdContactAge = nullptr;
    double* compactPCold2DNodeSpecificEnthalpy = nullptr;
    double* compactPCold2DRingContactAge = nullptr;
    double* compactPCold2DFrozenArea = nullptr;
    unsigned long long* compactPRng = nullptr;
    unsigned long long* compactPOrigId = nullptr;
    int trackingWorkGrid = 0;

};

}
