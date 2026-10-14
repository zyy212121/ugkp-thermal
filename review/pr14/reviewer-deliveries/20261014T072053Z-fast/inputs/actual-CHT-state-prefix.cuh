#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/ContactCapabilityValidation.h"
#define GPU_CONTACT_SEPARATE_AGE 1
#define GPU_PAIRED_PARTICLE_HEAT 1
#define GPU_THERMAL_RELAX_NATIVE_ORDER 0
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
#define GPU_CONTACT_AGE(s, i) s.pContactAge[i]
#define GPU_CONTACT_TIME_ZERO GpuTime(0)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
#define GPU_OPERATOR_THERMAL 1
#define GPU_GAS_WALL_EXPOSURE(s, f, neighbour, kind) \
    ((neighbour) < 0 && (kind) == 2 ? gasWallExposedAreaFraction(s, f) : GPU_R(1.0))
#define GPU_RECOVERY_INLINE 
#define GPU_PERIODIC_FACE_KIND 6
#define GPU_PERIODIC_COORDINATE(hit, shift, eps, normal) insideFaceCoordinate(hit - shift, -normal, eps)
#define GPU_WALL_COORDINATE(hit, eps, normal) insideFaceCoordinate(hit, normal, eps)
#define GPU_RESET_CONTACT_AGE(s, i) s.pContactAge[i] = GpuTime(0);
#define GPU_CONTACT_DURATION(value) static_cast<GpuTime>(value)
#define GPU_CONTACT_PEAK(value) static_cast<double>(value)
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/applications/CHT/gpu/GpuWallEnergy64.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuParticleCaloric.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuPairedParticleHeat.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuParticleThermalCarry.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuParticleThermalConsumers.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuPrecisionTypes.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuOperatorContract.cuh"
#include <cuda_runtime.h>
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuGasOperatorDependencies.cuh"

#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/applications/CHT/gpu/GpuBackendApi.H"
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
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/applications/CHT/gpu/GpuParticleRadiationMath.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuParticleWallContactHeat.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuCapillaryDetachment.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuSommerfeldSticking.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuFiniteWallContact.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuColdWallSolidification.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuColdWall2DSolidification.H"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuColdWallThermalCarry.cuh"
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/wall/GpuWallContactDirectory.cuh"

#include <algorithm>
#include <type_traits>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <new>

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



constexpr int coldWallBlockThreads = 32;
constexpr int coldWallSmBlocks = 0;
constexpr int particlePayloadThreads = 128;
constexpr int particlePayloadSmBlocks = 0;

char lastError[2048] = "no GPU resident strict error";
static constexpr GpuReal OfSmall = GPU_R(2.22044604925031308085e-16);
static constexpr GpuReal OfVSmall = GPU_REAL_MIN;
static constexpr GpuReal OfGreat = GPU_R(1.0)/OfSmall;
static constexpr GpuReal OfPi = GPU_R(3.141592653589793238462643383279502884);

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
    GpuWallEnergyState wallEnergy;
#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/common/GpuAutomaticCsrScheduleFields.inl"
    DeviceState* deviceState = nullptr;

    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nCellPlanes = 0;
    int particleCapacity = 0;
    int particleWorkGrid = 0;
    bool particlesMayBePresent = false;
    // Host-only progress for real restore allocation/initialization/publication.
    unsigned char particleWallEnergyRestoreProgress = 0;
    unsigned short particleStuckInitialisationPending = 0;
    int maxFaceWalkHops = 8;
    int* particleCountDevice = nullptr;
    GpuReal injectionParcelMass = GPU_R(0.0);
    unsigned long long rngSeed = 0;
    GpuReal gammaGas = GPU_R(1.4);
    GpuReal Rgas = GPU_R(287.0);
    GpuReal gasCp = GPU_R(0.0);
    GpuReal rhoSolid = GPU_R(2500.0);
    GpuReal invRhoSolid = GPU_R(0.0);
    int solveParticleTemperature = 0;
    GpuReal gasMu = GPU_R(0.0);
    GpuReal gasPr = GPU_R(0.4);
    GpuReal gasPrClamped = GPU_R(0.4);
    GpuReal gasPrOneThird = GPU_R(0.0);
    int dragModelId = 1;
    int particleGasHeatTransferModelId = 1;
    GpuReal dragResidualRe = GPU_R(1.0e-3);
    GpuReal gravityX = GPU_R(0.0);
    GpuReal gravityY = GPU_R(0.0);
    GpuReal gravityZ = GPU_R(0.0);
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
    GpuReal lesDeltaCoeff = GPU_R(1.0);
    GpuReal turbulentPrandtl = GPU_R(0.9);
    GpuReal waleCw = GPU_R(0.325);
    GpuReal smagorinskyCs = GPU_R(0.17);
    GpuReal maxDiffusionNumber = GPU_R(0.25);
    int sstConfigured = 0;
    ugkwp::SstCoefficients sstCoefficients =
        ugkwp::defaultSstCoefficients();
    GpuReal sstKMin = GPU_R(1.0e-12);
    GpuReal sstOmegaMin = GPU_R(1.0e-6);
    GpuReal sstMaxSourceNumber = GPU_R(0.25);
    int sstWallTreatment = 0;
    GpuReal sstWallKappa = GPU_R(0.41);
    GpuReal sstWallE = GPU_R(9.8);
    GpuReal sstWallCmu = GPU_R(0.09);
    GpuReal sstJayatillekeP = GPU_R(0.0);
    GpuReal sstThermalYPlus = GPU_R(0.0);
    GpuReal particleDiameterFallback = GPU_R(0.0);
    GpuReal particleDiameterMin = GPU_R(0.0);
    GpuReal particleDiameterMax = GPU_R(0.0);
    GpuReal particleDiameterSigma = GPU_R(0.0);
    GpuReal injectionTheta = GPU_R(0.0);
    GpuReal rhoMin = GPU_R(1.0e-12);
    GpuReal TgasMin = GPU_R(1.0e-12);
    GpuReal epsSMin = GPU_R(1.0e-12);
    GpuReal thetaMin = GPU_R(1.0e-12);
    GpuReal TpMin = GPU_R(1.0e-12);
    GpuReal TpMax = GPU_R(1.0e30);
    int collisionalPressureEnabled = 0;
    GpuReal collisionalRestitution = GPU_R(0.9);
    GpuReal pressureKickFraction = GPU_R(0.25);
    int jammingPressureEnabled = 0;
    GpuReal packingFraction = GPU_R(0.63);
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
    GpuTime gasGraphDt = -1;
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
    GpuReal* facePeriodicDx = nullptr;
    GpuReal* facePeriodicDy = nullptr;
    GpuReal* facePeriodicDz = nullptr;
    GpuReal* V = nullptr;
    GpuReal* Cx = nullptr;
    GpuReal* Cy = nullptr;
    GpuReal* Cz = nullptr;
    GpuReal* faceCx = nullptr;
    GpuReal* faceCy = nullptr;
    GpuReal* faceCz = nullptr;
    GpuReal* Sfx = nullptr;
    GpuReal* Sfy = nullptr;
    GpuReal* Sfz = nullptr;
    GpuReal* magSf = nullptr;
    GpuReal* deltaCoeffs = nullptr;
    GpuReal* faceWeight = nullptr;
    GpuReal* cellLength = nullptr;
    int* cellPlaneStart = nullptr;
    int* cellPlaneCount = nullptr;
    int* cellFaceId = nullptr;
    int* cellFaceNeighbor = nullptr;
    int* cellFaceKind = nullptr;
    GpuReal* cellFaceRestitution = nullptr;
    GpuReal* cellFaceTangential = nullptr;
    GpuReal* planeNx = nullptr;
    GpuReal* planeNy = nullptr;
    GpuReal* planeNz = nullptr;
    GpuReal* planeD = nullptr;

    GpuReal* rho = nullptr;
    GpuReal* rhoUx = nullptr;
    GpuReal* rhoUy = nullptr;
    GpuReal* rhoUz = nullptr;
    double* rhoE = nullptr; // authoritative energy density, independent of GpuReal
    GpuReal* rhoNext = nullptr;
    GpuReal* rhoUxNext = nullptr;
    GpuReal* rhoUyNext = nullptr;
    GpuReal* rhoUzNext = nullptr;
    double* rhoENext = nullptr; // full-precision RK initial state
    GpuReal* gasPhiRho = nullptr;
    GpuReal* gasPhiRhoUx = nullptr;
    GpuReal* gasPhiRhoUy = nullptr;
    GpuReal* gasPhiRhoUz = nullptr;
    GpuReal* gasPhiRhoE = nullptr;
    GpuReal* gasFluxPositivityScale = nullptr;
    GpuReal* gasHllcAdcSensor = nullptr;
    unsigned char* particleStuckCandidateMask = nullptr;
    GpuWallEnergy* particleWallDepositedEnergy = nullptr;
    GpuWallEnergy* particleWallReflectedEnergy = nullptr;
    GpuReal* particleWallRepresentedContactArea = nullptr;
    GpuReal* particleWallContactAreaScale = nullptr;
    GpuReal* particleWallEffusivityByFace = nullptr;
    int particleStuckModelConfigured = 0;
    int particleWallHeatTransferEnabled = 0;
    GpuReal sommerfeldThreshold = GPU_R(20.0);
    GpuReal particleWallMaximumCoverage = GPU_R(0.63);
    GpuReal particleWallDepositionHeatTransferEfficiency = GPU_R(1.0);
    GpuReal particleWallReflectionHeatTransferEfficiency = GPU_R(1.0);
    GpuReal particleWallAdhesionEnergyScale = GPU_R(1.0);
    GpuReal particleWallContactAngleCosine =
        ::cos(GPU_R(145.0)*M_PI/GPU_R(180.0));
    GpuReal particleWallDensityKgM3 = GPU_R(1800.0);
    GpuReal particleWallSpecificHeatJkgK = GPU_R(710.0);
    GpuReal particleWallConductivityWmK = GPU_R(100.0);
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
    GpuReal* gasBoundaryPWaveGamma = nullptr;
    GpuReal* gasBoundaryPWaveFieldInf = nullptr;
    GpuReal* gasBoundaryPWaveLInf = nullptr;
    GpuReal* gasBoundaryRho = nullptr;
    GpuReal* gasBoundaryUx = nullptr;
    GpuReal* gasBoundaryUy = nullptr;
    GpuReal* gasBoundaryUz = nullptr;
    GpuReal* gasBoundaryP = nullptr;
    GpuReal* gasBoundaryT = nullptr;
                                                                           
                                                                              
                                                                            
    int* riemannBoundaryKind = nullptr;
    int* riemannBoundaryRhoFix = nullptr;
    int* riemannBoundaryUFix = nullptr;
    int* riemannBoundaryPFix = nullptr;
    int* riemannBoundaryTFix = nullptr;
    int* riemannBoundaryPWave = nullptr;
    GpuReal* riemannBoundaryPWaveGamma = nullptr;
    GpuReal* riemannBoundaryPWaveFieldInf = nullptr;
    GpuReal* riemannBoundaryPWaveLInf = nullptr;
    GpuReal* riemannBoundaryRho = nullptr;
    GpuReal* riemannBoundaryUx = nullptr;
    GpuReal* riemannBoundaryUy = nullptr;
    GpuReal* riemannBoundaryUz = nullptr;
    GpuReal* riemannBoundaryP = nullptr;
    GpuReal* riemannBoundaryT = nullptr;
    int nScheduledInletFaces = 0;
    int* scheduledInletFaceMask = nullptr;
    GpuReal scheduledInletTemperature = GPU_R(0.0);
    int nPressureScheduleRows = 0;
    GpuTime* pressureScheduleTimes = nullptr;
    GpuReal* pressureScheduleValues = nullptr;
    int nVolumeFractionScheduleRows = 0;
    GpuTime* volumeFractionScheduleTimes = nullptr;
    GpuReal* volumeFractionScheduleValues = nullptr;
    GpuReal* gradPx = nullptr;
    GpuReal* gradPy = nullptr;
    GpuReal* gradPz = nullptr;
    GpuReal* gradRhoX = nullptr;
    GpuReal* gradRhoY = nullptr;
    GpuReal* gradRhoZ = nullptr;
    GpuReal* gradUxX = nullptr;
    GpuReal* gradUxY = nullptr;
    GpuReal* gradUxZ = nullptr;
    GpuReal* gradUyX = nullptr;
    GpuReal* gradUyY = nullptr;
    GpuReal* gradUyZ = nullptr;
    GpuReal* gradUzX = nullptr;
    GpuReal* gradUzY = nullptr;
    GpuReal* gradUzZ = nullptr;
    GpuReal* gradTX = nullptr;
    GpuReal* gradTY = nullptr;
    GpuReal* gradTZ = nullptr;
                                                                     
                                                                   
                                                                           
                                                                  
    GpuReal* gasGradientLimiterRho = nullptr;
    GpuReal* gasGradientLimiterUx = nullptr;
    GpuReal* gasGradientLimiterUy = nullptr;
    GpuReal* gasGradientLimiterUz = nullptr;
    GpuReal* gasGradientLimiterP = nullptr;
    GpuReal* gasGradientLimiterT = nullptr;
    GpuReal* nut = nullptr;
    GpuReal* gasDiffusionNumber = nullptr;
    GpuReal* rhoK = nullptr;
    GpuReal* rhoOmega = nullptr;
    GpuReal* rhoKInitial = nullptr;
    GpuReal* rhoOmegaInitial = nullptr;
    GpuReal* k = nullptr;
    GpuReal* omega = nullptr;
    GpuReal* sstWallDistance = nullptr;
    GpuReal* sstF1 = nullptr;
    GpuReal* sstF2 = nullptr;
    GpuReal* gradKX = nullptr;
    GpuReal* gradKY = nullptr;
    GpuReal* gradKZ = nullptr;
    GpuReal* gradOmegaX = nullptr;
    GpuReal* gradOmegaY = nullptr;
    GpuReal* gradOmegaZ = nullptr;
    GpuReal* sstPhiRhoK = nullptr;
    GpuReal* sstPhiRhoOmega = nullptr;
    int* sstBoundaryKMode = nullptr;
    int* sstBoundaryOmegaMode = nullptr;
    GpuReal* sstBoundaryK = nullptr;
    GpuReal* sstBoundaryOmega = nullptr;
    GpuReal* sstSourceNumber = nullptr;
    GpuReal* Ux = nullptr;
    GpuReal* Uy = nullptr;
    GpuReal* Uz = nullptr;
    GpuReal* p = nullptr;
    GpuReal* Tgas = nullptr;
    GpuReal* couplingRhoOld = nullptr;
    GpuReal* couplingUxOld = nullptr;
    GpuReal* couplingUyOld = nullptr;
    GpuReal* couplingUzOld = nullptr;
    GpuReal* couplingTgasOld = nullptr;
    double* particleHeatCoefficient = nullptr;
    double* particleHeatBudget = nullptr;

    GpuReal* epsS = nullptr;
    GpuReal* rhoUsx = nullptr;
    GpuReal* rhoUsy = nullptr;
    GpuReal* rhoUsz = nullptr;
    GpuReal* rhoEs = nullptr;
    GpuReal* rhoDs = nullptr;
    GpuReal* rhoHp = nullptr;
    GpuReal* Usx = nullptr;
    GpuReal* Usy = nullptr;
    GpuReal* Usz = nullptr;
    GpuReal* theta = nullptr;
    GpuReal* Tp = nullptr;
    GpuReal* dMeanCell = nullptr;
    GpuReal* epsGPrev = nullptr;
    GpuReal* collisionalPressure = nullptr;
    GpuReal* pressureKickScale = nullptr;
    GpuReal* pressureDeltaMomX = nullptr;
    GpuReal* pressureDeltaMomY = nullptr;
    GpuReal* pressureDeltaMomZ = nullptr;
    GpuReal* pressureDeltaEnergy = nullptr;
    GpuReal* pressureParticleMoments = nullptr;
    int* pressureParticleCount = nullptr;
    GpuReal* solidPressurePhiMomX = nullptr;
    GpuReal* solidPressurePhiMomY = nullptr;
    GpuReal* solidPressurePhiMomZ = nullptr;
    GpuReal* solidPressurePhiEnergy = nullptr;
                                                                             
                                                                       
    GpuReal* mobilePackingRho = nullptr;
    GpuReal* packingStuckRho = nullptr;
    GpuReal* mobilePackingMomX = nullptr;
    GpuReal* mobilePackingMomY = nullptr;
    GpuReal* mobilePackingMomZ = nullptr;
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
    
                                                                     
                                                                                
                                                            
    GpuReal* thetaDragAlpha = nullptr;

    GpuReal* momRhoP = nullptr;
    GpuReal* momRhoUPx = nullptr;
    GpuReal* momRhoUPy = nullptr;
    GpuReal* momRhoUPz = nullptr;
    GpuReal* momRhoEP = nullptr;
    GpuReal* momRhoPD = nullptr;
    GpuReal* momRhoHpP = nullptr;
    GpuReal* radiationMobileMass = nullptr;
    GpuReal* radiationMobileTemperatureMass = nullptr;
    GpuReal* radiationMobileDiameterMass = nullptr;
    int* poolThermalCount = nullptr;
    GpuReal* poolThermalSumUx = nullptr;
    GpuReal* poolThermalSumUy = nullptr;
    GpuReal* poolThermalSumUz = nullptr;
    GpuReal* poolThermalSumU2 = nullptr;
    int* poissonPoolSampleTargetCount = nullptr;
    GpuReal* poissonPoolMass = nullptr;
    GpuReal* poissonPoolMomX = nullptr;
    GpuReal* poissonPoolMomY = nullptr;
    GpuReal* poissonPoolMomZ = nullptr;
    GpuReal* poissonPoolEnergy = nullptr;
    GpuReal* poissonPoolDiameter = nullptr;
    GpuReal* poissonPoolDiameter2 = nullptr;

    GpuReal* px = nullptr;
    GpuReal* py = nullptr;
    GpuReal* pz = nullptr;
    GpuReal* pux = nullptr;
    GpuReal* puy = nullptr;
    GpuReal* puz = nullptr;
    GpuReal* puxOld = nullptr;
    GpuReal* puyOld = nullptr;
    GpuReal* puzOld = nullptr;
    GpuReal* pT = nullptr;
    double* pThermalEnthalpyResidual = nullptr;
    GpuReal* pTheta = nullptr;
    GpuTime* pContactAge = nullptr;
    GpuReal* pd = nullptr;
    GpuReal* pm = nullptr;
    int* pCellId = nullptr;
    int* pStatus = nullptr;
    unsigned char* pStuck = nullptr;
    int* pStuckFaceId = nullptr;
    double* pDepositionArea = nullptr;
    double* pContactDuration = nullptr;
    double* pContactMaximumArea = nullptr;
    double* pContactPeakFraction = nullptr;
    float* pColdNodeSpecificEnthalpy = nullptr;
    double* pColdRingSolidMass = nullptr;
    double* pColdFrozenArea = nullptr;
    double* pColdContactAge = nullptr;
    float* pCold2DNodeSpecificEnthalpy = nullptr;
    double* pCold2DRingContactAge = nullptr;
    double* pCold2DFrozenArea = nullptr;
    unsigned long long* pRng = nullptr;
    unsigned long long* pOrigId = nullptr;
    Foam::gpuThermal::RadiationAffineTemperatureUpdate*
        particleRadiationAffineUpdate = nullptr;
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
    GpuReal* csrHeavyPartials = nullptr;
    int* csrHeavyInjectionTaskCount = nullptr;
    int* csrHeavyInjectionTaskCursor = nullptr;
    int* csrHeavyInjectionTaskCell = nullptr;
    int* csrHeavyInjectionTaskBegin = nullptr;
    int* csrHeavyInjectionTaskEnd = nullptr;
    int* csrHeavyInjectionCellTaskStart = nullptr;
    int* csrHeavyInjectionCellTaskCount = nullptr;
    GpuReal* csrHeavyInjectionPartials = nullptr;

    int nBoundarySources = 0;
    int* sourceCell = nullptr;
    int* sourceFace = nullptr;
    GpuReal* sourcePx = nullptr;
    GpuReal* sourcePy = nullptr;
    GpuReal* sourcePz = nullptr;
    GpuReal* sourceUx = nullptr;
    GpuReal* sourceUy = nullptr;
    GpuReal* sourceUz = nullptr;
    GpuReal* sourceT = nullptr;
    GpuReal* sourceTheta = nullptr;
    GpuReal* sourceD = nullptr;
    GpuReal* sourceMassRate = nullptr;
    GpuReal* sourceResidualMass = nullptr;

    GpuReal* compactPx = nullptr;
    GpuReal* compactPy = nullptr;
    GpuReal* compactPz = nullptr;
    GpuReal* compactPux = nullptr;
    GpuReal* compactPuxOld = nullptr;
    GpuReal* compactPuy = nullptr;
    GpuReal* compactPuyOld = nullptr;
    GpuReal* compactPuz = nullptr;
    GpuReal* compactPuzOld = nullptr;
    GpuReal* compactPT = nullptr;
    double* compactPThermalEnthalpyResidual = nullptr;
    GpuReal* compactPTheta = nullptr;
    GpuTime* compactPContactAge = nullptr;
    GpuReal* compactPd = nullptr;
    GpuReal* compactPm = nullptr;
    int* compactPCellId = nullptr;
    int* compactPStatus = nullptr;
    unsigned char* compactPStuck = nullptr;
    int* compactPStuckFaceId = nullptr;
    double* compactPDepositionArea = nullptr;
    double* compactPContactDuration = nullptr;
    double* compactPContactMaximumArea = nullptr;
    double* compactPContactPeakFraction = nullptr;
    float* compactPColdNodeSpecificEnthalpy = nullptr;
    double* compactPColdRingSolidMass = nullptr;
    double* compactPColdFrozenArea = nullptr;
    double* compactPColdContactAge = nullptr;
    float* compactPCold2DNodeSpecificEnthalpy = nullptr;
    double* compactPCold2DRingContactAge = nullptr;
    double* compactPCold2DFrozenArea = nullptr;
    unsigned long long* compactPRng = nullptr;
    unsigned long long* compactPOrigId = nullptr;
    // Cold-wall launch geometry must not inherit reduction-kernel occupancy.
    int coldWallWorkGrid = 0;
    int trackingWorkGrid = 0;

};

}
