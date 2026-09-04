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
#include "GpuParticleRadiationMath.H"
#include "../thermal/GpuParticleWallContactHeat.H"
#include "../thermal/GpuCapillaryDetachment.H"
#include "../thermal/GpuSommerfeldSticking.H"
#include "../thermal/GpuFiniteWallContact.H"
#include "../../../common/wall/GpuColdWallSolidification.H"
#include "../../../common/wall/GpuColdWall2DSolidification.H"
#include "../../../common/wall/GpuWallContactDirectory.cuh"

#include <algorithm>
#include <cfloat>
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

struct CsrReductionTask
{
    int cell;
    int begin;
    int end;
    int source;
};

struct ParticleRadiationValidationError
{
    int code = 0;
    int particleArrayIndex = -1;
    int cellId = -1;
    unsigned long long particleOriginalId = 0;
    double oldTemperatureK = 0.0;
    double scale = 0.0;
    double offset = 0.0;
    double proposedTemperatureK = 0.0;
};

struct WallContactAreaDiagnosticError
{
    int code = 0;
    int directoryEntry = -1;
    int directoryCount = 0;
    int particleArrayIndex = -1;
    int particleStatus = -1;
    int wallState = -1;
    int faceId = -1;
    int faceCount = 0;
    int candidateType = -1;
    unsigned long long particleOriginalId = 0;
    double depositionArea = 0.0;
    double contactDuration = 0.0;
    double contactMaximumArea = 0.0;
    double contactPeakFraction = 0.0;
    double contactAge = 0.0;
    double frozenArea = 0.0;
    double kinematicArea = 0.0;
    double physicalContactArea = 0.0;
    double representedArea = 0.0;
    double diameter = 0.0;
    double parcelMass = 0.0;
    double solidDensity = 0.0;
    double temperature = 0.0;
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
    int csrHeavyReductionEnabled = 0;
    int csrHeavyReductionMode = 0;
    int csrHeavyAutoInterval = 100;
    int csrHeavyReductionActive = 0;
    unsigned long long schedulingAdvanceCount = 0;
    int csrWarpAggregatedBinning = 0;
    int fixedCellBlockThreads = 128;
    int fixedFaceBlockThreads = 128;
    int fixedWorkBlockTuned = 0;
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
    double* gasWallEnergy = nullptr;
    unsigned char* gasWallEnergyMask = nullptr;
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
    float* finiteContactRateTable = nullptr;
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
    double* radiationMobileMass = nullptr;
    double* radiationMobileTemperatureMass = nullptr;
    double* radiationMobileDiameterMass = nullptr;
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
    float* pDepositionArea = nullptr;
    float* pContactDuration = nullptr;
    float* pContactMaximumArea = nullptr;
    float* pContactPeakFraction = nullptr;
    float* pColdNodeSpecificEnthalpy = nullptr;
    float* pColdRingSolidMass = nullptr;
    float* pColdFrozenArea = nullptr;
    float* pColdContactAge = nullptr;
    float* pCold2DNodeSpecificEnthalpy = nullptr;
    float* pCold2DRingContactAge = nullptr;
    float* pCold2DFrozenArea = nullptr;
    unsigned long long* pRng = nullptr;
    unsigned long long* pOrigId = nullptr;
    Foam::gpuThermal::RadiationAffineTemperatureUpdate*
        particleRadiationAffineUpdate = nullptr;
    ParticleRadiationValidationError* particleRadiationValidationError = nullptr;
    int wallContactAreaDiagnosticsEnabled = 0;
    WallContactAreaDiagnosticError* wallContactAreaDiagnosticError = nullptr;
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
    int* csrHeavyTaskCount = nullptr;
    int* csrHeavyTaskCursor = nullptr;
    int* csrHeavyCellCount = nullptr;
    int* csrMaximumOccupancy = nullptr;
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
    double* compactPuy = nullptr;
    double* compactPuz = nullptr;
    double* compactPT = nullptr;
    double* compactPTheta = nullptr;
    double* compactPd = nullptr;
    double* compactPm = nullptr;
    int* compactPCellId = nullptr;
    int* compactPStatus = nullptr;
    unsigned char* compactPStuck = nullptr;
    int* compactPStuckFaceId = nullptr;
    float* compactPDepositionArea = nullptr;
    float* compactPContactDuration = nullptr;
    float* compactPContactMaximumArea = nullptr;
    float* compactPContactPeakFraction = nullptr;
    float* compactPColdNodeSpecificEnthalpy = nullptr;
    float* compactPColdRingSolidMass = nullptr;
    float* compactPColdFrozenArea = nullptr;
    float* compactPColdContactAge = nullptr;
    float* compactPCold2DNodeSpecificEnthalpy = nullptr;
    float* compactPCold2DRingContactAge = nullptr;
    float* compactPCold2DFrozenArea = nullptr;
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
    int particlePath = 0;
    int particleCount = -1;
    int particleCapacity = 0;
    double particleUtilisation = 0.0;
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
    cudaEvent_t events[ProbeStageCount + 1]{};
    DevelopmentProbeDeviceSummary* deviceSummary = nullptr;
    std::vector<int> occupancy;
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

int syncGasWallLedgerPointers(DeviceState* s, const char* name)
{
    static_assert
    (
        offsetof(DeviceState, gasWallEnergyMask)
     == offsetof(DeviceState, gasWallEnergy) + sizeof(double*),
        "gas-wall ledger pointers must remain adjacent"
    );
    const cudaError_t err = cudaMemcpy
    (
        &(s->deviceState->gasWallEnergy),
        &(s->gasWallEnergy),
        sizeof(s->gasWallEnergy) + sizeof(s->gasWallEnergyMask),
        cudaMemcpyHostToDevice
    );
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
    release(s->gasWallEnergy);
    release(s->gasWallEnergyMask);
    release(s->gasFluxPositivityScale);
    release(s->gasHllcAdcSensor);
    release(s->particleStuckCandidateMask);
    release(s->finiteContactRateTable);
    release(s->pColdNodeSpecificEnthalpy);
    release(s->pColdRingSolidMass);
    release(s->pColdFrozenArea);
    release(s->pColdContactAge);
    release(s->pCold2DNodeSpecificEnthalpy);
    release(s->pCold2DRingContactAge);
    release(s->pCold2DFrozenArea);
    release(s->particleWallDepositedEnergy);
    release(s->particleWallReflectedEnergy);
    release(s->particleWallRepresentedContactArea);
    release(s->particleWallContactAreaScale);
    release(s->particleWallEffusivityByFace);
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
    release(s->radiationMobileMass);
    release(s->radiationMobileTemperatureMass);
    release(s->radiationMobileDiameterMass);
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
    release(s->pStuck);
    release(s->pStuckFaceId);
    release(s->pDepositionArea);
    release(s->pContactDuration);
    release(s->pContactMaximumArea);
    release(s->pContactPeakFraction);
    release(s->pRng);
    release(s->pOrigId);
    release(s->particleRadiationAffineUpdate);
    release(s->particleRadiationValidationError);
    release(s->wallContactAreaDiagnosticError);
    release(s->wallBoundParticleIndex);
    release(s->wallBoundParticleCountDevice);
    release(s->sortedParticleIndex);
    release(s->cellParticleCount);
    release(s->cellParticleOffset);
    release(s->compactCellOffset);
    release(s->cellParticleWrite);
    release(s->preBaseCellOffset);
    release(s->preBaseParticleCountDevice);
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
    release(s->csrMaximumOccupancy);
    release(s->csrHeavyCellList);
    release(s->csrHeavyTaskCell);
    release(s->csrHeavyTaskBegin);
    release(s->csrHeavyTaskEnd);
    release(s->csrHeavyCellTaskStart);
    release(s->csrHeavyCellTaskCount);
    release(s->csrHeavyPartials);
    release(s->csrHeavyInjectionTaskCount);
    release(s->csrHeavyInjectionTaskCursor);
    release(s->csrHeavyInjectionTaskCell);
    release(s->csrHeavyInjectionTaskBegin);
    release(s->csrHeavyInjectionTaskEnd);
    release(s->csrHeavyInjectionCellTaskStart);
    release(s->csrHeavyInjectionCellTaskCount);
    release(s->csrHeavyInjectionPartials);

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
    release(s->compactPStuck);
    release(s->compactPStuckFaceId);
    release(s->compactPDepositionArea);
    release(s->compactPContactDuration);
    release(s->compactPContactMaximumArea);
    release(s->compactPContactPeakFraction);
    release(s->compactPColdNodeSpecificEnthalpy);
    release(s->compactPColdRingSolidMass);
    release(s->compactPColdFrozenArea);
    release(s->compactPColdContactAge);
    release(s->compactPCold2DNodeSpecificEnthalpy);
    release(s->compactPCold2DRingContactAge);
    release(s->compactPCold2DFrozenArea);
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
    rc |= allocate(s->radiationMobileMass, nc, "cudaMalloc mobile radiation mass");
    rc |= allocate(s->radiationMobileTemperatureMass, nc, "cudaMalloc mobile radiation temperature mass");
    rc |= allocate(s->radiationMobileDiameterMass, nc, "cudaMalloc mobile radiation diameter mass");
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
    rc |= allocate(s->pStuck, np, "cudaMalloc strict particle stuck");
    rc |= allocate(s->pStuckFaceId, np, "cudaMalloc strict particle stuck face");
    rc |= allocate(s->pDepositionArea, np, "cudaMalloc strict particle deposition area");
    rc |= allocate(s->pContactDuration, np, "cudaMalloc finite contact duration");
    rc |= allocate(s->pContactMaximumArea, np, "cudaMalloc finite contact maximum area");
    rc |= allocate(s->pContactPeakFraction, np, "cudaMalloc finite contact peak fraction");
    rc |= allocate(s->pRng, np, "cudaMalloc strict particle rng");
    rc |= allocate(s->pOrigId, np, "cudaMalloc strict particle origId");
    rc |= allocate(s->wallBoundParticleIndex, np, "cudaMalloc wall-bound particle index");
    rc |= allocate(s->wallBoundParticleCountDevice, 1, "cudaMalloc wall-bound particle count");
    rc |= allocate(s->sortedParticleIndex, np, "cudaMalloc strict sorted particle index");
    rc |= allocate(s->cellParticleCount, nc + 1, "cudaMalloc strict cellParticleCount");
    rc |= allocate(s->cellParticleOffset, nc + 1, "cudaMalloc strict cellParticleOffset");
    rc |= allocate(s->compactCellOffset, nc + 1, "cudaMalloc strict compactCellOffset");
    rc |= allocate(s->cellParticleWrite, nc, "cudaMalloc strict cellParticleWrite");
    rc |= allocate(s->preBaseCellOffset, nc + 1, "cudaMalloc split-Dpre base offsets");
    rc |= allocate(s->preBaseParticleCountDevice, 1, "cudaMalloc split-Dpre base count");
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
        rc |= allocate(s->csrMaximumOccupancy, 1, "cudaMalloc CSR maximum occupancy");
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
    rc |= allocate(s->compactPStuck, np, "cudaMalloc strict compact particle stuck");
    rc |= allocate(s->compactPStuckFaceId, np, "cudaMalloc strict compact particle stuck face");
    rc |= allocate(s->compactPDepositionArea, np, "cudaMalloc strict compact particle deposition area");
    rc |= allocate(s->compactPContactDuration, np, "cudaMalloc compact finite contact duration");
    rc |= allocate(s->compactPContactMaximumArea, np, "cudaMalloc compact finite contact maximum area");
    rc |= allocate(s->compactPContactPeakFraction, np, "cudaMalloc compact finite contact peak fraction");
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
                                                                        
                                                                            
                                                                      
                                    
    const int capacityGrid =
        (s->particleCapacity + s->particleBlockThreads - 1)
       /s->particleBlockThreads;
    const int saturatedGrid = multiprocessorCount;
    s->particleWorkGrid = capacityGrid < saturatedGrid
      ? capacityGrid
      : saturatedGrid;
    s->multiprocessorCount = multiprocessorCount;
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
    err = cudaMemset(s->preBaseParticleCountDevice, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->wallBoundParticleCountDevice, 0, sizeof(int));
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset
        (
            s->preBaseCellOffset,
            0,
            (static_cast<size_t>(s->nCells) + 1u)*sizeof(int)
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("initialise split-Dpre base directory", err);
        releaseState(s);
        return 1;
    }
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
         || s.pm[i] < 0.0
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
         || s.pStuck[i] > Foam::gpuThermal::particleWallTransientDeposit
         || (
                s.pStuck[i] == Foam::gpuThermal::particleWallDeposited
             && (
                    !(s.pDepositionArea[i] > 0.0f)
                 || !
                    (
                        (
                            s.pContactDuration[i] == 0.0f
                         && s.pContactMaximumArea[i] == 0.0f
                         && s.pContactPeakFraction[i] == 0.0f
                        )
                     || (
                            s.pContactDuration[i] > 0.0f
                         && s.pContactMaximumArea[i] > 0.0f
                         && s.pContactPeakFraction[i] > 0.0f
                         && s.pContactPeakFraction[i] < 1.0f
                        )
                    )
                )
            )
         || (
                (
                    s.pStuck[i]
                 == Foam::gpuThermal::particleWallTransientRebound
                 || s.pStuck[i]
                 == Foam::gpuThermal::particleWallTransientDeposit
                )
             && (
                    !(s.pContactDuration[i] > 0.0f)
                 || !(s.pContactMaximumArea[i] > 0.0f)
                 || !(s.pContactPeakFraction[i] > 0.0f)
                 || !(s.pContactPeakFraction[i] < 1.0f)
                 || s.pTheta[i] > s.pContactDuration[i]
                 || s.pDepositionArea[i] < 0.0f
                )
            )
        )
        {
            mask |= ProbeBadParticleMetadata;
        }

        recordDevelopmentProbeParticleFailure(summary, i, mask);
    }
}

#endif

__device__ double particleSpecificHeatDevice(const double temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificHeatJkgK(temperatureK);
}

__device__ double particleSpecificEnthalpyDevice(const double temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificEnthalpyJkg(temperatureK);
}

                                                                          
                                                                           
                                                                               
                                                                              
                                                                               
__device__ double particleMomentThetaDevice
(
    const DeviceState& s,
    const int i
)
{
    return s.pStuck[i] == Foam::gpuThermal::particleWallMobile
      ? clampMin(finiteOr(s.pTheta[i], 0.0), 0.0)
      : 0.0;
}

__device__ double particleTemperatureFromSpecificEnthalpyDevice
(
    const double specificEnthalpyJkg
)
{
    return Foam::gpuThermal::aluminaTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg
    );
}

enum ParticleRadiationValidationCode
{
    particleRadiationValid = 0,
    particleRadiationInvalidCell = 1,
    particleRadiationNonfiniteOldTemperature = 2,
    particleRadiationNonfiniteAffineUpdate = 3,
    particleRadiationNonfiniteProposedTemperature = 4,
    particleRadiationProposedTemperatureOutOfBounds = 5
};

__global__ void validateParticleRadiationAffineTemperatureKernel
(
    DeviceState* sp,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell,
    ParticleRadiationValidationError* validationError
)
{
    DeviceState& s = *sp;
    const int nParticles = *s.particleCountDevice;
    const int stride = blockDim.x*gridDim.x;
    for
    (
        int particleI = blockIdx.x*blockDim.x + threadIdx.x;
        particleI < nParticles;
        particleI += stride
    )
    {
        if
        (
            s.pStatus[particleI] != 1
         || s.pStuck[particleI] != Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const int cellI = s.pCellId[particleI];
        const double oldTemperatureK = s.pT[particleI];
        double scale = nan("");
        double offset = nan("");
        double proposedTemperatureK = nan("");
        int errorCode = particleRadiationValid;
        if (cellI < 0 || cellI >= s.nCells)
        {
            errorCode = particleRadiationInvalidCell;
        }
        else if (!isfinite(oldTemperatureK))
        {
            errorCode = particleRadiationNonfiniteOldTemperature;
        }
        else
        {
            const Foam::gpuThermal::RadiationAffineTemperatureUpdate update =
                updateByCell[cellI];
            scale = update.scale;
            offset = update.offset;
            proposedTemperatureK =
                Foam::gpuThermal::applyRadiationAffineTemperature
                (
                    oldTemperatureK,
                    update
                );
            if (!isfinite(scale) || !isfinite(offset))
            {
                errorCode = particleRadiationNonfiniteAffineUpdate;
            }
            else if (!isfinite(proposedTemperatureK))
            {
                errorCode = particleRadiationNonfiniteProposedTemperature;
            }
            else if
            (
                proposedTemperatureK < s.TpMin
             || proposedTemperatureK > s.TpMax
            )
            {
                errorCode = particleRadiationProposedTemperatureOutOfBounds;
            }
        }
        if
        (
            errorCode != particleRadiationValid
         && atomicCAS(&(validationError->code), 0, errorCode) == 0
        )
        {
            validationError->particleArrayIndex = particleI;
            validationError->cellId = cellI;
            validationError->particleOriginalId = s.pOrigId[particleI];
            validationError->oldTemperatureK = oldTemperatureK;
            validationError->scale = scale;
            validationError->offset = offset;
            validationError->proposedTemperatureK = proposedTemperatureK;
        }
    }
}

__global__ void applyParticleRadiationAffineTemperatureKernel
(
    DeviceState* sp,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
)
{
    DeviceState& s = *sp;
    const int nParticles = *s.particleCountDevice;
    const int stride = blockDim.x*gridDim.x;
    for
    (
        int particleI = blockIdx.x*blockDim.x + threadIdx.x;
        particleI < nParticles;
        particleI += stride
    )
    {
        if
        (
            s.pStatus[particleI] != 1
         || s.pStuck[particleI] != Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const int cellI = s.pCellId[particleI];
        s.pT[particleI] = Foam::gpuThermal::applyRadiationAffineTemperature
        (
            s.pT[particleI],
            updateByCell[cellI]
        );
    }
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
        const int other = isPeriodicFace(s, f) ? nei : (c == own ? nei : own);
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
            const int other = isPeriodicFace(s, f) ? nei : (c == own ? nei : own);
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
            const int other = isPeriodicFace(s, f) ? nei : (own == c ? nei : own);
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
            const int other = isPeriodicFace(s, f) ? nei : (c == own ? nei : own);
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
            const double symmetricGradient = 0.5*(g[i][j] + g[j][i]);
            symmetricGradientSquared += symmetricGradient*symmetricGradient;
            strain[i][j] = symmetricGradient
              - (i == j ? tr/3.0 : 0.0);
            ss += strain[i][j]*strain[i][j];
        }
    }
    const double delta = s.lesDeltaCoeff*clampMin(s.cellLength[c], OfSmall);
    double nuT = 0.0;
    if (s.turbulenceModel == 2)
    {
        nuT = ugkwp::smagorinskyNut(s.smagorinskyCs, delta, ss);
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
        nuT = ugkwp::waleNut
        (
            s.waleCw,
            delta,
            symmetricGradientSquared,
            sd2,
            OfSmall
        );
                                                                        
                                                                          
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

__device__ double gasWallExposedAreaFraction
(
    const DeviceState& s,
    const int f
)
{
    if
    (
        f < s.nInternalFaces
     || f >= s.nFaces
     || s.particleStuckModelConfigured == 0
     || !s.particlesMayBePresent
     || s.particleStuckCandidateMask == nullptr
     || s.particleStuckCandidateMask[f] == 0
    )
    {
        return 1.0;
    }
    if
    (
        s.particleWallRepresentedContactArea == nullptr
     || s.particleWallContactAreaScale == nullptr
    )
    {
        asm("trap;");
        return 0.0;
    }

    const double faceArea = s.magSf[f];
    const double representedArea =
        s.particleWallRepresentedContactArea[f];
    const double contactAreaScale =
        s.particleWallContactAreaScale[f];
    if
    (
        !finiteDevice(faceArea)
     || !finiteDevice(representedArea)
     || !finiteDevice(contactAreaScale)
     || !(faceArea > 0.0)
     || representedArea < 0.0
     || !(contactAreaScale > 0.0)
     || contactAreaScale > 1.0
    )
    {
        asm("trap;");
        return 0.0;
    }

    const double occupiedArea = representedArea*contactAreaScale;
    if (!finiteDevice(occupiedArea) || occupiedArea < 0.0)
    {
        asm("trap;");
        return 0.0;
    }
    return clampRange(1.0 - occupiedArea/faceArea, 0.0, 1.0);
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
    const int boundaryKind = nei >= 0 ? 0 : s.riemannBoundaryKind[f];
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
        const double wallThermalAreaFraction =
            nei < 0 && boundaryKind == 2
          ? gasWallExposedAreaFraction(s, f)
          : 1.0;
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
                energyFlux +=
                    wallThermalAreaFraction*directWallHeatFlux;
            }
            else
            {
                energyFlux -=
                    wallThermalAreaFraction
                   *kEffective*normalTemperatureGradient;
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

__global__ void applyGasFluxPositivityScaleKernel
(
    DeviceState* sp,
    const double ledgerDt
)
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

    if
    (
        s.gasWallEnergy != nullptr
     && s.gasWallEnergyMask != nullptr
     && s.gasWallEnergyMask[f] != 0
     && f >= s.nInternalFaces
     && s.gasBoundaryKind[f] == 2
    )
    {
        s.gasWallEnergy[f] += ledgerDt*s.gasPhiRhoE[f];
    }
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
    const int boundaryKind = nei >= 0 ? 0 : s.riemannBoundaryKind[f];
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
        const int boundaryKind = (f < s.nInternalFaces || isPeriodicFace(s, f))
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

    if (s.solveParticleTemperature != 0)
    {
        const double specificEnthalpy =
            finiteOr(s.rhoHp[c]/solidMass, -1.0);
        double Tp = particleTemperatureFromSpecificEnthalpyDevice
        (
            specificEnthalpy
        );
        Tp = finiteOr(Tp, s.TpMin);
        Tp = clampRange(Tp, s.TpMin, s.TpMax);
        s.Tp[c] = Tp;
        s.rhoHp[c] = solidMass*particleSpecificEnthalpyDevice(Tp);
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
    if (eps <= s.epsSMin)
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = 0.0;
        return;
    }

    s.Tp[c] = clampRange(s.Tp[c], s.TpMin, s.TpMax);
    s.rhoHp[c] =
        eps*s.rhoSolid*particleSpecificEnthalpyDevice(s.Tp[c]);
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
    const double eps = clampRange(finiteOr(s.epsS[c], 0.0), 0.0, 1.0);
    s.epsGPrev[c] = 1.0 - eps;
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

__global__ void applyGasGravityKernel(DeviceState* sp, const double dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const double rho = clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const double oldMomX = finiteOr(s.rhoUx[c], 0.0);
    const double oldMomY = finiteOr(s.rhoUy[c], 0.0);
    const double oldMomZ = finiteOr(s.rhoUz[c], 0.0);
    const double work =
        dt*(oldMomX*s.gravityX + oldMomY*s.gravityY + oldMomZ*s.gravityZ);
    const double gravitySquared =
        sqr3(s.gravityX, s.gravityY, s.gravityZ);
    s.rhoUx[c] = oldMomX + rho*s.gravityX*dt;
    s.rhoUy[c] = oldMomY + rho*s.gravityY*dt;
    s.rhoUz[c] = oldMomZ + rho*s.gravityZ*dt;
    s.rhoE[c] = finiteOr(s.rhoE[c], 0.0)
              + work + 0.5*rho*dt*dt*gravitySquared;
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

__host__ __device__ inline bool gasDragModelActive(const int modelId)
{
    return modelId != 0;
}

template<class DragModel>
__device__ double dragInverseTimeDevice
(
    const DeviceState& s,
    const double gasDensity,
    const double gasVolumeFraction,
    const double diameter,
    const double relativeSpeed,
    const DragModel& model
)
{
    const ugkwpGpuDrag::DragInput input
    {
        gasDensity,
        gasVolumeFraction,
        s.gasMu,
        s.rhoSolid,
        diameter,
        relativeSpeed
    };
    return model.inverseRelaxationTime(input);
}

__global__ void applyGasVolumeFractionSourceKernel
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

    const double ugx0 = finiteOr(s.Ux[c], 0.0);
    const double ugy0 = finiteOr(s.Uy[c], 0.0);
    const double ugz0 = finiteOr(s.Uz[c], 0.0);

    const double cepsG =
        -((epsG - epsGOld)/(dt + OfSmall)
          + ugx0*gradEx + ugy0*gradEy + ugz0*gradEz)/epsGsafe;

    const double mgOld =
        clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const double pressureOld =
        clampMin(finiteOr(s.p[c], 0.0), 0.0);
    const double enerGOld = finiteOr(s.rhoE[c], 0.0);
    const double massScale = 1.0 + dt*cepsG;
    const double mgCandidate = mgOld*massScale;
    const double momGXCandidate = s.rhoUx[c]*massScale;
    const double momGYCandidate = s.rhoUy[c]*massScale;
    const double momGZCandidate = s.rhoUz[c]*massScale;
    const double enerGCandidate =
        enerGOld*massScale + dt*cepsG*pressureOld;
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
     || !finiteDevice(pressureOld)
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
}

template<class DragModel>
__global__ void applyEulerianGasSolidDragKernelStatic
(
    DeviceState* sp,
    const double dt,
    const DragModel dragModel
)
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

    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }

    const double eps = solidEpsFromMomentDevice(s, c);
    const double epsG = 1.0 - eps;
    const double epsGsafe = clampMin(epsG, OfSmall);
    const double rhoG = s.couplingRhoOld[c];
    const double mg = epsG*rhoG;
    const double ugx0 = s.couplingUxOld[c];
    const double ugy0 = s.couplingUyOld[c];
    const double ugz0 = s.couplingUzOld[c];

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
    const double beta = dragInverseTimeDevice
    (
        s,
        rhoG,
        epsG,
        dLocal,
        urMag,
        dragModel
    );

    if (!finiteDevice(beta) || beta <= OfSmall)
    {
        return;
    }

    const double tauDragCell = clampMin(1.0/beta, OfSmall);

    const double momGX0 = epsG*s.rhoUx[c];
    const double momGY0 = epsG*s.rhoUy[c];
    const double momGZ0 = epsG*s.rhoUz[c];
    const double enerG0 = epsG*s.rhoE[c];

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

    s.rho[c] = rhoG;
    s.rhoUx[c] = rhoG*ugNewX;
    s.rhoUy[c] = rhoG*ugNewY;
    s.rhoUz[c] = rhoG*ugNewZ;
    s.rhoE[c] = (kgNew + igAfter + diss)/epsGsafe;
}
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

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }
    const double epsG = 1.0 - solidEpsFromMomentDevice(s, c);
    const double epsGsafe = clampMin(epsG, OfSmall);

    const double hp = clampMin(finiteOr(s.momRhoHpP[c], 0.0), 0.0);
    if (hp <= 0.0)
    {
        return;
    }

    const double specificEnthalpy = finiteOr(hp/(rhoP + OfSmall), -1.0);
    const double tpAvg =
        clampRange
        (
            finiteOr
            (
                particleTemperatureFromSpecificEnthalpyDevice
                (
                    specificEnthalpy
                ),
                s.TpMin
            ),
            s.TpMin,
            s.TpMax
        );
    const double particleCp = particleSpecificHeatDevice(tpAvg);
    if (!(particleCp > 0.0))
    {
        return;
    }

    const double rhoG =
        clampMin(finiteOr(s.couplingRhoOld[c], 0.0), s.rhoMin);
    const double tg =
        clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            1.0e30
        );

    const double rhoDP = clampMin(finiteOr(s.momRhoPD[c], 0.0), 0.0);
    const double dLocal =
        clampMin
        (
            finiteOr(rhoDP/rhoP, s.particleDiameterFallback),
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

    const double nu = ugkwp::ranzMarshallNuFromPr
    (
        clampMin(re, 0.0),
        clampMin(pr, 1.0e-12)
    );

    const double rate =
        6.0*nu*gasConductivity
       /(s.rhoSolid*particleCp*dLocal*dLocal + 1.0e-300);

    const double gasCv =
        s.Rgas/clampMin(s.gammaGas - 1.0, OfSmall);
    const double gasCapacity = epsG*rhoG*gasCv;
    const double solidCapacity = rhoP*particleCp;

                                                                           
                                                                             
                                                                        
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
    double rhoEnew = rhoEold - dHp/epsGsafe;

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
    return ugkwp::radialDistributionG0FromRatio(cRatio);
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
        pColl = ugkwp::collisionalPressure
        (
            s.collisionalRestitution,
            s.rhoSolid,
            eps,
            g0,
            theta
        );
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

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        if (i < 0 || i >= s.particleCapacity || s.pStatus[i] == 0)
        {
            continue;
        }
        if (s.pStuck[i] != 0)
        {
            s.pux[i] = 0.0;
            s.puy[i] = 0.0;
            s.puz[i] = 0.0;
            if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
            {
                s.puxOld[i] = 0.0;
                s.puyOld[i] = 0.0;
                s.puzOld[i] = 0.0;
            }
            continue;
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

template<bool DirectBase>
__global__ void applyCollisionalPressureProjectionSplitSegmentKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return;
    }
    const double px0 = finiteOr(s.momRhoUPx[c], 0.0);
    const double py0 = finiteOr(s.momRhoUPy[c], 0.0);
    const double pz0 = finiteOr(s.momRhoUPz[c], 0.0);
    const double e0 = clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0);
    const double dpx = finiteOr(s.pressureDeltaMomX[c], 0.0);
    const double dpy = finiteOr(s.pressureDeltaMomY[c], 0.0);
    const double dpz = finiteOr(s.pressureDeltaMomZ[c], 0.0);
    const double de = finiteOr(s.pressureDeltaEnergy[c], 0.0);
    const double px1 = px0 + dpx;
    const double py1 = py0 + dpy;
    const double pz1 = pz0 + dpz;
    const double e1 = e0 + de;
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

    const int start = DirectBase
      ? s.preBaseCellOffset[c]
      : s.cellParticleOffset[c];
    const int end = DirectBase
      ? s.preBaseCellOffset[c + 1]
      : s.cellParticleOffset[c + 1];
    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = DirectBase ? pos : s.sortedParticleIndex[pos];
        if
        (
            i < 0 || i >= s.particleCapacity
         || s.pStatus[i] == 0 || s.pCellId[i] != c
        )
        {
            continue;
        }
        if (s.pStuck[i] != 0)
        {
            s.pux[i] = 0.0;
            s.puy[i] = 0.0;
            s.puz[i] = 0.0;
            if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
            {
                s.puxOld[i] = 0.0;
                s.puyOld[i] = 0.0;
                s.puzOld[i] = 0.0;
            }
            continue;
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
}

__global__ void finalizeCollisionalPressureProjectionSplitCellsKernel
(
    DeviceState* sp
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
        return;
    }
    const double px1 =
        finiteOr(s.momRhoUPx[c], 0.0) + finiteOr(s.pressureDeltaMomX[c], 0.0);
    const double py1 =
        finiteOr(s.momRhoUPy[c], 0.0) + finiteOr(s.pressureDeltaMomY[c], 0.0);
    const double pz1 =
        finiteOr(s.momRhoUPz[c], 0.0) + finiteOr(s.pressureDeltaMomZ[c], 0.0);
    const double e1 =
        clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0)
      + finiteOr(s.pressureDeltaEnergy[c], 0.0);
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
        if (s.pStuck[i] != 0)
        {
            s.pux[i] = 0.0;
            s.puy[i] = 0.0;
            s.puz[i] = 0.0;
            if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
            {
                s.puxOld[i] = 0.0;
                s.puyOld[i] = 0.0;
                s.puzOld[i] = 0.0;
            }
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
    const int block
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
    if
    (
        s->csrCellLocalPathEnabled != 0
     && s->splitPreDirectoryActive != 0
    )
    {
        PRESSURE_LAUNCH
        (
            (applyCollisionalPressureProjectionSplitSegmentKernel<true>
                <<<s->nCells, block>>>(s->deviceState)),
            "apply split-Dpre base pressure projection launch"
        );
        if (s->nBoundarySources > 0)
        {
            PRESSURE_LAUNCH
            (
                (applyCollisionalPressureProjectionSplitSegmentKernel<false>
                    <<<s->nCells, block>>>(s->deviceState)),
                "apply split-Dpre injection pressure projection launch"
            );
        }
        PRESSURE_LAUNCH
        (
            (finalizeCollisionalPressureProjectionSplitCellsKernel
                <<<cellGrid, block>>>(s->deviceState)),
            "finalize split-Dpre pressure projection cells launch"
        );
    }
    else if (s->csrCellLocalPathEnabled != 0)
    {
        PRESSURE_LAUNCH
        (
            (applyCollisionalPressureProjectionKernel<<<s->nCells, block>>>(s->deviceState, kickDt)),
            "applyCollisionalPressureProjectionKernel launch"
        );
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

__device__ int checkedMobilePackingCount
(
    const DeviceState& s,
    const int* count
)
{
    const int value = *count;
    if (value < 0 || value > s.nCells)
    {
        asm("trap;");
    }
    return value;
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
        if (s.pStatus[i] != 1 || s.pStuck[i] != 0)
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

#include "../../../common/gasNumerics/GpuPackingProjectionAlgebra.cuh"

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

__device__ bool mobilePackingParticleEligible
(
    const DeviceState& s,
    const int i
)
{
    return s.pStatus[i] == 1 && s.pStuck[i] == 0;
}

#include "../../../common/gasNumerics/GpuPackingProjectionCooperative.cuh"

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

    if (s->mobilePackingCooperativeGrid <= 0)
    {
        setLastErrorText("mobile packing cooperative grid is unavailable");
        return 1;
    }
    double kernelDt = dt;
    void* cooperativeArguments[] = {&s->deviceState, &kernelDt};
    err = cudaLaunchCooperativeKernel
    (
        reinterpret_cast<const void*>
        (completeMobilePackingProjectionCooperativeKernel),
        dim3(s->mobilePackingCooperativeGrid),
        dim3(block),
        cooperativeArguments,
        0,
        nullptr
    );
    if (err != cudaSuccess)
    {
        setLastError("complete mobile packing projection launch", err);
        return 1;
    }
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
    const double lmfp = ugkwp::granularMeanFreePath
    (
        OfPi,
        dPart,
        eps,
        g0,
        OfSmall
    );

    return clampMin
    (
        ugkwp::granularCollisionTime(lmfp, theta, OfSmall),
        OfSmall
    );
}

__device__ void clearColdWallParticleState(DeviceState&, int);
__device__ void initialiseColdWallParticleState(DeviceState&, int, double);
__device__ void clearColdWall2DParticleState(DeviceState&, int);
__device__ void initialiseColdWall2DParticleState(DeviceState&, int, double);

__global__ void injectBoundaryParticlesKernel
(
    DeviceState* sp,
    const double dt,
    const double simulationTime
)
{
    DeviceState& s = *sp;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if
    (
        j >= s.nBoundarySources
     || s.particleCapacity <= 0
     || s.injectionParcelMass <= 0.0
    )
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

    const double parcelMass = s.injectionParcelMass;
    double available =
        clampMin(finiteOr(s.sourceResidualMass[j], 0.0), 0.0) + rate*dt;
    if (available < parcelMass)
    {
        s.sourceResidualMass[j] = available;
        return;
    }

    const int nNew = static_cast<int>(floor(available/parcelMass));
    double consumed = 0.0;

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
        s.pm[slot] = parcelMass;
        s.pCellId[slot] = sourceCell;
        s.pStatus[slot] = 1;
        s.pStuck[slot] = 0;
        s.pStuckFaceId[slot] = -1;
        s.pDepositionArea[slot] = 0.0f;
        s.pContactDuration[slot] = 0.0f;
        s.pContactMaximumArea[slot] = 0.0f;
        s.pContactPeakFraction[slot] = 0.0f;
        clearColdWallParticleState(s, slot);
        clearColdWall2DParticleState(s, slot);
        s.pRng[slot] = rng;
        s.pOrigId[slot] =
            (static_cast<unsigned long long>(j) << 40)
          ^ static_cast<unsigned long long>(slot);
        consumed += parcelMass;
    }

    s.sourceResidualMass[j] = available - consumed;
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

__device__ void atomicAddParticleWallEnergyByFace
(
    DeviceState& s,
    double* const wallEnergyLedger,
    const int globalFaceId,
    const double wallEnergyJ
)
{
    if (wallEnergyJ == 0.0)
    {
        return;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned int active = __activemask();
    const int ledgerChannel =
        wallEnergyLedger == s.particleWallDepositedEnergy ? 0 : 1;
    const int faceChannelKey = 2*globalFaceId + ledgerChannel;
    const unsigned int group =
        __match_any_sync(active, faceChannelKey);
    const int leader = __ffs(static_cast<int>(group)) - 1;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    double groupEnergy = 0.0;
    unsigned int remaining = group;
    while (remaining != 0u)
    {
        const int sourceLane = __ffs(static_cast<int>(remaining)) - 1;
        groupEnergy += __shfl_sync(group, wallEnergyJ, sourceLane);
        remaining &= remaining - 1u;
    }
    if (lane == leader)
    {
        atomicAdd(&wallEnergyLedger[globalFaceId], groupEnergy);
    }
#else
    atomicAdd(&wallEnergyLedger[globalFaceId], wallEnergyJ);
#endif
}

__device__ void clearColdWallParticleState
(
    DeviceState& s,
    const int particleI
)
{
    if (s.coldWallSolidificationEnabled == 0)
    {
        return;
    }
    const int nodeBase = particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase = particleI*Foam::gpuThermal::coldWallRadialRingCount;
    for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
    {
        s.pColdNodeSpecificEnthalpy[nodeBase + node] = 0.0f;
    }
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0f;
    }
    s.pColdFrozenArea[particleI] = 0.0f;
    s.pColdContactAge[particleI] = 0.0f;
}

__device__ void initialiseColdWallParticleState
(
    DeviceState& s,
    const int particleI,
    const double temperatureK
)
{
    if
    (
        s.coldWallSolidificationEnabled == 0
     || s.pColdNodeSpecificEnthalpy == nullptr
     || s.pColdRingSolidMass == nullptr
     || s.pColdFrozenArea == nullptr
     || s.pColdContactAge == nullptr
    )
    {
        asm("trap;");
    }
    double nodeEnthalpy[Foam::gpuThermal::coldWallAxialNodeCount];
    double ringSolidMass[Foam::gpuThermal::coldWallRadialRingCount];
    Foam::gpuThermal::initialiseColdWallProfile
    (
        nodeEnthalpy,
        ringSolidMass,
        temperatureK,
        s.coldWallSolidificationParameters
    );
    const int nodeBase = particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase = particleI*Foam::gpuThermal::coldWallRadialRingCount;
    for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
    {
        s.pColdNodeSpecificEnthalpy[nodeBase + node] =
            static_cast<float>(nodeEnthalpy[node]);
    }
    for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
    {
        s.pColdRingSolidMass[ringBase + ring] = 0.0f;
    }
    s.pColdFrozenArea[particleI] = 0.0f;
    s.pColdContactAge[particleI] = 0.0f;
}

#include "../../../common/wall/GpuColdWall2DDevice.cuh"
#include "../../../common/wall/GpuColdWall1DDevice.cuh"

__global__ void accumulateParticleWallRepresentedContactAreaKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*blockDim.x + threadIdx.x;
        entry < nWallBound;
        entry += blockDim.x*gridDim.x
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        if (i < 0 || i >= s.particleCapacity)
        {
            asm("trap;");
        }
        if
        (
            s.pStatus[i] == 0
         || s.pStuck[i] == Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const unsigned char wallState = s.pStuck[i];
        const int faceI = s.pStuckFaceId[i];
        if
        (
            faceI < 0
         || faceI >= s.nFaces
         || s.particleStuckCandidateMask[faceI] == 0
        )
        {
            asm("trap;");
        }

        double physicalContactArea = 0.0;
        if (wallState == Foam::gpuThermal::particleWallDeposited)
        {
            physicalContactArea =
                static_cast<double>(s.pDepositionArea[i]);
            if (!(physicalContactArea > 0.0))
            {
                asm("trap;");
            }
        }
        else if
        (
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit
        )
        {
            const double duration =
                static_cast<double>(s.pContactDuration[i]);
            const double maximumArea =
                static_cast<double>(s.pContactMaximumArea[i]);
            const double peakTimeFraction =
                static_cast<double>(s.pContactPeakFraction[i]);
            const double damageArea =
                static_cast<double>(s.pDepositionArea[i]);
            const double age =
                clampRange(finiteOr(s.pTheta[i], 0.0), 0.0, duration);
            if
            (
                !(duration > 0.0) || !(maximumArea > 0.0)
             || !(peakTimeFraction > 0.0) || !(peakTimeFraction < 1.0)
             || damageArea < 0.0
            )
            {
                asm("trap;");
            }
            const double kinematicArea =
                maximumArea*Foam::gpuThermal::normalizedKinematicArea
                (
                    age/duration, peakTimeFraction
                );
            const unsigned char interactionType =
                s.particleStuckCandidateMask[faceI];
            const double frozenArea =
                interactionType
             == Foam::gpuThermal::particleWallSolidifyingDeposition
              ? static_cast<double>(s.pColdFrozenArea[i])
              : interactionType == Foam::gpuThermal::particleWallColdWall2D
              ? static_cast<double>(s.pCold2DFrozenArea[i])
              : 0.0;
            physicalContactArea = clampMin
            (
                fmax(kinematicArea, frozenArea) - damageArea,
                0.0
            );
                                                                           
                                                                           
            if (!(physicalContactArea > 0.0))
            {
                continue;
            }
        }
        else
        {
            asm("trap;");
        }

        const double representedArea =
            Foam::gpuThermal::representedDepositionContactArea
            (
                s.rhoSolid,
                s.pd[i],
                physicalContactArea,
                s.pm[i]
            );
        if (!(representedArea > 0.0))
        {
            asm("trap;");
        }
        atomicAdd
        (
            &s.particleWallRepresentedContactArea[faceI],
            representedArea
        );
    }
}

__device__ void recordWallContactAreaDiagnosticError
(
    WallContactAreaDiagnosticError* error,
    const int code,
    DeviceState& s,
    const int directoryEntry,
    const int directoryCount,
    const int particleI,
    const int faceI,
    const int candidateType,
    const double depositionArea,
    const double duration,
    const double maximumArea,
    const double peakTimeFraction,
    const double age,
    const double frozenArea,
    const double kinematicArea,
    const double physicalContactArea,
    const double representedArea
)
{
    if (atomicCAS(&error->code, 0, code) != 0)
    {
        return;
    }
    error->directoryEntry = directoryEntry;
    error->directoryCount = directoryCount;
    error->particleArrayIndex = particleI;
    error->faceId = faceI;
    error->faceCount = s.nFaces;
    error->candidateType = candidateType;
    error->depositionArea = depositionArea;
    error->contactDuration = duration;
    error->contactMaximumArea = maximumArea;
    error->contactPeakFraction = peakTimeFraction;
    error->contactAge = age;
    error->frozenArea = frozenArea;
    error->kinematicArea = kinematicArea;
    error->physicalContactArea = physicalContactArea;
    error->representedArea = representedArea;
    error->solidDensity = s.rhoSolid;
    if (particleI >= 0 && particleI < s.particleCapacity)
    {
        error->particleStatus = s.pStatus[particleI];
        error->wallState = static_cast<int>(s.pStuck[particleI]);
        error->particleOriginalId = s.pOrigId[particleI];
        error->diameter = s.pd[particleI];
        error->parcelMass = s.pm[particleI];
        error->temperature = s.pT[particleI];
    }
}

__global__ void diagnoseParticleWallRepresentedContactAreaKernel
(
    DeviceState* sp,
    WallContactAreaDiagnosticError* error
)
{
    DeviceState& s = *sp;
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*blockDim.x + threadIdx.x;
        entry < nWallBound;
        entry += blockDim.x*gridDim.x
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        if (i < 0 || i >= s.particleCapacity)
        {
            recordWallContactAreaDiagnosticError
            (
                error, 1, s, entry, nWallBound, i, -1, -1,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            );
            continue;
        }
        if
        (
            s.pStatus[i] == 0
         || s.pStuck[i] == Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const unsigned char wallState = s.pStuck[i];
        const int faceI = s.pStuckFaceId[i];
        const int candidateType =
            faceI >= 0 && faceI < s.nFaces
          ? static_cast<int>(s.particleStuckCandidateMask[faceI])
          : -1;
        if
        (
            faceI < 0
         || faceI >= s.nFaces
         || candidateType == 0
        )
        {
            recordWallContactAreaDiagnosticError
            (
                error, 2, s, entry, nWallBound, i, faceI, candidateType,
                static_cast<double>(s.pDepositionArea[i]),
                static_cast<double>(s.pContactDuration[i]),
                static_cast<double>(s.pContactMaximumArea[i]),
                static_cast<double>(s.pContactPeakFraction[i]),
                s.pTheta[i], 0.0, 0.0, 0.0, 0.0
            );
            continue;
        }

        double physicalContactArea = 0.0;
        double duration = 0.0;
        double maximumArea = 0.0;
        double peakTimeFraction = 0.0;
        double age = 0.0;
        double frozenArea = 0.0;
        double kinematicArea = 0.0;
        const double depositionArea =
            static_cast<double>(s.pDepositionArea[i]);
        if (wallState == Foam::gpuThermal::particleWallDeposited)
        {
            physicalContactArea = depositionArea;
            if (!(physicalContactArea > 0.0))
            {
                recordWallContactAreaDiagnosticError
                (
                    error, 3, s, entry, nWallBound, i, faceI, candidateType,
                    depositionArea, 0.0, 0.0, 0.0, s.pTheta[i],
                    0.0, 0.0, physicalContactArea, 0.0
                );
                continue;
            }
        }
        else if
        (
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit
        )
        {
            duration = static_cast<double>(s.pContactDuration[i]);
            maximumArea = static_cast<double>(s.pContactMaximumArea[i]);
            peakTimeFraction = static_cast<double>(s.pContactPeakFraction[i]);
            age = clampRange(finiteOr(s.pTheta[i], 0.0), 0.0, duration);
            if
            (
                !(duration > 0.0) || !(maximumArea > 0.0)
             || !(peakTimeFraction > 0.0) || !(peakTimeFraction < 1.0)
             || depositionArea < 0.0
            )
            {
                recordWallContactAreaDiagnosticError
                (
                    error, 4, s, entry, nWallBound, i, faceI, candidateType,
                    depositionArea, duration, maximumArea, peakTimeFraction,
                    age, 0.0, 0.0, 0.0, 0.0
                );
                continue;
            }
            kinematicArea =
                maximumArea*Foam::gpuThermal::normalizedKinematicArea
                (
                    age/duration, peakTimeFraction
                );
            frozenArea =
                candidateType
             == Foam::gpuThermal::particleWallSolidifyingDeposition
              ? static_cast<double>(s.pColdFrozenArea[i])
              : candidateType == Foam::gpuThermal::particleWallColdWall2D
              ? static_cast<double>(s.pCold2DFrozenArea[i])
              : 0.0;
            physicalContactArea = clampMin
            (
                fmax(kinematicArea, frozenArea) - depositionArea,
                0.0
            );
            if (!(physicalContactArea > 0.0))
            {
                continue;
            }
        }
        else
        {
            recordWallContactAreaDiagnosticError
            (
                error, 5, s, entry, nWallBound, i, faceI, candidateType,
                depositionArea,
                static_cast<double>(s.pContactDuration[i]),
                static_cast<double>(s.pContactMaximumArea[i]),
                static_cast<double>(s.pContactPeakFraction[i]),
                s.pTheta[i], 0.0, 0.0, 0.0, 0.0
            );
            continue;
        }

        const double representedArea =
            Foam::gpuThermal::representedDepositionContactArea
            (
                s.rhoSolid,
                s.pd[i],
                physicalContactArea,
                s.pm[i]
            );
        if (!(representedArea > 0.0))
        {
            recordWallContactAreaDiagnosticError
            (
                error, 6, s, entry, nWallBound, i, faceI, candidateType,
                depositionArea, duration, maximumArea, peakTimeFraction,
                age, frozenArea, kinematicArea, physicalContactArea,
                representedArea
            );
            continue;
        }
        atomicAdd
        (
            &s.particleWallRepresentedContactArea[faceI],
            representedArea
        );
    }
}

__global__ void rebuildWallBoundParticleDirectoryKernel(DeviceState* sp)
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
        if
        (
            s.pStatus[i] == 0
         || s.pStuck[i] == Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        Foam::gpuWall::publishWallBoundParticleIndex(s, i);
    }
}

__global__ void finalizeParticleWallContactAreaScaleKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int faceI = blockIdx.x*blockDim.x + threadIdx.x;
    if (faceI >= s.nFaces)
    {
        return;
    }

    double scale = 1.0;
    if (s.particleStuckCandidateMask[faceI] != 0)
    {
        const double representedArea =
            s.particleWallRepresentedContactArea[faceI];
        const double maximumArea =
            s.particleWallMaximumCoverage*s.magSf[faceI];
        if
        (
            !finiteDevice(representedArea)
         || !finiteDevice(maximumArea)
         || representedArea < 0.0
         || !(maximumArea > 0.0)
        )
        {
            asm("trap;");
        }
        if (representedArea > maximumArea)
        {
            scale = maximumArea/representedArea;
        }
    }
    if (!(scale > 0.0) || scale > 1.0)
    {
        asm("trap;");
    }
    s.particleWallContactAreaScale[faceI] = scale;
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

    if (s.pStuck[i] != 0)
    {
        s.pux[i] = 0.0;
        s.puy[i] = 0.0;
        s.puz[i] = 0.0;
        if (s.pStuck[i] == Foam::gpuThermal::particleWallDeposited)
        {
            s.puxOld[i] = 0.0;
            s.puyOld[i] = 0.0;
            s.puzOld[i] = 0.0;
        }
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
        else if (kind == 6 && next >= 0 && next < s.nCells)
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
        else if (kind == 5)
        {
            const double nx = s.planeNx[plane];
            const double ny = s.planeNy[plane];
            const double nz = s.planeNz[plane];
            const int globalFaceId = s.cellFaceId[plane];
            if
            (
                s.particleStuckModelConfigured == 0
             || globalFaceId < 0
             || globalFaceId >= s.nFaces
             || s.particleStuckCandidateMask[globalFaceId] == 0
            )
            {
                asm("trap;");
            }
            const unsigned char wallInteractionType =
                s.particleStuckCandidateMask[globalFaceId];
            const double un =
                s.pux[i]*nx + s.puy[i]*ny + s.puz[i]*nz;
            const double normalSpeed = fabs
            (
                (s.pux[i] - s.gasBoundaryUx[globalFaceId])*nx
              + (s.puy[i] - s.gasBoundaryUy[globalFaceId])*ny
              + (s.puz[i] - s.gasBoundaryUz[globalFaceId])*nz
            );
            const Foam::gpuThermal::SommerfeldImpact impact =
                Foam::gpuThermal::evaluateSommerfeldImpact
                (
                    s.pT[i],
                    normalSpeed,
                    s.pd[i],
                    s.sommerfeldThreshold
                );
            if (!impact.valid)
            {
                asm("trap;");
            }
            const Foam::gpuThermal::FiniteWallContactImpact finiteImpact =
                Foam::gpuThermal::evaluateFiniteWallContactImpact
                (
                    s.pT[i],
                    s.pd[i],
                    normalSpeed,
                    s.particleWallContactAngleCosine
                );
            if
            (
                !finiteImpact.valid
             || finiteImpact.contactDurationS > static_cast<double>(FLT_MAX)
             || finiteImpact.maximumAreaM2 > static_cast<double>(FLT_MAX)
             || finiteImpact.peakTimeFraction > static_cast<double>(FLT_MAX)
            )
            {
                asm("trap;");
            }
            const double eps = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
            x = xHit - eps*nx;
            y = yHit - eps*ny;
            z = zHit - eps*nz;
            const double restitution = s.cellFaceRestitution[plane];
            s.puxOld[i] = s.pux[i] - (1.0 + restitution)*un*nx;
            s.puyOld[i] = s.puy[i] - (1.0 + restitution)*un*ny;
            s.puzOld[i] = s.puz[i] - (1.0 + restitution)*un*nz;
            s.pux[i] = 0.0;
            s.puy[i] = 0.0;
            s.puz[i] = 0.0;
            s.pStuck[i] =
                wallInteractionType
             == Foam::gpuThermal::particleWallReboundContact
              ? Foam::gpuThermal::particleWallTransientRebound
              :
                (
                    impact.deposit
                  ? Foam::gpuThermal::particleWallTransientDeposit
                  : Foam::gpuThermal::particleWallTransientRebound
                );
            s.pStuckFaceId[i] = globalFaceId;
            s.pTheta[i] = 0.0;
            s.pDepositionArea[i] = 0.0f;
            s.pContactDuration[i] =
                static_cast<float>(finiteImpact.contactDurationS);
            s.pContactMaximumArea[i] =
                static_cast<float>(finiteImpact.maximumAreaM2);
            s.pContactPeakFraction[i] =
                static_cast<float>(finiteImpact.peakTimeFraction);
            if
            (
                wallInteractionType
             == Foam::gpuThermal::particleWallSolidifyingDeposition
            )
            {
                initialiseColdWallParticleState(s, i, s.pT[i]);
                clearColdWall2DParticleState(s, i);
            }
            else if
            (
                wallInteractionType
             == Foam::gpuThermal::particleWallColdWall2D
            )
            {
                clearColdWallParticleState(s, i);
                initialiseColdWall2DParticleState(s, i, s.pT[i]);
            }
            else
            {
                clearColdWallParticleState(s, i);
                clearColdWall2DParticleState(s, i);
            }
            remainingDt = 0.0;
            break;
        }
        else
        {
            const double exitMass =
                clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
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
__device__ void relaxOneParticleToResidentGas
(
    DeviceState& s,
    const int i,
    const double dt,
    const DragModel dragModel
)
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
    const unsigned char wallStateAtStepStart = s.pStuck[i];
    const bool stuck = wallStateAtStepStart != Foam::gpuThermal::particleWallMobile;
    const bool finiteContact =
        wallStateAtStepStart == Foam::gpuThermal::particleWallTransientRebound
     || wallStateAtStepStart == Foam::gpuThermal::particleWallTransientDeposit;
    const int wallFaceAtStepStart = stuck ? s.pStuckFaceId[i] : -1;
    const unsigned char wallInteractionType =
        wallFaceAtStepStart >= 0 && wallFaceAtStepStart < s.nFaces
      ? s.particleStuckCandidateMask[wallFaceAtStepStart]
      : Foam::gpuThermal::particleWallInteractionNone;
    if
    (
        stuck
     && wallInteractionType == Foam::gpuThermal::particleWallColdWall2D
    )
    {
        return;
    }
    const bool coldWallContact =
        wallInteractionType
     == Foam::gpuThermal::particleWallSolidifyingDeposition;
    const double ux = stuck ? 0.0 : s.pux[i];
    const double uy = stuck ? 0.0 : s.puy[i];
    const double uz = stuck ? 0.0 : s.puz[i];
    if (!finiteContact)
    {
        s.puxOld[i] = ux;
        s.puyOld[i] = uy;
        s.puzOld[i] = uz;
    }

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
    const double re = rhoG*dPart*relMag/mu;
    if (gasDragModelActive(s.dragModelId))
    {
    const double invTauDrag = dragInverseTimeDevice
    (
        s,
        rhoG,
        clampRange(1.0 - solidEpsFromMomentDevice(s, c), 0.0, 1.0),
        dPart,
        relMag,
        dragModel
    );

    if (!stuck)
    {
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
    }
    else
    {
        s.pux[i] = 0.0;
        s.puy[i] = 0.0;
        s.puz[i] = 0.0;
    }

                                                                         
                                                                             
                                                                      
                                                                  
    if (!finiteContact)
    {
        const double alphaTheta =
            clampRange(finiteOr(s.thetaDragAlpha[c], 1.0), 0.0, 1.0);
        const double unresolvedTheta =
            clampMin(finiteOr(s.pTheta[i], 0.0), 0.0);
        s.pTheta[i] = unresolvedTheta*alphaTheta;
    }
    }
    else if (stuck)
    {
        s.pux[i] = 0.0;
        s.puy[i] = 0.0;
        s.puz[i] = 0.0;
    }

    if
    (
        s.solveParticleTemperature != 0
     && s.particleGasHeatTransferModelId != 0
     && !coldWallContact
    )
    {
        const double gasConductivity = molecularGasConductivity(s);
        const double nu = ugkwp::ranzMarshallNuFromPrOneThird
        (
            clampMin(re, 0.0),
            s.gasPrOneThird
        );
        const double tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const double particleCp = particleSpecificHeatDevice(tpOld);
        const double rate =
            6.0*nu*gasConductivity
           /(s.rhoSolid*particleCp*dPart*dPart + 1.0e-300);
        const double decay = exp(-clampMin(rate, 0.0)*dt);
        const double gasTemperatureK = clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            1.0e30
        );
        const double tpNew = clampRange
        (
            gasTemperatureK + (tpOld - gasTemperatureK)*decay,
            s.TpMin,
            s.TpMax
        );
        s.pT[i] = tpNew;
    }

    if (finiteContact)
    {
        const int faceI = s.pStuckFaceId[i];
        const double duration = static_cast<double>(s.pContactDuration[i]);
        const double maximumArea = static_cast<double>(s.pContactMaximumArea[i]);
        const double peakTimeFraction =
            static_cast<double>(s.pContactPeakFraction[i]);
        const double damageArea = static_cast<double>(s.pDepositionArea[i]);
        const double age0 = clampRange(finiteOr(s.pTheta[i], 0.0), 0.0, duration);
        if
        (
            faceI < 0 || faceI >= s.nFaces
         || s.particleStuckCandidateMask[faceI] == 0
         || !(duration > 0.0) || !(maximumArea > 0.0)
         || !(peakTimeFraction > 0.0) || !(peakTimeFraction < 1.0)
         || damageArea < 0.0
        )
        {
            asm("trap;");
        }
        const double activeDt = clampRange(dt, 0.0, duration - age0);
        const double age1 = age0 + activeDt;
        if
        (
            activeDt > 0.0
         && s.particleWallHeatTransferEnabled != 0
         && !coldWallContact
        )
        {
            const double thetaMid = (age0 + 0.5*activeDt)/duration;
            const double contactAreaMid = fmax
            (
                maximumArea
               *Foam::gpuThermal::normalizedKinematicArea
                (
                    thetaMid, peakTimeFraction
                )
              - damageArea,
                0.0
            );
            const double wallEffusivity =
                s.particleWallEffusivityByFace != nullptr
              ? s.particleWallEffusivityByFace[faceI]
              : sqrt
                (
                    s.particleWallDensityKgM3
                   *s.particleWallSpecificHeatJkgK
                   *s.particleWallConductivityWmK
                );
            const Foam::gpuThermal::AluminaLiquidProperties material =
                Foam::gpuThermal::liquidAluminaProperties(s.pT[i]);
            const double physicalMass =
                (Foam::gpuThermal::finiteContactPi/6.0)
               *material.densityKgM3*s.pd[i]*s.pd[i]*s.pd[i];
            const double conductanceTimeIntegral =
                Foam::gpuThermal::finiteContactWallConductanceTimeIntegral
                (
                    maximumArea,
                    contactAreaMid,
                    duration,
                    peakTimeFraction,
                    age0,
                    activeDt,
                    s.particleWallReflectionHeatTransferEfficiency
                   *s.particleWallContactAreaScale[faceI],
                    s.coldWallSolidificationParameters
                     .interfaceResistanceM2KW,
                    0.0,
                    wallEffusivity,
                    s.coldWallSolidificationParameters
                     .wallTransientResistance != 0
                );
            const Foam::gpuThermal::ParticleWallContactResult result =
                Foam::gpuThermal::lumpedParticleWallInterfaceContact
                (
                    s.pT[i],
                    s.gasBoundaryT[faceI],
                    physicalMass,
                    s.pm[i],
                    conductanceTimeIntegral
                );
            if
            (
                contactAreaMid < 0.0 || !(wallEffusivity > 0.0)
             || !material.valid || !(physicalMass > 0.0)
             || !(conductanceTimeIntegral >= 0.0) || !result.valid
            )
            {
                asm("trap;");
            }
            s.pT[i] = result.particleTemperatureK;
            atomicAddParticleWallEnergyByFace
            (
                s,
                s.particleWallReflectedEnergy,
                faceI,
                result.wallEnergyJ
            );
        }
        s.pTheta[i] = age1;

        bool detach = false;
        bool enterLongDeposit = false;
        double longDepositArea = 0.0;
        const double theta1 = age1/duration;
        const double kinematicArea =
            maximumArea
           *Foam::gpuThermal::normalizedKinematicArea
            (
                theta1, peakTimeFraction
            );
        const double frozenArea = coldWallContact
          ? static_cast<double>(s.pColdFrozenArea[i])
          : 0.0;
        const double effectiveContactArea =
            fmax(kinematicArea, frozenArea) - damageArea;
        if
        (
            wallStateAtStepStart
         == Foam::gpuThermal::particleWallTransientRebound
        )
        {
            detach = !(effectiveContactArea > 0.0);
            enterLongDeposit =
                coldWallContact && !detach && !(kinematicArea > 0.0);
            longDepositArea = effectiveContactArea;
        }
        else
        {
            const Foam::gpuThermal::CapillaryDetachmentState capillary =
                Foam::gpuThermal::evaluateCapillaryDetachmentState
                (
                    s.pT[i], s.pd[i], s.particleWallAdhesionEnergyScale,
                    s.particleWallContactAngleCosine
                );
            const double equilibriumArea =
                capillary.equilibriumContactAreaM2;
            const double targetContactArea = fmin(equilibriumArea, maximumArea);
            if (!coldWallContact)
            {
                asm("trap;");
            }
            longDepositArea =
                fmax(targetContactArea, frozenArea) - damageArea;
            detach = !(effectiveContactArea > 0.0);
            enterLongDeposit =
                !detach
             && theta1 >= peakTimeFraction
             &&
                (
                    kinematicArea <= targetContactArea
                 || !(kinematicArea > 0.0)
                );
            if (!capillary.valid)
            {
                asm("trap;");
            }
        }

        if (detach)
        {
            s.pux[i] = s.puxOld[i];
            s.puy[i] = s.puyOld[i];
            s.puz[i] = s.puzOld[i];
            s.pStuck[i] = Foam::gpuThermal::particleWallMobile;
            s.pStuckFaceId[i] = -1;
            s.pTheta[i] = 0.0;
            s.pDepositionArea[i] = 0.0f;
            s.pContactDuration[i] = 0.0f;
            s.pContactMaximumArea[i] = 0.0f;
            s.pContactPeakFraction[i] = 0.0f;
            clearColdWallParticleState(s, i);
            clearColdWall2DParticleState(s, i);
        }
        else if (enterLongDeposit)
        {
            s.pStuck[i] = Foam::gpuThermal::particleWallDeposited;
            s.pTheta[i] = 0.0;
            s.pDepositionArea[i] = static_cast<float>(longDepositArea);
            s.puxOld[i] = 0.0;
            s.puyOld[i] = 0.0;
            s.puzOld[i] = 0.0;
        }
    }
    else if
    (
        wallStateAtStepStart == Foam::gpuThermal::particleWallDeposited
     && s.particleWallHeatTransferEnabled != 0
    )
    {
        const int stuckFaceId = s.pStuckFaceId[i];
        const double storedDepositionArea =
            static_cast<double>(s.pDepositionArea[i]);
        if
        (
            stuckFaceId < 0
         || stuckFaceId >= s.nFaces
         || s.particleStuckCandidateMask[stuckFaceId] == 0
         || !(storedDepositionArea > 0.0)
        )
        {
            asm("trap;");
        }
        const double contactAreaScale =
            s.particleWallContactAreaScale[stuckFaceId];
        if (!(contactAreaScale > 0.0) || contactAreaScale > 1.0)
        {
            asm("trap;");
        }
        if (!coldWallContact)
        {
            asm("trap;");
        }
    }
}

template<class DragModel>
__global__ void relaxMobileParticlesToResidentGasKernelStatic
(
    DeviceState* sp,
    const double dt,
    const DragModel dragModel
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
        if
        (
            s.pStatus[i] != 0
         && s.pStuck[i] == Foam::gpuThermal::particleWallMobile
        )
        {
            relaxOneParticleToResidentGas(s, i, dt, dragModel);
        }
    }
}

template<class DragModel>
__global__ void relaxWallBoundParticlesToResidentGasKernelStatic
(
    DeviceState* sp,
    const double dt,
    const DragModel dragModel
)
{
    DeviceState& s = *sp;
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*blockDim.x + threadIdx.x;
        entry < nWallBound;
        entry += blockDim.x*gridDim.x
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        if (i < 0 || i >= s.particleCapacity)
        {
            asm("trap;");
        }
        if
        (
            s.pStatus[i] != 0
         && s.pStuck[i] != Foam::gpuThermal::particleWallMobile
        )
        {
            relaxOneParticleToResidentGas(s, i, dt, dragModel);
        }
    }
}

int prepareParticleWallContactAreaScale(DeviceState* s, const int block)
{
    if (s->particleStuckModelConfigured == 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemset
    (
        s->particleWallRepresentedContactArea,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset particle-wall represented contact area", err);
        return 1;
    }

    if (s->wallContactAreaDiagnosticsEnabled != 0)
    {
        if
        (
            s->wallContactAreaDiagnosticError == nullptr
         && allocate
            (
                s->wallContactAreaDiagnosticError,
                1,
                "cudaMalloc wall-contact area diagnostic error"
            ) != 0
        )
        {
            return 1;
        }
        err = cudaMemset
        (
            s->wallContactAreaDiagnosticError,
            0,
            sizeof(WallContactAreaDiagnosticError)
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMemset wall-contact area diagnostic error", err);
            return 1;
        }
        diagnoseParticleWallRepresentedContactAreaKernel
            <<<s->particleWorkGrid, block>>>
            (
                s->deviceState,
                s->wallContactAreaDiagnosticError
            );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "diagnoseParticleWallRepresentedContactAreaKernel launch",
                err
            );
            return 1;
        }
        WallContactAreaDiagnosticError hostError;
        if
        (
            copyToHost
            (
                &hostError,
                s->wallContactAreaDiagnosticError,
                1,
                "cudaMemcpy wall-contact area diagnostic error"
            ) != 0
        )
        {
            return 1;
        }
        if (hostError.code != 0)
        {
            std::snprintf
            (
                lastError,
                sizeof(lastError),
                "wall-contact area diagnostic failed: code=%d, entry=%d/%d, "
                "particle=%d, originalId=%llu, status=%d, wallState=%d, "
                "face=%d/%d, candidateType=%d, depositionArea=%.17g, "
                "duration=%.17g, maximumArea=%.17g, peakFraction=%.17g, "
                "age=%.17g, frozenArea=%.17g, kinematicArea=%.17g, "
                "physicalArea=%.17g, representedArea=%.17g, diameter=%.17g, "
                "parcelMass=%.17g, solidDensity=%.17g, temperature=%.17g",
                hostError.code,
                hostError.directoryEntry,
                hostError.directoryCount,
                hostError.particleArrayIndex,
                static_cast<unsigned long long>(hostError.particleOriginalId),
                hostError.particleStatus,
                hostError.wallState,
                hostError.faceId,
                hostError.faceCount,
                hostError.candidateType,
                hostError.depositionArea,
                hostError.contactDuration,
                hostError.contactMaximumArea,
                hostError.contactPeakFraction,
                hostError.contactAge,
                hostError.frozenArea,
                hostError.kinematicArea,
                hostError.physicalContactArea,
                hostError.representedArea,
                hostError.diameter,
                hostError.parcelMass,
                hostError.solidDensity,
                hostError.temperature
            );
            return 1;
        }
    }
    else
    {
        accumulateParticleWallRepresentedContactAreaKernel
            <<<s->particleWorkGrid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "accumulateParticleWallRepresentedContactAreaKernel launch",
                err
            );
            return 1;
        }
    }

    const int faceGrid = (s->nFaces + block - 1)/block;
    finalizeParticleWallContactAreaScaleKernel
        <<<faceGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "finalizeParticleWallContactAreaScaleKernel launch",
            err
        );
        return 1;
    }
    return 0;
}

__global__ void clearPoissonThermalPoolKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
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

__global__ void clearMobileParticleRadiationSumsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.radiationMobileMass[c] = 0.0;
    s.radiationMobileTemperatureMass[c] = 0.0;
    s.radiationMobileDiameterMass[c] = 0.0;
}

__global__ void accumulateMobileParticleRadiationSumsAtomicKernel
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
        if
        (
            s.pStatus[i] != 1
         || s.pStuck[i] != Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const double mass = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double temperature =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        const double diameter =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        atomicAdd(&s.radiationMobileMass[c], mass);
        atomicAdd(&s.radiationMobileTemperatureMass[c], mass*temperature);
        atomicAdd(&s.radiationMobileDiameterMass[c], mass*diameter);
    }
}

__global__ void accumulatePackedMobileParticleRadiationSumsKernel
(
    DeviceState* sp
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    double mass = 0.0;
    double temperatureMass = 0.0;
    double diameterMass = 0.0;
    const int begin = s.preBaseCellOffset[c];
    const int end = s.preBaseCellOffset[c + 1];
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
         || s.pCellId[i] != c
         || s.pStuck[i] != Foam::gpuThermal::particleWallMobile
        )
        {
            continue;
        }
        const double particleMass = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double temperature =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        const double diameter =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        mass += particleMass;
        temperatureMass += particleMass*temperature;
        diameterMass += particleMass*diameter;
    }
    double sums[3] = {mass, temperatureMass, diameterMass};
    extern __shared__ double warpPartials[];
    blockReduceComponentSums<3>(sums, warpPartials);
    if (threadIdx.x == 0)
    {
        s.radiationMobileMass[c] = sums[0];
        s.radiationMobileTemperatureMass[c] = sums[1];
        s.radiationMobileDiameterMass[c] = sums[2];
    }
}

__device__ void publishParticleEnthalpyMoment
(
    DeviceState& s,
    const int c,
    const double heatDensity
)
{
    const double rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if
    (
        s.solveParticleTemperature == 0
     || rhoP <= s.epsSMin*s.rhoSolid
    )
    {
        s.momRhoHpP[c] = 0.0;
        s.rhoHp[c] = 0.0;
        s.Tp[c] = s.TpMin;
        return;
    }
    const double totalHeat = clampMin(finiteOr(heatDensity, 0.0), 0.0);
    s.momRhoHpP[c] = totalHeat;
    s.rhoHp[c] = totalHeat;
    const double specificEnthalpy =
        finiteOr(totalHeat/(rhoP + OfSmall), -1.0);
    s.Tp[c] = clampRange
    (
        finiteOr
        (
            particleTemperatureFromSpecificEnthalpyDevice(specificEnthalpy),
            s.TpMin
        ),
        s.TpMin,
        s.TpMax
    );
}

__global__ void clearParticleEnthalpyMomentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < s.nCells)
    {
        s.momRhoHpP[c] = 0.0;
    }
}

__global__ void accumulateParticleEnthalpyMomentAtomicKernel(DeviceState* sp)
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
        const double mass = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double temperature =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        atomicAdd
        (
            &s.momRhoHpP[c],
            mass*particleSpecificEnthalpyDevice(temperature)
        );
    }
}

__global__ void recoverParticleEnthalpyMomentAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const double heatDensity =
        s.momRhoHpP[c]/clampMin(s.V[c], s.rhoMin);
    publishParticleEnthalpyMoment(s, c, heatDensity);
}

__global__ void refreshPackedParticleEnthalpyKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    double heat = 0.0;
    const int begin = s.preBaseCellOffset[c];
    const int end = s.preBaseCellOffset[c + 1];
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
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
        const double mass = clampMin(finiteOr(s.pm[i], 0.0), 0.0);
        const double temperature =
            clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        heat += mass*particleSpecificEnthalpyDevice(temperature);
    }
    double sums[1] = {heat};
    extern __shared__ double warpPartials[];
    blockReduceComponentSums<1>(sums, warpPartials);
    if (threadIdx.x == 0)
    {
        const double heatDensity =
            sums[0]/clampMin(s.V[c], s.rhoMin);
        publishParticleEnthalpyMoment(s, c, heatDensity);
    }
}

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
    if (s.csrHeavyReductionEnabled != 0)
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

        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] == 0
        )
        {
            continue;
        }

        if (s.pCellId[i] != c)
        {
            continue;
        }

        unsigned long long rng = s.pRng[i];

        if (uniform01Device(rng) >= prob)
        {
            s.pRng[i] = rng;
            continue;
        }

        const double m =
            clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
        const double ux = finiteOr(s.pux[i], 0.0);
        const double uy = finiteOr(s.puy[i], 0.0);
        const double uz = finiteOr(s.puz[i], 0.0);
        const double theta = particleMomentThetaDevice(s, i);
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

template<bool DirectBase, bool AddToCell>
__global__ void accumulatePoissonPoolSplitSegmentKernel
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

    const int baseBegin = s.preBaseCellOffset[c];
    const int baseEnd = s.preBaseCellOffset[c + 1];
    const int injectionBegin = s.cellParticleOffset[c];
    const int injectionEnd = s.cellParticleOffset[c + 1];
    if (s.csrHeavyReductionEnabled != 0)
    {
        return;
    }

    __shared__ double cellCollisionProbability;
    if (threadIdx.x == 0)
    {
        const double tauColl = granularCollisionTauFromCellDevice(s, c);
        cellCollisionProbability =
            (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
          ? 0.0
          : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
    }
    __syncthreads();

    const int start = DirectBase ? baseBegin : injectionBegin;
    const int end = DirectBase ? baseEnd : injectionEnd;
    double sums[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double prob = cellCollisionProbability;
    if (prob > 0.0)
    {
        for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
        {
            const int i = DirectBase ? pos : s.sortedParticleIndex[pos];
            if
            (
                i < 0 || i >= s.particleCapacity
             || s.pStatus[i] == 0 || s.pCellId[i] != c
            )
            {
                continue;
            }
            unsigned long long rng = s.pRng[i];
            if (uniform01Device(rng) >= prob)
            {
                s.pRng[i] = rng;
                continue;
            }

            const double m =
                clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
            const double ux = finiteOr(s.pux[i], 0.0);
            const double uy = finiteOr(s.puy[i], 0.0);
            const double uz = finiteOr(s.puz[i], 0.0);
            const double theta = particleMomentThetaDevice(s, i);
            const double d =
                clampMin(finiteOr(s.pd[i], s.particleDiameterFallback), 1.0e-12);
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
            sums[0] += m;
            sums[1] += m*ux;
            sums[2] += m*uy;
            sums[3] += m*uz;
            sums[4] += m*specificEnergy;
            sums[5] += m*d;
            sums[6] += m*d*d;
            sums[7] += 1.0;
            s.pRng[i] = rng;
            s.pStatus[i] = 2;
        }
    }

    extern __shared__ double warpPartials[];
    blockReduceComponentSums<8>(sums, warpPartials);
    if (threadIdx.x == 0)
    {
        if (AddToCell)
        {
            s.poissonPoolMass[c] += sums[0];
            s.poissonPoolMomX[c] += sums[1];
            s.poissonPoolMomY[c] += sums[2];
            s.poissonPoolMomZ[c] += sums[3];
            s.poissonPoolEnergy[c] += sums[4];
            s.poissonPoolDiameter[c] += sums[5];
            s.poissonPoolDiameter2[c] += sums[6];
            s.poolThermalCount[c] += static_cast<int>(sums[7]);
        }
        else
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
    }
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

        const double theta = particleMomentThetaDevice(s, i);
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

        const double m =
            clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
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
    const bool directIndex,
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
            const int i = directIndex ? pos : s.sortedParticleIndex[pos];
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

            const double theta = particleMomentThetaDevice(s, i);
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

            const double m =
                clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
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

template<bool PoissonMode>
__device__ __forceinline__ void accumulateCsrSplitLogicalPoolParticle
(
    DeviceState& s,
    const int c,
    const int i,
    const double collisionProbability,
    double& mass,
    double& momX,
    double& momY,
    double& momZ,
    double& energy,
    double& diameter,
    double& diameter2,
    double& count
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

    const double theta = particleMomentThetaDevice(s, i);
    if (!PoissonMode && theta <= 10.0*s.thetaMin)
    {
        return;
    }

    if (PoissonMode)
    {
        unsigned long long rng = s.pRng[i];
        if (uniform01Device(rng) >= collisionProbability)
        {
            s.pRng[i] = rng;
            return;
        }
        s.pRng[i] = rng;
    }

    const double m =
        clampMin(finiteOr(s.pm[i], s.injectionParcelMass), 0.0);
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

template<bool PoissonMode>
__device__ void accumulateCsrSplitLogicalPoolTask
(
    DeviceState& s,
    const int c,
    const int logicalBegin,
    const int logicalEnd,
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

    const int baseBegin = s.preBaseCellOffset[c];
    const int baseCount = s.preBaseCellOffset[c + 1] - baseBegin;
    const int injectionBegin = s.cellParticleOffset[c];
    const int baseLogicalEnd = min(logicalEnd, baseCount);
    for
    (
        int logical = logicalBegin + threadIdx.x;
        logical < baseLogicalEnd;
        logical += blockDim.x
    )
    {
        accumulateCsrSplitLogicalPoolParticle<PoissonMode>
        (
            s, c, baseBegin + logical, collisionProbability,
            mass, momX, momY, momZ, energy, diameter, diameter2, count
        );
    }

    const int injectionLogicalBegin = max(logicalBegin, baseCount);
    for
    (
        int logical = injectionLogicalBegin + threadIdx.x;
        logical < logicalEnd;
        logical += blockDim.x
    )
    {
        const int injectionPosition =
            injectionBegin + logical - baseCount;
        accumulateCsrSplitLogicalPoolParticle<PoissonMode>
        (
            s, c, s.sortedParticleIndex[injectionPosition],
            collisionProbability,
            mass, momX, momY, momZ, energy, diameter, diameter2, count
        );
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


template<bool PoissonMode>
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
        if (task >= *s.csrHeavyTaskCount)
        {
            return;
        }

        const int c = s.csrHeavyTaskCell[task];
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
        accumulateCsrHeavyPoolTask<PoissonMode>
        (
            s,
            c,
            s.csrHeavyTaskBegin[task],
            s.csrHeavyTaskEnd[task],
            false,
            collisionProbability,
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

#include "../../../gpu/CsrSegmentedPoolWorkers.cuh"

int launchCsrHeavyPoolReduction
(
    DeviceState* s,
    const double dt,
    const bool poissonMode,
    const int block
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
    if (poissonMode)
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulateCsrHeavyPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR heavy pool persistent kernel launch", err);
        return 1;
    }

    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR heavy pool finalize cursor", err);
        return 1;
    }
    const int finalizeGridFromSm =
        s->csrHeavyWorkerGrid/s->heavyResidentBlocksPerSm;
    const int finalizeGrid =
        finalizeGridFromSm < s->nCells ? finalizeGridFromSm : s->nCells;
    finalizeCsrHeavyPoolCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrHeavyPoolCellsKernel launch", err);
        return 1;
    }
    return 0;
#endif
}

template<bool DirectBase>
__global__ void accumulateSplitCsrHeavyPoolTasksPersistentKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ double collisionProbability;
    extern __shared__ double warpPartials[];
    int* const cursor = DirectBase
      ? s.csrHeavyTaskCursor
      : s.csrHeavyInjectionTaskCursor;
    const int* const count = DirectBase
      ? s.csrHeavyTaskCount
      : s.csrHeavyInjectionTaskCount;

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            task = atomicAdd(cursor, 1);
        }
        __syncthreads();
        if (task >= *count)
        {
            return;
        }

        const int c = DirectBase
          ? s.csrHeavyTaskCell[task]
          : s.csrHeavyInjectionTaskCell[task];
        if (threadIdx.x == 0)
        {
            const double tauColl = granularCollisionTauFromCellDevice(s, c);
            collisionProbability =
                (!(tauColl < 0.5*OfGreat) || tauColl <= OfSmall)
              ? 0.0
              : clampRange(1.0 - exp(-dt/tauColl), 0.0, 1.0);
        }
        __syncthreads();

        double sums[8];
        const int begin = DirectBase
          ? s.csrHeavyTaskBegin[task]
          : s.csrHeavyInjectionTaskBegin[task];
        const int end = DirectBase
          ? s.csrHeavyTaskEnd[task]
          : s.csrHeavyInjectionTaskEnd[task];
        accumulateCsrHeavyPoolTask<true>
        (
            s,
            c,
            begin,
            end,
            DirectBase,
            collisionProbability,
            sums,
            warpPartials
        );
        if (threadIdx.x == 0)
        {
            double* const partials = DirectBase
              ? s.csrHeavyPartials
              : s.csrHeavyInjectionPartials;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
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

__global__ void finalizeSplitCsrHeavyPoolCellsKernel(DeviceState* sp)
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
        const int baseFirst = s.csrHeavyCellTaskStart[c];
        const int baseCount = s.csrHeavyCellTaskCount[c];
        for (int local = threadIdx.x; local < baseCount; local += blockDim.x)
        {
            const int task = baseFirst + local;
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
        const int injectionFirst = s.csrHeavyInjectionCellTaskStart[c];
        const int injectionCount = s.csrHeavyInjectionCellTaskCount[c];
        for
        (
            int local = threadIdx.x;
            local < injectionCount;
            local += blockDim.x
        )
        {
            const int task = injectionFirst + local;
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyInjectionPartials
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

int launchSplitCsrHeavyPoolReduction
(
    DeviceState* s,
    const double dt,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyInjectionTaskCursor, 0, sizeof(int));
    }
    if (err != cudaSuccess)
    {
        setLastError("reset split-Dpre heavy pool cursors", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    accumulateSplitCsrHeavyPoolTasksPersistentKernel<true>
        <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("split-Dpre heavy base pool launch", err);
        return 1;
    }
    if (s->nBoundarySources > 0)
    {
        accumulateSplitCsrHeavyPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("split-Dpre heavy injection pool launch", err);
            return 1;
        }
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset split-Dpre heavy finalizer cursor", err);
        return 1;
    }
    const int finalizeGridFromSm =
        s->csrHeavyWorkerGrid/s->heavyResidentBlocksPerSm;
    const int finalizeGrid =
        finalizeGridFromSm < s->nCells ? finalizeGridFromSm : s->nCells;
    finalizeSplitCsrHeavyPoolCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("split-Dpre heavy pool finalizer launch", err);
        return 1;
    }
    return 0;
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

__device__ void appendSelectedStuckParticleIndex
(
    DeviceState& s,
    const int i
)
{
    if (s.pStuck[i] == 0)
    {
        return;
    }

    const int slot = atomicAdd(s.compactCountDevice, 1);
    if (slot < 0 || slot >= s.particleCapacity)
    {
        asm("trap;");
    }
    s.compactPStatus[slot] = i;
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

    const bool wasStuck = s.pStuck[i] != 0;
    const bool wasFiniteContact =
        s.pStuck[i] == Foam::gpuThermal::particleWallTransientRebound
     || s.pStuck[i] == Foam::gpuThermal::particleWallTransientDeposit;
    const double finiteContactAge = wasFiniteContact ? s.pTheta[i] : 0.0;
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

    if (!finiteDevice(s.pm[i]) || s.pm[i] <= 0.0)
    {
        asm("trap;");
    }
    if (!wasStuck)
    {
        s.pStuckFaceId[i] = -1;
        s.pDepositionArea[i] = 0.0f;
        s.pContactDuration[i] = 0.0f;
        s.pContactMaximumArea[i] = 0.0f;
        s.pContactPeakFraction[i] = 0.0f;
    }
    else if (wasFiniteContact)
    {
        s.pTheta[i] = finiteContactAge;
    }
    s.pRng[i] = rng;

    const double sampleMass = s.pm[i];
    const double sampleFluctuationX = s.pux[i] - targetUx;
    const double sampleFluctuationY = s.puy[i] - targetUy;
    const double sampleFluctuationZ = s.puz[i] - targetUz;
    atomicAdd
    (
        &s.poolThermalSumUx[c],
        sampleMass*sampleFluctuationX
    );
    atomicAdd
    (
        &s.poolThermalSumUy[c],
        sampleMass*sampleFluctuationY
    );
    atomicAdd
    (
        &s.poolThermalSumUz[c],
        sampleMass*sampleFluctuationZ
    );
    atomicAdd
    (
        &s.poolThermalSumU2[c],
        sampleMass*sqr3
        (
            sampleFluctuationX,
            sampleFluctuationY,
            sampleFluctuationZ
        )
    );
    appendSelectedStuckParticleIndex(s, i);
}

__device__ void finalizeOneThermalizedMobileParticle
(
    DeviceState& s,
    const int i,
    const double candidateUx,
    const double candidateUy,
    const double candidateUz
)
{
    s.pux[i] = candidateUx;
    s.puy[i] = candidateUy;
    s.puz[i] = candidateUz;
    s.pTheta[i] = 0.0;
    s.pStatus[i] = 1;
}

__device__ void finalizeOneThermalizedStuckParticle
(
    DeviceState& s,
    const int i,
    const double candidateUx,
    const double candidateUy,
    const double candidateUz,
    const double poolMeanUx,
    const double poolMeanUy,
    const double poolMeanUz
)
{
    if (s.pStuck[i] == 0)
    {
        asm("trap;");
    }

    const int faceI = s.pStuckFaceId[i];
    if
    (
        faceI < s.nInternalFaces
     || faceI >= s.nFaces
     || s.particleStuckCandidateMask[faceI] == 0
    )
    {
        asm("trap;");
    }
    const Foam::gpuThermal::CapillaryDetachmentState capillary =
        Foam::gpuThermal::evaluateCapillaryDetachmentState
        (
            s.pT[i],
            s.pd[i],
            s.particleWallAdhesionEnergyScale,
            s.particleWallContactAngleCosine
        );
    const double wallUx = s.gasBoundaryUx[faceI];
    const double wallUy = s.gasBoundaryUy[faceI];
    const double wallUz = s.gasBoundaryUz[faceI];
                                                                        
                                                                            
    const double fluctuationUx = candidateUx - poolMeanUx;
    const double fluctuationUy = candidateUy - poolMeanUy;
    const double fluctuationUz = candidateUz - poolMeanUz;
    const double sampledSpecificEnergy =
        0.5*sqr3(fluctuationUx, fluctuationUy, fluctuationUz);
    const double contactAreaScale =
        s.particleWallContactAreaScale[faceI];
    if (!(contactAreaScale > 0.0) || contactAreaScale > 1.0)
    {
        asm("trap;");
    }
    const bool finiteContact =
        s.pStuck[i] == Foam::gpuThermal::particleWallTransientRebound
     || s.pStuck[i] == Foam::gpuThermal::particleWallTransientDeposit;
    const double contactAge = finiteContact ? s.pTheta[i] : 0.0;
    Foam::gpuThermal::CapillaryContactDamageResult damage;
    double accumulatedDamageArea = 0.0;
    if (finiteContact)
    {
        const double duration = static_cast<double>(s.pContactDuration[i]);
        const double maximumArea = static_cast<double>(s.pContactMaximumArea[i]);
        const double peakTimeFraction =
            static_cast<double>(s.pContactPeakFraction[i]);
        const double oldDamage = static_cast<double>(s.pDepositionArea[i]);
        const double kinematicArea =
            maximumArea*Foam::gpuThermal::normalizedKinematicArea
            (
                contactAge/duration, peakTimeFraction
            );
        const unsigned char interactionType =
            s.particleStuckCandidateMask[faceI];
        const double frozenArea =
            interactionType
         == Foam::gpuThermal::particleWallSolidifyingDeposition
          ? static_cast<double>(s.pColdFrozenArea[i])
          : interactionType == Foam::gpuThermal::particleWallColdWall2D
          ? static_cast<double>(s.pCold2DFrozenArea[i])
          : 0.0;
        const double remainingArea =
            fmax(kinematicArea, frozenArea) - oldDamage;
        const double adhesionSpecificEnergyPerArea =
            capillary.adhesionSpecificEnergyJkg
           /capillary.equilibriumContactAreaM2;
        const double requiredEnergy =
            contactAreaScale*adhesionSpecificEnergyPerArea
           *clampMin(remainingArea, 0.0);
        if
        (
            !(duration > 0.0) || !(maximumArea > 0.0)
         || !(peakTimeFraction > 0.0) || !(peakTimeFraction < 1.0)
         || oldDamage < 0.0 || !(adhesionSpecificEnergyPerArea > 0.0)
        )
        {
            asm("trap;");
        }
        if (sampledSpecificEnergy >= requiredEnergy)
        {
            const double residual = sampledSpecificEnergy - requiredEnergy;
            damage = {finiteDevice(residual), true, 0.0, residual};
        }
        else
        {
            const double consumedArea =
                sampledSpecificEnergy
               /(contactAreaScale*adhesionSpecificEnergyPerArea);
            accumulatedDamageArea = oldDamage + consumedArea;
            damage =
            {
                finiteDevice(accumulatedDamageArea),
                false,
                remainingArea - consumedArea,
                0.0
            };
        }
    }
    else
    {
        const double currentContactAreaM2 =
            static_cast<double>(s.pDepositionArea[i]);
        damage = Foam::gpuThermal::applyCapillaryContactDamage
        (
            capillary,
            currentContactAreaM2,
            contactAreaScale,
            sampledSpecificEnergy
        );
    }
    if
    (
        !damage.valid
     || nonFiniteDevice(sampledSpecificEnergy)
     || sampledSpecificEnergy < 0.0
    )
    {
        asm("trap;");
    }

    s.pTheta[i] = finiteContact ? contactAge : 0.0;
    if (damage.detached)
    {
        const double area = s.magSf[faceI];
        if (!(area > 0.0) || nonFiniteDevice(area))
        {
            asm("trap;");
        }
        const double fluctuationScale =
            sampledSpecificEnergy > 0.0
          ? sqrt
            (
                clampRange
                (
                    damage.residualSpecificEnergyJkg/sampledSpecificEnergy,
                    0.0,
                    1.0
                )
            )
          : 0.0;
        const double outwardNx = s.Sfx[faceI]/area;
        const double outwardNy = s.Sfy[faceI]/area;
        const double outwardNz = s.Sfz[faceI]/area;
        double releaseUx = poolMeanUx + fluctuationScale*fluctuationUx;
        double releaseUy = poolMeanUy + fluctuationScale*fluctuationUy;
        double releaseUz = poolMeanUz + fluctuationScale*fluctuationUz;
        const double relativeNormal =
            (releaseUx - wallUx)*outwardNx
          + (releaseUy - wallUy)*outwardNy
          + (releaseUz - wallUz)*outwardNz;
        if (relativeNormal > 0.0)
        {
            releaseUx -= 2.0*relativeNormal*outwardNx;
            releaseUy -= 2.0*relativeNormal*outwardNy;
            releaseUz -= 2.0*relativeNormal*outwardNz;
        }
        s.pux[i] = releaseUx;
        s.puy[i] = releaseUy;
        s.puz[i] = releaseUz;
        s.puxOld[i] = releaseUx;
        s.puyOld[i] = releaseUy;
        s.puzOld[i] = releaseUz;
        s.pStuck[i] = 0;
        s.pStuckFaceId[i] = -1;
        s.pDepositionArea[i] = 0.0f;
        s.pContactDuration[i] = 0.0f;
        s.pContactMaximumArea[i] = 0.0f;
        s.pContactPeakFraction[i] = 0.0f;
        clearColdWallParticleState(s, i);
        clearColdWall2DParticleState(s, i);
    }
    else
    {
        if
        (
            !(damage.remainingContactAreaM2 > 0.0)
         || damage.remainingContactAreaM2 > static_cast<double>(FLT_MAX)
        )
        {
            asm("trap;");
        }
        s.pux[i] = 0.0;
        s.puy[i] = 0.0;
        s.puz[i] = 0.0;
        if (finiteContact)
        {
            s.pDepositionArea[i] =
                static_cast<float>(accumulatedDamageArea);
        }
        else
        {
            s.puxOld[i] = 0.0;
            s.puyOld[i] = 0.0;
            s.puzOld[i] = 0.0;
            s.pDepositionArea[i] =
                static_cast<float>(damage.remainingContactAreaM2);
        }
    }
    s.pStatus[i] = 1;
}

template<bool StuckPath>
__device__ void finalizeOneThermalizedParticlePath
(
    DeviceState& s,
    const int i,
    const double candidateUx,
    const double candidateUy,
    const double candidateUz,
    const double poolMeanUx,
    const double poolMeanUy,
    const double poolMeanUz
)
{
    if (StuckPath)
    {
        finalizeOneThermalizedStuckParticle
        (
            s, i,
            candidateUx, candidateUy, candidateUz,
            poolMeanUx, poolMeanUy, poolMeanUz
        );
    }
    else
    {
        finalizeOneThermalizedMobileParticle
        (
            s, i,
            candidateUx, candidateUy, candidateUz
        );
    }
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

template<bool StuckPath>
__device__ void correctOnePoissonThermalizedParticlePath
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
    if (StuckPath)
    {
        if (s.pStuck[i] == 0)
        {
            asm("trap;");
        }
    }
    else if (s.pStuck[i] != 0)
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

    const double targetUx = s.poissonPoolMomX[c]/poolMass;
    const double targetUy = s.poissonPoolMomY[c]/poolMass;
    const double targetUz = s.poissonPoolMomZ[c]/poolMass;
    const double targetMean2 = sqr3(targetUx, targetUy, targetUz);
    const double targetThetaRaw =
        clampMin(s.poissonPoolEnergy[c]/poolMass - 0.5*targetMean2, 0.0)/1.5;
    const double meanParticleMass = poolMass/static_cast<double>(n);
    const double sampleMeanDeltaX = s.poolThermalSumUx[c]/poolMass;
    const double sampleMeanDeltaY = s.poolThermalSumUy[c]/poolMass;
    const double sampleMeanDeltaZ = s.poolThermalSumUz[c]/poolMass;
    const double sampleMeanDelta2 = sqr3
    (
        sampleMeanDeltaX,
        sampleMeanDeltaY,
        sampleMeanDeltaZ
    );
    const double sampleFluctuationEnergy = 0.5*clampMin
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

    const double targetRandomEnergy = 1.5*poolMass*targetTheta;

    if (targetRandomEnergy <= 0.0)
    {
        finalizeOneThermalizedParticlePath<StuckPath>
        (
            s, i,
            targetUx, targetUy, targetUz,
            targetUx, targetUy, targetUz
        );
        return;
    }

    const bool retainSingletonTheta =
        n <= 1
     || sampleFluctuationEnergy <= OfSmall*meanParticleMass;
    if (retainSingletonTheta)
    {
        if (StuckPath)
        {
            finalizeOneThermalizedParticlePath<true>
            (
                s, i,
                s.pux[i], s.puy[i], s.puz[i],
                targetUx, targetUy, targetUz
            );
        }
        else
        {
            s.pux[i] = targetUx;
            s.puy[i] = targetUy;
            s.puz[i] = targetUz;
            s.pTheta[i] = targetTheta;
            s.pStatus[i] = 1;
        }
        return;
    }

    double scale = 1.0;
    scale = sqrt(targetRandomEnergy/sampleFluctuationEnergy);
    const double correctedUx = targetUx + scale*
        ((s.pux[i] - targetUx) - sampleMeanDeltaX);
    const double correctedUy = targetUy + scale*
        ((s.puy[i] - targetUy) - sampleMeanDeltaY);
    const double correctedUz = targetUz + scale*
        ((s.puz[i] - targetUz) - sampleMeanDeltaZ);
    if
    (
        nonFiniteDevice(correctedUx) || nonFiniteDevice(correctedUy)
     || nonFiniteDevice(correctedUz)
    )
    {
        asm("trap;");
    }
    finalizeOneThermalizedParticlePath<StuckPath>
    (
        s, i,
        correctedUx, correctedUy, correctedUz,
        targetUx, targetUy, targetUz
    );
}

__device__ void correctOnePoissonThermalizedMobileParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    correctOnePoissonThermalizedParticlePath<false>
    (
        s,
        i,
        applyThetaDrag
    );
}

__device__ void correctOnePoissonThermalizedStuckParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    correctOnePoissonThermalizedParticlePath<true>
    (
        s,
        i,
        applyThetaDrag
    );
}

__global__ void correctPoissonThermalizedMobileParticlesKernel
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
        if (s.pStuck[i] != 0)
        {
            continue;
        }
        correctOnePoissonThermalizedMobileParticle
        (
            s,
            i,
            applyThetaDrag != 0
        );
    }
}

__global__ void correctPoissonThermalizedStuckParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
)
{
    DeviceState& s = *sp;
    const int selectedStuckCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    for
    (
        int pos = blockIdx.x*blockDim.x + threadIdx.x;
        pos < selectedStuckCount;
        pos += blockDim.x*gridDim.x
    )
    {
        const int i = s.compactPStatus[pos];
        if (i < 0 || i >= s.particleCapacity)
        {
            asm("trap;");
        }
        correctOnePoissonThermalizedStuckParticle
        (
            s,
            i,
            applyThetaDrag != 0
        );
    }
}

template<int BlockSize>
__global__ void gatherCellLocalParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;
    if (c >= s.nCells || blockDim.x != BlockSize)
    {
        return;
    }

    using BlockScan = cub::BlockScan<int, BlockSize>;
    __shared__ typename BlockScan::TempStorage scanStorage;

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    const int outputStart = s.compactCellOffset[c];
    int tileOutputOffset = 0;

    for (int tileStart = start; tileStart < end; tileStart += BlockSize)
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
            s.compactPStuck[dst] = s.pStuck[i];
            s.compactPStuckFaceId[dst] = s.pStuckFaceId[i];
            s.compactPDepositionArea[dst] = s.pDepositionArea[i];
            s.compactPContactDuration[dst] = s.pContactDuration[i];
            s.compactPContactMaximumArea[dst] = s.pContactMaximumArea[i];
            s.compactPContactPeakFraction[dst] = s.pContactPeakFraction[i];
            if (s.coldWallSolidificationEnabled != 0)
            {
                for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
                {
                    s.compactPColdNodeSpecificEnthalpy
                    [dst*Foam::gpuThermal::coldWallAxialNodeCount + node] =
                        s.pColdNodeSpecificEnthalpy
                        [i*Foam::gpuThermal::coldWallAxialNodeCount + node];
                }
                for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
                {
                    s.compactPColdRingSolidMass
                    [dst*Foam::gpuThermal::coldWallRadialRingCount + ring] =
                        s.pColdRingSolidMass
                        [i*Foam::gpuThermal::coldWallRadialRingCount + ring];
                }
                s.compactPColdFrozenArea[dst] = s.pColdFrozenArea[i];
                s.compactPColdContactAge[dst] = s.pColdContactAge[i];
            }
            if (s.coldWall2DEnabled != 0)
            {
                for
                (
                    int node = 0;
                    node < Foam::gpuThermal::coldWall2DNodeCount;
                    ++node
                )
                {
                    s.compactPCold2DNodeSpecificEnthalpy
                    [dst*Foam::gpuThermal::coldWall2DNodeCount + node] =
                        s.pCold2DNodeSpecificEnthalpy
                        [i*Foam::gpuThermal::coldWall2DNodeCount + node];
                }
                for
                (
                    int ring = 0;
                    ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
                    ++ring
                )
                {
                    s.compactPCold2DRingContactAge
                    [dst*Foam::gpuThermal::coldWall2DRadialNodeCount + ring] =
                        s.pCold2DRingContactAge
                        [i*Foam::gpuThermal::coldWall2DRadialNodeCount + ring];
                }
                s.compactPCold2DFrozenArea[dst] = s.pCold2DFrozenArea[i];
            }
            s.compactPRng[dst] = s.pRng[i];
            s.compactPOrigId[dst] = s.pOrigId[i];
            if
            (
                s.compactPStuck[dst]
             != Foam::gpuThermal::particleWallMobile
            )
            {
                Foam::gpuWall::publishWallBoundParticleIndex(s, dst);
            }
        }

        tileOutputOffset += tileCount;
        __syncthreads();
    }
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
        s.compactPStuck[dst] = s.pStuck[i];
        s.compactPStuckFaceId[dst] = s.pStuckFaceId[i];
        s.compactPDepositionArea[dst] = s.pDepositionArea[i];
        s.compactPContactDuration[dst] = s.pContactDuration[i];
        s.compactPContactMaximumArea[dst] = s.pContactMaximumArea[i];
        s.compactPContactPeakFraction[dst] = s.pContactPeakFraction[i];
        if (s.coldWallSolidificationEnabled != 0)
        {
            for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
            {
                s.compactPColdNodeSpecificEnthalpy
                [dst*Foam::gpuThermal::coldWallAxialNodeCount + node] =
                    s.pColdNodeSpecificEnthalpy
                    [i*Foam::gpuThermal::coldWallAxialNodeCount + node];
            }
            for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
            {
                s.compactPColdRingSolidMass
                [dst*Foam::gpuThermal::coldWallRadialRingCount + ring] =
                    s.pColdRingSolidMass
                    [i*Foam::gpuThermal::coldWallRadialRingCount + ring];
            }
            s.compactPColdFrozenArea[dst] = s.pColdFrozenArea[i];
            s.compactPColdContactAge[dst] = s.pColdContactAge[i];
        }
        if (s.coldWall2DEnabled != 0)
        {
            for
            (
                int node = 0;
                node < Foam::gpuThermal::coldWall2DNodeCount;
                ++node
            )
            {
                s.compactPCold2DNodeSpecificEnthalpy
                [dst*Foam::gpuThermal::coldWall2DNodeCount + node] =
                    s.pCold2DNodeSpecificEnthalpy
                    [i*Foam::gpuThermal::coldWall2DNodeCount + node];
            }
            for
            (
                int ring = 0;
                ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
                ++ring
            )
            {
                s.compactPCold2DRingContactAge
                [dst*Foam::gpuThermal::coldWall2DRadialNodeCount + ring] =
                    s.pCold2DRingContactAge
                    [i*Foam::gpuThermal::coldWall2DRadialNodeCount + ring];
            }
            s.compactPCold2DFrozenArea[dst] = s.pCold2DFrozenArea[i];
        }
        s.compactPRng[dst] = s.pRng[i];
        s.compactPOrigId[dst] = s.pOrigId[i];
        if
        (
            s.compactPStuck[dst]
         != Foam::gpuThermal::particleWallMobile
        )
        {
            Foam::gpuWall::publishWallBoundParticleIndex(s, dst);
        }
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
    swapParticlePointerDevice(s.pStuck, s.compactPStuck);
    swapParticlePointerDevice(s.pStuckFaceId, s.compactPStuckFaceId);
    swapParticlePointerDevice(s.pDepositionArea, s.compactPDepositionArea);
    swapParticlePointerDevice(s.pContactDuration, s.compactPContactDuration);
    swapParticlePointerDevice(s.pContactMaximumArea, s.compactPContactMaximumArea);
    swapParticlePointerDevice(s.pContactPeakFraction, s.compactPContactPeakFraction);
    if (s.coldWallSolidificationEnabled != 0)
    {
        swapParticlePointerDevice(s.pColdNodeSpecificEnthalpy, s.compactPColdNodeSpecificEnthalpy);
        swapParticlePointerDevice(s.pColdRingSolidMass, s.compactPColdRingSolidMass);
        swapParticlePointerDevice(s.pColdFrozenArea, s.compactPColdFrozenArea);
        swapParticlePointerDevice(s.pColdContactAge, s.compactPColdContactAge);
    }
    if (s.coldWall2DEnabled != 0)
    {
        swapParticlePointerDevice(s.pCold2DNodeSpecificEnthalpy, s.compactPCold2DNodeSpecificEnthalpy);
        swapParticlePointerDevice(s.pCold2DRingContactAge, s.compactPCold2DRingContactAge);
        swapParticlePointerDevice(s.pCold2DFrozenArea, s.compactPCold2DFrozenArea);
    }
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
    swapParticlePointerDevice(s.pStuck, s.compactPStuck);
    swapParticlePointerDevice(s.pStuckFaceId, s.compactPStuckFaceId);
    swapParticlePointerDevice(s.pDepositionArea, s.compactPDepositionArea);
    swapParticlePointerDevice(s.pContactDuration, s.compactPContactDuration);
    swapParticlePointerDevice(s.pContactMaximumArea, s.compactPContactMaximumArea);
    swapParticlePointerDevice(s.pContactPeakFraction, s.compactPContactPeakFraction);
    if (s.coldWallSolidificationEnabled != 0)
    {
        swapParticlePointerDevice(s.pColdNodeSpecificEnthalpy, s.compactPColdNodeSpecificEnthalpy);
        swapParticlePointerDevice(s.pColdRingSolidMass, s.compactPColdRingSolidMass);
        swapParticlePointerDevice(s.pColdFrozenArea, s.compactPColdFrozenArea);
        swapParticlePointerDevice(s.pColdContactAge, s.compactPColdContactAge);
    }
    if (s.coldWall2DEnabled != 0)
    {
        swapParticlePointerDevice(s.pCold2DNodeSpecificEnthalpy, s.compactPCold2DNodeSpecificEnthalpy);
        swapParticlePointerDevice(s.pCold2DRingContactAge, s.compactPCold2DRingContactAge);
        swapParticlePointerDevice(s.pCold2DFrozenArea, s.compactPCold2DFrozenArea);
    }
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
    swapParticlePointerHost(s->pStuck, s->compactPStuck);
    swapParticlePointerHost(s->pStuckFaceId, s->compactPStuckFaceId);
    swapParticlePointerHost(s->pDepositionArea, s->compactPDepositionArea);
    swapParticlePointerHost(s->pContactDuration, s->compactPContactDuration);
    swapParticlePointerHost(s->pContactMaximumArea, s->compactPContactMaximumArea);
    swapParticlePointerHost(s->pContactPeakFraction, s->compactPContactPeakFraction);
    if (s->coldWallSolidificationEnabled != 0)
    {
        swapParticlePointerHost(s->pColdNodeSpecificEnthalpy, s->compactPColdNodeSpecificEnthalpy);
        swapParticlePointerHost(s->pColdRingSolidMass, s->compactPColdRingSolidMass);
        swapParticlePointerHost(s->pColdFrozenArea, s->compactPColdFrozenArea);
        swapParticlePointerHost(s->pColdContactAge, s->compactPColdContactAge);
    }
    if (s->coldWall2DEnabled != 0)
    {
        swapParticlePointerHost(s->pCold2DNodeSpecificEnthalpy, s->compactPCold2DNodeSpecificEnthalpy);
        swapParticlePointerHost(s->pCold2DRingContactAge, s->compactPCold2DRingContactAge);
        swapParticlePointerHost(s->pCold2DFrozenArea, s->compactPCold2DFrozenArea);
    }
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
        const double theta = particleMomentThetaDevice(s, i);
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
        atomicAdd(&s.momRhoHpP[c], m*particleSpecificEnthalpyDevice(tp));
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
    if (s.csrHeavyReductionEnabled != 0)
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
        const double theta = particleMomentThetaDevice(s, i);
        const double d =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        const double tp =
            clampRange
            (
                finiteOr(s.pT[i], s.TpMin),
                s.TpMin,
                s.TpMax
            );
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += m*(0.5*sqr3(ux, uy, uz) + 1.5*theta);
        diameter += m*d;
        heat += m*particleSpecificEnthalpyDevice(tp);
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
        const double theta = particleMomentThetaDevice(s, i);
        const double d =
            clampMin
            (
                finiteOr(s.pd[i], s.particleDiameterFallback),
                1.0e-12
            );
        const double tp =
            clampRange
            (
                finiteOr(s.pT[i], s.TpMin),
                s.TpMin,
                s.TpMax
            );
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += m*(0.5*sqr3(ux, uy, uz) + 1.5*theta);
        diameter += m*d;
        heat += m*particleSpecificEnthalpyDevice(tp);
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

#include "../../../gpu/CsrSegmentedMomentWorkers.cuh"

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
    const int finalizeGridFromSm =
        s->csrHeavyWorkerGrid/s->heavyResidentBlocksPerSm;
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

__global__ void clearSplitPreInjectionBinsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c <= s.nCells; c += stride)
    {
        s.cellParticleCount[c] = 0;
        s.cellParticleOffset[c] = 0;
        if (c < s.nCells)
        {
            s.cellParticleWrite[c] = 0;
        }
    }
}

template<bool WarpAggregated>
__global__ void countSplitPreInjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int begin = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int end = clampRange
    (
        *s.particleCountDevice,
        begin,
        s.particleCapacity
    );
    for
    (
        int i = begin + blockIdx.x*blockDim.x + threadIdx.x;
        i < end;
        i += blockDim.x*gridDim.x
    )
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

__global__ void initialiseSplitPreInjectionWritesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.cellParticleWrite[c] = s.cellParticleOffset[c];
    }
}

template<bool WarpAggregated>
__global__ void scatterSplitPreInjectionParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int begin = clampRange
    (
        *s.preBaseParticleCountDevice,
        0,
        s.particleCapacity
    );
    const int end = clampRange
    (
        *s.particleCountDevice,
        begin,
        s.particleCapacity
    );
    for
    (
        int i = begin + blockIdx.x*blockDim.x + threadIdx.x;
        i < end;
        i += blockDim.x*gridDim.x
    )
    {
        const int c = s.pCellId[i];
        const bool valid = s.pStatus[i] != 0 && c >= 0 && c < s.nCells;
        if (!WarpAggregated)
        {
            if (valid)
            {
                const int pos = atomicAdd(&s.cellParticleWrite[c], 1);
                if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
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
            if (pos >= s.cellParticleOffset[c] && pos < s.cellParticleOffset[c + 1])
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

__global__ void countCsrReductionTasksKernel(DeviceState* sp, const int splitDirectory)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c > s.nCells) return;
    if (c == s.nCells)
    {
        s.csrCellTaskCount[c] = 0;
        return;
    }
    const int tile = s.csrHeavyTileParticles;
    if (tile <= 0) asm("trap;");
    if (splitDirectory == 0)
    {
        const int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        s.csrCellTaskCount[c] = count > 0 ? 1 + (count - 1)/tile : 0;
        return;
    }
    const int baseCount = s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
    const int injectionCount = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
    const int totalCount = baseCount + injectionCount;
    s.csrCellTaskCount[c] =
        totalCount > 0 ? 1 + (totalCount - 1)/tile : 0;
}

__device__ void writeCsrReductionTask
(
    DeviceState& s, const int task, const int cell,
    const int begin, const int end, const CsrReductionTaskSource taskSource
)
{
    if (task < 0 || task >= s.csrHeavyTaskCapacity || begin >= end) asm("trap;");
    const int source = static_cast<int>(taskSource);
    CsrReductionTask descriptor = {cell, begin, end, source};
    s.csrReductionTasks[task] = descriptor;
}

__global__ void materializeCsrReductionTasksKernel(DeviceState* sp, const int splitDirectory)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        const int taskStart = s.csrCellTaskOffset[c];
        const int nTasks = s.csrCellTaskCount[c];
        if (nTasks == 0) continue;
        if (taskStart < 0 || taskStart > s.csrHeavyTaskCapacity - nTasks) asm("trap;");
        if (nTasks > 1)
        {
            const int multiIndex = atomicAdd(s.csrHeavyCellCount, 1);
            if (multiIndex < 0 || multiIndex >= s.nCells) asm("trap;");
            s.csrMultiTaskCellList[multiIndex] = c;
        }
        const int tile = s.csrHeavyTileParticles;
        int localTask = 0;
        if (splitDirectory == 0)
        {
            const int begin = s.cellParticleOffset[c];
            const int end = s.cellParticleOffset[c + 1];
            for (; localTask < nTasks; ++localTask)
            {
                const int taskBegin = begin + localTask*tile;
                writeCsrReductionTask(s, taskStart + localTask, c, taskBegin,
                    min(end, taskBegin + tile), CsrReductionTaskSource::fullIndexed);
            }
            continue;
        }
        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        const int baseCount = baseEnd - baseBegin;
        const int injectionCount = injectionEnd - injectionBegin;
        const int totalCount = baseCount + injectionCount;
        for (int begin = 0; begin < totalCount; begin += tile)
        {
            writeCsrReductionTask(s, taskStart + localTask++, c, begin,
                min(totalCount, begin + tile), CsrReductionTaskSource::splitLogical);
        }
        if (localTask != nTasks) asm("trap;");
    }
}

__global__ void publishCsrReductionTaskCountKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    *s.csrHeavyTaskCount = s.csrCellTaskOffset[s.nCells];
}

int prepareCsrSegmentedReductionTasks(DeviceState* s, const int block, const bool splitDirectory)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    if (err != cudaSuccess) { setLastError("clear CSR multi-task cell count", err); return 1; }
    const int countGrid = (s->nCells + 1 + block - 1)/block;
    countCsrReductionTasksKernel<<<countGrid, block>>>(s->deviceState, splitDirectory ? 1 : 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("countCsrReductionTasksKernel launch", err); return 1; }
    err = cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage, s->cellScanTempBytes,
        s->csrCellTaskCount, s->csrCellTaskOffset, s->nCells + 1);
    if (err != cudaSuccess) { setLastError("CSR cell task count exclusive scan", err); return 1; }
    publishCsrReductionTaskCountKernel<<<1, 1>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("publish CSR segmented task count", err); return 1; }
    const int cellGrid = (s->nCells + block - 1)/block;
    materializeCsrReductionTasksKernel<<<cellGrid, block>>>(s->deviceState, splitDirectory ? 1 : 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) { setLastError("materializeCsrReductionTasksKernel launch", err); return 1; }
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
        if (count <= s.dynamicHeavyThreshold)
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

__global__ void buildSplitCsrHeavyReductionTasksKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.csrHeavyCellTaskStart[c] = 0;
        s.csrHeavyCellTaskCount[c] = 0;
        s.csrHeavyInjectionCellTaskStart[c] = 0;
        s.csrHeavyInjectionCellTaskCount[c] = 0;

        const int baseBegin = s.preBaseCellOffset[c];
        const int baseEnd = s.preBaseCellOffset[c + 1];
        const int injectionBegin = s.cellParticleOffset[c];
        const int injectionEnd = s.cellParticleOffset[c + 1];
        const int baseCount = baseEnd - baseBegin;
        const int injectionCount = injectionEnd - injectionBegin;
        if (baseCount + injectionCount <= s.dynamicHeavyThreshold)
        {
            continue;
        }

        const int heavyCellIndex = atomicAdd(s.csrHeavyCellCount, 1);
        if (heavyCellIndex < 0 || heavyCellIndex >= s.nCells)
        {
            asm("trap;");
        }
        s.csrHeavyCellList[heavyCellIndex] = c;

        if (baseCount > 0)
        {
            const int nTasks = 1 + (baseCount - 1)/s.csrHeavyTileParticles;
            const int taskStart = atomicAdd(s.csrHeavyTaskCount, nTasks);
            if
            (
                taskStart < 0 || nTasks <= 0
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
                const int begin = baseBegin + localTask*s.csrHeavyTileParticles;
                const int end = min(begin + s.csrHeavyTileParticles, baseEnd);
                s.csrHeavyTaskCell[task] = c;
                s.csrHeavyTaskBegin[task] = begin;
                s.csrHeavyTaskEnd[task] = end;
            }
        }

        if (injectionCount > 0)
        {
            const int nTasks =
                1 + (injectionCount - 1)/s.csrHeavyTileParticles;
            const int taskStart =
                atomicAdd(s.csrHeavyInjectionTaskCount, nTasks);
            if
            (
                taskStart < 0 || nTasks <= 0
             || taskStart > s.csrHeavyTaskCapacity - nTasks
            )
            {
                asm("trap;");
            }
            s.csrHeavyInjectionCellTaskStart[c] = taskStart;
            s.csrHeavyInjectionCellTaskCount[c] = nTasks;
            for (int localTask = 0; localTask < nTasks; ++localTask)
            {
                const int task = taskStart + localTask;
                const int begin =
                    injectionBegin + localTask*s.csrHeavyTileParticles;
                const int end =
                    min(begin + s.csrHeavyTileParticles, injectionEnd);
                s.csrHeavyInjectionTaskCell[task] = c;
                s.csrHeavyInjectionTaskBegin[task] = begin;
                s.csrHeavyInjectionTaskEnd[task] = end;
            }
        }
    }
}

int prepareSplitCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks(s, block, true);
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
    }
    cudaError_t err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyInjectionTaskCount, 0, sizeof(int));
    }
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    }
    if (err != cudaSuccess)
    {
        setLastError("clear split-Dpre heavy schedule", err);
        return 1;
    }
    const int cellGrid = (s->nCells + block - 1)/block;
    buildSplitCsrHeavyReductionTasksKernel
        <<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("build split-Dpre heavy schedule launch", err);
        return 1;
    }
    return 0;
#endif
}

int prepareCsrHeavyReductionTasks(DeviceState* s, const int block)
{
    return prepareCsrSegmentedReductionTasks(s, block, false);
#if 0
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0)
    {
        return 0;
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

#include "../../../common/GpuToolB1.cuh"

int configureLaunchOccupancy(DeviceState* s)
{
    const int block = s->reductionBlockThreads;
    const int warpCount = (block + 31)/32;
    const size_t poolSharedBytes =
        8u*static_cast<size_t>(block)*sizeof(double);
    const size_t momentSharedBytes =
        8u*static_cast<size_t>(warpCount)*sizeof(double);
    cudaError_t err = cudaSuccess;
    if (s->csrHeavyReductionEnabled != 0)
    {
        int segmentedPool = 0;
        int segmentedMoment = 0;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (
            &segmentedPool,
            accumulateCsrSegmentedPoolTasksPersistentKernel<true>,
            block,
            momentSharedBytes
        );
        if (err == cudaSuccess)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &segmentedMoment,
                accumulateCsrSegmentedMomentTasksPersistentKernel,
                block,
                momentSharedBytes
            );
        }
        if (err != cudaSuccess || segmentedPool <= 0 || segmentedMoment <= 0)
        {
            if (err != cudaSuccess)
            {
                setLastError("CUDA segmented-kernel occupancy validation", err);
            }
            else
            {
                setLastErrorText("selected bn gives zero segmented-kernel occupancy");
            }
            return 1;
        }
        s->heavyResidentBlocksPerSm =
            segmentedPool < segmentedMoment ? segmentedPool : segmentedMoment;
        s->lightResidentBlocksPerSm = s->heavyResidentBlocksPerSm;
        s->csrHeavyWorkerGrid =
            s->multiprocessorCount*s->heavyResidentBlocksPerSm;
    }
    else
    {
        int lightPool = 0;
        int lightMoment = 0;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (
            &lightPool,
            accumulatePoissonPoolParticlesByCellKernel,
            block,
            poolSharedBytes
        );
        if (err == cudaSuccess)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &lightMoment,
                accumulateParticleMomentsSegmentedKernel,
                block,
                momentSharedBytes
            );
        }
        if (err != cudaSuccess || lightPool <= 0 || lightMoment <= 0)
        {
            if (err != cudaSuccess)
            {
                setLastError("CUDA light-kernel occupancy validation", err);
            }
            else
            {
                setLastErrorText("selected bn gives zero light-kernel occupancy");
            }
            return 1;
        }
        s->lightResidentBlocksPerSm =
            lightPool < lightMoment ? lightPool : lightMoment;
        s->heavyResidentBlocksPerSm = 0;
        s->csrHeavyWorkerGrid = 0;
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
                s->particleBlockThreads,
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

    const int capacityGrid =
        (s->particleCapacity + s->particleBlockThreads - 1)
       /s->particleBlockThreads;
    const int saturationGrid =
        s->multiprocessorCount*s->lightResidentBlocksPerSm;
    s->particleWorkGrid = capacityGrid < saturationGrid
      ? capacityGrid
      : saturationGrid;
    std::fprintf
    (
        stderr,
        "Launch geometry: B1cell=%d B1face=%d (ToolB1 pending) "
        "B2=%d B3=%d SM=%d "
        "lightBlocksPerSM=%d heavyBlocksPerSM=%d\n",
        s->fixedCellBlockThreads,
        s->fixedFaceBlockThreads,
        s->particleBlockThreads,
        s->reductionBlockThreads,
        s->multiprocessorCount,
        s->lightResidentBlocksPerSm,
        s->heavyResidentBlocksPerSm
    );
    return syncDeviceState(s, "sync occupancy-derived launch geometry");
}

__global__ void updateDynamicHeavyPolicyKernel
(
    DeviceState* sp,
    const int splitPreDirectoryActive
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    long long population = s.cellParticleOffset[s.nCells];
    if (splitPreDirectoryActive != 0)
    {
        population += *s.preBaseParticleCountDevice;
    }
    const long long concurrency =
        static_cast<long long>(s.reductionBlockThreads)
       *static_cast<long long>(s.multiprocessorCount)
       *static_cast<long long>(s.lightResidentBlocksPerSm);
    if (population < 0 || population > s.particleCapacity || concurrency <= 0)
    {
        asm("trap;");
    }
    const long long total = population > 0 ? population : 1;
    long long shares = (total + concurrency - 1)/concurrency;
    shares = shares > 0 ? shares : 1;
    const long long threshold =
        static_cast<long long>(s.reductionBlockThreads)*shares;
    if (threshold <= 0 || threshold > 2147483647LL)
    {
        asm("trap;");
    }
    s.dynamicHeavyThreshold = static_cast<int>(threshold);
    s.csrHeavyTileParticles = static_cast<int>(threshold);
}

int updateDynamicHeavyPolicy(DeviceState* s)
{
    if (s->csrHeavyReductionMode == 0)
    {
        return 0;
    }
    updateDynamicHeavyPolicyKernel<<<1, 1>>>
    (
        s->deviceState,
        s->splitPreDirectoryActive
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("update dynamic heavy policy launch", err);
        return 1;
    }
    return 0;
}

__global__ void maximumDirectoryOccupancyKernel
(
    DeviceState* sp,
    const int splitPreDirectoryActive,
    int* maximumOccupancy
)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        int count = s.cellParticleOffset[c + 1] - s.cellParticleOffset[c];
        if (splitPreDirectoryActive != 0)
        {
            count += s.preBaseCellOffset[c + 1] - s.preBaseCellOffset[c];
        }
        atomicMax(maximumOccupancy, count);
    }
}

__global__ void publishHeavyReductionDecisionKernel
(
    DeviceState* sp,
    const int active
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        sp->csrHeavyReductionActive = active;
        sp->csrHeavyReductionEnabled = active;
    }
}

int runToolB3(DeviceState* s, const int block)
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
    if (updateDynamicHeavyPolicy(s) != 0)
    {
        return 1;
    }
    cudaError_t err = cudaMemset(s->csrMaximumOccupancy, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 clear maximum occupancy", err);
        return 1;
    }
    const int grid = (s->nCells + block - 1)/block;
    maximumDirectoryOccupancyKernel<<<grid, block>>>
    (
        s->deviceState,
        s->splitPreDirectoryActive,
        s->csrMaximumOccupancy
    );
    err = cudaGetLastError();
    int maximumOccupancy = 0;
    int threshold = 0;
    if (err == cudaSuccess)
    {
        err = cudaMemcpy
        (
            &maximumOccupancy,
            s->csrMaximumOccupancy,
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
              + offsetof(DeviceState, dynamicHeavyThreshold),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    }
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 maximum occupancy", err);
        return 1;
    }
    const int active = maximumOccupancy > threshold ? 1 : 0;
    s->dynamicHeavyThreshold = threshold;
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
    if (active != 0)
    {
        return prepareCsrSegmentedReductionTasks
        (
            s,
            block,
            s->splitPreDirectoryActive != 0
        );
    }
    err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
    if (err == cudaSuccess)
    {
        err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
    }
    if (err != cudaSuccess)
    {
        setLastError("ToolB3 clear inactive segmented schedule", err);
        return 1;
    }
    return 0;
}

int binParticlesByCell(DeviceState* s, const int block)
{
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
    if (updateDynamicHeavyPolicy(s) != 0)
    {
        return 1;
    }
    return prepareCsrHeavyReductionTasks(s, block);
}

int buildSplitPreDirectory(DeviceState* s, const int block)
{
    if
    (
        s->csrCellLocalPathEnabled == 0
     || s->preBaseDirectoryReady == 0
    )
    {
        s->splitPreDirectoryActive = 0;
        return binParticlesByCell(s, block);
    }

    s->splitPreDirectoryActive = 1;

    const int cellGrid = (s->nCells + 1 + block - 1)/block;
    clearSplitPreInjectionBinsKernel<<<cellGrid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clear split-Dpre injection bins launch", err);
        return 1;
    }

                                                                             
                                                                      
    if (s->nBoundarySources == 0)
    {
        if (updateDynamicHeavyPolicy(s) != 0)
        {
            return 1;
        }
        return prepareSplitCsrHeavyReductionTasks(s, block);
    }

    if (s->csrWarpAggregatedBinning != 0)
    {
        countSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        countSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("count split-Dpre injection particles launch", err);
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
        setLastError("split-Dpre injection count exclusive scan", err);
        return 1;
    }

    initialiseSplitPreInjectionWritesKernel<<<cellGrid, block>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("initialise split-Dpre injection writes launch", err);
        return 1;
    }
    if (s->csrWarpAggregatedBinning != 0)
    {
        scatterSplitPreInjectionParticlesKernel<true>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    else
    {
        scatterSplitPreInjectionParticlesKernel<false>
            <<<s->particleWorkGrid, block>>>(s->deviceState);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("scatter split-Dpre injection particles launch", err);
        return 1;
    }

                                                                         
                                                                             
                                                     
    if (updateDynamicHeavyPolicy(s) != 0)
    {
        return 1;
    }
    return prepareSplitCsrHeavyReductionTasks(s, block);
}

int buildPostTransportDirectory(DeviceState* s, const int block)
{
    s->splitPreDirectoryActive = 0;
    return binParticlesByCell(s, block);
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

            accumulateParticleMomentsSegmentedKernel
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
            if (launchCsrHeavyMomentReduction(s, block) != 0)
            {
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

    if (s.solveParticleTemperature != 0)
    {
        const double specificEnthalpy =
            finiteOr(totalHeat/(rhoP + OfSmall), -1.0);
        s.Tp[c] = clampRange
        (
            finiteOr
            (
                particleTemperatureFromSpecificEnthalpyDevice
                (
                    specificEnthalpy
                ),
                s.TpMin
            ),
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
    for (int i = 0; i <= ProbeStageCount; ++i)
    {
        if (developmentProbe.events[i] != nullptr)
        {
            cudaEventDestroy(developmentProbe.events[i]);
            developmentProbe.events[i] = nullptr;
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
        "particle_path,particle_count,particle_capacity,particle_utilisation,"
        "occupancy_sum,occupancy_nonempty,occupancy_min,occupancy_mean,"
        "occupancy_stddev,occupancy_cv,occupancy_p50,occupancy_p95,"
        "occupancy_p99,occupancy_max,occupancy_matches_count,bad_cells,"
        "bad_particles,bad_field_mask,first_bad_cell,first_bad_particle,"
        "total_ms,gas_flux_ms,eulerian_coupling_ms,injection_ms,bin_pre_ms,"
        "pressure_pre_ms,collision_pool_ms,relax_ms,track_ms,bin_post_ms,"
        "theta_pool_ms,moments_ms,pressure_post_ms,compaction_ms,boundary_ms\n";

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

    std::fprintf(file, "1,");
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
        ",%d,%d,%d,%d,%.17g,%lld,%d,%d,%.17g,%.17g,%.17g,"
        "%d,%d,%d,%d,%d,%llu,%llu,0x%016llx,%d,%d,%.9g",
        sample.nCells,
        sample.particlePath,
        sample.particleCount,
        sample.particleCapacity,
        sample.particleUtilisation,
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

    for (int i = 0; i <= ProbeStageCount; ++i)
    {
        const cudaError_t err = cudaEventCreate(&developmentProbe.events[i]);
        if (err != cudaSuccess)
        {
            setLastError("cudaEventCreate UGKP development probe", err);
            shutdownDevelopmentProbe();
            return 1;
        }
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
    cudaError_t err = cudaEventSynchronize
    (
        developmentProbe.events[ProbeStageCount]
    );
    if (err != cudaSuccess)
    {
        setLastError("UGKP development probe final event synchronization", err);
        return 1;
    }

    err = cudaEventElapsedTime
    (
        &sample.totalMs,
        developmentProbe.events[0],
        developmentProbe.events[ProbeStageCount]
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaEventElapsedTime UGKP development probe total", err);
        return 1;
    }
    for (int i = 0; i < ProbeStageCount; ++i)
    {
        err = cudaEventElapsedTime
        (
            &sample.stageMs[i],
            developmentProbe.events[i],
            developmentProbe.events[i + 1]
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventElapsedTime UGKP development probe stage", err);
            return 1;
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
    for (const int count : developmentProbe.occupancy)
    {
        sample.occupancySum += static_cast<long long>(count);
        sumSquares += static_cast<long double>(count)*count;
        occupancyMin = std::min(occupancyMin, count);
        occupancyMax = std::max(occupancyMax, count);
        sample.occupancyNonEmpty += count > 0 ? 1 : 0;
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
            const cudaError_t err = cudaEventRecord(developmentProbe.events[0], 0);
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
    }

    int leave(const DevelopmentProbeStage stage)
    {
        if (!sampled_)
        {
            return 0;
        }
        const cudaError_t err = cudaEventRecord
        (
            developmentProbe.events[static_cast<int>(stage) + 1],
            0
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventRecord UGKP development probe stage", err);
            failed_ = true;
            return 1;
        }
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
        DevelopmentProbeSample sample;
        sample.step = step_;
        sample.simulationTime = simulationTime_;
        sample.dt = dt_;
        sample.nCells = state_->nCells;
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
            setLastError("enforcePeriodicGasFluxAntisymmetryKernel launch", err);
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
    applyGasFluxPositivityScaleKernel<<<faceGrid, faceBlock>>>
    (
        s->deviceState,
        dt
    );
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
                setLastError("enforcePeriodicSstFluxAntisymmetryKernel launch", err);
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

int advanceGasFluxStage(DeviceState* s, const double dt, const double simulationTime)
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
            <<<preparationFaceGrid, preparationFaceBlock>>>(s->deviceState, simulationTime);
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

int finaliseGasBoundaryStage(DeviceState* s, const double dt, const double simulationTime)
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
    double gasMu,
    double gasPr,
    int dragModelId,
    int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX,
    double gravityY,
    double gravityZ,
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
     || !std::isfinite(gammaGas) || gammaGas <= 1.0 || gammaGas > 5.0/3.0
     || !std::isfinite(Rgas) || Rgas <= 0.0
     || !std::isfinite(gasMu) || gasMu < 0.0
     || !std::isfinite(gasPr) || gasPr <= 0.0
     || dragModelId < 0 || dragModelId > 2
     || particleGasHeatTransferModelId < 0
     || particleGasHeatTransferModelId > 1
     || !std::isfinite(dragResidualRe) || dragResidualRe <= 0.0
     || !std::isfinite(gravityX)
     || !std::isfinite(gravityY)
     || !std::isfinite(gravityZ)
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
     || csrHeavyReductionMode < 0 || csrHeavyReductionMode > 2
     || csrHeavyAutoInterval < 1
     || (particleBlockThreads != 32 && particleBlockThreads != 64
      && particleBlockThreads != 128 && particleBlockThreads != 256)
     || (reductionBlockThreads != 32 && reductionBlockThreads != 64
      && reductionBlockThreads != 128 && reductionBlockThreads != 256)
     || (csrWarpAggregatedBinning != 0 && csrWarpAggregatedBinning != 1)
     || (jammingPressureEnabled != 0 && jammingPressureEnabled != 1)
     || packingProjectionIterations < 1
     || (!csrCellLocalPathEnabled
      && (csrHeavyReductionMode != 0 || csrWarpAggregatedBinning))
    )
    {
        setLastErrorText("invalid GPU resident strict configuration");
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
    s->gasMu = gasMu;
    s->gasPr = gasPr;
    s->gasPrClamped = gasPr < 1.0e-12 ? 1.0e-12 : gasPr;
    s->gasPrOneThird = std::pow(s->gasPrClamped, 1.0/3.0);
    s->dragModelId = dragModelId;
    s->particleGasHeatTransferModelId = particleGasHeatTransferModelId;
    s->dragResidualRe = dragResidualRe;
    s->gravityX = gravityX;
    s->gravityY = gravityY;
    s->gravityZ = gravityZ;
    s->gravityEnabled =
        gravityX != 0.0 || gravityY != 0.0 || gravityZ != 0.0 ? 1 : 0;
    s->gasFluxScheme = gasFluxScheme;
    s->gasReconstruction = gasReconstruction;
    s->gasLimiter = gasLimiter;
    s->gasTimeIntegrator = gasTimeIntegrator;
    s->gasRobustFallback = gasRobustFallback;
    s->turbulenceModel = turbulenceModel;
    s->hostGasFluxScheme = gasFluxScheme;
    s->hostGasTimeIntegrator = gasTimeIntegrator;
    s->hostTurbulenceModel = turbulenceModel;
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
    s->csrWarpAggregatedBinning = csrWarpAggregatedBinning;
    s->particleBlockThreads = particleBlockThreads;
    s->reductionBlockThreads = reductionBlockThreads;
    s->dynamicHeavyThreshold = s->reductionBlockThreads;
    s->csrHeavyTileParticles = s->reductionBlockThreads;
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
    const char* const wallContactDiagnostics =
        std::getenv("UGKP_WALL_CONTACT_DIAGNOSTICS");
    s->wallContactAreaDiagnosticsEnabled =
        wallContactDiagnostics != nullptr
     && wallContactDiagnostics[0] == '1'
     && wallContactDiagnostics[1] == '\0'
      ? 1 : 0;
    if (s->wallContactAreaDiagnosticsEnabled != 0)
    {
        std::fprintf
        (
            stderr,
            "CHT wall-contact diagnostics enabled; production contact-area "
            "kernel replaced by fail-fast diagnostic kernel.\n"
        );
    }

    if (allocateFields(s) != 0)
    {
        return 1;
    }
    if (configureLaunchOccupancy(s) != 0)
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
        faceOwner == nullptr
     || faceNeighbour == nullptr
     || facePeriodicPair == nullptr
     || facePeriodicDx == nullptr
     || facePeriodicDy == nullptr
     || facePeriodicDz == nullptr
     || cellFaceId == nullptr
     || cellFaceNeighbor == nullptr
     || cellFaceKind == nullptr
    )
    {
        setLastErrorText("null topology array in mesh upload");
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
        if (!std::isfinite(pressureTimes[i]) || !std::isfinite(pressureValues[i]) || pressureTimes[i] < 0.0 || pressureValues[i] <= 0.0 || (i > 0 && pressureTimes[i] <= pressureTimes[i - 1]))
        {
            setLastErrorText("invalid scheduled inlet pressure table");
            return 1;
        }
    }
    bool futureParticleInflow = false;
    for (int i = 0; i < nVolumeFractionRows; ++i)
    {
        if (!std::isfinite(volumeFractionTimes[i]) || !std::isfinite(volumeFractionValues[i]) || volumeFractionTimes[i] < 0.0 || volumeFractionValues[i] < 0.0 || volumeFractionValues[i] >= 1.0 || (i > 0 && volumeFractionTimes[i] <= volumeFractionTimes[i - 1]))
        {
            setLastErrorText("invalid scheduled inlet volume-fraction table");
            return 1;
        }
        futureParticleInflow = futureParticleInflow || volumeFractionValues[i] > 0.0;
    }

    release(s->scheduledInletFaceMask);
    release(s->pressureScheduleTimes);
    release(s->pressureScheduleValues);
    release(s->volumeFractionScheduleTimes);
    release(s->volumeFractionScheduleValues);
    int rc = 0;
    rc |= allocate(s->scheduledInletFaceMask, static_cast<size_t>(s->nFaces), "cudaMalloc strict scheduled inlet face mask");
    rc |= allocate(s->pressureScheduleTimes, static_cast<size_t>(nPressureRows), "cudaMalloc strict pressure schedule times");
    rc |= allocate(s->pressureScheduleValues, static_cast<size_t>(nPressureRows), "cudaMalloc strict pressure schedule values");
    rc |= allocate(s->volumeFractionScheduleTimes, static_cast<size_t>(nVolumeFractionRows), "cudaMalloc strict volume-fraction schedule times");
    rc |= allocate(s->volumeFractionScheduleValues, static_cast<size_t>(nVolumeFractionRows), "cudaMalloc strict volume-fraction schedule values");
    if (rc != 0)
    {
        return 1;
    }
    rc |= copyToDevice(s->scheduledInletFaceMask, faceMask.data(), static_cast<size_t>(s->nFaces), "cudaMemcpy strict scheduled inlet face mask");
    rc |= copyToDevice(s->pressureScheduleTimes, pressureTimes, static_cast<size_t>(nPressureRows), "cudaMemcpy strict pressure schedule times");
    rc |= copyToDevice(s->pressureScheduleValues, pressureValues, static_cast<size_t>(nPressureRows), "cudaMemcpy strict pressure schedule values");
    rc |= copyToDevice(s->volumeFractionScheduleTimes, volumeFractionTimes, static_cast<size_t>(nVolumeFractionRows), "cudaMemcpy strict volume-fraction schedule times");
    rc |= copyToDevice(s->volumeFractionScheduleValues, volumeFractionValues, static_cast<size_t>(nVolumeFractionRows), "cudaMemcpy strict volume-fraction schedule values");
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

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch
(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* temperatures
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "gas boundary temperature patch upload") != 0)
    {
        return 1;
    }
    if (patchFaceCount < 0)
    {
        setLastErrorText("negative gas boundary temperature patch face count");
        return 1;
    }
    if (patchFaceCount == 0)
    {
        return 0;
    }
    if
    (
        patchStartFace < s->nInternalFaces
     || patchStartFace >= s->nFaces
     || patchFaceCount > s->nFaces - patchStartFace
     || temperatures == nullptr
    )
    {
        setLastErrorText("invalid gas boundary temperature patch input");
        return 1;
    }
    for (int i = 0; i < patchFaceCount; ++i)
    {
        if (!std::isfinite(temperatures[i]) || temperatures[i] < s->TgasMin)
        {
            setLastErrorText("invalid gas boundary temperature patch value");
            return 1;
        }
    }
    return copyToDevice
    (
        s->gasBoundaryT + patchStartFace,
        temperatures,
        static_cast<size_t>(patchFaceCount),
        "cudaMemcpy strict gas boundary temperature patch"
    );
}

extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "particle-wall effusivity patch upload") != 0)
    {
        return 1;
    }
    if (patchFaceCount < 0)
    {
        setLastErrorText("negative particle-wall effusivity patch face count");
        return 1;
    }
    if (patchFaceCount == 0)
    {
        return 0;
    }
    if
    (
        patchStartFace < s->nInternalFaces
     || patchStartFace >= s->nFaces
     || patchFaceCount > s->nFaces - patchStartFace
     || wallEffusivity == nullptr
     || s->particleWallHeatTransferEnabled == 0
     || s->particleWallEffusivityByFace == nullptr
    )
    {
        setLastErrorText("invalid particle-wall effusivity patch input");
        return 1;
    }
    for (int i = 0; i < patchFaceCount; ++i)
    {
        if (!std::isfinite(wallEffusivity[i]) || wallEffusivity[i] <= 0.0)
        {
            setLastErrorText("particle-wall effusivity must be finite and positive");
            return 1;
        }
    }
    return copyToDevice
    (
        s->particleWallEffusivityByFace + patchStartFace,
        wallEffusivity,
        static_cast<size_t>(patchFaceCount),
        "cudaMemcpy CHT particle-wall effusivity patch"
    );
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
        updateLegacyGasBoundaryMirrorKernel<<<allFaceGrid, faceBlock>>>(s->deviceState, scheduleTime);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("updateLegacyGasBoundaryMirrorKernel for gas Courant", err);
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
#define UGKP_DEV_PROBE_LEAVE(STAGE) \
    do \
    { \
        if (developmentAdvanceProbe.leave(STAGE) != 0) \
        { \
            return 1; \
        } \
    } while (false)
#else
#define UGKP_DEV_PROBE_ENTER(STAGE) ((void)0)
#define UGKP_DEV_PROBE_LEAVE(STAGE) ((void)0)
#endif

    UGKP_DEV_PROBE_ENTER(ProbeGasFlux);
    if
    (
        s->particlesMayBePresent
     && s->particleStuckModelConfigured != 0
     && s->particleWorkGrid > 0
     && prepareParticleWallContactAreaScale(s, block) != 0
    )
    {
        return 1;
    }
    if (advanceGasFluxStage(s, dt, simulationTime) != 0)
    {
        return 1;
    }
    if (s->gravityEnabled != 0)
    {
        const int gasBlock = s->fixedCellBlockThreads;
        const int gasGrid = (s->nCells + gasBlock - 1)/gasBlock;
        applyGasGravityKernel<<<gasGrid, gasBlock>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applyGasGravityKernel launch", err);
            return 1;
        }
        recoverGasPrimitivesKernel<<<gasGrid, gasBlock>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "recoverGasPrimitivesKernel after gravity launch",
                err
            );
            return 1;
        }
    }

    UGKP_DEV_PROBE_LEAVE(ProbeGasFlux);

    const bool skipParticlePath = !s->particlesMayBePresent;
    const bool dragActive = gasDragModelActive(s->dragModelId);

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

    if (s->dragModelId == 2)
    {
        applyEulerianGasSolidDragKernelStatic
            <<<grid, block>>>
            (
                s->deviceState,
                dt,
                ugkwpGpuDrag::GidaspowErgunWenYuDrag{s->dragResidualRe}
            );
    }
    else
    {
        applyEulerianGasSolidDragKernelStatic
            <<<grid, block>>>
            (
                s->deviceState,
                dt,
                ugkwpGpuDrag::SchillerNaumannDrag{}
            );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyEulerianGasSolidDragKernel launch", err);
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
    if (s->nBoundarySources > 0)
    {
        const int sourceGrid =
            (s->nBoundarySources + s->particleBlockThreads - 1)
           /s->particleBlockThreads;
        injectBoundaryParticlesKernel
            <<<sourceGrid, s->particleBlockThreads>>>(s->deviceState, dt, simulationTime);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("injectBoundaryParticlesKernel launch", err);
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeInjection);


    UGKP_DEV_PROBE_ENTER(ProbeBinPre);
    const int particleGrid = s->particleWorkGrid;
    if
    (
        s->csrCellLocalPathEnabled != 0
     && buildSplitPreDirectory(s, block) != 0
    )
    {
        return 1;
    }
    if (s->csrCellLocalPathEnabled != 0 && runToolB3(s, block) != 0)
    {
        return 1;
    }
    UGKP_DEV_PROBE_LEAVE(ProbeBinPre);

    UGKP_DEV_PROBE_ENTER(ProbePressurePre);
    if (applyCollisionalPressureKick(s, 0.5*dt, block) != 0)
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
                const int heavyPoolStatus =
                    launchCsrHeavyPoolReduction(s, dt, true, block);
                if (heavyPoolStatus != 0)
                {
                    return 1;
                }
            }
            else if (s->splitPreDirectoryActive != 0)
            {
                const size_t splitSharedBytes =
                    8u*static_cast<size_t>((block + 31)/32)*sizeof(double);
                accumulatePoissonPoolSplitSegmentKernel<true, false>
                    <<<s->nCells, block, splitSharedBytes>>>
                    (s->deviceState, dt);
                if (s->nBoundarySources > 0)
                {
                    accumulatePoissonPoolSplitSegmentKernel<false, true>
                        <<<s->nCells, block, splitSharedBytes>>>
                        (s->deviceState, dt);
                }
            }
            else
            {
                accumulatePoissonPoolParticlesByCellKernel
                    <<<s->nCells, block, poolReduceSharedBytes>>>
                    (s->deviceState, dt);
            }
            if (s->csrHeavyReductionEnabled == 0)
            {
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    setLastError
                    (
                        "accumulate split/full Poisson pool by cell launch",
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

        err = cudaMemset(s->compactCountDevice, 0, sizeof(int));
        if (err != cudaSuccess)
        {
            setLastError("clear selected stuck collision count", err);
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

        correctPoissonThermalizedMobileParticlesKernel<<<particleGrid, block>>>
        (
            s->deviceState,
            0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "correctPoissonThermalizedMobileParticlesKernel launch",
                err
            );
            return 1;
        }

        correctPoissonThermalizedStuckParticlesKernel<<<particleGrid, block>>>
        (
            s->deviceState,
            0
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "correctPoissonThermalizedStuckParticlesKernel launch",
                err
            );
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeCollisionPool);

    UGKP_DEV_PROBE_ENTER(ProbeRelax);
    if (particleGrid > 0)
    {
        if (s->coldWallSolidificationEnabled != 0)
        {
            relaxColdWall1DParticlesToResidentGasKernel
                <<<particleGrid, block>>>(s->deviceState, dt);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "relaxColdWall1DParticlesToResidentGasKernel launch",
                    err
                );
                return 1;
            }
        }
        if (s->dragModelId == 2)
        {
            relaxMobileParticlesToResidentGasKernelStatic
                <<<particleGrid, block>>>
                (
                    s->deviceState,
                    dt,
                    ugkwpGpuDrag::GidaspowErgunWenYuDrag{s->dragResidualRe}
                );
        }
        else
        {
            relaxMobileParticlesToResidentGasKernelStatic
                <<<particleGrid, block>>>
                (
                    s->deviceState,
                    dt,
                    ugkwpGpuDrag::SchillerNaumannDrag{}
                );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("relaxMobileParticlesToResidentGasKernel launch", err);
            return 1;
        }
        if (s->dragModelId == 2)
        {
            relaxWallBoundParticlesToResidentGasKernelStatic
                <<<particleGrid, block>>>
                (
                    s->deviceState,
                    dt,
                    ugkwpGpuDrag::GidaspowErgunWenYuDrag{s->dragResidualRe}
                );
        }
        else
        {
            relaxWallBoundParticlesToResidentGasKernelStatic
                <<<particleGrid, block>>>
                (
                    s->deviceState,
                    dt,
                    ugkwpGpuDrag::SchillerNaumannDrag{}
                );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("relaxWallBoundParticlesToResidentGasKernel launch", err);
            return 1;
        }
        if (s->coldWall2DEnabled != 0)
        {
            relaxColdWall2DParticlesToResidentGasKernel
                <<<particleGrid, block>>>(s->deviceState, dt);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError
                (
                    "relaxColdWall2DParticlesToResidentGasKernel launch",
                    err
                );
                return 1;
            }
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
    if (particleGrid > 0 && s->csrCellLocalPathEnabled != 0)
    {
        if (buildPostTransportDirectory(s, block) != 0)
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbeBinPost);

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
                accumulateParticleMomentsSegmentedKernel
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
        if (applyCollisionalPressureKick(s, 0.5*dt, block) != 0)
        {
            return 1;
        }
    }
    UGKP_DEV_PROBE_LEAVE(ProbePressurePost);

    UGKP_DEV_PROBE_ENTER(ProbeCompaction);
    err = cudaMemset(s->wallBoundParticleCountDevice, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset wall-bound particle directory count", err);
        return 1;
    }
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
            default:
                gatherCellLocalParticlesKernel<256>
                    <<<s->nCells, 256>>>(s->deviceState);
                break;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("gatherCellLocalParticlesKernel launch", err);
            return 1;
        }

        commitCellLocalParticleBuffersKernel<<<1, 1>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("commitCellLocalParticleBuffersKernel launch", err);
            return 1;
        }
        err = cudaMemcpy
        (
            s->preBaseCellOffset,
            s->compactCellOffset,
            (static_cast<size_t>(s->nCells) + 1u)*sizeof(int),
            cudaMemcpyDeviceToDevice
        );
        if (err == cudaSuccess)
        {
            err = cudaMemcpy
            (
                s->preBaseParticleCountDevice,
                s->compactCellOffset + s->nCells,
                sizeof(int),
                cudaMemcpyDeviceToDevice
            );
        }
        if (err != cudaSuccess)
        {
            setLastError("capture compaction-seeded split-Dpre base", err);
            return 1;
        }
        s->preBaseDirectoryReady = 1;
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
        s->preBaseDirectoryReady = 0;
    }
    swapParticleBufferPointersHost(s);
    s->splitPreDirectoryActive = 0;
    UGKP_DEV_PROBE_LEAVE(ProbeCompaction);
    }
    else
    {
                                                                              
                                                                            
        UGKP_DEV_PROBE_ENTER(ProbeEulerianCoupling);
        UGKP_DEV_PROBE_LEAVE(ProbeEulerianCoupling);
        UGKP_DEV_PROBE_ENTER(ProbeInjection);
        UGKP_DEV_PROBE_LEAVE(ProbeInjection);
        UGKP_DEV_PROBE_ENTER(ProbeBinPre);
        UGKP_DEV_PROBE_LEAVE(ProbeBinPre);
        UGKP_DEV_PROBE_ENTER(ProbePressurePre);
        UGKP_DEV_PROBE_LEAVE(ProbePressurePre);
        UGKP_DEV_PROBE_ENTER(ProbeCollisionPool);
        UGKP_DEV_PROBE_LEAVE(ProbeCollisionPool);
        UGKP_DEV_PROBE_ENTER(ProbeRelax);
        UGKP_DEV_PROBE_LEAVE(ProbeRelax);
        UGKP_DEV_PROBE_ENTER(ProbeTrack);
        UGKP_DEV_PROBE_LEAVE(ProbeTrack);
        UGKP_DEV_PROBE_ENTER(ProbeBinPost);
        UGKP_DEV_PROBE_LEAVE(ProbeBinPost);
        UGKP_DEV_PROBE_ENTER(ProbeMoments);
        UGKP_DEV_PROBE_LEAVE(ProbeMoments);
        UGKP_DEV_PROBE_ENTER(ProbePressurePost);
        UGKP_DEV_PROBE_LEAVE(ProbePressurePost);
        UGKP_DEV_PROBE_ENTER(ProbeCompaction);
        UGKP_DEV_PROBE_LEAVE(ProbeCompaction);
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

    if (!std::isfinite(simulationTime) || simulationTime < 0.0 || advanceGasFluxStage(s, dt, simulationTime) != 0)
    {
        return 1;
    }
    if (s->gravityEnabled != 0)
    {
        const int block = s->fixedCellBlockThreads;
        const int grid = (s->nCells + block - 1)/block;
        applyGasGravityKernel<<<grid, block>>>(s->deviceState, dt);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applyGasGravityKernel gas-only launch", err);
            return 1;
        }
        recoverGasPrimitivesKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "recoverGasPrimitivesKernel gas-only after gravity",
                err
            );
            return 1;
        }
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

extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature
(
    void* handle,
    int nCells,
    const Foam::gpuThermal::RadiationAffineTemperatureUpdate* updateByCell
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "apply particle-radiation affine temperature") != 0
     || nCells != s->nCells
     || nCells <= 0
     || updateByCell == nullptr
    )
    {
        setLastErrorText("invalid particle-radiation affine update");
        return 1;
    }
    if (!s->particlesMayBePresent)
    {
        return 0;
    }
    if
    (
        (s->particleRadiationAffineUpdate == nullptr)
     != (s->particleRadiationValidationError == nullptr)
    )
    {
        setLastErrorText("inconsistent particle-radiation lazy buffer state");
        return 1;
    }
    if (s->particleRadiationAffineUpdate == nullptr)
    {
        if
        (
            allocate
            (
                s->particleRadiationAffineUpdate,
                static_cast<size_t>(nCells),
                "cudaMalloc particle-radiation affine updates"
            ) != 0
         || allocate
            (
                s->particleRadiationValidationError,
                1,
                "cudaMalloc particle-radiation validation error"
            ) != 0
        )
        {
            release(s->particleRadiationAffineUpdate);
            release(s->particleRadiationValidationError);
            return 1;
        }
    }
    if
    (
        copyToDevice
        (
            s->particleRadiationAffineUpdate,
            updateByCell,
            static_cast<size_t>(nCells),
            "cudaMemcpy particle-radiation affine updates"
        ) != 0
    )
    {
        return 1;
    }
    cudaError_t err = cudaMemset
    (
        s->particleRadiationValidationError,
        0,
        sizeof(ParticleRadiationValidationError)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset particle-radiation validation error", err);
        return 1;
    }
    const int block = s->particleBlockThreads;
    validateParticleRadiationAffineTemperatureKernel
        <<<s->particleWorkGrid, block>>>
    (
        s->deviceState,
        s->particleRadiationAffineUpdate,
        s->particleRadiationValidationError
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "validateParticleRadiationAffineTemperatureKernel launch", err
        );
        return 1;
    }
    ParticleRadiationValidationError hostError;
    if
    (
        copyToHost
        (
            &hostError,
            s->particleRadiationValidationError,
            1,
            "cudaMemcpy particle-radiation validation error"
        ) != 0
    )
    {
        return 1;
    }
    if (hostError.code != particleRadiationValid)
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "particle-radiation validation failed: code=%d, particle=%d, "
            "originalId=%llu, cell=%d, oldT=%.17g, scale=%.17g, "
            "offset=%.17g, proposedT=%.17g",
            hostError.code,
            hostError.particleArrayIndex,
            static_cast<unsigned long long>(hostError.particleOriginalId),
            hostError.cellId,
            hostError.oldTemperatureK,
            hostError.scale,
            hostError.offset,
            hostError.proposedTemperatureK
        );
        return 1;
    }
    applyParticleRadiationAffineTemperatureKernel
        <<<s->particleWorkGrid, block>>>
    (
        s->deviceState,
        s->particleRadiationAffineUpdate
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "applyParticleRadiationAffineTemperatureKernel launch", err
        );
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "applyParticleRadiationAffineTemperatureKernel completion", err
        );
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums
(
    void* handle,
    int nCells,
    double* particleMassKg,
    double* particleTemperatureMassKgK,
    double* particleDiameterMassKgM
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "mobile particle radiation material download") != 0
    )
    {
        return 1;
    }
    if
    (
        nCells != s->nCells
     || particleMassKg == nullptr
     || particleTemperatureMassKgK == nullptr
     || particleDiameterMassKgM == nullptr
    )
    {
        setLastErrorText("invalid mobile particle radiation download arrays");
        return 1;
    }
    const int block = s->reductionBlockThreads;
    const int grid = (s->nCells + block - 1)/block;
    clearMobileParticleRadiationSumsKernel<<<grid, block>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("clear mobile particle radiation sums launch", err);
        return 1;
    }
    if (s->particleCapacity > 0 && s->particleWorkGrid > 0)
    {
        if (s->csrCellLocalPathEnabled != 0)
        {
            if (s->preBaseDirectoryReady == 0)
            {
                setLastErrorText
                (
                    "mobile radiation snapshot requires packed base offsets"
                );
                return 1;
            }
            const int warpCount = (block + 31)/32;
            const size_t sharedBytes =
                3u*static_cast<size_t>(warpCount)*sizeof(double);
            accumulatePackedMobileParticleRadiationSumsKernel
                <<<s->nCells, block, sharedBytes>>>(s->deviceState);
        }
        else
        {
            accumulateMobileParticleRadiationSumsAtomicKernel
                <<<s->particleWorkGrid, block>>>(s->deviceState);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("accumulate mobile particle radiation sums launch", err);
            return 1;
        }
    }
    if
    (
        copyToHost
        (
            particleMassKg,
            s->radiationMobileMass,
            static_cast<size_t>(s->nCells),
            "cudaMemcpy mobile radiation mass download"
        ) != 0
     || copyToHost
        (
            particleTemperatureMassKgK,
            s->radiationMobileTemperatureMass,
            static_cast<size_t>(s->nCells),
            "cudaMemcpy mobile radiation temperature mass download"
        ) != 0
     || copyToHost
        (
            particleDiameterMassKgM,
            s->radiationMobileDiameterMass,
            static_cast<size_t>(s->nCells),
            "cudaMemcpy mobile radiation diameter mass download"
        ) != 0
    )
    {
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea
(
    void* handle,
    int nFaces,
    double* occupiedAreaM2
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "particle-wall occupied area download") != 0)
    {
        return 1;
    }
    if (nFaces != s->nFaces || occupiedAreaM2 == nullptr)
    {
        setLastErrorText("invalid particle-wall occupied area download array");
        return 1;
    }
    if (s->particleStuckModelConfigured == 0 || s->particleWorkGrid <= 0)
    {
        std::fill(occupiedAreaM2, occupiedAreaM2 + nFaces, 0.0);
        return 0;
    }
    const int block = s->reductionBlockThreads;
    if (prepareParticleWallContactAreaScale(s, block) != 0)
    {
        return 1;
    }
    std::vector<double> areaScale(static_cast<size_t>(nFaces), 1.0);
    if
    (
        copyToHost
        (
            occupiedAreaM2,
            s->particleWallRepresentedContactArea,
            static_cast<size_t>(nFaces),
            "cudaMemcpy particle-wall represented area download"
        ) != 0
     || copyToHost
        (
            areaScale.data(),
            s->particleWallContactAreaScale,
            static_cast<size_t>(nFaces),
            "cudaMemcpy particle-wall area scale download"
        ) != 0
    )
    {
        return 1;
    }
    for (int faceI = 0; faceI < nFaces; ++faceI)
    {
        const double represented = occupiedAreaM2[faceI];
        const double scale = areaScale[static_cast<size_t>(faceI)];
        if
        (
            !std::isfinite(represented)
         || !std::isfinite(scale)
         || represented < 0.0
         || !(scale > 0.0)
         || scale > 1.0
        )
        {
            setLastErrorText("invalid particle-wall occupied area state");
            return 1;
        }
        occupiedAreaM2[faceI] = represented*scale;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked
(
    void* handle
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "post-radiation particle enthalpy refresh") != 0)
    {
        return 1;
    }
    const int block = s->reductionBlockThreads;
    const int grid = (s->nCells + block - 1)/block;
    cudaError_t err = cudaSuccess;
    if (s->csrCellLocalPathEnabled != 0)
    {
        if (s->preBaseDirectoryReady == 0)
        {
            setLastErrorText
            (
                "post-radiation enthalpy refresh requires packed base offsets"
            );
            return 1;
        }
        const int warpCount = (block + 31)/32;
        const size_t sharedBytes =
            static_cast<size_t>(warpCount)*sizeof(double);
        refreshPackedParticleEnthalpyKernel
            <<<s->nCells, block, sharedBytes>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("refresh packed particle enthalpy launch", err);
            return 1;
        }
    }
    else
    {
        clearParticleEnthalpyMomentKernel<<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("clear particle enthalpy moment launch", err);
            return 1;
        }
        if (s->particleCapacity > 0 && s->particleWorkGrid > 0)
        {
            accumulateParticleEnthalpyMomentAtomicKernel
                <<<s->particleWorkGrid, block>>>(s->deviceState);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("accumulate particle enthalpy moment launch", err);
                return 1;
            }
        }
        recoverParticleEnthalpyMomentAtomicKernel
            <<<grid, block>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recover particle enthalpy moment launch", err);
            return 1;
        }
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError("post-radiation particle enthalpy refresh completion", err);
        return 1;
    }
    return 0;
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

extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger
(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "gas-wall energy ledger configuration") != 0)
    {
        return 1;
    }
    if
    (
        nEnabledFaces < 0
     || nEnabledFaces > s->nFaces
     || (nEnabledFaces > 0 && enabledFaceIds == nullptr)
    )
    {
        setLastErrorText("invalid gas-wall energy ledger face list");
        return 1;
    }
    if
    (
        (s->gasWallEnergy == nullptr) != (s->gasWallEnergyMask == nullptr)
     || s->gasWallEnergy != nullptr
    )
    {
        setLastErrorText("gas-wall energy ledger is already configured");
        return 1;
    }
    if (nEnabledFaces == 0)
    {
        return 0;
    }

    std::vector<unsigned char> hostMask
    (
        static_cast<size_t>(s->nFaces),
        static_cast<unsigned char>(0)
    );
    for (int enabledI = 0; enabledI < nEnabledFaces; ++enabledI)
    {
        const int faceI = enabledFaceIds[enabledI];
        if
        (
            faceI < s->nInternalFaces
         || faceI >= s->nFaces
         || hostMask[static_cast<size_t>(faceI)] != 0
        )
        {
            setLastErrorText("invalid or duplicate gas-wall ledger face");
            return 1;
        }
        hostMask[static_cast<size_t>(faceI)] = 1;
    }

    double* newEnergy = nullptr;
    unsigned char* newMask = nullptr;
    if
    (
        allocate
        (
            newEnergy,
            static_cast<size_t>(s->nFaces),
            "cudaMalloc gas-wall energy ledger"
        ) != 0
     || allocate
        (
            newMask,
            static_cast<size_t>(s->nFaces),
            "cudaMalloc gas-wall energy mask"
        ) != 0
    )
    {
        release(newEnergy);
        release(newMask);
        return 1;
    }

    cudaError_t err = cudaMemset
    (
        newEnergy,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(double)
    );
    if
    (
        err != cudaSuccess
     || copyToDevice
        (
            newMask,
            hostMask.data(),
            hostMask.size(),
            "cudaMemcpy gas-wall energy mask"
        ) != 0
    )
    {
        if (err != cudaSuccess)
        {
            setLastError("cudaMemset gas-wall energy ledger", err);
        }
        release(newEnergy);
        release(newMask);
        return 1;
    }

    s->gasWallEnergy = newEnergy;
    s->gasWallEnergyMask = newMask;
    if
    (
        syncGasWallLedgerPointers
        (
            s,
            "cudaMemcpy configure gas-wall energy ledger pointers"
        ) != 0
    )
    {
        s->gasWallEnergy = nullptr;
        s->gasWallEnergyMask = nullptr;
        release(newEnergy);
        release(newMask);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictPeekGasWallEnergy
(
    void* handle,
    int nFaces,
    double* gasWallEnergy
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "gas-wall energy ledger peek") != 0
     || nFaces != s->nFaces
     || nFaces <= 0
     || gasWallEnergy == nullptr
     || s->gasWallEnergy == nullptr
     || s->gasWallEnergyMask == nullptr
    )
    {
        setLastErrorText("invalid gas-wall energy ledger peek input");
        return 1;
    }
    return copyToHost
    (
        gasWallEnergy,
        s->gasWallEnergy,
        static_cast<size_t>(nFaces),
        "cudaMemcpy gas-wall energy ledger peek"
    );
}

extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange
(
    void* handle,
    int firstFace,
    int nFaces,
    double* gasWallEnergyJ,
    double* particleDepositedWallEnergyJ,
    double* particleReflectedWallEnergyJ
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "pending wall-energy ledger range peek") != 0
     || firstFace < s->nInternalFaces
     || nFaces <= 0
     || firstFace > s->nFaces - nFaces
     || gasWallEnergyJ == nullptr
     || particleDepositedWallEnergyJ == nullptr
     || particleReflectedWallEnergyJ == nullptr
     || s->gasWallEnergy == nullptr
     || s->gasWallEnergyMask == nullptr
    )
    {
        setLastErrorText("invalid pending wall-energy ledger range peek");
        return 1;
    }
    const size_t count = static_cast<size_t>(nFaces);
    if
    (
        copyToHost
        (
            gasWallEnergyJ,
            s->gasWallEnergy + firstFace,
            count,
            "cudaMemcpy compact gas-wall energy ledger peek"
        ) != 0
    )
    {
        return 1;
    }
    if (s->particleWallHeatTransferEnabled == 0)
    {
        std::fill_n(particleDepositedWallEnergyJ, count, 0.0);
        std::fill_n(particleReflectedWallEnergyJ, count, 0.0);
        return 0;
    }
    if
    (
        s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
     || copyToHost
        (
            particleDepositedWallEnergyJ,
            s->particleWallDepositedEnergy + firstFace,
            count,
            "cudaMemcpy compact deposited-particle wall energy peek"
        ) != 0
     || copyToHost
        (
            particleReflectedWallEnergyJ,
            s->particleWallReflectedEnergy + firstFace,
            count,
            "cudaMemcpy compact reflected-particle wall energy peek"
        ) != 0
    )
    {
        setLastErrorText("invalid configured particle wall-energy ledgers");
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange
(
    void* handle,
    int firstFace,
    int nFaces,
    const double* gasWallEnergyJ,
    const double* particleDepositedWallEnergyJ,
    const double* particleReflectedWallEnergyJ
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "pending wall-energy ledger range restore") != 0
     || firstFace < s->nInternalFaces
     || nFaces <= 0
     || firstFace > s->nFaces - nFaces
     || gasWallEnergyJ == nullptr
     || particleDepositedWallEnergyJ == nullptr
     || particleReflectedWallEnergyJ == nullptr
     || s->gasWallEnergy == nullptr
     || s->gasWallEnergyMask == nullptr
    )
    {
        setLastErrorText("invalid pending wall-energy ledger range restore");
        return 1;
    }
    const size_t count = static_cast<size_t>(nFaces);
    if
    (
        copyToDevice
        (
            s->gasWallEnergy + firstFace,
            gasWallEnergyJ,
            count,
            "cudaMemcpy compact gas-wall energy ledger restore"
        ) != 0
    )
    {
        return 1;
    }
    if (s->particleWallHeatTransferEnabled == 0)
    {
        if
        (
            !std::all_of
            (
                particleDepositedWallEnergyJ,
                particleDepositedWallEnergyJ + count,
                [](const double value){ return value == 0.0; }
            )
         || !std::all_of
            (
                particleReflectedWallEnergyJ,
                particleReflectedWallEnergyJ + count,
                [](const double value){ return value == 0.0; }
            )
        )
        {
            setLastErrorText
            (
                "nonzero particle wall-energy restore for disabled heat transfer"
            );
            return 1;
        }
        return 0;
    }
    if
    (
        s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
     || copyToDevice
        (
            s->particleWallDepositedEnergy + firstFace,
            particleDepositedWallEnergyJ,
            count,
            "cudaMemcpy compact deposited-particle wall energy restore"
        ) != 0
     || copyToDevice
        (
            s->particleWallReflectedEnergy + firstFace,
            particleReflectedWallEnergyJ,
            count,
            "cudaMemcpy compact reflected-particle wall energy restore"
        ) != 0
    )
    {
        setLastErrorText("invalid configured particle wall-energy ledgers");
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy
(
    void* handle,
    int nFaces,
    double* gasWallEnergy
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "gas-wall energy ledger download/reset") != 0
     || nFaces != s->nFaces
     || nFaces <= 0
     || gasWallEnergy == nullptr
     || s->gasWallEnergy == nullptr
     || s->gasWallEnergyMask == nullptr
    )
    {
        setLastErrorText("invalid gas-wall energy ledger download/reset input");
        return 1;
    }
    if
    (
        copyToHost
        (
            gasWallEnergy,
            s->gasWallEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy gas-wall energy ledger download"
        ) != 0
    )
    {
        return 1;
    }
    cudaError_t err = cudaMemset
    (
        s->gasWallEnergy,
        0,
        static_cast<size_t>(nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset gas-wall energy ledger reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError("cudaDeviceSynchronize gas-wall ledger reset", err);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel
(
    void* handle,
    int nFaces,
    const unsigned char* candidateFaceMask,
    double sommerfeldThreshold,
    int heatTransferEnabled,
    double maximumCoverage,
    double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale,
    double contactAngleDegree,
    int wallTransientResistance,
    int nonlinearIterations,
    double meltingTemperatureK,
    double mushyRangeK,
    double latentHeatJkg,
    double solidDensityKgM3,
    double solidSpecificHeatJkgK,
    double solidThermalConductivityWmK,
    double pinningThicknessFraction,
    double interfaceResistanceM2KW
)
{
    DeviceState* s = asState(handle);
    if (validateState(s, "particle stuck-model configuration") != 0)
    {
        return 1;
    }
    const bool basicParametersValid =
        std::isfinite(sommerfeldThreshold)
     && std::isfinite(depositionHeatTransferEfficiency)
     && std::isfinite(reflectionHeatTransferEfficiency)
     && std::isfinite(adhesionEnergyScale)
     && std::isfinite(contactAngleDegree)
     && sommerfeldThreshold > 0.0
     && depositionHeatTransferEfficiency > 0.0
     && depositionHeatTransferEfficiency <= 1.0
     && reflectionHeatTransferEfficiency > 0.0
     && reflectionHeatTransferEfficiency <= 1.0
     && adhesionEnergyScale > 0.0
     && contactAngleDegree > 0.0
     && contactAngleDegree < 180.0;
    const Foam::gpuThermal::ColdWallSolidificationParameters coldWallParameters
    {
        meltingTemperatureK,
        mushyRangeK,
        latentHeatJkg,
        solidDensityKgM3,
        solidSpecificHeatJkgK,
        solidThermalConductivityWmK,
        pinningThicknessFraction,
        interfaceResistanceM2KW,
        wallTransientResistance,
        nonlinearIterations
    };
    if
    (
        nFaces != s->nFaces
     || nFaces <= 0
     || candidateFaceMask == nullptr
     || !basicParametersValid
     || !Foam::gpuThermal::validColdWallSolidificationParameters
        (
            coldWallParameters
        )
     || (heatTransferEnabled != 0 && heatTransferEnabled != 1)
     || s->particleStuckModelConfigured != 0
     || s->particleStuckCandidateMask != nullptr
    )
    {
        setLastErrorText("invalid or duplicate particle stuck-model configuration");
        return 1;
    }

    int candidateCount = 0;
    bool coldWall1DEnabled = false;
    bool coldWall2DEnabled = false;
    for (int faceI = 0; faceI < nFaces; ++faceI)
    {
        if
        (
            candidateFaceMask[faceI] > 4
         || candidateFaceMask[faceI] == 2
         || (candidateFaceMask[faceI] != 0 && faceI < s->nInternalFaces)
        )
        {
            setLastErrorText
            (
                "particle wall interaction type must be one of 0, 1, 3, 4 and boundary-only"
            );
            return 1;
        }
        candidateCount += candidateFaceMask[faceI] != 0 ? 1 : 0;
        coldWall1DEnabled =
            coldWall1DEnabled
         || candidateFaceMask[faceI]
            == Foam::gpuThermal::particleWallSolidifyingDeposition;
        coldWall2DEnabled =
            coldWall2DEnabled
         || candidateFaceMask[faceI]
            == Foam::gpuThermal::particleWallColdWall2D;
    }

    if
    (
        heatTransferEnabled != 0
     && (
            candidateCount == 0
         || !std::isfinite(maximumCoverage)
         || maximumCoverage <= 0.0
         || maximumCoverage > 1.0
        )
    )
    {
        setLastErrorText
        (
            "enabled particle wall heat transfer requires candidate faces "
            "and finite positive heat-transfer/coverage parameters"
        );
        return 1;
    }

    unsigned char* candidateMask = nullptr;
    double* depositedWallEnergy = nullptr;
    double* reflectedWallEnergy = nullptr;
    double* representedArea = nullptr;
    double* areaScale = nullptr;
    double* wallEffusivityByFace = nullptr;
    float* finiteContactTable = nullptr;
    float* coldNodeSpecificEnthalpy = nullptr;
    float* coldRingSolidMass = nullptr;
    float* coldFrozenArea = nullptr;
    float* coldContactAge = nullptr;
    float* compactColdNodeSpecificEnthalpy = nullptr;
    float* compactColdRingSolidMass = nullptr;
    float* compactColdFrozenArea = nullptr;
    float* compactColdContactAge = nullptr;
    float* cold2DNodeSpecificEnthalpy = nullptr;
    float* cold2DRingContactAge = nullptr;
    float* cold2DFrozenArea = nullptr;
    float* compactCold2DNodeSpecificEnthalpy = nullptr;
    float* compactCold2DRingContactAge = nullptr;
    float* compactCold2DFrozenArea = nullptr;
    const size_t particleCapacity = static_cast<size_t>(s->particleCapacity);
    const size_t coldNodeStorage = particleCapacity
       *static_cast<size_t>(Foam::gpuThermal::coldWallAxialNodeCount);
    const size_t coldRingStorage = particleCapacity
       *static_cast<size_t>(Foam::gpuThermal::coldWallRadialRingCount);
    const size_t cold2DNodeStorage = particleCapacity
       *static_cast<size_t>(Foam::gpuThermal::coldWall2DNodeCount);
    const size_t cold2DRingStorage = particleCapacity
       *static_cast<size_t>(Foam::gpuThermal::coldWall2DRadialNodeCount);
    if
    (
        allocate
        (
            candidateMask,
            static_cast<size_t>(nFaces),
            "cudaMalloc particle stuck candidate mask"
        ) != 0
     || (
            heatTransferEnabled != 0
         && (
                allocate
                (
                    depositedWallEnergy,
                    static_cast<size_t>(nFaces),
                    "cudaMalloc deposited-particle wall energy ledger"
                ) != 0
             || allocate
                (
                    reflectedWallEnergy,
                    static_cast<size_t>(nFaces),
                    "cudaMalloc reflected-particle wall energy ledger"
                ) != 0
             || allocate
                (
                    representedArea,
                    static_cast<size_t>(nFaces),
                    "cudaMalloc particle wall represented contact area"
                ) != 0
             || allocate
                (
                    areaScale,
                    static_cast<size_t>(nFaces),
                    "cudaMalloc particle wall contact area scale"
                ) != 0
             || allocate
                (
                    wallEffusivityByFace,
                    static_cast<size_t>(nFaces),
                    "cudaMalloc CHT particle wall effusivity"
                ) != 0
             || allocate
                (
                    finiteContactTable,
                    static_cast<size_t>
                    (
                        Foam::gpuThermal::finiteContactTimeTableSize
                       *Foam::gpuThermal::finiteContactDamageTableSize
                       *Foam::gpuThermal::finiteContactPeakTableSize
                    ),
                    "cudaMalloc finite-contact rate table"
                ) != 0
            )
        )
    )
    {
        release(candidateMask);
        release(depositedWallEnergy);
        release(reflectedWallEnergy);
        release(representedArea);
        release(areaScale);
        release(wallEffusivityByFace);
        release(finiteContactTable);
        return 1;
    }

    if
    (
        copyToDevice
        (
            candidateMask,
            candidateFaceMask,
            static_cast<size_t>(nFaces),
            "cudaMemcpy particle stuck candidate mask"
        ) != 0
    )
    {
        release(candidateMask);
        release(depositedWallEnergy);
        release(reflectedWallEnergy);
        release(representedArea);
        release(areaScale);
        release(wallEffusivityByFace);
        release(finiteContactTable);
        return 1;
    }
    if
    (
        coldWall1DEnabled
     &&
        (
            allocate(coldNodeSpecificEnthalpy, coldNodeStorage, "cudaMalloc cold-wall axial enthalpy") != 0
         || allocate(coldRingSolidMass, coldRingStorage, "cudaMalloc cold-wall radial solid mass") != 0
         || allocate(coldFrozenArea, particleCapacity, "cudaMalloc cold-wall frozen footprint") != 0
         || allocate(coldContactAge, particleCapacity, "cudaMalloc cold-wall profile contact age") != 0
         || allocate(compactColdNodeSpecificEnthalpy, coldNodeStorage, "cudaMalloc compact cold-wall axial enthalpy") != 0
         || allocate(compactColdRingSolidMass, coldRingStorage, "cudaMalloc compact cold-wall radial solid mass") != 0
         || allocate(compactColdFrozenArea, particleCapacity, "cudaMalloc compact cold-wall frozen footprint") != 0
         || allocate(compactColdContactAge, particleCapacity, "cudaMalloc compact cold-wall profile contact age") != 0
        )
    )
    {
        release(candidateMask);
        release(depositedWallEnergy);
        release(reflectedWallEnergy);
        release(representedArea);
        release(areaScale);
        release(wallEffusivityByFace);
        release(finiteContactTable);
        release(coldNodeSpecificEnthalpy);
        release(coldRingSolidMass);
        release(coldFrozenArea);
        release(coldContactAge);
        release(compactColdNodeSpecificEnthalpy);
        release(compactColdRingSolidMass);
        release(compactColdFrozenArea);
        release(compactColdContactAge);
        return 1;
    }
    if
    (
        coldWall2DEnabled
     &&
        (
            allocate(cold2DNodeSpecificEnthalpy, cold2DNodeStorage, "cudaMalloc cold-wall-2D enthalpy") != 0
         || allocate(cold2DRingContactAge, cold2DRingStorage, "cudaMalloc cold-wall-2D ring contact age") != 0
         || allocate(cold2DFrozenArea, particleCapacity, "cudaMalloc cold-wall-2D frozen footprint") != 0
         || allocate(compactCold2DNodeSpecificEnthalpy, cold2DNodeStorage, "cudaMalloc compact cold-wall-2D enthalpy") != 0
         || allocate(compactCold2DRingContactAge, cold2DRingStorage, "cudaMalloc compact cold-wall-2D ring contact age") != 0
         || allocate(compactCold2DFrozenArea, particleCapacity, "cudaMalloc compact cold-wall-2D frozen footprint") != 0
        )
    )
    {
        release(candidateMask);
        release(depositedWallEnergy);
        release(reflectedWallEnergy);
        release(representedArea);
        release(areaScale);
        release(wallEffusivityByFace);
        release(finiteContactTable);
        release(coldNodeSpecificEnthalpy);
        release(coldRingSolidMass);
        release(coldFrozenArea);
        release(coldContactAge);
        release(compactColdNodeSpecificEnthalpy);
        release(compactColdRingSolidMass);
        release(compactColdFrozenArea);
        release(compactColdContactAge);
        release(cold2DNodeSpecificEnthalpy);
        release(cold2DRingContactAge);
        release(cold2DFrozenArea);
        release(compactCold2DNodeSpecificEnthalpy);
        release(compactCold2DRingContactAge);
        release(compactCold2DFrozenArea);
        return 1;
    }
    if (coldWall1DEnabled)
    {
        const cudaError_t e0 = cudaMemset(coldNodeSpecificEnthalpy, 0, coldNodeStorage*sizeof(float));
        const cudaError_t e1 = cudaMemset(coldRingSolidMass, 0, coldRingStorage*sizeof(float));
        const cudaError_t e2 = cudaMemset(coldFrozenArea, 0, particleCapacity*sizeof(float));
        const cudaError_t e3 = cudaMemset(coldContactAge, 0, particleCapacity*sizeof(float));
        if (e0 != cudaSuccess || e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess)
        {
            setLastErrorText("cudaMemset cold-wall particle state");
            release(candidateMask);
            release(depositedWallEnergy);
            release(reflectedWallEnergy);
            release(representedArea);
            release(areaScale);
            release(wallEffusivityByFace);
            release(finiteContactTable);
            release(coldNodeSpecificEnthalpy);
            release(coldRingSolidMass);
            release(coldFrozenArea);
            release(coldContactAge);
            release(compactColdNodeSpecificEnthalpy);
            release(compactColdRingSolidMass);
            release(compactColdFrozenArea);
            release(compactColdContactAge);
            return 1;
        }
    }
    if (coldWall2DEnabled)
    {
        const cudaError_t e0 = cudaMemset(cold2DNodeSpecificEnthalpy, 0, cold2DNodeStorage*sizeof(float));
        const cudaError_t e1 = cudaMemset(cold2DRingContactAge, 0, cold2DRingStorage*sizeof(float));
        const cudaError_t e2 = cudaMemset(cold2DFrozenArea, 0, particleCapacity*sizeof(float));
        if (e0 != cudaSuccess || e1 != cudaSuccess || e2 != cudaSuccess)
        {
            setLastErrorText("cudaMemset cold-wall-2D particle state");
            release(candidateMask);
            release(depositedWallEnergy);
            release(reflectedWallEnergy);
            release(representedArea);
            release(areaScale);
            release(wallEffusivityByFace);
            release(finiteContactTable);
            release(coldNodeSpecificEnthalpy);
            release(coldRingSolidMass);
            release(coldFrozenArea);
            release(coldContactAge);
            release(compactColdNodeSpecificEnthalpy);
            release(compactColdRingSolidMass);
            release(compactColdFrozenArea);
            release(compactColdContactAge);
            release(cold2DNodeSpecificEnthalpy);
            release(cold2DRingContactAge);
            release(cold2DFrozenArea);
            release(compactCold2DNodeSpecificEnthalpy);
            release(compactCold2DRingContactAge);
            release(compactCold2DFrozenArea);
            return 1;
        }
    }
    if (heatTransferEnabled != 0)
    {
        const std::vector<double> initialAreaScale
        (
            static_cast<size_t>(nFaces),
            1.0
        );
        if
        (
            copyToDevice
            (
                areaScale,
                initialAreaScale.data(),
                initialAreaScale.size(),
                "cudaMemcpy initial particle wall contact area scale"
            ) != 0
        )
        {
            release(candidateMask);
            release(depositedWallEnergy);
            release(reflectedWallEnergy);
            release(representedArea);
            release(areaScale);
            release(wallEffusivityByFace);
            release(finiteContactTable);
            return 1;
        }
        std::vector<float> hostFiniteContactTable
        (
            static_cast<size_t>
            (
                Foam::gpuThermal::finiteContactTimeTableSize
               *Foam::gpuThermal::finiteContactDamageTableSize
               *Foam::gpuThermal::finiteContactPeakTableSize
            )
        );
        Foam::gpuThermal::buildFiniteContactRateTable
        (
            hostFiniteContactTable.data()
        );
        if
        (
            copyToDevice
            (
                finiteContactTable,
                hostFiniteContactTable.data(),
                hostFiniteContactTable.size(),
                "cudaMemcpy finite-contact rate table"
            ) != 0
        )
        {
            release(candidateMask);
            release(depositedWallEnergy);
            release(reflectedWallEnergy);
            release(representedArea);
            release(areaScale);
            release(wallEffusivityByFace);
            release(finiteContactTable);
            return 1;
        }
        const cudaError_t depositedEnergyErr = cudaMemset
        (
            depositedWallEnergy,
            0,
            static_cast<size_t>(nFaces)*sizeof(double)
        );
        const cudaError_t reflectedEnergyErr = cudaMemset
        (
            reflectedWallEnergy,
            0,
            static_cast<size_t>(nFaces)*sizeof(double)
        );
        const cudaError_t areaErr = cudaMemset
        (
            representedArea, 0, static_cast<size_t>(nFaces)*sizeof(double)
        );
        const cudaError_t effusivityErr = cudaMemset
        (
            wallEffusivityByFace, 0, static_cast<size_t>(nFaces)*sizeof(double)
        );
        if
        (
            depositedEnergyErr != cudaSuccess
         || reflectedEnergyErr != cudaSuccess
         || areaErr != cudaSuccess
         || effusivityErr != cudaSuccess
        )
        {
            const cudaError_t firstError =
                depositedEnergyErr != cudaSuccess
              ? depositedEnergyErr
              : (
                    reflectedEnergyErr != cudaSuccess
                  ? reflectedEnergyErr
                  : (areaErr != cudaSuccess ? areaErr : effusivityErr)
                );
            setLastError
            (
                depositedEnergyErr != cudaSuccess
              ? "cudaMemset deposited-particle wall energy ledger"
              : (
                    reflectedEnergyErr != cudaSuccess
                  ? "cudaMemset reflected-particle wall energy ledger"
                  : (
                        areaErr != cudaSuccess
                      ? "cudaMemset particle wall represented contact area"
                      : "cudaMemset CHT particle wall effusivity"
                    )
                ),
                firstError
            );
            release(candidateMask);
            release(depositedWallEnergy);
            release(reflectedWallEnergy);
            release(representedArea);
            release(areaScale);
            release(wallEffusivityByFace);
            release(finiteContactTable);
            return 1;
        }
    }

    s->particleStuckCandidateMask = candidateMask;
    s->particleWallDepositedEnergy = depositedWallEnergy;
    s->particleWallReflectedEnergy = reflectedWallEnergy;
    s->particleWallRepresentedContactArea = representedArea;
    s->particleWallContactAreaScale = areaScale;
    s->particleWallEffusivityByFace = wallEffusivityByFace;
    s->finiteContactRateTable = finiteContactTable;
    s->pColdNodeSpecificEnthalpy = coldNodeSpecificEnthalpy;
    s->pColdRingSolidMass = coldRingSolidMass;
    s->pColdFrozenArea = coldFrozenArea;
    s->pColdContactAge = coldContactAge;
    s->compactPColdNodeSpecificEnthalpy = compactColdNodeSpecificEnthalpy;
    s->compactPColdRingSolidMass = compactColdRingSolidMass;
    s->compactPColdFrozenArea = compactColdFrozenArea;
    s->compactPColdContactAge = compactColdContactAge;
    s->pCold2DNodeSpecificEnthalpy = cold2DNodeSpecificEnthalpy;
    s->pCold2DRingContactAge = cold2DRingContactAge;
    s->pCold2DFrozenArea = cold2DFrozenArea;
    s->compactPCold2DNodeSpecificEnthalpy = compactCold2DNodeSpecificEnthalpy;
    s->compactPCold2DRingContactAge = compactCold2DRingContactAge;
    s->compactPCold2DFrozenArea = compactCold2DFrozenArea;
    s->coldWallSolidificationEnabled = coldWall1DEnabled ? 1 : 0;
    s->coldWall2DEnabled = coldWall2DEnabled ? 1 : 0;
    s->coldWallSolidificationParameters = coldWallParameters;
    s->particleStuckModelConfigured = 1;
    s->particleWallHeatTransferEnabled = heatTransferEnabled;
    s->sommerfeldThreshold = sommerfeldThreshold;
    s->particleWallMaximumCoverage = maximumCoverage;
    s->particleWallDepositionHeatTransferEfficiency =
        depositionHeatTransferEfficiency;
    s->particleWallReflectionHeatTransferEfficiency =
        reflectionHeatTransferEfficiency;
    s->particleWallAdhesionEnergyScale = adhesionEnergyScale;
    s->particleWallContactAngleCosine =
        ::cos(contactAngleDegree*M_PI/180.0);
    if
    (
        syncDeviceState
        (
            s, "cudaMemcpy configure particle stuck-model state"
        ) != 0
    )
    {
        s->particleStuckCandidateMask = nullptr;
        s->particleWallDepositedEnergy = nullptr;
        s->particleWallReflectedEnergy = nullptr;
        s->particleWallRepresentedContactArea = nullptr;
        s->particleWallContactAreaScale = nullptr;
        s->particleWallEffusivityByFace = nullptr;
        s->finiteContactRateTable = nullptr;
        s->pColdNodeSpecificEnthalpy = nullptr;
        s->pColdRingSolidMass = nullptr;
        s->pColdFrozenArea = nullptr;
        s->pColdContactAge = nullptr;
        s->compactPColdNodeSpecificEnthalpy = nullptr;
        s->compactPColdRingSolidMass = nullptr;
        s->compactPColdFrozenArea = nullptr;
        s->compactPColdContactAge = nullptr;
        s->pCold2DNodeSpecificEnthalpy = nullptr;
        s->pCold2DRingContactAge = nullptr;
        s->pCold2DFrozenArea = nullptr;
        s->compactPCold2DNodeSpecificEnthalpy = nullptr;
        s->compactPCold2DRingContactAge = nullptr;
        s->compactPCold2DFrozenArea = nullptr;
        s->coldWallSolidificationEnabled = 0;
        s->coldWall2DEnabled = 0;
        s->particleStuckModelConfigured = 0;
        s->particleWallHeatTransferEnabled = 0;
        release(candidateMask);
        release(depositedWallEnergy);
        release(reflectedWallEnergy);
        release(representedArea);
        release(areaScale);
        release(wallEffusivityByFace);
        release(finiteContactTable);
        release(coldNodeSpecificEnthalpy);
        release(coldRingSolidMass);
        release(coldFrozenArea);
        release(coldContactAge);
        release(compactColdNodeSpecificEnthalpy);
        release(compactColdRingSolidMass);
        release(compactColdFrozenArea);
        release(compactColdContactAge);
        release(cold2DNodeSpecificEnthalpy);
        release(cold2DRingContactAge);
        release(cold2DFrozenArea);
        release(compactCold2DNodeSpecificEnthalpy);
        release(compactCold2DRingContactAge);
        release(compactCold2DFrozenArea);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictPeekParticleWallHeatLedgers
(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "particle wall heat-ledger peek") != 0
     || nFaces != s->nFaces
     || nFaces <= 0
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
     || s->particleWallHeatTransferEnabled == 0
     || s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
    )
    {
        setLastErrorText("invalid particle wall heat-ledger peek");
        return 1;
    }
    if
    (
        copyToHost
        (
            depositedWallEnergyJ,
            s->particleWallDepositedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy deposited-particle wall energy peek"
        ) != 0
     || copyToHost
        (
            reflectedWallEnergyJ,
            s->particleWallReflectedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy reflected-particle wall energy peek"
        ) != 0
    )
    {
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers
(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
)
{
    DeviceState* s = asState(handle);
    if
    (
        validateState(s, "particle wall heat-ledger download/reset") != 0
     || nFaces != s->nFaces
     || nFaces <= 0
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
     || s->particleWallHeatTransferEnabled == 0
     || s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
    )
    {
        setLastErrorText("invalid particle wall heat-ledger download/reset");
        return 1;
    }
    if
    (
        copyToHost
        (
            depositedWallEnergyJ,
            s->particleWallDepositedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy deposited-particle wall energy download"
        ) != 0
     || copyToHost
        (
            reflectedWallEnergyJ,
            s->particleWallReflectedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy reflected-particle wall energy download"
        ) != 0
    )
    {
        return 1;
    }
    cudaError_t err = cudaMemset
    (
        s->particleWallDepositedEnergy,
        0,
        static_cast<size_t>(nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset deposited-particle wall energy reset", err);
        return 1;
    }
    err = cudaMemset
    (
        s->particleWallReflectedEnergy,
        0,
        static_cast<size_t>(nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset reflected-particle wall energy reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError("cudaDeviceSynchronize particle wall ledgers reset", err);
        return 1;
    }
    return 0;
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
    const unsigned char* pStuck,
    const int* pStuckFaceId,
    const float* pDepositionArea,
    const float* pContactDuration,
    const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const float* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const float* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
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
    err = cudaMemset
    (
        s->pStuck,
        0,
        static_cast<size_t>(s->particleCapacity)*sizeof(unsigned char)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict particle restart stuck", err);
        return 1;
    }
    err = cudaMemset
    (
        s->pContactDuration,
        0,
        static_cast<size_t>(s->particleCapacity)*sizeof(float)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset finite contact duration", err);
        return 1;
    }
    err = cudaMemset
    (
        s->pContactMaximumArea,
        0,
        static_cast<size_t>(s->particleCapacity)*sizeof(float)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset finite contact maximum area", err);
        return 1;
    }
    err = cudaMemset
    (
        s->pContactPeakFraction,
        0,
        static_cast<size_t>(s->particleCapacity)*sizeof(float)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset finite contact peak fraction", err);
        return 1;
    }
    err = cudaMemset
    (
        s->pStuckFaceId,
        0xff,
        static_cast<size_t>(s->particleCapacity)*sizeof(int)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict particle restart stuck face", err);
        return 1;
    }
    err = cudaMemset
    (
        s->pDepositionArea,
        0,
        static_cast<size_t>(s->particleCapacity)*sizeof(float)
    );
    if (err != cudaSuccess)
    {
        setLastError("cudaMemset strict particle restart deposition area", err);
        return 1;
    }

    int rc = 0;
    rc |= copyToDevice(s->particleCountDevice, &nParticles, 1, "cudaMemcpy strict particle restart count");
    s->preBaseDirectoryReady = 0;
    s->splitPreDirectoryActive = 0;
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
     || pCellId == nullptr || pStatus == nullptr || pStuck == nullptr
     || pStuckFaceId == nullptr || pDepositionArea == nullptr
     || pContactDuration == nullptr || pContactMaximumArea == nullptr
     || pContactPeakFraction == nullptr
     || pColdNodeSpecificEnthalpy == nullptr
     || pColdRingSolidMass == nullptr
     || pColdFrozenArea == nullptr || pColdContactAge == nullptr
     || pCold2DNodeSpecificEnthalpy == nullptr
     || pCold2DRingContactAge == nullptr
     || pCold2DFrozenArea == nullptr
     || pRng == nullptr || pOrigId == nullptr
    )
    {
        setLastErrorText("null particle restart mirror upload array");
        return 1;
    }

    const size_t n = static_cast<size_t>(nParticles);
    for (int i = 0; i < nParticles; ++i)
    {
        if
        (
            !std::isfinite(pm[i]) || pm[i] <= 0.0
         || !std::isfinite(pTheta[i]) || pTheta[i] < 0.0
         || pStuck[i] > Foam::gpuThermal::particleWallTransientDeposit
         || pStuckFaceId[i] < -2
         || pStuckFaceId[i] >= s->nFaces
         || !std::isfinite(static_cast<double>(pDepositionArea[i]))
         || pDepositionArea[i] < 0.0f
         || !std::isfinite(static_cast<double>(pContactDuration[i]))
         || !std::isfinite(static_cast<double>(pContactMaximumArea[i]))
         || !std::isfinite(static_cast<double>(pContactPeakFraction[i]))
         || pContactDuration[i] < 0.0f
         || pContactMaximumArea[i] < 0.0f
         || pContactPeakFraction[i] < 0.0f
         || !std::isfinite(static_cast<double>(pColdFrozenArea[i]))
         || !std::isfinite(static_cast<double>(pColdContactAge[i]))
         || pColdFrozenArea[i] < 0.0f || pColdContactAge[i] < 0.0f
         || !std::isfinite(static_cast<double>(pCold2DFrozenArea[i]))
         || pCold2DFrozenArea[i] < 0.0f
         || (pStuck[i] == 0 && pStuckFaceId[i] != -1)
         || (pStuck[i] == 0 && pDepositionArea[i] != 0.0f)
         || (
                pStuckFaceId[i] < 0
             && (
                    pDepositionArea[i] != 0.0f
                 || pContactDuration[i] != 0.0f
                 || pContactMaximumArea[i] != 0.0f
                 || pContactPeakFraction[i] != 0.0f
                )
            )
         || (pStuck[i] != 0 && pStuckFaceId[i] == -1)
         || (
                pStuck[i] == Foam::gpuThermal::particleWallDeposited
             && (
                    pStuckFaceId[i] < 0 || pDepositionArea[i] <= 0.0f
                 || !
                    (
                        (
                            pContactDuration[i] == 0.0f
                         && pContactMaximumArea[i] == 0.0f
                         && pContactPeakFraction[i] == 0.0f
                        )
                     || (
                            pContactDuration[i] > 0.0f
                         && pContactMaximumArea[i] > 0.0f
                         && pContactPeakFraction[i] > 0.0f
                         && pContactPeakFraction[i] < 1.0f
                        )
                    )
                )
            )
         || (
                (
                    pStuck[i] == Foam::gpuThermal::particleWallTransientRebound
                 || pStuck[i] == Foam::gpuThermal::particleWallTransientDeposit
                )
             && (
                    pStuckFaceId[i] < 0
                 || !(pContactDuration[i] > 0.0f)
                 || !(pContactMaximumArea[i] > 0.0f)
                 || !(pContactPeakFraction[i] > 0.0f)
                 || !(pContactPeakFraction[i] < 1.0f)
                 || pTheta[i] > pContactDuration[i]
                )
            )
        )
        {
            setLastErrorText("particle restart stuck state/face/area is invalid");
            return 1;
        }
        for (int node = 0; node < Foam::gpuThermal::coldWallAxialNodeCount; ++node)
        {
            const float value = pColdNodeSpecificEnthalpy
            [
                i*Foam::gpuThermal::coldWallAxialNodeCount + node
            ];
            if (!std::isfinite(static_cast<double>(value)))
            {
                setLastErrorText("particle restart cold-wall enthalpy is invalid");
                return 1;
            }
        }
        for (int ring = 0; ring < Foam::gpuThermal::coldWallRadialRingCount; ++ring)
        {
            const float value = pColdRingSolidMass
            [
                i*Foam::gpuThermal::coldWallRadialRingCount + ring
            ];
            if (!std::isfinite(static_cast<double>(value)) || value < 0.0f)
            {
                setLastErrorText("particle restart cold-wall ring mass is invalid");
                return 1;
            }
        }
        for (int node = 0; node < Foam::gpuThermal::coldWall2DNodeCount; ++node)
        {
            const float value = pCold2DNodeSpecificEnthalpy
            [i*Foam::gpuThermal::coldWall2DNodeCount + node];
            if (!std::isfinite(static_cast<double>(value)) || value < 0.0f)
            {
                setLastErrorText("particle restart cold-wall-2D enthalpy is invalid");
                return 1;
            }
        }
        for
        (
            int ring = 0;
            ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
            ++ring
        )
        {
            const float value = pCold2DRingContactAge
            [i*Foam::gpuThermal::coldWall2DRadialNodeCount + ring];
            if (!std::isfinite(static_cast<double>(value)) || value < 0.0f)
            {
                setLastErrorText("particle restart cold-wall-2D contact age is invalid");
                return 1;
            }
        }
    }
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
    rc |= copyToDevice(s->pStuck, pStuck, n, "cudaMemcpy strict restart pStuck");
    rc |= copyToDevice(s->pStuckFaceId, pStuckFaceId, n, "cudaMemcpy strict restart pStuckFaceId");
    rc |= copyToDevice(s->pDepositionArea, pDepositionArea, n, "cudaMemcpy strict restart pDepositionArea");
    rc |= copyToDevice(s->pContactDuration, pContactDuration, n, "cudaMemcpy restart pContactDuration");
    rc |= copyToDevice(s->pContactMaximumArea, pContactMaximumArea, n, "cudaMemcpy restart pContactMaximumArea");
    rc |= copyToDevice(s->pContactPeakFraction, pContactPeakFraction, n, "cudaMemcpy restart pContactPeakFraction");
    if (s->coldWallSolidificationEnabled != 0)
    {
        rc |= copyToDevice
        (
            s->pColdNodeSpecificEnthalpy,
            pColdNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWallAxialNodeCount,
            "cudaMemcpy restart pColdNodeSpecificEnthalpy"
        );
        rc |= copyToDevice
        (
            s->pColdRingSolidMass,
            pColdRingSolidMass,
            n*Foam::gpuThermal::coldWallRadialRingCount,
            "cudaMemcpy restart pColdRingSolidMass"
        );
        rc |= copyToDevice
        (
            s->pColdFrozenArea,
            pColdFrozenArea,
            n,
            "cudaMemcpy restart pColdFrozenArea"
        );
        rc |= copyToDevice
        (
            s->pColdContactAge,
            pColdContactAge,
            n,
            "cudaMemcpy restart pColdContactAge"
        );
    }
    if (s->coldWall2DEnabled != 0)
    {
        rc |= copyToDevice
        (
            s->pCold2DNodeSpecificEnthalpy,
            pCold2DNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWall2DNodeCount,
            "cudaMemcpy restart pCold2DNodeSpecificEnthalpy"
        );
        rc |= copyToDevice
        (
            s->pCold2DRingContactAge,
            pCold2DRingContactAge,
            n*Foam::gpuThermal::coldWall2DRadialNodeCount,
            "cudaMemcpy restart pCold2DRingContactAge"
        );
        rc |= copyToDevice
        (
            s->pCold2DFrozenArea,
            pCold2DFrozenArea,
            n,
            "cudaMemcpy restart pCold2DFrozenArea"
        );
    }
    rc |= copyToDevice(s->pRng, pRng, n, "cudaMemcpy strict restart pRng");
    rc |= copyToDevice(s->pOrigId, pOrigId, n, "cudaMemcpy strict restart pOrigId");

    if (rc != 0)
    {
        return 1;
    }

    err = cudaMemset
    (
        s->wallBoundParticleCountDevice,
        0,
        sizeof(int)
    );
    if (err != cudaSuccess)
    {
        setLastError("reset restart wall-bound particle directory", err);
        return 1;
    }
    if (nParticles > 0)
    {
        rebuildWallBoundParticleDirectoryKernel
            <<<s->particleWorkGrid, s->particleBlockThreads>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("rebuild restart wall-bound particle directory", err);
            return 1;
        }
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
    unsigned char* pStuck,
    int* pStuckFaceId,
    float* pDepositionArea,
    float* pContactDuration,
    float* pContactMaximumArea,
    float* pContactPeakFraction,
    float* pColdNodeSpecificEnthalpy,
    float* pColdRingSolidMass,
    float* pColdFrozenArea,
    float* pColdContactAge,
    float* pCold2DNodeSpecificEnthalpy,
    float* pCold2DRingContactAge,
    float* pCold2DFrozenArea,
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
     || pCellId == nullptr || pStatus == nullptr || pStuck == nullptr
     || pStuckFaceId == nullptr || pDepositionArea == nullptr
     || pContactDuration == nullptr || pContactMaximumArea == nullptr
     || pContactPeakFraction == nullptr
     || pColdNodeSpecificEnthalpy == nullptr
     || pColdRingSolidMass == nullptr
     || pColdFrozenArea == nullptr || pColdContactAge == nullptr
     || pCold2DNodeSpecificEnthalpy == nullptr
     || pCold2DRingContactAge == nullptr
     || pCold2DFrozenArea == nullptr
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
    rc |= copyToHost(pStuck, s->pStuck, n, "cudaMemcpy strict restart pStuck");
    rc |= copyToHost(pStuckFaceId, s->pStuckFaceId, n, "cudaMemcpy strict restart pStuckFaceId");
    rc |= copyToHost(pDepositionArea, s->pDepositionArea, n, "cudaMemcpy strict restart pDepositionArea");
    rc |= copyToHost(pContactDuration, s->pContactDuration, n, "cudaMemcpy restart pContactDuration");
    rc |= copyToHost(pContactMaximumArea, s->pContactMaximumArea, n, "cudaMemcpy restart pContactMaximumArea");
    rc |= copyToHost(pContactPeakFraction, s->pContactPeakFraction, n, "cudaMemcpy restart pContactPeakFraction");
    if (s->coldWallSolidificationEnabled != 0)
    {
        rc |= copyToHost
        (
            pColdNodeSpecificEnthalpy,
            s->pColdNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWallAxialNodeCount,
            "cudaMemcpy restart pColdNodeSpecificEnthalpy"
        );
        rc |= copyToHost
        (
            pColdRingSolidMass,
            s->pColdRingSolidMass,
            n*Foam::gpuThermal::coldWallRadialRingCount,
            "cudaMemcpy restart pColdRingSolidMass"
        );
        rc |= copyToHost
        (
            pColdFrozenArea,
            s->pColdFrozenArea,
            n,
            "cudaMemcpy restart pColdFrozenArea"
        );
        rc |= copyToHost
        (
            pColdContactAge,
            s->pColdContactAge,
            n,
            "cudaMemcpy restart pColdContactAge"
        );
    }
    else
    {
        std::fill_n
        (
            pColdNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWallAxialNodeCount,
            0.0f
        );
        std::fill_n
        (
            pColdRingSolidMass,
            n*Foam::gpuThermal::coldWallRadialRingCount,
            0.0f
        );
        std::fill_n(pColdFrozenArea, n, 0.0f);
        std::fill_n(pColdContactAge, n, 0.0f);
    }
    if (s->coldWall2DEnabled != 0)
    {
        rc |= copyToHost
        (
            pCold2DNodeSpecificEnthalpy,
            s->pCold2DNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWall2DNodeCount,
            "cudaMemcpy restart pCold2DNodeSpecificEnthalpy"
        );
        rc |= copyToHost
        (
            pCold2DRingContactAge,
            s->pCold2DRingContactAge,
            n*Foam::gpuThermal::coldWall2DRadialNodeCount,
            "cudaMemcpy restart pCold2DRingContactAge"
        );
        rc |= copyToHost
        (
            pCold2DFrozenArea,
            s->pCold2DFrozenArea,
            n,
            "cudaMemcpy restart pCold2DFrozenArea"
        );
    }
    else
    {
        std::fill_n
        (
            pCold2DNodeSpecificEnthalpy,
            n*Foam::gpuThermal::coldWall2DNodeCount,
            0.0f
        );
        std::fill_n
        (
            pCold2DRingContactAge,
            n*Foam::gpuThermal::coldWall2DRadialNodeCount,
            0.0f
        );
        std::fill_n(pCold2DFrozenArea, n, 0.0f);
    }
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
