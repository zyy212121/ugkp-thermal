#include "GpuBackendApi.H"
#include "GpuBackendProtocol.H"

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

using namespace ugkwpGpuIpc;

struct ClientState
{
    int fd = -1;
    pid_t childPid = -1;
    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nCellPlanes = 0;
    int particleCapacity = 0;
};

thread_local std::string lastError = "no GPU backend client error";

int fail(const std::string& message)
{
    lastError = message;
    return -1;
}

bool writeAll(const int fd, const void* data, std::uint64_t bytes)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    while (bytes != 0)
    {
        const std::size_t chunk =
            bytes > static_cast<std::uint64_t>(SSIZE_MAX)
          ? static_cast<std::size_t>(SSIZE_MAX)
          : static_cast<std::size_t>(bytes);
        const ssize_t wrote = ::send(fd, p, chunk, MSG_NOSIGNAL);
        if (wrote < 0 && errno == EINTR)
        {
            continue;
        }
        if (wrote <= 0)
        {
            lastError = std::string("GPU backend IPC write failed: ")
                      + std::strerror(errno);
            return false;
        }
        p += wrote;
        bytes -= static_cast<std::uint64_t>(wrote);
    }
    return true;
}

bool readAll(const int fd, void* data, std::uint64_t bytes)
{
    unsigned char* p = static_cast<unsigned char*>(data);
    while (bytes != 0)
    {
        const std::size_t chunk =
            bytes > static_cast<std::uint64_t>(SSIZE_MAX)
          ? static_cast<std::size_t>(SSIZE_MAX)
          : static_cast<std::size_t>(bytes);
        const ssize_t got = ::read(fd, p, chunk);
        if (got < 0 && errno == EINTR)
        {
            continue;
        }
        if (got == 0)
        {
            lastError = "GPU backend exited or closed its IPC channel";
            return false;
        }
        if (got < 0)
        {
            lastError = std::string("GPU backend IPC read failed: ")
                      + std::strerror(errno);
            return false;
        }
        p += got;
        bytes -= static_cast<std::uint64_t>(got);
    }
    return true;
}

template<class T>
bool sendObject(const int fd, const T& value)
{
    return writeAll(fd, &value, sizeof(T));
}

template<class T>
bool sendArray(const int fd, const T* values, const std::uint64_t count)
{
    if (count == 0)
    {
        return true;
    }
    if (values == nullptr)
    {
        lastError = "null array passed to GPU backend IPC";
        return false;
    }
    return writeAll(fd, values, count*sizeof(T));
}

template<class T>
bool receiveArray(const int fd, T* values, const std::uint64_t count)
{
    if (count == 0)
    {
        return true;
    }
    if (values == nullptr)
    {
        lastError = "null output array passed to GPU backend IPC";
        return false;
    }
    return readAll(fd, values, count*sizeof(T));
}

bool startRequest(ClientState* state, const Op op, const std::uint64_t bytes)
{
    if (state == nullptr || state->fd < 0)
    {
        lastError = "invalid GPU backend client handle";
        return false;
    }
    if (bytes > maxPayloadBytes)
    {
        lastError = "GPU backend request exceeds protocol payload limit";
        return false;
    }
    const RequestHeader header
    {
        magic, protocolMajor, protocolMinor,
        static_cast<std::uint32_t>(op), 0U, bytes
    };
    return sendObject(state->fd, header);
}

int receiveResponse
(
    ClientState* state,
    std::uint64_t& payloadBytes,
    const bool requireExact,
    const std::uint64_t expectedBytes
)
{
    ResponseHeader response{};
    if (!readAll(state->fd, &response, sizeof(response)))
    {
        return -1;
    }
    if
    (
        response.magicValue != magic
     || response.major != protocolMajor
     || response.minor != protocolMinor
     || response.errorBytes > 1024U*1024U
     || response.payloadBytes > maxPayloadBytes
    )
    {
        return fail("invalid or incompatible GPU backend response header");
    }

    std::string backendError(response.errorBytes, '\0');
    if (response.errorBytes != 0 && !readAll(state->fd, &backendError[0], response.errorBytes))
    {
        return -1;
    }
    if (response.status != 0)
    {
        lastError = backendError.empty()
          ? "GPU backend returned an unspecified error"
          : backendError;
        return response.status;
    }
    if (requireExact && response.payloadBytes != expectedBytes)
    {
        return fail("GPU backend response payload size mismatch");
    }
    payloadBytes = response.payloadBytes;
    lastError = "no GPU backend client error";
    return 0;
}

int finishNoPayload(ClientState* state)
{
    std::uint64_t bytes = 0;
    return receiveResponse(state, bytes, true, 0);
}

std::string backendExecutablePath()
{
    if (const char* configured = std::getenv("FSH_CUDA_BACKEND"))
    {
        if (*configured != '\0')
        {
            return configured;
        }
    }

    char executable[PATH_MAX + 1]{};
    const ssize_t length = ::readlink("/proc/self/exe", executable, PATH_MAX);
    if (length <= 0)
    {
        return "FSHCudaBackend";
    }
    executable[length] = '\0';
    std::string path(executable);
    const std::size_t slash = path.find_last_of('/');
    return (slash == std::string::npos ? std::string() : path.substr(0, slash + 1))
         + "FSHCudaBackend";
}

bool spawnBackend(ClientState& state)
{
    int sockets[2] = {-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
    {
        lastError = std::string("cannot create GPU backend IPC socket: ")
                  + std::strerror(errno);
        return false;
    }

    const pid_t child = ::fork();
    if (child < 0)
    {
        lastError = std::string("cannot fork GPU backend: ") + std::strerror(errno);
        ::close(sockets[0]);
        ::close(sockets[1]);
        return false;
    }

    if (child == 0)
    {
        ::close(sockets[0]);
        char fdText[32]{};
        std::snprintf(fdText, sizeof(fdText), "%d", sockets[1]);
        const std::string executable = backendExecutablePath();
        ::execl(executable.c_str(), executable.c_str(), "--ipc-fd", fdText,
                static_cast<char*>(nullptr));
        std::fprintf(stderr, "Cannot exec GPU backend %s: %s\n",
                     executable.c_str(), std::strerror(errno));
        ::_exit(127);
    }

    ::close(sockets[1]);
    state.fd = sockets[0];
    state.childPid = child;
    return true;
}

bool addArrays(std::uint64_t& total, const std::uint64_t count,
               const std::uint64_t elementBytes, const unsigned copies)
{
    for (unsigned i = 0; i < copies; ++i)
    {
        if (!addBytes(total, count, elementBytes))
        {
            lastError = "GPU backend payload size overflow";
            return false;
        }
    }
    return true;
}

}             

extern "C" const char* ugkwpGpuResidentStrictLastError()
{
    return lastError.c_str();
}

extern "C" int ugkwpGpuResidentStrictCreate
(
    int nCells, int nFaces, int nInternalFaces, int nCellPlanes,
    int particleCapacity, int maxFaceWalkHops, double injectionParcelMass,
    unsigned long long rngSeed, double gammaGas, double Rgas,
    double rhoSolid, int solveParticleTemperature, double gasMu,
    double gasPr, int dragModelId, int particleGasHeatTransferModelId,
    double dragResidualRe,
    double gravityX, double gravityY, double gravityZ,
    double particleDiameterFallback,
    double particleDiameterMin, double particleDiameterMax,
    double particleDiameterSigma, double injectionTheta, double rhoMin,
    double TgasMin, double epsSMin, double thetaMin, double TpMin,
    double TpMax, int collisionalPressureEnabled,
    double collisionalRestitution, double pressureKickFraction,
    int jammingPressureEnabled, double packingFraction,
    int packingProjectionIterations,
    int gasFluxScheme, int gasReconstruction, int gasLimiter,
    int gasTimeIntegrator, int gasRobustFallback, int turbulenceModel,
    double lesDeltaCoeff,
    double turbulentPrandtl, double waleCw, double smagorinskyCs,
    double maxDiffusionNumber, int csrCellLocalPathEnabled,
    int csrHeavyReductionMode, int csrHeavyAutoInterval,
    int particleBlockThreads, int reductionBlockThreads,
    int csrWarpAggregatedBinning,
    void** handle
)
{
    if (handle == nullptr || nCells <= 0 || nFaces <= 0
     || nInternalFaces < 0 || nInternalFaces > nFaces || nCellPlanes < 0
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
     || gasFluxScheme < static_cast<int>(GasFluxScheme::RusanovTadmor)
     || gasFluxScheme > static_cast<int>(GasFluxScheme::Slau2_2)
     || gasReconstruction < static_cast<int>(GasReconstruction::firstOrder)
     || gasReconstruction
        > static_cast<int>
          (
              GasReconstruction::OpenFoamEnergyLimitedLinear
          )
     || gasLimiter < static_cast<int>(GasLimiter::none)
     || gasLimiter > static_cast<int>(GasLimiter::venkatakrishnan)
     || gasTimeIntegrator < static_cast<int>(GasTimeIntegrator::Euler)
     || gasTimeIntegrator > static_cast<int>(GasTimeIntegrator::SSPRK3)
     || (gasRobustFallback != 0 && gasRobustFallback != 1)
     || ((gasFluxScheme == static_cast<int>(GasFluxScheme::Hllc)
       || gasFluxScheme == static_cast<int>(GasFluxScheme::Roe)
       || gasFluxScheme == static_cast<int>(GasFluxScheme::Hllem))
      && gasRobustFallback == 0)
     || turbulenceModel < static_cast<int>(TurbulenceModel::laminar)
     || turbulenceModel > static_cast<int>(TurbulenceModel::kOmegaSST)
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
     || (csrWarpAggregatedBinning != 0 && csrWarpAggregatedBinning != 1)
     || (particleBlockThreads != 32 && particleBlockThreads != 64
      && particleBlockThreads != 128 && particleBlockThreads != 256)
     || (reductionBlockThreads != 32 && reductionBlockThreads != 64
      && reductionBlockThreads != 128 && reductionBlockThreads != 256)
     || (!csrCellLocalPathEnabled
      && (csrHeavyReductionMode != 0 || csrWarpAggregatedBinning))
     || (jammingPressureEnabled != 0 && jammingPressureEnabled != 1)
     || packingProjectionIterations < 1
     || (jammingPressureEnabled != 0
      && (!std::isfinite(packingFraction)
       || packingFraction <= 0.0 || packingFraction >= 1.0)))
    {
        return fail("invalid GPU backend create arguments");
    }

    ClientState* state = new ClientState;
    state->nCells = nCells;
    state->nFaces = nFaces;
    state->nInternalFaces = nInternalFaces;
    state->nCellPlanes = nCellPlanes;
    state->particleCapacity = particleCapacity;
    if (!spawnBackend(*state))
    {
        delete state;
        return -1;
    }

    const CreateArgs args
    {
        nCells, nFaces, nInternalFaces, nCellPlanes, particleCapacity,
        maxFaceWalkHops, injectionParcelMass, rngSeed, gammaGas, Rgas,
        rhoSolid,
        solveParticleTemperature, gasMu, gasPr,
        dragModelId, particleGasHeatTransferModelId, dragResidualRe,
        gravityX, gravityY, gravityZ,
        particleDiameterFallback, particleDiameterMin,
        particleDiameterMax, particleDiameterSigma, injectionTheta, rhoMin,
        TgasMin, epsSMin, thetaMin, TpMin, TpMax,
        collisionalPressureEnabled, collisionalRestitution,
        pressureKickFraction, jammingPressureEnabled, packingFraction,
        packingProjectionIterations,
        gasFluxScheme, gasReconstruction, gasLimiter, gasTimeIntegrator,
        gasRobustFallback, turbulenceModel,
        lesDeltaCoeff, turbulentPrandtl,
        waleCw, smagorinskyCs, maxDiffusionNumber,
        csrCellLocalPathEnabled, csrHeavyReductionMode,
        csrHeavyAutoInterval, particleBlockThreads,
        reductionBlockThreads, csrWarpAggregatedBinning
    };
    if (!startRequest(state, Op::create, sizeof(args)) || !sendObject(state->fd, args))
    {
        const std::string savedError = lastError;
        ugkwpGpuResidentStrictRelease(state);
        lastError = savedError;
        return -1;
    }
    const int status = finishNoPayload(state);
    if (status != 0)
    {
        const std::string savedError = lastError;
        ugkwpGpuResidentStrictRelease(state);
        lastError = savedError;
        return status;
    }
    *handle = state;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictUploadMesh
(
    void* handle, const int* faceOwner, const int* faceNeighbour,
    const int* facePeriodicPair,
    const double* facePeriodicDx, const double* facePeriodicDy,
    const double* facePeriodicDz,
    const double* V, const double* Cx, const double* Cy, const double* Cz,
    const double* faceCx, const double* faceCy, const double* faceCz,
    const double* Sfx, const double* Sfy, const double* Sfz,
    const double* magSf, const double* deltaCoeffs, const double* faceWeight,
    const double* cellLength, const int* cellPlaneStart,
    const int* cellPlaneCount, const int* cellFaceId,
    const int* cellFaceNeighbor, const int* cellFaceKind,
    const double* cellFaceRestitution, const double* cellFaceTangential,
    const double* planeNx, const double* planeNy, const double* planeNz,
    const double* planeD
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr) return fail("invalid GPU backend client handle");
    std::uint64_t bytes = 0;
    if (!addArrays(bytes, s->nFaces, sizeof(int), 3)
     || !addArrays(bytes, s->nCells, sizeof(double), 5)
     || !addArrays(bytes, s->nFaces, sizeof(double), 12)
     || !addArrays(bytes, s->nCells, sizeof(int), 2)
     || !addArrays(bytes, s->nCellPlanes, sizeof(int), 3)
     || !addArrays(bytes, s->nCellPlanes, sizeof(double), 6)
     || !startRequest(s, Op::uploadMesh, bytes)) return -1;
#define SEND(P,N) do { if (!sendArray(s->fd, (P), (N))) return -1; } while (false)
    SEND(faceOwner, s->nFaces); SEND(faceNeighbour, s->nFaces);
    SEND(facePeriodicPair, s->nFaces);
    SEND(facePeriodicDx, s->nFaces); SEND(facePeriodicDy, s->nFaces);
    SEND(facePeriodicDz, s->nFaces);
    SEND(V, s->nCells); SEND(Cx, s->nCells); SEND(Cy, s->nCells);
    SEND(Cz, s->nCells); SEND(faceCx, s->nFaces); SEND(faceCy, s->nFaces);
    SEND(faceCz, s->nFaces); SEND(Sfx, s->nFaces); SEND(Sfy, s->nFaces);
    SEND(Sfz, s->nFaces); SEND(magSf, s->nFaces);
    SEND(deltaCoeffs, s->nFaces); SEND(faceWeight, s->nFaces);
    SEND(cellLength, s->nCells); SEND(cellPlaneStart, s->nCells);
    SEND(cellPlaneCount, s->nCells); SEND(cellFaceId, s->nCellPlanes);
    SEND(cellFaceNeighbor, s->nCellPlanes); SEND(cellFaceKind, s->nCellPlanes);
    SEND(cellFaceRestitution, s->nCellPlanes);
    SEND(cellFaceTangential, s->nCellPlanes); SEND(planeNx, s->nCellPlanes);
    SEND(planeNy, s->nCellPlanes); SEND(planeNz, s->nCellPlanes);
    SEND(planeD, s->nCellPlanes);
#undef SEND
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictUploadBoundarySources
(
    void* handle, int count, const int* sourceCell, const int* sourceFace,
    const double* sourcePx, const double* sourcePy, const double* sourcePz,
    const double* sourceUx, const double* sourceUy, const double* sourceUz,
    const double* sourceT, const double* sourceTheta, const double* sourceD,
    const double* sourceMassRate
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || count < 0 || count > s->nFaces)
        return fail("invalid GPU boundary source count");
    std::uint64_t bytes = sizeof(CountArgs);
    if (!addArrays(bytes, count, sizeof(int), 2)
     || !addArrays(bytes, count, sizeof(double), 10)
     || !startRequest(s, Op::uploadBoundarySources, bytes)
     || !sendObject(s->fd, CountArgs{count})) return -1;
#define SEND(P) do { if (!sendArray(s->fd, (P), count)) return -1; } while (false)
    SEND(sourceCell); SEND(sourceFace); SEND(sourcePx); SEND(sourcePy);
    SEND(sourcePz); SEND(sourceUx); SEND(sourceUy); SEND(sourceUz);
    SEND(sourceT); SEND(sourceTheta); SEND(sourceD); SEND(sourceMassRate);
#undef SEND
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel
(
    void* handle, int nFaces, const unsigned char* candidateFaceMask,
    double sommerfeldThreshold, int heatTransferEnabled,
    double maximumCoverage, double depositionHeatTransferEfficiency,
    double reflectionHeatTransferEfficiency,
    double adhesionEnergyScale, double wallDensityKgM3,
    double wallSpecificHeatJkgK, double wallConductivityWmK,
    double contactAngleDegree, int wallTransientResistance,
    int nonlinearIterations, double meltingTemperatureK,
    double mushyRangeK, double latentHeatJkg, double solidDensityKgM3,
    double solidSpecificHeatJkgK, double solidThermalConductivityWmK,
    double pinningThicknessFraction, double interfaceResistanceM2KW
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if
    (
        s == nullptr
     || nFaces != s->nFaces
     || candidateFaceMask == nullptr
     || (heatTransferEnabled != 0 && heatTransferEnabled != 1)
     || !std::isfinite(sommerfeldThreshold)
     || !std::isfinite(maximumCoverage)
     || !std::isfinite(depositionHeatTransferEfficiency)
     || !std::isfinite(reflectionHeatTransferEfficiency)
     || !std::isfinite(adhesionEnergyScale)
     || !std::isfinite(wallDensityKgM3)
     || !std::isfinite(wallSpecificHeatJkgK)
     || !std::isfinite(wallConductivityWmK)
     || !std::isfinite(contactAngleDegree)
     || !std::isfinite(meltingTemperatureK)
     || !std::isfinite(mushyRangeK)
     || !std::isfinite(latentHeatJkg)
     || !std::isfinite(solidDensityKgM3)
     || !std::isfinite(solidSpecificHeatJkgK)
     || !std::isfinite(solidThermalConductivityWmK)
     || !std::isfinite(pinningThicknessFraction)
     || !std::isfinite(interfaceResistanceM2KW)
     || sommerfeldThreshold <= 0.0
     || maximumCoverage <= 0.0
     || maximumCoverage > 1.0
     || depositionHeatTransferEfficiency <= 0.0
     || depositionHeatTransferEfficiency > 1.0
     || reflectionHeatTransferEfficiency <= 0.0
     || reflectionHeatTransferEfficiency > 1.0
     || adhesionEnergyScale <= 0.0
     || wallDensityKgM3 <= 0.0
     || wallSpecificHeatJkgK <= 0.0
     || wallConductivityWmK <= 0.0
     || contactAngleDegree <= 0.0
     || contactAngleDegree >= 180.0
     || (wallTransientResistance != 0 && wallTransientResistance != 1)
     || nonlinearIterations < 1
     || nonlinearIterations > 12
     || meltingTemperatureK <= 0.0
     || mushyRangeK <= 0.0
     || latentHeatJkg <= 0.0
     || solidDensityKgM3 <= 0.0
     || solidSpecificHeatJkgK <= 0.0
     || solidThermalConductivityWmK <= 0.0
     || pinningThicknessFraction <= 0.0
     || pinningThicknessFraction > 1.0
     || interfaceResistanceM2KW < 0.0
    )
    {
        return fail("invalid particle stuck-model configuration");
    }
    const ParticleStuckConfigArgs args
    {
        nFaces,
        heatTransferEnabled,
        sommerfeldThreshold,
        maximumCoverage,
        depositionHeatTransferEfficiency,
        reflectionHeatTransferEfficiency,
        adhesionEnergyScale,
        wallDensityKgM3,
        wallSpecificHeatJkgK,
        wallConductivityWmK,
        contactAngleDegree,
        wallTransientResistance,
        nonlinearIterations,
        meltingTemperatureK,
        mushyRangeK,
        latentHeatJkg,
        solidDensityKgM3,
        solidSpecificHeatJkgK,
        solidThermalConductivityWmK,
        pinningThicknessFraction,
        interfaceResistanceM2KW
    };
    const std::uint64_t bytes =
        sizeof(args) + std::uint64_t(nFaces)*sizeof(unsigned char);
    if
    (
        !startRequest(s, Op::configureParticleStuckModel, bytes)
     || !sendObject(s->fd, args)
     || !sendArray(s->fd, candidateFaceMask, nFaces)
    )
    {
        return -1;
    }
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictConfigureScheduledInlet
(
    void* handle, int count, const int* faceIds, double inletTemperature,
    int nPressureRows, const double* pressureTimes,
    const double* pressureValues, int nVolumeFractionRows,
    const double* volumeFractionTimes, const double* volumeFractionValues
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if
    (
        s == nullptr || count <= 0 || count > s->nFaces
     || nPressureRows <= 0 || nVolumeFractionRows <= 0
    )
        return fail("invalid scheduled inlet configuration counts");
    const ScheduledInletConfigArgs args
    {
        count, nPressureRows, nVolumeFractionRows, inletTemperature
    };
    std::uint64_t bytes = sizeof(args);
    if
    (
        !addArrays(bytes, count, sizeof(int), 1)
     || !addArrays(bytes, nPressureRows, sizeof(double), 2)
     || !addArrays(bytes, nVolumeFractionRows, sizeof(double), 2)
     || !startRequest(s, Op::configureScheduledInlet, bytes)
     || !sendObject(s->fd, args)
     || !sendArray(s->fd, faceIds, count)
     || !sendArray(s->fd, pressureTimes, nPressureRows)
     || !sendArray(s->fd, pressureValues, nPressureRows)
     || !sendArray(s->fd, volumeFractionTimes, nVolumeFractionRows)
     || !sendArray(s->fd, volumeFractionValues, nVolumeFractionRows)
    ) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass
(
    void* handle, int* nSources, int maxSources, int* sourceFace,
    double* residualMass
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || nSources == nullptr || maxSources < 0)
        return fail("invalid source residual download arguments");
    if (!startRequest(s, Op::downloadSourceResidualMass, sizeof(CountArgs))
     || !sendObject(s->fd, CountArgs{maxSources})) return -1;
    std::uint64_t payload = 0;
    const int status = receiveResponse(s, payload, false, 0);
    if (status != 0) return status;
    int count = 0;
    if (payload < sizeof(count) || !readAll(s->fd, &count, sizeof(count))) return -1;
    const std::uint64_t expected = sizeof(count)
      + (maxSources == 0 ? 0 : std::uint64_t(count)*(sizeof(int) + sizeof(double)));
    if (count < 0 || (maxSources != 0 && count > maxSources) || payload != expected)
        return fail("invalid source residual response");
    *nSources = count;
    if (maxSources != 0)
    {
        if (!receiveArray(s->fd, sourceFace, count)
         || !receiveArray(s->fd, residualMass, count)) return -1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass
(
    void* handle, int count, const int* sourceFace, const double* residualMass
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || count < 0 || count > s->nFaces)
        return fail("invalid source residual upload count");
    const std::uint64_t bytes = sizeof(CountArgs)
      + std::uint64_t(count)*(sizeof(int) + sizeof(double));
    if (!startRequest(s, Op::uploadSourceResidualMass, bytes)
     || !sendObject(s->fd, CountArgs{count})
     || !sendArray(s->fd, sourceFace, count)
     || !sendArray(s->fd, residualMass, count)) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields
(
    void* handle, const int* kind, const int* rhoFix, const int* uFix,
    const int* pFix, const int* tFix, const int* pWave,
    const double* pWaveGamma, const double* pWaveFieldInf,
    const double* pWaveLInf, const double* rho, const double* ux,
    const double* uy, const double* uz, const double* p, const double* t
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr) return fail("invalid GPU backend client handle");
    std::uint64_t bytes = 0;
    if (!addArrays(bytes, s->nFaces, sizeof(int), 6)
     || !addArrays(bytes, s->nFaces, sizeof(double), 9)
     || !startRequest(s, Op::uploadGasBoundaryFields, bytes)) return -1;
#define SEND(P) do { if (!sendArray(s->fd, (P), s->nFaces)) return -1; } while (false)
    SEND(kind); SEND(rhoFix); SEND(uFix); SEND(pFix); SEND(tFix); SEND(pWave);
    SEND(pWaveGamma); SEND(pWaveFieldInf); SEND(pWaveLInf); SEND(rho);
    SEND(ux); SEND(uy); SEND(uz); SEND(p); SEND(t);
#undef SEND
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictUploadFields
(
    void* handle, const double* rho, const double* rhoUx,
    const double* rhoUy, const double* rhoUz, const double* rhoE,
    const double* Ux, const double* Uy, const double* Uz, const double* p,
    const double* Tgas
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr) return fail("invalid GPU backend client handle");
    std::uint64_t bytes = 0;
    if (!addArrays(bytes, s->nCells, sizeof(double), 10)
     || !startRequest(s, Op::uploadFields, bytes)) return -1;
    const double* fields[] = {rho,rhoUx,rhoUy,rhoUz,rhoE,Ux,Uy,Uz,p,Tgas};
    for (const double* values : fields)
        if (!sendArray(s->fd, values, s->nCells)) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictAdvance
(void* handle, double dt, double simulationTime)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const AdvanceArgs args{dt, simulationTime};
    if (!startRequest(s, Op::advance, sizeof(args)) || !sendObject(s->fd, args))
        return -1;
    return finishNoPayload(s);
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
    ClientState* s = static_cast<ClientState*>(handle);
    if
    (
        s == nullptr
     || k == nullptr
     || omega == nullptr
     || wallDistance == nullptr
     || boundaryKMode == nullptr
     || boundaryOmegaMode == nullptr
     || boundaryK == nullptr
     || boundaryOmega == nullptr
    )
    {
        return fail("invalid SST configuration request");
    }

    const SstConfigArgs args
    {
        alphaK1, alphaK2, alphaOmega1, alphaOmega2,
        beta1, beta2, betaStar, gamma1, gamma2,
        a1, b1, c1, kMin, omegaMin, maxSourceNumber,
        wallTreatment, wallKappa, wallE, wallCmu
    };
    std::uint64_t bytes = sizeof(args);
    if
    (
        !addArrays(bytes, s->nCells, sizeof(double), 3)
     || !addArrays(bytes, s->nFaces, sizeof(int), 2)
     || !addArrays(bytes, s->nFaces, sizeof(double), 2)
     || !startRequest(s, Op::configureSst, bytes)
     || !sendObject(s->fd, args)
     || !sendArray(s->fd, k, s->nCells)
     || !sendArray(s->fd, omega, s->nCells)
     || !sendArray(s->fd, wallDistance, s->nCells)
     || !sendArray(s->fd, boundaryKMode, s->nFaces)
     || !sendArray(s->fd, boundaryOmegaMode, s->nFaces)
     || !sendArray(s->fd, boundaryK, s->nFaces)
     || !sendArray(s->fd, boundaryOmega, s->nFaces)
    )
    {
        return -1;
    }
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadSst
(
    void* handle,
    double* k,
    double* omega,
    double* nut
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || k == nullptr || omega == nullptr || nut == nullptr)
    {
        return fail("invalid SST download request");
    }
    const std::uint64_t expected =
        3ULL*static_cast<std::uint64_t>(s->nCells)*sizeof(double);
    if (!startRequest(s, Op::downloadSst, 0))
    {
        return -1;
    }
    std::uint64_t payload = 0;
    const int status = receiveResponse(s, payload, true, expected);
    if (status != 0)
    {
        return status;
    }
    if
    (
        !receiveArray(s->fd, k, s->nCells)
     || !receiveArray(s->fd, omega, s->nCells)
     || !receiveArray(s->fd, nut, s->nCells)
    )
    {
        return -1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(void* handle, double dt, double targetMaxCo, double scheduleTime, double* maxCo)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (maxCo == nullptr) return fail("null gas Courant output");
    const CourantArgs args{dt, targetMaxCo, scheduleTime};
    if (!startRequest(s, Op::computeGasCourant, sizeof(args))
     || !sendObject(s->fd, args)) return -1;
    std::uint64_t bytes = 0;
    const int status = receiveResponse(s, bytes, true, sizeof(double));
    if (status != 0) return status;
    return readAll(s->fd, maxCo, sizeof(double)) ? 0 : -1;
}

extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly
(void* handle, double dt, double simulationTime)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (!startRequest(s, Op::advanceGasOnly, sizeof(AdvanceArgs))
     || !sendObject(s->fd, AdvanceArgs{dt, simulationTime})) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadFields
(
    void* handle, double* rho, double* rhoUx, double* rhoUy, double* rhoUz,
    double* rhoE, double* Ux, double* Uy, double* Uz, double* p,
    double* Tgas, double* epsS, double* rhoUsx, double* rhoUsy,
    double* rhoUsz, double* rhoEs, double* rhoDs, double* rhoHp,
    double* Usx, double* Usy, double* Usz, double* theta, double* Tp,
    double* dMeanCell
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const std::uint64_t bytes = std::uint64_t(s ? s->nCells : 0)*23*sizeof(double);
    if (!startRequest(s, Op::downloadFields, 0)) return -1;
    std::uint64_t got = 0;
    const int status = receiveResponse(s, got, true, bytes);
    if (status != 0) return status;
    double* fields[] = {rho,rhoUx,rhoUy,rhoUz,rhoE,Ux,Uy,Uz,p,Tgas,epsS,
        rhoUsx,rhoUsy,rhoUsz,rhoEs,rhoDs,rhoHp,Usx,Usy,Usz,theta,Tp,
        dMeanCell};
    for (double* values : fields)
        if (!receiveArray(s->fd, values, s->nCells)) return -1;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadGasBoundaryFields
(
    void* handle, double* rho, double* ux, double* uy, double* uz,
    double* p, double* t
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const std::uint64_t bytes = std::uint64_t(s ? s->nFaces : 0)*6*sizeof(double);
    if (!startRequest(s, Op::downloadGasBoundaryFields, 0)) return -1;
    std::uint64_t got = 0;
    const int status = receiveResponse(s, got, true, bytes);
    if (status != 0) return status;
    double* fields[] = {rho,ux,uy,uz,p,t};
    for (double* values : fields)
        if (!receiveArray(s->fd, values, s->nFaces)) return -1;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror
(
    void* handle, int count, const double* px, const double* py,
    const double* pz, const double* pux, const double* puy,
    const double* puz, const double* pT, const double* pTheta,
    const double* pd, const double* pm, const int* pCellId,
    const int* pStatus, const unsigned char* pStuck,
    const int* pStuckFaceId, const float* pDepositionArea,
    const float* pContactDuration, const float* pContactMaximumArea,
    const float* pContactPeakFraction,
    const float* pColdNodeSpecificEnthalpy,
    const float* pColdRingSolidMass,
    const float* pColdFrozenArea,
    const float* pColdContactAge,
    const float* pCold2DNodeSpecificEnthalpy,
    const float* pCold2DRingContactAge,
    const float* pCold2DFrozenArea,
    const unsigned long long* pRng, const unsigned long long* pOrigId
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || count < 0 || count > s->particleCapacity)
        return fail("invalid particle restart upload count");
    std::uint64_t bytes = sizeof(CountArgs);
    if (!addArrays(bytes, count, sizeof(double), 10)
     || !addArrays(bytes, count, sizeof(int), 3)
     || !addArrays(bytes, count, sizeof(unsigned char), 1)
     || !addArrays(bytes, count, sizeof(float), 95)
     || !addArrays(bytes, count, sizeof(unsigned long long), 2)
     || !startRequest(s, Op::uploadParticleRestartMirror, bytes)
     || !sendObject(s->fd, CountArgs{count})) return -1;
    const double* doubles[] = {px,py,pz,pux,puy,puz,pT,pTheta,pd,pm};
    for (const double* values : doubles)
        if (!sendArray(s->fd, values, count)) return -1;
    if (!sendArray(s->fd, pCellId, count)
     || !sendArray(s->fd, pStatus, count)
     || !sendArray(s->fd, pStuck, count)
     || !sendArray(s->fd, pStuckFaceId, count)
     || !sendArray(s->fd, pDepositionArea, count)
     || !sendArray(s->fd, pContactDuration, count)
     || !sendArray(s->fd, pContactMaximumArea, count)
     || !sendArray(s->fd, pContactPeakFraction, count)
     || !sendArray(s->fd, pColdNodeSpecificEnthalpy, count*8)
     || !sendArray(s->fd, pColdRingSolidMass, count*8)
     || !sendArray(s->fd, pColdFrozenArea, count)
     || !sendArray(s->fd, pColdContactAge, count)
     || !sendArray(s->fd, pCold2DNodeSpecificEnthalpy, count*64)
     || !sendArray(s->fd, pCold2DRingContactAge, count*8)
     || !sendArray(s->fd, pCold2DFrozenArea, count)
     || !sendArray(s->fd, pRng, count)
     || !sendArray(s->fd, pOrigId, count)) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadParticleRestartMirror
(
    void* handle, int* nParticles, int maxParticles, double* px, double* py,
    double* pz, double* pux, double* puy, double* puz, double* pT,
    double* pTheta, double* pd, double* pm, int* pCellId, int* pStatus,
    unsigned char* pStuck, int* pStuckFaceId, float* pDepositionArea,
    float* pContactDuration, float* pContactMaximumArea,
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
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || nParticles == nullptr || maxParticles < 0
     || maxParticles > s->particleCapacity)
        return fail("invalid particle restart download arguments");
    if (!startRequest(s, Op::downloadParticleRestartMirror, sizeof(CountArgs))
     || !sendObject(s->fd, CountArgs{maxParticles})) return -1;
    std::uint64_t payload = 0;
    const int status = receiveResponse(s, payload, false, 0);
    if (status != 0) return status;
    int count = 0;
    if (payload < sizeof(count) || !readAll(s->fd, &count, sizeof(count))) return -1;
    std::uint64_t perParticle = 0;
    addArrays(perParticle, 1, sizeof(double), 10);
    addArrays(perParticle, 1, sizeof(int), 3);
    addArrays(perParticle, 1, sizeof(unsigned char), 1);
    addArrays(perParticle, 1, sizeof(float), 95);
    addArrays(perParticle, 1, sizeof(unsigned long long), 2);
    const std::uint64_t expected = sizeof(count)
      + (maxParticles == 0 ? 0 : std::uint64_t(count)*perParticle);
    if (count < 0 || count > s->particleCapacity
     || (maxParticles != 0 && count > maxParticles) || payload != expected)
        return fail("invalid particle restart response");
    *nParticles = count;
    if (maxParticles == 0) return 0;
    double* doubles[] = {px,py,pz,pux,puy,puz,pT,pTheta,pd,pm};
    for (double* values : doubles)
        if (!receiveArray(s->fd, values, count)) return -1;
    if (!receiveArray(s->fd, pCellId, count)
     || !receiveArray(s->fd, pStatus, count)
     || !receiveArray(s->fd, pStuck, count)
     || !receiveArray(s->fd, pStuckFaceId, count)
     || !receiveArray(s->fd, pDepositionArea, count)
     || !receiveArray(s->fd, pContactDuration, count)
     || !receiveArray(s->fd, pContactMaximumArea, count)
     || !receiveArray(s->fd, pContactPeakFraction, count)
     || !receiveArray(s->fd, pColdNodeSpecificEnthalpy, count*8)
     || !receiveArray(s->fd, pColdRingSolidMass, count*8)
     || !receiveArray(s->fd, pColdFrozenArea, count)
     || !receiveArray(s->fd, pColdContactAge, count)
     || !receiveArray(s->fd, pCold2DNodeSpecificEnthalpy, count*64)
     || !receiveArray(s->fd, pCold2DRingContactAge, count*8)
     || !receiveArray(s->fd, pCold2DFrozenArea, count)
     || !receiveArray(s->fd, pRng, count)
     || !receiveArray(s->fd, pOrigId, count)) return -1;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev
(void* handle, double* epsGPrev)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const std::uint64_t bytes = std::uint64_t(s ? s->nCells : 0)*sizeof(double);
    if (!startRequest(s, Op::downloadEpsGPrev, 0)) return -1;
    std::uint64_t got = 0;
    const int status = receiveResponse(s, got, true, bytes);
    if (status != 0) return status;
    return receiveArray(s->fd, epsGPrev, s->nCells) ? 0 : -1;
}

extern "C" int ugkwpGpuResidentStrictUploadEpsGPrev
(void* handle, const double* epsGPrev)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const std::uint64_t bytes = std::uint64_t(s ? s->nCells : 0)*sizeof(double);
    if (!startRequest(s, Op::uploadEpsGPrev, bytes)
     || !sendArray(s->fd, epsGPrev, s->nCells)) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadNut
(void* handle, double* nut)
{
    ClientState* s = static_cast<ClientState*>(handle);
    const std::uint64_t bytes = std::uint64_t(s ? s->nCells : 0)*sizeof(double);
    if (!startRequest(s, Op::downloadNut, 0)) return -1;
    std::uint64_t got = 0;
    const int status = receiveResponse(s, got, true, bytes);
    if (status != 0) return status;
    return receiveArray(s->fd, nut, s->nCells) ? 0 : -1;
}

extern "C" int ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers
(
    void* handle,
    int nFaces,
    double* depositedWallEnergyJ,
    double* reflectedWallEnergyJ
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if
    (
        s == nullptr
     || nFaces != s->nFaces
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
    )
    {
        return fail("invalid particle-wall heat-ledger download request");
    }
    const std::uint64_t bytes =
        2ULL*std::uint64_t(nFaces)*sizeof(double);
    if
    (
        !startRequest(s, Op::downloadAndResetParticleWallHeatLedgers, 0)
    )
    {
        return -1;
    }
    std::uint64_t got = 0;
    const int status = receiveResponse(s, got, true, bytes);
    if (status != 0)
    {
        return status;
    }
    if (!receiveArray(s->fd, depositedWallEnergyJ, nFaces))
    {
        return -1;
    }
    return receiveArray(s->fd, reflectedWallEnergyJ, nFaces) ? 0 : -1;
}

extern "C" void ugkwpGpuResidentStrictRelease(void* handle)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr) return;
    if (s->fd >= 0)
    {
        if (startRequest(s, Op::release, 0))
        {
            (void)finishNoPayload(s);
        }
        ::close(s->fd);
        s->fd = -1;
    }
    if (s->childPid > 0)
    {
        int childStatus = 0;
        while (::waitpid(s->childPid, &childStatus, 0) < 0 && errno == EINTR) {}
        s->childPid = -1;
    }
    delete s;
}
