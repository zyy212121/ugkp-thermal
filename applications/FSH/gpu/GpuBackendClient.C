#include "../../../common/DiagnosticsCodec.H"
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
#include <algorithm>

namespace
{

using namespace ugkwpGpuIpc;

struct ClientState
{
    int fd = -1;
    bool terminalDiagnostic = false;
    std::uint64_t diagnosticSequence=0;
    std::int64_t diagnosticStep=-1;
    double diagnosticTime=0;
    ugkp::diagnostics::Snapshot diagnostics;
    ugkp::diagnostics::HostSink diagnosticSink;
    bool responseActive = false;
    std::uint64_t responseRemaining = 0;
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

bool readAll(const int fd, void* data, std::uint64_t bytes, ClientState* state = nullptr)
{
    if (state && state->responseActive && bytes>state->responseRemaining)
    {
        lastError="short response field: needed "+std::to_string(bytes)+" available "+std::to_string(state->responseRemaining);
        return false;
    }
    const auto requested=bytes;
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
            lastError = "GPU backend exited or closed its IPC channel: needed "+std::to_string(requested)+" received "+std::to_string(requested-bytes);
            return false;
        }
        if (got < 0)
        {
            lastError = std::string("GPU backend IPC read failed: ")
                      + std::strerror(errno);
            return false;
        }
        if (state && state->responseActive) state->responseRemaining-=static_cast<std::uint64_t>(got);
        p += got;
        bytes -= static_cast<std::uint64_t>(got);
    }
    return true;
}

template<class T>
bool sendObject(const int fd, const T& value)
{
    if (wireLittleEndian() && WireCodec<T>::nativeLayout)
        return writeAll(fd,&value,WireCodec<T>::size);
    const auto bytes=encodeWire(value);
    return writeAll(fd, bytes.data(), bytes.size());
}

template<class T> bool receiveObject(const int fd,T& value,ClientState* state = nullptr)
{
    if constexpr (std::is_same<T,ResponseHeader>::value) {
        std::array<unsigned char,WireCodec<T>::size> bytes{};
        if(!readAll(fd,bytes.data(),8,state))return false;
        std::uint32_t receivedMagic;std::uint16_t major,minor;
        decodeWireScalar(bytes.data(),receivedMagic);decodeWireScalar(bytes.data()+4,major);decodeWireScalar(bytes.data()+6,minor);
        if(receivedMagic!=magic || major!=protocolMajor || minor!=protocolMinor) {
            lastError="GPU backend response framing/version mismatch: expected "+std::to_string(protocolMajor)+"."+std::to_string(protocolMinor)
                +", received "+std::to_string(major)+"."+std::to_string(minor);
            return false;
        }
        if(!readAll(fd,bytes.data()+8,bytes.size()-8,state))return false;
        WireCodec<T>::decode(bytes.data(),value);return true;
    }

    if (wireLittleEndian() && WireCodec<T>::nativeLayout)
        return readAll(fd,&value,WireCodec<T>::size,state);
    std::array<unsigned char,WireCodec<T>::size> bytes{};
    if (!readAll(fd,bytes.data(),bytes.size(),state)) return false;
    WireCodec<T>::decode(bytes.data(),value); return true;
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
    if (wireLittleEndian() || sizeof(T)==1) return writeAll(fd,values,count*sizeof(T));
    for (std::uint64_t i=0;i<count;++i)
    { const auto bytes=encodeWire(values[i]); if (!writeAll(fd,bytes.data(),bytes.size())) return false; }
    return true;
}

template<class T>
bool receiveArray(const int fd, T* values, const std::uint64_t count, ClientState* state = nullptr)
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
    if (!readAll(fd,values,count*sizeof(T),state)) return false;
    if (!wireLittleEndian() && sizeof(T)!=1)
        for (std::uint64_t i=0;i<count;++i)
        { auto* bytes=reinterpret_cast<unsigned char*>(values+i); std::reverse(bytes,bytes+sizeof(T)); }
    return true;
}

bool discardResponse(ClientState* state)
{
    if (!state->responseActive) return true;
    char buffer[8192];
    while (state->responseRemaining)
    {
        const auto n=std::min<std::uint64_t>(state->responseRemaining,sizeof(buffer));
        if (!readAll(state->fd,buffer,n,state)) return false;
    }
    state->responseActive=false;
    return true;
}

bool startRequest(ClientState* state, const Op op, const std::uint64_t bytes)
{
    if (state == nullptr || state->fd < 0)
    {
        lastError = "invalid GPU backend client handle";
        return false;
    }
    if (state->terminalDiagnostic && op!=Op::release) {lastError="GPU backend has a fatal diagnostic";return false;}
    if (!discardResponse(state)) return false;
    const RequestHeader header
    {
        magic, protocolMajor, protocolMinor,
        static_cast<std::uint32_t>(op), 0U, bytes,
        state->diagnosticSequence,state->diagnosticStep,state->diagnosticTime
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
    if (!receiveObject(state->fd, response))
    {
        return -1;
    }
    std::string backendError(response.errorBytes, '\0');
    if (response.errorBytes != 0 && !readAll(state->fd, &backendError[0], response.errorBytes))
    {
        return -1;
    }
    std::vector<std::uint8_t> diagnosticBytes(response.diagnosticBytes);
    if(!diagnosticBytes.empty() && !readAll(state->fd,diagnosticBytes.data(),diagnosticBytes.size())) return -1;
    state->diagnostics=ugkp::diagnostics::Snapshot{};
    if(!diagnosticBytes.empty()) {
        std::string decodeError;
        if(!ugkp::diagnostics::decode(diagnosticBytes.data(),diagnosticBytes.size(),state->diagnostics,decodeError))
            return fail("GPU backend diagnostic envelope: "+decodeError);
    }
    state->terminalDiagnostic=state->diagnostics.hasFatal;
    if(state->diagnosticSink.fatal || state->diagnosticSink.warning)
        ugkp::diagnostics::dispatch(state->diagnostics,state->diagnosticSink);
    state->responseRemaining=response.payloadBytes;
    state->responseActive=true;
    if (response.status != 0 || state->terminalDiagnostic)
    {
        if (!discardResponse(state)) return -1;
        lastError = backendError.empty()
          ? "GPU backend returned an unspecified error"
          : backendError;
        return response.status ? response.status : -1;
    }
    if (requireExact && response.payloadBytes < expectedBytes)
    {
        return fail("short GPU backend response: needed "+std::to_string(expectedBytes)+" available "+std::to_string(response.payloadBytes));
    }
    payloadBytes = response.payloadBytes;
    lastError = "no GPU backend client error";
    return 0;
}

int finishNoPayload(ClientState* state)
{
    std::uint64_t bytes = 0;
    const int status=receiveResponse(state, bytes, true, 0);
    return status ? status : (discardResponse(state)?0:-1);
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
    state.responseActive=false;
    state.responseRemaining=0;
    state.childPid = child;
    return true;
}

void addArrays(std::uint64_t& total, const std::uint64_t count,
               const std::uint64_t elementBytes, const unsigned copies)
{
    for (unsigned i = 0; i < copies; ++i)
        addBytes(total, count, elementBytes);
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
    if (handle == nullptr) return fail("missing backend output handle");
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
    if (!startRequest(state, Op::create, wireBytes(args)) || !sendObject(state->fd, args))
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
    addArrays(bytes, s->nFaces, sizeof(int), 3);
    addArrays(bytes, s->nCells, sizeof(double), 5);
    addArrays(bytes, s->nFaces, sizeof(double), 12);
    addArrays(bytes, s->nCells, sizeof(int), 2);
    addArrays(bytes, s->nCellPlanes, sizeof(int), 3);
    addArrays(bytes, s->nCellPlanes, sizeof(double), 6);
    if (!startRequest(s, Op::uploadMesh, bytes)) return -1;
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
    if (s == nullptr)
        return fail("invalid GPU boundary source count");
    std::uint64_t bytes = WireCodec<CountArgs>::size;
    addArrays(bytes, count, sizeof(int), 2);
    addArrays(bytes, count, sizeof(double), 10);
    if (!startRequest(s, Op::uploadBoundarySources, bytes) || !sendObject(s->fd, CountArgs{count})) return -1;
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
    if (!s || (nFaces && !candidateFaceMask))
        return fail("missing stuck-model handle or face mask extent");
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
        wireBytes(args) + std::uint64_t(nFaces)*sizeof(unsigned char);
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
    if (s == nullptr)
        return fail("invalid scheduled inlet configuration counts");
    const ScheduledInletConfigArgs args
    {
        count, nPressureRows, nVolumeFractionRows, inletTemperature
    };
    std::uint64_t bytes = wireBytes(args);
    addArrays(bytes, count, sizeof(int), 1);
    addArrays(bytes, nPressureRows, sizeof(double), 2);
    addArrays(bytes, nVolumeFractionRows, sizeof(double), 2);
    if (!startRequest(s, Op::configureScheduledInlet, bytes)
     || !sendObject(s->fd, args)
     || !sendArray(s->fd, faceIds, count)
     || !sendArray(s->fd, pressureTimes, nPressureRows)
     || !sendArray(s->fd, pressureValues, nPressureRows)
     || !sendArray(s->fd, volumeFractionTimes, nVolumeFractionRows)
     || !sendArray(s->fd, volumeFractionValues, nVolumeFractionRows)) return -1;
    return finishNoPayload(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass
(
    void* handle, int* nSources, int maxSources, int* sourceFace,
    double* residualMass
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || nSources == nullptr)
        return fail("invalid source residual download arguments");
    if (!startRequest(s, Op::downloadSourceResidualMass, WireCodec<CountArgs>::size)
     || !sendObject(s->fd, CountArgs{maxSources})) return -1;
    std::uint64_t payload = 0;
    const int status = receiveResponse(s, payload, false, 0);
    if (status != 0) return status;
    int count = 0;
    if (payload < wireBytes(count) || !receiveObject(s->fd, count,s)) return -1;
    const std::uint64_t expected = wireBytes(count)
      + (maxSources == 0 ? 0 : std::uint64_t(count)*(sizeof(int) + sizeof(double)));
    if (payload < expected)
        return fail("invalid source residual response");
    *nSources = count;
    if (maxSources != 0)
    {
        if (!receiveArray(s->fd, sourceFace, count,s)
         || !receiveArray(s->fd, residualMass, count,s)) return -1;
    }
    return discardResponse(s)?0:-1;
}

extern "C" int ugkwpGpuResidentStrictUploadSourceResidualMass
(
    void* handle, int count, const int* sourceFace, const double* residualMass
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr)
        return fail("invalid source residual upload count");
    const std::uint64_t bytes = WireCodec<CountArgs>::size
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
    addArrays(bytes, s->nFaces, sizeof(int), 6);
    addArrays(bytes, s->nFaces, sizeof(double), 9);
    if (!startRequest(s, Op::uploadGasBoundaryFields, bytes)) return -1;
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
    addArrays(bytes, s->nCells, sizeof(double), 10);
    if (!startRequest(s, Op::uploadFields, bytes)) return -1;
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
    if (!startRequest(s, Op::advance, wireBytes(args)) || !sendObject(s->fd, args))
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
    std::uint64_t bytes = wireBytes(args);
    addArrays(bytes, s->nCells, sizeof(double), 3);
    addArrays(bytes, s->nFaces, sizeof(int), 2);
    addArrays(bytes, s->nFaces, sizeof(double), 2);
    if (!startRequest(s, Op::configureSst, bytes)
     || !sendObject(s->fd, args)
     || !sendArray(s->fd, k, s->nCells)
     || !sendArray(s->fd, omega, s->nCells)
     || !sendArray(s->fd, wallDistance, s->nCells)
     || !sendArray(s->fd, boundaryKMode, s->nFaces)
     || !sendArray(s->fd, boundaryOmegaMode, s->nFaces)
     || !sendArray(s->fd, boundaryK, s->nFaces)
     || !sendArray(s->fd, boundaryOmega, s->nFaces))
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
        !receiveArray(s->fd, k, s->nCells,s)
     || !receiveArray(s->fd, omega, s->nCells,s)
     || !receiveArray(s->fd, nut, s->nCells,s)
    )
    {
        return -1;
    }
    return discardResponse(s)?0:-1;
}

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(void* handle, double dt, double targetMaxCo, double scheduleTime, double* maxCo)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (maxCo == nullptr) return fail("null gas Courant output");
    const CourantArgs args{dt, targetMaxCo, scheduleTime};
    if (!startRequest(s, Op::computeGasCourant, wireBytes(args))
     || !sendObject(s->fd, args)) return -1;
    std::uint64_t bytes = 0;
    const int status = receiveResponse(s, bytes, true, sizeof(double));
    if (status != 0) return status;
    return receiveArray(s->fd,maxCo,1,s) && discardResponse(s) ? 0 : -1;
}

extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly
(void* handle, double dt, double simulationTime)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (!startRequest(s, Op::advanceGasOnly, WireCodec<AdvanceArgs>::size)
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
        if (!receiveArray(s->fd, values, s->nCells,s)) return -1;
    return discardResponse(s)?0:-1;
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
        if (!receiveArray(s->fd, values, s->nFaces,s)) return -1;
    return discardResponse(s)?0:-1;
}

extern "C" int ugkwpGpuResidentStrictUploadParticleSavedVelocity
(void* handle, int count, const double* x, const double* y, const double* z)
{
    ClientState* s=static_cast<ClientState*>(handle);
    if (!s || count<0 || count>s->particleCapacity || (count && (!x || !y || !z)))
        return fail("invalid saved velocity upload");
    std::uint64_t bytes=WireCodec<CountArgs>::size;
    addArrays(bytes,count,sizeof(double),3);
    if (!startRequest(s,Op::uploadParticleSavedVelocity,bytes)
     || !sendObject(s->fd,CountArgs{count})
     || !sendArray(s->fd,x,count)
     || !sendArray(s->fd,y,count)
     || !sendArray(s->fd,z,count)) return -1;
    return finishNoPayload(s);
}
extern "C" int ugkwpGpuResidentStrictDownloadParticleSavedVelocity
(void* handle, int count, double* x, double* y, double* z)
{
    ClientState* s=static_cast<ClientState*>(handle);
    if (!s || count<0 || count>s->particleCapacity || (count && (!x || !y || !z)))
        return fail("invalid saved velocity download");
    if (!startRequest(s,Op::downloadParticleSavedVelocity,WireCodec<CountArgs>::size)
        || !sendObject(s->fd,CountArgs{count})) return -1;
    std::uint64_t bytes=0,expected=0;addArrays(expected,count,sizeof(double),3);
    const int status=receiveResponse(s,bytes,true,expected);
    if (status!=0) return status;
    return receiveArray(s->fd,x,count,s) && receiveArray(s->fd,y,count,s)
        && receiveArray(s->fd,z,count,s) && discardResponse(s) ? 0 : -1;
}

extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror
(
    void* handle, int count, const double* px, const double* py,
    const double* pz, const double* pux, const double* puy,
    const double* puz, const double* pT, const double* pTheta,
    const double* pd, const double* pm, const int* pCellId,
    const int* pStatus, const unsigned char* pStuck,
    const int* pStuckFaceId, const double* pDepositionArea,
    const double* pContactDuration, const double* pContactMaximumArea,
    const double* pContactPeakFraction,
    const double* pColdNodeSpecificEnthalpy,
    const double* pColdRingSolidMass,
    const double* pColdFrozenArea,
    const double* pColdContactAge,
    const double* pCold2DNodeSpecificEnthalpy,
    const double* pCold2DRingContactAge,
    const double* pCold2DFrozenArea,
    const unsigned long long* pRng, const unsigned long long* pOrigId
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr)
        return fail("invalid particle restart upload count");
    std::uint64_t bytes = WireCodec<CountArgs>::size;
    addArrays(bytes, count, sizeof(double), 10);
    addArrays(bytes, count, sizeof(int), 3);
    addArrays(bytes, count, sizeof(unsigned char), 1);
    addArrays(bytes, count, sizeof(double), 95);
    addArrays(bytes, count, sizeof(unsigned long long), 2);
    if (!startRequest(s, Op::uploadParticleRestartMirror, bytes) || !sendObject(s->fd, CountArgs{count})) return -1;
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
     || !sendArray(s->fd, pColdNodeSpecificEnthalpy, std::uint64_t(count)*8)
     || !sendArray(s->fd, pColdRingSolidMass, std::uint64_t(count)*8)
     || !sendArray(s->fd, pColdFrozenArea, count)
     || !sendArray(s->fd, pColdContactAge, count)
     || !sendArray(s->fd, pCold2DNodeSpecificEnthalpy, std::uint64_t(count)*64)
     || !sendArray(s->fd, pCold2DRingContactAge, std::uint64_t(count)*8)
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
    unsigned char* pStuck, int* pStuckFaceId, double* pDepositionArea,
    double* pContactDuration, double* pContactMaximumArea,
    double* pContactPeakFraction,
    double* pColdNodeSpecificEnthalpy,
    double* pColdRingSolidMass,
    double* pColdFrozenArea,
    double* pColdContactAge,
    double* pCold2DNodeSpecificEnthalpy,
    double* pCold2DRingContactAge,
    double* pCold2DFrozenArea,
    unsigned long long* pRng,
    unsigned long long* pOrigId
)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr || nParticles == nullptr)
        return fail("invalid particle restart download arguments");
    if (!startRequest(s, Op::downloadParticleRestartMirror, WireCodec<CountArgs>::size)
     || !sendObject(s->fd, CountArgs{maxParticles})) return -1;
    std::uint64_t payload = 0;
    const int status = receiveResponse(s, payload, false, 0);
    if (status != 0) return status;
    int count = 0;
    if (payload < wireBytes(count) || !receiveObject(s->fd, count,s)) return -1;
    std::uint64_t perParticle = 0;
    addArrays(perParticle, 1, sizeof(double), 10);
    addArrays(perParticle, 1, sizeof(int), 3);
    addArrays(perParticle, 1, sizeof(unsigned char), 1);
    addArrays(perParticle, 1, sizeof(double), 95);
    addArrays(perParticle, 1, sizeof(unsigned long long), 2);
    const std::uint64_t expected = wireBytes(count)
      + (maxParticles == 0 ? 0 : std::uint64_t(count)*perParticle);
    if (payload < expected)
        return fail("invalid particle restart response");
    *nParticles = count;
    if (maxParticles == 0) return discardResponse(s)?0:-1;
    double* doubles[] = {px,py,pz,pux,puy,puz,pT,pTheta,pd,pm};
    for (double* values : doubles)
        if (!receiveArray(s->fd, values, count,s)) return -1;
    if (!receiveArray(s->fd, pCellId, count,s)
     || !receiveArray(s->fd, pStatus, count,s)
     || !receiveArray(s->fd, pStuck, count,s)
     || !receiveArray(s->fd, pStuckFaceId, count,s)
     || !receiveArray(s->fd, pDepositionArea, count,s)
     || !receiveArray(s->fd, pContactDuration, count,s)
     || !receiveArray(s->fd, pContactMaximumArea, count,s)
     || !receiveArray(s->fd, pContactPeakFraction, count,s)
     || !receiveArray(s->fd, pColdNodeSpecificEnthalpy, std::uint64_t(count)*8,s)
     || !receiveArray(s->fd, pColdRingSolidMass, std::uint64_t(count)*8,s)
     || !receiveArray(s->fd, pColdFrozenArea, count,s)
     || !receiveArray(s->fd, pColdContactAge, count,s)
     || !receiveArray(s->fd, pCold2DNodeSpecificEnthalpy, std::uint64_t(count)*64,s)
     || !receiveArray(s->fd, pCold2DRingContactAge, std::uint64_t(count)*8,s)
     || !receiveArray(s->fd, pCold2DFrozenArea, count,s)
     || !receiveArray(s->fd, pRng, count,s)
     || !receiveArray(s->fd, pOrigId, count,s)) return -1;
    return discardResponse(s)?0:-1;
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
    return receiveArray(s->fd, epsGPrev, s->nCells,s) && discardResponse(s) ? 0 : -1;
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
    return receiveArray(s->fd, nut, s->nCells,s) && discardResponse(s) ? 0 : -1;
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
     || nFaces < s->nFaces
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
    )
    {
        return fail("invalid particle-wall heat-ledger download request");
    }
    const std::uint64_t bytes =
        2ULL*std::uint64_t(s->nFaces)*sizeof(double);
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
    if (!receiveArray(s->fd, depositedWallEnergyJ, s->nFaces,s))
    {
        return -1;
    }
    return receiveArray(s->fd, reflectedWallEnergyJ, s->nFaces,s) && discardResponse(s) ? 0 : -1;
}

extern "C" void ugkwpGpuResidentStrictReleaseDiagnostics(void* handle,ugkp::diagnostics::Snapshot* snapshot)
{
    ClientState* s = static_cast<ClientState*>(handle);
    if (s == nullptr) return;
    if (s->fd >= 0)
    {
        if (startRequest(s, Op::release, 0))
        {
            (void)finishNoPayload(s);
        }
        s->responseActive=false;
        s->responseRemaining=0;
        ::close(s->fd);
        s->fd = -1;
    }
    if (s->childPid > 0)
    {
        int childStatus = 0;
        while (::waitpid(s->childPid, &childStatus, 0) < 0 && errno == EINTR) {}
        s->childPid = -1;
    }
    if(snapshot)*snapshot=s->diagnostics;
    delete s;
}

extern "C" void ugkwpGpuResidentStrictSetDiagnosticSink(void* handle,const ugkp::diagnostics::HostSink* sink)
{
    auto* state=static_cast<ClientState*>(handle);
    if(state && sink) state->diagnosticSink=*sink;
}
extern "C" int ugkwpGpuResidentStrictCollectDiagnostics(void* handle,ugkp::diagnostics::Snapshot* snapshot)
{
    auto* state=static_cast<ClientState*>(handle);
    if(!state || !snapshot)return 1;
    *snapshot=state->diagnostics;
    return snapshot->hasFatal ? 1 : 0;
}

extern "C" void ugkwpGpuResidentStrictRelease(void* handle)
{
    ugkwpGpuResidentStrictReleaseDiagnostics(handle,nullptr);
}

extern "C" int ugkwpGpuResidentStrictSetDiagnosticOperation(void* handle,std::uint64_t sequence,std::int64_t step,double time)
{
    auto* state=static_cast<ClientState*>(handle);
    if(!state)return -1;
    state->diagnosticSequence=sequence;state->diagnosticStep=step;state->diagnosticTime=time;
    return 0;
}
