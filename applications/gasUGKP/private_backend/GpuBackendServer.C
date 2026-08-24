#include "GpuBackendApi.H"
#include "GpuBackendProtocol.H"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

using namespace ugkwpGpuIpc;

struct ServerState
{
    void* backend = nullptr;
    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nCellPlanes = 0;
    int particleCapacity = 0;
};

bool writeAll(const int fd, const void* data, std::uint64_t bytes)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    while (bytes != 0)
    {
        const ssize_t wrote = ::write(fd, p, static_cast<std::size_t>(bytes));
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return false;
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
        const ssize_t got = ::read(fd, p, static_cast<std::size_t>(bytes));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        p += got;
        bytes -= static_cast<std::uint64_t>(got);
    }
    return true;
}

template<class T>
bool readObject(const int fd, T& value)
{
    return readAll(fd, &value, sizeof(T));
}

template<class T>
bool readVector(const int fd, std::vector<T>& values, const std::size_t count)
{
    values.resize(count);
    return count == 0 || readAll(fd, values.data(), count*sizeof(T));
}

template<class T>
bool writeVector(const int fd, const std::vector<T>& values)
{
    return values.empty() || writeAll(fd, values.data(), values.size()*sizeof(T));
}

bool responseHeader
(
    const int fd,
    const int status,
    const std::uint64_t payloadBytes,
    const std::string& overrideError = std::string()
)
{
    std::string error;
    if (status != 0)
    {
        if (!overrideError.empty()) error = overrideError;
        else if (const char* backendError = ugkwpGpuResidentStrictLastError())
            error = backendError;
        if (error.empty()) error = "GPU backend returned an unspecified error";
    }
    const ResponseHeader response
    {
        magic, protocolMajor, protocolMinor, status,
        static_cast<std::uint32_t>(error.size()), payloadBytes
    };
    return writeAll(fd, &response, sizeof(response))
        && (error.empty() || writeAll(fd, error.data(), error.size()));
}

bool success(const int fd)
{
    return responseHeader(fd, 0, 0);
}

bool protocolError(const int fd, const std::string& message)
{
    (void)responseHeader(fd, -1, 0, message);
    return false;
}

bool addArrays(std::uint64_t& total, const std::uint64_t count,
               const std::uint64_t elementBytes, const unsigned copies)
{
    for (unsigned i = 0; i < copies; ++i)
        if (!addBytes(total, count, elementBytes)) return false;
    return true;
}

bool validatePayload
(
    const int fd,
    const RequestHeader& request,
    const std::uint64_t expected
)
{
    return request.payloadBytes == expected
        || protocolError(fd, "GPU backend request payload size mismatch");
}

template<class T>
T* ptr(std::vector<T>& values)
{
    return values.empty() ? nullptr : values.data();
}

bool handleCreate
(
    const int fd,
    const RequestHeader& request,
    ServerState& state
)
{
    if (state.backend != nullptr)
        return protocolError(fd, "GPU backend session was already created");
    if (!validatePayload(fd, request, sizeof(CreateArgs))) return false;
    CreateArgs a{};
    if (!readObject(fd, a)) return false;
    if (a.nCells <= 0 || a.nFaces <= 0 || a.nInternalFaces < 0
     || a.nInternalFaces > a.nFaces || a.nCellPlanes < 0
     || a.particleCapacity < 0
     || !std::isfinite(a.gammaGas) || a.gammaGas <= 1.0
     || a.gammaGas > 5.0/3.0
     || !std::isfinite(a.Rgas) || a.Rgas <= 0.0
     || !std::isfinite(a.gasMu) || a.gasMu < 0.0
     || !std::isfinite(a.gasPr) || a.gasPr <= 0.0
     || a.gasFluxScheme < static_cast<int>(GasFluxScheme::RusanovTadmor)
     || a.gasFluxScheme > static_cast<int>(GasFluxScheme::Slau2_2)
     || a.gasReconstruction
        < static_cast<int>(GasReconstruction::firstOrder)
     || a.gasReconstruction
        > static_cast<int>
          (
              GasReconstruction::OpenFoamEnergyLimitedLinear
          )
     || a.gasLimiter < static_cast<int>(GasLimiter::none)
     || a.gasLimiter > static_cast<int>(GasLimiter::venkatakrishnan)
     || a.gasTimeIntegrator < static_cast<int>(GasTimeIntegrator::Euler)
     || a.gasTimeIntegrator > static_cast<int>(GasTimeIntegrator::SSPRK3)
     || (a.gasRobustFallback != 0 && a.gasRobustFallback != 1)
     || ((a.gasFluxScheme == static_cast<int>(GasFluxScheme::Hllc)
       || a.gasFluxScheme == static_cast<int>(GasFluxScheme::Roe)
       || a.gasFluxScheme == static_cast<int>(GasFluxScheme::Hllem))
      && a.gasRobustFallback == 0)
     || a.turbulenceModel < static_cast<int>(TurbulenceModel::laminar)
     || a.turbulenceModel > static_cast<int>(TurbulenceModel::kOmegaSST)
     || !std::isfinite(a.lesDeltaCoeff)
     || a.lesDeltaCoeff <= 0.0
     || !std::isfinite(a.turbulentPrandtl)
     || a.turbulentPrandtl <= 0.0
     || !std::isfinite(a.waleCw)
     || a.waleCw < 0.0
     || !std::isfinite(a.smagorinskyCs)
     || a.smagorinskyCs < 0.0
     || !std::isfinite(a.maxDiffusionNumber)
     || a.maxDiffusionNumber <= 0.0
     || (a.csrCellLocalPathEnabled != 0
      && a.csrCellLocalPathEnabled != 1)
     || a.csrHeavyReductionMode < 0
     || a.csrHeavyReductionMode > 2
     || a.csrHeavyAutoInterval < 1
     || (a.particleBlockThreads != 32 && a.particleBlockThreads != 64
      && a.particleBlockThreads != 128 && a.particleBlockThreads != 256)
     || (a.reductionBlockThreads != 32 && a.reductionBlockThreads != 64
      && a.reductionBlockThreads != 128 && a.reductionBlockThreads != 256)
     || (a.csrWarpAggregatedBinning != 0
      && a.csrWarpAggregatedBinning != 1)
     || (a.csrSplitPreDirectoryEnabled != 0
      && a.csrSplitPreDirectoryEnabled != 1)
     || (!a.csrCellLocalPathEnabled
      && (a.csrHeavyReductionMode != 0 || a.csrWarpAggregatedBinning))
     || a.dragModel < 0
     || a.dragModel > 2
     || a.particleGasHeatTransferModelId < 0
     || a.particleGasHeatTransferModelId > 1
     || !std::isfinite(a.dragParameter0)
     || !std::isfinite(a.dragParameter1)
     || !std::isfinite(a.dragParameter2)
     || !std::isfinite(a.dragParameter3)
     || (a.dragModel == 1 && a.dragParameter0 <= 0.0)
     || !std::isfinite(a.gravityX)
     || !std::isfinite(a.gravityY)
     || !std::isfinite(a.gravityZ)
     || (a.jammingPressureEnabled != 0 && a.jammingPressureEnabled != 1)
     || a.packingProjectionIterations < 1
     || (a.jammingPressureEnabled != 0
      && (!std::isfinite(a.packingFraction)
       || a.packingFraction <= 0.0 || a.packingFraction >= 1.0)))
        return protocolError(fd, "invalid GPU backend create arguments");

    const int status = ugkwpGpuResidentStrictCreate
    (
        a.nCells, a.nFaces, a.nInternalFaces, a.nCellPlanes,
        a.particleCapacity, a.maxFaceWalkHops, a.injectionParcelMass, a.rngSeed,
        a.gammaGas, a.Rgas, a.rhoSolid, a.solveParticleTemperature,
        a.particleGasHeatTransferModelId,
        a.particleThermalRho, a.particleCp, a.gasMu, a.gasPr,
        a.particleDiameterFallback, a.particleDiameterMin,
        a.particleDiameterMax, a.particleDiameterSigma, a.injectionTheta,
        a.rhoMin, a.TgasMin, a.epsSMin, a.thetaMin, a.TpMin, a.TpMax,
        a.collisionalPressureEnabled, a.collisionalRestitution,
        a.pressureKickFraction, a.jammingPressureEnabled,
        a.packingFraction, a.packingProjectionIterations,
        a.gasFluxScheme, a.gasReconstruction,
        a.gasLimiter, a.gasTimeIntegrator, a.gasRobustFallback,
        a.turbulenceModel, a.lesDeltaCoeff,
        a.turbulentPrandtl, a.waleCw, a.smagorinskyCs,
        a.maxDiffusionNumber, a.csrCellLocalPathEnabled,
        a.csrHeavyReductionMode, a.csrHeavyAutoInterval,
        a.particleBlockThreads, a.reductionBlockThreads,
        a.csrWarpAggregatedBinning,
        a.csrSplitPreDirectoryEnabled,
        a.dragModel,
        a.dragParameter0, a.dragParameter1,
        a.dragParameter2, a.dragParameter3,
        a.gravityX, a.gravityY, a.gravityZ,
        &state.backend
    );
    if (status == 0)
    {
        state.nCells = a.nCells;
        state.nFaces = a.nFaces;
        state.nInternalFaces = a.nInternalFaces;
        state.nCellPlanes = a.nCellPlanes;
        state.particleCapacity = a.particleCapacity;
    }
    return responseHeader(fd, status, 0);
}

bool handleMesh(const int fd, const RequestHeader& request, ServerState& s)
{
    std::uint64_t expected = 0;
    if (!addArrays(expected, s.nFaces, sizeof(int), 3)
     || !addArrays(expected, s.nCells, sizeof(double), 5)
     || !addArrays(expected, s.nFaces, sizeof(double), 12)
     || !addArrays(expected, s.nCells, sizeof(int), 2)
     || !addArrays(expected, s.nCellPlanes, sizeof(int), 3)
     || !addArrays(expected, s.nCellPlanes, sizeof(double), 6)
     || !validatePayload(fd, request, expected)) return false;

    std::vector<int> owner, neighbour, facePeriodicPair, planeStart, planeCount,
        faceId, faceNeighbor, faceKind;
    std::vector<double> facePeriodicDx,facePeriodicDy,facePeriodicDz,V,Cx,Cy,Cz,faceCx,
        faceCy,faceCz,Sfx,Sfy,Sfz,magSf,delta,weight,cellLength,restitution,
        tangential,nx,ny,nz,d;
#define READ(VEC,N) do { if (!readVector(fd, (VEC), (N))) return false; } while (false)
    READ(owner,s.nFaces); READ(neighbour,s.nFaces);
    READ(facePeriodicPair,s.nFaces); READ(facePeriodicDx,s.nFaces);
    READ(facePeriodicDy,s.nFaces); READ(facePeriodicDz,s.nFaces); READ(V,s.nCells);
    READ(Cx,s.nCells); READ(Cy,s.nCells); READ(Cz,s.nCells);
    READ(faceCx,s.nFaces); READ(faceCy,s.nFaces); READ(faceCz,s.nFaces);
    READ(Sfx,s.nFaces); READ(Sfy,s.nFaces); READ(Sfz,s.nFaces);
    READ(magSf,s.nFaces); READ(delta,s.nFaces); READ(weight,s.nFaces);
    READ(cellLength,s.nCells); READ(planeStart,s.nCells);
    READ(planeCount,s.nCells); READ(faceId,s.nCellPlanes);
    READ(faceNeighbor,s.nCellPlanes); READ(faceKind,s.nCellPlanes);
    READ(restitution,s.nCellPlanes); READ(tangential,s.nCellPlanes);
    READ(nx,s.nCellPlanes); READ(ny,s.nCellPlanes); READ(nz,s.nCellPlanes);
    READ(d,s.nCellPlanes);
#undef READ
    const int status = ugkwpGpuResidentStrictUploadMesh
    (
        s.backend, ptr(owner), ptr(neighbour), ptr(facePeriodicPair),
        ptr(facePeriodicDx), ptr(facePeriodicDy), ptr(facePeriodicDz),
        ptr(V), ptr(Cx), ptr(Cy), ptr(Cz),
        ptr(faceCx), ptr(faceCy), ptr(faceCz), ptr(Sfx), ptr(Sfy), ptr(Sfz),
        ptr(magSf), ptr(delta), ptr(weight), ptr(cellLength), ptr(planeStart),
        ptr(planeCount), ptr(faceId), ptr(faceNeighbor), ptr(faceKind),
        ptr(restitution), ptr(tangential), ptr(nx), ptr(ny), ptr(nz), ptr(d)
    );
    return responseHeader(fd, status, 0);
}

bool handleBoundarySources
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    CountArgs a{};
    if (request.payloadBytes < sizeof(a) || !readObject(fd, a)) return false;
    if (!validCount(a.count, s.nFaces))
        return protocolError(fd, "invalid GPU boundary source count");
    std::uint64_t expected = sizeof(a);
    if (!addArrays(expected,a.count,sizeof(int),2)
     || !addArrays(expected,a.count,sizeof(double),10)
     || !validatePayload(fd,request,expected)) return false;
    std::vector<int> cell,face;
    std::vector<double> px,py,pz,ux,uy,uz,t,theta,d,massRate;
#define READ(VEC) do { if (!readVector(fd,(VEC),a.count)) return false; } while(false)
    READ(cell); READ(face); READ(px); READ(py); READ(pz); READ(ux); READ(uy);
    READ(uz); READ(t); READ(theta); READ(d); READ(massRate);
#undef READ
    const int status = ugkwpGpuResidentStrictUploadBoundarySources
    (
        s.backend,a.count,ptr(cell),ptr(face),ptr(px),ptr(py),ptr(pz),ptr(ux),
        ptr(uy),ptr(uz),ptr(t),ptr(theta),ptr(d),ptr(massRate)
    );
    return responseHeader(fd,status,0);
}

bool handleScheduledInletConfiguration
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    ScheduledInletConfigArgs a{};
    if (request.payloadBytes < sizeof(a) || !readObject(fd,a)) return false;
    if
    (
        a.count <= 0 || !validCount(a.count,s.nFaces)
     || a.nPressureRows <= 0 || a.nVolumeFractionRows <= 0
    )
        return protocolError(fd,"invalid scheduled inlet configuration count");
    std::uint64_t expected = sizeof(a);
    if
    (
        !addArrayBytes<int>(expected,a.count)
     || !addArrayBytes<double>(expected,2ULL*a.nPressureRows)
     || !addArrayBytes<double>(expected,2ULL*a.nVolumeFractionRows)
    ) return protocolError(fd,"scheduled inlet configuration is too large");
    if (!validatePayload(fd,request,expected)) return false;
    std::vector<int> faces;
    std::vector<double> pressureTimes, pressureValues;
    std::vector<double> volumeFractionTimes, volumeFractionValues;
    if
    (
        !readVector(fd,faces,a.count)
     || !readVector(fd,pressureTimes,a.nPressureRows)
     || !readVector(fd,pressureValues,a.nPressureRows)
     || !readVector(fd,volumeFractionTimes,a.nVolumeFractionRows)
     || !readVector(fd,volumeFractionValues,a.nVolumeFractionRows)
    ) return false;
    const int status = ugkwpGpuResidentStrictConfigureScheduledInlet
    (
        s.backend,a.count,ptr(faces),a.inletTemperature,
        a.nPressureRows,ptr(pressureTimes),ptr(pressureValues),
        a.nVolumeFractionRows,ptr(volumeFractionTimes),
        ptr(volumeFractionValues)
    );
    return responseHeader(fd,status,0);
}

bool handleDownloadResidual
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    CountArgs a{};
    if (!validatePayload(fd,request,sizeof(a)) || !readObject(fd,a)) return false;
    if (!validCount(a.count,s.nFaces))
        return protocolError(fd,"invalid source residual capacity");
    int count = 0;
    std::vector<int> faces(a.count);
    std::vector<double> residual(a.count);
    const int status = ugkwpGpuResidentStrictDownloadSourceResidualMass
    (s.backend,&count,a.count,ptr(faces),ptr(residual));
    if (status != 0) return responseHeader(fd,status,0);
    if (count < 0 || (a.count != 0 && count > a.count))
        return protocolError(fd,"backend returned an invalid residual count");
    const std::uint64_t bytes = sizeof(count)
      +(a.count==0?0:std::uint64_t(count)*(sizeof(int)+sizeof(double)));
    if (!responseHeader(fd,0,bytes) || !writeAll(fd,&count,sizeof(count))) return false;
    if (a.count != 0)
    {
        faces.resize(count); residual.resize(count);
        return writeVector(fd,faces) && writeVector(fd,residual);
    }
    return true;
}

bool handleUploadResidual
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    CountArgs a{};
    if (request.payloadBytes < sizeof(a) || !readObject(fd,a)) return false;
    if (!validCount(a.count,s.nFaces))
        return protocolError(fd,"invalid source residual count");
    const std::uint64_t expected=sizeof(a)+std::uint64_t(a.count)*(sizeof(int)+sizeof(double));
    if (!validatePayload(fd,request,expected)) return false;
    std::vector<int> faces; std::vector<double> residual;
    if (!readVector(fd,faces,a.count) || !readVector(fd,residual,a.count)) return false;
    const int status=ugkwpGpuResidentStrictUploadSourceResidualMass
    (s.backend,a.count,ptr(faces),ptr(residual));
    return responseHeader(fd,status,0);
}

bool handleGasBoundary
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    std::uint64_t expected=0;
    if (!addArrays(expected,s.nFaces,sizeof(int),6)
     || !addArrays(expected,s.nFaces,sizeof(double),9)
     || !validatePayload(fd,request,expected)) return false;
    std::vector<int> kind,rhoFix,uFix,pFix,tFix,pWave;
    std::vector<double> gamma,fieldInf,lInf,rho,ux,uy,uz,p,t;
#define READ(VEC) do { if(!readVector(fd,(VEC),s.nFaces)) return false; } while(false)
    READ(kind); READ(rhoFix); READ(uFix); READ(pFix); READ(tFix); READ(pWave);
    READ(gamma); READ(fieldInf); READ(lInf); READ(rho); READ(ux); READ(uy);
    READ(uz); READ(p); READ(t);
#undef READ
    const int status=ugkwpGpuResidentStrictUploadGasBoundaryFields
    (s.backend,ptr(kind),ptr(rhoFix),ptr(uFix),ptr(pFix),ptr(tFix),ptr(pWave),
     ptr(gamma),ptr(fieldInf),ptr(lInf),ptr(rho),ptr(ux),ptr(uy),ptr(uz),ptr(p),ptr(t));
    return responseHeader(fd,status,0);
}

bool handleUploadFields
(
    const int fd, const RequestHeader& request, ServerState& s
)
{
    const std::uint64_t expected=std::uint64_t(s.nCells)*10*sizeof(double);
    if (!validatePayload(fd,request,expected)) return false;
    std::vector<std::vector<double>> f(10);
    for(auto& v:f) if(!readVector(fd,v,s.nCells)) return false;
    const int status=ugkwpGpuResidentStrictUploadFields
    (s.backend,ptr(f[0]),ptr(f[1]),ptr(f[2]),ptr(f[3]),ptr(f[4]),ptr(f[5]),
     ptr(f[6]),ptr(f[7]),ptr(f[8]),ptr(f[9]));
    return responseHeader(fd,status,0);
}

bool handleAdvance(const int fd,const RequestHeader& request,ServerState& s)
{
    AdvanceArgs a{};
    if(!validatePayload(fd,request,sizeof(a)) || !readObject(fd,a)) return false;
    const int status=ugkwpGpuResidentStrictAdvance(s.backend,a.dt,a.simulationTime);
    return responseHeader(fd,status,0);
}

bool handleConfigureSst
(
    const int fd,
    const RequestHeader& request,
    ServerState& s
)
{
    SstConfigArgs a{};
    if (request.payloadBytes < sizeof(a) || !readObject(fd, a))
    {
        return false;
    }
    std::uint64_t expected = sizeof(a);
    if
    (
        !addArrays(expected, s.nCells, sizeof(double), 3)
     || !addArrays(expected, s.nFaces, sizeof(int), 2)
     || !addArrays(expected, s.nFaces, sizeof(double), 2)
     || !validatePayload(fd, request, expected)
    )
    {
        return false;
    }

    std::vector<double> k, omega, wallDistance, boundaryK, boundaryOmega;
    std::vector<int> boundaryKMode, boundaryOmegaMode;
    if
    (
        !readVector(fd, k, s.nCells)
     || !readVector(fd, omega, s.nCells)
     || !readVector(fd, wallDistance, s.nCells)
     || !readVector(fd, boundaryKMode, s.nFaces)
     || !readVector(fd, boundaryOmegaMode, s.nFaces)
     || !readVector(fd, boundaryK, s.nFaces)
     || !readVector(fd, boundaryOmega, s.nFaces)
    )
    {
        return false;
    }

    const int status = ugkwpGpuResidentStrictConfigureSst
    (
        s.backend,
        a.alphaK1, a.alphaK2, a.alphaOmega1, a.alphaOmega2,
        a.beta1, a.beta2, a.betaStar, a.gamma1, a.gamma2,
        a.a1, a.b1, a.c1, a.kMin, a.omegaMin, a.maxSourceNumber,
        a.wallTreatment, a.wallKappa, a.wallE, a.wallCmu,
        ptr(k), ptr(omega), ptr(wallDistance),
        ptr(boundaryKMode), ptr(boundaryOmegaMode),
        ptr(boundaryK), ptr(boundaryOmega)
    );
    return responseHeader(fd, status, 0);
}

bool handleDownloadSst
(
    const int fd,
    const RequestHeader& request,
    ServerState& s
)
{
    if (!validatePayload(fd, request, 0))
    {
        return false;
    }
    std::vector<double> k(s.nCells), omega(s.nCells), nut(s.nCells);
    const int status = ugkwpGpuResidentStrictDownloadSst
    (
        s.backend, ptr(k), ptr(omega), ptr(nut)
    );
    if (status != 0)
    {
        return responseHeader(fd, status, 0);
    }
    const std::uint64_t bytes =
        3ULL*static_cast<std::uint64_t>(s.nCells)*sizeof(double);
    return responseHeader(fd, 0, bytes)
        && writeVector(fd, k)
        && writeVector(fd, omega)
        && writeVector(fd, nut);
}

bool handleCourant(const int fd,const RequestHeader& request,ServerState& s)
{
    CourantArgs a{};
    if(!validatePayload(fd,request,sizeof(a)) || !readObject(fd,a)) return false;
    double maxCo=0;
    const int status=ugkwpGpuResidentStrictComputeGasCourant
    (
        s.backend,a.dt,a.targetMaxCo,a.scheduleTime,&maxCo
    );
    if(status!=0) return responseHeader(fd,status,0);
    return responseHeader(fd,0,sizeof(maxCo)) && writeAll(fd,&maxCo,sizeof(maxCo));
}

bool handleGasOnly(const int fd,const RequestHeader& request,ServerState& s)
{
    AdvanceArgs a{};
    if(!validatePayload(fd,request,sizeof(a)) || !readObject(fd,a)) return false;
    const int status=ugkwpGpuResidentStrictAdvanceGasOnly
    (s.backend,a.dt,a.simulationTime);
    return responseHeader(fd,status,0);
}

bool handleDownloadFields
(
    const int fd,const RequestHeader& request,ServerState& s
)
{
    if(!validatePayload(fd,request,0)) return false;
    std::vector<std::vector<double>> f(23,std::vector<double>(s.nCells));
    const int status=ugkwpGpuResidentStrictDownloadFields
    (s.backend,ptr(f[0]),ptr(f[1]),ptr(f[2]),ptr(f[3]),ptr(f[4]),ptr(f[5]),
     ptr(f[6]),ptr(f[7]),ptr(f[8]),ptr(f[9]),ptr(f[10]),ptr(f[11]),ptr(f[12]),
     ptr(f[13]),ptr(f[14]),ptr(f[15]),ptr(f[16]),ptr(f[17]),ptr(f[18]),
     ptr(f[19]),ptr(f[20]),ptr(f[21]),ptr(f[22]));
    if(status!=0) return responseHeader(fd,status,0);
    const std::uint64_t bytes=std::uint64_t(s.nCells)*23*sizeof(double);
    if(!responseHeader(fd,0,bytes)) return false;
    for(const auto& v:f) if(!writeVector(fd,v)) return false;
    return true;
}

bool handleDownloadGasBoundary
(
    const int fd,const RequestHeader& request,ServerState& s
)
{
    if(!validatePayload(fd,request,0)) return false;
    std::vector<std::vector<double>> f(6,std::vector<double>(s.nFaces));
    const int status=ugkwpGpuResidentStrictDownloadGasBoundaryFields
    (s.backend,ptr(f[0]),ptr(f[1]),ptr(f[2]),ptr(f[3]),ptr(f[4]),ptr(f[5]));
    if(status!=0) return responseHeader(fd,status,0);
    const std::uint64_t bytes=std::uint64_t(s.nFaces)*6*sizeof(double);
    if(!responseHeader(fd,0,bytes)) return false;
    for(const auto& v:f) if(!writeVector(fd,v)) return false;
    return true;
}

bool handleUploadParticles
(
    const int fd,const RequestHeader& request,ServerState& s
)
{
    CountArgs a{};
    if(request.payloadBytes<sizeof(a) || !readObject(fd,a)) return false;
    if(!validCount(a.count,s.particleCapacity))
        return protocolError(fd,"invalid particle restart upload count");
    std::uint64_t expected=sizeof(a);
    if(!addArrays(expected,a.count,sizeof(double),10)
     || !addArrays(expected,a.count,sizeof(int),2)
     || !addArrays(expected,a.count,sizeof(unsigned long long),2)
     || !validatePayload(fd,request,expected)) return false;
    std::vector<std::vector<double>> d(10);
    for(auto& v:d) if(!readVector(fd,v,a.count)) return false;
    std::vector<int> cell,status;
    std::vector<unsigned long long> rng,origId;
    if(!readVector(fd,cell,a.count)||!readVector(fd,status,a.count)
     ||!readVector(fd,rng,a.count)
     ||!readVector(fd,origId,a.count)) return false;
    const int rc=ugkwpGpuResidentStrictUploadParticleRestartMirror
    (s.backend,a.count,ptr(d[0]),ptr(d[1]),ptr(d[2]),ptr(d[3]),ptr(d[4]),
     ptr(d[5]),ptr(d[6]),ptr(d[7]),ptr(d[8]),ptr(d[9]),ptr(cell),ptr(status),
     ptr(rng),ptr(origId));
    return responseHeader(fd,rc,0);
}

bool handleDownloadParticles
(
    const int fd,const RequestHeader& request,ServerState& s
)
{
    CountArgs a{};
    if(!validatePayload(fd,request,sizeof(a)) || !readObject(fd,a)) return false;
    if(!validCount(a.count,s.particleCapacity))
        return protocolError(fd,"invalid particle restart download capacity");
    int count=0;
    std::vector<std::vector<double>> d(10,std::vector<double>(a.count));
    std::vector<int> cell(a.count),status(a.count);
    std::vector<unsigned long long> rng(a.count),origId(a.count);
    const int rc=ugkwpGpuResidentStrictDownloadParticleRestartMirror
    (s.backend,&count,a.count,ptr(d[0]),ptr(d[1]),ptr(d[2]),ptr(d[3]),ptr(d[4]),
     ptr(d[5]),ptr(d[6]),ptr(d[7]),ptr(d[8]),ptr(d[9]),ptr(cell),ptr(status),
     ptr(rng),ptr(origId));
    if(rc!=0) return responseHeader(fd,rc,0);
    if(count<0 || count>s.particleCapacity || (a.count!=0 && count>a.count))
        return protocolError(fd,"backend returned an invalid particle count");
    std::uint64_t per=0;
    addArrays(per,1,sizeof(double),10); addArrays(per,1,sizeof(int),2);
    addArrays(per,1,sizeof(unsigned long long),2);
    const std::uint64_t bytes=sizeof(count)+(a.count==0?0:std::uint64_t(count)*per);
    if(!responseHeader(fd,0,bytes)||!writeAll(fd,&count,sizeof(count))) return false;
    if(a.count==0) return true;
    for(auto& v:d){v.resize(count);if(!writeVector(fd,v))return false;}
    cell.resize(count);status.resize(count);rng.resize(count);origId.resize(count);
    return writeVector(fd,cell)&&writeVector(fd,status)&&writeVector(fd,rng)
        &&writeVector(fd,origId);
}

bool handleDownloadEpsGPrev
(const int fd,const RequestHeader& request,ServerState& s)
{
    if(!validatePayload(fd,request,0)) return false;
    std::vector<double> values(s.nCells);
    const int rc=ugkwpGpuResidentStrictDownloadEpsGPrev(s.backend,ptr(values));
    if(rc!=0) return responseHeader(fd,rc,0);
    return responseHeader(fd,0,values.size()*sizeof(double))&&writeVector(fd,values);
}

bool handleUploadEpsGPrev
(const int fd,const RequestHeader& request,ServerState& s)
{
    const std::uint64_t expected=std::uint64_t(s.nCells)*sizeof(double);
    if(!validatePayload(fd,request,expected)) return false;
    std::vector<double> values;
    if(!readVector(fd,values,s.nCells)) return false;
    const int rc=ugkwpGpuResidentStrictUploadEpsGPrev(s.backend,ptr(values));
    return responseHeader(fd,rc,0);
}

bool handleDownloadNut
(const int fd,const RequestHeader& request,ServerState& s)
{
    if(!validatePayload(fd,request,0)) return false;
    std::vector<double> values(s.nCells);
    const int rc=ugkwpGpuResidentStrictDownloadNut(s.backend,ptr(values));
    if(rc!=0) return responseHeader(fd,rc,0);
    return responseHeader(fd,0,values.size()*sizeof(double))&&writeVector(fd,values);
}

bool dispatch
(
    const int fd,const RequestHeader& request,ServerState& s,bool& done
)
{
    const Op op=static_cast<Op>(request.operation);
    if(op!=Op::create && s.backend==nullptr)
        return protocolError(fd,"GPU backend request arrived before create");
    switch(op)
    {
        case Op::create:return handleCreate(fd,request,s);
        case Op::uploadMesh:return handleMesh(fd,request,s);
        case Op::uploadBoundarySources:return handleBoundarySources(fd,request,s);
        case Op::configureScheduledInlet:
            return handleScheduledInletConfiguration(fd,request,s);
        case Op::downloadSourceResidualMass:return handleDownloadResidual(fd,request,s);
        case Op::uploadSourceResidualMass:return handleUploadResidual(fd,request,s);
        case Op::uploadGasBoundaryFields:return handleGasBoundary(fd,request,s);
        case Op::uploadFields:return handleUploadFields(fd,request,s);
        case Op::advance:return handleAdvance(fd,request,s);
        case Op::computeGasCourant:return handleCourant(fd,request,s);
        case Op::advanceGasOnly:return handleGasOnly(fd,request,s);
        case Op::downloadFields:return handleDownloadFields(fd,request,s);
        case Op::downloadGasBoundaryFields:return handleDownloadGasBoundary(fd,request,s);
        case Op::uploadParticleRestartMirror:return handleUploadParticles(fd,request,s);
        case Op::downloadParticleRestartMirror:return handleDownloadParticles(fd,request,s);
        case Op::downloadEpsGPrev:return handleDownloadEpsGPrev(fd,request,s);
        case Op::uploadEpsGPrev:return handleUploadEpsGPrev(fd,request,s);
        case Op::downloadNut:return handleDownloadNut(fd,request,s);
        case Op::configureSst:return handleConfigureSst(fd,request,s);
        case Op::downloadSst:return handleDownloadSst(fd,request,s);
        case Op::release:
        {
            if(!validatePayload(fd,request,0)) return false;
            ugkwpGpuResidentStrictRelease(s.backend); s.backend=nullptr;
            done=true; return success(fd);
        }
        default:return protocolError(fd,"unknown GPU backend operation");
    }
}

}             

int main(int argc,char** argv)
{
    if(argc!=3 || std::strcmp(argv[1],"--ipc-fd")!=0)
    {
        std::fprintf(stderr,"usage: %s --ipc-fd FD\n",argv[0]);
        return 2;
    }
    char* end=nullptr;
    const long parsed=std::strtol(argv[2],&end,10);
    if(end==argv[2] || *end!='\0' || parsed<0)
    {
        std::fprintf(stderr,"invalid IPC descriptor\n");
        return 2;
    }
    const int fd=static_cast<int>(parsed);
    ServerState state;
    bool done=false;
    while(!done)
    {
        RequestHeader request{};
        if(!readObject(fd,request)) break;
        if(request.magicValue!=magic || request.major!=protocolMajor
         || request.minor!=protocolMinor || request.payloadBytes>maxPayloadBytes)
        {
            protocolError(fd,"invalid or incompatible GPU backend request header");
            break;
        }
        if(!dispatch(fd,request,state,done)) break;
    }
    if(state.backend!=nullptr) ugkwpGpuResidentStrictRelease(state.backend);
    ::close(fd);
    return done?0:1;
}
