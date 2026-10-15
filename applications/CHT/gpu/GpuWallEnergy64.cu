#include "GpuWallEnergy64.H"
#include <algorithm>
#include <cstdio>
#include <vector>

namespace
{
void setLastErrorText(GpuWallEnergyState* s, const char* message)
{
    if (s != nullptr && s->errorBuffer != nullptr)
        std::snprintf(s->errorBuffer, 2048, "%s", message);
}

void setLastError(GpuWallEnergyState* s, const char* name, cudaError_t error)
{
    if (s != nullptr && s->errorBuffer != nullptr)
        std::snprintf(s->errorBuffer, 2048, "%s: %s", name, cudaGetErrorString(error));
}

GpuWallEnergyState* asState(void* handle)
{
    return static_cast<GpuWallEnergyState*>(handle);
}

int validateState(GpuWallEnergyState* s, const char* name)
{
    if (s == nullptr || s->deviceState == nullptr)
    {
        setLastErrorText(s, name);
        return 1;
    }
    return 0;
}

template<class T>
void release(T*& pointer)
{
    if (pointer != nullptr) cudaFree(pointer);
    pointer = nullptr;
}

template<class T>
int allocate(GpuWallEnergyState* s, T*& pointer, size_t count, const char* name)
{
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&pointer), count*sizeof(T));
    if (error == cudaSuccess) return 0;
    setLastError(s, name, error);
    return 1;
}

template<class T>
int copyToDevice(GpuWallEnergyState* s, T* destination, const T* source, size_t count, const char* name)
{
    if (count == 0) return 0;
    const cudaError_t error = cudaMemcpy(destination, source, count*sizeof(T), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) return 0;
    setLastError(s, name, error);
    return 1;
}

template<class T>
int copyToHost(GpuWallEnergyState* s, T* destination, const T* source, size_t count, const char* name)
{
    if (count == 0) return 0;
    const cudaError_t error = cudaMemcpy(destination, source, count*sizeof(T), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) return 0;
    setLastError(s, name, error);
    return 1;
}

int syncGasWallLedgerPointers(GpuWallEnergyState* s, const char* name)
{
    const cudaError_t error = cudaMemcpy(s->deviceState, s, sizeof(*s), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) return 0;
    setLastError(s, name, error);
    return 1;
}

__global__ void accumulateGasWallEnergy64Kernel
(
    double* energy,
    const double* flux,
    const int* faces,
    const int count,
    const double dt
)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < count)
    {
        const int face = faces[index];
        energy[face] += dt*flux[face];
    }
}
}

int ugkpAccumulateGasWallEnergy64(GpuWallEnergyState* s, double dt, cudaStream_t stream)
{
    if (s->nEnabledFaces == 0) return 0;
    accumulateGasWallEnergy64Kernel<<<(s->nEnabledFaces+255)/256, 256, 0, stream>>>
    (s->gasWallEnergy, s->gasWallFlux, s->enabledFaceIds, s->nEnabledFaces, dt);
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) return 0;
    setLastError(s, "accumulateGasWallEnergy64Kernel launch", error);
    return 1;
}

void ugkpReleaseGasWallEnergy64(GpuWallEnergyState* s)
{
    release(s->gasWallEnergy);
    release(s->gasWallEnergyMask);
    release(s->gasWallFlux);
    release(s->enabledFaceIds);
    release(s->retiredFaceIds);
    s->enabledFaceCapacity = 0;
    s->nEnabledFaces = 0;
}

extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger
(
    void* handle,
    int nEnabledFaces,
    const int* enabledFaceIds
)
{

    GpuWallEnergyState* s = asState(handle);
    if (s == nullptr || s->deviceState == nullptr)
    {
        return 1;
    }
    if (nEnabledFaces > 0 && enabledFaceIds == nullptr)
    {
        setLastErrorText(s, "missing gas-wall energy ledger face list");
        return 1;
    }
    if (nEnabledFaces == 0 && s->gasWallEnergy == nullptr)
    {
        return 0;
    }

    if (ugkpInvalidateGasWallEnergyGraph(handle) != 0) return 1;
    if (s->retiredFaceIds != nullptr)
    {

        if (syncGasWallLedgerPointers(s, "cudaMemcpy retry gas-wall ledger pointers") != 0)
            return 1;
        const cudaError_t error = cudaFree(s->retiredFaceIds);
        if (error != cudaSuccess)
        {
            setLastError(s, "cudaFree retired gas-wall face ids", error);
            return 1;
        }
        s->retiredFaceIds = nullptr;
    }

    std::vector<unsigned char> hostMask
    (
        static_cast<size_t>(s->nFaces),
        static_cast<unsigned char>(0)
    );
    for (int enabledI = 0; enabledI < nEnabledFaces; ++enabledI)
    {
        const int faceI = enabledFaceIds[enabledI];
        hostMask[static_cast<size_t>(faceI)] = 1;
    }

    if (s->gasWallEnergy != nullptr)
    {
        int* faces = s->enabledFaceIds;
        const size_t count = static_cast<size_t>(nEnabledFaces);
        const bool replaceFaces = count > s->enabledFaceCapacity;
        if (replaceFaces)
        {
            faces = nullptr;
            if (allocate(s, faces, count, "cudaMalloc replacement wall face ids") != 0)
                return 1;
        }
        if (copyToDevice(s, faces, enabledFaceIds, count, "cudaMemcpy wall face ids") != 0
            || copyToDevice(s, s->gasWallEnergyMask, hostMask.data(), hostMask.size(), "cudaMemcpy gas-wall energy mask") != 0)
        {
            if (replaceFaces)
            {

                if (cudaFree(faces) != cudaSuccess) s->retiredFaceIds = faces;
            }
            return 1;
        }
        if (replaceFaces)
        {
            s->retiredFaceIds = s->enabledFaceIds;
            s->enabledFaceIds = faces;
            s->enabledFaceCapacity = count;
        }
        s->nEnabledFaces = nEnabledFaces;
        if (syncGasWallLedgerPointers(s, "cudaMemcpy configure gas-wall energy ledger pointers") != 0)
            return 1;
        if (s->retiredFaceIds != nullptr)
        {
            const cudaError_t error = cudaFree(s->retiredFaceIds);
            if (error != cudaSuccess)
            {
                setLastError(s, "cudaFree retired gas-wall face ids", error);
                return 1;
            }
            s->retiredFaceIds = nullptr;
        }
        return 0;
    }

    double* newEnergy = nullptr;
    unsigned char* newMask = nullptr;
    double* newFlux = nullptr;
    int* newFaces = nullptr;
    if
    (
        allocate(s, 
            newEnergy,
            static_cast<size_t>(s->nFaces),
            "cudaMalloc gas-wall energy ledger"
        ) != 0
     || allocate(s, 
            newMask,
            static_cast<size_t>(s->nFaces),
            "cudaMalloc gas-wall energy mask"
        ) != 0
    )
    {
        release(newEnergy);
        release(newMask);
        release(newFlux);
        release(newFaces);
        return 1;
    }

    if (allocate(s, newFlux, static_cast<size_t>(s->nFaces), "cudaMalloc wall flux") != 0
        || allocate(s, newFaces, static_cast<size_t>(nEnabledFaces), "cudaMalloc wall face ids") != 0
        || copyToDevice(s, newFaces, enabledFaceIds, static_cast<size_t>(nEnabledFaces), "cudaMemcpy wall face ids") != 0)
    {
        release(newEnergy);
        release(newMask);
        release(newFlux);
        release(newFaces);
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
     || copyToDevice(s, 
            newMask,
            hostMask.data(),
            hostMask.size(),
            "cudaMemcpy gas-wall energy mask"
        ) != 0
    )
    {
        if (err != cudaSuccess)
        {
            setLastError(s, "cudaMemset gas-wall energy ledger", err);
        }
        release(newEnergy);
        release(newMask);
        release(newFlux);
        release(newFaces);
        return 1;
    }

    s->gasWallEnergy = newEnergy;
    s->gasWallEnergyMask = newMask;
    s->gasWallFlux = newFlux;
    s->enabledFaceIds = newFaces;
    s->enabledFaceCapacity = static_cast<size_t>(nEnabledFaces);
    s->nEnabledFaces = nEnabledFaces;
    if
    (
        syncGasWallLedgerPointers
        (
            s,
            "cudaMemcpy configure gas-wall energy ledger pointers"
        ) != 0
    )
    {

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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "gas-wall energy ledger peek") != 0
     || gasWallEnergy == nullptr
     || s->gasWallEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid gas-wall energy ledger peek input");
        return 1;
    }
    return copyToHost(s, 
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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "pending wall-energy ledger range peek") != 0
     || gasWallEnergyJ == nullptr
     || particleDepositedWallEnergyJ == nullptr
     || particleReflectedWallEnergyJ == nullptr
     || s->gasWallEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid pending wall-energy ledger range peek");
        return 1;
    }
    const size_t count = static_cast<size_t>(nFaces);
    if
    (
        copyToHost(s, 
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
     || copyToHost(s, 
            particleDepositedWallEnergyJ,
            s->particleWallDepositedEnergy + firstFace,
            count,
            "cudaMemcpy compact deposited-particle wall energy peek"
        ) != 0
     || copyToHost(s, 
            particleReflectedWallEnergyJ,
            s->particleWallReflectedEnergy + firstFace,
            count,
            "cudaMemcpy compact reflected-particle wall energy peek"
        ) != 0
    )
    {
        setLastErrorText(s, "invalid configured particle wall-energy ledgers");
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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        s == nullptr
     || s->deviceState == nullptr
     || gasWallEnergyJ == nullptr
     || particleDepositedWallEnergyJ == nullptr
     || particleReflectedWallEnergyJ == nullptr
     || s->gasWallEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid pending wall-energy ledger range restore");
        return 1;
    }
    const size_t count = static_cast<size_t>(nFaces);
    if
    (
        copyToDevice(s, 
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
            if (ugkpPrepareParticleWallEnergyRestore(handle) != 0) return 1;
        }
        else return 0;
    }
    if
    (
        s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
     || copyToDevice(s, 
            s->particleWallDepositedEnergy + firstFace,
            particleDepositedWallEnergyJ,
            count,
            "cudaMemcpy compact deposited-particle wall energy restore"
        ) != 0
     || copyToDevice(s, 
            s->particleWallReflectedEnergy + firstFace,
            particleReflectedWallEnergyJ,
            count,
            "cudaMemcpy compact reflected-particle wall energy restore"
        ) != 0
    )
    {
        setLastErrorText(s, "invalid configured particle wall-energy ledgers");
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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "gas-wall energy ledger download/reset") != 0
     || gasWallEnergy == nullptr
     || s->gasWallEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid gas-wall energy ledger download/reset input");
        return 1;
    }
    if
    (
        copyToHost(s, 
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
        setLastError(s, "cudaMemset gas-wall energy ledger reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaDeviceSynchronize gas-wall ledger reset", err);
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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "particle wall heat-ledger peek") != 0
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
     || s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid particle wall heat-ledger peek");
        return 1;
    }
    if
    (
        copyToHost(s, 
            depositedWallEnergyJ,
            s->particleWallDepositedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy deposited-particle wall energy peek"
        ) != 0
     || copyToHost(s, 
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

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "particle wall heat-ledger download/reset") != 0
     || depositedWallEnergyJ == nullptr
     || reflectedWallEnergyJ == nullptr
     || s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid particle wall heat-ledger download/reset");
        return 1;
    }
    if
    (
        copyToHost(s, 
            depositedWallEnergyJ,
            s->particleWallDepositedEnergy,
            static_cast<size_t>(nFaces),
            "cudaMemcpy deposited-particle wall energy download"
        ) != 0
     || copyToHost(s, 
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
        setLastError(s, "cudaMemset deposited-particle wall energy reset", err);
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
        setLastError(s, "cudaMemset reflected-particle wall energy reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaDeviceSynchronize particle wall ledgers reset", err);
        return 1;
    }
    return 0;
}


extern "C" int ugkwpGpuResidentStrictResetGasWallEnergy(void* handle)
{

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "gas-wall energy ledger reset") != 0
     || s->gasWallEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid gas-wall energy ledger reset input");
        return 1;
    }
    cudaError_t err = cudaMemset
    (
        s->gasWallEnergy,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaMemset gas-wall energy ledger reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaDeviceSynchronize gas-wall ledger reset", err);
        return 1;
    }
    return 0;
}

extern "C" int ugkwpGpuResidentStrictResetParticleWallHeatLedgers(void* handle)
{

    GpuWallEnergyState* s = asState(handle);
    if
    (
        validateState(s, "particle wall heat-ledger reset") != 0
     || s->particleWallDepositedEnergy == nullptr
     || s->particleWallReflectedEnergy == nullptr
    )
    {
        setLastErrorText(s, "invalid particle wall heat-ledger reset");
        return 1;
    }
    cudaError_t err = cudaMemset
    (
        s->particleWallDepositedEnergy,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaMemset deposited-particle wall energy reset", err);
        return 1;
    }
    err = cudaMemset
    (
        s->particleWallReflectedEnergy,
        0,
        static_cast<size_t>(s->nFaces)*sizeof(double)
    );
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaMemset reflected-particle wall energy reset", err);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError(s, "cudaDeviceSynchronize particle wall ledgers reset", err);
        return 1;
    }
    return 0;
}
