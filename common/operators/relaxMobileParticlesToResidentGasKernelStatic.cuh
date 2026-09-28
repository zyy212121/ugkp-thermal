#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class DragModel>
__global__ void relaxMobileParticlesToResidentGasKernelStatic
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt,
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
    const GPU_OPERATOR_TIME dt,
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
