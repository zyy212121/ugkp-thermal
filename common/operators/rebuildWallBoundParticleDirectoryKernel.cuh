#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
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

    GPU_OPERATOR_REAL scale = GPU_OPERATOR_R(1.0);
    if (s.particleStuckCandidateMask[faceI] != 0)
    {
        const GPU_OPERATOR_REAL representedArea =
            s.particleWallRepresentedContactArea[faceI];
        const GPU_OPERATOR_REAL maximumArea =
            s.particleWallMaximumCoverage*s.magSf[faceI];
        if
        (
            !finiteDevice(representedArea)
         || !finiteDevice(maximumArea)
         || representedArea < GPU_OPERATOR_R(0.0)
         || !(maximumArea > GPU_OPERATOR_R(0.0))
        )
        {
            asm("trap;");
        }
        if (representedArea > maximumArea)
        {
            scale = maximumArea/representedArea;
        }
    }
    if (!(scale > GPU_OPERATOR_R(0.0)) || scale > GPU_OPERATOR_R(1.0))
    {
        asm("trap;");
    }
    s.particleWallContactAreaScale[faceI] = scale;
}
