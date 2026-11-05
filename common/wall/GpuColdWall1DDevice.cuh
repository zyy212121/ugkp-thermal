#include "GpuPrecisionTypes.H"
#include "GpuColdWallThermalCarry.cuh"
#include "GpuColdWall1DAlgebra.cuh"
#include "GpuColdWall1DGasOnly.cuh"
#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH








__device__ bool advanceColdWall1DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int lane,
    const unsigned int mask,
    const GpuReal physicalVolumeM3,
    const GpuReal physicalMassKg,
    const GpuReal maximumAreaM2,
    const GpuReal intrinsicContactAreaM2,
    const GpuTime contactDurationS,
    const GpuReal peakTimeFraction,
    const GpuTime deltaTSeconds,
    const GpuReal wallTemperatureK,
    const GpuReal wallEffusivity,
    const GpuReal thermalAreaFactor,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK,
    GpuReal& meanTemperatureK,
    GpuReal& frozenFootprintAreaM2,
    GpuReal& wallEnergyJ,
    const GpuTime gasExchangeDurationS = -1.0
)
{
#include "GpuColdWall1DAdvance.inl"
}

#ifndef GPU_COLD_WALL_1D_ALGEBRA_ONLY

__global__ void relaxColdWall1DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const GpuTime dt
)
{
    DeviceState& s = *sp;
    if ((blockDim.x & 7) != 0)
    {
        asm("trap;");
    }
    const int lane = threadIdx.x & 7;
    const int groupsPerBlock = blockDim.x/8;
    const int groupInBlock = threadIdx.x/8;
    const int groupStride = gridDim.x*groupsPerBlock;
    const unsigned int mask = 0xffu << ((threadIdx.x & 31)/8*8);
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*groupsPerBlock + groupInBlock;
        entry < nWallBound;
        entry += groupStride
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        int active = i >= 0 && i < s.particleCapacity;
        active = active
          && s.pStatus[i] != 0
          && s.pStuck[i] != Foam::gpuThermal::particleWallMobile
          && s.particleWallHeatTransferEnabled != 0;
        const int faceI = active ? s.pStuckFaceId[i] : -1;
        active = active
          && faceI >= 0
          && faceI < s.nFaces
          && s.particleStuckCandidateMask[faceI]
             == Foam::gpuThermal::particleWallSolidifyingDeposition;
        if (!__all_sync(mask, active))
        {
            continue;
        }
#include "GpuColdWall1DInputs.inl"
        const bool valid = advanceColdWall1DThermalGroup
        (
            s,
            i,
            lane,
            mask,
            physicalVolume,
            physicalMass,
            maximumArea,
            intrinsicArea,
            duration,
            peakTimeFraction,
            activeDt,
            ugkwp::particleWallTemperature(s,faceI),
            wallEffusivity,
            efficiency*s.particleWallContactAreaScale[faceI],
            gasTemperatureK,
            gasConductanceWK,
            meanTemperature,
            frozenArea,
            wallEnergy,
            dt
        );
        if (!valid)
        {
            asm("trap;");
        }
        if (lane == 0)
        {
            s.pT[i] = meanTemperature;
            atomicAddParticleWallEnergyByFace
            (
                s,
                finiteContact
                  ? s.particleWallReflectedEnergy
                  : s.particleWallDepositedEnergy,
                faceI,
                parcelMultiplicity*wallEnergy
            );
        }
    }
}

#endif

#endif
