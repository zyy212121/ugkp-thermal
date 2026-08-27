#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../../../../common/wall/GpuColdWallSolidification.H"

struct DeviceState
{
    int coldWallSolidificationEnabled;
    float* pColdNodeSpecificEnthalpy;
    float* pColdRingSolidMass;
    float* pColdFrozenArea;
    float* pColdContactAge;
    Foam::gpuThermal::ColdWallSolidificationParameters
        coldWallSolidificationParameters;
};

#define GPU_COLD_WALL_1D_ALGEBRA_ONLY
#include "../../../../common/wall/GpuColdWall1DDevice.cuh"

__global__ void advanceGroup
(
    DeviceState* state,
    const double physicalVolume,
    const double physicalMass,
    const double maximumArea,
    const double contactArea,
    const double duration,
    const double peakFraction,
    const double dt,
    double* output
)
{
    const int lane = threadIdx.x;
    double meanTemperature = 0.0;
    double frozenArea = 0.0;
    double wallEnergy = 0.0;
    const bool valid = advanceColdWall1DThermalGroup
    (
        *state,
        0,
        lane,
        0xffu,
        physicalVolume,
        physicalMass,
        maximumArea,
        contactArea,
        duration,
        peakFraction,
        dt,
        300.0,
        13000.0,
        0.1,
        3200.0,
        2.0e-4,
        meanTemperature,
        frozenArea,
        wallEnergy
    );
    if (lane == 0)
    {
        output[0] = valid ? 1.0 : 0.0;
        output[1] = meanTemperature;
        output[2] = frozenArea;
        output[3] = wallEnergy;
    }
}

int main()
{
    using namespace Foam::gpuThermal;
    ColdWallSolidificationParameters parameters
    {
        2327.0,
        20.0,
        1.16e6,
        3990.0,
        1273.0,
        5.9,
        0.25,
        0.0,
        1,
        4
    };
    const double diameter = 120.0e-6;
    const double volume = coldWallPi/6.0*diameter*diameter*diameter;
    const double mass = 3200.0*volume;
    const double maximumArea = 2.2e-8;
    const double contactArea = 1.7e-8;
    const double duration = 8.0e-6;
    const double peakFraction = 0.42;
    const double dt = 7.5e-8;
    float* node = nullptr;
    float* ring = nullptr;
    float* frozen = nullptr;
    float* age = nullptr;
    double* output = nullptr;
    DeviceState* state = nullptr;
    cudaMallocManaged(&node, 8*sizeof(float));
    cudaMallocManaged(&ring, 8*sizeof(float));
    cudaMallocManaged(&frozen, sizeof(float));
    cudaMallocManaged(&age, sizeof(float));
    cudaMallocManaged(&output, 4*sizeof(double));
    cudaMallocManaged(&state, sizeof(DeviceState));
    const float initialEnthalpy = static_cast<float>
    (
        coldWallSpecificEnthalpyJkg(3500.0, parameters)
    );
    double serialNode[8];
    double serialRing[8];
    for (int i = 0; i < 8; ++i)
    {
        node[i] = initialEnthalpy;
        ring[i] = 0.0f;
        serialNode[i] = static_cast<double>(node[i]);
        serialRing[i] = 0.0;
    }
    *frozen = 0.0f;
    *age = 0.0f;
    *state = {1, node, ring, frozen, age, parameters};
    double serialAge = 0.0;
    double serialFrozen = 0.0;
    double maximumRelativeError = 0.0;
    for (int step = 0; step < 1000; ++step)
    {
        const ColdWallSolidificationStep serial = advanceColdWallProfile
        (
            serialNode,
            serialRing,
            serialAge,
            serialFrozen,
            parameters,
            volume,
            mass,
            maximumArea,
            contactArea,
            duration,
            peakFraction,
            dt,
            300.0,
            13000.0,
            0.1,
            3200.0,
            2.0e-4
        );
        if (!serial.valid)
        {
            return 2;
        }
        advanceGroup<<<1, 8>>>
        (
            state,
            volume,
            mass,
            maximumArea,
            contactArea,
            duration,
            peakFraction,
            dt,
            output
        );
        if (cudaDeviceSynchronize() != cudaSuccess || output[0] != 1.0)
        {
            return 3;
        }
        maximumRelativeError = std::max
        (
            maximumRelativeError,
            std::fabs(output[1] - serial.meanTemperatureK)
           /std::max(1.0, std::fabs(serial.meanTemperatureK))
        );
        maximumRelativeError = std::max
        (
            maximumRelativeError,
            std::fabs(output[3] - serial.wallEnergyJ)
           /std::max(1.0e-30, std::fabs(serial.wallEnergyJ))
        );
        for (int i = 0; i < 8; ++i)
        {
            serialNode[i] = static_cast<double>
            (
                static_cast<float>(serialNode[i])
            );
            serialRing[i] = static_cast<double>
            (
                static_cast<float>(serialRing[i])
            );
            maximumRelativeError = std::max
            (
                maximumRelativeError,
                std::fabs(static_cast<double>(node[i]) - serialNode[i])
               /std::max(1.0, std::fabs(serialNode[i]))
            );
        }
        serialAge = static_cast<double>(static_cast<float>(serialAge));
        serialFrozen = static_cast<double>(static_cast<float>(serialFrozen));
    }
    std::printf("maximumRelativeError=%.17g\n", maximumRelativeError);
    return maximumRelativeError <= 2.0e-10 ? 0 : 4;
}
