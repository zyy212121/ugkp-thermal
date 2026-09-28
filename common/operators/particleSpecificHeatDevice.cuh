#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL particleSpecificHeatDevice(const GPU_OPERATOR_REAL temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificHeatJkgK(temperatureK);
}

__device__ GPU_OPERATOR_REAL particleSpecificEnthalpyDevice(const GPU_OPERATOR_REAL temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificEnthalpyJkg(temperatureK);
}
