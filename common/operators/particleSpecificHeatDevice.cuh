#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL particleSpecificHeatDevice(const GPU_OPERATOR_REAL temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificHeatJkgK(temperatureK);
}

__device__ GPU_OPERATOR_REAL particleSpecificEnthalpyDevice(const GPU_OPERATOR_REAL temperatureK)
{

    if (!Foam::gpuThermal::finiteAluminaValue(temperatureK)
        || temperatureK < GPU_OPERATOR_R(0.0))
        return -GPU_OPERATOR_R(1.0);
    return Foam::gpuThermal::aluminaSpecificEnthalpyJkg(temperatureK);
}
