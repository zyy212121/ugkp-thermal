#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ double particleSpecificHeatDevice(const double temperatureK)
{
    return Foam::gpuThermal::aluminaSpecificHeatJkgK(temperatureK);
}

__device__ double particleSpecificEnthalpyDevice(const double temperatureK)
{

    if (!Foam::gpuThermal::finiteAluminaValue(temperatureK)
        || temperatureK < GPU_OPERATOR_R(0.0))
        return -GPU_OPERATOR_R(1.0);
    return Foam::gpuThermal::aluminaSpecificEnthalpyJkg(temperatureK);
}
