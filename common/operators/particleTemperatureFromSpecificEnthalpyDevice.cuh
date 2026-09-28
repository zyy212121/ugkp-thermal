#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL particleTemperatureFromSpecificEnthalpyDevice
(
    const GPU_OPERATOR_REAL specificEnthalpyJkg
)
{
    return Foam::gpuThermal::aluminaTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg
    );
}
