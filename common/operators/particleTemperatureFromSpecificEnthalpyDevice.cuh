#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL particleTemperatureFromSpecificEnthalpyDevice
(
    const GPU_OPERATOR_REAL specificEnthalpyJkg
)
{
    // These recovery consumers used the original -1 as a normal fallback.
    if (!Foam::gpuThermal::finiteAluminaValue(specificEnthalpyJkg)
        || specificEnthalpyJkg < GPU_OPERATOR_R(0.0))
        return -GPU_OPERATOR_R(1.0);
    return Foam::gpuThermal::aluminaTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg
    );
}
