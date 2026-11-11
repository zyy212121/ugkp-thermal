#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ double particleTemperatureFromSpecificEnthalpyDevice
(
    const double specificEnthalpyJkg
)
{

    if (!Foam::gpuThermal::finiteAluminaValue(specificEnthalpyJkg)
        || specificEnthalpyJkg < GPU_OPERATOR_R(0.0))
        return -GPU_OPERATOR_R(1.0);
    return Foam::gpuThermal::aluminaTemperatureFromSpecificEnthalpyK
    (
        specificEnthalpyJkg
    );
}
