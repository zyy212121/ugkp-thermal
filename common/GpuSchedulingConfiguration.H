#ifndef GpuSchedulingConfiguration_H
#define GpuSchedulingConfiguration_H

#include "fvCFD.H"
#include <cmath>

namespace Foam
{

enum class GpuHeavyReductionMode : label
{
    disabled = 0,
    enabled = 1,
    automatic = 2
};

enum class GpuCsrLevel : label
{
    L0 = 0,
    L1 = 1,
    L2 = 2,
    automatic = 3
};

struct GpuSchedulingConfiguration
{
    bool pureGasOnly = false;
    bool dynamicInlet = false;
    label particleCapacity = 1000000;
    label maxFaceWalkHops = 32;
    label courantUpdateInterval = 1000;
    scalar maxDeltaTGrowth = 1.05;
    GpuCsrLevel csrLevel = GpuCsrLevel::L0;
    bool csrCellLocalPath = false;
    GpuHeavyReductionMode heavyReductionMode =
        GpuHeavyReductionMode::disabled;
    label heavyReductionAutoInterval = 100;
    label particleBlockThreads = 128;
    label reductionBlockThreads = 128;
    bool warpAggregatedBinning = false;
    bool splitPreDirectory = false;
};

inline bool validGpuBlockThreads(const label value)
{
    return value == 32 || value == 64 || value == 128 || value == 256;
}

inline GpuCsrLevel readGpuCsrLevel
(
    const dictionary& properties
)
{
    if (!properties.found("gpuCsrLevel"))
    {
        FatalErrorInFunction
            << "schedulingProperties requires gpuCsrLevel L0|L1|L2|auto"
            << exit(FatalError);
    }
    const word csrLevelValue(properties.lookup("gpuCsrLevel"));
    if (csrLevelValue == "L0")
    {
        return GpuCsrLevel::L0;
    }
    if (csrLevelValue == "L1")
    {
        return GpuCsrLevel::L1;
    }
    if (csrLevelValue == "L2")
    {
        return GpuCsrLevel::L2;
    }
    if (csrLevelValue == "auto")
    {
        return GpuCsrLevel::automatic;
    }
    FatalErrorInFunction
        << "gpuCsrLevel must be L0, L1, L2 or auto. Current value = "
        << csrLevelValue << exit(FatalError);
    return GpuCsrLevel::L0;
}

inline GpuSchedulingConfiguration readGpuSchedulingConfiguration
(
    const dictionary& properties
)
{
    GpuSchedulingConfiguration value;
    const word forbiddenGpuSchedulingKeys[] =
    {
        "gpuResidentStrict",
        "gpuCsrCellLocalPath",
        "gpuCsrHeavyReduction",
        "gpuCsrWarpAggregatedBinning",
        "gpuCsrSplitPreDirectory"
    };
    for (const word& key : forbiddenGpuSchedulingKeys)
    {
        if (properties.found(key))
        {
            FatalErrorInFunction
                << key << " has been removed from schedulingProperties. "
                << "Use gpuCsrLevel L0|L1|L2|auto; the solver derives the "
                << "complete scheduling bundle."
                << exit(FatalError);
        }
    }
    value.pureGasOnly =
        properties.lookupOrDefault<bool>("gpuResidentPureGasOnly", false);
    value.dynamicInlet =
        properties.lookupOrDefault<bool>("gpuResidentDynamicInlet", false);
    value.particleCapacity =
        properties.lookupOrDefault<label>("gpuResidentParticleCapacity", 1000000);
    value.maxFaceWalkHops =
        properties.lookupOrDefault<label>("gpuResidentMaxFaceWalkHops", 32);
    value.courantUpdateInterval =
        properties.lookupOrDefault<label>("gpuResidentCourantUpdateInterval", 1000);
    value.maxDeltaTGrowth =
        properties.lookupOrDefault<scalar>("gpuResidentMaxDeltaTGrowth", 1.05);
    value.csrLevel = readGpuCsrLevel(properties);
    value.csrCellLocalPath = value.csrLevel != GpuCsrLevel::L0;
    value.warpAggregatedBinning = value.csrLevel != GpuCsrLevel::L0;
    value.splitPreDirectory = value.csrLevel != GpuCsrLevel::L0;
    if (value.csrLevel == GpuCsrLevel::L2)
    {
        value.heavyReductionMode = GpuHeavyReductionMode::enabled;
    }
    else if (value.csrLevel == GpuCsrLevel::automatic)
    {
        value.heavyReductionMode = GpuHeavyReductionMode::automatic;
        value.heavyReductionAutoInterval =
            properties.lookupOrDefault<label>
            (
                "gpuCsrHeavyReductionAutoInterval",
                100
            );
    }
    else if (properties.found("gpuCsrHeavyReductionAutoInterval"))
    {
        FatalErrorInFunction
            << "gpuCsrHeavyReductionAutoInterval is only valid when "
            << "gpuCsrLevel auto" << exit(FatalError);
    }
    value.particleBlockThreads =
        properties.lookupOrDefault<label>("gpuParticleBlockThreads", 128);
    value.reductionBlockThreads =
        properties.lookupOrDefault<label>("gpuReductionBlockThreads", 128);

    if (value.particleCapacity < 0)
    {
        FatalErrorInFunction
            << "gpuResidentParticleCapacity must be non-negative. Current value = "
            << value.particleCapacity << exit(FatalError);
    }
    if (value.maxFaceWalkHops < 1)
    {
        FatalErrorInFunction
            << "gpuResidentMaxFaceWalkHops must be positive. Current value = "
            << value.maxFaceWalkHops << exit(FatalError);
    }
    if (value.courantUpdateInterval < 1)
    {
        FatalErrorInFunction
            << "gpuResidentCourantUpdateInterval must be positive. Current value = "
            << value.courantUpdateInterval << exit(FatalError);
    }
    if
    (
        !std::isfinite(value.maxDeltaTGrowth)
     || value.maxDeltaTGrowth < scalar(1)
    )
    {
        FatalErrorInFunction
            << "gpuResidentMaxDeltaTGrowth must be finite and at least one. "
            << "Current value = " << value.maxDeltaTGrowth << exit(FatalError);
    }
    if (!validGpuBlockThreads(value.particleBlockThreads))
    {
        FatalErrorInFunction
            << "gpuParticleBlockThreads must be 32, 64, 128 or 256. "
            << "Current value = " << value.particleBlockThreads
            << exit(FatalError);
    }
    if (!validGpuBlockThreads(value.reductionBlockThreads))
    {
        FatalErrorInFunction
            << "gpuReductionBlockThreads must be 32, 64, 128 or 256. "
            << "Current value = " << value.reductionBlockThreads
            << exit(FatalError);
    }
    if
    (
        value.heavyReductionMode == GpuHeavyReductionMode::automatic
     && value.heavyReductionAutoInterval < 1
    )
    {
        FatalErrorInFunction
            << "gpuCsrHeavyReductionAutoInterval must be positive in auto mode. "
            << "Current value = " << value.heavyReductionAutoInterval
            << exit(FatalError);
    }
    return value;
}

}

#endif
