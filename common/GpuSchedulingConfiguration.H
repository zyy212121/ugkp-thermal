#ifndef GpuSchedulingConfiguration_H
#define GpuSchedulingConfiguration_H

#include "fvCFD.H"
#include <cmath>
#include "gpu/WorkPlan.H"

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
        return GpuCsrLevel::L0;
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
    return GpuCsrLevel::L0;
}

inline GpuSchedulingConfiguration readGpuSchedulingConfiguration
(
    const dictionary& properties,
    const label trackingMaxFaceWalkHops = 32
)
{
    GpuSchedulingConfiguration value;
    value.pureGasOnly =
        properties.lookupOrDefault<bool>("gpuResidentPureGasOnly", false);
    value.particleCapacity =
        properties.lookupOrDefault<label>("gpuResidentParticleCapacity", 1000000);
    value.maxFaceWalkHops = trackingMaxFaceWalkHops;
    // Gas adaptive-step numerics now come only from constant/scheme/gasTimeStep.
    value.csrLevel = readGpuCsrLevel(properties);
    if (value.csrLevel == GpuCsrLevel::automatic)
    {
        value.heavyReductionAutoInterval = properties.lookupOrDefault<label>
        (
            "gpuCsrHeavyReductionAutoInterval", 100
        );
        value.heavyReductionMode = GpuHeavyReductionMode::automatic;
    }
    value.csrCellLocalPath = value.csrLevel != GpuCsrLevel::L0;
    value.warpAggregatedBinning = value.csrLevel != GpuCsrLevel::L0;
    value.splitPreDirectory = value.csrLevel != GpuCsrLevel::L0;
    if (value.csrLevel == GpuCsrLevel::L2)
    {
        value.heavyReductionMode = GpuHeavyReductionMode::enabled;
    }
    // One startup plan shared by applications and GPU owners.
    auto& plan=ugkp::gpu::workPlan();
    plan.cellBlockThreads=properties.lookupOrDefault<label>("gpuCellBlockThreads",128);
    plan.faceBlockThreads=properties.lookupOrDefault<label>("gpuFaceBlockThreads",128);
    plan.sparseBlockThreads=properties.lookupOrDefault<label>("gpuSparseBlockThreads",128);
    plan.radiationBlockThreads=properties.lookupOrDefault<label>("gpuRadiationBlockThreads",128);
    plan.thermalBlockThreads=properties.lookupOrDefault<label>("gpuThermalBlockThreads",128);
    plan.materialBlockThreads=properties.lookupOrDefault<label>("gpuMaterialBlockThreads",128);
    plan.coldWallBlockThreads=properties.lookupOrDefault<label>("gpuColdWallBlockThreads",128);
    value.particleBlockThreads =
        properties.lookupOrDefault<label>("gpuParticleBlockThreads", 128);
    value.reductionBlockThreads =
        properties.lookupOrDefault<label>("gpuReductionBlockThreads", 128);

    plan.particleBlockThreads=value.particleBlockThreads;
    plan.reductionBlockThreads=value.reductionBlockThreads;

    if
    (
        value.heavyReductionMode == GpuHeavyReductionMode::automatic
     && value.heavyReductionAutoInterval == 0
    )
    {
        FatalErrorInFunction
            << "gpuCsrHeavyReductionAutoInterval is a zero modulo divisor in auto mode. "
            << "Current value = " << value.heavyReductionAutoInterval
            << exit(FatalError);
    }
    return value;
}

}

#endif
