#pragma once
// Shared probe option parsing, event allocation and initialization.
int initialiseDevelopmentProbe(DeviceState* owner)
{
    if (developmentProbe.owner != nullptr)
    {
        setLastErrorText
        (
            "UGKP development probe already belongs to another backend handle"
        );
        return 1;
    }
    shutdownDevelopmentProbe();

    const char* const configuredMode = std::getenv("UGKP_DEV_PROBE_MODE");
    if
    (
        configuredMode == nullptr
     || *configuredMode == '\0'
     || std::strcmp(configuredMode, "off") == 0
     || std::strcmp(configuredMode, "0") == 0
    )
    {
        return 0;
    }

    if (std::strcmp(configuredMode, "timing") == 0)
    {
        developmentProbe.mode = DevelopmentProbeMode::timing;
        developmentProbe.modeName = "timing";
    }
    else if
    (
        std::strcmp(configuredMode, "full") == 0
     || std::strcmp(configuredMode, "1") == 0
    )
    {
        developmentProbe.mode = DevelopmentProbeMode::full;
        developmentProbe.modeName = "full";
    }
    else
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "invalid UGKP_DEV_PROBE_MODE '%s' (expected off, timing, or full)",
            configuredMode
        );
        return 1;
    }

    const char* const configuredLog = std::getenv("UGKP_DEV_PROBE_LOG");
    if (configuredLog == nullptr || *configuredLog == '\0')
    {
        setLastErrorText
        (
            "UGKP_DEV_PROBE_LOG is required when UGKP_DEV_PROBE_MODE is enabled"
        );
        shutdownDevelopmentProbe();
        return 1;
    }
    if (configuredLog[0] != '/')
    {
        setLastErrorText
        (
            "UGKP_DEV_PROBE_LOG must be an absolute path"
        );
        shutdownDevelopmentProbe();
        return 1;
    }

    if (const char* configuredInterval = std::getenv("UGKP_DEV_PROBE_INTERVAL"))
    {
        errno = 0;
        char* end = nullptr;
        const unsigned long long interval =
            std::strtoull(configuredInterval, &end, 10);
        if
        (
            errno != 0
         || end == configuredInterval
         || *end != '\0'
         || interval == 0
        )
        {
            std::snprintf
            (
                lastError,
                sizeof(lastError),
                "invalid UGKP_DEV_PROBE_INTERVAL '%s'",
                configuredInterval
            );
            shutdownDevelopmentProbe();
            return 1;
        }
        developmentProbe.interval = interval;
    }

    developmentProbe.failOnNonFinite = developmentProbeBoolean
    (
        std::getenv("UGKP_DEV_PROBE_FAIL_ON_NONFINITE")
    );
    developmentProbe.pid = static_cast<int>(::getpid());
    developmentProbe.logPath = developmentProbePathForPid(configuredLog);
    if (const char* value = std::getenv("UGKP_DEV_PROBE_RUN_ID"))
    {
        developmentProbe.runId = value;
    }
    if (const char* value = std::getenv("UGKP_DEV_PROBE_VARIANT"))
    {
        developmentProbe.variant = value;
    }

    developmentProbe.log = std::fopen(developmentProbe.logPath.c_str(), "a+");
    if (developmentProbe.log == nullptr)
    {
        std::snprintf
        (
            lastError,
            sizeof(lastError),
            "cannot open UGKP development probe log '%s': %s",
            developmentProbe.logPath.c_str(),
            std::strerror(errno)
        );
        shutdownDevelopmentProbe();
        return 1;
    }
    std::setvbuf(developmentProbe.log, nullptr, _IOLBF, 0);
    if (std::fseek(developmentProbe.log, 0, SEEK_END) != 0)
    {
        setLastErrorText("cannot seek UGKP development probe log");
        shutdownDevelopmentProbe();
        return 1;
    }
    const long logBytes = std::ftell(developmentProbe.log);
    if (logBytes < 0)
    {
        setLastErrorText("cannot query UGKP development probe log length");
        shutdownDevelopmentProbe();
        return 1;
    }
    if (logBytes == 0 && writeDevelopmentProbeHeader() != 0)
    {
        shutdownDevelopmentProbe();
        return 1;
    }

    cudaError_t err = cudaEventCreate(&developmentProbe.totalStartEvent);
    if (err == cudaSuccess)
    {
        err = cudaEventCreate(&developmentProbe.totalStopEvent);
    }
    for
    (
        int stage = 0;
        err == cudaSuccess && stage < ProbeStageCount;
        ++stage
    )
    {
        for
        (
            int occurrence = 0;
            err == cudaSuccess && occurrence < ProbeMaxOccurrences;
            ++occurrence
        )
        {
            err = cudaEventCreate
            (
                &developmentProbe.stageStartEvents[stage][occurrence]
            );
            if (err == cudaSuccess)
            {
                err = cudaEventCreate
                (
                    &developmentProbe.stageStopEvents[stage][occurrence]
                );
            }
        }
    }
    if (err != cudaSuccess)
    {
        setLastError("cudaEventCreate UGKP development probe", err);
        shutdownDevelopmentProbe();
        return 1;
    }

    if (developmentProbeFullValidation())
    {
        const cudaError_t err = cudaMalloc
        (
            reinterpret_cast<void**>(&developmentProbe.deviceSummary),
            sizeof(DevelopmentProbeDeviceSummary)
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaMalloc UGKP development probe summary", err);
            shutdownDevelopmentProbe();
            return 1;
        }
    }

    try
    {
        developmentProbe.occupancy.resize(static_cast<size_t>(owner->nCells));
    }
    catch (...)
    {
        setLastErrorText("cannot allocate UGKP development probe occupancy mirror");
        shutdownDevelopmentProbe();
        return 1;
    }
    developmentProbe.owner = owner;
    return 0;
}
