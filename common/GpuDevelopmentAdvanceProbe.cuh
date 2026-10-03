#pragma once
// Shared stage timing, failure publication and final sample lifecycle.
// The detailed scheduling schema is a compile-time field capability.
template<bool IncludesScheduling, class State, class Sample>
inline void publishDevelopmentProbeScheduling(const State* state, Sample& sample)
{
    if constexpr (IncludesScheduling)
    {
        sample.blockExponent = 0;
        sample.blockThreads = state->reductionBlockThreads;
        sample.smCount = state->multiprocessorCount;
        sample.particleBlocksPerSm = state->particleBlocksPerSm;
        sample.lightBlocksPerSm = state->lightBlocksPerSm;
        sample.heavyBlocksPerSm = state->heavyBlocksPerSm;
        sample.heavyReductionEnabled = state->csrHeavyReductionEnabled;
    }
}

class DevelopmentAdvanceProbe
{
    DeviceState* state_ = nullptr;
    unsigned long long step_ = 0;
    double simulationTime_ = 0.0;
    double dt_ = 0.0;
    const char* currentStage_ = "advance_begin";
    bool enabled_ = false;
    bool sampled_ = false;
    bool failed_ = false;
    bool completed_ = false;
    bool rowWritten_ = false;

public:
    DevelopmentAdvanceProbe
    (
        DeviceState* state,
        const double dt,
        const double simulationTime
    )
    :
        state_(state),
        simulationTime_(simulationTime),
        dt_(dt),
        enabled_(developmentProbeEnabled())
    {
        if (!enabled_)
        {
            return;
        }

        step_ = ++developmentProbe.advanceIndex;
        sampled_ = (step_ % developmentProbe.interval) == 0;
        if (sampled_)
        {
            for (int stage = 0; stage < ProbeStageCount; ++stage)
            {
                developmentProbe.stageOccurrenceCount[stage] = 0;
                for
                (
                    int occurrence = 0;
                    occurrence < ProbeMaxOccurrences;
                    ++occurrence
                )
                {
                    developmentProbe.stageOccurrenceExecuted[stage][occurrence] =
                        false;
                }
            }
            const cudaError_t err = cudaEventRecord
            (
                developmentProbe.totalStartEvent,
                0
            );
            if (err != cudaSuccess)
            {
                setLastError("cudaEventRecord UGKP development probe start", err);
                failed_ = true;
            }
        }
    }

    ~DevelopmentAdvanceProbe()
    {
        if (!enabled_ || completed_ || rowWritten_)
        {
            return;
        }

        DevelopmentProbeSample sample;
        sample.step = step_;
        sample.simulationTime = simulationTime_;
        sample.dt = dt_;
        sample.status = "error";
        sample.errorStage = currentStage_;
        sample.errorMessage = lastError;
        if (state_ != nullptr)
        {
            sample.nCells = state_->nCells;
        publishDevelopmentProbeScheduling<developmentProbeIncludesScheduling>(state_, sample);
            sample.particlePath = state_->particlesMayBePresent ? 1 : 0;
            sample.particleCapacity = state_->particleCapacity;
        }
        char preservedError[sizeof(lastError)]{};
        std::snprintf
        (
            preservedError,
            sizeof(preservedError),
            "%s",
            lastError
        );
        (void)writeDevelopmentProbeSample(sample);
        std::snprintf(lastError, sizeof(lastError), "%s", preservedError);
        rowWritten_ = true;
    }

    bool failed() const
    {
        return failed_;
    }

    void enter(const DevelopmentProbeStage stage)
    {
        currentStage_ = developmentProbeStageNames[stage];
        if (!sampled_ || failed_)
        {
            return;
        }
        const int occurrence = developmentProbe.stageOccurrenceCount[stage];
        if (occurrence < 0 || occurrence >= ProbeMaxOccurrences)
        {
            setLastErrorText
            (
                "UGKP development probe stage occurrence capacity exceeded"
            );
            failed_ = true;
            return;
        }
        const cudaError_t err = cudaEventRecord
        (
            developmentProbe.stageStartEvents[stage][occurrence],
            0
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventRecord UGKP development probe stage start", err);
            failed_ = true;
        }
    }

    int leave
    (
        const DevelopmentProbeStage stage,
        const bool executed = true
    )
    {
        if (!sampled_)
        {
            return 0;
        }
        if (failed_)
        {
            return 1;
        }
        const int occurrence = developmentProbe.stageOccurrenceCount[stage];
        if (occurrence < 0 || occurrence >= ProbeMaxOccurrences)
        {
            setLastErrorText
            (
                "UGKP development probe stage occurrence capacity exceeded"
            );
            failed_ = true;
            return 1;
        }
        const cudaError_t err = cudaEventRecord
        (
            developmentProbe.stageStopEvents[stage][occurrence],
            0
        );
        if (err != cudaSuccess)
        {
            setLastError("cudaEventRecord UGKP development probe stage", err);
            failed_ = true;
            return 1;
        }
        developmentProbe.stageOccurrenceExecuted[stage][occurrence] = executed;
        developmentProbe.stageOccurrenceCount[stage] = occurrence + 1;
        return 0;
    }

    int finish(const bool particlePath)
    {
        if (!enabled_ || !sampled_)
        {
            completed_ = true;
            return 0;
        }

        currentStage_ = "probe_collect";
        const cudaError_t stopError = cudaEventRecord
        (
            developmentProbe.totalStopEvent,
            0
        );
        if (stopError != cudaSuccess)
        {
            setLastError
            (
                "cudaEventRecord UGKP development probe total stop",
                stopError
            );
            return 1;
        }
        DevelopmentProbeSample sample;
        sample.step = step_;
        sample.simulationTime = simulationTime_;
        sample.dt = dt_;
        sample.nCells = state_->nCells;
        publishDevelopmentProbeScheduling<developmentProbeIncludesScheduling>(state_, sample);
        sample.particlePath = particlePath ? 1 : 0;
        sample.particleCapacity = state_->particleCapacity;
        if (collectDevelopmentProbeSample(state_, particlePath, sample) != 0)
        {
            return 1;
        }
        if (writeDevelopmentProbeSample(sample) != 0)
        {
            rowWritten_ = true;
            return 1;
        }
        rowWritten_ = true;

        if
        (
            developmentProbe.failOnNonFinite
         && sample.badFieldMask != 0
        )
        {
            setLastErrorText
            (
                "UGKP development probe found non-finite or invalid resident state"
            );
            return 1;
        }

        completed_ = true;
        return 0;
    }
};
