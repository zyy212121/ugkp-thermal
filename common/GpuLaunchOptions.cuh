#pragma once
// Two independent launch decisions. Named types prevent reversed bool arguments.
enum class MomentPayload { momentsOnly, gatherSurvivors };
enum class MomentRecovery { completeHere, deferToAdvance };
struct SegmentedMomentOptions
{
    MomentPayload payload;
    MomentRecovery recovery;
};
