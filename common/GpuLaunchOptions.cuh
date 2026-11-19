#pragma once

enum class MomentPayload { momentsOnly, gatherSurvivors };
enum class MomentRecovery { completeHere, deferToAdvance };
struct SegmentedMomentOptions
{
    MomentPayload payload;
    MomentRecovery recovery;
};
