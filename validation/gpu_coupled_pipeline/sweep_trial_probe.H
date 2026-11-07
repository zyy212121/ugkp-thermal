#pragma once
#include "../../common/gpu/SweepTrialMath.H"
#include <vector>
struct SweepTrialInput {double origin,correction,alpha;};
std::vector<ugkp::gpu::geometry::SweepTrialCoordinate> probeSweepTrial(const std::vector<SweepTrialInput>&);
