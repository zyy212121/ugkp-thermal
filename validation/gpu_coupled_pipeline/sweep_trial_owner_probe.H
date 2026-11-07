#include "../../common/gpu/SweepTrial.H"
struct SweepTrialOwnerProbe {std::vector<ugkp::gpu::geometry::Point> points;ugkp::gpu::geometry::SweepTrialStatus status;};
SweepTrialOwnerProbe probeSweepTrialOwner(ugkp::gpu::geometry::ResidentSweepTrial&,const std::vector<ugkp::gpu::geometry::Point>&,const std::vector<ugkp::gpu::geometry::Point>&,int);
