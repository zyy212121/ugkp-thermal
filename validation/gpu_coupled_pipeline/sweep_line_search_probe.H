#include "../../common/gpu/SweepLineSearch.H"
struct SweepSearchProbe {std::vector<ugkp::gpu::geometry::Point> points;ugkp::gpu::geometry::SweepLineSearchStatus status;};
std::vector<SweepSearchProbe> probeSweepSearch(const ugkp::gpu::geometry::SweepResidualDefinition&,const std::vector<ugkp::gpu::geometry::Point>&,const std::vector<ugkp::gpu::geometry::Point>&,double,int);
