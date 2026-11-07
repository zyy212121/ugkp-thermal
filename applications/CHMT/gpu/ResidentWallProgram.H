#ifndef CHMT_RESIDENT_WALL_PROGRAM_H
#define CHMT_RESIDENT_WALL_PROGRAM_H
#include "gpu/GasWindowProgram.H"
#include "../../../common/CompensatedTime.H"
namespace chmt {
struct ResidentWallKnotTime {ugkp::CompensatedTime local{};Real absolute=0;};
struct ResidentWallClock {ugkp::CompensatedTime elapsed{};Real origin=0;std::uint64_t sequence=0;};
// Conversion is only used when installing a program. Numerical advancement and
// knot selection use the same FP64 representation in every device invocation.
inline ugkp::CompensatedTime residentWindowTime(const WindowTime& time){
    ugkp::CompensatedTime result(static_cast<double>(time.sum));
    result.add(static_cast<double>((time.sum-static_cast<long double>(result.sum))+time.correction));return result;
}
CHMT_HD inline GasWallProgramPosition locateResidentGasWallProgram(const ResidentWallKnotTime* knots,
    int count,Real origin,ugkp::CompensatedTime time){
    // Program shape/order and coverage are validated by the existing window
    // contract when installed. This is upper_bound, including exact knot hits.
    int lo=0,hi=count;
    while(lo<hi){const int mid=lo+(hi-lo)/2;if(time<knots[mid].local)hi=mid;else lo=mid+1;}
    if(lo==count)return {count-1,count-1,0};
    const int lower=lo-1;const auto numerator=time-knots[lower].local,denominator=knots[lo].local-knots[lower].local;
    const Real stamp=ugkp::absoluteTime(origin,time),a=stamp-knots[lower].absolute,b=knots[lo].absolute-knots[lower].absolute;
    const Real fraction=numerator==ugkp::CompensatedTime(a)&&denominator==ugkp::CompensatedTime(b)
        ?a/b:ugkp::compensatedTimeRatio(numerator,denominator);
    return {lower,lo,fraction};
}
}
#endif
