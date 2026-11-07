#pragma once
#include "CompensatedTime.H"
#include <cstddef>
#if defined(__CUDACC__)
#define UGKP_SLICE_HD __host__ __device__
#else
#define UGKP_SLICE_HD
#endif
namespace ugkp {
struct IntervalSliceCursor {CompensatedTime time{};std::size_t record=0;};
enum class IntervalSliceError : int {None,ExceedsHistory,RadiationLayout};
struct IntervalSlicePlan {
    IntervalSliceCursor begin{},end{};
    CompensatedTime target{};
    std::size_t past=0;
    IntervalSliceError error=IntervalSliceError::None;
    UGKP_SLICE_HD bool hasRecords()const{return begin.record<past;}
};
// This is the temporal part of consuming extensive accepted transfers. The
// history owns the intervals; callers keep old/candidate accumulators distinct
// until the physical consumer accepts a slice. A final piece uses the original
// transfer minus prior pieces, rather than another rounded time fraction.
template<class History>
UGKP_SLICE_HD inline IntervalSlicePlan planIntervalSlice(const History& h,
    IntervalSliceCursor cursor,CompensatedTime target){
    IntervalSlicePlan p;p.begin=p.end=cursor;p.target=target;p.past=cursor.record;
    while(p.end.time<target){
        if(p.end.record>=h.frames){p.error=IntervalSliceError::ExceedsHistory;return p;}
        const auto end=h.interval[p.end.record].end;
        p.end.time=target<end?target:end;p.past=p.end.record+1;
        if(p.end.time==end)++p.end.record;
    }
    return p;
}
template<class History>
UGKP_SLICE_HD inline CompensatedTime sliceRecordStop(const History& h,const IntervalSlicePlan& p,std::size_t r){
    return p.target<h.interval[r].end?p.target:h.interval[r].end;
}
template<class History>
UGKP_SLICE_HD inline double sliceRecordFraction(const History& h,const IntervalSlicePlan& p,std::size_t r){
    const auto begin=r==p.begin.record?p.begin.time:h.interval[r-1].end;
    return (sliceRecordStop(h,p,r)-begin).value()/h.interval[r].actualDt;
}
}
#undef UGKP_SLICE_HD
