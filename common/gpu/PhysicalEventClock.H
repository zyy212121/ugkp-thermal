#pragma once
#include <cfloat>
#include <cmath>

namespace ugkp { namespace gpu {
#ifdef __CUDACC__
#define UGKP_CLOCK_HD __host__ __device__
#else
#define UGKP_CLOCK_HD
#endif

UGKP_CLOCK_HD inline bool sameEventTime(double a, double b)
{
    const double scale = ::fmax(::fabs(a), ::fabs(b));
    return a == b || (scale <= DBL_MAX && ::fabs(a-b) <= 8*DBL_EPSILON*scale);
}

UGKP_CLOCK_HD inline bool reachedEventTime(double now, double target)
{
    return now >= target || sameEventTime(now,target);
}



UGKP_CLOCK_HD inline bool reachedEventTime(double now, double target, double interval)
{
    return now >= target || (sameEventTime(now,target) && target-now < 0.5*interval);
}

UGKP_CLOCK_HD inline double nextPeriodicTime(double origin, double period, double now)
{
    double tick = ::floor((now-origin)/period) + 1;
    double target = origin + tick*period;
    if (reachedEventTime(now,target,period))
        target = origin + (tick+1)*period;
    return target;
}


UGKP_CLOCK_HD inline bool radiationDueAt(double begin,double end,double origin,double heatPeriod,
    unsigned long long frequency,unsigned long long exchange,bool output)
{
    if(!frequency || !(end>begin))return false;
    if(output || frequency==1)return true;
    if(!(heatPeriod>0))return exchange%frequency==0;
    return reachedEventTime(end,nextPeriodicTime(origin,heatPeriod*frequency,begin),end-begin);
}

UGKP_CLOCK_HD inline double eventDeadline(double begin, double target, double limit)
{
    return reachedEventTime(target,limit,target-begin) ? limit : target;
}

struct PhysicalStep { double dt, end; };

UGKP_CLOCK_HD inline PhysicalStep stepToEvent(double now, double nominalDt, double deadline)
{
    const double remaining = deadline-now;
    const double trialEnd = now+nominalDt;
    if (reachedEventTime(trialEnd,deadline,nominalDt))
        return {remaining, deadline};


    const double dt = remaining < 1.5*nominalDt ? 0.5*remaining : nominalDt;
    return {dt, now+dt};
}

#undef UGKP_CLOCK_HD
}}
