#ifndef CHMT_COUPLING_WINDOWTIME_H
#define CHMT_COUPLING_WINDOWTIME_H
#include <cmath>
namespace chmt {
// Transient time measured from a coupling-window origin. Keeping the residual
// separate lets a paid increment remain visible below either clock's spacing.
struct WindowTime {
    long double sum=0,correction=0;
    WindowTime()=default;
    WindowTime(long double value):sum(value){}
    void add(long double value){
        const long double next=sum+value;
        const long double residual=std::fabs(sum)>=std::fabs(value)?(sum-next)+value:(value-next)+sum;
        const long double tail=correction+residual;
        sum=next+tail;correction=tail-(sum-next);
    }
    WindowTime& operator+=(long double value){add(value);return *this;}
    WindowTime& operator-=(long double value){add(-value);return *this;}
};
inline WindowTime operator+(WindowTime a,long double b){a+=b;return a;}
inline WindowTime operator-(WindowTime a,long double b){a-=b;return a;}
inline WindowTime operator-(WindowTime a,const WindowTime& b){a.add(-b.sum);a.add(-b.correction);return a;}
inline bool operator<(const WindowTime& a,const WindowTime& b){return a.sum<b.sum||(a.sum==b.sum&&a.correction<b.correction);}
inline bool operator>(const WindowTime& a,const WindowTime& b){return b<a;}
inline bool operator==(const WindowTime& a,const WindowTime& b){return a.sum==b.sum&&a.correction==b.correction;}
inline bool operator!=(const WindowTime& a,const WindowTime& b){return !(a==b);}
inline bool operator<=(const WindowTime& a,const WindowTime& b){return !(b<a);}
inline bool operator>=(const WindowTime& a,const WindowTime& b){return !(a<b);}
inline long double windowDifference(const WindowTime& end,const WindowTime& begin){const auto d=end-begin;return d.sum+d.correction;}
inline double windowDuration(const WindowTime& end,const WindowTime& begin){return static_cast<double>(windowDifference(end,begin));}
inline double boundedWindowDuration(const WindowTime& end,const WindowTime& begin){
    const auto span=end-begin;double dt=windowDuration(end,begin);
    if(WindowTime(dt)>span)dt=std::nextafter(dt,0.0);
    return dt;
}
inline double windowAbsolute(double origin,const WindowTime& elapsed){WindowTime time(origin);time.add(elapsed.sum);time.add(elapsed.correction);return static_cast<double>(time.sum+time.correction);}
inline WindowTime windowOffset(double origin,double time){WindowTime value(time);value-=origin;return value;}
inline double windowFraction(const WindowTime& time,const WindowTime& begin,const WindowTime& end,
    double stamp,double beginStamp,double endStamp){
    const auto numerator=time-begin,denominator=end-begin;
    // Retain the established expression/order whenever its binary64 differences
    // are exactly the paid local differences.
    if(numerator==WindowTime(stamp-beginStamp)&&denominator==WindowTime(endStamp-beginStamp))
        return (stamp-beginStamp)/(endStamp-beginStamp);
    return static_cast<double>((numerator.sum+numerator.correction)/(denominator.sum+denominator.correction));
}
}
#endif
