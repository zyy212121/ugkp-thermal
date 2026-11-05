#ifndef UGKP_GPU_EXTENDED_GEOMETRY_MATH_H
#define UGKP_GPU_EXTENDED_GEOMETRY_MATH_H
#include "PolyhedralGeometryMath.H"
#include <limits>
#ifdef __CUDACC__
#define UGKP_WIDE_HD __host__ __device__
#else
#define UGKP_WIDE_HD
#endif
namespace ugkp { namespace gpu { namespace geometry { namespace extended {
// Scaled double-double arithmetic keeps the original extended-range geometry
// formulas on CUDA without narrowing intermediate determinants to FP64. The
// mantissa has approximately 106 bits; a separate exponent covers all products
// of finite input coordinates. Final distances retain the public FP64 format.
constexpr double invalidDistance=std::numeric_limits<double>::quiet_NaN();
struct Wide {
    double high=0,low=0;int exponent=0;
    UGKP_WIDE_HD Wide():high(0),low(0),exponent(0){}
    UGKP_WIDE_HD explicit Wide(double x){high=::frexp(x,&exponent);}
};
UGKP_WIDE_HD inline Wide normalized(double high,double low,int exponent){
    const double sum=high+low,part=sum-high;const double error=(high-(sum-part))+(low-part);
    Wide r;if(sum==0){r=Wide(error);r.exponent+=exponent;return r;}
    int shift=0;r.high=::frexp(sum,&shift);r.low=::ldexp(error,-shift);r.exponent=exponent+shift;return r;
}
UGKP_WIDE_HD inline Wide operator-(Wide a){a.high=-a.high;a.low=-a.low;return a;}
UGKP_WIDE_HD inline Wide operator+(Wide a,Wide b){
    if(a.high==0)return b;if(b.high==0)return a;
    if(a.exponent<b.exponent){const Wide temp=a;a=b;b=temp;}
    const int shift=b.exponent-a.exponent;if(shift < -120)return a;
    const double bh=::ldexp(b.high,shift),bl=::ldexp(b.low,shift);
    const double sum=a.high+bh,part=sum-a.high;
    const double error=(a.high-(sum-part))+(bh-part)+a.low+bl;
    return normalized(sum,error,a.exponent);
}
UGKP_WIDE_HD inline Wide operator-(Wide a,Wide b){return a+(-b);}
UGKP_WIDE_HD inline Wide operator*(Wide a,Wide b){
    const double product=a.high*b.high;
    const double error=::fma(a.high,b.high,-product)+a.high*b.low+a.low*b.high+a.low*b.low;
    return normalized(product,error,a.exponent+b.exponent);
}
UGKP_WIDE_HD inline double value(Wide x){
    // Scale through the half value near the FP64 overflow boundary so a low
    // negative mantissa part can still produce the largest finite distance.
    if(x.exponent==1025)return (::ldexp(x.high,1024)+::ldexp(x.low,1024))*2;
    return ::ldexp(x.high,x.exponent)+::ldexp(x.low,x.exponent);
}
UGKP_WIDE_HD inline Wide operator/(Wide a,Wide b){
    const int scale=a.exponent-b.exponent;a.exponent=b.exponent=0;
    Wide result(a.high/b.high);
    for(int i=0;i<2;++i){const Wide remainder=a-b*result;result=result+Wide(value(remainder)/b.high);}
    result.exponent+=scale;return result;
}
UGKP_WIDE_HD inline Wide squareRoot(Wide x){
    if(x.high==0)return x;
    const int half=x.exponent/2,remainder=x.exponent-2*half;
    Wide result(::sqrt(::ldexp(x.high,remainder)));result.exponent+=half;
    for(int i=0;i<2;++i)result=result+(x-result*result)/(Wide(2)*result);
    return result;
}
UGKP_WIDE_HD inline bool finite(Wide x){return std::isfinite(x.high)&&std::isfinite(x.low);}
UGKP_WIDE_HD inline bool less(Wide a,Wide b){return (a-b).high<0;}
UGKP_WIDE_HD inline Wide absolute(Wide x){return x.high<0?-x:x;}
struct Vector {Wide x,y,z;};
UGKP_WIDE_HD inline Vector difference(Point a,Point b){return {Wide(a.x)-Wide(b.x),Wide(a.y)-Wide(b.y),Wide(a.z)-Wide(b.z)};}
UGKP_WIDE_HD inline Wide inner(Vector a,Vector b){return a.x*b.x+a.y*b.y+a.z*b.z;}
UGKP_WIDE_HD inline Vector outer(Vector a,Vector b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
UGKP_WIDE_HD inline Wide segmentDistance(Point p,Point x,Point y){
    const Vector edge=difference(y,x),offset=difference(p,x);const Wide length=inner(edge,edge);
    Wide t;if(length.high>0){t=inner(offset,edge)/length;if(t.high<0)t=Wide(0);else if(less(Wide(1),t))t=Wide(1);}
    const Vector separation{offset.x-t*edge.x,offset.y-t*edge.y,offset.z-t*edge.z};
    return squareRoot(inner(separation,separation));
}
UGKP_WIDE_HD inline double triangleDistance(Point p,Point a,Point b,Point c){
    if(!geometry::finite(p)||!geometry::finite(a)||!geometry::finite(b)||!geometry::finite(c))return invalidDistance;
    const Vector ab=difference(b,a),ac=difference(c,a),ap=difference(p,a),normal=outer(ab,ac);
    const Wide squared=inner(normal,normal);
    if(!finite(squared))return invalidDistance;
    if(squared.high>0){
        const Wide first=inner(outer(ab,ap),normal),second=inner(outer(difference(c,b),difference(p,b)),normal),
            third=inner(outer(difference(a,c),difference(p,c)),normal);
        if(!finite(first)||!finite(second)||!finite(third))return invalidDistance;
        if(first.high>=0&&second.high>=0&&third.high>=0)return value(absolute(inner(ap,normal))/squareRoot(squared));
    }
    const Wide abDistance=segmentDistance(p,a,b),bcDistance=segmentDistance(p,b,c),caDistance=segmentDistance(p,c,a);
    if(!finite(abDistance)||!finite(bcDistance)||!finite(caDistance))return invalidDistance;
    const Wide last=less(caDistance,bcDistance)?caDistance:bcDistance;
    return value(less(last,abDistance)?last:abDistance);
}
}}}}
#undef UGKP_WIDE_HD
#endif
