#ifndef UGKP_GPU_EXTENDED_ACCUMULATOR_H
#define UGKP_GPU_EXTENDED_ACCUMULATOR_H
#include <cstdint>
#include <cstring>
#if defined(__CUDACC__)
#define UGKP_EXT_HD __host__ __device__
#else
#define UGKP_EXT_HD
#endif
namespace ugkp { namespace numerics {
// Ordered binary arithmetic with a 64-bit significand, matching the x86
// extended accumulator used by the original material inventory accounting.
// Integer guard/sticky bits retain cancellation and range beyond binary64;
// neither atomics nor a different summation order are introduced.
struct ExtendedAccumulator {
    std::uint64_t significand=0;
    int exponent=0;
    bool negative=false;
    unsigned kind=0; // finite, infinity, NaN

    UGKP_EXT_HD static std::uint64_t bits(double x) {
#if defined(__CUDA_ARCH__)
        return static_cast<std::uint64_t>(__double_as_longlong(x));
#else
        std::uint64_t r;std::memcpy(&r,&x,sizeof(r));return r;
#endif
    }
    UGKP_EXT_HD static double real(std::uint64_t x) {
#if defined(__CUDA_ARCH__)
        return __longlong_as_double(static_cast<long long>(x));
#else
        double r;std::memcpy(&r,&x,sizeof(r));return r;
#endif
    }
    UGKP_EXT_HD static int leadingZeros(std::uint64_t x) {
#if defined(__CUDA_ARCH__)
        return __clzll(x);
#else
        return __builtin_clzll(x);
#endif
    }
    UGKP_EXT_HD ExtendedAccumulator()=default;
    UGKP_EXT_HD explicit ExtendedAccumulator(double x) {
        const auto u=bits(x);negative=(u>>63)!=0;
        const unsigned e=unsigned((u>>52)&2047);
        const auto fraction=u&0xfffffffffffffULL;
        if(e==2047){kind=fraction?2:1;return;}
        if(e){significand=((1ULL<<52)|fraction)<<11;exponent=int(e)-1023;}
        else if(fraction){const int shift=leadingZeros(fraction);significand=fraction<<shift;exponent=-1011-shift;}
    }
    struct Pair {std::uint64_t hi,lo;};
    UGKP_EXT_HD static Pair shifted(std::uint64_t value,unsigned distance) {
        if(!distance)return {value,0};
        if(distance<64)return {value>>distance,value<<(64-distance)};
        if(distance==64)return {0,value};
        if(distance<128){const unsigned d=distance-64;
            return {0,(value>>d)|((value<<(64-d))!=0)};}
        return {0,std::uint64_t(value!=0)};
    }
    UGKP_EXT_HD void rounded(Pair x,int e) {
        if(x.lo>0x8000000000000000ULL
            ||(x.lo==0x8000000000000000ULL&&(x.hi&1))){
            ++x.hi;
            if(!x.hi){x.hi=0x8000000000000000ULL;++e;}
        }
        significand=x.hi;exponent=e;
    }
    UGKP_EXT_HD int compareMagnitude(const ExtendedAccumulator& b)const {
        if(kind==2||b.kind==2)return 0;
        if(kind==1)return b.kind==1?0:1;
        if(b.kind==1)return -1;
        if(!significand)return b.significand?-1:0;
        if(!b.significand)return 1;
        if(exponent!=b.exponent)return exponent>b.exponent?1:-1;
        return significand==b.significand?0:(significand>b.significand?1:-1);
    }
    UGKP_EXT_HD void addValue(ExtendedAccumulator b) {
        if(kind==2||b.kind==2){kind=2;return;}
        if(kind||b.kind){
            if(kind==1&&b.kind==1&&negative!=b.negative)kind=2;
            else if(b.kind)*this=b;
            return;
        }
        if(!b.significand){if(!significand)negative=negative&&b.negative;return;}
        if(!significand){*this=b;return;}
        if(compareMagnitude(b)<0){auto temporary=*this;*this=b;b=temporary;}
        const Pair small=shifted(b.significand,unsigned(exponent-b.exponent));
        if(negative==b.negative){
            Pair sum{significand+small.hi,small.lo};
            const bool carry=sum.hi<significand;
            if(carry){sum={0x8000000000000000ULL|(sum.hi>>1),
                (sum.hi<<63)|(sum.lo>>1)|(sum.lo&1)};++exponent;}
            rounded(sum,exponent);
        }else{
            Pair difference{significand-small.hi-std::uint64_t(small.lo!=0),0-small.lo};
            if(!difference.hi&&!difference.lo){significand=0;exponent=0;negative=false;return;}
            if(!difference.hi){difference.hi=difference.lo;difference.lo=0;exponent-=64;}
            const int shift=leadingZeros(difference.hi);
            if(shift){difference.hi=(difference.hi<<shift)|(difference.lo>>(64-shift));
                difference.lo<<=shift;exponent-=shift;}
            rounded(difference,exponent);
        }
    }
    UGKP_EXT_HD void add(double x){addValue(ExtendedAccumulator(x));}
    UGKP_EXT_HD static Pair product(std::uint64_t a,std::uint64_t b) {
#if defined(__CUDA_ARCH__)
        return {__umul64hi(a,b),a*b};
#else
        const std::uint64_t mask=0xffffffffULL;
        const auto a0=a&mask,a1=a>>32,b0=b&mask,b1=b>>32;
        const auto w0=a0*b0;
        const auto t=a1*b0+(w0>>32);
        const auto w1=(t&mask)+a0*b1;
        return {a1*b1+(t>>32)+(w1>>32),(w1<<32)|(w0&mask)};
#endif
    }
    UGKP_EXT_HD ExtendedAccumulator multiplied(double factor)const {
        ExtendedAccumulator result=*this,b(factor);result.negative=negative!=b.negative;
        if(kind==2||b.kind==2){result.kind=2;return result;}
        if(kind||b.kind){
            if((!kind&&!significand)||(!b.kind&&!b.significand))result.kind=2;
            else result.kind=1;
            return result;
        }
        if(!significand||!b.significand){result.significand=0;result.exponent=0;return result;}
        Pair x=product(significand,b.significand);int e=exponent+b.exponent+1;
        if(!(x.hi>>63)){x.hi=(x.hi<<1)|(x.lo>>63);x.lo<<=1;--e;}
        result.rounded(x,e);return result;
    }
    UGKP_EXT_HD double value()const {
        const std::uint64_t sign=std::uint64_t(negative)<<63;
        if(kind)return real(sign|0x7ff0000000000000ULL|(kind==2?0x8000000000000ULL:0));
        if(!significand)return real(sign);
        int e=exponent;
        if(e>1023)return real(sign|0x7ff0000000000000ULL);
        if(e>=-1022){
            std::uint64_t m=significand>>11;
            const auto rest=significand&2047;
            if(rest>1024||(rest==1024&&(m&1)))++m;
            if(m==(1ULL<<53)){m>>=1;++e;}
            if(e>1023)return real(sign|0x7ff0000000000000ULL);
            return real(sign|(std::uint64_t(e+1023)<<52)|(m&0xfffffffffffffULL));
        }
        const int shift=-e-1011;
        std::uint64_t m=0;
        if(shift<64){
            m=significand>>shift;
            const auto rest=significand&((1ULL<<shift)-1),half=1ULL<<(shift-1);
            if(rest>half||(rest==half&&(m&1)))++m;
        }else if(shift==64&&significand>0x8000000000000000ULL)m=1;
        return real(sign|m);
    }
};
}}
#undef UGKP_EXT_HD
#endif
