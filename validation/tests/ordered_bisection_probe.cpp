#include "gpu/OrderedBisectionProbe.H"
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
struct Value {double residual;bool valid;};
Value evaluate(int mode,double x){
    if(mode==2&&x>.62&&x<.69)return {0,false};
    if(mode==3&&x>.48&&x<.52)return {0,false};
    if(mode==1)return {(x-.1234567)*(x-.4567891)*(x-.7891234),true};
    return {x-.23456789123456789,true};
}
struct Result {std::vector<double> consumed;double value=0;bool valid=false;};
Result reference(int mode,double lo,double hi){
    Result out;auto left=evaluate(mode,lo);
    for(int i=0;i<100;++i){double x=.5*(lo+hi);out.consumed.push_back(x);auto v=evaluate(mode,x);
        if(!v.valid)return out;
        out.value=x;out.valid=true;
        if(v.residual==0||x==lo||x==hi||hi-lo<1e-13)return out;
        out.valid=false;
        if((v.residual>0)==(left.residual>0)){lo=x;left=v;}else hi=x;
    }
    return out;
}
template<int Depth> Result batched(int mode,double lo,double hi){
    Result out;auto left=evaluate(mode,lo);ugkp::gpu::OrderedBisectionProbe<double,Depth> cache;
    Value values[decltype(cache)::count];
    for(int i=0;i<100;++i){double x=.5*(lo+hi);int selected=cache.find(x);
        if(selected<0){cache.prepare(lo,hi);for(int j=0;j<decltype(cache)::count;++j)values[j]=evaluate(mode,cache.temperature[j]);selected=0;}
        out.consumed.push_back(x);auto v=values[selected];if(!v.valid)return out;
        out.value=x;out.valid=true;
        if(v.residual==0||x==lo||x==hi||hi-lo<1e-13)return out;
        out.valid=false;
        if((v.residual>0)==(left.residual>0)){lo=x;left=v;}else hi=x;
    }
    return out;
}
int main(){
    for(int mode=0;mode<4;++mode)for(int variant=0;variant<3;++variant){
        const double lo=variant==2?.2345678912345678:0.;
        const double hi=variant==2?std::nextafter(lo,1.):variant==1?.75:1.;
        const auto a=reference(mode,lo,hi),b=batched<2>(mode,lo,hi),c=batched<3>(mode,lo,hi);
        assert(a.consumed==b.consumed&&a.consumed==c.consumed);
        assert(a.valid==b.valid&&a.valid==c.valid&&a.value==b.value&&a.value==c.value);
    }
    std::puts("PASS ordered probes preserve consumed root trajectory, multiple roots, ignored/selected holes and adjacent endpoints");
}
