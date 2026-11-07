#include "common/gpu/TrajectoryPolynomialMath.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace ugkp::gpu::geometry;
struct Control{double value[4];};
struct Bounds{double minimum,maximum;};
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::abort();}}
// Frozen native formula from CHMT Geometry.C at dafe4abb's parent source.
Bounds nativeBounds(const double* control){
    Bounds out;double& minimum=out.minimum;double& maximum=out.maximum;
    const auto minValue=[](double a,double b){return a<b?a:b;};
    const auto maxValue=[](double a,double b){return a>b?a:b;};
    minimum=minValue(control[0],control[3]);maximum=maxValue(control[0],control[3]);
    const long double a=3*(-static_cast<long double>(control[0])+3*static_cast<long double>(control[1])-3*static_cast<long double>(control[2])+control[3]);
    const long double b=6*(static_cast<long double>(control[0])-2*static_cast<long double>(control[1])+control[2]);
    const long double c=3*(static_cast<long double>(control[1])-control[0]);
    long double roots[2]={-1,-1};
    if(a==0){if(b!=0)roots[0]=-c/b;}
    else{const long double discriminant=b*b-4*a*c;
        if(discriminant>=0){const long double root=std::sqrt(discriminant);
            const long double q=-.5L*(b+(b>=0?root:-root));
            if(q!=0){roots[0]=q/a;roots[1]=c/q;}else roots[0]=-b/(2*a);}}
    for(long double t:roots)if(t>0&&t<1){long double value[4]={control[0],control[1],control[2],control[3]};
        for(int degree=3;degree>0;--degree)for(int k=0;k<degree;++k)value[k]=(1-t)*value[k]+t*value[k+1];
        minimum=minValue(minimum,static_cast<double>(value[0]));maximum=maxValue(maximum,static_cast<double>(value[0]));}
    return out;
}
__global__ void boundsKernel(const Control* input,int count,Bounds* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)
        trajectoryPolynomialBounds<extended::Wide>(input[i].value,output[i].minimum,output[i].maximum);
}
int main(){
    const double tiny=std::numeric_limits<double>::denorm_min(),large=std::numeric_limits<double>::max();
    std::vector<Control> controls={{{0.,-0.,0.,-0.}},{{1,1,1,1}},{{1,-1,-1,1}},{{1,-1./3,-1./3,1}},
        {{1,.5,1e-200,1e-300}},{{1e-300,1e300,-1e300,1e-300}},{{tiny,0,0,tiny}},{{large,-large,-large,large}}};
    std::mt19937_64 rng(32771);std::uniform_real_distribution<double> value(-1.,1.);
    for(int exponent:{-1000,-600,-60,0,60,600,1000})for(int sample=0;sample<4096;++sample){Control input;
        for(double& x:input.value)x=std::ldexp(value(rng),exponent);controls.push_back(input);}
    std::vector<Bounds> expected,actual(controls.size());
    for(const auto& c:controls){const auto native=nativeBounds(c.value);Bounds shared;
        trajectoryPolynomialBounds<long double>(c.value,shared.minimum,shared.maximum);
        assert(!std::memcmp(&native,&shared,sizeof(Bounds)));expected.push_back(native);}
    Control* input=nullptr;Bounds* output=nullptr;
    check(cudaMalloc(&input,controls.size()*sizeof(Control)));check(cudaMalloc(&output,actual.size()*sizeof(Bounds)));
    check(cudaMemcpy(input,controls.data(),controls.size()*sizeof(Control),cudaMemcpyHostToDevice));
    boundsKernel<<<(controls.size()+127)/128,128>>>(input,controls.size(),output);check(cudaGetLastError());
    check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(Bounds),cudaMemcpyDeviceToHost));
    long double maximum=0;int classifications=0;
    for(std::size_t i=0;i<controls.size();++i){long double scale=0;for(double x:controls[i].value)scale=std::max(scale,std::abs(static_cast<long double>(x)));
        assert(std::isfinite(actual[i].minimum)&&std::isfinite(actual[i].maximum));
        const long double error=std::max(std::abs(static_cast<long double>(actual[i].minimum)-expected[i].minimum),
            std::abs(static_cast<long double>(actual[i].maximum)-expected[i].maximum));
        if(scale>0){maximum=std::max(maximum,error/scale);assert(error<=2e-15L*scale+tiny);}
        else assert(error==0);
        if((actual[i].minimum<=0)!=(expected[i].minimum<=0)){std::fprintf(stderr,"classification %zu %.17g %.17g\n",i,actual[i].minimum,expected[i].minimum);++classifications;}
    }
    assert(classifications==0);check(cudaFree(input));check(cudaFree(output));
    std::printf("PASS %zu trajectory polynomials: shared host bitwise; GPU maximum scaled error %.18Lg; positivity classifications identical\n",controls.size(),maximum);
    return 0;
}
