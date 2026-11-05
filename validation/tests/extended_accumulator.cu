#include "gpu/ExtendedAccumulator.H"
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
using Wide=ugkp::numerics::ExtendedAccumulator;
constexpr int width=64;
struct Result {Wide sum,scale,bound;};
struct ProductInput {double a,b,factor;};
static long double expanded(const Wide& x){
    return (x.negative?-1.L:1.L)*std::ldexp(static_cast<long double>(x.significand),x.exponent-63);
}
static void check(const Wide& got,long double expected,int sample,int operation){
    if(got.kind||expanded(got)!=expected||Wide::bits(got.value())!=Wide::bits(static_cast<double>(expected))){
        std::fprintf(stderr,"Mismatch sample=%d operation=%d extended=%La expected=%La narrowed=%a expected=%a\n",
            sample,operation,expanded(got),expected,got.value(),static_cast<double>(expected));
        std::exit(1);
    }
}
#if defined(__CUDACC__)
__global__ void products(const ProductInput* input,int count,Wide* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Wide value(input[i].a);value.add(input[i].b);output[i]=value.multiplied(input[i].factor);
}
__global__ void evaluate(const double* input,int count,Result* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Wide sum(input[width*i]),scale(::fabs(input[width*i]));
    for(int j=1;j<width;++j){sum.add(input[width*i+j]);scale.add(::fabs(input[width*i+j]));}
    output[i]={sum,scale,scale.multiplied((width+2)*0x1p-52)};
}
#endif
int main(){
    if(LDBL_MANT_DIG!=64){std::puts("SKIP: oracle requires binary80 long double");return 77;}
    constexpr int count=8192;
    std::vector<double> input(count*width);
    std::mt19937_64 random(0x948723edULL);
    for(int i=0;i<count;++i)for(int j=0;j<width;++j){
        auto bits=random();if(((bits>>52)&2047)==2047)bits^=1ULL<<62;
        input[width*i+j]=Wide::real(bits);
        if(j%4==3)input[width*i+j]=-input[width*i+j-1];
    }
    const std::array<double,width> edge={1e308,1e308,-1e308,-1e308,0x1p-1074,-0x1p-1074,
        1,0x1p-64,0x1p-65,-1,1,0x1p-63,0x1p-64,-1,
        -0.0,-0.0,0.0,1e-308,1e-308,-1e-308,-1e-308,
        0x1p-1022,-0x1p-1022,0x1p-1074,0x1p-1074};
    for(int j=0;j<width;++j)input[j]=edge[j];
    // Whole rows at the bottom of the binary64 range exercise scale and
    // donor bounds too; mixed full-range random rows cannot cover this.
    const double tiny=std::numeric_limits<double>::denorm_min();
    const double normal=std::numeric_limits<double>::min();
    const double subnormal=std::nextafter(normal,0.);
    for(int row=1;row<=8;++row)for(int j=0;j<width;++j){
        double value=row<=2?tiny:(row<=4?subnormal:normal);
        if(row%2==0&&j%2)value=-value;
        if(row==7)value=j%3==0?normal:(j%3==1?-subnormal:-tiny);
        if(row==8)value=-0.;
        input[width*row+j]=value;
    }
    auto row=[&](int i,std::initializer_list<double> values){
        std::fill(input.begin()+width*i,input.begin()+width*(i+1),0.);
        std::copy(values.begin(),values.end(),input.begin()+width*i);
    };
    row(9,{1.,0x1p-53});row(10,{std::nextafter(1.,2.),0x1p-53});
    row(11,{1.,0x1p-53,0x1p-63});row(12,{1.,0x1p-53,-0x1p-63});
    row(13,{-1.,-0x1p-53});row(14,{-std::nextafter(1.,2.),-0x1p-53});
    row(15,{DBL_MAX,0x1p970});row(16,{DBL_MAX,0x1p970,-0x1p960});
    row(17,{DBL_MAX,0x1p970,0x1p960});row(18,{-DBL_MAX,-0x1p970});
    row(19,{normal,-subnormal});row(20,{-normal,subnormal});
    row(21,{-0.,0.});row(22,{0.,-0.});
    const std::vector<ProductInput> productInput={
        {tiny,0.,.5},{-tiny,-0.,.5},{3*tiny,0.,.5},{5*tiny,0.,.5},
        {tiny,0.,std::nextafter(.5,1.)},{tiny,0.,std::nextafter(.5,0.)},
        {normal,0.,.5},{subnormal,0.,.5},{normal,-subnormal,.5},
        {DBL_MAX,0.,std::nextafter(1.,2.)},{DBL_MAX,0x1p970,1.},
        {DBL_MAX,0x1p970,-1.},{1.,0x1p-53,1.},{std::nextafter(1.,2.),0x1p-53,1.},
        {-0.,-0.,1.},{-0.,-0.,-1.},{0.,0.,-1.}};
    std::vector<long double> productReference;
    for(std::size_t i=0;i<productInput.size();++i){
        const auto x=productInput[i];long double reference=x.a;reference+=x.b;reference*=x.factor;
        Wide value(x.a);value.add(x.b);check(value.multiplied(x.factor),reference,int(i),-1);
        productReference.push_back(reference);
    }
    std::vector<Result> expected(count);
    for(int i=0;i<count;++i){
        Wide sum(input[width*i]),scale(std::abs(input[width*i]));
        long double reference=input[width*i],magnitude=std::abs(input[width*i]);
        for(int j=1;j<width;++j){
            const double x=input[width*i+j];reference+=x;magnitude+=std::abs(x);
            sum.add(x);scale.add(std::abs(x));
            check(sum,reference,i,j);check(scale,magnitude,i,j);
            const double factor=(j+2)*0x1p-52;
            check(scale.multiplied(factor),factor*magnitude,i,j);
        }
        expected[i]={sum,scale,scale.multiplied((width+2)*0x1p-52)};
    }
#if defined(__CUDACC__)
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    double* values=nullptr;Result* result=nullptr;
    if(cudaMalloc(&values,input.size()*sizeof(double))!=cudaSuccess
        ||cudaMalloc(&result,count*sizeof(Result))!=cudaSuccess)return 2;
    if(cudaMemcpy(values,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
    evaluate<<<(count+127)/128,128>>>(values,count,result);
    std::vector<Result> actual(count);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess
        ||cudaMemcpy(actual.data(),result,count*sizeof(Result),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
    for(int i=0;i<count;++i){
        check(actual[i].sum,expanded(expected[i].sum),i,width);
        check(actual[i].scale,expanded(expected[i].scale),i,width);
        check(actual[i].bound,expanded(expected[i].bound),i,width);
    }
    ProductInput* productValues=nullptr;Wide* productResults=nullptr;
    if(cudaMalloc(&productValues,productInput.size()*sizeof(ProductInput))!=cudaSuccess
        ||cudaMalloc(&productResults,productInput.size()*sizeof(Wide))!=cudaSuccess)return 2;
    if(cudaMemcpy(productValues,productInput.data(),productInput.size()*sizeof(ProductInput),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
    products<<<1,32>>>(productValues,int(productInput.size()),productResults);
    std::vector<Wide> productActual(productInput.size());
    if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(productActual.data(),productResults,productInput.size()*sizeof(Wide),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
    for(std::size_t i=0;i<productInput.size();++i)check(productActual[i],productReference[i],int(i),-1);
    cudaFree(productValues);cudaFree(productResults);
    cudaFree(values);cudaFree(result);
    std::puts("PASS CUDA ordered binary80 sum, range, cancellation and donor bounds");
#else
    std::puts("PASS host ordered binary80 sum, range, cancellation and donor bounds");
#endif
}
