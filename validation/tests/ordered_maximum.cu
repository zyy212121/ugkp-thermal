#include "gpu/RadiationMaximum.cuh"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ugkp::gpu;
using Real=RadiationReal;
using Segment=OrderedMaximumSegment<Real>;
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
int main(){
    std::mt19937 rng(32997);int tested=0;
    for(int width:{32,64,96,128,256}) for(int count:{0,1,2,255,256,257,511,512,65539}){
        const int batches=32,blocks=std::max(1,(count+width-1)/width);
        std::vector<Real> input(std::size_t(count)*batches);
        std::vector<unsigned char> mask(input.size());
        for(std::size_t i=0;i<input.size();++i){input[i]=Real(rng()%2001)-1000;mask[i]=rng()%5!=0;
            if(i%29==0)input[i]=std::numeric_limits<Real>::quiet_NaN();
            if(i%53==0)input[i]=std::numeric_limits<Real>::infinity();
            if(i%59==0)input[i]=-std::numeric_limits<Real>::infinity();
            if(i%43==0)input[i]=-0.;if(i%47==0)input[i]=0.;}
        Real* deviceResult=nullptr;check(cudaMalloc(&deviceResult,sizeof(Real)));
        Real* data=nullptr;unsigned char* valid=nullptr;Segment *a=nullptr,*b=nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&data),std::max(std::size_t(1),input.size())*sizeof(Real)));
        check(cudaMalloc(reinterpret_cast<void**>(&valid),std::max(std::size_t(1),mask.size())));
        check(cudaMalloc(reinterpret_cast<void**>(&a),blocks*batches*sizeof(Segment)));
        check(cudaMalloc(reinterpret_cast<void**>(&b),blocks*batches*sizeof(Segment)));
        if(count){check(cudaMemcpy(data,input.data(),input.size()*sizeof(Real),cudaMemcpyHostToDevice));check(cudaMemcpy(valid,mask.data(),mask.size(),cudaMemcpyHostToDevice));}
        for(bool masked:{false,true}){
            orderedMaximumPartials<<<dim3(blocks,batches),width,width*sizeof(Segment)>>>(count,data,masked?valid:nullptr,a);
            int remaining=blocks;Segment *current=a,*next=b;
            while(remaining>1){const int n=(remaining+width-1)/width;orderedMaximumMerge<<<dim3(n,batches),width,width*sizeof(Segment)>>>(remaining,current,next);std::swap(current,next);remaining=n;}
            std::vector<Segment> result(batches);check(cudaMemcpy(result.data(),current,batches*sizeof(Segment),cudaMemcpyDeviceToHost));
            enqueueOrderedMaximum<true>(count,data,masked?valid:nullptr,Real(1e-15),deviceResult,a,b,width,nullptr);
            Real absolute=0,expected=Real(1e-15);
            check(cudaMemcpy(&absolute,deviceResult,sizeof(Real),cudaMemcpyDeviceToHost));
            for(int i=0;i<count;++i)if(!masked||mask[i]){
                const Real value=std::fabs(input[i]);expected=expected>value?expected:value;
            }
            if(!(std::isnan(expected)&&std::isnan(absolute))&&std::memcmp(&expected,&absolute,sizeof(Real)))
                throw std::runtime_error("fused absolute maximum mismatch");
            for(int batch=0;batch<batches;++batch){Real reference=Real(1e-15);
                for(int i=0;i<count;++i){const auto index=std::size_t(batch)*count+i;if(masked&&!mask[index])continue;reference=reference>input[index]?reference:input[index];}
                const Real actual=applyMaximum(1e-15,result[batch]);
                if(!(std::isnan(reference)&&std::isnan(actual))&&std::memcmp(&reference,&actual,sizeof(Real)))throw std::runtime_error("ordered max mismatch");
                ++tested;
            }
        }
        cudaFree(deviceResult);cudaFree(data);cudaFree(valid);cudaFree(a);cudaFree(b);
    }
    std::cout<<"PASS "<<tested<<" batched ordered maximum reductions, IEEE values, ignored entries and empty ranges\n";
}
