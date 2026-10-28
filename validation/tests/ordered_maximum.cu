#include "gpu/OrderedMaximum.cuh"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ugkp::gpu;
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
int main(){
    std::mt19937 rng(32997);int tested=0;
    for(int count:{0,1,2,255,256,257,511,512,65539}){
        const int batches=32,blocks=std::max(1,(count+255)/256);
        std::vector<double> input(std::size_t(count)*batches);
        std::vector<unsigned char> mask(input.size());
        for(std::size_t i=0;i<input.size();++i){input[i]=double(rng()%2001)-1000;mask[i]=rng()%5!=0;
            if(i%29==0)input[i]=std::numeric_limits<double>::quiet_NaN();
            if(i%53==0)input[i]=std::numeric_limits<double>::infinity();
            if(i%59==0)input[i]=-std::numeric_limits<double>::infinity();
            if(i%43==0)input[i]=-0.;if(i%47==0)input[i]=0.;}
        double* data=nullptr;unsigned char* valid=nullptr;OrderedMaximum *a=nullptr,*b=nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&data),std::max(std::size_t(1),input.size())*sizeof(double)));
        check(cudaMalloc(reinterpret_cast<void**>(&valid),std::max(std::size_t(1),mask.size())));
        check(cudaMalloc(reinterpret_cast<void**>(&a),blocks*batches*sizeof(OrderedMaximum)));
        check(cudaMalloc(reinterpret_cast<void**>(&b),blocks*batches*sizeof(OrderedMaximum)));
        if(count){check(cudaMemcpy(data,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));check(cudaMemcpy(valid,mask.data(),mask.size(),cudaMemcpyHostToDevice));}
        for(bool masked:{false,true}){
            orderedMaximumPartials<<<dim3(blocks,batches),256>>>(count,data,masked?valid:nullptr,a);
            int remaining=blocks;OrderedMaximum *current=a,*next=b;
            while(remaining>1){const int n=(remaining+255)/256;orderedMaximumMerge<<<dim3(n,batches),256>>>(remaining,current,next);std::swap(current,next);remaining=n;}
            std::vector<OrderedMaximum> result(batches);check(cudaMemcpy(result.data(),current,batches*sizeof(OrderedMaximum),cudaMemcpyDeviceToHost));
            for(int batch=0;batch<batches;++batch){double reference=1e-15;
                for(int i=0;i<count;++i){const auto index=std::size_t(batch)*count+i;if(masked&&!mask[index])continue;reference=reference>input[index]?reference:input[index];}
                const double actual=applyMaximum(1e-15,result[batch]);
                if(!(std::isnan(reference)&&std::isnan(actual))&&std::memcmp(&reference,&actual,sizeof(double)))throw std::runtime_error("ordered max mismatch");
                ++tested;
            }
        }
        cudaFree(data);cudaFree(valid);cudaFree(a);cudaFree(b);
    }
    std::cout<<"PASS "<<tested<<" batched ordered maximum reductions, IEEE values, ignored entries and empty ranges\n";
}
