#include "gasTransport/GasStatusCompletion.cuh"
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t code){
    if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

int main(){
    using namespace ugkp::diagnostics;
    const int block=ugkp::gpu::workPlan().reductionBlockThreads;
    const int invalid=int(ugkwp::GasTransportCode::InvalidStorage);
    int* values;DeviceContext* diagnostics;unsigned long long* first;
    check(cudaMalloc(&values,(block+5)*sizeof(int)));
    check(cudaMalloc(&diagnostics,sizeof(DeviceContext)));check(cudaMalloc(&first,sizeof(*first)));
    for(const int count:{0,1,7,block,block+1}){
        const int cells=count/2;
        for(int mode=0;mode<4;++mode){
            std::vector<int> source(count,0);DeviceContext context{};
            if(mode&&count)source.back()=invalid;
            if(mode==2&&cells)source[cells-1]=invalid;
            if(mode==3){context.fatalClaim=1;context.firstFatal.sourceSiteId=9;}
            if(count)check(cudaMemcpy(values,source.data(),count*sizeof(int),cudaMemcpyHostToDevice));
            check(cudaMemcpy(diagnostics,&context,sizeof(context),cudaMemcpyHostToDevice));
            check(ugkwp::enqueueGasStatusCheck(values,cells,values+cells,count-cells,diagnostics,first,23,8,nullptr));
            unsigned long long selected;
            check(cudaMemcpy(&selected,first,sizeof(selected),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&context,diagnostics,sizeof(context),cudaMemcpyDeviceToHost));
            unsigned long long expected=~0ULL;
            if(mode!=3)for(int i=0;i<count;++i)if(source[i]==invalid){expected=i;break;}
            if(selected!=expected)return 1;
            if(mode==3){if(context.firstFatal.sourceSiteId!=9)return 2;continue;}
            if(expected==~0ULL){if(context.fatalClaim)return 3;continue;}
            const bool face=expected>=static_cast<unsigned>(cells);
            if(!context.fatalClaim||context.firstFatal.sourceSiteId!=23
                ||context.firstFatal.entityKind!=(face?EntityKind::Face:EntityKind::Cell)
                ||context.firstFatal.entityIndex!=static_cast<std::int64_t>(face?expected-cells:expected))return 4;
        }
    }
    check(cudaFree(values));check(cudaFree(diagnostics));check(cudaFree(first));
    std::cout<<"PASS 20 gas status completions\n";
}
