#include "sweep_trial_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
__global__ void evaluateSweepTrial(int count,const SweepTrialInput* input,g::SweepTrialCoordinate* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){const auto x=input[i];output[i]=g::sweepTrialCoordinate(x.origin,x.correction,x.alpha);}
}
}
std::vector<g::SweepTrialCoordinate> probeSweepTrial(const std::vector<SweepTrialInput>& input){
    std::vector<g::SweepTrialCoordinate> result(input.size());if(input.empty())return result;
    SweepTrialInput* data=nullptr;g::SweepTrialCoordinate* output=nullptr;
    check(cudaMalloc(&data,input.size()*sizeof(*data)));check(cudaMalloc(&output,result.size()*sizeof(*output)));
    check(cudaMemcpy(data,input.data(),input.size()*sizeof(*data),cudaMemcpyHostToDevice));
    evaluateSweepTrial<<<(input.size()-1)/128+1,128>>>(input.size(),data,output);check(cudaGetLastError());
    check(cudaMemcpy(result.data(),output,result.size()*sizeof(*output),cudaMemcpyDeviceToHost));check(cudaFree(output));check(cudaFree(data));return result;
}
