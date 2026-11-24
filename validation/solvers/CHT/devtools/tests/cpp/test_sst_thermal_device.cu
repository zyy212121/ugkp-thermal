#include "../legacy/common/gasNumerics/LegacySstWallFunction.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>

__global__ void evaluate(ugkpwall::JayatillekeThermalTransport* results)
{
    constexpr GpuReal Pr=GpuReal(.4), Prt=GpuReal(.85), kappa=GpuReal(.41), E=GpuReal(9.8);
    constexpr GpuReal conductivity=GpuReal(.4);
    const auto P=ugkpwall::jayatillekeSmoothP(Pr/Prt);
    const auto transition=ugkpwall::jayatillekeThermalYPlus(Pr/Prt,kappa,E);
    const GpuReal gradient=GpuReal(1000)*(GpuReal(threadIdx.x)-GpuReal(1));
    results[threadIdx.x]=ugkpwall::sstJayatillekeThermalTransport
        (conductivity,Pr,Prt,kappa,E,P,transition,GpuReal(8150),gradient);
}

int main()
{
    ugkpwall::JayatillekeThermalTransport* device;
    ugkpwall::JayatillekeThermalTransport host[3];
    if (cudaMalloc(&device,sizeof(host)) != cudaSuccess) return 1;
    evaluate<<<1,3>>>(device);
    const auto copy=cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (copy != cudaSuccess) return 2;
    const double ratio=.4/.85;
    const double P=9.24*(std::pow(ratio,.75)-1)*(1+.28*std::exp(-.007*ratio));
    const double TPlus=.85*(std::log(9.8*8150)/.41+P);
    const double expected=.4*.4*8150/TPlus;
    for (int i=0; i<3; ++i)
    {
        const double heat=-expected*1000*(i-1);
        if (!host[i].valid || std::abs(host[i].conductivity/expected-1)>3e-6
            || std::abs(host[i].heatFlux-heat)>3e-6*std::max(1.,std::abs(heat)))
            return 3;
    }
    std::cout<<"PASS CUDA Spalding y+ and signed Fourier heat\n";
}
