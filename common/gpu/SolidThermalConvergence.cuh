#pragma once
using ThermalScalar=decltype(SolidThermalConvergence::temperatureChange);
__device__ void mergeConvergence(SolidThermalConvergence& a,const SolidThermalConvergence& b){
    a.boundaryChanged=a.boundaryChanged||b.boundaryChanged;
    a.enthalpyChange.merge(b.enthalpyChange);a.externalEnergy.merge(b.externalEnergy);a.interfaceEnergy.merge(b.interfaceEnergy);
    a.temperatureChange=std::fmax(a.temperatureChange,b.temperatureChange);a.surfaceChange=std::fmax(a.surfaceChange,b.surfaceChange);
    a.surfaceFluxChange=std::fmax(a.surfaceFluxChange,b.surfaceFluxChange);
}
__global__ void thermalConvergenceParts(int cells,int surfaces,int boundary,const ThermalPropertyResult* properties,const ThermalScalar* volume,
    const int* surfaceIndex,const ThermalScalar* flux,const ThermalScalar* temperature,ThermalScalar* previous,const ThermalScalar* gradient,
    const ThermalScalar* energy,const ThermalScalar* area,ThermalScalar interval,SolidThermalConvergence* partial,const double* deviceInterval=nullptr){
    if(deviceInterval)interval=static_cast<ThermalScalar>(*deviceInterval);
    extern __shared__ __align__(16) unsigned char storage[];auto* values=reinterpret_cast<SolidThermalConvergence*>(storage);const int i=blockIdx.x*blockDim.x+threadIdx.x;SolidThermalConvergence v;
    if(i<cells){const auto value=properties[surfaces+i];v.enthalpyChange.product(volume[i],value.energy);v.temperatureChange=value.relativeChange;}
    if(i<boundary){const int source=surfaceIndex[i];
        if(source==-2)v.externalEnergy.product(-interval,flux[i]);
        else v.interfaceEnergy.add(energy[i]);
        if(source>=0){const ThermalScalar T=temperature[i],old=previous[i];
            v.surfaceChange=std::fabs(T-old)/std::fmax(std::fmax(std::fabs(T),std::fabs(old)),ThermalScalar(1));
            const ThermalScalar prescribed=energy[i]/(area[i]*interval),actual=properties[source].conductivity*gradient[i];
            v.surfaceFluxChange=std::fabs(actual-prescribed)/std::fmax(std::fmax(std::fabs(actual),std::fabs(prescribed)),ThermalScalar(1));}
        v.boundaryChanged=temperature[i]!=previous[i];
        previous[i]=temperature[i];}
    values[threadIdx.x]=v;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){const int offset=(active+1)/2;if(threadIdx.x<active-offset)mergeConvergence(values[threadIdx.x],values[threadIdx.x+offset]);__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void thermalConvergenceFinish(int count,const SolidThermalConvergence* partial,SolidThermalConvergence* result){
    extern __shared__ __align__(16) unsigned char storage[];auto* values=reinterpret_cast<SolidThermalConvergence*>(storage);SolidThermalConvergence v;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)mergeConvergence(v,partial[i]);values[threadIdx.x]=v;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){const int offset=(active+1)/2;if(threadIdx.x<active-offset)mergeConvergence(values[threadIdx.x],values[threadIdx.x+offset]);__syncthreads();}
    if(!threadIdx.x)*result=values[0];
}
