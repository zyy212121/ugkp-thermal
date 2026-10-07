#include "film/ResolvedNormalThermal.H"
#include "gas/KernelSupport.cuh"
namespace chmt {
namespace {
__global__ void normalRecoverKernel(NormalThermalView v, PhysicsConfig p, DeviceStatus* status) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    if (column>=v.nFaces || gasStopped(status)) return;
    const int begin=v.offsets[column], end=v.offsets[column+1];
    if (begin<0 || end<=begin || end>v.nLayers || !(v.area[column]>0)) {
        gasDeviceError(status,ErrorCode::InvalidInput,column,begin,ErrorLocation::Face); return;
    }
    for (int layer=begin; layer<end; ++layer) {
        Real T=0, fraction=0;
        if (!finite(v.coordinate[layer]) || !finite(v.thickness[layer]) || v.thickness[layer]<=0
            || !closeEnough(v.oldVolume[layer],v.newVolume[layer],p.tolerances.absoluteGeometry,
                p.tolerances.relativeGeometry)
            || !closeEnough(v.newVolume[layer],v.area[column]*v.thickness[layer],
                p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)
            || !recoverNormalEnthalpy(v.mass[layer],v.enthalpy[layer],v.newVolume[layer],
                v.pressure[column],p,T,fraction)) {
            gasDeviceError(status,ErrorCode::PropertyRange,layer,v.enthalpy[layer],ErrorLocation::Layer);
            return;
        }
        v.temperature[layer]=T;
        v.liquidFraction[layer]=fraction;
    }
}
__device__ Real layerConductivity(int layer, NormalThermalView v, PhysicsConfig p) {
    const Real f=v.liquidFraction[layer];
    return (1-f)*p.condensed[p.material.phaseCondensed].conductivity+f*p.liquid.conductivity;
}
__global__ void normalRhsKernel(NormalThermalView v, PhysicsConfig p, DeviceStatus* status) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    if (column>=v.nFaces || gasStopped(status)) return;
    const int begin=v.offsets[column], end=v.offsets[column+1];
    for (int layer=begin; layer<end; ++layer) {
        v.rhs[layer]=v.newVolume[layer]*v.pressureRate[column];
    }
    for (int layer=begin; layer+1<end; ++layer) {
        const Real distance=v.coordinate[layer+1]-v.coordinate[layer];
        const Real expected=0.5*(v.thickness[layer]+v.thickness[layer+1]);
        if (!closeEnough(distance,expected,p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)) {
            gasDeviceError(status,ErrorCode::InvalidInput,layer,distance,ErrorLocation::Layer); return;
        }
        const Real power=normalConductivePower(v.temperature[layer],v.temperature[layer+1],
            v.thickness[layer]/2,v.thickness[layer+1]/2,
            layerConductivity(layer,v,p),layerConductivity(layer+1,v,p),v.area[column]);
        v.rhs[layer]-=power;
        v.rhs[layer+1]+=power;
    }
    Real bottom=0, top=0;
    if (v.bottomTemperature) {
        bottom=v.area[column]*layerConductivity(begin,v,p)
            *(v.bottomTemperature[column]-v.temperature[begin])/(v.thickness[begin]/2);
    } else if (v.bottomHeatFlux) bottom=v.area[column]*v.bottomHeatFlux[column];
    if (v.topTemperature) {
        top=v.area[column]*layerConductivity(end-1,v,p)
            *(v.topTemperature[column]-v.temperature[end-1])/(v.thickness[end-1]/2);
    } else if (v.topHeatFlux) top=v.area[column]*v.topHeatFlux[column];
    if (!finite(bottom) || !finite(top) || !finite(v.pressureRate[column])) {
        gasDeviceError(status,ErrorCode::Nonfinite,column,bottom+top,ErrorLocation::Face); return;
    }
    v.rhs[begin]+=bottom;
    v.rhs[end-1]+=top;
    v.bottomHeatRate[column]=bottom;
    v.topHeatRate[column]=top;
    // acceptedWallHeat is deliberately never mutated here.
}
bool validNormalPointers(NormalThermalView v) {
    return v.nFaces>0 && v.nLayers>0 && v.offsets && v.coordinate && v.thickness && v.area
        && v.pressure && v.pressureRate && v.mass && v.oldVolume && v.newVolume
        && v.enthalpy && v.temperature && v.liquidFraction;
}
}
int launchNormalThermalValidate(NormalThermalView v, const PhysicsConfig& p,
    DeviceStatus* status, void* stream) {
    if (v.nFaces==0 && v.nLayers==0) return 0;
    if (!status || !validNormalPointers(v)) return static_cast<int>(cudaErrorInvalidValue);
    normalRecoverKernel<<<(v.nFaces+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(v,p,status);
    return launchError();
}
int launchResolvedNormalThermalRhs(NormalThermalView v, const PhysicsConfig& p,
    Real dt, DeviceStatus* status, void* stream) {
    if (v.nFaces==0 && v.nLayers==0) return 0;
    if (!status || !validNormalPointers(v) || !v.rhs || !v.bottomHeatRate || !v.topHeatRate
        || (v.bottomTemperature && v.bottomHeatFlux) || (v.topTemperature && v.topHeatFlux)
        || !finite(dt) || dt<=0) return static_cast<int>(cudaErrorInvalidValue);
    int code=launchNormalThermalValidate(v,p,status,stream);
    if (code) return code;
    normalRhsKernel<<<(v.nFaces+63)/64,64,0,static_cast<cudaStream_t>(stream)>>>(v,p,status);
    return launchError();
}
} // namespace chmt
