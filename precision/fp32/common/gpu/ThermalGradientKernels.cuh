#pragma once
#include "ThermalFieldOperators.H"
namespace ugkp { namespace gpu { namespace thermalGradientDetail {
static __global__ void cellGradient(int n,const int* offsets,const int* faces,const int* signs,
    const GpuReal* value,const ThermalVector* area,const GpuReal* volume,ThermalVector* output){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;ThermalVector g;
    for(int j=offsets[c];j<offsets[c+1];++j){const int f=faces[j];const ThermalVector v{area[f].x*value[f],area[f].y*value[f],area[f].z*value[f]};
        if(signs[j]>0){g.x+=v.x;g.y+=v.y;g.z+=v.z;}else{g.x-=v.x;g.y-=v.y;g.z-=v.z;}}
    output[c]={g.x/volume[c],g.y/volume[c],g.z/volume[c]};
}
}}}
