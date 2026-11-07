#pragma once
#include "FaceAreaPartition.H"
#include <cuda_runtime.h>
namespace ugkwp { namespace gaswall {
template<class R,class Area>
__global__ void sumFaceAreaRows(int cells,const int* offsets,const int* indices,Area area,R* total){
    for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<cells;c+=blockDim.x*gridDim.x)
        total[c]=faceAreaRow<R>(indices,offsets[c],offsets[c+1],area);
}
}}
