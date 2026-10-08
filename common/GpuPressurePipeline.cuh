
#pragma once
// Shared pressure launch protocol. LaunchBackend owns closure, precision selection,
// sorted/unsorted projection and global or segmented launch policy; this function
// owns face-flux/limit/scale/project order and its launch-error boundaries.
template<class LaunchBackend>
int launchCommonPressureKick(DeviceState* s,PressureTime dt,int block,
    int projectionBlock,bool split,bool compact)
{
    if(!s->collisionalPressureEnabled)return 0;
    if(!(dt>0) || !std::isfinite(dt))
    {setLastErrorText("invalid collisional-pressure kick dt");return 1;}
    const int cellGrid=(s->nCells+block-1)/block;
    const int faceGrid=(s->nFaces+block-1)/block;
    computeCollisionalPressureFaceFluxKernel<<<faceGrid,block>>>(s->deviceState);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("pressure face flux launch",err);return 1;}
    err=LaunchBackend::limit(s,dt,cellGrid,block);
    if(err!=cudaSuccess){setLastError("pressure cell limiter launch",err);return 1;}
    scaleCollisionalPressureFaceFluxKernel<<<faceGrid,block>>>(s->deviceState);
    err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("pressure limited face flux launch",err);return 1;}
    if(s->csrCellLocalPathEnabled)
        err=LaunchBackend::project(s,dt,cellGrid,block,projectionBlock,split,compact);
    else
        err=LaunchBackend::projectUnsorted(s,dt,cellGrid,block);
    if(err!=cudaSuccess){setLastError("pressure projection launch",err);return 1;}
    return 0;
}
