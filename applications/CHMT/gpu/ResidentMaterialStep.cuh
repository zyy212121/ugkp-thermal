#pragma once
#include "gpu/ResidentMaterialStep.H"
#include "materials/CpuMaterialDriver.H"
#include "materials/MaterialTransportMath.H"
#include <cuda_runtime.h>
namespace chmt {
__global__ void requestResidentMaterialStep(const ugkp::IntervalSliceCursor* cursor,
    const ugkp::CompensatedTime* total,CpuMaterialControls controls,
    ResidentMaterialStep* step,ugkwp::GasDriveControls* drive){
    if(!blockIdx.x&&!threadIdx.x)*step=requestMaterialStep(cursor->time,*total,controls.maxSubstep,
        controls.driveRelativeTolerance,controls.driveTractionScale,*drive);
}
__global__ void alignResidentMaterialStep(ResidentMaterialStep* step,const ugkwp::GasDriveSelection* drive,
    const ResidentWallKnotTime* knots,std::size_t count){
    if(!blockIdx.x&&!threadIdx.x)alignMaterialStep(*step,*drive,knots,count);
}
__global__ void limitResidentMaterialStep(ResidentMaterialStep* step,const ugkp::CompensatedTime* total,
    const MaterialTransportDeviceResult* transport){
    if(!blockIdx.x&&!threadIdx.x)limitMaterialStep(*step,*total,transport->dtLimit,transport->errorCode);
}
}
