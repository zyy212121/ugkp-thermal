#pragma once
#include "gpu/ResidentMaterialSlice.H"
#include "gpu/ResidentMaterialState.H"
#include "gpu/ResidentMaterialStep.H"
#include "materials/MaterialRecessionMath.H"
namespace chmt {
struct MaterialRecessionStatus {
    MaterialRecessionError error=MaterialRecessionError::None;
    std::size_t packet=0;
};
// Each face traverses the accepted producer's fixed slots in record order.
// No atomics or copied/compacted packet history are needed. The consumer checks
// the slice status before this node and all face statuses before accepting it.
__global__ void prepareResidentMaterialRecession(ResidentMaterialSlice slice,
    const ResidentMaterialState* state,const ResidentMaterialStep* step,
    const PhysicsConfig* physics,const int* gasFaces,const int* solidFaces,
    const FilmAux* previous,Real* targets,FilmAux* candidate,MaterialRecessionStatus* status){
    if(slice.plan->error!=ugkp::IntervalSliceError::None)return;
    const auto material=*state;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x){
        Real target=0;MaterialRecessionStatus result;
        for(std::size_t r=slice.plan->begin.record;r<slice.plan->past;++r){
            for(unsigned k=0;k<3;++k){
                const auto slot=((r-slice.plan->begin.record)*slice.history.faces+f)*3+k;
                if(!slice.active(slot))continue;
                const auto packet=slice.packet(slot);if(packet.kind!=ExchangeKind::GasSolid)continue;
                result.error=addMaterialRecessionTarget(packet.condensed,material.current[packet.solidCell],
                    material.mesh.volumes[packet.solidCell],*physics,target);
                if(result.error!=MaterialRecessionError::None){result.packet=slot;break;}
            }
            if(result.error!=MaterialRecessionError::None)break;
        }
        targets[f]=target;status[f]=result;candidate[f]=previous[f];
        if(result.error==MaterialRecessionError::None&&gasFaces[f]>=0){
            const int sf=solidFaces[f];const Real area=sf>=0?mag(material.mesh.areaVectors[sf]):0;
            candidate[f]=materialRecessionFront(previous[f],target,area,step->dt);
        }
    }
}
}
