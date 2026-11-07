#pragma once
#include "ablation/MaterialSurfaceInput.H"
#include "gpu/ResidentMaterialDrive.H"
#include "gpu/ResidentMaterialState.H"
#include "gpu/ResidentMaterialStep.H"
#include "gpu/MaterialSurfaceKernels.cuh"
#include "../../../common/gpu/ResidentSurfaceGeometry.H"
#include "../../../common/gasWall/WallGeometryTypes.H"
#include "../../../common/gasWall/FaceAreaPartition.cuh"
namespace chmt {
enum class MaterialSurfaceInputError : int {None,MissingMatching,MechanicalPressure,GasFace,GasCell,SolidCell,SolidFace};
struct MaterialSurfaceInputBinding {
    int faces=0;
    const int* requests=nullptr;
    const std::size_t* records=nullptr;
    ResidentMaterialDriveHistory history{};
    const ResidentMaterialState* material=nullptr;
    const MaterialMeshView* gasMesh=nullptr;
    const ugkp::gpu::geometry::SurfaceFaceGeometry* surface=nullptr;
    const int *gasFaces=nullptr,*solidFaces=nullptr,*solidCells=nullptr;
    const unsigned char* ablating=nullptr;
    const Vec3* baseVelocity=nullptr;
    // Retain the macro-window's base particle fraction, as in the original
    // material state. The advancing gas endpoint's fraction is a different
    // physical snapshot and must not silently replace this input.
    const Real* baseGasVoidFraction=nullptr;
    const Real* donorArea=nullptr;
    const Real* sideVolume=nullptr;
    const ugkwp::gaswall::WallMatchingGeometry* matchingGeometry=nullptr;
    ugkwp::GasMechanismView<Real,Ns> mechanism{};
    bool matching=false,acceptedFlux=false,candidate=false,useRates=false;
    const SolidQ* rates=nullptr;
    SurfacePhysicsInput* inputs=nullptr;
    MaterialSurfaceInputError* errors=nullptr;
};
struct MaterialSurfaceGasArea {
    const MaterialMeshView* mesh=nullptr;
    const int* faces=nullptr;
    __device__ Real operator()(int i)const{return mag(mesh->areaVectors[faces[i]]);}
};
// Topology/addressing and output capacity are initialized once by the owner.
// Gas history, geometry, timestep, selected records and material pointer parity
// are all read on device. The existing closure receives the same input fields.
__global__ void prepareResidentMaterialSurfaceInput(MaterialSurfaceInputBinding binding,
    const ResidentMaterialStep* step,const PhysicsConfig* physics,
    const MaterialSurfaceView* prototype,MaterialSurfaceView* operation){
    const int count=binding.faces*(*binding.requests);
    const auto state=*binding.material;const auto solid=binding.candidate?state.candidateMesh:state.mesh;
    const auto gas=*binding.gasMesh;
    if(!blockIdx.x&&!threadIdx.x){auto view=*prototype;view.count=count;
        view.solid=binding.candidate?state.candidate:state.current;view.volumes=solid.volumes;
        view.materialRate=binding.useRates?binding.rates:nullptr;*operation=view;}
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
        const int f=i%binding.faces,request=i/binding.faces;auto& error=binding.errors[i];error=MaterialSurfaceInputError::None;
        SurfacePhysicsInput in;
        const int gf=binding.gasFaces[f];if(gf<0){binding.inputs[i]=in;continue;}
        const auto record=binding.history.record(binding.records[request]);
        GasWallClosureContext context;
        if(binding.matching){
            if(!record.wall){error=MaterialSurfaceInputError::MissingMatching;continue;}
            const auto& sample=record.wall[f].matching;
            if(!finite(sample.mechanicalPressure)||sample.mechanicalPressure<=0){error=MaterialSurfaceInputError::MechanicalPressure;continue;}
            context.mechanism=binding.mechanism;context.matching=sample.state;
            context.matchingPressure=sample.pressure;context.mechanicalPressure=sample.mechanicalPressure;
            context.matchingDistance=binding.matchingGeometry[f].matchingDistance;
            context.ownerDistance=binding.matchingGeometry[f].ownerDistance;
        }
        if(gf>=gas.faces){error=MaterialSurfaceInputError::GasFace;continue;}
        const int gc=gas.owner[gf];if(gc<0||gc>=gas.cells){error=MaterialSurfaceInputError::GasCell;continue;}
        in.ablating=!binding.ablating||binding.ablating[f];
        const Real phaseVolume=gas.volumes[gc]*(binding.baseGasVoidFraction?binding.baseGasVoidFraction[gc]:Real(1));
        const Real share=ugkwp::gaswall::faceAreaShare(mag(gas.areaVectors[gf]),binding.donorArea[gc]);
        setMaterialSurfaceGasDrive(in,record.bulk(f),&record.gradient(f),phaseVolume,share,
            binding.acceptedFlux?&record.surface[f].acceptedWall:nullptr,*physics);
        if(binding.matching)setMaterialSurfaceMatching(in,context);
        const auto metric=binding.surface[f];
        setMaterialSurfaceGasGeometry(in,metric.area,{metric.normal.x,metric.normal.y,metric.normal.z},gas.areaVectors[gf],metric.gasDistance,
            binding.baseVelocity?binding.baseVelocity[f]:Vec3{},step->dt);
        const int sc=binding.solidCells?binding.solidCells[f]:-1;in.hasSolid=sc>=0;
        if(in.hasSolid){
            if(sc>=solid.cells){error=MaterialSurfaceInputError::SolidCell;continue;}
            const int sf=binding.solidFaces[f];if(sf<0||sf>=solid.faces){error=MaterialSurfaceInputError::SolidFace;continue;}
            setMaterialSurfaceSolidGeometry(in,metric.solidDistance,solid.areaVectors[sf]);
        }
        if(binding.sideVolume)in.sideMeshVolumeRate=binding.sideVolume[f];
        binding.inputs[i]=in;
    }
}
}
