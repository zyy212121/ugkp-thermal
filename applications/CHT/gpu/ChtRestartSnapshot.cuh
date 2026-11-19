#pragma once
#include "ChtRestartSnapshot.H"
#include "GpuChtParticleRestart.H"
#include "../../../common/GpuParticleRestartTheta.H"
#include "../../../common/gpu/FrozenReadback.cuh"
#include <fstream>
#include <limits>
#include <type_traits>

namespace Foam { namespace gpuThermal {
namespace chtSnapshot {



template<class View>
View layout(ugkp::gpu::SnapshotLayout& layout, unsigned char* buffer, std::size_t count) {
    View out{};
    out.count=count;
    const auto allocate=[&](auto& field, unsigned width){
        using T=std::remove_cv_t<std::remove_pointer_t<std::decay_t<decltype(field)>>>;
        const auto slot=layout.allocate<T>(count*width);
        field=buffer ? slot.at(buffer) : nullptr;
    };
#define UGKP_PARTICLE_WIRE_THETA(field) allocate(out.field,1);
#define UGKP_PARTICLE_FIELD(field,value) allocate(out.field,1);
#define UGKP_PARTICLE_OPTIONAL(field,value) allocate(out.field,1);
#define UGKP_PARTICLE_ARRAY(field,value,n) allocate(out.field,n);
#include "../../../common/thermal/ParticleStateFields.inc"
#undef UGKP_PARTICLE_WIRE_THETA
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
    return out;
}
__global__ void pack(const DeviceState* state,gpuFshRestart::MutableView out) {
    const auto& s=*state;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<int(out.count);i+=blockDim.x*gridDim.x){
#define UGKP_PARTICLE_WIRE_THETA(field) out.field[i]=particleRestartTheta(s.pStuck[i],s.field[i],s.pContactAge[i]);
#define UGKP_PARTICLE_FIELD(field,value) out.field[i]=s.field[i];
#define UGKP_PARTICLE_OPTIONAL(field,value) { const int j=s.field ? gpuWall::coldWallFieldSlot(s,s.field,i) : -1; out.field[i]=j>=0?s.field[j]:0; }
#define UGKP_PARTICLE_ARRAY(field,value,n) { const int j=s.field ? gpuWall::coldWallFieldSlot(s,s.field,i) : -1; for(int k=0;k<n;++k)out.field[i*n+k]=j>=0?s.field[j*n+k]:0; }
#include "../../../common/thermal/ParticleStateFields.inc"
#undef UGKP_PARTICLE_WIRE_THETA
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
    }
}
template<class Writer>
void publish(const std::string& path,Writer encode){
    const std::string temporary=path+".tmp";
    std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot open "+temporary);
    encode(out);
    out.close();
    if(!out||std::rename(temporary.c_str(),path.c_str())!=0)
        throw std::runtime_error("cannot publish "+path);
}
}
struct ChtRestartSnapshot::Storage {
    ugkp::gpu::FrozenReadback transfer;
    ugkp::gpu::SnapshotRange<GpuReal> previousFraction,residualMass;
    ugkp::gpu::SnapshotRange<int> sourceFaces;
    int count=0;
    cudaStream_t producer=nullptr;
};
ChtRestartSnapshot::ChtRestartSnapshot():storage_(new Storage){}
ChtRestartSnapshot::~ChtRestartSnapshot()=default;
void* ChtRestartSnapshot::producerStream() const{return storage_->producer;}
int ChtRestartSnapshot::freeze(void* handle){
    auto* s=asState(handle);
    if(validateState(s,"freeze CHT checkpoint"))ugkp::gpu::snapshotDataFailure(lastError);
    auto& target=*storage_;
    target.producer=s->advanceGraph?s->advanceGraph->stream:s->gasCaptureStream;
    const auto shape=ugkp::thermal::particleSnapshotShape(handle,target.producer);
    target.count=shape.count;
    ugkp::gpu::SnapshotLayout size;
    chtSnapshot::layout<gpuFshRestart::MutableView>(size,nullptr,target.count);
    target.previousFraction=size.allocate<GpuReal>(s->nCells);
    target.sourceFaces=size.allocate<int>(shape.sources);
    target.residualMass=size.allocate<GpuReal>(shape.sources);
    target.transfer.freeze(size.bytes,target.producer,[&](unsigned char* storage){
        if(target.count){
            ugkp::gpu::SnapshotLayout offsets;
            auto view=chtSnapshot::layout<gpuFshRestart::MutableView>(offsets,storage,target.count);
            const int width=ugkp::gpu::workPlan().particleBlockThreads;
            chtSnapshot::pack<<<(target.count+width-1)/width,width,0,target.producer>>>(s->deviceState,view);
            ugkp::gpu::checkSnapshotCopy(cudaGetLastError(),"freeze CHT particle history");
        }
        target.previousFraction.copy(storage,s->epsGPrev,target.producer);
        target.sourceFaces.copy(storage,shape.sourceFaces,target.producer);
        target.residualMass.copy(storage,shape.sourceResidualMass,target.producer);
    });
    return target.count;
}
void ChtRestartSnapshot::write(const std::string& directory){
    auto& s=*storage_;
    const auto* bytes=s.transfer.collect();
    ugkp::gpu::SnapshotLayout offsets;
    auto particles=chtSnapshot::layout<gpuFshRestart::ConstView>(offsets,const_cast<unsigned char*>(bytes),s.count);
    chtSnapshot::publish(directory+"/gpuResidentStrictParticles.dat",[&](auto& out){gpuFshRestart::writeCurrent(out,particles);});
    chtSnapshot::publish(directory+"/gpuResidentStrictEpsGPrev.dat",[&](auto& out){
        out.precision(std::numeric_limits<double>::max_digits10);out<<std::scientific<<s.previousFraction.count<<'\n';
        const auto* values=s.previousFraction.at(bytes);
        for(std::size_t i=0;i<s.previousFraction.count;++i)out<<values[i]<<'\n';
    });
    chtSnapshot::publish(directory+"/gpuResidentStrictSourceResidual.dat",[&](auto& out){
        out.precision(std::numeric_limits<double>::max_digits10);out<<std::scientific<<"UGKP_SOURCE_RESIDUAL_SCHEMA1 "<<s.sourceFaces.count<<'\n';
        const auto* ids=s.sourceFaces.at(bytes);const auto* mass=s.residualMass.at(bytes);
        for(std::size_t i=0;i<s.sourceFaces.count;++i)out<<ids[i]<<' '<<mass[i]<<'\n';
    });
}
}}
