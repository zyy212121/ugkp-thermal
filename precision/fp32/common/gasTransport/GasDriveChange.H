#pragma once
#include <cmath>
#include "../GpuPrecisionTypes.H"
#include <cstddef>
#if defined(__CUDACC__)
#define UGKP_GAS_DRIVE_HD __host__ __device__
#else
#define UGKP_GAS_DRIVE_HD
#endif
namespace ugkwp {
struct GasDriveComparison {GpuReal change=0;bool layoutMatches=true,receiverChanged=false;};
namespace drive_detail {
UGKP_GAS_DRIVE_HD inline GpuReal maximum(GpuReal a,GpuReal b){return a>b?a:b;}
UGKP_GAS_DRIVE_HD inline GpuReal absolute(GpuReal a){return a<0?-a:a;}
UGKP_GAS_DRIVE_HD inline GpuReal relative(GpuReal a,GpuReal b,GpuReal floor){
    return absolute(a-b)/maximum(floor,maximum(absolute(a),absolute(b)));
}
template<class Vector> UGKP_GAS_DRIVE_HD inline GpuReal length(const Vector& v){
    const GpuReal x=GpuReal(v.x),y=GpuReal(v.y),z=GpuReal(v.z);return std::sqrt(x*x+y*y+z*z);
}
template<class A,class B> UGKP_GAS_DRIVE_HD inline GpuReal distance(const A& a,const B& b){
    const GpuReal x=GpuReal(a.x)-GpuReal(b.x),y=GpuReal(a.y)-GpuReal(b.y),z=GpuReal(a.z)-GpuReal(b.z);
    return std::sqrt(x*x+y*y+z*z);
}
}
// Ordered local driving-state variation, shared by host publication adapters
// and resident history consumers. Views expose indexed fields without owning
// storage. Keep category/face/component ordering, including IEEE max semantics;
// this is a local lag criterion, not a bound on global PDE error.
template<int Ns,class Held,class Sample>
UGKP_GAS_DRIVE_HD GasDriveComparison compareGasDrive(const Held& held,const Sample& r,GpuReal tractionScale){
    using namespace drive_detail;GasDriveComparison result;
    if(r.bulkCount()!=held.bulkCount()||r.tractionCount()!=held.tractionCount()
        ||r.gradientCount()!=held.gradientCount()||r.matchingCount()!=held.matchingCount()){
        result.layoutMatches=false;return result;}
    GpuReal& change=result.change;
    for(std::size_t f=0;f<r.bulkCount();++f){const auto& a=held.bulk(f);const auto& b=r.bulk(f);
        change=maximum(change,relative(a.temperature,b.temperature,1));change=maximum(change,relative(a.pressure,b.pressure,1));
        change=maximum(change,relative(a.rho,b.rho,1e-30f));
        change=maximum(change,distance(a.velocity,b.velocity)/maximum(1,maximum(a.soundSpeed,b.soundSpeed)));
        for(int s=0;s<Ns;++s)change=maximum(change,absolute(GpuReal(a.Y[s])-GpuReal(b.Y[s])));
    }
    for(std::size_t f=0;f<r.matchingCount();++f){const auto& a=held.matching(f);const auto& b=r.matching(f);
        change=maximum(change,relative(a.pressure,b.pressure,1));
        change=maximum(change,relative(a.mechanicalPressure,b.mechanicalPressure,1));
        change=maximum(change,relative(a.state.temperature,b.state.temperature,1));
        const GpuReal speedScale=f<r.bulkCount()?maximum(1,maximum(r.bulk(f).soundSpeed,held.bulk(f).soundSpeed)):1;
        GpuReal velocityChange=0;for(int d=0;d<3;++d){const GpuReal delta=GpuReal(a.state.velocity[d])-GpuReal(b.state.velocity[d]);velocityChange+=delta*delta;}
        change=maximum(change,std::sqrt(velocityChange)/speedScale);
        for(int s=0;s<Ns;++s)change=maximum(change,absolute(GpuReal(a.state.massFraction[s])-GpuReal(b.state.massFraction[s])));
        change=maximum(change,relative(a.state.k,b.state.k,1e-12f));
        change=maximum(change,relative(a.state.omega,b.state.omega,1e-12f));
    }
    for(std::size_t f=0;f<r.tractionCount();++f)change=maximum(change,distance(r.traction(f),held.traction(f))
        /maximum(tractionScale,maximum(length(r.traction(f)),length(held.traction(f)))));
    for(std::size_t f=0;f<r.gradientCount();++f)for(int j=0;j<3;++j)change=maximum(change,
        distance(r.gradient(f).velocity[j],held.gradient(f).velocity[j])
        /maximum(1,maximum(length(r.gradient(f).velocity[j]),length(held.gradient(f).velocity[j]))));
    result.receiverChanged=r.receiverCount()!=held.receiverCount();
    for(std::size_t f=0;!result.receiverChanged&&f<r.receiverCount();++f)result.receiverChanged=r.receiver(f)!=held.receiver(f);
    return result;
}
}
#undef UGKP_GAS_DRIVE_HD
