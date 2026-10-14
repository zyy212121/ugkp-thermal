#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <type_traits>
namespace r4obs {
template<class T> auto view(T* s,int)->decltype(s->storage.hostView()){return s->storage.hostView();}
template<class T> T& view(T* s,long){return *s;}
template<class T> std::vector<T> read(const T* p,int n){
    std::vector<T> a;
    if(!p||n<=0)return a;
    a.resize(n);
    const auto e=cudaMemcpy(a.data(),p,std::size_t(n)*sizeof(T),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess){std::fprintf(stderr,"R4_OBSERVER_RESOURCE_ERROR %s\n",cudaGetErrorString(e));a.clear();}
    return a;
}
template<class T> void field(const char* name,const T* p,int n){
    const auto a=read(p,n);std::printf(",\"%s\":[",name);
    for(std::size_t i=0;i<a.size();++i){if(i)std::printf(",");
        if constexpr(std::is_integral<T>::value)std::printf("%lld",static_cast<long long>(a[i]));
        else std::printf("\"%a\"",static_cast<double>(a[i]));}
    std::printf("]");
}
template<class Owner> void dump(Owner* owner,const char* tag,double dt,double ledgerDt,double stageTime,double w0=0.,double w1=0.){
    const auto e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){std::fprintf(stderr,"R4_OBSERVER_RESOURCE_ERROR sync %s\n",cudaGetErrorString(e));return;}
    auto& s=view(owner,0);const int nc=s.nCells,nf=s.nFaces;
    const auto counts=read(s.cellPlaneCount,nc);const auto starts=read(s.cellPlaneStart,nc);int planes=0;
    for(int c=0;c<nc && c<int(counts.size()) && c<int(starts.size());++c)if(starts[c]+counts[c]>planes)planes=starts[c]+counts[c];
    std::printf("R4_OBS {\"tag\":\"%s\",\"dt\":\"%a\",\"ledgerDt\":\"%a\",\"time\":\"%a\",\"w0\":\"%a\",\"w1\":\"%a\",\"rk\":%d,\"nc\":%d,\"nf\":%d,\"ni\":%d,\"moving\":%d,\"sst\":%d",tag,dt,ledgerDt,stageTime,w0,w1,owner->hostGasTimeIntegrator,nc,nf,s.nInternalFaces,int(s.gasGeometry.enabled),s.sstConfigured);
#define R4_CELL(name) field(#name,s.name,nc)
#define R4_FACE(name) field(#name,s.name,nf)
    R4_CELL(V);R4_CELL(rho);R4_CELL(rhoE);R4_CELL(rhoUx);R4_CELL(rhoUy);R4_CELL(rhoUz);
    R4_CELL(rhoNext);R4_CELL(rhoENext);R4_CELL(rhoUxNext);R4_CELL(rhoUyNext);R4_CELL(rhoUzNext);
    R4_CELL(Tgas);R4_CELL(p);R4_CELL(gasFluxPositivityScale);
    R4_CELL(cellPlaneStart);R4_CELL(cellPlaneCount);field("cellFaceId",s.cellFaceId,planes);
    R4_FACE(faceOwner);R4_FACE(faceNeighbour);R4_FACE(facePeriodicPair);R4_FACE(riemannBoundaryKind);
    R4_FACE(gasPhiRho);R4_FACE(gasPhiRhoE);R4_FACE(gasPhiRhoUx);R4_FACE(gasPhiRhoUy);R4_FACE(gasPhiRhoUz);
    if(s.gasGeometry.enabled){field("oldVolume",s.gasGeometry.oldVolume,nc);field("newVolume",s.gasGeometry.newVolume,nc);field("faceSweptVolume",s.gasGeometry.faceSweptVolume,nf);}
    if(s.sstConfigured){R4_CELL(rhoK);R4_CELL(rhoOmega);R4_CELL(rhoKInitial);R4_CELL(rhoOmegaInitial);R4_CELL(k);R4_CELL(omega);R4_CELL(sstSourceNumber);R4_FACE(sstPhiRhoK);R4_FACE(sstPhiRhoOmega);}
    using State=typename std::remove_reference<decltype(s)>::type;
    if constexpr(::ugkwp::GasStateTraits<State>::speciesCount>0){
        constexpr int ns=::ugkwp::GasStateTraits<State>::speciesCount;
        std::printf(",\"ns\":%d,\"gasMode\":%d",ns,int(s.gasSpecies.mode));
        field("speciesRho",s.gasSpecies.rho,ns*nc);field("speciesInitial",s.gasSpecies.initial,ns*nc);field("speciesFlux",s.gasSpecies.flux,ns*nf);
        field("cellStatus",s.gasSpecies.cellStatus,nc);field("faceStatus",s.gasSpecies.faceStatus,nf);
    }
#undef R4_CELL
#undef R4_FACE
    std::printf("}\n");std::fflush(stdout);
}
}
