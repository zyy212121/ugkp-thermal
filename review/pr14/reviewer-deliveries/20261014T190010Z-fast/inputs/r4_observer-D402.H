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

    if(s.sstConfigured){
        std::printf(",\"gasMu\":\"%a\",\"rhoMin\":\"%a\",\"sstKMin\":\"%a\",\"sstOmegaMin\":\"%a\",\"sstWallTreatment\":%d",double(s.gasMu),double(s.rhoMin),double(s.sstKMin),double(s.sstOmegaMin),s.sstWallTreatment);
        R4_CELL(sstWallDistance);R4_CELL(nut);R4_CELL(sstF1);R4_CELL(sstF2);
        R4_CELL(gradKX);R4_CELL(gradKY);R4_CELL(gradKZ);R4_CELL(gradOmegaX);R4_CELL(gradOmegaY);R4_CELL(gradOmegaZ);
        R4_CELL(gradUxX);R4_CELL(gradUxY);R4_CELL(gradUxZ);R4_CELL(gradUyX);R4_CELL(gradUyY);R4_CELL(gradUyZ);R4_CELL(gradUzX);R4_CELL(gradUzY);R4_CELL(gradUzZ);
        R4_FACE(sstBoundaryKMode);R4_FACE(sstBoundaryOmegaMode);
        std::printf(",\"coeff_alphaK1\":\"%a\"",double(s.sstCoefficients.alphaK1));
        std::printf(",\"coeff_alphaK2\":\"%a\"",double(s.sstCoefficients.alphaK2));
        std::printf(",\"coeff_alphaOmega1\":\"%a\"",double(s.sstCoefficients.alphaOmega1));
        std::printf(",\"coeff_alphaOmega2\":\"%a\"",double(s.sstCoefficients.alphaOmega2));
        std::printf(",\"coeff_beta1\":\"%a\"",double(s.sstCoefficients.beta1));
        std::printf(",\"coeff_beta2\":\"%a\"",double(s.sstCoefficients.beta2));
        std::printf(",\"coeff_betaStar\":\"%a\"",double(s.sstCoefficients.betaStar));
        std::printf(",\"coeff_gamma1\":\"%a\"",double(s.sstCoefficients.gamma1));
        std::printf(",\"coeff_gamma2\":\"%a\"",double(s.sstCoefficients.gamma2));
        std::printf(",\"coeff_a1\":\"%a\"",double(s.sstCoefficients.a1));
        std::printf(",\"coeff_b1\":\"%a\"",double(s.sstCoefficients.b1));
        std::printf(",\"coeff_c1\":\"%a\"",double(s.sstCoefficients.c1));
    }
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
