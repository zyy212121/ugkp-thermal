#pragma once
#include "Definition.H"
#include <cuda_runtime.h>
#include <cstring>
#include <utility>

namespace ugkp { namespace boundary {
class Resident {
    Face* faces_=nullptr;
    Function* functions_=nullptr;
    Term* terms_=nullptr;
    cudaStream_t stream_=nullptr;
    Definition saved_;
    template<class T> static bool copy(T*& dst,const std::vector<T>& src,cudaStream_t stream,std::string& error) {
        if(src.empty())return true;
        auto code=cudaMalloc(reinterpret_cast<void**>(&dst),src.size()*sizeof(T));
        if(code==cudaSuccess)code=cudaMemcpyAsync(dst,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice,stream);
        if(code!=cudaSuccess){error=cudaGetErrorString(code);return false;}return true;
    }
public:
    Resident()=default;
    Resident(const Resident&)=delete;
    Resident& operator=(const Resident&)=delete;
    ~Resident(){clear();}
    void clear(){
        if(faces_||functions_||terms_)cudaStreamSynchronize(stream_);
        if(faces_)cudaFree(faces_);
        if(functions_)cudaFree(functions_);
        if(terms_)cudaFree(terms_);
        faces_=nullptr;functions_=nullptr;terms_=nullptr;
    }
    void swap(Resident& b)noexcept {
        std::swap(faces_,b.faces_);std::swap(functions_,b.functions_);std::swap(terms_,b.terms_);
        std::swap(stream_,b.stream_);saved_.faces.swap(b.saved_.faces);
        saved_.functions.swap(b.saved_.functions);saved_.terms.swap(b.saved_.terms);
    }
    View view()const{return {faces_,functions_,terms_,int(saved_.faces.size())};}
    bool configure(const Definition& input,cudaStream_t stream,std::string& error) {
        try{validate(input);}catch(const std::exception& e){error=e.what();return false;}
        Resident next;next.stream_=stream;next.saved_=input;
        bool ok=copy(next.faces_,next.saved_.faces,stream,error)
            &&copy(next.functions_,next.saved_.functions,stream,error)
            &&copy(next.terms_,next.saved_.terms,stream,error);
        // Configuration owns its sources and completes once, outside advancement.
        const auto code=cudaStreamSynchronize(stream);
        if(code!=cudaSuccess){error=cudaGetErrorString(code);ok=false;}
        if(!ok)return false;
        swap(next);return true;
    }
};
// Publish the shared definition to the existing SoA consumer ABI. All clients
// use this implementation; the callback only supplies their transfer stream.
template<class State,class Copy> bool publish(const Definition& d,State& s,Copy copy) {
    using Real=std::remove_pointer_t<decltype(s.gasBoundaryP)>;
    const auto n=d.faces.size();std::vector<int> integers(n);std::vector<Real> values(n);
#define INT_FIELD(member,field) \
    for(std::size_t i=0;i<n;++i)integers[i]=d.faces[i].member; \
    if(!copy(s.gasBoundary##field,integers)||!copy(s.riemannBoundary##field,integers))return false
    INT_FIELD(geometry,Kind);INT_FIELD(rhoMode,RhoFix);INT_FIELD(uMode,UFix);
    INT_FIELD(pMode,PFix);INT_FIELD(tMode,TFix);INT_FIELD(wave,PWave);
#undef INT_FIELD
#define VALUE_FIELD(member,field) \
    for(std::size_t i=0;i<n;++i)values[i]=Real(d.faces[i].member); \
    if(!copy(s.gasBoundary##field,values)||!copy(s.riemannBoundary##field,values))return false
    VALUE_FIELD(value[Density],Rho);VALUE_FIELD(value[Ux],Ux);VALUE_FIELD(value[Uy],Uy);VALUE_FIELD(value[Uz],Uz);
    VALUE_FIELD(value[Pressure],P);VALUE_FIELD(value[Temperature],T);
    VALUE_FIELD(waveGamma,PWaveGamma);VALUE_FIELD(waveFieldInf,PWaveFieldInf);VALUE_FIELD(waveLInf,PWaveLInf);
#undef VALUE_FIELD
    return true;
}
}}
