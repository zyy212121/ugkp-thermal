#pragma once
#include "Definition.H"
namespace ugkp { namespace boundary {
constexpr int FaceInts=14,FaceReals=11,FunctionInts=4;
// Term time and polynomial coefficient have different storage widths. The
// legacy wire remains two doubles per term; function metadata selects meaning.
inline std::vector<unsigned char> termUsage(const Definition& d) {
    std::vector<unsigned char> usage(d.terms.size(),0);
    for(const auto& f:d.functions)if(f.begin>=0&&f.count>0)
        for(std::size_t i=std::size_t(f.begin),end=i+std::size_t(f.count);i<end&&i<usage.size();++i)
            usage[i]|=f.kind==FunctionKind::Polynomial?2:1;
    return usage;
}
inline void encode(const Definition& d,std::vector<int>& iv,std::vector<double>& rv) {
    iv={int(d.faces.size()),int(d.functions.size()),int(d.terms.size())};
    for(const auto& f:d.faces) {
        for(int v:{f.geometry,f.rhoMode,f.uMode,f.pMode,f.tMode,f.wave})iv.push_back(v);
        for(int v:f.function)iv.push_back(v);
        for(double v:{f.waveGamma,f.waveFieldInf,f.waveLInf,f.pressureGamma,f.temperatureGamma})rv.push_back(v);
        for(double v:f.value)rv.push_back(v);
        iv.push_back(f.compressiblePressure);iv.push_back(f.compositionMode);
    }
    for(const auto& f:d.functions) {
        for(int v:{int(f.kind),int(f.bounds),f.begin,f.count})iv.push_back(v);
        rv.push_back(f.constant);
    }
    const auto usage=termUsage(d);
    for(std::size_t i=0;i<d.terms.size();++i) {
        const auto& t=d.terms[i];
        rv.push_back(usage[i]==2?double(t.coefficient):t.time);
        rv.push_back(double(t.value));
    }
}
inline Definition decode(const std::vector<int>& iv,const std::vector<double>& rv) {
    if(iv.size()<3||iv[0]<0||iv[1]<0||iv[2]<0)throw std::runtime_error("invalid boundary wire counts");
    if(iv.size()!=3+std::size_t(iv[0])*14+std::size_t(iv[1])*4
        ||rv.size()!=std::size_t(iv[0])*11+iv[1]+std::size_t(iv[2])*2)
        throw std::runtime_error("invalid boundary wire size");
    Definition d;d.faces.resize(iv[0]);d.functions.resize(iv[1]);d.terms.resize(iv[2]);
    std::size_t i=3,j=0;
    for(auto& f:d.faces) {
        f.geometry=iv[i++];f.rhoMode=iv[i++];f.uMode=iv[i++];f.pMode=iv[i++];f.tMode=iv[i++];f.wave=iv[i++];
        for(auto& v:f.function)v=iv[i++];
        f.compressiblePressure=iv[i++];f.compositionMode=iv[i++];
        f.waveGamma=rv[j++];f.waveFieldInf=rv[j++];f.waveLInf=rv[j++];f.pressureGamma=rv[j++];f.temperatureGamma=rv[j++];
        for(auto& v:f.value)v=rv[j++];
    }
    for(auto& f:d.functions){f.kind=FunctionKind(iv[i++]);f.bounds=Bounds(iv[i++]);f.begin=iv[i++];f.count=iv[i++];f.constant=rv[j++];}
    const auto usage=termUsage(d);
    for(std::size_t k=0;k<d.terms.size();++k) {
        auto& t=d.terms[k];const double x=rv[j++],y=rv[j++];
        t.time=usage[k]==2?GpuTime(0):x;
        t.coefficient=usage[k]&2?GpuReal(x):GpuReal(0);t.value=GpuReal(y);
    }
    validate(d);return d;
}
}}
