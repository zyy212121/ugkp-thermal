#pragma once
#include "Definition.H"
namespace ugkp { namespace boundary {
constexpr int FaceInts=16,FaceReals=12,FunctionInts=4;
constexpr int LegacyFaceInts=14,LegacyFaceReals=11,LegacyComponents=6;
inline void encode(const Definition& d,std::vector<int>& iv,std::vector<double>& rv) {
    iv={int(d.faces.size()),int(d.functions.size()),int(d.terms.size())};rv.clear();
    for(const auto& f:d.faces) {
        for(int v:{f.geometry,f.rhoMode,f.uMode,f.pMode,f.tMode,f.wave})iv.push_back(v);
        for(int v:f.function)iv.push_back(v);
        for(double v:{f.waveGamma,f.waveFieldInf,f.waveLInf,f.pressureGamma,f.temperatureGamma})rv.push_back(v);
        for(double v:f.value)rv.push_back(v);
        iv.push_back(f.compressiblePressure);iv.push_back(f.compositionMode);iv.push_back(f.solidMode);
    }
    for(const auto& f:d.functions) {
        for(int v:{int(f.kind),int(f.bounds),f.begin,f.count})iv.push_back(v);
        rv.push_back(f.constant);
    }
    for(const auto& t:d.terms){rv.push_back(t.x);rv.push_back(t.y);}
}
inline Definition decode(const std::vector<int>& iv,const std::vector<double>& rv) {
    if(iv.size()<3||iv[0]<0||iv[1]<0||iv[2]<0)throw std::runtime_error("invalid boundary wire counts");
    // Array lengths distinguish the historical six-component record from
    // the extended inlet record. Old checkpoints remain readable as written.
    const auto matches=[&](int ints,int reals) {
        return iv.size()==3+std::size_t(iv[0])*ints+std::size_t(iv[1])*FunctionInts
            &&rv.size()==std::size_t(iv[0])*reals+iv[1]+std::size_t(iv[2])*2;
    };
    const bool current=matches(FaceInts,FaceReals);
    const bool legacy=!current&&matches(LegacyFaceInts,LegacyFaceReals);
    if(!current&&!legacy)throw std::runtime_error("invalid boundary wire size");
    const int components=legacy?LegacyComponents:Components;
    Definition d;d.faces.resize(iv[0]);d.functions.resize(iv[1]);d.terms.resize(iv[2]);
    std::size_t i=3,j=0;
    for(auto& f:d.faces) {
        f.geometry=iv[i++];f.rhoMode=iv[i++];f.uMode=iv[i++];f.pMode=iv[i++];f.tMode=iv[i++];f.wave=iv[i++];
        for(int c=0;c<components;++c)f.function[c]=iv[i++];
        f.compressiblePressure=iv[i++];f.compositionMode=iv[i++];
        if(!legacy)f.solidMode=iv[i++];
        f.waveGamma=rv[j++];f.waveFieldInf=rv[j++];f.waveLInf=rv[j++];f.pressureGamma=rv[j++];f.temperatureGamma=rv[j++];
        for(int c=0;c<components;++c)f.value[c]=rv[j++];
    }
    for(auto& f:d.functions){f.kind=FunctionKind(iv[i++]);f.bounds=Bounds(iv[i++]);f.begin=iv[i++];f.count=iv[i++];f.constant=rv[j++];}
    for(auto& t:d.terms){t.x=rv[j++];t.y=rv[j++];}validate(d);return d;
}
}}
