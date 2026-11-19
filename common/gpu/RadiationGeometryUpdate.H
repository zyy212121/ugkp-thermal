#pragma once
#include "DiscreteOrdinateRadiation.H"
#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace ugkp { namespace gpu { namespace radiationGeometryDetail {
inline int sign(double value){
    const RadiationReal stored=static_cast<RadiationReal>(value);
    return (stored>0)-(stored<0);
}
inline bool sameTopology(const RadiationGeometry& a,const RadiationGeometry& b){
    return a.cells==b.cells&&a.directions==b.directions&&a.owner==b.owner
        &&a.neighbour==b.neighbour&&a.cellOffsets==b.cellOffsets&&a.cellFaces==b.cellFaces
        &&a.boundaryOwner==b.boundaryOwner;
}
inline void validate(const RadiationGeometry& g){
    const auto fail=[](const char* field){throw std::invalid_argument(std::string("unusable radiation geometry field: ")+field);};
    if(g.cells<=0||g.directions<=0)fail("cell/direction count");
    const std::size_t cells=g.cells,angles=g.directions,faces=g.owner.size(),boundary=g.boundaryOwner.size();
    if(cells*angles>INT_MAX||angles*angles>INT_MAX||faces>INT_MAX||boundary>INT_MAX
        ||angles*boundary>INT_MAX||cells+boundary>INT_MAX)fail("addressable extent");
    const auto extent=[&](std::size_t actual,std::size_t expected,const char* name){if(actual!=expected)fail(name);};
    extent(g.neighbour.size(),faces,"neighbour");extent(g.cellOffsets.size(),cells+1,"cellOffsets");
    if(g.cellOffsets.front()!=0||g.cellOffsets.back()<0||std::size_t(g.cellOffsets.back())!=g.cellFaces.size())fail("cellOffsets extent");
    std::vector<unsigned char> incidence(faces,0);
    for(std::size_t f=0;f<faces;++f)if(g.owner[f]<0||g.owner[f]>=g.cells||g.neighbour[f]<0
        ||g.neighbour[f]>=g.cells||g.owner[f]==g.neighbour[f])fail("internal face owner");
    for(int cell=0;cell<g.cells;++cell){
        if(g.cellOffsets[cell]<0||g.cellOffsets[cell]>g.cellOffsets[cell+1]||g.cellOffsets[cell+1]>g.cellOffsets.back())fail("cellOffsets order");
        for(int i=g.cellOffsets[cell];i<g.cellOffsets[cell+1];++i){const int face=g.cellFaces[i];
            if(face<0||std::size_t(face)>=faces)fail("cellFaces index");
            const unsigned bit=cell==g.owner[face]?1:cell==g.neighbour[face]?2:0;
            if(!bit||(incidence[face]&bit))fail("cell face incidence");incidence[face]|=bit;
        }
    }
    for(auto present:incidence)if(present!=3)fail("missing cell face incidence");
    extent(g.boundaryPatch.size(),boundary,"boundaryPatch");extent(g.boundaryFace.size(),boundary,"boundaryFace");
    extent(g.boundaryType.size(),boundary,"boundaryType");
    for(std::size_t f=0;f<boundary;++f){
        if(g.boundaryOwner[f]<0||g.boundaryOwner[f]>=g.cells||g.boundaryPatch[f]<0||g.boundaryFace[f]<0)fail("boundary address");
        if(int(g.boundaryType[f])<int(RadiationBoundary::Diffuse)||int(g.boundaryType[f])>int(RadiationBoundary::Specular))fail("boundary type");
    }
    const auto values=[&](const std::vector<double>& v,std::size_t count,const char* name,int lower){
        extent(v.size(),count,name);
        for(double value:v){const RadiationReal stored=static_cast<RadiationReal>(value);
            if(!std::isfinite(value)||!std::isfinite(stored)||(lower==1&&!(stored>0))||(lower==0&&stored<0))fail(name);
        }
    };
    values(g.volume,cells,"volume",1);values(g.weights,angles,"weights",1);
    values(g.directionCosines,angles*angles,"directionCosines",-1);
    values(g.internalFlux,angles*faces,"internalFlux",-1);
    values(g.transportDiagonal,angles*cells,"transportDiagonal",0);
    values(g.boundaryFlux,angles*boundary,"boundaryFlux",-1);
    values(g.boundaryCosine,angles*boundary,"boundaryCosine",-1);
    values(g.boundaryArea,boundary,"boundaryArea",1);
    values(g.diffuseDenominator,boundary,"diffuseDenominator",0);
    values(g.fixedIntensity,boundary,"fixedIntensity",0);

    for(std::size_t f=0;f<boundary;++f){
        if(g.boundaryType[f]!=RadiationBoundary::Diffuse||RadiationReal(g.diffuseDenominator[f])>0)continue;
        for(std::size_t a=0;a<angles;++a){
            const auto row=a*boundary+f;
            if(sign(g.boundaryFlux[row])<0||sign(g.boundaryCosine[row])<0)
                fail("diffuse denominator on incoming ordinate");
        }
    }
    if(!g.reflectedDirection.empty()){
        extent(g.reflectedDirection.size(),angles*boundary,"reflectedDirection");
        for(int direction:g.reflectedDirection)if(direction<0||direction>=g.directions)fail("reflected direction index");
    }
    const bool specular=std::find(g.boundaryType.begin(),g.boundaryType.end(),RadiationBoundary::Specular)!=g.boundaryType.end();
    if(specular||!g.reflectionOffsets.empty()){
        extent(g.reflectionOffsets.size(),angles*boundary+1,"reflectionOffsets");
        if(g.reflectionOffsets.front()!=0||g.reflectionOffsets.back()<0
            ||std::size_t(g.reflectionOffsets.back())!=g.reflectionDonors.size())fail("reflectionOffsets extent");
        for(std::size_t i=0;i+1<g.reflectionOffsets.size();++i)if(g.reflectionOffsets[i]<0
            ||g.reflectionOffsets[i]>g.reflectionOffsets[i+1]||g.reflectionOffsets[i+1]>g.reflectionOffsets.back())fail("reflectionOffsets order");
    }else if(!g.reflectionDonors.empty()||!g.reflectionWeights.empty())fail("reflectionOffsets missing");
    values(g.reflectionWeights,g.reflectionDonors.size(),"reflectionWeights",0);
    for(int direction:g.reflectionDonors)if(direction<0||direction>=g.directions)fail("reflection donor index");
}
}}}
