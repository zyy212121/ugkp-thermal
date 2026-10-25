#pragma once
#include <ostream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <cmath>
namespace ugkwp {
// Transport storage keeps its V1 ABI. Only available quantities have CSV
// interfaces: flow/species here; finite off-wall SST in its separate opt-in file.
inline bool writeGasBoundaryLayerProfiles(std::ostream& output,
    const std::vector<int>& faces,const std::vector<double>& centres,
    const std::vector<double>& geometry,const std::vector<std::string>& species,
    int nodes,const double* profiles,double stageTime,double outputTime)
{
    const std::size_t columns=7+species.size();
    for(std::size_t i=0;i<faces.size()*nodes;++i)
        for(std::size_t col=0;col<columns;++col)
            if(col!=5 && col!=6 && !std::isfinite(profiles[i*columns+col]))return false;
    output<<std::setprecision(std::numeric_limits<double>::max_digits10);
    output<<"face,faceX,faceY,faceZ,normalX,normalY,normalZ,node,stageTime,outputTime,y,Ux,Uy,Uz,T";
    for(const auto& name:species)output<<",Y_"<<name;
    output<<'\n';
    for(std::size_t slot=0;slot<faces.size();++slot)
        for(int node=0;node<nodes;++node)
        {
            output<<faces[slot];
            for(int d=0;d<3;++d)output<<','<<centres[3*slot+d];
            for(int d=0;d<3;++d)output<<','<<geometry[7*slot+d];
            output<<','<<node<<','<<stageTime<<','<<outputTime;
            const double* row=profiles+(slot*nodes+node)*columns;
            for(std::size_t col=0;col<columns;++col)if(col!=5 && col!=6)output<<','<<row[col];
            output<<'\n';
        }
    return bool(output);
}
inline bool writeGasBoundaryLayerSstProfiles(std::ostream& output,
    bool localSst,const std::vector<int>& faces,int nodes,std::size_t species,
    const double* profiles,double stageTime,double outputTime)
{
    if(!localSst)return false; // No local-SST export interface for algebraic ODE.
    const std::size_t columns=7+species;
    // node 0 is the wall boundary, where omega has a singular asymptotic limit.
    // Do not manufacture a finite wall omega or put an infinity into a data row.
    for(std::size_t slot=0;slot<faces.size();++slot)
        for(int node=1;node<nodes;++node){
            const double* r=profiles+(slot*nodes+node)*columns;
            if(!std::isfinite(r[0])||!std::isfinite(r[5])||!std::isfinite(r[6]))return false;
        }
    output<<std::setprecision(std::numeric_limits<double>::max_digits10);
    output<<"face,node,stageTime,outputTime,y,k,omega\n";
    for(std::size_t slot=0;slot<faces.size();++slot)
        for(int node=1;node<nodes;++node){
            const double* r=profiles+(slot*nodes+node)*columns;
            output<<faces[slot]<<','<<node<<','<<stageTime<<','<<outputTime<<','<<r[0]<<','<<r[5]<<','<<r[6]<<'\n';
        }
    return bool(output);
}
inline bool writeGasBoundaryLayerDiagnostics(std::ostream& output,
    bool localSst,std::size_t walls,const std::vector<std::string>& species,const double* data)
{
    const char* names[]={"face","owner","matchingDistance","ownerDistance","heatFluxIntoGas",
        "tractionX","tractionY","tractionZ","wallKFlux","integratedKSource","ownerOmega",
        "residual","iterations","traceDensity","viscosity","stageTime","heatConduction",
        "heatViscous","heatTransport","heatBalanceResidual","ownerKContent","ownerKDestruction",
        "ownerVolume","ownerKBalanceResidual","volumeSstClosure"};
    const auto active=[&](std::size_t col){return localSst ||
        !(col==8||col==9||col==10||col==20||col==21||col==23||col==24);};
    const std::size_t columns=25+2*species.size();
    for(std::size_t row=0;row<walls;++row)
        for(std::size_t col=0;col<columns;++col)
            if(active(col)&&!std::isfinite(data[row*columns+col]))return false;
    output<<std::setprecision(std::numeric_limits<double>::max_digits10);
    for(std::size_t col=0;col<25;++col)if(active(col)){if(col)output<<',';output<<names[col];}
    for(const auto& s:species)output<<",speciesFlux_"<<s;
    for(const auto& s:species)output<<",reactionIntegral_"<<s;
    output<<'\n';
    for(std::size_t row=0;row<walls;++row){
        for(std::size_t col=0;col<columns;++col)if(active(col)){if(col)output<<',';output<<data[row*columns+col];}
        output<<'\n';
    }
    return bool(output);
}
}
