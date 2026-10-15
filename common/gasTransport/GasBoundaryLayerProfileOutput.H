#pragma once
#include <ostream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
namespace ugkwp {
// Shared CSV serializer: rows retain the solver's node and mechanism order.
// stageTime labels the accepted wall solve; outputTime labels the field directory.
inline void writeGasBoundaryLayerProfiles(std::ostream& output,
    const std::vector<int>& faces,const std::vector<double>& centres,
    const std::vector<double>& geometry,const std::vector<std::string>& species,
    int nodes,const double* profiles,double stageTime,double outputTime)
{
    output<<std::setprecision(std::numeric_limits<double>::max_digits10);
    output<<"face,faceX,faceY,faceZ,normalX,normalY,normalZ,node,stageTime,outputTime,y,Ux,Uy,Uz,T,k,omega";
    for(const auto& name:species)output<<",Y_"<<name;
    output<<'\n';
    const std::size_t columns=7+species.size();
    for(std::size_t slot=0;slot<faces.size();++slot)
        for(int node=0;node<nodes;++node)
        {
            output<<faces[slot];
            for(int d=0;d<3;++d)output<<','<<centres[3*slot+d];
            for(int d=0;d<3;++d)output<<','<<geometry[7*slot+d];
            output<<','<<node<<','<<stageTime<<','<<outputTime;
            const double* row=profiles+(slot*nodes+node)*columns;
            for(std::size_t col=0;col<columns;++col)output<<','<<row[col];
            output<<'\n';
        }
}
}
