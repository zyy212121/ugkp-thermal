#pragma once
#include <fstream>
#include <iomanip>
// Optional observations only: copy existing raw residuals/face fluxes after
// their production evaluation. No operator, state update, or stop criterion.
void exportSteadyValidationSnapshot(DeviceState* s)
{
    const char* directory=std::getenv("GAS_UGKP_STEADY_DIAGNOSTICS_DIR");
    if(!directory || !*directory)return;
    static unsigned snapshot=0;
    const auto suffix=std::to_string(snapshot++);
    const int components=7+ugkwp::compiledGasSpecies;
    std::vector<double> residual(std::size_t(components)*s->nCells),volume(s->nCells);
    if(copyToHost(residual.data(),s->gasSteadyResidual,residual.size(),"export raw steady residual")
        ||copyToHost(volume.data(),s->V,volume.size(),"export cell volumes"))return;
    std::ofstream out(std::string(directory)+"/spatialResidual-"+suffix+".csv");
    if(!out){std::fprintf(stderr,"[validation-export] cannot open residual output\n");return;}
    out<<std::setprecision(17)<<"cell,volume";
    for(int k=0;k<components;++k)out<<",R"<<k;
    out<<'\n';
    for(int c=0;c<s->nCells;++c){out<<c<<','<<volume[c];for(int k=0;k<components;++k)out<<','<<residual[k*s->nCells+c];out<<'\n';}
    const double* pointers[5]={s->gasPhiRho,s->gasPhiRhoUx,s->gasPhiRhoUy,s->gasPhiRhoUz,s->gasPhiRhoE};
    std::vector<double> flux[5];
    for(int k=0;k<5;++k){flux[k].resize(s->nFaces);if(copyToHost(flux[k].data(),pointers[k],s->nFaces,"export raw FV face flux"))return;}
    std::ofstream faces(std::string(directory)+"/faceFlux-"+suffix+".csv");
    faces<<std::setprecision(17)<<"face,boundary,mass,momentumX,momentumY,momentumZ,energy\n";
    for(int f=0;f<s->nFaces;++f){faces<<f<<','<<(f>=s->nInternalFaces);for(int k=0;k<5;++k)faces<<','<<flux[k][f];faces<<'\n';}
}
