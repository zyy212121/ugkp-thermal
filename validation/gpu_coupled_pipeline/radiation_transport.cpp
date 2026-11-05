#include "fvCFD.H"
#include "MieDoRadiationSolver.H"
#include "MieDoRadiationMath.H"
#include "IStringStream.H"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace Foam;
using namespace Foam::gpuThermal;
int main(int argc,char** argv){
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject(fvMesh::defaultRegion,runTime.timeName(),runTime,IOobject::MUST_READ));
    // Materialize the immutable cell-face cache before native OpenMP sweeps.
    (void)mesh.cells();
    MieTable table(runTime.path()/"constant/mie.table");
    const label right=mesh.boundaryMesh().findPatchID("right");labelList coupled(1,right);
    volScalarField wall(IOobject("wallTemperature",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("wall",dimTemperature,350),"fixedValue");
    List<scalarField> emissivity(mesh.boundary().size());
    forAll(mesh.boundary(),p){emissivity[p].setSize(mesh.boundary()[p].size());forAll(emissivity[p],f){
        const vector x=mesh.boundary()[p].Cf()[f];wall.boundaryFieldRef()[p][f]=350+100*x.y();emissivity[p][f]=.6+.1*std::cos(100*x.y());}}
    scalarField epsilon(mesh.nCells()),temperature(mesh.nCells()),diameter(mesh.nCells());
    forAll(epsilon,c){const vector x=mesh.C()[c];epsilon[c]=1e-4*(1+.1*std::sin(100*x.x()));temperature[c]=600+4000*x.x()+2000*x.z();diameter[c]=1.5e-5+5e-4*x.y();}
    IOdictionary configuration(IOobject("radiationProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    MieDoControls controls{3,8,300,1e-10,1e-12,200,0,1e-8,100,runTime.controlDict().lookupOrDefault<label>("angleThreads",1)};
    MieDoRadiationSolver solver(mesh,configuration,table,coupled,controls);auto workspace=solver.makeWorkspace();
    const int repeats=runTime.controlDict().lookupOrDefault<label>("radiationRepeats",1);
    for(int repeat=0;repeat<=repeats;++repeat){
        if(repeat==2)temperature+=scalar(3);
        MieDoSnapshot snapshot{epsilon,temperature,diameter,wall,emissivity};
        const auto start=std::chrono::steady_clock::now();const auto result=solver.solve(snapshot,workspace);
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        double particle=0,boundary=0,gross=0;
        forAll(epsilon,c){const double q=result.particlePowerDensityWm3[c]*mesh.V()[c];particle+=q;gross+=std::abs(q);}
        forAll(mesh.boundary(),p)forAll(mesh.boundary()[p],f){const double q=result.boundaryQrOutwardWm2[p][f]*mesh.magSf().boundaryField()[p][f];boundary+=q;gross+=std::abs(q);}
        if(!std::isfinite(particle)||!std::isfinite(boundary)||std::abs(particle+boundary)>1e-9+1e-8*gross)return 2;
        if(particle!=result.integratedParticlePowerW||boundary!=result.integratedBoundaryPowerW||particle+boundary!=result.conservationResidualW)return 3;
        std::ofstream output((runTime.path()/("radiation-values-"+std::to_string(repeat)+".txt")).c_str());output<<std::setprecision(17);
        for(const auto* field:{&result.absorptionCoefficientInvM,&result.scatteringCoefficientInvM,&result.extinctionCoefficientInvM,&result.asymmetryFactor,&result.incidentRadiationG_Wm2,&result.particlePowerDensityWm3})
            forAll(*field,c){if(!std::isfinite((*field)[c]))return 4;output<<(*field)[c]<<'\n';}
        forAll(result.boundaryQrOutwardWm2,p)forAll(result.boundaryQrOutwardWm2[p],f){const double q=result.boundaryQrOutwardWm2[p][f];if(!std::isfinite(q))return 4;output<<q<<'\n';}
        std::cout<<std::setprecision(17)<<"RADIATION_RESULT "<<repeat<<' '<<elapsed<<' '<<result.sourceIterations<<' '<<result.angleThreadsUsed<<' '<<particle<<' '<<boundary<<' '<<particle+boundary<<' '<<gross<<'\n';
    }
}
