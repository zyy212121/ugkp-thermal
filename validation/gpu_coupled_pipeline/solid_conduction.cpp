#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuSolidThermalProperties.H"
#include "fixedGradientFvPatchFields.H"
#include "IStringStream.H"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
using namespace Foam;
using namespace Foam::gpuThermal;
int main(int argc,char** argv){
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    wordList types(mesh.boundary().size(),"zeroGradient");
    labelList patches(2);patches[0]=mesh.boundaryMesh().findPatchID("left");patches[1]=mesh.boundaryMesh().findPatchID("right");
    forAll(patches,p)types[patches[p]]="fixedGradient";
    forAll(mesh.boundary(),p)if(mesh.boundary()[p].coupled())types[p]=mesh.boundary()[p].type();
    volScalarField temperature(IOobject("T",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("initial",dimTemperature,300),types);
    const double pi=std::acos(-1.);
    forAll(temperature,c){const auto x=mesh.C()[c];temperature[c]=300+80*std::sin(pi*x.x())*std::sin(pi*x.y())*std::sin(2*pi*x.z());}
    temperature.correctBoundaryConditions();
    IStringStream propertyText("rho {type table; outOfBounds error; values ((100 0.9)(350 1.1)(1000 1.5));} Cp {type table; outOfBounds error; values ((100 1)(300 1.5)(450 2)(1000 3));} kappa {type table; outOfBounds error; values ((100 0.5)(300 1)(500 2)(1000 4));} emissivity {type constant; value 0.8;}");
    dictionary propertiesDictionary(propertyText);GpuSolidThermalProperties properties(propertiesDictionary);
    GpuSolidThermalControls controls;controls.nonlinearMaxIterations=100;
    controls.nonlinearRelativeTolerance=1e-10;controls.energyRelativeTolerance=1e-10;controls.energyAbsoluteTolerance=1e-10;
    controls.nonlinearAitken=runTime.controlDict().lookupOrDefault<bool>("conductionAitken",false);
    controls.nonlinearInitialRelaxation=.7;
    const double dt=runTime.controlDict().lookupOrDefault<scalar>("conductionInterval",.05);
    List<scalarField> energy(2);double prescribed=0;
    forAll(patches,p){const auto& patch=mesh.boundary()[patches[p]];energy[p].setSize(patch.size());
        forAll(patch,f){const vector x=patch.Cf()[f];const double flux=(p?-1.:3.)*(1+.2*std::cos(2*pi*x.y())*std::cos(2*pi*x.z()));
            energy[p][f]=flux*patch.magSf()[f]*dt;prescribed+=energy[p][f];}}
    GpuSolidThermalCandidateSolver solver(mesh,properties,patches,controls);
    const int repeats=runTime.controlDict().lookupOrDefault<label>("conductionRepeats",0);
    for(int repeat=0;repeat<=repeats;++repeat){
    const auto start=std::chrono::steady_clock::now();
    const auto candidate=solver.solveTemporarySolidCandidate(temperature,energy,dt);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double integrated=0,fluxError=0;
    forAll(temperature,c)integrated+=mesh.V()[c]*properties.deltaHv(candidate.temperature()[c],temperature[c]);
    forAll(patches,p){const label patchI=patches[p];const auto& field=refCast<const fixedGradientFvPatchScalarField>(candidate.temperature().boundaryField()[patchI]);
        forAll(field,f){const double wanted=energy[p][f]/(mesh.magSf().boundaryField()[patchI][f]*dt);
            fluxError=std::max(fluxError,std::abs(properties.kappa(field[f])*field.gradient()[f]-wanted));}}
    std::ofstream output((runTime.path()/("conduction-values-"+std::to_string(repeat)+".txt")).c_str());output<<std::setprecision(17);
    forAll(temperature,c)output<<candidate.temperature()[c]<<'\n';
    forAll(patches,p)forAll(candidate.temperature().boundaryField()[patches[p]],f)output<<candidate.temperature().boundaryField()[patches[p]][f]<<'\n';
    std::cout<<std::setprecision(17)<<"CONDUCTION_RESULT "<<mesh.nCells()<<' '<<candidate.nonlinearIterations<<' '<<seconds<<' '<<integrated-prescribed<<' '<<fluxError<<' '<<candidate.integratedEnergyResidualJ<<'\n';
    if(!std::isfinite(integrated)||std::abs(integrated-prescribed)>1e-8||!std::isfinite(fluxError)||fluxError>1e-8)return 2;
    }
}
