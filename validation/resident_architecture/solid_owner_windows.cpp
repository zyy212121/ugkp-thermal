#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuSolidThermalProperties.H"
#include "IStringStream.H"
#include <iomanip>
#include <iostream>
#ifdef UGKP_TEST_RESIDENT_OWNER
#include <cuda_runtime_api.h>
static void cudaCheck(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
#endif
using namespace Foam;
using namespace Foam::gpuThermal;
int main(int argc,char** argv){
    argList::addBoolOption("variable-properties","use temperature-dependent solid properties");
    argList::addBoolOption("aitken","enable the existing nonlinear Aitken policy");
    argList::addBoolOption("varying-initial-state","initialize a smooth nonuniform temperature");
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    volScalarField temperature(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
    labelList interfaces(2);interfaces[0]=mesh.boundaryMesh().findPatchID("left");interfaces[1]=mesh.boundaryMesh().findPatchID("right");
    if(args.optionFound("varying-initial-state"))forAll(temperature,c){const auto& point=mesh.C()[c];
        temperature[c]=300+20*std::sin(point.x())+30*std::cos(point.y())+7*point.z();}
    temperature.correctBoundaryConditions();
    const bool variable=args.optionFound("variable-properties");
    IStringStream text(variable
        ? "rho {type table; outOfBounds error; values ((100 0.9)(350 1.1)(1000 1.5));} Cp {type table; outOfBounds error; values ((100 1)(300 1.5)(450 2)(1000 3));} kappa {type table; outOfBounds error; values ((100 0.5)(300 1)(500 2)(1000 4));} emissivity {type constant; value 0;}"
        : "rho {type constant; value 1;} Cp {type constant; value 1;} kappa {type constant; value 1;} emissivity {type constant; value 0;}");
    dictionary dict(text);GpuSolidThermalProperties properties(dict);
    GpuSolidThermalControls controls;controls.nonlinearMaxIterations=100;controls.nonlinearRelativeTolerance=1e-12;
    controls.nonlinearAitken=args.optionFound("aitken");
    controls.energyRelativeTolerance=1e-12;controls.energyAbsoluteTolerance=1e-10;
    List<scalarField> energy(2);label count=0;
    forAll(interfaces,p){energy[p]=scalarField(mesh.boundary()[interfaces[p]].size(),0);count+=energy[p].size();}
    GpuSolidThermalCandidateSolver solver(mesh,properties,interfaces,controls);
    const scalar dt=.25;
#ifdef UGKP_TEST_RESIDENT_OWNER
    double *deviceEnergy=nullptr,*deviceInterval=nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceEnergy),count*sizeof(double)));
    cudaCheck(cudaMemset(deviceEnergy,0,count*sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceInterval),sizeof(double)));
    cudaCheck(cudaMemcpy(deviceInterval,&dt,sizeof(double),cudaMemcpyHostToDevice));
    if(!solver.prepareResident(temperature,{true,int(count),deviceEnergy,nullptr,deviceInterval},nullptr))
        throw std::runtime_error("resident owner was not prepared");
#endif
    std::cout<<std::setprecision(17);
    for(int window=1;window<=4;++window){
        const scalarField oldTemperature(temperature.primitiveField());
        runTime.setTime(window*dt,window);
#ifdef UGKP_TEST_RESIDENT_OWNER
        solver.advanceResident(nullptr);
        solver.publishResident(temperature);
#else
        const auto result=solver.solveTemporarySolidCandidate(temperature,energy,dt);
        solver.publishSolidCandidate(result,temperature);
#endif
        long double energyChange=0;
        forAll(temperature,c)energyChange+=static_cast<long double>(properties.deltaHv(temperature[c],oldTemperature[c]))*mesh.V()[c];
        std::cout<<"WINDOW_ENERGY "<<window<<' '<<energyChange<<'\n';
        if(!std::isfinite(energyChange)||std::abs(energyChange)>2e-8)return 2;
        forAll(temperature,c)std::cout<<"WINDOW_CELL "<<window<<' '<<c<<' '<<temperature[c]<<'\n';
        // Native evaluation supplies an independent boundary/flux check at the
        // same physical time; it is an output observer, not a device input.
        temperature.correctBoundaryConditions();
        const auto gradient=fvc::snGrad(temperature);
        forAll(gradient().boundaryField(),p){const auto& grad=gradient().boundaryField()[p];
            forAll(grad,f)std::cout<<"WINDOW_FLUX "<<window<<' '<<p<<' '<<f<<' '<<-grad[f]*mesh.magSf().boundaryField()[p][f]<<'\n';}
    }
#ifdef UGKP_TEST_RESIDENT_OWNER
    cudaCheck(cudaFree(deviceEnergy));cudaCheck(cudaFree(deviceInterval));
#endif
    return 0;
}
