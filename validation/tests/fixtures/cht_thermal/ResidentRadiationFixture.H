#ifndef UGKP_TEST_RESIDENT_RADIATION_FIXTURE_H
#define UGKP_TEST_RESIDENT_RADIATION_FIXTURE_H
#include "gpu/RadiationResidentView.H"
#include "gpu/ThermalInterfaceDeviceView.H"
#ifndef UGKP_TEST_LEGACY_THERMAL_OWNER
#include "gpu/ParticleRadiationResponse.H"
#endif
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
namespace Foam {
// Test-only owner of the same fixed physical snapshot used by the former host
// fixture. Production radiation still runs unchanged CUDA operators.
struct ResidentRadiationFixture {
    double* storage=nullptr;std::size_t capacity=0;scalarField mean;
    double* interfaceStorage=nullptr;std::size_t interfaceCapacity=0;
    ~ResidentRadiationFixture(){if(storage)cudaFree(storage);if(interfaceStorage)cudaFree(interfaceStorage);}
    static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
    ugkp::gpu::RadiationParticleDeviceView stage(const fvMesh& mesh,const scalarField& mass,const scalarField& temperature,
        const scalarField& diameter,double density,double minimum){
        const std::size_t n=mesh.nCells();if(3*n>capacity){if(storage)check(cudaFree(storage));storage=nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&storage),3*n*sizeof(double)));capacity=3*n;}
        mean=temperature;std::vector<double> values(3*n);
        for(std::size_t c=0;c<n;++c){values[c]=mass[c];values[n+c]=mass[c]*temperature[c];values[2*n+c]=mass[c]*diameter[c];}
        if(n)check(cudaMemcpy(storage,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice));
        return {true,int(n),mesh.nInternalFaces(),mesh.nFaces()-mesh.nInternalFaces(),storage,n?storage+n:nullptr,n?storage+2*n:nullptr,
            nullptr,nullptr,nullptr,density,minimum,nullptr};
    }
    ugkp::gpu::ThermalInterfaceEnergyDeviceView stageInterfaceEnergy(const List<scalarField>& energy){
        std::vector<double> values;forAll(energy,p)forAll(energy[p],f)values.push_back(energy[p][f]);
        if(values.size()>interfaceCapacity){if(interfaceStorage)check(cudaFree(interfaceStorage));interfaceStorage=nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&interfaceStorage),values.size()*sizeof(double)));interfaceCapacity=values.size();}
        if(!values.empty())check(cudaMemcpy(interfaceStorage,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice));
        return {true,int(values.size()),interfaceStorage,nullptr};
    }
    void readEnergy(const ugkp::gpu::RadiationMaterialDeviceView& result,scalarField& energy)const{
        energy.setSize(result.cells);if(result.cells)check(cudaMemcpy(energy.begin(),result.energy,result.cells*sizeof(double),cudaMemcpyDeviceToHost));
    }
};
}
#endif
