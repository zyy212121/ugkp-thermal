#include "ParticleRadiationMaterial.H"
#include "RadiationImplicitMath.H"
#include "PinnedTransferQueue.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checkedMaterial(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA particle radiation: ")+cudaGetErrorString(status));}
template<class T> struct MaterialArray {
    T* ptr=nullptr;std::size_t capacity=0;
    ~MaterialArray(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t size){if(size<=capacity)return;T* next=nullptr;checkedMaterial(cudaMalloc(reinterpret_cast<void**>(&next),size*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=size;}
    void upload(const std::vector<T>& values,PinnedTransferQueue& transfer){reserve(values.size());transfer.upload(ptr,values.data(),values.size()*sizeof(T));}
};
constexpr double fourPiMaterial=12.56637061435917295385057353311801153679;
__host__ __device__ double materialBlack(double t,double band){const double square=t*t;return band*5.670374419e-8*square*square/(fourPiMaterial/4);}
struct MaterialBalance {
    MieOpticalTableView table;double diameter,incident;
    __host__ __device__ double operator()(double temperature)const{return incident-fourPiMaterial*materialBlack(temperature,mieBand(table,temperature));}
};
struct MaterialPower {
    MaterialBalance balance;double epsilon,volume;
    __host__ __device__ double operator()(double temperature)const{
        const auto d=mieLogBracket(balance.table.diameter,balance.table.diameters,balance.diameter);
        const auto t=mieLinearBracket(balance.table.temperature,balance.table.temperatures,temperature);
        const double absorption=3*epsilon*mieBilinear(balance.table.absorption,balance.table.temperatures,d,t)/(2*balance.diameter);
        // The original zero-absorption short circuit is significant for IEEE values.
        return volume*(absorption==0?0:absorption*balance(temperature));
    }
};
__global__ void materialRoots(int count,MieOpticalTableView table,const double* epsilon,const double* mass,const double* temperature,
    const double* diameter,const double* incident,const double* volume,double heatFactor,double elapsed,double minimum,double maximum,double cutoff,
    double* next,double* energy){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=count)return;
    const double old=temperature[cell];next[cell]=old;energy[cell]=0;if(!(epsilon[cell]>cutoff))return;
    const double capacity=mass[cell]*heatFactor;
    const MaterialBalance balance{table,diameter[cell],incident[cell]};const MaterialPower power{balance,epsilon[cell],volume[cell]};
    const auto root=radiationMath::solveBoundedImplicitMaterialTemperature(old,minimum,maximum,capacity,elapsed,100,1e-10,1e-9,balance,power);
    next[cell]=root.temperatureK;energy[cell]=capacity*(root.temperatureK-old);
}
__global__ void rawBoundaryEnergy(int count,double elapsed,const double* area,const double* flux,double* energy){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face<count)energy[face]=elapsed*area[face]*flux[face];
}
__global__ void conservativeMaterialBoundary(int cells,int faces,const double* accepted,const double* raw,const unsigned char* coupled,
    double* boundary,double* escaped){
    if(threadIdx.x)return;
    // These are the native ordered transaction sums, deliberately unreordered.
    double acceptedTotal=0,gross=0,rawTotal=0;
    for(int cell=0;cell<cells;++cell){acceptedTotal+=accepted[cell];gross+=fabs(accepted[cell]);}
    for(int face=0;face<faces;++face){rawTotal+=raw[face];gross+=fabs(raw[face]);}
    const auto scaling=radiationMath::conservativeBoundaryEnergyScale(acceptedTotal,rawTotal,gross,1e-10);
    double other=0;
    for(int face=0;face<faces;++face){const double value=scaling.scale*raw[face];boundary[face]=0;if(coupled[face])boundary[face]=value;else other+=value;}
    *escaped=other;
}
}
struct ParticleRadiationMaterial::Implementation {
    MieOpticalProperties optical;cudaStream_t stream=nullptr;PinnedTransferQueue transfer;
    MaterialArray<double> epsilon,mass,temperature,diameter,incident,volume,area,flux,next,energy,raw,boundary,escaped;
    MaterialArray<unsigned char> coupled;
    explicit Implementation(const MieOpticalTable& table):optical(table){checkedMaterial(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);}
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
ParticleRadiationMaterial::ParticleRadiationMaterial(const MieOpticalTable& table):data_(new Implementation(table)){}
ParticleRadiationMaterial::~ParticleRadiationMaterial()=default;
ParticleRadiationMaterialResult ParticleRadiationMaterial::solve(const ParticleRadiationMaterialInput& input){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);const int cells=input.temperature.size(),faces=input.boundaryArea.size();
    d.epsilon.upload(input.epsilon,d.transfer);d.mass.upload(input.mass,d.transfer);d.temperature.upload(input.temperature,d.transfer);
    d.diameter.upload(input.diameter,d.transfer);d.incident.upload(input.incident,d.transfer);d.volume.upload(input.volume,d.transfer);
    d.area.upload(input.boundaryArea,d.transfer);d.flux.upload(input.boundaryFlux,d.transfer);d.coupled.upload(input.coupledBoundary,d.transfer);
    d.next.reserve(cells);d.energy.reserve(cells);d.raw.reserve(faces);d.boundary.reserve(faces);d.escaped.reserve(1);
    if(cells)materialRoots<<<(cells+127)/128,128,0,d.stream>>>(cells,d.optical.deviceTable(),d.epsilon.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr,
        d.incident.ptr,d.volume.ptr,input.heatFactor,input.elapsed,input.minimumTemperature,input.maximumTemperature,input.activeEpsilonMinimum,d.next.ptr,d.energy.ptr);
    if(faces)rawBoundaryEnergy<<<(faces+127)/128,128,0,d.stream>>>(faces,input.elapsed,d.area.ptr,d.flux.ptr,d.raw.ptr);
    conservativeMaterialBoundary<<<1,1,0,d.stream>>>(cells,faces,d.energy.ptr,d.raw.ptr,d.coupled.ptr,d.boundary.ptr,d.escaped.ptr);
    ParticleRadiationMaterialResult result;result.temperature.resize(cells);result.energy.resize(cells);result.boundaryEnergy.resize(faces);
    d.transfer.download(result.temperature.data(),d.next.ptr,cells*sizeof(double));d.transfer.download(result.energy.data(),d.energy.ptr,cells*sizeof(double));
    d.transfer.download(result.boundaryEnergy.data(),d.boundary.ptr,faces*sizeof(double));d.transfer.download(&result.escapedEnergy,d.escaped.ptr,sizeof(double));d.transfer.finish();return result;
}
}}
