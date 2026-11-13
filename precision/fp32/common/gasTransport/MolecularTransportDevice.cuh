#pragma once
#include "MolecularTransportIO.H"
#include "GasModelCast.H"
#include "../GpuPrecisionTypes.H"
#include <cuda_runtime.h>
#include <memory>
namespace ugkwp {
// Initialization/reconfiguration owner only; no per-step work or host fields.
class MolecularTransportDevice {
    SutherlandSpecies<GpuReal>* species_=nullptr;
    PowerLawDiffusivity<GpuReal>* diffusion_=nullptr;
    GpuReal *quarter_=nullptr,*denominator_=nullptr;
    MolecularViscosityView<GpuReal> view_;
    MolecularTransportConfiguration host_;
    template<class T,class S> static cudaError_t upload(T*& dst,const std::vector<S>& src){
        if(src.empty())return cudaSuccess;
        std::vector<T> staged(src.size());for(std::size_t i=0;i<src.size();++i)staged[i]=castGasModel<T>(src[i]);
        auto e=cudaMalloc(reinterpret_cast<void**>(&dst),src.size()*sizeof(T));
        return e==cudaSuccess?cudaMemcpy(dst,staged.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice):e;
    }
public:
    MolecularTransportDevice()=default;
    MolecularTransportDevice(const MolecularTransportDevice&)=delete;
    MolecularTransportDevice& operator=(const MolecularTransportDevice&)=delete;
    ~MolecularTransportDevice(){cudaFree(species_);cudaFree(quarter_);cudaFree(denominator_);cudaFree(diffusion_);}
    cudaError_t initialize(const MolecularTransportConfiguration& config){
        host_=config;
        auto e=upload(species_,config.species);if(e!=cudaSuccess)return e;
        e=upload(quarter_,config.massRatioQuarter);if(e!=cudaSuccess)return e;
        e=upload(denominator_,config.wilkeDenominator);if(e!=cudaSuccess)return e;
        e=upload(diffusion_,config.diffusion);if(e!=cudaSuccess)return e;
        view_=castMolecularViscosityMetadata<GpuReal>(config.view());view_.species=species_;view_.massRatioQuarter=quarter_;view_.wilkeDenominator=denominator_;view_.diffusion=diffusion_;return cudaSuccess;
    }
    MolecularViscosityView<GpuReal> view()const{return view_;}
    MolecularViscosityView<GpuReal> hostView()const{return host_.view();}
};
} // namespace ugkwp
