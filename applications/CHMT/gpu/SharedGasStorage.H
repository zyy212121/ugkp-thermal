#ifndef CHMT_GPU_SHAREDGASSTORAGE_H
#define CHMT_GPU_SHAREDGASSTORAGE_H
#include "core/HostState.H"
#include "../../../common/gasTransport/GasStateView.H"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include <climits>
namespace chmt {
using CommonGasView=ugkwp::GasStateView<Real,Real,ugkwp::SstCoefficients>;
// Explicit integral/density staging, not a numerical backend. This only binds
// the five conserved arrays plus volume. Geometry, gradients, boundaries, RK
// buffers and device upload must be bound before any common operator launch.
// The current common G1 view has no species state: reject mixture inventories
// instead of losing species or pretending density is integrated mass.
class SharedGasDensityStorage {
    std::vector<Real> rho_,px_,py_,pz_,energy_,volume_;
public:
    bool importSingleSpecies(const HostState& source,int activeSpecies,std::string& error) {
        if(activeSpecies<0||activeSpecies>=Ns||source.gas.empty()
            ||source.gas.size()>INT_MAX||source.gas.size()!=source.gasMesh.volumes.size()){
            error="invalid single-species staging dimensions";return false;}
        SharedGasDensityStorage candidate;const auto count=source.gas.size();
        candidate.rho_.resize(count);candidate.px_.resize(count);candidate.py_.resize(count);
        candidate.pz_.resize(count);candidate.energy_.resize(count);candidate.volume_=source.gasMesh.volumes;
        for(std::size_t c=0;c<count;++c){
            const auto& q=source.gas[c];const Real volume=candidate.volume_[c];
            if(!finite(volume)||volume<=0||!finite(q.mass)||q.mass<=0||!finite(q.momentum)||!finite(q.energy)){
                error="nonfinite/invalid integral gas state or volume";return false;}
            for(int s=0;s<Ns;++s)if(q.species[s]!=(s==activeSpecies?q.mass:0)){
                error="common legacy density view cannot transport a mixture";return false;}
            candidate.rho_[c]=q.mass/volume;candidate.px_[c]=q.momentum.x/volume;
            candidate.py_[c]=q.momentum.y/volume;candidate.pz_[c]=q.momentum.z/volume;candidate.energy_[c]=q.energy/volume;
            if(!finite(candidate.rho_[c])||!finite(candidate.px_[c])||!finite(candidate.py_[c])
                ||!finite(candidate.pz_[c])||!finite(candidate.energy_[c])){
                error="integral-to-density conversion overflow";return false;}
        }
        *this=std::move(candidate);error.clear();return true;
    }
    CommonGasView view() {
        CommonGasView result;result.nCells=static_cast<int>(rho_.size());
        result.rho=rho_.data();result.rhoUx=px_.data();result.rhoUy=py_.data();result.rhoUz=pz_.data();
        result.rhoE=energy_.data();result.V=volume_.data();return result;
    }
    bool exportSingleSpecies(const std::vector<Real>& endpointVolume,int activeSpecies,
        std::vector<GasQ>& accepted,std::string& error) const {
        if(activeSpecies<0||activeSpecies>=Ns||endpointVolume.size()!=rho_.size()){
            error="invalid density-to-integral staging dimensions";return false;}
        // G1 is fixed-grid only. ALE conversion requires the common stage-volume
        // policy, not an application-local rescaling by arbitrary new volumes.
        if(endpointVolume!=volume_){error="ALE stage-volume policy is not wired to the shared gas view";return false;}
        std::vector<GasQ> candidate(rho_.size());
        for(std::size_t c=0;c<candidate.size();++c){auto& q=candidate[c];const Real v=endpointVolume[c];
            q.mass=rho_[c]*v;q.momentum={px_[c]*v,py_[c]*v,pz_[c]*v};q.energy=energy_[c]*v;q.species[activeSpecies]=q.mass;
            if(!finite(v)||v<=0||!finite(q.mass)||q.mass<=0||!finite(q.momentum)||!finite(q.energy)){
                error="invalid density candidate; accepted integral state unchanged";return false;}
        }
        accepted=std::move(candidate);error.clear();return true;
    }
};
}
#endif
