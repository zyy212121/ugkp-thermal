#ifndef CHMT_GPU_SHAREDMIXTURESTORAGE_H
#define CHMT_GPU_SHAREDMIXTURESTORAGE_H
#include "gpu/SharedGasStorage.H"
#include "configuration/SharedGasModel.H"
#include "materials/Thermodynamics.H"
namespace chmt {
struct CommonMixtureGasView : CommonGasView {
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
};
// Host staging for the actual common species capability. This is deliberately
// not a launch-ready backend: mesh metrics, boundary policies, common bulk
// gradients/RK buffers and CUDA copies are bound by the future device owner.
// No flux, reconstruction, chemistry or time-step algorithm is defined here.
class SharedMixtureStorage {
    ugkwp::GasModelConfiguration model_;
    std::array<std::vector<Real>,5> bulk_;
    std::vector<Real> volume_,nut_,rhoSpecies_,initial_,flux_,gx_,gy_,gz_,limiter_;
    std::vector<Real> boundary_,positivity_,sound_,cp_,gasConstant_;
    std::vector<int> boundaryFixed_,cellStatus_,faceStatus_;
public:
    bool import(const HostState& source,const ugkwp::GasModelConfiguration& shared,
        const ModelConfig& material,std::string& error) {
        const auto cells=source.gas.size(),faces=source.gasMesh.owner.size();
        if(!cells||cells>INT_MAX||faces>INT_MAX||source.gasMesh.volumes.size()!=cells){
            error="invalid mixture staging dimensions";return false;}
        ModelConfig validated=material;
        if(!bindSharedGasModel(shared,validated,error))return false;
        SharedMixtureStorage candidate;candidate.model_=shared;candidate.volume_=source.gasMesh.volumes;
        for(auto& field:candidate.bulk_)field.resize(cells);
        candidate.nut_.assign(cells,0); // Required even by laminar shared gas helpers.
        candidate.rhoSpecies_.resize(Ns*cells);candidate.initial_.resize(Ns*cells);
        candidate.flux_.assign(Ns*faces,0);candidate.gx_.assign(Ns*cells,0);
        candidate.gy_.assign(Ns*cells,0);candidate.gz_.assign(Ns*cells,0);
        candidate.limiter_.assign(cells,1);candidate.boundary_.assign(Ns*faces,0);
        candidate.boundaryFixed_.assign(faces,0);candidate.positivity_.assign(Ns*cells,1);
        candidate.sound_.resize(cells);candidate.cp_.resize(cells);candidate.gasConstant_.resize(cells);
        candidate.cellStatus_.assign(cells,0);candidate.faceStatus_.assign(faces,0);
        const auto thermo=shared.thermoView<Ns>();
        for(std::size_t c=0;c<cells;++c){const auto& q=source.gas[c];const auto volume=candidate.volume_[c];
            GasPrimitive primitive;if(!recoverGas(q,volume,validated.physics,primitive,nullptr,static_cast<int>(c))){
                error="invalid integral mixture state for shared storage";return false;}
            candidate.bulk_[0][c]=q.mass/volume;candidate.bulk_[1][c]=q.momentum.x/volume;
            candidate.bulk_[2][c]=q.momentum.y/volume;candidate.bulk_[3][c]=q.momentum.z/volume;
            candidate.bulk_[4][c]=q.energy/volume;Real heatCapacity=0,gasConstant=0;
            for(int s=0;s<Ns;++s){const auto i=s*cells+c;candidate.rhoSpecies_[i]=q.species[s]/volume;
                candidate.initial_[i]=candidate.rhoSpecies_[i];
                heatCapacity+=primitive.Y[s]*ugkwp::speciesCp(s,primitive.temperature,thermo);
                gasConstant+=primitive.Y[s]*validated.physics.species[s].R;
            }
            candidate.sound_[c]=primitive.soundSpeed;candidate.cp_[c]=heatCapacity;candidate.gasConstant_[c]=gasConstant;
            for(const auto& field:candidate.bulk_)if(!finite(field[c])){error="integral-to-density conversion overflow";return false;}
        }
        if(source.gasMesh.boundaryPrimitive.size()==faces)
            for(std::size_t f=0;f<faces;++f)for(int s=0;s<Ns;++s)
                candidate.boundary_[s*faces+f]=source.gasMesh.boundaryPrimitive[f].Y[s];
        *this=std::move(candidate);error.clear();return true;
    }
    CommonMixtureGasView view() {
        // Refresh every borrowed pointer after owner copy/move/reallocation.
        CommonMixtureGasView out;out.nCells=static_cast<int>(volume_.size());out.nFaces=static_cast<int>(faceStatus_.size());
        out.rho=bulk_[0].data();out.rhoUx=bulk_[1].data();out.rhoUy=bulk_[2].data();out.rhoUz=bulk_[3].data();out.rhoE=bulk_[4].data();
        out.V=volume_.data();out.nut=nut_.data();auto& s=out.gasSpecies;
        s.mode=model_.mode;s.thermo=model_.thermoView<Ns>();s.thermoControls=model_.chemistryControls.thermo;
        s.rho=rhoSpecies_.data();s.initial=initial_.data();s.flux=flux_.data();
        s.gradX=gx_.data();s.gradY=gy_.data();s.gradZ=gz_.data();s.limiter=limiter_.data();
        s.boundaryMassFraction=boundary_.data();s.compositionBoundaryFixed=boundaryFixed_.data();
        s.positivityScale=positivity_.data();s.diffusivity=model_.diffusionCoefficients.data();
        s.turbulentSchmidt=model_.turbulentSchmidt;s.soundSpeed=sound_.data();s.heatCapacity=cp_.data();s.gasConstant=gasConstant_.data();
        s.cellStatus=cellStatus_.data();s.faceStatus=faceStatus_.data();return out;
    }

};
}
#endif
