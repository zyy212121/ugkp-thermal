#pragma once
#include "materials/ImplicitMaterialBalance.H"
#include "materials/MaterialTransport.H"
#include "materials/MaterialWindowExchange.H"
#include <array>

namespace chmt {
class ImplicitMaterialMass {
    using Amount=implicitMaterial::Amount;
    using Face=implicitMaterial::Face;
    static constexpr auto components=implicitMaterial::components;
    std::vector<Amount> initial_,masses_;
    std::vector<std::array<PhysicalReal,Nr>> extents_;
    std::vector<Face> faces_;

public:
    bool begin(const HostState& base,const HostMesh& mesh,const MaterialWindowExchange& input,
        const Tolerances& tolerance,std::string& error,const std::vector<Real>& swept={}){
        const auto n=base.solid.size();
        if(input.cells.size()!=n||(!swept.empty()&&swept.size()!=mesh.owner.size())){error="material window exchange cell count mismatch";return false;}
        if(!validateMaterialTransportMesh(mesh,n,error))return false;
        initial_.resize(n);extents_.resize(n);
        for(std::size_t i=0;i<n;++i){
            const auto& q=base.solid[i];const auto& received=input.cells[i];
            for(int k=0;k<Nc+Ns;++k){
                const PhysicalReal before=k<Nc?q.condensed[k]:q.pore[k-Nc];
                auto accepted=received.mass[k];
                accepted.add(before);
                const PhysicalReal after=accepted.value();
                const PhysicalReal roundoff=8*std::numeric_limits<PhysicalReal>::epsilon()*std::abs(before);
                if(!std::isfinite(after)||after<-std::max(roundoff,PhysicalReal(tolerance.absoluteMass))){
                    error="fixed material exchange overdraws cell "+std::to_string(i);return false;
                }
                initial_[i][k]=std::max(PhysicalReal(0),after);
            }
            for(int r=0;r<Nr;++r)initial_[i][Nc+Ns+r]=q.progress[r];
        }
        for(int f=0;f<int(mesh.owner.size());++f){
            const auto kind=mesh.boundaryKind[f];
            int right=mesh.neighbour[f];
            if(kind==BoundaryKind::Periodic){
                const auto pair=mesh.periodicPartner[f];
                if(f>=pair)continue;
                right=mesh.owner[pair];
            }
            if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty||right==mesh.owner[f])continue;
            if(right>=0||(!swept.empty()&&swept[f]!=0))
                faces_.push_back({mesh.owner[f],right,f});
        }
        masses_=initial_;
        return true;
    }

    void reconstruct(const PhysicsConfig& p){
        masses_=initial_;
        for(const auto& f:faces_)for(int k=0;k<components;++k){
            masses_[f.left][k]-=f.transfer[k];
            if(f.right>=0)masses_[f.right][k]+=f.transfer[k];
        }
        if(!p.enableReactions)return;
        for(std::size_t i=0;i<masses_.size();++i)for(int r=0;r<p.nReactions;++r){
            implicitMaterial::applyExtent(masses_[i],p.reactions[r],r,extents_[i][r]);
        }
    }

    bool prepare(const std::vector<PhysicalReal>& temperatures,const std::vector<Real>& volumes,
        const PhysicsConfig& p,std::vector<SolidQ>& state,std::vector<MaterialCaloric>& caloric,
        std::vector<MaterialPrimitive>& primitive,std::string& error) const {
        for(std::size_t i=0;i<masses_.size();++i)
            if(const auto* failure=implicitMaterial::prepareCell(masses_[i],initial_[i],temperatures[i],
                PhysicalReal(volumes[i]),p,state[i],caloric[i],primitive[i])){
                error=std::string(failure)+" in cell "+std::to_string(i);return false;
            }
        return true;
    }

    bool transport(const HostMesh& mesh,const std::vector<Real>& swept,const PhysicsConfig& p,
        const std::vector<MaterialPrimitive>& primitive,const MaterialTransportResult& transport,
        PhysicalReal dt,PhysicalReal& residual,std::string& error){
        const bool darcy=p.permeability>0;
        const bool diffusion=std::any_of(std::begin(p.poreEffectiveDiffusivity),std::end(p.poreEffectiveDiffusivity),
            [](PhysicalReal value){return value>0;});
        for(auto& f:faces_){
            const auto slot=f.geometry;
            PhysicalReal spacing=0,area=0;
            if(f.right>=0){
                const auto areaVector=mesh.areaVectors[slot];
                const auto distance=mesh.neighbour[slot]>=0
                    ?mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                    :mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                        +mesh.faceCentres[slot]-mesh.faceCentres[mesh.periodicPartner[slot]];
                spacing=PhysicalReal(dot(distance,normalized(areaVector)));area=PhysicalReal(mag(areaVector));
            }
            if(const auto* failure=implicitMaterial::updateMass(f,masses_[f.left],f.right>=0?&masses_[f.right]:nullptr,
                mesh.volumes.data(),p,primitive.data(),transport.faceFlux[slot],transport.faceTransport[slot],
                transport.faceLossLeft[slot]+transport.faceLossRight[slot],spacing,area,
                swept.empty()?PhysicalReal(0):PhysicalReal(swept[slot]),dt,darcy,diffusion,residual)){
                error=failure;return false;
            }
        }
        return true;
    }

    bool correctEnergy(const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        const std::vector<MaterialCaloric>& caloric,const std::vector<EnergyReal>& conduction,
        std::vector<EnergyReal>& target,EnergyReal& boundaryInput,std::string& error){
        if(faces_.empty())return true;
        std::vector<implicitMaterial::CellThermal> cells(temperature.size());
        for(std::size_t c=0;c<cells.size();++c)
            if(!implicitMaterial::prepareThermalCell(caloric[c],temperature[c],cells[c])){
                error="inadmissible material state for implicit face energy";return false;
            }
        for(const auto& face:faces_)implicitMaterial::applyFaceEnergy(face,face.energy,target.data());
        for(auto& face:faces_){
            if(const auto* failure=implicitMaterial::updateEnergy(face,p,temperature.data(),cells.data(),conduction.data(),target.data())){
                error=failure;return false;
            }
            if(face.right<0)boundaryInput-=face.energy;
        }
        return true;
    }

    bool checkEnergy(const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        const std::vector<SolidQ>& state,EnergyReal absolute,EnergyReal relative,EnergyReal& residual,
        std::string& error)const{
        for(const auto& face:faces_)
            if(!implicitMaterial::checkFaceEnergy(face,p,temperature.data(),state.data(),absolute,relative,residual)){
                error="nonfinite implicit transport energy residual";return false;
            }
        return true;
    }

    bool react(const HostMesh& mesh,const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        PhysicalReal dt,PhysicalReal& residual,std::string& error){
        if(!p.enableReactions)return true;
        for(std::size_t i=0;i<masses_.size();++i)
            if(!implicitMaterial::updateReactions(masses_[i],extents_[i].data(),PhysicalReal(mesh.volumes[i]),
                temperature[i],p,dt,residual)){
                error="nonfinite implicit material reaction in cell "+std::to_string(i);return false;
            }
        return true;
    }
};
}
