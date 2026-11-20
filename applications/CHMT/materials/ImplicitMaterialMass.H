#pragma once
#include "materials/MaterialTransport.H"
#include "materials/MaterialWindowExchange.H"
#include "materials/ReactionStepControl.H"
#include <algorithm>
#include <array>

namespace chmt {
class ImplicitMaterialMass {
    static constexpr int components=Nc+Ns+Nr;
    using Amount=std::array<Real,components>;
    struct Face {
        int left,right,geometry;
        Amount transfer{};
    };
    std::vector<Amount> initial_,masses_;
    std::vector<std::array<Real,Nr>> extents_;
    std::vector<Face> faces_;

    static SolidQ inventory(const Amount& mass){
        SolidQ q;
        for(int c=0;c<Nc;++c)q.condensed[c]=PhysicalReal(mass[c]);
        for(int s=0;s<Ns;++s)q.pore[s]=PhysicalReal(mass[Nc+s]);
        for(int r=0;r<Nr;++r)q.progress[r]=PhysicalReal(mass[Nc+Ns+r]);
        return q;
    }

    static Real residualRatio(Real residual,Real a,Real b,Real amount,const Tolerances& tolerance){
        const Real scale=std::max(std::abs(a),std::abs(b));
        const Real roundoff=8*std::numeric_limits<PhysicalReal>::epsilon()*std::abs(amount);
        const Real allowed=tolerance.absoluteMass+tolerance.relativeMass*scale+roundoff;
        return residual==0?0:std::abs(residual)/std::max(allowed,std::numeric_limits<Real>::min());
    }
    static bool poreCorrections(const HostMesh& mesh,const Face& face,const PhysicsConfig& p,
        const MaterialPrimitive& left,const MaterialPrimitive& right,const MaterialTransportResult& transport,
        const Amount& donor,const Amount& defect,Real dt,Real aleSensitivity,Real spacing,Real area,
        bool leftDonor,Amount& correction) {
        const auto slot=face.geometry;
        const Real leftVolume=left.porosity*mesh.volumes[face.left];
        const Real rightVolume=right.porosity*mesh.volumes[face.right];
        const Real donorVolume=leftDonor?leftVolume:rightVolume;
        const Real donorDensity=leftDonor?left.poreDensity:right.poreDensity;
        const Real conductance=p.permeability>0&&spacing>0?p.permeability*area/(p.poreViscosity*spacing):0;
        const Real orthogonalAdvection=donorVolume>0?dt*conductance*std::abs(Real(left.pressure)-right.pressure)/donorVolume:0;
        const Real baseDiagonal=1+aleSensitivity+orthogonalAdvection;
        const Real compositionDiagonal=baseDiagonal+dt*(transport.faceLossLeft[slot]+transport.faceLossRight[slot]);
        const Real pressureResponse=dt*conductance*donorDensity
            *((leftVolume>0?left.temperature/leftVolume:0)+(rightVolume>0?right.temperature/rightVolume:0));
        if(!(baseDiagonal>0)||!(compositionDiagonal>0)||!finite(compositionDiagonal)
            ||!finite(pressureResponse)||pressureResponse<0)return false;

        Real donorMass=0,netDefect=0;
        for(int s=0;s<Ns;++s){
            donorMass+=std::max(Real(0),donor[Nc+s]);
            netDefect+=defect[Nc+s];
        }
        Real fraction[Ns]{},netDiagonal=baseDiagonal;
        for(int s=0;s<Ns;++s){
            fraction[s]=donorMass>0?std::max(Real(0),donor[Nc+s])/donorMass:Real(1)/Ns;
            correction[Nc+s]=(defect[Nc+s]-fraction[s]*netDefect)/compositionDiagonal;
        }
        for(int s=0;s<Ns;++s){
            const Real sensitivity=pressureResponse*p.species[s].R;
            netDiagonal+=sensitivity*fraction[s];
            netDefect-=sensitivity*correction[Nc+s];
        }
        if(!(netDiagonal>0)||!finite(netDiagonal)||!finite(netDefect))return false;
        const Real netCorrection=netDefect/netDiagonal;
        for(int s=0;s<Ns;++s)correction[Nc+s]+=fraction[s]*netCorrection;
        return true;
    }

public:
    bool begin(const HostState& base,const HostMesh& mesh,const MaterialWindowExchange& input,
        const Tolerances& tolerance,std::string& error){
        const auto n=base.solid.size();
        if(input.cells.size()!=n){error="material window exchange cell count mismatch";return false;}
        if(!validateMaterialTransportMesh(mesh,n,error))return false;
        initial_.resize(n);extents_.resize(n);
        for(std::size_t i=0;i<n;++i){
            const auto& q=base.solid[i];const auto& received=input.cells[i];
            for(int k=0;k<Nc+Ns;++k){
                const Real before=k<Nc?q.condensed[k]:q.pore[k-Nc];
                auto accepted=received[k];
                accepted.add(before);
                const Real after=accepted.value();
                const Real roundoff=8*std::numeric_limits<PhysicalReal>::epsilon()*std::abs(before);
                if(!std::isfinite(after)||after<-std::max(roundoff,Real(tolerance.absoluteMass))){
                    error="fixed material exchange overdraws cell "+std::to_string(i);return false;
                }
                initial_[i][k]=std::max(Real(0),after);
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
            if(right>=0&&right!=mesh.owner[f]&&kind!=BoundaryKind::Interface&&kind!=BoundaryKind::Empty)
                faces_.push_back({mesh.owner[f],right,f,{}});
        }
        masses_=initial_;
        return true;
    }

    void reconstruct(const PhysicsConfig& p){
        masses_=initial_;
        for(const auto& f:faces_)for(int k=0;k<components;++k){
            masses_[f.left][k]-=f.transfer[k];masses_[f.right][k]+=f.transfer[k];
        }
        if(!p.enableReactions)return;
        for(std::size_t i=0;i<masses_.size();++i)for(int r=0;r<p.nReactions;++r){
            const Real extent=extents_[i][r];const auto& reaction=p.reactions[r];
            for(int c=0;c<Nc;++c)masses_[i][c]+=extent*reaction.condensedNu[c];
            for(int s=0;s<Ns;++s)masses_[i][Nc+s]+=extent*reaction.gasNu[s];
            masses_[i][Nc+Ns+r]+=extent;
        }
    }

    bool prepare(const std::vector<Real>& temperatures,const std::vector<Real>& volumes,
        const PhysicsConfig& p,std::vector<SolidQ>& state,std::vector<MaterialCaloric>& caloric,
        std::vector<MaterialPrimitive>& primitive,std::string& error) const {
        for(std::size_t i=0;i<masses_.size();++i){
            auto mass=masses_[i];
            for(int k=0;k<components;++k){
                if(!finite(mass[k])||mass[k]<-32*std::numeric_limits<Real>::epsilon()
                    *std::max(Real(1),std::abs(initial_[i][k]))){
                    error="invalid implicit inventory in cell "+std::to_string(i);return false;
                }
                if(mass[k]<0)mass[k]=0;
            }
            auto q=inventory(mass);
            if(!materialCaloric(q,p,caloric[i],error))return false;
            if(!finite(temperatures[i])||temperatures[i]<=0){
                error="invalid coupled material temperature in cell "+std::to_string(i);return false;
            }
            q.energy=EnergyReal(caloric[i].energy(PhysicalReal(temperatures[i])));
            if(!recoverMaterial(q,volumes[i],p,primitive[i])){
                error="inadmissible implicit material constitutive state in cell "+std::to_string(i);return false;
            }
            q.porosity=primitive[i].porosity;state[i]=q;
        }
        return true;
    }

    bool transport(const HostMesh& mesh,const std::vector<Real>& swept,const PhysicsConfig& p,
        const std::vector<MaterialPrimitive>& primitive,const MaterialTransportResult& transport,
        Real dt,Real& residual,std::string& error){
        for(auto& f:faces_){
            auto& left=masses_[f.left];auto& right=masses_[f.right];const auto& flux=transport.faceFlux[f.geometry];
            const auto area=mesh.areaVectors[f.geometry];
            const auto distance=mesh.neighbour[f.geometry]>=0
                ?mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                :mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                    +mesh.faceCentres[f.geometry]-mesh.faceCentres[mesh.periodicPartner[f.geometry]];
            const Real spacing=dot(distance,normalized(area));
            const auto& a=primitive[f.left];const auto& b=primitive[f.right];
            const Real ale=swept.empty()?0:swept[f.geometry];
            const Real aleSensitivity=std::abs(ale)/mesh.volumes[ale<=0?f.left:f.right];
            Amount defect{},correction{};
            for(int k=0;k<components;++k){
                const Real flow=k<Nc?flux.condensed[k]:k<Nc+Ns?flux.pore[k-Nc]:flux.progress[k-Nc-Ns];
                const Real desired=dt*flow;
                defect[k]=desired-f.transfer[k];
                correction[k]=defect[k]/(1+aleSensitivity);
                if(!finite(desired)||!finite(defect[k])){
                    error="nonfinite implicit material mass flux";return false;
                }
                residual=std::max(residual,residualRatio(defect[k],desired,f.transfer[k],
                    std::max(left[k],right[k]),p.tolerances));
            }
            const bool leftDonor=p.permeability>0?a.pressure>=b.pressure:ale<=0;
            if(!poreCorrections(mesh,f,p,a,b,transport,leftDonor?left:right,defect,dt,aleSensitivity,
                spacing,mag(area),leftDonor,correction)){
                error="nonfinite implicit pore mass preconditioner";return false;
            }
            for(int k=0;k<components;++k){
                if(!finite(correction[k])){error="nonfinite implicit mass correction";return false;}
                const Real limited=std::clamp(correction[k],-std::max(Real(0),right[k]),std::max(Real(0),left[k]));
                left[k]-=limited;right[k]+=limited;f.transfer[k]+=limited;
            }
        }
        return true;
    }

    bool react(const HostMesh& mesh,const PhysicsConfig& p,const std::vector<Real>& temperature,
        Real dt,Real& residual,std::string& error){
        if(!p.enableReactions)return true;
        for(std::size_t i=0;i<masses_.size();++i){
            auto& mass=masses_[i];const Real volume=mesh.volumes[i];
            const auto T=PhysicalReal(temperature[i]);
            for(int r=0;r<p.nReactions;++r){
                const auto& reaction=p.reactions[r];Real& extent=extents_[i][r];
                const auto q=inventory(mass);
                const Real rate=reactionRate<false>(reaction,q.condensed,PhysicalReal(volume),T);
                Real derivative=0,lower=-extent,upper=std::numeric_limits<Real>::max(),scale=0;
                for(int c=0;c<Nc;++c)derivative+=reaction.condensedNu[c]
                    *reactionMassDerivative(reaction,c,q,PhysicalReal(volume),T,PhysicalReal(rate));
                for(int k=0;k<Nc+Ns;++k){
                    const Real nu=k<Nc?reaction.condensedNu[k]:reaction.gasNu[k-Nc];
                    if(nu<0)upper=std::min(upper,std::max(Real(0),mass[k])/(-nu));
                    if(nu>0)lower=std::max(lower,-std::max(Real(0),mass[k])/nu);
                    if(nu!=0)scale=std::max(scale,std::abs(mass[k]/nu));
                }
                const Real desired=dt*rate,defect=desired-extent;
                if(!finite(desired)||rate<0||!finite(derivative)){
                    error="nonfinite implicit material reaction in cell "+std::to_string(i);return false;
                }
                const Real admissible=std::clamp(defect,lower,upper);
                residual=std::max(residual,residualRatio(admissible,desired,extent,scale,p.tolerances));
                const Real delta=std::clamp(defect/(1+dt*std::max(Real(0),-derivative)),lower,upper);
                extent+=delta;
                for(int c=0;c<Nc;++c)mass[c]+=delta*reaction.condensedNu[c];
                for(int s=0;s<Ns;++s)mass[Nc+s]+=delta*reaction.gasNu[s];
                mass[Nc+Ns+r]+=delta;
            }
        }
        return true;
    }
};
}
