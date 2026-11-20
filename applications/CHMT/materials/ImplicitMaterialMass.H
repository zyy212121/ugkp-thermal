#pragma once
#include "materials/MaterialTransport.H"
#include "materials/MaterialWindowExchange.H"
#include "materials/ReactionStepControl.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace chmt {
class ImplicitMaterialMass {
    static constexpr int components=Nc+Ns+Nr;
    using Amount=std::array<PhysicalReal,components>;
    struct Face {
        int left,right,geometry;
        Amount transfer{};
        PhysicalReal sweptPore[Ns]{},darcyPore[Ns]{};
        bool sweptFromLeft=true,darcyFromLeft=true;
        EnergyReal sweptTemperatureOffset=0,darcyTemperatureOffset=0,energy=0;
    };
    struct PoreResponse {
        PhysicalReal velocityDiagonal=0;
        PhysicalReal pressure[Ns]{},fraction[Ns]{};
    };
    struct FaceEnergy {
        EnergyReal value=0,leftSlope=0,rightSlope=0;
    };
    std::vector<Amount> initial_,masses_;
    std::vector<std::array<PhysicalReal,Nr>> extents_;
    std::vector<Face> faces_;

    static SolidQ inventory(const Amount& mass){
        SolidQ q;
        for(int c=0;c<Nc;++c)q.condensed[c]=PhysicalReal(mass[c]);
        for(int s=0;s<Ns;++s)q.pore[s]=PhysicalReal(mass[Nc+s]);
        for(int r=0;r<Nr;++r)q.progress[r]=PhysicalReal(mass[Nc+Ns+r]);
        return q;
    }

    static PhysicalReal residualRatio(PhysicalReal residual,PhysicalReal a,PhysicalReal b,PhysicalReal amount,const Tolerances& tolerance){
        const PhysicalReal scale=std::max(std::abs(a),std::abs(b));
        const PhysicalReal roundoff=8*std::numeric_limits<PhysicalReal>::epsilon()*std::abs(amount);
        const PhysicalReal allowed=tolerance.absoluteMass+tolerance.relativeMass*scale+roundoff;
        return residual==0?0:std::abs(residual)/std::max(allowed,std::numeric_limits<PhysicalReal>::min());
    }
    static bool poreCorrections(const HostMesh& mesh,const Face& face,const PhysicsConfig& p,
        const MaterialPrimitive& left,const MaterialPrimitive& right,const MaterialTransportResult& transport,
        const Amount& donor,const Amount& defect,PhysicalReal dt,PhysicalReal aleSensitivity,PhysicalReal spacing,PhysicalReal area,
        bool leftDonor,Amount& correction,PoreResponse& response) {
        const auto slot=face.geometry;
        const PhysicalReal leftVolume=left.porosity*PhysicalReal(mesh.volumes[face.left]);
        const PhysicalReal rightVolume=right.porosity*PhysicalReal(mesh.volumes[face.right]);
        const PhysicalReal donorVolume=leftDonor?leftVolume:rightVolume;
        const PhysicalReal donorDensity=leftDonor?left.poreDensity:right.poreDensity;
        const PhysicalReal conductance=p.permeability>0&&spacing>0?p.permeability*area/(p.poreViscosity*spacing):0;
        const PhysicalReal orthogonalAdvection=donorVolume>0?dt*conductance*std::abs(PhysicalReal(left.pressure)-right.pressure)/donorVolume:0;
        response.velocityDiagonal=orthogonalAdvection;
        const PhysicalReal baseDiagonal=PhysicalReal(1)+aleSensitivity+orthogonalAdvection;
        const PhysicalReal compositionDiagonal=baseDiagonal+dt*(transport.faceLossLeft[slot]+transport.faceLossRight[slot]);
        const PhysicalReal pressureResponse=dt*conductance*donorDensity
            *((leftVolume>0?left.temperature/leftVolume:0)+(rightVolume>0?right.temperature/rightVolume:0));
        if(!(baseDiagonal>0)||!(compositionDiagonal>0)||!finite(compositionDiagonal)
            ||!finite(pressureResponse)||pressureResponse<0)return false;

        PhysicalReal donorMass=0,netDefect=0;
        for(int s=0;s<Ns;++s){
            donorMass+=std::max(PhysicalReal(0),donor[Nc+s]);
            netDefect+=defect[Nc+s];
        }
        PhysicalReal netDiagonal=baseDiagonal;
        for(int s=0;s<Ns;++s){
            response.fraction[s]=donorMass>0?std::max(PhysicalReal(0),donor[Nc+s])/donorMass:PhysicalReal(1)/Ns;
            correction[Nc+s]=(defect[Nc+s]-response.fraction[s]*netDefect)/compositionDiagonal;
        }
        for(int s=0;s<Ns;++s){
            const PhysicalReal sensitivity=pressureResponse*p.species[s].R;
            response.pressure[s]=sensitivity;
            netDiagonal+=sensitivity*response.fraction[s];
            netDefect-=sensitivity*correction[Nc+s];
        }
        if(!(netDiagonal>0)||!finite(netDiagonal)||!finite(netDefect))return false;
        const PhysicalReal netCorrection=netDefect/netDiagonal;
        for(int s=0;s<Ns;++s)correction[Nc+s]+=response.fraction[s]*netCorrection;
        return true;
    }

    template<bool Linearize>
    static FaceEnergy faceEnergy(const Face& face,const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature){
        const EnergyReal left=EnergyReal(temperature[face.left]);
        const EnergyReal right=face.right>=0?EnergyReal(temperature[face.right]):left;
        const EnergyReal sweptT=(face.sweptFromLeft?left:right)+face.sweptTemperatureOffset;
        const EnergyReal meanT=EnergyReal(.5)*left+EnergyReal(.5)*right;
        FaceEnergy result;
        auto& sweptSlope=face.sweptFromLeft?result.leftSlope:result.rightSlope;
        for(int c=0;c<Nc;++c){
            const EnergyReal mass=EnergyReal(face.transfer[c]);
            if(mass==0)continue;
            const auto& phase=p.condensed[c];
            result.value+=mass*condensedE(phase,sweptT);
            if constexpr(Linearize)sweptSlope+=mass*(EnergyReal(phase.cp0)+EnergyReal(phase.cp1)*sweptT);
        }
        for(int s=0;s<Ns;++s){
            const auto& species=p.species[s];
            const EnergyReal swept=EnergyReal(face.sweptPore[s]);
            const EnergyReal darcy=EnergyReal(face.darcyPore[s]);
            const EnergyReal diffusive=EnergyReal(face.transfer[Nc+s])-swept-darcy;
            if(swept!=0){
                result.value+=swept*speciesE(species,sweptT);
                if constexpr(Linearize)sweptSlope+=swept*speciesCv(species,sweptT);
            }
            if(darcy!=0){
                const bool fromLeft=darcy>0;
                const EnergyReal T=(fromLeft?left:right)
                    +(fromLeft==face.darcyFromLeft?face.darcyTemperatureOffset:EnergyReal(0));
                result.value+=darcy*speciesH(species,T);
                if constexpr(Linearize)(fromLeft?result.leftSlope:result.rightSlope)+=darcy*speciesCp(species,T);
            }
            if(diffusive!=0){
                result.value+=diffusive*speciesH(species,meanT);
                if constexpr(Linearize){
                    const EnergyReal slope=EnergyReal(.5)*diffusive*speciesCp(species,meanT);
                    result.leftSlope+=slope;result.rightSlope+=slope;
                }
            }
        }
        return result;
    }

    static void applyFaceEnergy(const Face& face,EnergyReal flux,std::vector<EnergyReal>& target){
        target[face.left]-=flux;
        if(face.right>=0)target[face.right]+=flux;
    }

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
            const PhysicalReal extent=extents_[i][r];const auto& reaction=p.reactions[r];
            for(int c=0;c<Nc;++c)masses_[i][c]+=extent*reaction.condensedNu[c];
            for(int s=0;s<Ns;++s)masses_[i][Nc+s]+=extent*reaction.gasNu[s];
            masses_[i][Nc+Ns+r]+=extent;
        }
    }

    bool prepare(const std::vector<PhysicalReal>& temperatures,const std::vector<Real>& volumes,
        const PhysicsConfig& p,std::vector<SolidQ>& state,std::vector<MaterialCaloric>& caloric,
        std::vector<MaterialPrimitive>& primitive,std::string& error) const {
        for(std::size_t i=0;i<masses_.size();++i){
            auto mass=masses_[i];
            for(int k=0;k<components;++k){
                if(!finite(mass[k])||mass[k]<-std::max(p.tolerances.absoluteMass,
                    PhysicalReal(32)*std::numeric_limits<PhysicalReal>::epsilon()*std::abs(initial_[i][k]))){
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
            if(!recoverMaterial(q,PhysicalReal(volumes[i]),p,primitive[i])){
                error="inadmissible implicit material constitutive state in cell "+std::to_string(i);return false;
            }
            q.porosity=primitive[i].porosity;state[i]=q;
        }
        return true;
    }

    bool transport(const HostMesh& mesh,const std::vector<Real>& swept,const PhysicsConfig& p,
        const std::vector<MaterialPrimitive>& primitive,const MaterialTransportResult& transport,
        PhysicalReal dt,PhysicalReal& residual,std::string& error){
        const bool darcy=p.permeability>0;
        const bool diffusion=std::any_of(std::begin(p.poreEffectiveDiffusivity),std::end(p.poreEffectiveDiffusivity),
            [](PhysicalReal diffusivity){return diffusivity>0;});
        for(auto& f:faces_){
            auto& left=masses_[f.left];
            auto* right=f.right>=0?&masses_[f.right]:nullptr;
            const auto& flux=transport.faceFlux[f.geometry];
            const auto& split=transport.faceTransport[f.geometry];
            const PhysicalReal ale=swept.empty()?PhysicalReal(0):PhysicalReal(swept[f.geometry]);
            f.sweptFromLeft=ale<=0||!right;
            const int sweptDonor=f.sweptFromLeft?f.left:f.right;
            const PhysicalReal aleSensitivity=std::abs(ale)/PhysicalReal(mesh.volumes[sweptDonor]);
            f.sweptTemperatureOffset=ale!=0?split.sweptTemperature-EnergyReal(primitive[sweptDonor].temperature):EnergyReal(0);
            f.darcyFromLeft=split.darcyFromLeft;
            if(right){
                const int darcyDonor=f.darcyFromLeft?f.left:f.right;
                f.darcyTemperatureOffset=split.darcyTemperature-EnergyReal(primitive[darcyDonor].temperature);
            }
            Amount defect{},correction{};
            for(int k=0;k<components;++k){
                const PhysicalReal flow=k<Nc?flux.condensed[k]:k<Nc+Ns?flux.pore[k-Nc]:flux.progress[k-Nc-Ns];
                const PhysicalReal desired=dt*flow;
                defect[k]=desired-f.transfer[k];
                correction[k]=defect[k]/(PhysicalReal(1)+aleSensitivity);
                if(!finite(desired)||!finite(defect[k])){
                    error="nonfinite implicit material mass flux";return false;
                }
                residual=std::max(residual,residualRatio(defect[k],desired,f.transfer[k],
                    right?std::max(left[k],(*right)[k]):left[k],p.tolerances));
            }
            PoreResponse response;
            if(right){
                const auto areaVector=mesh.areaVectors[f.geometry];
                const auto distance=mesh.neighbour[f.geometry]>=0
                    ?mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                    :mesh.cellCentres[f.right]-mesh.cellCentres[f.left]
                        +mesh.faceCentres[f.geometry]-mesh.faceCentres[mesh.periodicPartner[f.geometry]];
                const auto& a=primitive[f.left];const auto& b=primitive[f.right];
                const bool leftDonor=darcy?f.darcyFromLeft:f.sweptFromLeft;
                if(!poreCorrections(mesh,f,p,a,b,transport,leftDonor?left:*right,defect,dt,aleSensitivity,
                    PhysicalReal(dot(distance,normalized(areaVector))),PhysicalReal(mag(areaVector)),
                    leftDonor,correction,response)){
                    error="nonfinite implicit pore mass preconditioner";return false;
                }
            }
            PhysicalReal fraction=1;
            for(int k=0;k<components;++k){
                const PhysicalReal delta=correction[k];
                if(!finite(delta)){error="nonfinite implicit mass correction";return false;}
                const PhysicalReal leftAvailable=std::max(PhysicalReal(0),left[k]);
                if(delta>leftAvailable)fraction=std::min(fraction,leftAvailable/delta);
                if(right){
                    const PhysicalReal rightAvailable=std::max(PhysicalReal(0),(*right)[k]);
                    if(-delta>rightAvailable)fraction=std::min(fraction,rightAvailable/(-delta));
                }
            }
            PhysicalReal pressureCorrection=0;
            for(int s=0;s<Ns;++s)pressureCorrection+=response.pressure[s]*correction[Nc+s];
            for(int s=0;s<Ns;++s){
                const PhysicalReal sweptGoal=dt*split.sweptPore[s],darcyGoal=dt*split.darcyPore[s];
                const auto inventory=right?std::max(left[Nc+s],(*right)[Nc+s]):left[Nc+s];
                const PhysicalReal priorDiffusion=f.transfer[Nc+s]-f.sweptPore[s]-f.darcyPore[s];
                const PhysicalReal diffusiveGoal=dt*split.diffusivePore[s];
                residual=std::max({residual,
                    residualRatio(sweptGoal-f.sweptPore[s],sweptGoal,f.sweptPore[s],inventory,p.tolerances),
                    residualRatio(darcyGoal-f.darcyPore[s],darcyGoal,f.darcyPore[s],inventory,p.tolerances),
                    residualRatio(diffusiveGoal-priorDiffusion,diffusiveGoal,priorDiffusion,inventory,p.tolerances)});
                f.sweptPore[s]+=fraction*(sweptGoal-f.sweptPore[s]-aleSensitivity*correction[Nc+s]);
                f.darcyPore[s]+=fraction*(darcyGoal-f.darcyPore[s]
                    -response.velocityDiagonal*correction[Nc+s]-response.fraction[s]*pressureCorrection);
            }
            for(int k=0;k<components;++k){
                const PhysicalReal minDelta=right?-std::max(PhysicalReal(0),(*right)[k]):-std::numeric_limits<PhysicalReal>::max();
                const PhysicalReal delta=std::clamp(fraction*correction[k],minDelta,std::max(PhysicalReal(0),left[k]));
                left[k]-=delta;
                if(right)(*right)[k]+=delta;
                f.transfer[k]+=delta;
            }
            for(int s=0;s<Ns;++s){
                if(!right||(!darcy&&!diffusion))f.sweptPore[s]=f.transfer[Nc+s];
                if(!right||!darcy)f.darcyPore[s]=0;
                else if(!diffusion)f.darcyPore[s]=f.transfer[Nc+s]-f.sweptPore[s];
            }
        }
        return true;
    }

    bool correctEnergy(const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        const std::vector<MaterialCaloric>& caloric,const std::vector<EnergyReal>& conduction,
        std::vector<EnergyReal>& target,EnergyReal& boundaryInput,std::string& error){
        if(faces_.empty())return true;
        struct CellThermal {EnergyReal anchor,minimum,capacity;};
        std::vector<CellThermal> cells(temperature.size());
        for(std::size_t c=0;c<cells.size();++c){
            const auto& law=caloric[c];
            auto& cell=cells[c];
            cell={EnergyReal(law.energy(temperature[c])),EnergyReal(law.energy(law.minimumTemperature)),
                EnergyReal(law.capacity(temperature[c]))};
            if(!finite(cell.anchor)||!finite(cell.minimum)||!(cell.capacity>0)||!finite(cell.capacity)){
                error="inadmissible material state for implicit face energy";return false;
            }
        }
        for(const auto& face:faces_)applyFaceEnergy(face,face.energy,target);
        for(auto& face:faces_){
            const auto flux=faceEnergy<true>(face,p,temperature);
            const auto& left=cells[face.left];
            const EnergyReal leftEnergy=target[face.left]+conduction[face.left];
            const EnergyReal leftResponse=flux.leftSlope/left.capacity;
            EnergyReal defect=flux.value-face.energy+leftResponse*(leftEnergy-left.anchor);
            EnergyReal diagonal=EnergyReal(1)+std::max(EnergyReal(0),leftResponse);
            EnergyReal lower=-std::numeric_limits<EnergyReal>::max();
            if(face.right>=0){
                const auto& right=cells[face.right];
                const EnergyReal rightEnergy=target[face.right]+conduction[face.right];
                const EnergyReal rightResponse=flux.rightSlope/right.capacity;
                defect+=rightResponse*(rightEnergy-right.anchor);
                diagonal+=std::max(EnergyReal(0),-rightResponse);
                lower=-std::max(EnergyReal(0),rightEnergy-right.minimum);
            }
            if(!finite(defect)||!finite(diagonal)){
                error="nonfinite implicit material face energy";return false;
            }
            const EnergyReal upper=std::max(EnergyReal(0),leftEnergy-left.minimum);
            const EnergyReal prior=face.energy;
            face.energy+=std::clamp(defect/diagonal,lower,upper);
            if(!finite(face.energy)){error="nonfinite material face energy transfer";return false;}
            applyFaceEnergy(face,face.energy-prior,target);
            if(face.right<0)boundaryInput-=face.energy;
        }
        return true;
    }

    bool checkEnergy(const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        const std::vector<SolidQ>& state,EnergyReal absolute,EnergyReal relative,EnergyReal& residual,
        std::string& error)const{
        for(const auto& face:faces_){
            const EnergyReal requested=faceEnergy<false>(face,p,temperature).value;
            const EnergyReal difference=requested-face.energy;
            if(!finite(difference)){error="nonfinite implicit transport energy residual";return false;}
            const EnergyReal inventory=face.right>=0
                ?std::max(std::abs(state[face.left].energy),std::abs(state[face.right].energy))
                :std::abs(state[face.left].energy);
            const EnergyReal scale=std::max({EnergyReal(1),std::abs(requested),std::abs(face.energy)});
            const EnergyReal allowed=absolute+relative*scale+EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon()*inventory;
            residual=std::max(residual,std::abs(difference)/std::max(allowed,std::numeric_limits<EnergyReal>::min()));
        }
        return true;
    }

    bool react(const HostMesh& mesh,const PhysicsConfig& p,const std::vector<PhysicalReal>& temperature,
        PhysicalReal dt,PhysicalReal& residual,std::string& error){
        if(!p.enableReactions)return true;
        for(std::size_t i=0;i<masses_.size();++i){
            auto& mass=masses_[i];const PhysicalReal volume=PhysicalReal(mesh.volumes[i]);
            const auto T=PhysicalReal(temperature[i]);
            for(int r=0;r<p.nReactions;++r){
                const auto& reaction=p.reactions[r];PhysicalReal& extent=extents_[i][r];
                const auto q=inventory(mass);
                const PhysicalReal rate=reactionRate<false>(reaction,q.condensed,PhysicalReal(volume),T);
                PhysicalReal derivative=0,lower=-extent,upper=std::numeric_limits<PhysicalReal>::max(),scale=0;
                for(int c=0;c<Nc;++c)derivative+=reaction.condensedNu[c]
                    *reactionMassDerivative(reaction,c,q,PhysicalReal(volume),T,PhysicalReal(rate));
                for(int k=0;k<Nc+Ns;++k){
                    const PhysicalReal nu=k<Nc?reaction.condensedNu[k]:reaction.gasNu[k-Nc];
                    if(nu<0)upper=std::min(upper,std::max(PhysicalReal(0),mass[k])/(-nu));
                    if(nu>0)lower=std::max(lower,-std::max(PhysicalReal(0),mass[k])/nu);
                    if(nu!=0)scale=std::max(scale,std::abs(mass[k]/nu));
                }
                const PhysicalReal desired=dt*rate,defect=desired-extent;
                if(!finite(desired)||rate<0||!finite(derivative)){
                    error="nonfinite implicit material reaction in cell "+std::to_string(i);return false;
                }
                const PhysicalReal admissible=std::clamp(defect,lower,upper);
                residual=std::max(residual,residualRatio(admissible,desired,extent,scale,p.tolerances));
                const PhysicalReal delta=std::clamp(defect/(PhysicalReal(1)+dt*std::max(PhysicalReal(0),-derivative)),lower,upper);
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
