#pragma once
#include "materials/MaterialTransportMath.H"
#include "materials/MaterialCoolingLimit.H"
#include "materials/ReactionStepControl.H"
#include <limits>

namespace chmt::implicitMaterial {
template<class Number>
CHMT_HD inline Number limited(Number value,Number lower,Number upper){
    return minValue(maxValue(value,lower),upper);
}

static constexpr int components=Nc+Ns+Nr;
struct Amount {
    PhysicalReal value[components]{};
    CHMT_HD PhysicalReal& operator[](int i){return value[i];}
    CHMT_HD PhysicalReal operator[](int i)const{return value[i];}
};
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
CHMT_HD inline SolidQ inventory(const Amount& mass){
    SolidQ q;
    for(int c=0;c<Nc;++c)q.condensed[c]=PhysicalReal(mass[c]);
    for(int s=0;s<Ns;++s)q.pore[s]=PhysicalReal(mass[Nc+s]);
    for(int r=0;r<Nr;++r)q.progress[r]=PhysicalReal(mass[Nc+Ns+r]);
    return q;
}

CHMT_HD inline PhysicalReal residualRatio(PhysicalReal residual,PhysicalReal a,PhysicalReal b,PhysicalReal amount,const Tolerances& tolerance){
    const PhysicalReal scale=maxValue(absValue(a),absValue(b));
    const PhysicalReal roundoff=8*std::numeric_limits<PhysicalReal>::epsilon()*absValue(amount);
    const PhysicalReal allowed=tolerance.absoluteMass+tolerance.relativeMass*scale+roundoff;
    return residual==0?0:absValue(residual)/maxValue(allowed,std::numeric_limits<PhysicalReal>::min());
}
CHMT_HD inline bool poreCorrections(const PhysicsConfig& p,
    const MaterialPrimitive& left,const MaterialPrimitive& right,PhysicalReal leftVolume,PhysicalReal rightVolume,PhysicalReal diffusionLoss,
    const Amount& donor,const Amount& defect,PhysicalReal dt,PhysicalReal aleSensitivity,PhysicalReal spacing,PhysicalReal area,
    bool leftDonor,Amount& correction,PoreResponse& response) {
    const PhysicalReal donorVolume=leftDonor?leftVolume:rightVolume;
    const PhysicalReal donorDensity=leftDonor?left.poreDensity:right.poreDensity;
    const PhysicalReal conductance=p.permeability>0&&spacing>0?p.permeability*area/(p.poreViscosity*spacing):0;
    const PhysicalReal orthogonalAdvection=donorVolume>0?dt*conductance*absValue(PhysicalReal(left.pressure)-right.pressure)/donorVolume:0;
    response.velocityDiagonal=orthogonalAdvection;
    const PhysicalReal baseDiagonal=PhysicalReal(1)+aleSensitivity+orthogonalAdvection;
    const PhysicalReal compositionDiagonal=baseDiagonal+dt*diffusionLoss;
    const PhysicalReal pressureResponse=dt*conductance*donorDensity
        *((leftVolume>0?left.temperature/leftVolume:0)+(rightVolume>0?right.temperature/rightVolume:0));
    if(!(baseDiagonal>0)||!(compositionDiagonal>0)||!finite(compositionDiagonal)
        ||!finite(pressureResponse)||pressureResponse<0)return false;

    PhysicalReal donorMass=0,netDefect=0;
    for(int s=0;s<Ns;++s){
        donorMass+=maxValue(PhysicalReal(0),donor[Nc+s]);
        netDefect+=defect[Nc+s];
    }
    PhysicalReal netDiagonal=baseDiagonal;
    for(int s=0;s<Ns;++s){
        response.fraction[s]=donorMass>0?maxValue(PhysicalReal(0),donor[Nc+s])/donorMass:PhysicalReal(1)/Ns;
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
CHMT_HD inline FaceEnergy faceEnergy(const Face& face,const PhysicsConfig& p,const PhysicalReal* temperature){
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

CHMT_HD inline void applyFaceEnergy(const Face& face,EnergyReal flux,EnergyReal* target){
    target[face.left]-=flux;
    if(face.right>=0)target[face.right]+=flux;
}

struct CellThermal {EnergyReal anchor,minimum,capacity;};

CHMT_HD inline const char* prepareCell(Amount mass,const Amount& initial,PhysicalReal temperature,
    PhysicalReal volume,const PhysicsConfig& p,SolidQ& q,MaterialCaloric& caloric,MaterialPrimitive& primitive){
    for(int k=0;k<components;++k){
        if(!finite(mass[k])||mass[k]<-maxValue(p.tolerances.absoluteMass,
            PhysicalReal(32)*std::numeric_limits<PhysicalReal>::epsilon()*absValue(initial[k]))){
            return "invalid implicit inventory";
        }
        if(mass[k]<0)mass[k]=0;
    }
    q=inventory(mass);
    bool numerical=true;
    if(makeMaterialCaloric(q,p,caloric,numerical)!=MaterialCaloricCode::Success)
        return "invalid implicit material caloric inventory";
    if(!finite(temperature)||temperature<=0){
        return "invalid coupled material temperature";
    }
    q.energy=EnergyReal(caloric.energy(PhysicalReal(temperature)));
    if(!recoverMaterial(q,volume,p,primitive)){
        return "inadmissible implicit material constitutive state";
    }
    q.porosity=primitive.porosity;
    return nullptr;
}

CHMT_HD inline const char* updateMass(Face& f,Amount& left,Amount* right,const Real* volumes,
    const PhysicsConfig& p,const MaterialPrimitive* primitive,const SolidQ& flux,const MaterialFaceTransport& split,
    PhysicalReal diffusionLoss,PhysicalReal spacing,PhysicalReal area,PhysicalReal ale,PhysicalReal dt,
    bool darcy,bool diffusion,PhysicalReal& residual){
    f.sweptFromLeft=ale<=0||!right;
    const int sweptDonor=f.sweptFromLeft?f.left:f.right;
    const PhysicalReal aleSensitivity=absValue(ale)/PhysicalReal(volumes[sweptDonor]);
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
            return "nonfinite implicit material mass flux";
        }
        residual=maxValue(residual,residualRatio(defect[k],desired,f.transfer[k],
            right?maxValue(left[k],(*right)[k]):left[k],p.tolerances));
    }
    PoreResponse response;
    if(right){
        const auto& a=primitive[f.left];const auto& b=primitive[f.right];
        const bool leftDonor=darcy?f.darcyFromLeft:f.sweptFromLeft;
        if(!poreCorrections(p,a,b,a.porosity*PhysicalReal(volumes[f.left]),b.porosity*PhysicalReal(volumes[f.right]),
            diffusionLoss,leftDonor?left:*right,defect,dt,aleSensitivity,spacing,area,leftDonor,correction,response))
            return "nonfinite implicit pore mass preconditioner";
    }
    PhysicalReal fraction=1;
    for(int k=0;k<components;++k){
        const PhysicalReal delta=correction[k];
        if(!finite(delta)){return "nonfinite implicit mass correction";}
        const PhysicalReal leftAvailable=maxValue(PhysicalReal(0),left[k]);
        if(delta>leftAvailable)fraction=minValue(fraction,leftAvailable/delta);
        if(right){
            const PhysicalReal rightAvailable=maxValue(PhysicalReal(0),(*right)[k]);
            if(-delta>rightAvailable)fraction=minValue(fraction,rightAvailable/(-delta));
        }
    }
    PhysicalReal pressureCorrection=0;
    for(int s=0;s<Ns;++s)pressureCorrection+=response.pressure[s]*correction[Nc+s];
    for(int s=0;s<Ns;++s){
        const PhysicalReal sweptGoal=dt*split.sweptPore[s],darcyGoal=dt*split.darcyPore[s];
        const auto inventory=right?maxValue(left[Nc+s],(*right)[Nc+s]):left[Nc+s];
        const PhysicalReal priorDiffusion=f.transfer[Nc+s]-f.sweptPore[s]-f.darcyPore[s];
        const PhysicalReal diffusiveGoal=dt*split.diffusivePore[s];
        residual=maxValue(residual,residualRatio(sweptGoal-f.sweptPore[s],sweptGoal,f.sweptPore[s],inventory,p.tolerances));
        residual=maxValue(residual,residualRatio(darcyGoal-f.darcyPore[s],darcyGoal,f.darcyPore[s],inventory,p.tolerances));
        residual=maxValue(residual,residualRatio(diffusiveGoal-priorDiffusion,diffusiveGoal,priorDiffusion,inventory,p.tolerances));
        f.sweptPore[s]+=fraction*(sweptGoal-f.sweptPore[s]-aleSensitivity*correction[Nc+s]);
        f.darcyPore[s]+=fraction*(darcyGoal-f.darcyPore[s]
            -response.velocityDiagonal*correction[Nc+s]-response.fraction[s]*pressureCorrection);
    }
    for(int k=0;k<components;++k){
        const PhysicalReal minDelta=right?-maxValue(PhysicalReal(0),(*right)[k]):-std::numeric_limits<PhysicalReal>::max();
        const PhysicalReal delta=limited(fraction*correction[k],minDelta,maxValue(PhysicalReal(0),left[k]));
        left[k]-=delta;
        if(right)(*right)[k]+=delta;
        f.transfer[k]+=delta;
    }
    for(int s=0;s<Ns;++s){
        if(!right||(!darcy&&!diffusion))f.sweptPore[s]=f.transfer[Nc+s];
        if(!right||!darcy)f.darcyPore[s]=0;
        else if(!diffusion)f.darcyPore[s]=f.transfer[Nc+s]-f.sweptPore[s];
    }
    return nullptr;
}

CHMT_HD inline bool prepareThermalCell(const MaterialCaloric& law,PhysicalReal temperature,CellThermal& cell,
    bool coolingLimit=true){
    cell.anchor=law.energy(temperature);cell.capacity=law.capacity(temperature);
    if(!finite(cell.anchor)||!(cell.capacity>0)||!finite(cell.capacity))return false;
    cell.minimum=cell.anchor;
    if(coolingLimit){
        const EnergyReal limit=lowestCoolingEnergy(law,EnergyReal(temperature),cell.capacity);
        if(!finite(limit))return false;
        const EnergyReal reserve=EnergyReal(4)*std::numeric_limits<EnergyReal>::epsilon()
            *maxValue(absValue(cell.anchor),absValue(limit));
        cell.minimum=minValue(cell.anchor,limit+reserve);
    }
    return true;
}

CHMT_HD inline const char* updateEnergy(Face& face,const PhysicsConfig& p,const PhysicalReal* temperature,
    const CellThermal* cells,const EnergyReal* conduction,EnergyReal* target){
    const auto flux=faceEnergy<true>(face,p,temperature);
    const auto& left=cells[face.left];
    const EnergyReal leftEnergy=target[face.left]+conduction[face.left];
    const EnergyReal leftResponse=flux.leftSlope/left.capacity;
    EnergyReal defect=flux.value-face.energy+leftResponse*(leftEnergy-left.anchor);
    EnergyReal diagonal=EnergyReal(1)+maxValue(EnergyReal(0),leftResponse);
    EnergyReal lower=-std::numeric_limits<EnergyReal>::max();
    if(face.right>=0){
        const auto& right=cells[face.right];
        const EnergyReal rightEnergy=target[face.right]+conduction[face.right];
        const EnergyReal rightResponse=flux.rightSlope/right.capacity;
        defect+=rightResponse*(rightEnergy-right.anchor);
        diagonal+=maxValue(EnergyReal(0),-rightResponse);
        lower=-maxValue(EnergyReal(0),rightEnergy-right.minimum);
    }
    if(!finite(defect)||!finite(diagonal)){
        return "nonfinite implicit material face energy";
    }
    const EnergyReal upper=maxValue(EnergyReal(0),leftEnergy-left.minimum);
    const EnergyReal prior=face.energy;
    face.energy+=limited(defect/diagonal,lower,upper);
    if(!finite(face.energy))return "nonfinite material face energy transfer";
    applyFaceEnergy(face,face.energy-prior,target);
    return nullptr;
}

CHMT_HD inline bool checkFaceEnergy(const Face& face,const PhysicsConfig& p,const PhysicalReal* temperature,
    const SolidQ* state,EnergyReal absolute,EnergyReal relative,EnergyReal& residual){
    const EnergyReal requested=faceEnergy<false>(face,p,temperature).value;
    const EnergyReal difference=requested-face.energy;
    if(!finite(difference))return false;
    const EnergyReal inventory=face.right>=0
        ?maxValue(absValue(state[face.left].energy),absValue(state[face.right].energy))
        :absValue(state[face.left].energy);
    const EnergyReal scale=maxValue(EnergyReal(1),maxValue(absValue(requested),absValue(face.energy)));
    const EnergyReal allowed=absolute+relative*scale+EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon()*inventory;
    residual=maxValue(residual,absValue(difference)/maxValue(allowed,std::numeric_limits<EnergyReal>::min()));
    return true;
}

CHMT_HD inline void applyExtent(Amount& mass,const Reaction& reaction,int r,PhysicalReal extent){
    for(int c=0;c<Nc;++c)mass[c]+=extent*reaction.condensedNu[c];
    for(int s=0;s<Ns;++s)mass[Nc+s]+=extent*reaction.gasNu[s];
    mass[Nc+Ns+r]+=extent;
}

CHMT_HD inline bool updateReactions(Amount& mass,PhysicalReal* extents,PhysicalReal volume,PhysicalReal T,
    const PhysicsConfig& p,PhysicalReal dt,PhysicalReal& residual){
    for(int r=0;r<p.nReactions;++r){
        const auto& reaction=p.reactions[r];PhysicalReal& extent=extents[r];
        const auto q=inventory(mass);
        const PhysicalReal rate=reactionRate<false>(reaction,q.condensed,PhysicalReal(volume),T);
        PhysicalReal derivative=0,lower=-extent,upper=std::numeric_limits<PhysicalReal>::max(),scale=0;
        for(int c=0;c<Nc;++c)derivative+=reaction.condensedNu[c]
            *reactionMassDerivative(reaction,c,q,PhysicalReal(volume),T,PhysicalReal(rate));
        for(int k=0;k<Nc+Ns;++k){
            const PhysicalReal nu=k<Nc?reaction.condensedNu[k]:reaction.gasNu[k-Nc];
            if(nu<0)upper=minValue(upper,maxValue(PhysicalReal(0),mass[k])/(-nu));
            if(nu>0)lower=maxValue(lower,-maxValue(PhysicalReal(0),mass[k])/nu);
            if(nu!=0)scale=maxValue(scale,absValue(mass[k]/nu));
        }
        const PhysicalReal desired=dt*rate,defect=desired-extent;
        if(!finite(desired)||rate<0||!finite(derivative)){
            return false;
        }
        const PhysicalReal admissible=limited(defect,lower,upper);
        residual=maxValue(residual,residualRatio(admissible,desired,extent,scale,p.tolerances));
        const PhysicalReal delta=limited(defect/(PhysicalReal(1)+dt*maxValue(PhysicalReal(0),-derivative)),lower,upper);
        extent+=delta;
        applyExtent(mass,reaction,r,delta);
    }
    return true;
}
}
