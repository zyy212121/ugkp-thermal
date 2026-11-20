#pragma once
#include "materials/Reaction.H"
namespace chmt {
enum class MaterialRecessionError : int {None,DonorEos,Constituent};


template<class Mass>
CHMT_HD inline MaterialRecessionError addMaterialRecessionTarget(
    const Mass* condensed,const SolidQ& donor,Real volume,const PhysicsConfig& physics,Real& target){
    MaterialPrimitive material;
    if(!recoverMaterial(donor,volume,physics,material))return MaterialRecessionError::DonorEos;
    for(int c=0;c<Nc;++c)if(condensed[c]!=0){
        if(!finite(condensed[c])||!finite(physics.condensed[c].rho)||!(physics.condensed[c].rho>0))
            return MaterialRecessionError::Constituent;
        target-=Real(condensed[c])/Real(physics.condensed[c].rho)/(Real(1)-Real(material.porosity));
    }
    return MaterialRecessionError::None;
}
CHMT_HD inline FilmAux materialRecessionFront(const FilmAux& previous,Real target,Real area,Real dt){
    FilmAux next=previous;
    next.solidNormalVelocity=PhysicalReal(area>0?target/(area*dt):0);
    next.normalVelocity=next.solidNormalVelocity;
    next.solidFront=previous.solidFront+dt*Real(next.solidNormalVelocity);
    next.gasFront=previous.gasFront+dt*Real(next.normalVelocity);
    return next;
}
}
