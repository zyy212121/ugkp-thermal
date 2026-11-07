#pragma once
#include "materials/Reaction.H"
namespace chmt {
enum class MaterialRecessionError : int {None,DonorEos,Constituent};
// Recession is fixed by accepted condensed transfer, never by the next wall
// constitutive proposal. Keep constituent order and the original EOS call.
CHMT_HD inline MaterialRecessionError addMaterialRecessionTarget(
    const Real* condensed,const SolidQ& donor,Real volume,const PhysicsConfig& physics,Real& target){
    MaterialPrimitive material;
    if(!recoverMaterial(donor,volume,physics,material))return MaterialRecessionError::DonorEos;
    for(int c=0;c<Nc;++c)if(condensed[c]!=0){
        if(!finite(condensed[c])||!finite(physics.condensed[c].rho)||!(physics.condensed[c].rho>0))
            return MaterialRecessionError::Constituent;
        target-=condensed[c]/physics.condensed[c].rho/(1-material.porosity);
    }
    return MaterialRecessionError::None;
}
CHMT_HD inline FilmAux materialRecessionFront(const FilmAux& previous,Real target,Real area,Real dt){
    FilmAux next=previous;
    next.solidNormalVelocity=area>0?target/(area*dt):0;
    next.normalVelocity=next.solidNormalVelocity;
    next.solidFront=previous.solidFront+dt*next.solidNormalVelocity;
    next.gasFront=previous.gasFront+dt*next.normalVelocity;
    return next;
}
}
