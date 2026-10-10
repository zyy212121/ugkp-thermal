#ifndef CHMT_GPU_GASWALLMATH_H
#define CHMT_GPU_GASWALLMATH_H
#include "ablation/InterfaceMath.H"
namespace chmt {
// Gross gas withdrawals do not borrow from an opposing transfer in the same
// interval. Pore geometric sweep and Darcy transport are distinct channels.
CHMT_HD inline Real gasPacketSpeciesWithdrawal(const ExchangePacket& packet,int species){
    return packet.kind==ExchangeKind::PoreGas
        ?maxValue(0,-(packet.pore[species]-packet.poreSweep[species]))+maxValue(0,-packet.poreSweep[species])
        :maxValue(0,-packet.species[species]);
}
// Read-only CPU boundary program plus the current gas-side driving state. Rates
// are per reference-surface area. No material state or material RHS is an input.
struct GasWallInput {
    GasPrimitive bulk{}; GasGradient gradient{};
    Real temperature=0,gasDistance=0,area=0,gasArea=0,dt=0,normalSpeed=0,sweptVolume=0;
    Vec3 normal{},velocity{};
    ExchangeKind primaryKind=ExchangeKind::GasSolid;
    Real speciesRate[Ns]{},condensedRate[Nc]{},poreRate[Ns]{},poreSweepRate[Ns]{};
};
struct GasWallResult { ExchangePacket primary{},pore{}; GasPrimitive trace{}; Vec3 traction{}; };
CHMT_HD inline bool evaluateGasWall(const GasWallInput& in,const SurfacePacketIdentity& identity,
    const PhysicsConfig& p,GasWallResult& output) {
    if(!(in.dt>0&&in.area>0&&in.gasArea>0&&in.gasDistance>0&&in.temperature>0)
        || !finite(in.normal)||!finite(in.velocity)||!finite(in.normalSpeed)||!finite(in.sweptVolume)
        || !closeEnough(mag(in.normal),1,1e-12,1e-12)
        || (in.primaryKind!=ExchangeKind::GasSolid&&in.primaryKind!=ExchangeKind::GasFilm))return false;
    const Real ratio=in.area/in.gasArea;
    Real species[Ns]{},Y[Ns]{},J=0,primaryJ=0;
    for(int s=0;s<Ns;++s){species[s]=(in.speciesRate[s]+in.poreRate[s])*ratio;J+=species[s];primaryJ+=in.speciesRate[s];}
    if(!interfaceComposition(species,in.bulk.Y,in.bulk.pressure,in.temperature,in.gasDistance,p,
        p.tolerances.absoluteMass/(in.dt*in.gasArea),Y))return false;
    GasWallResult result;
    if(!makeInterfacePrimitive(Y,in.temperature,in.bulk.pressure,in.velocity,in.normal,in.normalSpeed,J,p,result.trace))return false;
    GasGradient gradient=in.gradient;
    const Vec3 difference=in.bulk.velocity-result.trace.velocity;
    gradient.velocity[0]+=in.normal*(difference.x/in.gasDistance-dot(gradient.velocity[0],in.normal));
    gradient.velocity[1]+=in.normal*(difference.y/in.gasDistance-dot(gradient.velocity[1],in.normal));
    gradient.velocity[2]+=in.normal*(difference.z/in.gasDistance-dot(gradient.velocity[2],in.normal));
    result.traction=viscousTraction(gradient,in.normal,p.gasViscosity,0);
    const Real scale=in.area*in.dt,gasScale=in.gasArea*in.dt;
    result.primary=surfacePacketIdentity(in.primaryKind,identity);
    result.pore=surfacePacketIdentity(identity.solidCell>=0?ExchangeKind::PoreGas:in.primaryKind,identity);
    const Real kinetic=.5*dot(result.trace.velocity,result.trace.velocity);
    for(int s=0;s<Ns;++s){
        result.primary.species[s]=scale*in.speciesRate[s];result.primary.mass+=result.primary.species[s];
        result.primary.advective+=result.primary.species[s]*(speciesH(p.species[s],in.temperature)+kinetic);
        if(identity.solidCell>=0){result.pore.species[s]=result.pore.pore[s]=scale*in.poreRate[s];result.pore.poreSweep[s]=scale*in.poreSweepRate[s];result.pore.mass+=result.pore.species[s];result.pore.advective+=result.pore.species[s]*(speciesH(p.species[s],in.temperature)+kinetic);}
        else if(in.poreRate[s]!=0||in.poreSweepRate[s]!=0)return false;
    }
    for(int c=0;c<Nc;++c)result.primary.condensed[c]=scale*in.condensedRate[c];
    const Real conductance=p.gasConductivity>0?1/(in.gasDistance/p.gasConductivity+p.material.gasContactResistance):0;
    result.primary.conductive=gasScale*conductance*(in.temperature-in.bulk.temperature);
    result.primary.pressureWork=in.bulk.pressure*in.sweptVolume;
    result.primary.viscousWork=-gasScale*dot(result.traction,result.trace.velocity);
    result.primary.energy=result.primary.advective+result.primary.conductive+result.primary.pressureWork+result.primary.viscousWork;
    result.primary.momentum=result.trace.velocity*result.primary.mass+(in.normal*in.bulk.pressure-result.traction)*gasScale;
    result.pore.energy=result.pore.advective;result.pore.momentum=result.trace.velocity*result.pore.mass;
    if(in.primaryKind==ExchangeKind::GasFilm){
        Vec3 liquid=in.velocity-in.normal*dot(in.velocity,in.normal);
        liquid+=in.normal*(in.normalSpeed+primaryJ*ratio/p.liquid.rho);
        result.primary.liquidKineticAdvection=.5*result.primary.mass*dot(liquid,liquid);
    }
    if(!validatePacketMath(result.primary,p.tolerances)||!validatePacketMath(result.pore,p.tolerances))return false;
    output=result;return true;
}
} // namespace chmt
#endif
