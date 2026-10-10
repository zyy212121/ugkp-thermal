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
    GasWallClosureContext wallContext{};
    const ugkwp::gaswall::WallOutput<Real,Ns>* preparedLayer=nullptr;
    GasPrimitive bulk{}; GasGradient gradient{};
    Real temperature=0,gasDistance=0,area=0,gasArea=0,dt=0,normalSpeed=0,sweptVolume=0;
    Vec3 normal{},velocity{};
    ExchangeKind primaryKind=ExchangeKind::GasSolid;
    Real speciesRate[Ns]{},condensedRate[Nc]{},poreRate[Ns]{},poreSweepRate[Ns]{};
};
struct GasWallResult { ExchangePacket primary{},pore{}; GasPrimitive trace{}; Vec3 traction{}; ugkwp::gaswall::WallOutput<Real,Ns> layer{}; };
CHMT_HD inline bool evaluateGasWall(const GasWallInput& in,const SurfacePacketIdentity& identity,
    const PhysicsConfig& p,GasWallResult& output) {
    if(!(in.dt>0&&in.area>0&&in.gasArea>0&&in.gasDistance>0&&in.temperature>0)
        || !finite(in.normal)||!finite(in.velocity)||!finite(in.normalSpeed)||!finite(in.sweptVolume)
        || !closeEnough(mag(in.normal),1,1e-12,1e-12)
        || (in.primaryKind!=ExchangeKind::GasSolid&&in.primaryKind!=ExchangeKind::GasFilm))return false;
    const Real ratio=in.area/in.gasArea;
    Real species[Ns]{},primaryJ=0;
    for(int s=0;s<Ns;++s){species[s]=(in.speciesRate[s]+in.poreRate[s])*ratio;primaryJ+=in.speciesRate[s];}
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::WallFunction){
        for(int s=0;s<Ns;++s)if(in.speciesRate[s]!=0||in.poreRate[s]!=0||in.poreSweepRate[s]!=0)return false;
    }
    GasWallResult result;InterfaceGasClosure closure;
    if(in.preparedLayer){
        if(p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer||p.material.gasContactResistance!=0)return false;
        const auto& layer=*in.preparedLayer;
        if(layer.traceTemperature!=in.temperature)return false;
        Real mass=0;for(int s=0;s<Ns;++s){if(layer.wallSpeciesFlux[s]!=species[s])return false;mass+=species[s];}
        if(!makeInterfacePrimitive(layer.traceMassFraction,in.temperature,in.wallContext.matchingPressure,in.velocity,
            in.normal,in.normalSpeed,mass,p,closure.trace))return false;
        const Vec3 preparedVelocity={layer.traceVelocity[0],layer.traceVelocity[1],layer.traceVelocity[2]};
        if(mag(preparedVelocity-closure.trace.velocity)>1e-11*maxValue(1,mag(closure.trace.velocity)))return false;
        closure.layer=layer;closure.traction={layer.traction[0],layer.traction[1],layer.traction[2]};
        closure.conductiveHeatFlux=layer.conductiveHeatFlux;
    }else if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,in.temperature,in.gasDistance,in.velocity,in.normal,
        in.normalSpeed,species,p.tolerances.absoluteMass/(in.dt*in.gasArea),p,in.wallContext,closure))return false;
    result.trace=closure.trace;result.traction=closure.traction;result.layer=closure.layer;
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
    result.primary.conductive=gasScale*closure.conductiveHeatFlux;
    result.primary.pressureWork=result.trace.pressure*in.sweptVolume;
    result.primary.viscousWork=-gasScale*dot(result.traction,result.trace.velocity);
    result.primary.energy=result.primary.advective+result.primary.conductive+result.primary.pressureWork+result.primary.viscousWork;
    result.primary.momentum=result.trace.velocity*result.primary.mass+(in.normal*result.trace.pressure-result.traction)*gasScale;
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
