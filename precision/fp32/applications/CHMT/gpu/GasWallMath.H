#ifndef CHMT_GPU_GASWALLMATH_H
#define CHMT_GPU_GASWALLMATH_H
#include "ablation/InterfaceMath.H"
namespace chmt {
// Gross gas withdrawals do not borrow from an opposing transfer in the same
// interval. Pore geometric sweep and Darcy transport are distinct channels.
CHMT_HD inline PhysicalReal gasPacketSpeciesWithdrawal(const ExchangePacket& packet,int species){
    return packet.kind==ExchangeKind::PoreGas
        ?energyArithmetic::maximum(0,-(packet.pore[species]-packet.poreSweep[species]))+energyArithmetic::maximum(0,-packet.poreSweep[species])
        :energyArithmetic::maximum(0,-packet.species[species]);
}
// Read-only CPU boundary program plus the current gas-side driving state. Rates
// are per reference-surface area. No material state or material RHS is an input.
struct GasWallInput {
    GasWallClosureContext wallContext{};
    const ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns>* preparedLayer=nullptr;

    const ugkwp::gaswall::WallMatchingState<EnergyReal,Ns>* preparedMatching=nullptr;
    const EnergyReal* preparedMatchingPressure=nullptr;
    GasPrimitive bulk{}; GasGradient gradient{};
    EnergyReal temperature=0;
    PhysicalReal gasDistance=0,area=0,gasArea=0,normalSpeed=0;Real dt=0,sweptVolume=0;
    PhysicalReal exposedGasHeatFraction=1;
    PhysicalVec3 normal{},velocity{};
    ExchangeKind primaryKind=ExchangeKind::GasSolid;
    PhysicalReal speciesRate[Ns]{},condensedRate[Nc]{},poreRate[Ns]{},poreSweepRate[Ns]{};
};
struct GasWallResult { ExchangePacket primary{},pore{}; GasPrimitive trace{}; PhysicalVec3 traction{}; ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns> layer{}; };
CHMT_HD inline bool evaluateGasWall(const GasWallInput& in,const SurfacePacketIdentity& identity,
    const PhysicsConfig& p,GasWallResult& output) {
    const PhysicalReal ratio=in.area/in.gasArea;
    PhysicalReal species[Ns]{};EnergyReal primaryJ=0;
    for(int s=0;s<Ns;++s){species[s]=(in.speciesRate[s]+in.poreRate[s])*ratio;primaryJ+=EnergyReal(in.speciesRate[s]);}
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::WallFunction){
        for(int s=0;s<Ns;++s)if(in.speciesRate[s]!=0||in.poreRate[s]!=0||in.poreSweepRate[s]!=0)return false;
    }
    GasWallResult result;InterfaceGasClosure closure;
    auto same=[](EnergyReal a,EnergyReal b){return a==b||(a!=a&&b!=b);};
    bool reuse=in.preparedLayer&&p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer;
    if(reuse){
        const auto& layer=*in.preparedLayer;


        reuse=same(layer.traceTemperature,in.temperature);
        if(in.preparedMatchingPressure)reuse=reuse&&same(*in.preparedMatchingPressure,in.wallContext.matchingPressure);
        if(in.preparedMatching){
            const auto& old=*in.preparedMatching;const auto& now=in.wallContext.matching;
            reuse=reuse&&same(old.temperature,now.temperature);
            for(int d=0;d<3;++d)reuse=reuse&&same(old.velocity[d],now.velocity[d]);
            if(p.enableSst)reuse=reuse&&same(old.k,now.k)&&same(old.omega,now.omega);
        }
        PhysicalReal mass=0;for(int s=0;s<Ns;++s){
            reuse=reuse&&same(layer.wallSpeciesFlux[s],species[s]);
            if(in.preparedMatching)reuse=reuse&&same(in.preparedMatching->massFraction[s],in.wallContext.matching.massFraction[s]);
            mass+=species[s];}
        if(reuse){
            if(!makeInterfacePrimitive(layer.traceMassFraction,in.temperature,in.wallContext.matchingPressure,in.velocity,
                in.normal,in.normalSpeed,mass,p,closure.trace))return false;
            const PhysicalVec3 preparedVelocity={layer.traceVelocity[0],layer.traceVelocity[1],layer.traceVelocity[2]};
            if(finite(preparedVelocity)&&finite(closure.trace.velocity))
                reuse=!(mag(preparedVelocity-closure.trace.velocity)>energyArithmetic::maximum(EnergyReal(PhysicalReal(1e-11)),EnergyReal(8)*std::numeric_limits<EnergyReal>::epsilon())*energyArithmetic::maximum(1,mag(closure.trace.velocity)));
            else{


                reuse=same(preparedVelocity.x,closure.trace.velocity.x)
                    &&same(preparedVelocity.y,closure.trace.velocity.y)
                    &&same(preparedVelocity.z,closure.trace.velocity.z);
                if(reuse){



                    const PhysicalVec3 velocity=in.velocity-in.normal*dot(in.velocity,in.normal)+in.normal*in.normalSpeed;
                    closure.trace.velocity=velocity-in.normal*dot(velocity,in.normal)
                        +in.normal*(in.normalSpeed+mass/closure.trace.rho);
                }
            }
        }
        if(reuse){closure.layer=layer;closure.traction={layer.traction[0],layer.traction[1],layer.traction[2]};
            closure.conductiveHeatFlux=layer.conductiveHeatFlux;}
    }
    if(!reuse){
        const bool evaluated=evaluateInterfaceGasClosure(in.bulk,in.gradient,in.temperature,in.gasDistance,in.velocity,in.normal,
            in.normalSpeed,species,p.tolerances.absoluteMass/(PhysicalReal(in.dt)*in.gasArea),p,in.wallContext,closure);
        if(!evaluated)return false;
    }
    result.trace=closure.trace;result.traction=closure.traction;result.layer=closure.layer;
    const PhysicalReal mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:result.trace.pressure;
    const PhysicalReal scale=in.area*PhysicalReal(in.dt),gasScale=in.gasArea*PhysicalReal(in.dt);
    const EnergyReal energyScale=EnergyReal(in.area)*EnergyReal(in.dt);
    result.primary=surfacePacketIdentity(in.primaryKind,identity);
    result.pore=surfacePacketIdentity(identity.solidCell>=0?ExchangeKind::PoreGas:in.primaryKind,identity);
    const EnergyReal kinetic=EnergyReal(.5)*surfaceEnergyDot(result.trace.velocity,result.trace.velocity);
    for(int s=0;s<Ns;++s){
        const GasThermoAdapter<1> thermo(&p.species[s]);
        const EnergyReal h=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
            ?ugkwp::gaswall::detail::wallProperty<true>(0,EnergyReal(in.temperature),thermo.view())
            :speciesH(p.species[s],in.temperature);
        result.primary.species[s]=scale*in.speciesRate[s];result.primary.mass+=result.primary.species[s];
        result.primary.advective+=energyScale*EnergyReal(in.speciesRate[s])*(h+kinetic);
        if(identity.solidCell>=0||in.poreRate[s]!=0||in.poreSweepRate[s]!=0){result.pore.species[s]=result.pore.pore[s]=scale*in.poreRate[s];result.pore.poreSweep[s]=scale*in.poreSweepRate[s];result.pore.mass+=result.pore.species[s];result.pore.advective+=energyScale*EnergyReal(in.poreRate[s])*(h+kinetic);}
    }
    for(int c=0;c<Nc;++c)result.primary.condensed[c]=scale*in.condensedRate[c];
    result.primary.conductive=EnergyReal(in.gasArea)*EnergyReal(in.dt)*closure.conductiveHeatFlux*EnergyReal(in.exposedGasHeatFraction);
    result.primary.pressureWork=EnergyReal(mechanicalPressure)*EnergyReal(in.sweptVolume);
    result.primary.viscousWork=-EnergyReal(in.gasArea)*EnergyReal(in.dt)*surfaceEnergyDot(result.traction,result.trace.velocity);
    result.primary.energy=result.primary.advective+result.primary.conductive+result.primary.pressureWork+result.primary.viscousWork;
    result.primary.momentum=result.trace.velocity*result.primary.mass+(in.normal*mechanicalPressure-result.traction)*gasScale;
    result.pore.energy=result.pore.advective;result.pore.momentum=result.trace.velocity*result.pore.mass;
    if(in.primaryKind==ExchangeKind::GasFilm){
        PhysicalVec3 liquid=in.velocity-in.normal*dot(in.velocity,in.normal);
        liquid+=in.normal*(in.normalSpeed+primaryJ*ratio/p.liquid.rho);
        result.primary.liquidKineticAdvection=EnergyReal(.5)*energyScale*EnergyReal(primaryJ)*surfaceEnergyDot(liquid,liquid);
    }
    output=result;return true;
}
} // namespace chmt
#endif
