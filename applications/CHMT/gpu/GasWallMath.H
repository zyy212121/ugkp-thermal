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
    // Borrow the existing same-stage sample; no second snapshot or cache.
    const ugkwp::gaswall::WallMatchingState<Real,Ns>* preparedMatching=nullptr;
    const Real* preparedMatchingPressure=nullptr;
    GasPrimitive bulk{}; GasGradient gradient{};
    Real temperature=0,gasDistance=0,area=0,gasArea=0,dt=0,normalSpeed=0,sweptVolume=0;
    Vec3 normal{},velocity{};
    ExchangeKind primaryKind=ExchangeKind::GasSolid;
    Real speciesRate[Ns]{},condensedRate[Nc]{},poreRate[Ns]{},poreSweepRate[Ns]{};
};
struct GasWallResult { ExchangePacket primary{},pore{}; GasPrimitive trace{}; Vec3 traction{}; ugkwp::gaswall::WallOutput<Real,Ns> layer{}; };
CHMT_HD inline bool evaluateGasWall(const GasWallInput& in,const SurfacePacketIdentity& identity,
    const PhysicsConfig& p,GasWallResult& output) {
    const Real ratio=in.area/in.gasArea;
    Real species[Ns]{},primaryJ=0;
    for(int s=0;s<Ns;++s){species[s]=(in.speciesRate[s]+in.poreRate[s])*ratio;primaryJ+=in.speciesRate[s];}
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::WallFunction){
        for(int s=0;s<Ns;++s)if(in.speciesRate[s]!=0||in.poreRate[s]!=0||in.poreSweepRate[s]!=0)return false;
    }
    GasWallResult result;InterfaceGasClosure closure;
    auto same=[](Real a,Real b){return a==b||(a!=a&&b!=b);};
    bool reuse=in.preparedLayer&&p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer;
    if(reuse){
        const auto& layer=*in.preparedLayer;
        // Equality selects reuse; a changed input selects a real new closure.
        // Identically undefined terminal values still belong to this profile.
        reuse=same(layer.traceTemperature,in.temperature);
        if(in.preparedMatchingPressure)reuse=reuse&&same(*in.preparedMatchingPressure,in.wallContext.matchingPressure);
        if(in.preparedMatching){
            const auto& old=*in.preparedMatching;const auto& now=in.wallContext.matching;
            reuse=reuse&&same(old.temperature,now.temperature);
            for(int d=0;d<3;++d)reuse=reuse&&same(old.velocity[d],now.velocity[d]);
            if(p.enableSst)reuse=reuse&&same(old.k,now.k)&&same(old.omega,now.omega);
        }
        Real mass=0;for(int s=0;s<Ns;++s){
            reuse=reuse&&same(layer.wallSpeciesFlux[s],species[s]);
            if(in.preparedMatching)reuse=reuse&&same(in.preparedMatching->massFraction[s],in.wallContext.matching.massFraction[s]);
            mass+=species[s];}
        if(reuse){
            if(!makeInterfacePrimitive<true>(layer.traceMassFraction,in.temperature,in.wallContext.matchingPressure,in.velocity,
                in.normal,in.normalSpeed,mass,p,closure.trace))return false;
            const Vec3 preparedVelocity={layer.traceVelocity[0],layer.traceVelocity[1],layer.traceVelocity[2]};
            if(finite(preparedVelocity)&&finite(closure.trace.velocity))
                reuse=!(mag(preparedVelocity-closure.trace.velocity)>1e-11*maxValue(1,mag(closure.trace.velocity)));
            else{
                // Unordered norm comparisons cannot certify reuse. Matching
                // terminal components still reuse; differences get a real solve.
                reuse=same(preparedVelocity.x,closure.trace.velocity.x)
                    &&same(preparedVelocity.y,closure.trace.velocity.y)
                    &&same(preparedVelocity.z,closure.trace.velocity.z);
                if(reuse){
                    // Fresh closure projects the supplied wall velocity before
                    // constructing its primitive. Finish that same terminal
                    // velocity arithmetic without evaluating the layer again.
                    const Vec3 velocity=in.velocity-in.normal*dot(in.velocity,in.normal)+in.normal*in.normalSpeed;
                    closure.trace.velocity=velocity-in.normal*dot(velocity,in.normal)
                        +in.normal*(in.normalSpeed+mass/closure.trace.rho);
                }
            }
        }
        if(reuse){closure.layer=layer;closure.traction={layer.traction[0],layer.traction[1],layer.traction[2]};
            closure.conductiveHeatFlux=layer.conductiveHeatFlux;}
    }
    if(!reuse){
        bool evaluated=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
            ?evaluateInterfaceGasClosure<true>(in.bulk,in.gradient,in.temperature,in.gasDistance,in.velocity,in.normal,
                in.normalSpeed,species,p.tolerances.absoluteMass/(in.dt*in.gasArea),p,in.wallContext,closure)
            :evaluateInterfaceGasClosure(in.bulk,in.gradient,in.temperature,in.gasDistance,in.velocity,in.normal,
                in.normalSpeed,species,p.tolerances.absoluteMass/(in.dt*in.gasArea),p,in.wallContext,closure);
        if(!evaluated && p.wallModel.family==ugkwp::gaswall::WallFamily::LowRe)
            evaluated=evaluateInterfaceGasClosure<true>(in.bulk,in.gradient,in.temperature,in.gasDistance,in.velocity,in.normal,
                in.normalSpeed,species,p.tolerances.absoluteMass/(in.dt*in.gasArea),p,in.wallContext,closure);
        if(!evaluated)return false;
    }
    result.trace=closure.trace;result.traction=closure.traction;result.layer=closure.layer;
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:result.trace.pressure;
    const Real scale=in.area*in.dt,gasScale=in.gasArea*in.dt;
    result.primary=surfacePacketIdentity(in.primaryKind,identity);
    result.pore=surfacePacketIdentity(identity.solidCell>=0?ExchangeKind::PoreGas:in.primaryKind,identity);
    const Real kinetic=.5*dot(result.trace.velocity,result.trace.velocity);
    for(int s=0;s<Ns;++s){
        const GasThermoAdapter<1> thermo(&p.species[s],
            p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer);
        const Real h=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
            ?ugkwp::gaswall::detail::wallProperty<true>(0,in.temperature,thermo.view())
            :speciesH(p.species[s],in.temperature);
        result.primary.species[s]=scale*in.speciesRate[s];result.primary.mass+=result.primary.species[s];
        result.primary.advective+=result.primary.species[s]*(h+kinetic);
        if(identity.solidCell>=0||in.poreRate[s]!=0||in.poreSweepRate[s]!=0){result.pore.species[s]=result.pore.pore[s]=scale*in.poreRate[s];result.pore.poreSweep[s]=scale*in.poreSweepRate[s];result.pore.mass+=result.pore.species[s];result.pore.advective+=result.pore.species[s]*(h+kinetic);}
    }
    for(int c=0;c<Nc;++c)result.primary.condensed[c]=scale*in.condensedRate[c];
    result.primary.conductive=gasScale*closure.conductiveHeatFlux;
    result.primary.pressureWork=mechanicalPressure*in.sweptVolume;
    result.primary.viscousWork=-gasScale*dot(result.traction,result.trace.velocity);
    result.primary.energy=result.primary.advective+result.primary.conductive+result.primary.pressureWork+result.primary.viscousWork;
    result.primary.momentum=result.trace.velocity*result.primary.mass+(in.normal*mechanicalPressure-result.traction)*gasScale;
    result.pore.energy=result.pore.advective;result.pore.momentum=result.trace.velocity*result.pore.mass;
    if(in.primaryKind==ExchangeKind::GasFilm){
        Vec3 liquid=in.velocity-in.normal*dot(in.velocity,in.normal);
        liquid+=in.normal*(in.normalSpeed+primaryJ*ratio/p.liquid.rho);
        result.primary.liquidKineticAdvection=.5*result.primary.mass*dot(liquid,liquid);
    }
    output=result;return true;
}
} // namespace chmt
#endif
