#pragma once

#include "coupling/IntervalAudit.H"
#include "verification/FilmInventory.H"
namespace chmt {
struct WindowConservationAudit {
    Real massResidual=0,energyResidual=0,elementResidual[Ne]{};
    Real maximumExchangeResidual=0;
};
struct IntervalInventoryTotals {
    long double mass=0,energy=0,elements[Ne]{};
};
inline bool intervalInventoryTotals(const HostState& state,const PhysicsConfig& p,
    IntervalInventoryTotals& result,std::string& error){
    IntervalInventoryTotals sum;
    if(p.nElements<0||p.nElements>Ne)return intervalFailure(error,"invalid element count in interval audit");
    for(const auto& q:state.gas){
        if(!validGasInventory(q))return intervalFailure(error,"invalid gas inventory in interval audit");
        sum.mass+=q.mass;sum.energy+=q.energy;
        for(int s=0;s<Ns;++s)for(int e=0;e<p.nElements;++e)sum.elements[e]+=static_cast<long double>(q.species[s])*p.species[s].element[e];
    }
    for(const auto& q:state.solid){
        if(!validSolidInventory(q))return intervalFailure(error,"invalid material inventory in interval audit");
        sum.energy+=q.energy;
        for(int c=0;c<Nc;++c){sum.mass+=q.condensed[c];for(int e=0;e<p.nElements;++e)sum.elements[e]+=static_cast<long double>(q.condensed[c])*p.condensed[c].element[e];}
        for(int s=0;s<Ns;++s){sum.mass+=q.pore[s];for(int e=0;e<p.nElements;++e)sum.elements[e]+=static_cast<long double>(q.pore[s])*p.species[s].element[e];}
    }
    if(state.film.size()!=state.filmAux.size()&&!state.film.empty())return intervalFailure(error,"film audit auxiliary count mismatch");
    for(std::size_t f=0;f<state.film.size();++f){const auto& q=state.film[f];const auto& a=state.filmAux[f];
        const long double kinetic=verification::filmKineticEnergy(q,a);
        if(!validFilmInventory(q)||!finite(a.pressure)||!std::isfinite(kinetic)||!(p.liquid.rho>0))return intervalFailure(error,"invalid film physical energy audit");
        sum.mass+=q.mass;sum.energy+=static_cast<long double>(q.enthalpy)-static_cast<long double>(a.pressure)*q.mass/p.liquid.rho+kinetic;
        for(int s=0;s<Ns;++s)for(int e=0;e<p.nElements;++e)sum.elements[e]+=static_cast<long double>(q.species[s])*p.species[s].element[e];
    }
    if(!state.normalEnthalpy.empty())return intervalFailure(error,"multirate volume/surface audit excludes separate normal-column state");
    for(const auto& q:state.particles){
        if(!finite(q.mass)||q.mass<0||!finite(q.energy)||!finite(q.contactMechanicalEnergy)||!finite(q.velocity))return intervalFailure(error,"invalid particle interval inventory");
        sum.mass+=q.mass;sum.energy+=static_cast<long double>(q.energy)+q.contactMechanicalEnergy+.5L*q.mass*dot(q.velocity,q.velocity);
        if(q.mass>0){if(p.particle.condensedIndex<0||p.particle.condensedIndex>=Nc)return intervalFailure(error,"particle material identity missing in interval audit");
            for(int e=0;e<p.nElements;++e)sum.elements[e]+=static_cast<long double>(q.mass)*p.condensed[p.particle.condensedIndex].element[e];}
    }
    if(!std::isfinite(sum.mass)||!std::isfinite(sum.energy))return intervalFailure(error,"nonfinite interval inventory total");
    for(int e=0;e<p.nElements;++e)if(!std::isfinite(sum.elements[e]))return intervalFailure(error,"nonfinite interval element total");
    result=sum;error.clear();return true;
}
inline bool intervalBalance(Real residual,long double before,long double after,Real absolute,Real relative){
    const long double tolerance=absolute+relative*std::max(std::fabs(before),std::fabs(after));
    return finite(residual)&&std::isfinite(tolerance)&&tolerance>=0&&std::fabs(static_cast<long double>(residual))<=tolerance;
}



inline bool auditCoupledCandidate(const HostState& base,const HostState& candidate,
    const IntervalHistory& history,const PhysicsConfig& p,WindowConservationAudit& report,std::string& error){
    if(!history.complete()||base.time!=history.interval().begin||candidate.time!=history.interval().end)
        return intervalFailure(error,"coupled audit requires complete synchronized interval endpoints");
    IntervalInventoryTotals old,next;
    if(!intervalInventoryTotals(base,p,old,error)||!intervalInventoryTotals(candidate,p,next,error))return false;
    WindowConservationAudit audit;
    const auto& b=base.budget;const auto& n=candidate.budget;
    audit.massResidual=static_cast<Real>(next.mass-old.mass+(n.boundaryMass-b.boundaryMass));
    const long double external=-(n.boundaryEnergy-b.boundaryEnergy)+(n.bodyWork-b.bodyWork)
        +(n.supportWork-b.supportWork)+(n.radiation-b.radiation);
    audit.energyResidual=static_cast<Real>(next.energy-old.energy-external-(n.filmKineticDefect-b.filmKineticDefect));
    for(int e=0;e<p.nElements;++e)audit.elementResidual[e]=static_cast<Real>(next.elements[e]-old.elements[e]+n.boundaryElements[e]-b.boundaryElements[e]);
    report=audit;
    if(!intervalBalance(audit.massResidual,old.mass,next.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))return intervalFailure(error,"coupled interval total mass residual");
    if(!intervalBalance(audit.energyResidual,old.energy,next.energy,p.tolerances.absoluteEnergy,p.tolerances.relativeEnergy))return intervalFailure(error,"coupled interval physical energy residual");
    for(int e=0;e<p.nElements;++e)if(!intervalBalance(audit.elementResidual[e],old.elements[e],next.elements[e],p.tolerances.absoluteMass,p.tolerances.relativeMass))return intervalFailure(error,"coupled interval conserved element residual");
    GasQ expectedGas;Real expectedParticleMass=0,expectedParticleEnergy=0;Vec3 expectedParticleMomentum{};Real expectedParticleSpecies[Ns]{};
    for(const auto& r:history.records())for(const auto& packet:r.packets){
        const auto d=packetDelta(packet);const unsigned mask=requiredConsumers(packet.kind);
        if(mask&ConsumeGas)expectedGas+=d.gas;
        if(mask&ConsumeParticle){expectedParticleMass+=d.particleMass;expectedParticleEnergy+=d.particleEnergy;expectedParticleMomentum+=d.particleMomentum;
            for(int s=0;s<Ns;++s)expectedParticleSpecies[s]-=packet.species[s];}
    }
    auto compare=[&](Real actual,Real expected,bool energy)->bool{
        const Real normalized=intervalNormalizedError(actual,expected,energy?p.tolerances.absoluteEnergy:p.tolerances.absoluteMass,energy?p.tolerances.relativeEnergy:p.tolerances.relativeMass);
        audit.maximumExchangeResidual=maxValue(audit.maximumExchangeResidual,normalized);return normalized<=1;
    };
    bool paired=compare(n.exchangeMass[GasParticipant]-b.exchangeMass[GasParticipant],expectedGas.mass,false)
        &&compare(n.exchangeEnergy[GasParticipant]-b.exchangeEnergy[GasParticipant],expectedGas.energy,true)
        &&compare(n.exchangeMass[ParticleParticipant]-b.exchangeMass[ParticleParticipant],expectedParticleMass,false)
        &&compare(n.exchangeEnergy[ParticleParticipant]-b.exchangeEnergy[ParticleParticipant],expectedParticleEnergy,true);


    Budget expectedCpu;
    auto addCpu=[&](const ExchangePacket& packet){
        const auto d=packetDelta(packet);const unsigned mask=requiredConsumers(packet.kind);
        if(mask&ConsumeSolid){const Real sign=packet.kind==ExchangeKind::ParticleWall?1:-1;
            expectedCpu.exchangeMass[SolidParticipant]+=sign*packet.mass;
            expectedCpu.exchangeEnergy[SolidParticipant]+=d.solid.energy;
            expectedCpu.exchangeMomentum[SolidParticipant]+=packet.momentum*sign;
            for(int s=0;s<Ns;++s)expectedCpu.exchangeSpecies[SolidParticipant][s]+=sign*packet.species[s];}
        if(mask&ConsumeFilm){const Real sign=packet.kind==ExchangeKind::SolidFilm?1:-1;
            expectedCpu.exchangeMass[FilmParticipant]+=d.film.mass;
            expectedCpu.exchangeEnergy[FilmParticipant]+=sign*packet.energy;
            expectedCpu.exchangeMomentum[FilmParticipant]+=packet.momentum*sign;
            for(int s=0;s<Ns;++s)expectedCpu.exchangeSpecies[FilmParticipant][s]+=d.film.species[s];}
    };
    for(const auto& r:history.records())for(const auto& packet:r.packets)addCpu(packet);
    for(const auto& packet:candidate.ledger)if(packet.kind==ExchangeKind::SolidFilm){
        if(packet.consumerMask!=requiredConsumers(packet.kind)||!validatePacketMath(packet,p.tolerances))
            return intervalFailure(error,"CPU internal phase record is invalid or pending");
        addCpu(packet);
    }
    for(int k=SolidParticipant;k<=FilmParticipant;++k){
        paired=compare(n.exchangeMass[k]-b.exchangeMass[k],expectedCpu.exchangeMass[k],false)&&paired;
        paired=compare(n.exchangeEnergy[k]-b.exchangeEnergy[k],expectedCpu.exchangeEnergy[k],true)&&paired;
        for(int s=0;s<Ns;++s)paired=compare(n.exchangeSpecies[k][s]-b.exchangeSpecies[k][s],expectedCpu.exchangeSpecies[k][s],false)&&paired;
        const Vec3 actual=n.exchangeMomentum[k]-b.exchangeMomentum[k],expected=expectedCpu.exchangeMomentum[k];
        paired=compare(actual.x,expected.x,false)&&compare(actual.y,expected.y,false)&&compare(actual.z,expected.z,false)&&paired;
    }
    Real massSum=0,energySum=0,speciesSum[Ns]{},massScale=0,energyScale=0,speciesScale[Ns]{};
    Vec3 momentumSum{},momentumScale{};
    for(int k=0;k<NParticipants;++k){
        const Real mass=n.exchangeMass[k]-b.exchangeMass[k],energy=n.exchangeEnergy[k]-b.exchangeEnergy[k];
        const Vec3 momentum=n.exchangeMomentum[k]-b.exchangeMomentum[k];
        massSum+=mass;energySum+=energy;momentumSum+=momentum;
        massScale+=absValue(mass);energyScale+=absValue(energy);
        momentumScale.x+=absValue(momentum.x);momentumScale.y+=absValue(momentum.y);momentumScale.z+=absValue(momentum.z);
        for(int s=0;s<Ns;++s){const Real species=n.exchangeSpecies[k][s]-b.exchangeSpecies[k][s];speciesSum[s]+=species;speciesScale[s]+=absValue(species);}
    }




    auto cancellation=[&](Real residual,Real scale,bool energy)->bool{
        const Real tolerance=(energy?p.tolerances.absoluteEnergy:p.tolerances.absoluteMass)
            +(energy?p.tolerances.relativeEnergy:p.tolerances.relativeMass)*scale;
        const Real normalized=finite(residual)&&finite(tolerance)&&tolerance>=0
            ?(tolerance>0?absValue(residual)/tolerance:(residual==0?0:std::numeric_limits<Real>::infinity()))
            :std::numeric_limits<Real>::infinity();
        audit.maximumExchangeResidual=maxValue(audit.maximumExchangeResidual,normalized);return normalized<=1;
    };
    paired=cancellation(massSum,massScale,false)&&cancellation(energySum,energyScale,true)&&paired;
    for(int s=0;s<Ns;++s){paired=compare(n.exchangeSpecies[GasParticipant][s]-b.exchangeSpecies[GasParticipant][s],expectedGas.species[s],false)&&paired;
        paired=compare(n.exchangeSpecies[ParticleParticipant][s]-b.exchangeSpecies[ParticleParticipant][s],expectedParticleSpecies[s],false)&&cancellation(speciesSum[s],speciesScale[s],false)&&paired;}


    const Vec3 gasMomentum=n.exchangeMomentum[GasParticipant]-b.exchangeMomentum[GasParticipant];
    const Vec3 particleMomentum=n.exchangeMomentum[ParticleParticipant]-b.exchangeMomentum[ParticleParticipant];
    paired=compare(gasMomentum.x,expectedGas.momentum.x,false)&&compare(gasMomentum.y,expectedGas.momentum.y,false)&&compare(gasMomentum.z,expectedGas.momentum.z,false)&&paired;
    paired=compare(particleMomentum.x,expectedParticleMomentum.x,false)&&compare(particleMomentum.y,expectedParticleMomentum.y,false)&&compare(particleMomentum.z,expectedParticleMomentum.z,false)&&paired;
    paired=cancellation(momentumSum.x,momentumScale.x,false)&&cancellation(momentumSum.y,momentumScale.y,false)&&cancellation(momentumSum.z,momentumScale.z,false)&&paired;
    report=audit;if(!paired)return intervalFailure(error,"coupled interval actual exchange ledger mismatch");
    error.clear();return true;
}
}
