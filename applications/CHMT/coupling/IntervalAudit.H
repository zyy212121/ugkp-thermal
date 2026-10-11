#ifndef CHMT_COUPLING_INTERVALAUDIT_H
#define CHMT_COUPLING_INTERVALAUDIT_H
#include "coupling/IntervalContracts.H"
#include <set>
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
        if(!validFilmInventory(q)||!finite(a.pressure)||!finite(a.kineticEnergy)||!(p.liquid.rho>0))return intervalFailure(error,"invalid film physical energy audit");
        sum.mass+=q.mass;sum.energy+=static_cast<long double>(q.enthalpy)-static_cast<long double>(a.pressure)*q.mass/p.liquid.rho+a.kineticEnergy;
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
inline Real intervalNormalizedError(Real a,Real b,Real absolute,Real relative){
    const Real difference=absValue(a-b),tolerance=absolute+relative*maxValue(absValue(a),absValue(b));
    if(!finite(a)||!finite(b)||!finite(tolerance)||absolute<0||relative<0)return std::numeric_limits<Real>::infinity();
    return tolerance>0?difference/tolerance:(difference==0?0:std::numeric_limits<Real>::infinity());
}
inline bool intervalBalance(Real residual,long double before,long double after,Real absolute,Real relative){
    const long double tolerance=absolute+relative*std::max(std::fabs(before),std::fabs(after));
    return finite(residual)&&std::isfinite(tolerance)&&tolerance>=0&&std::fabs(static_cast<long double>(residual))<=tolerance;
}
// This audits full endpoint inventory, conserved elements and physical energy,
// with the baseline reduced-film kinetic defect kept explicit. It does not
// claim a full momentum equation for the rigid solid/reduced thin-film model.
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
    // Audit each CPU recipient independently. Global cancellation alone can
    // hide a GasSolid transfer wrongly attributed to FilmParticipant.
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
    // A zero-sum exchange is normalized by the participating transfer, not by
    // its cancellation residue. Per-recipient comparisons above still verify
    // exactly which owner received the amount. This permits only the declared
    // relative tolerance, including cumulative-budget subtraction roundoff.
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
    // Momentum has no dedicated tolerance in the existing schema; use the
    // existing packet mass tolerance consistently, without claiming bulk closure.
    const Vec3 gasMomentum=n.exchangeMomentum[GasParticipant]-b.exchangeMomentum[GasParticipant];
    const Vec3 particleMomentum=n.exchangeMomentum[ParticleParticipant]-b.exchangeMomentum[ParticleParticipant];
    paired=compare(gasMomentum.x,expectedGas.momentum.x,false)&&compare(gasMomentum.y,expectedGas.momentum.y,false)&&compare(gasMomentum.z,expectedGas.momentum.z,false)&&paired;
    paired=compare(particleMomentum.x,expectedParticleMomentum.x,false)&&compare(particleMomentum.y,expectedParticleMomentum.y,false)&&compare(particleMomentum.z,expectedParticleMomentum.z,false)&&paired;
    paired=cancellation(momentumSum.x,momentumScale.x,false)&&cancellation(momentumSum.y,momentumScale.y,false)&&cancellation(momentumSum.z,momentumScale.z,false)&&paired;
    report=audit;if(!paired)return intervalFailure(error,"coupled interval actual exchange ledger mismatch");
    error.clear();return true;
}
// Call after the candidate inventory/budget audit, never to stand in for it.
// Existing CPU-owned SolidFilm records are retained; prior macro records are
// not copied. Bookkeeping only: no inventory or budget source is applied here.
inline bool finalizeIntervalLedger(const IntervalHistory& history,HostState& candidate,std::string& error){
    if(!history.complete())return intervalFailure(error,"cannot finalize incomplete gas history");
    std::vector<ExchangePacket> ledger;
    for(const auto& p:candidate.ledger)if(p.kind==ExchangeKind::SolidFilm){
        if(p.consumerMask!=requiredConsumers(p.kind)||!validatePacketMath(p,Tolerances{}))return intervalFailure(error,"pending CPU internal phase packet");
        ledger.push_back(p);}
    for(const auto& r:history.records())for(auto p:r.packets){p.consumerMask=requiredConsumers(p.kind);ledger.push_back(p);}
    candidate.ledger=std::move(ledger);error.clear();return true;
}
inline bool checkSynchronizedInterval(const HostState& state,const IntervalHistory& history,std::string& error){
    if(!history.complete()||state.time!=history.interval().end)return intervalFailure(error,"only synchronized macro endpoints may be checkpointed");
    for(const auto& p:state.ledger)if(p.consumerMask!=requiredConsumers(p.kind)||!validatePacketMath(p,Tolerances{}))return intervalFailure(error,"checkpoint contains pending interval transfer");
    for(const auto& pv:state.filmStorage)if(!pv.consumed||!finite(pv.oldPV)||!finite(pv.newPV))return intervalFailure(error,"checkpoint contains pending film pressure-volume storage");
    error.clear();return true;
}
inline bool validateMaterialDonorHistory(const HostState& base,const IntervalHistory& history,
    const MaterialDonorPlan* correctedPlan,std::string& error,DonorRoundoffReport* roundoff=nullptr){
    MaterialDonorReserve reserve;
    if(!history.complete()||!reserve.reset(base.solid,base.film,history.interval().begin,error))return false;
    if(correctedPlan&&!correctedPlan->knots.empty()){
        if(correctedPlan->interval.identity.sequence!=history.interval().identity.sequence
            ||correctedPlan->interval.identity.epoch!=history.interval().identity.epoch
            ||correctedPlan->interval.end!=history.interval().end)return intervalFailure(error,"corrected donor plan identity mismatch");
        if(!reserve.setPlan(*correctedPlan,error))return false;
    }
    for(const auto& r:history.records())if(!reserve.apply(r,error))return false;
    return reserve.validatePlanPrefix(error,roundoff);
}
struct WallProgramComparison { Real thermalError=0,massPredictionError=0,geometryError=0;bool regimeMatches=true; };
inline bool intervalSampleWall(const WallProgram& program,Real time,WallKnot& result,std::string& error){
    if(program.knots.size()<2||time<program.interval.begin||time>program.interval.end)return intervalFailure(error,"wall comparison time outside program");
    std::size_t hi=1;while(hi+1<program.knots.size()&&time>program.knots[hi].time)++hi;
    const auto& a=program.knots[hi-1];const auto& b=program.knots[hi];
    if(!finite(a.time)||!finite(b.time)||b.time<=a.time||a.faces.size()!=b.faces.size()
        ||a.gasPoints.size()!=b.gasPoints.size()||a.solidPoints.size()!=b.solidPoints.size())return intervalFailure(error,"invalid wall program interpolation layout");
    const Real f=(time-a.time)/(b.time-a.time);WallKnot out=a;out.time=time;
    for(std::size_t i=0;i<out.faces.size();++i){auto& q=out.faces[i];const auto& t=b.faces[i];
        q.temperature+=f*(t.temperature-q.temperature);q.velocity=q.velocity+(t.velocity-q.velocity)*f;
        q.primaryKind=time==b.time?t.primaryKind:q.primaryKind;
        for(int s=0;s<Ns;++s){q.speciesRate[s]+=f*(t.speciesRate[s]-q.speciesRate[s]);q.poreRate[s]+=f*(t.poreRate[s]-q.poreRate[s]);q.poreSweepRate[s]+=f*(t.poreSweepRate[s]-q.poreSweepRate[s]);}
        for(int c=0;c<Nc;++c)q.condensedRate[c]+=f*(t.condensedRate[c]-q.condensedRate[c]);
    }
    for(std::size_t i=0;i<out.gasPoints.size();++i)out.gasPoints[i]=a.gasPoints[i]+(b.gasPoints[i]-a.gasPoints[i])*f;
    for(std::size_t i=0;i<out.solidPoints.size();++i)out.solidPoints[i]=a.solidPoints[i]+(b.solidPoints[i]-a.solidPoints[i])*f;
    result=std::move(out);error.clear();return true;
}
// absoluteGeometry is the swept-volume tolerance [m^3]. For each
// reference surface face A [m^2], convert it to a displacement tolerance
// absoluteGeometry/A + relativeGeometry*sqrt(A) [m]. The program does not
// carry point-to-interface incidence, so use the strictest face tolerance
// for all points rather than allowing a coarse face to hide a fine-face error.
// This is a point-trajectory comparison only; actual swept-volume/GCL checks
// remain mandatory. No absolute world coordinate enters its scale.
inline bool wallPointLengthTolerance(const WallProgram& first,const WallProgram& second,
    const WallKnot& a,const WallKnot& b,const Tolerances& tol,Real& lengthTolerance,std::string& error){
    if(!finite(tol.absoluteGeometry)||tol.absoluteGeometry<0||!finite(tol.relativeGeometry)||tol.relativeGeometry<0)
        return intervalFailure(error,"invalid wall geometry comparison tolerance");
    Real result=std::numeric_limits<Real>::infinity();
    const std::vector<Real>* areas[]={&first.surface.area,&second.surface.area};
    for(const auto* faceAreas:areas)for(Real area:*faceAreas){
        if(!finite(area)||area<=0)return intervalFailure(error,"nonpositive wall reference face area");
        const Real threshold=tol.absoluteGeometry/area+tol.relativeGeometry*::sqrt(area);
        if(!finite(threshold))return intervalFailure(error,"nonfinite wall displacement tolerance");
        result=minValue(result,threshold);
    }
    if(first.surface.area.empty()&&second.surface.area.empty()){
        // Gas-only programs have no interface face area. Use their intrinsic
        // point-cloud extent; a degenerate cloud has no positive length scale.
        // Reject any nonzero mismatch in that case instead of inventing a unit
        // length or using the distance to the global origin.
        Real length=0;
        const std::vector<Vec3>* clouds[]={&a.gasPoints,&a.solidPoints,&b.gasPoints,&b.solidPoints};
        for(const auto* cloud:clouds)if(!cloud->empty()){
            Vec3 lo=cloud->front(),hi=lo;
            for(const auto& p:*cloud){
                if(!finite(p))return intervalFailure(error,"nonfinite geometry comparison point");
                lo.x=minValue(lo.x,p.x);lo.y=minValue(lo.y,p.y);lo.z=minValue(lo.z,p.z);
                hi.x=maxValue(hi.x,p.x);hi.y=maxValue(hi.y,p.y);hi.z=maxValue(hi.z,p.z);
            }
            length=maxValue(length,mag(hi-lo));
        }
        if(!finite(length))return intervalFailure(error,"nonfinite geometry comparison extent");
        result=length>0?tol.absoluteGeometry/(length*length)+tol.relativeGeometry*length:0;
        if(!finite(result))return intervalFailure(error,"nonfinite intrinsic displacement tolerance");
    }
    lengthTolerance=result;error.clear();return true;
}
inline bool compareWallPrograms(const WallProgram& first,const WallProgram& second,const Tolerances& tol,
    WallProgramComparison& result,std::string& error){
    if(!validCouplingInterval(first.interval)||first.interval.begin!=second.interval.begin||first.interval.end!=second.interval.end)
        return intervalFailure(error,"wall comparison interval mismatch");
    std::set<Real> times;for(const auto& k:first.knots)times.insert(k.time);for(const auto& k:second.knots)times.insert(k.time);
    if(times.empty()||*times.begin()!=first.interval.begin||*times.rbegin()!=first.interval.end)return intervalFailure(error,"wall program does not cover interval");
    WallProgramComparison comparison;
    for(Real time:times){WallKnot a,b;if(!intervalSampleWall(first,time,a,error)||!intervalSampleWall(second,time,b,error))return false;
        if(a.faces.size()!=b.faces.size()||a.gasPoints.size()!=b.gasPoints.size()||a.solidPoints.size()!=b.solidPoints.size()||first.surface.area.size()!=a.faces.size()||second.surface.area.size()!=b.faces.size())return intervalFailure(error,"wall comparison topology/layout mismatch");
        for(std::size_t i=0;i<a.faces.size();++i){const auto& x=a.faces[i];const auto& y=b.faces[i];comparison.regimeMatches=comparison.regimeMatches&&x.primaryKind==y.primaryKind;
            comparison.thermalError=maxValue(comparison.thermalError,intervalNormalizedError(x.temperature,y.temperature,tol.absoluteTemperature,tol.relativeTemperature));
            const Real scale=(first.interval.end-first.interval.begin)*maxValue(first.surface.area[i],second.surface.area[i]);
            for(int s=0;s<Ns;++s){comparison.massPredictionError=maxValue(comparison.massPredictionError,intervalNormalizedError(x.speciesRate[s]*scale,y.speciesRate[s]*scale,tol.absoluteMass,tol.relativeMass));
                comparison.massPredictionError=maxValue(comparison.massPredictionError,intervalNormalizedError(x.poreRate[s]*scale,y.poreRate[s]*scale,tol.absoluteMass,tol.relativeMass));
                comparison.massPredictionError=maxValue(comparison.massPredictionError,intervalNormalizedError(x.poreSweepRate[s]*scale,y.poreSweepRate[s]*scale,tol.absoluteMass,tol.relativeMass));}
            for(int c=0;c<Nc;++c)comparison.massPredictionError=maxValue(comparison.massPredictionError,intervalNormalizedError(x.condensedRate[c]*scale,y.condensedRate[c]*scale,tol.absoluteMass,tol.relativeMass));
        }
        Real pointTolerance=0;
        if(!wallPointLengthTolerance(first,second,a,b,tol,pointTolerance,error))return false;
        auto points=[&](const std::vector<Vec3>& x,const std::vector<Vec3>& y){for(std::size_t i=0;i<x.size();++i){
            const Real distance=mag(x[i]-y[i]);
            const Real normalized=finite(distance)?(pointTolerance>0?distance/pointTolerance
                :(distance==0?0:std::numeric_limits<Real>::infinity())):std::numeric_limits<Real>::infinity();
            comparison.geometryError=maxValue(comparison.geometryError,normalized);}};
        points(a.gasPoints,b.gasPoints);points(a.solidPoints,b.solidPoints);
    }
    result=comparison;error.clear();return true;
}
} // namespace chmt
#endif
