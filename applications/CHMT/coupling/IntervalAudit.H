#ifndef CHMT_COUPLING_INTERVALAUDIT_H
#define CHMT_COUPLING_INTERVALAUDIT_H
#include "coupling/IntervalContracts.H"
#include <set>
namespace chmt {
inline Real intervalNormalizedError(Real a,Real b,Real absolute,Real relative){
    const Real difference=absValue(a-b),tolerance=absolute+relative*maxValue(absValue(a),absValue(b));
    if(!finite(a)||!finite(b)||!finite(tolerance)||absolute<0||relative<0)return std::numeric_limits<Real>::infinity();
    return tolerance>0?difference/tolerance:(difference==0?0:std::numeric_limits<Real>::infinity());
}

// Existing CPU-owned SolidFilm records are retained; prior macro records are
// not copied. Bookkeeping only: no inventory or budget source is applied here.
inline bool finalizeIntervalLedger(const IntervalHistory& history,HostState& candidate,std::string& error){
    std::vector<ExchangePacket> ledger;
    for(const auto& p:candidate.ledger)if(p.kind==ExchangeKind::SolidFilm){
        ledger.push_back(p);}
    for(const auto& r:history.records())for(auto p:r.packets){p.consumerMask=requiredConsumers(p.kind);ledger.push_back(p);}
    candidate.ledger=std::move(ledger);error.clear();return true;
}
struct WallProgramComparison { Real thermalError=0,massPredictionError=0,geometryError=0;bool regimeMatches=true; };
inline bool intervalSampleWallLocal(const WallProgram& program,WindowTime localTime,WallKnot& result,std::string& error){
    const Real time=windowAbsolute(program.interval.begin,localTime);
    if(program.knots.size()<2)return intervalFailure(error,"wall comparison time outside program");
    std::size_t hi=1;while(hi+1<program.knots.size()&&localTime>knotOffset(program.interval,program.knots[hi]))++hi;
    const auto& a=program.knots[hi-1];const auto& b=program.knots[hi];
    const Real f=windowFraction(localTime,knotOffset(program.interval,a),knotOffset(program.interval,b),time,a.time,b.time);WallKnot out=a;out.time=time;out.localTime=localTime;
    for(std::size_t i=0;i<out.faces.size();++i){auto& q=out.faces[i];const auto& t=b.faces[i];
        q.temperature+=f*(t.temperature-q.temperature);q.velocity=q.velocity+(t.velocity-q.velocity)*f;
        q.primaryKind=localTime==knotOffset(program.interval,b)?t.primaryKind:q.primaryKind;
        for(int s=0;s<Ns;++s){q.speciesRate[s]+=f*(t.speciesRate[s]-q.speciesRate[s]);q.poreRate[s]+=f*(t.poreRate[s]-q.poreRate[s]);q.poreSweepRate[s]+=f*(t.poreSweepRate[s]-q.poreSweepRate[s]);}
        for(int c=0;c<Nc;++c)q.condensedRate[c]+=f*(t.condensedRate[c]-q.condensedRate[c]);
    }
    for(std::size_t i=0;i<out.gasPoints.size();++i)out.gasPoints[i]=a.gasPoints[i]+(b.gasPoints[i]-a.gasPoints[i])*f;
    for(std::size_t i=0;i<out.solidPoints.size();++i)out.solidPoints[i]=a.solidPoints[i]+(b.solidPoints[i]-a.solidPoints[i])*f;
    result=std::move(out);error.clear();return true;
}
inline bool intervalSampleWall(const WallProgram& program,Real time,WallKnot& result,std::string& error){return intervalSampleWallLocal(program,windowOffset(program.interval.begin,time),result,error);}
// absoluteGeometry is the swept-volume tolerance [m^3]. For each
// reference surface face A [m^2], convert it to a displacement tolerance
// absoluteGeometry/A + relativeGeometry*sqrt(A) [m]. The program does not
// carry point-to-interface incidence, so use the strictest face tolerance
// for all points rather than allowing a coarse face to hide a fine-face error.
// This is a point-trajectory comparison only; actual swept-volume/GCL checks
// remain mandatory. No absolute world coordinate enters its scale.
inline bool wallPointLengthTolerance(const WallProgram& first,const WallProgram& second,
    const WallKnot& a,const WallKnot& b,const Tolerances& tol,Real& lengthTolerance,std::string& error){
    Real result=std::numeric_limits<Real>::infinity();
    const std::vector<Real>* areas[]={&first.surface.area,&second.surface.area};
    for(const auto* faceAreas:areas)for(Real area:*faceAreas){
        const Real threshold=tol.absoluteGeometry/area+tol.relativeGeometry*::sqrt(area);
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
                lo.x=minValue(lo.x,p.x);lo.y=minValue(lo.y,p.y);lo.z=minValue(lo.z,p.z);
                hi.x=maxValue(hi.x,p.x);hi.y=maxValue(hi.y,p.y);hi.z=maxValue(hi.z,p.z);
            }
            length=maxValue(length,mag(hi-lo));
        }
        result=length>0?tol.absoluteGeometry/(length*length)+tol.relativeGeometry*length:0;
    }
    lengthTolerance=result;error.clear();return true;
}
inline bool compareWallPrograms(const WallProgram& first,const WallProgram& second,const Tolerances& tol,
    WallProgramComparison& result,std::string& error){
    std::set<WindowTime> times;for(const auto& k:first.knots)times.insert(knotOffset(first.interval,k));for(const auto& k:second.knots)times.insert(knotOffset(second.interval,k));
    if(times.empty())return intervalFailure(error,"wall program does not cover interval");
    WallProgramComparison comparison;
    for(WindowTime time:times){WallKnot a,b;if(!intervalSampleWallLocal(first,time,a,error)||!intervalSampleWallLocal(second,time,b,error))return false;
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
