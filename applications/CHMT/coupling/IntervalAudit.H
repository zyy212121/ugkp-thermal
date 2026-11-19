#ifndef CHMT_COUPLING_INTERVALAUDIT_H
#define CHMT_COUPLING_INTERVALAUDIT_H
#include "coupling/IntervalContracts.H"
#include "coupling/WallProgramComparison.H"
#include <set>
#include "../../../common/HostMemory.H"
namespace chmt {
inline Real intervalNormalizedError(Real a,Real b,Real absolute,Real relative){
    const Real difference=absValue(a-b),tolerance=absolute+relative*maxValue(absValue(a),absValue(b));
    if(!finite(a)||!finite(b)||!finite(difference)||!finite(absolute)||!finite(relative)||!finite(tolerance)||absolute<0||relative<0)return undefinedValue();



    if(tolerance==0)return difference==0?0:std::numeric_limits<Real>::max();
    if(tolerance<1&&difference>std::numeric_limits<Real>::max()*tolerance)return std::numeric_limits<Real>::max();
    return difference/tolerance;
}



template<class Records,class Release>
inline bool finalizeIntervalLedgerRecords(Records& records,HostState& candidate,std::string& error,Release release){
    std::vector<ExchangePacket> ledger;std::size_t size=0;
    for(const auto& p:candidate.ledger.values())if(p.kind==ExchangeKind::SolidFilm)++size;
    for(const auto& r:records)size+=r.packets.size();
    ledger.reserve(size);
    for(const auto& p:candidate.ledger.values())if(p.kind==ExchangeKind::SolidFilm)ledger.push_back(p);
    for(auto& r:records){
        for(auto p:r.packets){p.consumerMask=requiredConsumers(p.kind);ledger.push_back(p);}
        release(r);
    }
    candidate.ledger=std::move(ledger);error.clear();return true;
}
inline bool finalizeReleasedIntervalLedger(std::vector<GasIntervalRecord> records,HostState& candidate,std::string& error){


    std::size_t count=0;
    const bool ok=finalizeIntervalLedgerRecords(records,candidate,error,[&](GasIntervalRecord& r){
        r=GasIntervalRecord{};if(++count%512==0)ugkp::releaseUnusedHostPages();
    });
    ugkp::releaseUnusedHostPages();return ok;
}


inline bool finalizeReleasedIntervalLedger(IntervalHistory history,HostState& candidate,std::string& error){
    if(!history.externalPayload())return finalizeReleasedIntervalLedger(history.releaseRecords(),candidate,error);
    std::vector<ExchangePacket> solidFilm;
    candidate.ledger.forEachChunk([&](const ExchangePacket* packets,std::size_t count){
        for(std::size_t i=0;i<count;++i)
            if(packets[i].kind==ExchangeKind::SolidFilm)solidFilm.push_back(packets[i]);
    });
    candidate.ledger.freeze([first=std::move(solidFilm),packets=history.releaseLedgerReader()](const AcceptedLedger::ChunkVisitor& visit){
        if(!first.empty())visit(first.data(),first.size());
        packets(visit);
    });
    error.clear();return true;
}
inline bool finalizeIntervalLedger(const IntervalHistory& history,HostState& candidate,std::string& error){
    if(history.externalPayload())return finalizeReleasedIntervalLedger(history,candidate,error);
    return finalizeIntervalLedgerRecords(history.records(),candidate,error,[](const GasIntervalRecord&){});
}
inline bool intervalSampleWallLocal(const WallProgram& program,WindowTime localTime,WallKnot& result,std::string& error){
    const Real time=windowAbsolute(program.interval.begin,localTime);
    if(program.knots.size()<2)return intervalFailure(error,"wall comparison time outside program");
    std::size_t hi=1;while(hi+1<program.knots.size()&&localTime>knotOffset(program.interval,program.knots[hi]))++hi;
    const auto& a=program.knots[hi-1];const auto& b=program.knots[hi];
    if(a.faces.size()!=b.faces.size()||a.gasPoints.size()!=b.gasPoints.size()
        ||a.solidPoints.size()!=b.solidPoints.size())return intervalFailure(error,"wall interpolation field layout is incomplete");
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







inline bool wallPointLengthTolerance(const WallProgram& first,const WallProgram& second,
    const WallKnot& a,const WallKnot& b,const Tolerances& tol,Real& lengthTolerance,std::string& error){
    Real result=std::numeric_limits<Real>::infinity();
    const std::vector<Real>* areas[]={&first.surface.area,&second.surface.area};
    for(const auto* faceAreas:areas)for(Real area:*faceAreas){
        if(!finite(area)||area<=0)return intervalFailure(error,"invalid wall geometry comparison area");
        const Real threshold=tol.absoluteGeometry/area+tol.relativeGeometry*::sqrt(area);
        if(!finite(threshold))return intervalFailure(error,"nonfinite wall geometry comparison scale");
        result=minValue(result,threshold);
    }
    if(first.surface.area.empty()&&second.surface.area.empty()){




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
    if(!finite(result)||result<0)return intervalFailure(error,"nonfinite wall point comparison tolerance");
    lengthTolerance=result;error.clear();return true;
}
inline bool compareWallProgramsAtTimes(const WallProgram& first,const WallProgram& second,const Tolerances& tol,
    const std::set<WindowTime>& times,WallProgramComparison& result,std::string& error){
    if(times.empty())return intervalFailure(error,"wall program does not cover interval");
    WallProgramComparison comparison=result;
    bool valid=true;
    auto accumulate=[&](Real old,Real value){if(!finite(value))valid=false;return maxValue(old,value);};
    for(WindowTime time:times){WallKnot a,b;if(!intervalSampleWallLocal(first,time,a,error)||!intervalSampleWallLocal(second,time,b,error))return false;
        if(a.faces.size()!=b.faces.size()||first.surface.area.size()<a.faces.size()
            ||second.surface.area.size()<a.faces.size()||a.gasPoints.size()!=b.gasPoints.size()
            ||a.solidPoints.size()!=b.solidPoints.size())return intervalFailure(error,"wall comparison field layout is incomplete");
        for(std::size_t i=0;i<a.faces.size();++i){const auto& x=a.faces[i];const auto& y=b.faces[i];comparison.regimeMatches=comparison.regimeMatches&&x.primaryKind==y.primaryKind;
            comparison.thermalError=accumulate(comparison.thermalError,intervalNormalizedError(x.temperature,y.temperature,tol.absoluteTemperature,tol.relativeTemperature));
            const Real scale=(first.interval.end-first.interval.begin)*maxValue(first.surface.area[i],second.surface.area[i]);
            for(int s=0;s<Ns;++s){comparison.massPredictionError=accumulate(comparison.massPredictionError,intervalNormalizedError(x.speciesRate[s]*scale,y.speciesRate[s]*scale,tol.absoluteMass,tol.relativeMass));
                comparison.massPredictionError=accumulate(comparison.massPredictionError,intervalNormalizedError(x.poreRate[s]*scale,y.poreRate[s]*scale,tol.absoluteMass,tol.relativeMass));
                comparison.massPredictionError=accumulate(comparison.massPredictionError,intervalNormalizedError(x.poreSweepRate[s]*scale,y.poreSweepRate[s]*scale,tol.absoluteMass,tol.relativeMass));}
            for(int c=0;c<Nc;++c)comparison.massPredictionError=accumulate(comparison.massPredictionError,intervalNormalizedError(x.condensedRate[c]*scale,y.condensedRate[c]*scale,tol.absoluteMass,tol.relativeMass));
        }
        Real pointTolerance=0;
        if(!wallPointLengthTolerance(first,second,a,b,tol,pointTolerance,error))return false;
        auto points=[&](const std::vector<Vec3>& x,const std::vector<Vec3>& y){for(std::size_t i=0;i<x.size();++i){
            const Real distance=mag(x[i]-y[i]);
            const Real normalized=intervalNormalizedError(distance,0,pointTolerance,0);
            comparison.geometryError=accumulate(comparison.geometryError,normalized);}};
        points(a.gasPoints,b.gasPoints);points(a.solidPoints,b.solidPoints);
    }
    if(!valid)return intervalFailure(error,"nonfinite wall waveform comparison");
    result=comparison;error.clear();return true;
}
inline bool compareWallPrograms(const WallProgram& first,const WallProgram& second,const Tolerances& tol,
    WallProgramComparison& result,std::string& error){
    std::set<WindowTime> times;
    for(const auto& k:first.knots)times.insert(knotOffset(first.interval,k));
    for(const auto& k:second.knots)times.insert(knotOffset(second.interval,k));
    WallProgramComparison comparison;
    if(!compareWallProgramsAtTimes(first,second,tol,times,comparison,error))return false;
    result=comparison;return true;
}




inline bool accumulateWallProgramSegment(const WallProgram& first,const WallProgram& second,const Tolerances& tol,
    WallProgramComparison& result,std::string& error){
    if(second.knots.size()<2)return intervalFailure(error,"wall comparison requires an adjacent knot pair");
    const auto begin=knotOffset(second.interval,second.knots[0]),end=knotOffset(second.interval,second.knots[1]);
    if(!(begin<end))return intervalFailure(error,"wall comparison segment does not advance");
    std::set<WindowTime> times{end};
    if(begin==WindowTime{})times.insert(begin);
    for(const auto& k:first.knots){const auto time=knotOffset(first.interval,k);if(begin<time&&time<=end)times.insert(time);}
    return compareWallProgramsAtTimes(first,second,tol,times,result,error);
}
}
#endif
