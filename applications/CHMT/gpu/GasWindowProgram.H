#ifndef CHMT_GPU_GASWINDOWPROGRAM_H
#define CHMT_GPU_GASWINDOWPROGRAM_H
#include "coupling/IntervalContracts.H"
#include <algorithm>
namespace chmt {
inline bool sampleGasWallProgramLocal(const WallProgram& program,WindowTime localTime,WallKnot& sample,std::string& error,bool includePoints=true){
    const Real time=windowAbsolute(program.interval.begin,localTime);
    if(program.knots.size()<2){error="wall program sample outside interval";return false;}
    auto upper=std::upper_bound(program.knots.begin(),program.knots.end(),localTime,[&](WindowTime t,const WallKnot& knot){return t<knotOffset(program.interval,knot);});
    if(upper==program.knots.end()){sample.time=time;sample.localTime=localTime;sample.faces=program.knots.back().faces;if(includePoints){sample.gasPoints=program.knots.back().gasPoints;sample.solidPoints=program.knots.back().solidPoints;}error.clear();return true;}
    const auto& a=*(upper-1);const auto& b=*upper;const Real alpha=windowFraction(localTime,knotOffset(program.interval,a),knotOffset(program.interval,b),time,a.time,b.time);
    sample.faces=a.faces;sample.time=time;sample.localTime=localTime;
    if(includePoints){sample.gasPoints=a.gasPoints;sample.solidPoints=a.solidPoints;}
    for(std::size_t i=0;includePoints&&i<sample.gasPoints.size();++i)sample.gasPoints[i]=a.gasPoints[i]+(b.gasPoints[i]-a.gasPoints[i])*alpha;
    for(std::size_t i=0;includePoints&&i<sample.solidPoints.size();++i)sample.solidPoints[i]=a.solidPoints[i]+(b.solidPoints[i]-a.solidPoints[i])*alpha;
    for(std::size_t f=0;f<sample.faces.size();++f){auto& q=sample.faces[f];const auto& x=a.faces[f];const auto& y=b.faces[f];
        q.temperature=x.temperature+alpha*(y.temperature-x.temperature);q.velocity=x.velocity+(y.velocity-x.velocity)*alpha;
        q.normalVelocity=x.normalVelocity+alpha*(y.normalVelocity-x.normalVelocity);q.solidNormalVelocity=x.solidNormalVelocity+alpha*(y.solidNormalVelocity-x.solidNormalVelocity);
        q.radiationFlux=x.radiationFlux+alpha*(y.radiationFlux-x.radiationFlux);
        for(int s=0;s<Ns;++s){q.speciesRate[s]=x.speciesRate[s]+alpha*(y.speciesRate[s]-x.speciesRate[s]);q.poreRate[s]=x.poreRate[s]+alpha*(y.poreRate[s]-x.poreRate[s]);q.poreSweepRate[s]=x.poreSweepRate[s]+alpha*(y.poreSweepRate[s]-x.poreSweepRate[s]);}
        for(int c=0;c<Nc;++c)q.condensedRate[c]=x.condensedRate[c]+alpha*(y.condensedRate[c]-x.condensedRate[c]);
    }
    error.clear();return true;
}
inline bool sampleGasWallProgram(const WallProgram& program,Real time,WallKnot& sample,std::string& error,bool includePoints=true){
    return sampleGasWallProgramLocal(program,windowOffset(program.interval.begin,time),sample,error,includePoints);
}
inline WindowTime nextGasWallKnotLocal(const WallProgram& program,WindowTime time){
    for(const auto& knot:program.knots){const auto local=knotOffset(program.interval,knot);if(local>time)return local;}
    return intervalDuration(program.interval);
}
inline Real nextGasWallKnot(const WallProgram& program,Real time){return windowAbsolute(program.interval.begin,nextGasWallKnotLocal(program,windowOffset(program.interval.begin,time)));}
inline Real boundedGasMicrostep(Real time,Real knot,Real cap,Real maximum){return minValue(cap,minValue(maximum,knot-time));}

inline Real gasCoMaxStep(Real referenceDt,Real targetCo,Real maximumCo,Real growthBase){
    const Real stable=maximumCo>0?referenceDt*(targetCo/maximumCo):referenceDt;
    return minValue(referenceDt,minValue(stable,1.05*growthBase));
}
}
#endif
