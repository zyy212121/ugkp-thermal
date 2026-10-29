#ifndef CHMT_GPU_GASWINDOWPROGRAM_H
#define CHMT_GPU_GASWINDOWPROGRAM_H
#include "coupling/IntervalContracts.H"
#include <algorithm>
namespace chmt {
// Keep the original subtract/multiply/add rounding order on both processors.
// Prepared wall reuse compares these interpolated inputs exactly; contracting
// them into a device FMA would spuriously invalidate an identical wall closure.
CHMT_HD inline Real interpolateGasWallValue(Real a,Real b,Real alpha){
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,__dmul_rn(alpha,__dsub_rn(b,a)));
#else
    return a+alpha*(b-a);
#endif
}
CHMT_HD inline WallFaceSample interpolateGasWallFace(const WallFaceSample& x,const WallFaceSample& y,Real alpha){
    WallFaceSample q=x;
    q.temperature=interpolateGasWallValue(x.temperature,y.temperature,alpha);
    q.materialEffusivity=interpolateGasWallValue(x.materialEffusivity,y.materialEffusivity,alpha);
    q.velocity={interpolateGasWallValue(x.velocity.x,y.velocity.x,alpha),
        interpolateGasWallValue(x.velocity.y,y.velocity.y,alpha),interpolateGasWallValue(x.velocity.z,y.velocity.z,alpha)};
    q.normalVelocity=interpolateGasWallValue(x.normalVelocity,y.normalVelocity,alpha);
    q.solidNormalVelocity=interpolateGasWallValue(x.solidNormalVelocity,y.solidNormalVelocity,alpha);
    q.radiationFlux=interpolateGasWallValue(x.radiationFlux,y.radiationFlux,alpha);
    for(int s=0;s<Ns;++s){
        q.speciesRate[s]=interpolateGasWallValue(x.speciesRate[s],y.speciesRate[s],alpha);
        q.poreRate[s]=interpolateGasWallValue(x.poreRate[s],y.poreRate[s],alpha);
        q.poreSweepRate[s]=interpolateGasWallValue(x.poreSweepRate[s],y.poreSweepRate[s],alpha);
    }
    for(int c=0;c<Nc;++c)q.condensedRate[c]=interpolateGasWallValue(x.condensedRate[c],y.condensedRate[c],alpha);
    return q;
}
struct GasWallProgramPosition {int lower=0,upper=0;Real alpha=0;};
inline bool locateGasWallProgramLocal(const WallProgram& program,WindowTime localTime,
    GasWallProgramPosition& position,std::string& error){
    if(program.knots.size()<2){error="wall program sample outside interval";return false;}
    const auto upper=std::upper_bound(program.knots.begin(),program.knots.end(),localTime,
        [&](WindowTime t,const WallKnot& knot){return t<knotOffset(program.interval,knot);});
    if(upper==program.knots.end()){
        position.lower=position.upper=int(program.knots.size()-1);position.alpha=0;
    }else{
        position.upper=int(upper-program.knots.begin());position.lower=position.upper-1;
        const auto& a=program.knots[position.lower];const auto& b=program.knots[position.upper];
        position.alpha=windowFraction(localTime,knotOffset(program.interval,a),knotOffset(program.interval,b),
            windowAbsolute(program.interval.begin,localTime),a.time,b.time);
    }
    error.clear();return true;
}
CHMT_HD inline WallFaceSample sampleResidentGasWallFace(const WallFaceSample* knots,int count,
    GasWallProgramPosition position,int face){
    const auto& a=knots[position.lower*count+face];
    return position.lower==position.upper?a:interpolateGasWallFace(a,knots[position.upper*count+face],position.alpha);
}
inline bool sampleGasWallProgramLocal(const WallProgram& program,WindowTime localTime,WallKnot& sample,std::string& error,bool includePoints=true){
    GasWallProgramPosition position;
    if(!locateGasWallProgramLocal(program,localTime,position,error))return false;
    const auto& a=program.knots[position.lower];const auto& b=program.knots[position.upper];
    sample.time=windowAbsolute(program.interval.begin,localTime);sample.localTime=localTime;sample.faces=a.faces;
    if(includePoints){sample.gasPoints=a.gasPoints;sample.solidPoints=a.solidPoints;}
    if(position.lower==position.upper)return true;
    const Real alpha=position.alpha;
    for(std::size_t i=0;includePoints&&i<sample.gasPoints.size();++i)sample.gasPoints[i]=a.gasPoints[i]+(b.gasPoints[i]-a.gasPoints[i])*alpha;
    for(std::size_t i=0;includePoints&&i<sample.solidPoints.size();++i)sample.solidPoints[i]=a.solidPoints[i]+(b.solidPoints[i]-a.solidPoints[i])*alpha;
    for(std::size_t f=0;f<sample.faces.size();++f)sample.faces[f]=interpolateGasWallFace(a.faces[f],b.faces[f],alpha);
    return true;
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
