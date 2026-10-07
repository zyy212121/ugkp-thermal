#ifndef CHMT_GPU_GASWINDOWPROGRAM_H
#define CHMT_GPU_GASWINDOWPROGRAM_H
#include "coupling/IntervalContracts.H"
#include <algorithm>
namespace chmt {
inline bool validateGasWallProgram(const WallProgram& program,const HostState& base,std::string& error){
    const auto& interval=program.interval;
    if(!finite(interval.begin)||!finite(interval.end)||interval.begin!=base.time||interval.end<=interval.begin
        ||program.knots.size()<2||program.knots.front().time!=interval.begin||program.knots.back().time!=interval.end){error="wall program must span the synchronized coupling interval";return false;}
    const std::size_t nf=program.surface.area.size();
    if(nf!=base.surface.area.size()||program.surface.gasFace!=base.surface.gasFace||program.surface.solidFace!=base.surface.solidFace
        ||program.surface.solidCell!=base.surface.solidCell||program.surface.persistentId!=base.surface.persistentId
        ||!program.surface.gasMapOffsets.empty()||!program.surface.gasMapFaces.empty()||!program.surface.gasMapWeights.empty()){
        error="gas window requires the existing conformal interface topology";return false;}
    if(program.surface.area!=base.surface.area||program.surface.gasDistance!=base.surface.gasDistance
        ||program.surface.solidDistance!=base.surface.solidDistance||program.surface.edgeOwner!=base.surface.edgeOwner
        ||program.surface.edgeNeighbour!=base.surface.edgeNeighbour||program.surface.normal.size()!=base.surface.normal.size()){
        error="wall program surface does not describe the synchronized macro geometry";return false;}
    for(std::size_t f=0;f<program.surface.normal.size();++f)if(mag(program.surface.normal[f]-base.surface.normal[f])!=0){error="wall program surface normal differs from macro geometry";return false;}
    if(!program.donorPlan.knots.empty()&&(program.donorPlan.interval.identity.sequence!=interval.identity.sequence
        ||program.donorPlan.interval.identity.epoch!=interval.identity.epoch||program.donorPlan.interval.begin!=interval.begin||program.donorPlan.interval.end!=interval.end)){
        error="donor plan belongs to another window/replay";return false;}
    for(std::size_t k=0;k<program.knots.size();++k){const auto& knot=program.knots[k];
        if(!finite(knot.time)||(k&&knot.time<=program.knots[k-1].time)||knot.faces.size()!=nf
            ||knot.gasPoints.size()!=base.gasMesh.points.size()||knot.solidPoints.size()!=base.solidMesh.points.size()){
            error="wall program knot time or topology mismatch";return false;}
        for(const auto& point:knot.gasPoints)if(!finite(point)){error="nonfinite gas trajectory";return false;}
        for(const auto& point:knot.solidPoints)if(!finite(point)){error="nonfinite solid trajectory";return false;}
        for(std::size_t f=0;f<nf;++f){const auto& w=knot.faces[f];
            if(!(w.temperature>0)||!finite(w.temperature)||!finite(w.velocity)||!finite(w.normalVelocity)||!finite(w.solidNormalVelocity)||!finite(w.radiationFlux)
                ||(w.primaryKind!=ExchangeKind::GasSolid&&w.primaryKind!=ExchangeKind::GasFilm)){
                error="invalid prescribed wall state";return false;}
            if(k&&w.primaryKind!=program.knots[k-1].faces[f].primaryKind){error="wet/dry owner change must terminate the coupling window";return false;}
            for(int s=0;s<Ns;++s)if(!finite(w.speciesRate[s])||!finite(w.poreRate[s])||!finite(w.poreSweepRate[s])){error="nonfinite wall species rate";return false;}
            for(int c=0;c<Nc;++c)if(!finite(w.condensedRate[c])){error="nonfinite wall component rate";return false;}
        }
    }
    for(std::size_t i=0;i<base.gasMesh.points.size();++i)if(mag(program.knots.front().gasPoints[i]-base.gasMesh.points[i])!=0){error="gas trajectory does not start at the macro base";return false;}
    for(std::size_t i=0;i<base.solidMesh.points.size();++i)if(mag(program.knots.front().solidPoints[i]-base.solidMesh.points[i])!=0){error="solid trajectory does not start at the macro base";return false;}
    error.clear();return true;
}
inline bool sampleGasWallProgram(const WallProgram& program,Real time,WallKnot& sample,std::string& error,bool includePoints=true){
    if(program.knots.size()<2||!finite(time)||time<program.interval.begin||time>program.interval.end){error="wall program sample outside interval";return false;}
    auto upper=std::upper_bound(program.knots.begin(),program.knots.end(),time,[](Real t,const WallKnot& knot){return t<knot.time;});
    if(upper==program.knots.end()){sample.time=time;sample.faces=program.knots.back().faces;if(includePoints){sample.gasPoints=program.knots.back().gasPoints;sample.solidPoints=program.knots.back().solidPoints;}error.clear();return true;}
    const auto& a=*(upper-1);const auto& b=*upper;const Real alpha=(time-a.time)/(b.time-a.time);
    sample.faces=a.faces;sample.time=time;
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
inline Real nextGasWallKnot(const WallProgram& program,Real time){
    for(const auto& knot:program.knots)if(knot.time>time)return knot.time;
    return program.interval.end;
}
}
#endif
