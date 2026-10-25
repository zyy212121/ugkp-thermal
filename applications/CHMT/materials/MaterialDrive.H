#ifndef CHMT_MATERIALS_MATERIALDRIVE_H
#define CHMT_MATERIALS_MATERIALDRIVE_H
#include "coupling/IntervalContracts.H"
#include <cmath>
namespace chmt {
inline Real materialDriveRelative(Real a,Real b,Real floor){return absValue(a-b)/maxValue(floor,maxValue(absValue(a),absValue(b)));}
// Each cursor belongs to one immutable completed history. A material consumer
// advances monotonically, so already-consumed records cannot match later times.
// Backward/nonfinite queries reset the hint, preserving the ordinary first-match
// semantics without imposing sorted-history assumptions on external callers.
class MaterialDriveCursor {
    const IntervalHistory& history_;
    std::size_t record_=0;
    WindowTime previous_{};
    bool positioned_=false;
public:
    explicit MaterialDriveCursor(const IntervalHistory& history):history_(history){}
    const GasIntervalRecord* at(WindowTime time){
        if(!positioned_||time<previous_||!std::isfinite(previous_.sum)||!std::isfinite(previous_.correction))record_=0;
        positioned_=true;previous_=time;
        const auto& records=history_.records();
        while(record_<records.size()&&!(time<records[record_].localEnd))++record_;
        return record_<records.size()?&records[record_]:nullptr;
    }
    // A LOCAL first-order lag control, not a bound on global PDE error. Keep
    // source-variation comparisons in their original order and arithmetic.
    bool slabEndLocal(WindowTime begin,WindowTime requested,Real tolerance,Real tractionScale,
        WindowTime& output,std::string& error){
        const auto& history=history_;
        if(!history.complete()||begin<WindowTime{}||requested>history.localEnd()||requested<=begin
            ||!finite(tolerance)||tolerance<0||!finite(tractionScale)||tractionScale<=0){error="invalid material driving control";return false;}
        const GasIntervalRecord* held=at(begin);
        if(!held){error="no driving sample at material slab start";return false;}WindowTime end=requested;
        for(const auto& r:history.records()){if(r.localBegin<=begin||r.localBegin>=end)continue;
            if(r.gasTrace.size()!=held->gasTrace.size()||r.gasTraction.size()!=held->gasTraction.size()||r.gasGradient.size()!=held->gasGradient.size()||r.gasWallMatching.size()!=held->gasWallMatching.size()){
                error="gas drive layout changed";return false;}
            Real change=0;
            for(std::size_t f=0;f<r.gasTrace.size();++f){const auto& a=held->gasTrace[f];const auto& b=r.gasTrace[f];
                change=maxValue(change,materialDriveRelative(a.temperature,b.temperature,1));change=maxValue(change,materialDriveRelative(a.pressure,b.pressure,1));
                change=maxValue(change,materialDriveRelative(a.rho,b.rho,1e-30));
                change=maxValue(change,mag(a.velocity-b.velocity)/maxValue(1,maxValue(a.soundSpeed,b.soundSpeed)));
                for(int s=0;s<Ns;++s)change=maxValue(change,absValue(a.Y[s]-b.Y[s]));
            }
            // New-family material and film constitutive data are driven by this
            // actual matching snapshot, which may change while the owner is steady.
            for(std::size_t f=0;f<r.gasWallMatching.size();++f){const auto& a=held->gasWallMatching[f];const auto& b=r.gasWallMatching[f];
                change=maxValue(change,materialDriveRelative(a.pressure,b.pressure,1));
                change=maxValue(change,materialDriveRelative(a.mechanicalPressure,b.mechanicalPressure,1));
                change=maxValue(change,materialDriveRelative(a.state.temperature,b.state.temperature,1));
                const Real speedScale=f<r.gasTrace.size()?maxValue(1,maxValue(r.gasTrace[f].soundSpeed,held->gasTrace[f].soundSpeed)):1;
                Real velocityChange=0;for(int d=0;d<3;++d){const Real delta=a.state.velocity[d]-b.state.velocity[d];velocityChange+=delta*delta;}
                change=maxValue(change,::sqrt(velocityChange)/speedScale);
                for(int s=0;s<Ns;++s)change=maxValue(change,absValue(a.state.massFraction[s]-b.state.massFraction[s]));
                change=maxValue(change,materialDriveRelative(a.state.k,b.state.k,1e-12));
                change=maxValue(change,materialDriveRelative(a.state.omega,b.state.omega,1e-12));
            }
            for(std::size_t f=0;f<r.gasTraction.size();++f)change=maxValue(change,mag(r.gasTraction[f]-held->gasTraction[f])/maxValue(tractionScale,maxValue(mag(r.gasTraction[f]),mag(held->gasTraction[f]))));
            for(std::size_t f=0;f<r.gasGradient.size();++f){for(int j=0;j<3;++j)change=maxValue(change,mag(r.gasGradient[f].velocity[j]-held->gasGradient[f].velocity[j])/maxValue(1,maxValue(mag(r.gasGradient[f].velocity[j]),mag(held->gasGradient[f].velocity[j]))));}
            if(!finite(change)){error="nonfinite gas driving change";return false;}
            if(change>tolerance||r.radiationReceiver!=held->radiationReceiver){end=r.localBegin;break;}
        }
        if(!(end>begin)){error="driving slab does not advance";return false;}output=end;error.clear();return true;
    }
};
inline bool materialDriveSlabEndLocal(const IntervalHistory& history,WindowTime begin,WindowTime requested,
    Real tolerance,Real tractionScale,WindowTime& output,std::string& error){
    MaterialDriveCursor cursor(history);
    return cursor.slabEndLocal(begin,requested,tolerance,tractionScale,output,error);
}
inline bool materialDriveSlabEnd(const IntervalHistory& history,Real begin,Real requested,Real tolerance,Real tractionScale,Real& output,std::string& error){
    WindowTime local;const Real origin=history.interval().begin;
    if(!materialDriveSlabEndLocal(history,windowOffset(origin,begin),windowOffset(origin,requested),tolerance,tractionScale,local,error))return false;
    output=windowAbsolute(origin,local);return true;
}
}
#endif
