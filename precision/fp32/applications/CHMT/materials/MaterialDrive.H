#ifndef CHMT_MATERIALS_MATERIALDRIVE_H
#define CHMT_MATERIALS_MATERIALDRIVE_H
#include "coupling/IntervalContracts.H"
#include <cmath>
#include "../../../common/gasTransport/GasDriveChange.H"
namespace chmt {
struct MaterialDriveRecordView {
    const GasIntervalRecord& record;
    std::size_t bulkCount()const{return record.gasTrace.size();}
    std::size_t tractionCount()const{return record.gasTraction.size();}
    std::size_t gradientCount()const{return record.gasGradient.size();}
    std::size_t matchingCount()const{return record.gasWallMatching.size();}
    std::size_t receiverCount()const{return record.radiationReceiver.size();}
    const GasPrimitive& bulk(std::size_t i)const{return record.gasTrace[i];}
    const PhysicalVec3& traction(std::size_t i)const{return record.gasTraction[i];}
    const GasGradient& gradient(std::size_t i)const{return record.gasGradient[i];}
    const GasWallMatchingSample& matching(std::size_t i)const{return record.gasWallMatching[i];}
    unsigned receiver(std::size_t i)const{return record.radiationReceiver[i];}
};
// Each cursor belongs to one immutable completed history. A material consumer
// advances monotonically, so already-consumed records cannot match later times.
// Backward/nonfinite queries reset the hint, preserving the ordinary first-match
// semantics without imposing sorted-history assumptions on external callers.
class MaterialDriveCursor {
    const IntervalHistory& history_;
    std::size_t record_=0;
    WindowTime previous_{};
    bool positioned_=false,ordered_=true;
public:
    explicit MaterialDriveCursor(const IntervalHistory& history):history_(history){
        const auto& records=history.records();
        for(std::size_t i=0;i<records.size();++i)
            if(!(records[i].localBegin<records[i].localEnd)
                ||(i&&records[i].localBegin<records[i-1].localEnd)){ordered_=false;break;}
    }
    const GasIntervalRecord* at(WindowTime time){
        if(!ordered_||!positioned_||time<previous_||!std::isfinite(previous_.sum)||!std::isfinite(previous_.correction))record_=0;
        positioned_=true;previous_=time;
        const auto& records=history_.records();
        while(record_<records.size()&&!(time<records[record_].localEnd))++record_;
        return record_<records.size()?&records[record_]:nullptr;
    }
    // Exact open-interval samples for the source-variation audit. Arbitrary
    // queries retain first-match ordering; monotone production queries reuse
    // the same index without rescanning already-consumed records.
    void samplesInside(WindowTime begin,WindowTime end,std::vector<const GasIntervalRecord*>& output){
        output.clear();at(begin);
        const auto& records=history_.records();
        for(std::size_t i=ordered_?record_:0;i<records.size();++i){
            const auto& r=records[i];if(ordered_&&r.localBegin>=end)break;
            if(r.localBegin>begin&&r.localBegin<end)output.push_back(&r);
        }
    }
    // A LOCAL first-order lag control, not a bound on global PDE error. Keep
    // source-variation comparisons in their original order and arithmetic.
    bool slabEndLocal(WindowTime begin,WindowTime requested,PhysicalReal tolerance,PhysicalReal tractionScale,
        WindowTime& output,std::string& error){
        const auto& history=history_;
        if(!history.complete()||begin<WindowTime{}||requested>history.localEnd()||requested<=begin
            ||!finite(tolerance)||tolerance<0||!finite(tractionScale)||tractionScale<=0){error="invalid material driving control";return false;}
        const GasIntervalRecord* held=at(begin);
        if(!held){error="no driving sample at material slab start";return false;}WindowTime end=requested;
        for(std::size_t index=ordered_?record_:0;index<history.records().size();++index){
            const auto& r=history.records()[index];
            if(ordered_&&r.localBegin>=end)break;
            if(r.localBegin<=begin||r.localBegin>=end)continue;
            const auto comparison=ugkwp::compareGasDrive<Ns>(MaterialDriveRecordView{*held},MaterialDriveRecordView{r},tractionScale);
            if(!comparison.layoutMatches){error="gas drive layout changed";return false;}
            const PhysicalReal change=comparison.change;
            if(!finite(change)){error="nonfinite gas driving change";return false;}
            if(change>tolerance||comparison.receiverChanged){end=r.localBegin;break;}
        }
        if(!(end>begin)){error="driving slab does not advance";return false;}output=end;error.clear();return true;
    }
};
inline bool materialDriveSlabEndLocal(const IntervalHistory& history,WindowTime begin,WindowTime requested,
    PhysicalReal tolerance,PhysicalReal tractionScale,WindowTime& output,std::string& error){
    MaterialDriveCursor cursor(history);
    return cursor.slabEndLocal(begin,requested,tolerance,tractionScale,output,error);
}
inline bool materialDriveSlabEnd(const IntervalHistory& history,Real begin,Real requested,PhysicalReal tolerance,PhysicalReal tractionScale,Real& output,std::string& error){
    WindowTime local;const Real origin=history.interval().begin;
    if(!materialDriveSlabEndLocal(history,windowOffset(origin,begin),windowOffset(origin,requested),tolerance,tractionScale,local,error))return false;
    output=windowAbsolute(origin,local);return true;
}
}
#endif
