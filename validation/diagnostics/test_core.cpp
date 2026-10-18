#include "Diagnostics.H"
#include "DiagnosticsHost.H"
#include "DiagnosticsCodec.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>
using namespace ugkp::diagnostics;
struct FatalCaught { Record record; };
static void fatal(void*, const Record& r, const SourceDescriptor*) { throw FatalCaught{r}; }
static void warning(void* p, const WarningSummary&, const SourceDescriptor*) { ++*static_cast<int*>(p); }
int main() {
    static_assert(std::is_trivial<DeviceContext>::value, "device storage is POD");
    assert(!isWarning(Code::InvalidPhysicalState));
    DeviceContext classification{};reset(classification,1,1);
    recordWarning(&classification,makeRecord(Code::NonFinite));assert(hasFatal(&classification));
    reset(classification,2,1);recordFatal(&classification,makeRecord(Code::AccuracyResidual));assert(!hasFatal(&classification));
    DeviceContext a{}, b{};
    reset(a, 91, 17, 7, 0.25); reset(b, 92, 18);
    Record w = makeRecord(Code::FiniteYClip, 101, 2, 3, EntityKind::Cell, 9007199254740993LL, 4, 1.5, 1., 1.);
    recordWarning(&a,w); assert(!hasFatal(&a)); assert(a.warnings[2].count == 1);
    Record f = makeRecord(Code::NegativeTemperature, 102, 2, 3, EntityKind::Cell, 9, -1, -2., 0., 0.);
    recordFatal(&a,f); recordFatal(&a,makeRecord(Code::NonFinite));
    assert(hasFatal(&a) && !hasFatal(&b)); assert(a.firstFatal.code == Code::NegativeTemperature);
    assert(a.firstFatal.operationSequence == 91 && a.firstFatal.instanceId == 17);
    assert(a.firstFatal.step == 7 && a.firstFatal.simulationTime == 0.25);
    beginOperation(a, 92, 8, 0.5); recordFatal(&a, makeRecord(Code::NonFinite));
    assert(a.firstFatal.operationSequence == 91 && a.firstFatal.step == 7);
    assert(a.operationSequence == 92);
    auto snapshot = takeSnapshot(a, {{101,"common/species.cuh","clip",10,"species clipping"},{102,"common/gas.cuh","temperature",20,"negative temperature"},{999,"unused.cuh","unused",1,"unused"}}, "test-catalogue");
    assert(snapshot.sources.size()==2);
    auto bytes = encode(snapshot); Snapshot decoded; std::string error;
    assert(decode(bytes.data(),bytes.size(),decoded,error));
    assert(decoded.operationSequence == 92 && decoded.hasFatal);
    assert(decoded.warnings[0].sample.entityIndex == 9007199254740993LL);
    assert(decoded.sources[0].file == "common/species.cuh");
    assert(encode(decoded) == bytes);
    for (std::size_t n=0;n<bytes.size();++n) { Snapshot bad; assert(!decode(bytes.data(),n,bad,error)); }
    auto extra=bytes; extra.push_back(0); assert(!decode(extra.data(),extra.size(),decoded,error));
    DeviceContext explicitContext{}; reset(explicitContext, 0, 18);
    auto explicitRecord=makeRecord(Code::NonFinite);explicitRecord.metadata=3;
    explicitRecord=withOperation(explicitRecord,400,12,0.75);
    assert((explicitRecord.metadata & detailCodeMask)==3);
    assert(Quantity::LinearSystem==12);
    recordFatal(&explicitContext,explicitRecord);
    assert(explicitContext.firstFatal.operationSequence==400 && explicitContext.firstFatal.step==12);
    assert(explicitContext.firstFatal.simulationTime==0.75 && explicitContext.firstFatal.instanceId==18);
    int warningCount=0; HostSink sink{&warningCount,fatal,warning,true};
    try { dispatch(snapshot,sink); assert(false); } catch (const FatalCaught& e) { assert(e.record.code == Code::NegativeTemperature); }
    assert(warningCount==1);
    sink.warningsEnabled=false;
    try { dispatch(snapshot,sink); assert(false); } catch (const FatalCaught&) {}
    assert(warningCount==1 && snapshot.warnings.size()==1);
    reset(a,93,17);
    std::vector<std::thread> workers;
    for(int i=0;i<8;++i) workers.emplace_back([&,i] { Record r=f;r.entityIndex=i;r.observed=i;for(int j=0;j<1000;++j) {recordFatal(&a,r);recordWarning(&a,w);} });
    for(auto& t:workers)t.join();
    assert(a.warnings[2].count==8000 && a.firstFatal.entityIndex==a.firstFatal.observed);
    a.warnings[2].count=std::numeric_limits<std::uint32_t>::max(); recordWarning(&a,w);
    assert(a.warnings[2].count==std::numeric_limits<std::uint32_t>::max() && a.warnings[2].overflow==1);
    for(Code code: {Code::NegativeTemperature,Code::NonFinite,Code::InvalidIndex,Code::MissingOperator,Code::ZeroDenominator,Code::InvalidPhysicalState}) {
        DeviceContext failure{};reset(failure,44,55);recordFatal(&failure,makeRecord(code));
        int spy=0;if(!hasFatal(&failure))++spy;assert(spy==0);
    }
    for(Code code: {Code::ConservationResidual,Code::AccuracyResidual}) {reset(b,1,2); recordWarning(&b,makeRecord(code)); assert(!hasFatal(&b));}
    auto temperature=makeRecord(Code::NegativeTemperature,0,0,0,EntityKind::Cell,1,-1,-2,0,0,Quantity::Temperature);
    assert(formatRecord(temperature,nullptr).find("quantity=temperature")!=std::string::npos);
    auto unknown=takeSnapshot(b,{},"other-build"); unknown.warnings[0].sample.sourceSiteId=999999;
    assert(formatRecord(unknown.warnings[0].sample,nullptr).find("999999")!=std::string::npos);
}
