import difflib,json
from r2_runner import *
p=D/'inputs/fixtures/chemistry_trial_host_device_split.cu';old=p.read_text()
new=old.replace('static std::vector<double> sourceDurations;', 'static std::vector<double> sourceDurations,acceptedDurations;\n    static double currentDuration;')
new=new.replace('sourceDurations.clear();}', 'sourceDurations.clear();acceptedDurations.clear();}')
new=new.replace('sourceDurations.push_back(dt);if(!native)', 'currentDuration=dt;sourceDurations.push_back(dt);if(!native)')
new=new.replace('++commits;return SharedGasTrialPolicy::commit(s);','++commits;acceptedDurations.push_back(currentDuration);return SharedGasTrialPolicy::commit(s);')
new=new.replace('std::vector<double> TracePolicy::sourceDurations;','std::vector<double> TracePolicy::sourceDurations,TracePolicy::acceptedDurations;\ndouble TracePolicy::currentDuration=0;')
start=new.index('void ownerTests(){');end=new.index('void failedCellTests(){',start)
body=r'''void prepareOwner(DeviceState* s){
    double co=-999;check(ugkwpGpuResidentStrictComputeGasCourant(s,4e-6,.5,2.,&co)==0,"real native Courant preparation");
    std::printf("OWNER_INPUT targetMaxCo=%a measuredCo=%a\n",s->gasTrial.targetMaxCo,co);
}
void ledger(DeviceState* s){
    auto* f=s->gasBudgetHost->output;std::rewind(f);char row[8192];
    while(std::fgets(row,sizeof(row),f))std::printf("OWNER_LEDGER %s",row);
}
void ownerTests(){
    auto* actual=fixture();auto* reference=fixture();prepareOwner(actual);prepareOwner(reference);TracePolicy::clear();
    GasRequestedIntervalControls<double> controls;
    check(prepareGasBudgetRequest(actual,4e-6,2.)==0,"prepare full requested budget");
    check(advanceGasRequestedInterval<TracePolicy>(actual,4e-6,2.,controls)==0,"requested interval did not complete");
    std::printf("OWNER_TRACE begins=%d commits=%d rollbacks=%d stepLimits=%d terminal=%d\n",TracePolicy::begins,TracePolicy::commits,TracePolicy::rollbacks,TracePolicy::stepLimits,TracePolicy::terminalStages);
    check(TracePolicy::stepLimits>0 && TracePolicy::rollbacks>0 && TracePolicy::commits>1 && TracePolicy::terminalStages==0,"owner must complete by genuine accepted native substeps after StepLimit");
    double elapsed=0.;for(double dt:TracePolicy::acceptedDurations){
        std::printf("OWNER_ACCEPT dt=%a stamp=%a\n",dt,2.+elapsed);
        check(dt<4e-6,"StepLimit shrink must be real");
        check(advanceGasTrial<SharedGasTrialPolicy>(reference,dt,2.+elapsed,true)==0,"explicit accepted native partition failed");elapsed+=dt;
    }
    check(elapsed==4e-6,"accepted native partition must cover exact original requested interval");
    compare(actual,reference);
    check(actual->gasBudgetHost->sequence==1&&!actual->gasBudgetHost->requestActive&&actual->gasBudgetHost->acceptedTime==2.+4e-6,"requested ledger completion differs");
    ledger(actual);std::puts("PASS actual requested interval with native Courant preparation: complete partition byteequal fields/audits/budget");
    actual=fixture();reference=fixture();prepareOwner(actual);prepareOwner(reference);TracePolicy::clear();controls.maximumAttempts=1;
    check(prepareGasBudgetRequest(actual,4e-6,2.)==0,"prepare terminal budget");
    check(advanceGasRequestedInterval<TracePolicy>(actual,4e-6,2.,controls)==0,"exhausted owner did not complete terminal interval");
    check(TracePolicy::begins==2&&TracePolicy::rollbacks==1&&TracePolicy::commits==1&&TracePolicy::terminalStages==1,"terminal fallback trace differs");
    check(advanceGasTrial<SharedGasTrialPolicy>(reference,4e-6,2.,false)==0,"direct public completion failed");compare(actual,reference);
    ledger(actual);std::puts("PASS exhausted native effort preserves complete original public terminal result");
    actual=fixture();reference=fixture();TracePolicy::clear();
    // Retain the originally requested1e-6 chemistry success control; the
    // single Trial witness does not invoke the omitted external Co selector.
    advanceGasChemistryKernel<<<1,32>>>(actual->deviceState,1e-6,actual->V,true);
    advanceGasChemistryKernel<<<1,32>>>(reference->deviceState,1e-6,reference->V,false);sync();
    check(actual->gasSpecies.chemistryStatus[0]&&reference->gasSpecies.chemistryStatus[0],"original chemistry success control failed");
    compare(actual,reference);std::puts("PASS original1e-6 chemistry success native/public fields and audits byteequal");
}
'''
new=new[:start]+body+new[end:];q=D/'inputs/fixtures/chemistry_owner_native_prepared.cu';q.write_text(new)
patch=D/'patches/S203-actual-native-courant-owner-fixture.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/reviewer/chemistry_trial_host_device_split.cu',tofile='b/reviewer/chemistry_owner_native_prepared.cu')))
(D/'patches/S203-actual-native-courant-owner-fixture.json').write_text(json.dumps({'classification':'NEW_TEST_SUPPORT','production_modified':False,'old_failure':'CHEM-OWNER-S202','cause':'Fixture omitted native ComputeGasCourant preparation; targetMaxCo default0. Original expectation of exactlytwo commits also excludes physical CFL shrinking for supplied V/face area. New distinct full-owner witness prepares actual production Co API with unchanged original target0.5, preserves4e-6 interval/geometry/material/maxSteps1. Actual accepted native partition must fully cover request and bitmatch explicit native stages; old two-part assertion/red is retained.','old_sha256':sha(p),'new_sha256':sha(q),'thresholds_changed':False,'chemistry_control':'Original1e-6 direct native/public comparison remains byteexact.'},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nA新增S203完整owner witness：S202原夹具没有调用实际ComputeGasCourant，targetMaxCo默认0，原强制两accepted assertion失败。保留红；新夹具调用真实native Co preparation，target0.5和原4e-6/物性/几何/maxSteps1均不变，观测全部acceptedpartition（允许真实CFL缩步），必须native完成整区间、逐字节等价显式同partition、public terminal对照和原1e-6chemical单kernel对照。不是扩大数值阈值；300s。\n')
args=json.loads((D/'inputs/fixtures/chemistry-command.json').read_text());binary=ROOT/'work/fixtures/chemistry/chemistry_owner_native_prepared';args[-3]=str(q);args[-1]=str(binary)
t=run('BUILD-CHEM-OWNER-S203',cmd(*args),[10,14,15,26],900,'Actual native requested owner including required Co preparation; independent new fixture.',inputs=[q,patch],kind='BUILD')
if t['exit_code']==0:
 t=run('CHEM-OWNER-S203',cmd(binary),[10,14,15,26],expected='StepLimit owner shrink+native completion whole original4e-6; all acceptedfields/audits/budget byteequal sameactualpartition; exhausted public exact;16 resource/metadata combinations.',inputs=[q,patch],binary=binary)
 raw=(D/t['log']).read_text();assertion(t,'PASS real CUDA chemistry Trial routing' in raw and 'FAIL ' not in raw,raw,test_support='S202+S203')
 print(raw)
