from r3_runner import *
import difflib
formal=ROOT/'work/fixtures/chemistry-trial-formal/chemistry_trial.cu';text=formal.read_text()
prefix=text[text.index('    for(int kind=0;kind<16;++kind){',text.index('void failedCellTests')):text.index('        const double savedSpecies',text.index('void failedCellTests'))]
body=r'''
void failedTransactionTests(){
    auto* successful=fixture();
    advanceGasChemistryKernel<<<1,32>>>(successful->deviceState,1e-6,successful->V,true);sync();
    check(successful->gasSpecies.chemistryStatus[0],"independent original direct control failed");
    const int expectedNewton=successful->gasSpecies.chemistryStatus[0].nonlinearIterations;
PREFIX
        auto* reference=fixture();const double interval=kind==15?0.:2e-6;
        startReference(s,interval,2.);startReference(reference,interval,2.);
        check(SharedGasTrialPolicy::begin(s)==0,"real failed transaction begin");
        const double* volume=kind==14?nullptr:s->V;
        cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));
        advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,volume,true);sync();
        using TC=decltype(g.chemistryStatus[0].thermoCode);
        auto code=ChemistryCode::InvalidModel;auto thermo=TC::Success;
        int reaction=-1,attempted=0,rejected=0,nonlinear=0;
        if(kind==0){code=ChemistryCode::StepLimit;attempted=1;nonlinear=expectedNewton;}
        if(kind==1||kind==2||kind==7||kind==12||kind==15)thermo=TC::InvalidModel;
        if(kind==4){attempted=rejected=nonlinear=1;}
        if(kind==5||kind==6||kind==9||kind==11){code=ChemistryCode::RateFailure;reaction=0;attempted=rejected=nonlinear=1;}
        if(kind==10){code=ChemistryCode::StepLimit;reaction=0;attempted=rejected=nonlinear=1;}
        if(kind==13)code=ChemistryCode::InvalidControls;
        const bool unavailable=(kind>=1&&kind<=6)||kind==10||kind==11||kind==12||kind==14;
        const bool retry=kind==0||kind==9;
        for(int c=0;c<s->nCells;++c){const auto& q=g.chemistryStatus[c];
            std::printf("FAIL_STATUS kind=%d cell=%d code=%d thermo=%d reaction=%d attempted=%d rejected=%d nonlinear=%d cellStatus=%d expectedNewton=%d\n",
                kind,c,int(q.code),int(q.thermoCode),q.reaction,q.attemptedSteps,q.rejectedSteps,q.nonlinearIterations,g.cellStatus[c],expectedNewton);
            check(q.code==code&&q.thermoCode==thermo&&q.reaction==reaction
                &&q.attemptedSteps==attempted&&q.rejectedSteps==rejected&&q.nonlinearIterations==nonlinear,
                "full failure-specific ChemistryStatus mismatch");
            check(g.cellStatus[c]==int(unavailable?GasTransportCode::InvalidStorage:GasTransportCode::ChemistryFailure),"full failure-specific cell status");
        }
        auto fields=sharedGasTrialDetail::cellFields(s),control=sharedGasTrialDetail::cellFields(reference);
        check(fields.size()==control.size(),"failure cell field count");
        for(std::size_t k=0;k<fields.size();++k)check(fields[k].count==control[k].count
            &&bool(fields[k].pointer)==bool(control[k].pointer)&&(!fields[k].pointer
            ||std::memcmp(fields[k].pointer,control[k].pointer,fields[k].count*sizeof(double))==0),"failure leaked transaction cell/budget state");
        fields=sharedGasTrialDetail::faceFields(s);control=sharedGasTrialDetail::faceFields(reference);
        check(fields.size()==control.size(),"failure face field count");
        for(std::size_t k=0;k<fields.size();++k)check(fields[k].count==control[k].count
            &&bool(fields[k].pointer)==bool(control[k].pointer)&&(!fields[k].pointer
            ||std::memcmp(fields[k].pointer,control[k].pointer,fields[k].count*sizeof(double))==0),"failure leaked transaction face state");
        for(int c=0;c<s->nCells;++c)check(sameAudit(g.chemistryAudit[c],reference->gasSpecies.chemistryAudit[c]),"failure leaked accepted audit");
        compareBudget(s,reference);
        check(SharedGasTrialPolicy::validate(s,true)!=0,"failure transaction validate falsely succeeds");
        check(SharedGasTrialPolicy::retryable(s)==retry&&SharedGasTrialPolicy::continueTerminal(s)==!unavailable,"failure native decision flags");
        const bool savedRetry=s->gasTrial.retryableFailure,savedTerminal=s->gasTrial.terminalContinuation;
        SharedGasTrialPolicy::rollback(s);sync();compare(s,reference);compareBudget(s,reference);
        check(s->gasTrial.retryableFailure==savedRetry&&s->gasTrial.terminalContinuation==savedTerminal
            &&!s->gasModelPoisoned,"rollback changed native decision flags or poisoned owner");
        std::printf("PASS failed-transaction kind=%d bothCellsFullStatus=1 allCellFaceBudgetBytes=1 fullSemanticAudit=1 hostLedger=1 rollbackRestored=1 retry=%d terminal=%d\n",kind,int(savedRetry),int(savedTerminal));
    }
}
'''.replace('PREFIX',prefix)
new=text.replace('void cleanup(){',body+'\nvoid cleanup(){');new=new.replace('chemistryTrialTest::failedCellTests();chemistryTrialTest::cleanup();','chemistryTrialTest::failedCellTests();chemistryTrialTest::failedTransactionTests();chemistryTrialTest::cleanup();')
source=D/'inputs/chemistry-fixture-S301.cu';source.write_text(new);patch=D/'patches/S301-complete-failure-transactions.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/generated-formal-chemistry_trial.cu',tofile='b/generated-observation-chemistry_trial-S301.cu')))
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS301新增支持观察范围（正式原fixture已actualGPU exit0并保留）：源码复核确认原失败矩阵只比较species/accepted审计/chemistrySpecies budget与cell0粗状态，kind7/15原状态被public重放覆盖。增加独立失败Transaction组合，保持所有原16输入/control与4个成功测试；真实begin/kernel/validate/rollback，打印两cell全部6ChemistryStatus成员，全部cell/face transaction arrays、各budget、accepted Audit与host账不泄漏，rollback恢复语义diagnostics但保留供owner读取的retry/terminal decision flags。kind10正确期望StepLimit+reaction0/1attempt/1reject/1nonlinear，不误要求RateFailure；kind0Newton计数由实际原1e-6direct成功控制获取。源/patch/hash单列，不修改生产或正式fixture，不用marker冒充完整语义。compile900s/run300s。\n')
command=json.loads((ROOT/'work/fixtures/chemistry-trial-formal/command.json').read_text());binary=ROOT/'work/fixtures/chemistry-trial-S301';command[-3]=str(source);command[-1]=str(binary)
t=run('BUILD-CHEMISTRY-S301',cmd(*command),[10,12,14,26],900,'Actualnative unchangedproduction sourceplusdistinctfullfailuretransaction support. Formaloriginalruntime retained.',inputs=[source,patch],kind='BUILD',variant='SUPPORT_S301');assert t['exit_code']==0
(D/'inputs/chemistry-fixture-S301-command.json').write_text(json.dumps(command,indent=2)+'\n')
print('S301binary ready',sha(binary),flush=True)
