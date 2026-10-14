from r4_runner import *
import difflib
formal=D/'fixtures/chemistry-formal/chemistry_trial.cu';old=formal.read_text();s=old
anchor='    check(a.size()==b.size(),"accepted cell field count differs");'
diag='''
    for(std::size_t k=0;k<a.size()&&k<b.size();++k)if(a[k].pointer&&b[k].pointer){
        for(std::size_t c=0;c<a[k].count&&c<b[k].count;++c)if(!sameBits(a[k].pointer[c],b[k].pointer[c])){
            std::printf("S403_FIELD_DIFFER index=%zu element=%zu actual=%a reference=%a actualCoefficientCount=%d referenceCoefficientCount=%d actualMaximumRejected=%d referenceMaximumRejected=%d\\n",k,c,a[k].pointer[c],b[k].pointer[c],left->gasSpecies.thermo.coefficientCount,right->gasSpecies.thermo.coefficientCount,left->gasSpecies.chemistryControls.maximumRejectedSteps,right->gasSpecies.chemistryControls.maximumRejectedSteps);
        }
    }
'''
assert s.count(anchor)==1;s=s.replace(anchor,anchor+diag)
# Extend physical test inventories with aggregate volume and vector momentum.
anchor='    values.push_back(mass);values.push_back(energy);'
extra='''    double sumVolume=0.,sumPx=0.,sumPy=0.,sumPz=0.;
    for(int c=0;c<s->nCells;++c){sumVolume+=s->V[c];sumPx+=s->rhoUx[c]*s->V[c];sumPy+=s->rhoUy[c]*s->V[c];sumPz+=s->rhoUz[c]*s->V[c];}
    values.push_back(sumVolume);values.push_back(sumPx);values.push_back(sumPy);values.push_back(sumPz);
'''
assert s.count(anchor)==1;s=s.replace(anchor,extra+anchor)
begin=s.index('        if(kind==7||kind==15){check(')
end=s.index('\n    }\n}\n// Formalize',begin)
original_block=s[begin:end]
replacement='''        if(kind==7||kind==15){
            check(g.chemistryStatus[0].code==ChemistryCode::InvalidModel,"thermo model status lost");
            ugkwp::ChemistryStatus originalStatus[2]={g.chemistryStatus[0],g.chemistryStatus[1]};
            const auto originalFailedInventory=physicalInventory(s);
            auto* replayActual=fixture();auto* replayReference=fixture();
            // Match the original metadata/control input on independent initial states.
            // Count6 valid-model arithmetic is a different public-path oracle.
            for(auto* peer:{replayActual,replayReference}){
                peer->gasSpecies.chemistryControls.maximumRejectedSteps=0;
                peer->gasSpecies.thermo.coefficientCount=1;
                if(kind==15)peer->gasSpecies.mechanism.stoichiometricBasis=nullptr;
                cudaCheck(cudaMemcpy(peer->deviceState,peer,sizeof(DeviceState),cudaMemcpyHostToDevice));
            }
            if(!supportMode){
                check(SharedGasTrialPolicy::resetErrors(s)==0,"reset metadata control");
                advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,s->V,false);sync();
                auto* unequalReference=fixture();
                advanceGasChemistryKernel<<<1,32>>>(unequalReference->deviceState,interval,unequalReference->V,false);sync();
                comparePhysicalFields(s,unequalReference);compareInventory(s,unequalReference);
            }else{
                for(auto* peer:{replayActual,replayReference}){
                    startReference(peer,interval,2.);
                    check(advanceGasTrial<SharedGasTrialPolicy>(peer,interval,2.,false)==0,"independent original public interval failed");
                    finishReference(peer);sync();
                }
                compare(replayActual,replayReference);compareInventory(replayActual,replayReference);
                for(int c=0;c<2;++c){const auto& x=g.chemistryStatus[c];const auto& y=originalStatus[c];
                    check(x.code==y.code&&x.thermoCode==y.thermoCode&&x.reaction==y.reaction&&x.attemptedSteps==y.attemptedSteps&&x.rejectedSteps==y.rejectedSteps,"independent replay overwrote original failure");}
                const auto afterReplay=physicalInventory(s);
                check(originalFailedInventory.size()==afterReplay.size()&&std::memcmp(originalFailedInventory.data(),afterReplay.data(),afterReplay.size()*sizeof(double))==0,"independent replay changed original failed state");
                int finiteCells=0;for(int c=0;c<2;++c)if(std::isfinite(replayActual->rhoE[c])&&std::isfinite(replayActual->gasSpecies.rho[c]))++finiteCells;
                std::printf("PASS public-replay kind=%d independentFullPublicTrial=1 originalFailurePreserved=1 matchingCoefficientCount=1 matchingMaximumRejected=0 finiteCells=%d physicalScope=pathological_metadata_completion_not_valid_conservation\\n",kind,finiteCells);
            }
        }'''
s=s[:begin]+replacement+s[end:]
anchor='using namespace ugkwp;';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\nbool supportMode=false;')
anchor='        SharedGasTrialPolicy::rollback(s);sync();compare(s,reference);compareInventory(s,reference);'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n        check(s->gasTrial.trial.valid&&s->gasTrial.interval.valid,"rollback snapshot semantic validity differs");\n        std::printf("S403_TRANSACTION_FLAGS kind=%d trialValid=%d intervalValid=%d retry=%d terminal=%d poisoned=%d\\n",kind,int(s->gasTrial.trial.valid),int(s->gasTrial.interval.valid),int(s->gasTrial.retryableFailure),int(s->gasTrial.terminalContinuation),int(s->gasModelPoisoned));')
anchor='int main(){setvbuf(stdout,nullptr,_IONBF,0);chemistryTrialTest::ownerTests();'
assert s.count(anchor)==1;s=s.replace(anchor,'int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);chemistryTrialTest::supportMode=argc>1&&std::strcmp(argv[1],"support")==0;chemistryTrialTest::ownerTests();')
cu=D/'inputs/chemistry-S403-independent-public.cu';cu.write_text(s)
patch=D/'patches/S403-independent-public-reference-and-aggregate-inventory.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/formal-generated-chemistry_trial.cu',tofile='b/test-only/chemistry-S403-independent-public.cu')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS403 formal原O3/GPU和memcheck均kind7物理field比较exit11（memcheck0errors但不是PASS）。源码跟踪：原faileds保留coefficientCount1/maxRejected0，而旧freshoracle是count6/defaultRejected100；其真实public caloric/rate路径不同，不能按相同数值预期。只改生成的测试sidecar：diagnose模式继续原不等输入对照并打印fieldhex；support模式保留16原失败和四原控制，kind7/15独立freshactual/freshreference均保持各自原metadata/control/interval，执行beginInterval→完整advanceGasTrial(false)→commitInterval且保留originalfailure。补聚合V/P3与snapshot-valid语义，不改生产/kernel/原成功阈值。病态metadata public路径中的NaN/diagnostic传播不按有效case守恒拒绝，也不称有限正确物理结果；支持绿不能变成正式夹具绿。\n')
command=json.loads((D/'fixtures/chemistry-formal/command.json').read_text());b=ROOT/'work/fixtures/chemistry-S403';b.parent.mkdir(parents=True,exist_ok=True);command[-3]=str(cu);command[-1]=str(b)
(D/'inputs/chemistry-S403-command.json').write_text(json.dumps(command,indent=2)+'\n')
t=run('BUILD-CHEMISTRY-S403',cmd(*command),[10,12,14,26],expected='SameformalO3/fmadtrue kernels/owner onfixedsource, onlytest-side diagnostic/reference/aggregateinventory extension; formalred intact.',inputs=[cu,patch],kind='BUILD',variant='SUPPORT_S403')
assert t['exit_code']==0
amend(t['id'],binary=str(b),binary_sha256=sha(b))
for name,args,kind in [('CHEMISTRY-S403-DIAGNOSE',cmd(b,'diagnose'),'DIAGNOSTIC'),('CHEMISTRY-S403-SUPPORT',cmd(b,'support'),'GPU'),('CHEMISTRY-S403-MEMCHECK',cmd('compute-sanitizer','--tool','memcheck','--error-exitcode','99',b,'support'),'SANITIZER')]:
 q=run(name,args,[10,12,14,26],expected='Diagnose original unequaloracle field; support16cell/16transaction32status2independentpublicTrial4originalcontrols actualassertions; memcheck0notgeneralrecovery.',inputs=[cu,patch],binary=b,variant='DIAGNOSTIC_S403' if 'DIAGNOSE' in name else 'SUPPORT_S403',kind=kind)
 raw=(D/q['log']).read_text();counts={k:raw.count(k) for k in ['PASS failed-cell kind=','PASS failed-transaction kind=','FAIL_STATUS kind=','PASS public-replay kind=','S403_TRANSACTION_FLAGS kind=']}
 if 'DIAGNOSE' in name:
  amend(q['id'],verdict='FAIL',observations='Originalsameunequalpublicreference failure retained withactualfieldhex diagnostics.',diagnostic_field_differences=[l for l in raw.splitlines() if l.startswith('S403_FIELD_DIFFER')]);print('DIAGNOSIS',[l for l in raw.splitlines() if l.startswith('S403_FIELD_DIFFER')],flush=True)
 else:
  ok=q['exit_code']==0 and list(counts.values())==[16,16,32,2,16]
  if kind=='SANITIZER':ok=ok and 'ERROR SUMMARY: 0 errors' in raw
  assertion(q,ok,'Support-onlyactualGPUassertions/matrix='+json.dumps(counts)+'; pathologicalmetadata routes maypublishnonfinite values, valid4controlphysicalcomparison separate, formalfixture remainsFAIL.',marker_counts=counts)
  print('S403_RESULT',name,ok,counts,flush=True)
