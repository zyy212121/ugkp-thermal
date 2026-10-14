import json,collections,re,copy
from r2_runner import *
data=json.loads((D/'results.json').read_text());tests=data['tests'];byid={x['id']:x for x in tests};units=[]
def unit(id,verdict,variant,pattern,topics,scope,limitations=''):
 ids=[x['id'] for x in tests if re.search(pattern,x['id'])];units.append({'id':id,'verdict':verdict,'variant':variant,'test_ids':ids,'topics':[f'R014-{n:03d}' for n in topics],'scope':scope,'limitations':limitations,'counts_separately_from_commands':True})
unit('R2-A-owner-full-semantic','PASS','FORMAL_SOURCE_TEST_SUPPORT',r'^CHEM-OWNER-S208$',[10,12,14,26],'RealNs2requested4e-6: StepLimit rollback, fouracceptedparts, exhaustedterminal originalpublicfalse, original1e-6 control; allsemanticfields/audits/status/budget and16failedresource combinations.','S202/S203/S208 fixture support only. Submittedoriginalfixture itself FAIL. Ns2 is speciescount, notclaimallNs10ownerconfigurations.')
unit('R2-A-Ns10-native','PASS','FORMAL',r'^CHMT-chemistry-(full|control)$',[10,24,26],'RealNs10originalfull/control inputs, originalchecker andalloutput bytes equal802; freshfullnativecompile serialretry.','Fullinitcheck300s BLOCKED separate.')
unit('R2-A-submitted-fixture','FAIL','ORIGINAL_TEST_SUPPORT',r'^CHEM-OWNER$|^CHEM-OWNER-S202$',[10,14,26],'Originalmanagedhost owner SIGSEGV; split-onlytest requireswrongtwopartpartition andfailsrealCFL.','TwoP2support findings, no production repair byreviewer.')
unit('R2-B-CHT-original-inputs','PASS','FORMAL',r'^NATIVE-CHT-coldWall|^RESTART-CHT-coldWall',[18,19,23,26,33],'Originalcold1D/cold2D/wet100, alltime1e-9 mean(double)+carry; original2Drestart terminalbytes.','P007/P010 notapplied. Multipleactivegroups/fullgasledger separate.')
unit('R2-B-FSH-double-restart','PASS','FORMAL_TEST_SUPPORT',r'^NATIVE-FSH-coldWall|^RESTART-FSH-coldWall',[18,19,23,26,33],'RealFSHdoublecold1D/2D andmidpointrestart allterminalbytes; S204 samecaloric1e-9 usingactualFSHschema.','FSHhasnoCHTcarryfield; do notinventone.')
unit('R2-B-init-config-copy','PASS','FORMAL',r'^COLD-FIXTURE-|^COLD-ACTUAL-COPY-RETRY$',[12,18,23,26,33],'Fourphases/nonzerocarry,actualconfig-before/afterpatch mixedfaces/toggle preserveuploads; actualCUDAcopyerror initializerretry.','Fixturegroupsinglethreadinitialization, not full8lane dynamics.')
unit('R2-B-16-activegroups','FAIL','FORMAL_NEW_INPUT',r'^NATIVE-.*valid16-S214$|^PROFILE-ALL-PARCELS-.*valid16-S214$',[18,23,26,33],'Actual16particles atall10outputs,correctcellIDs; strictprofile-pT error CHT2.076931505e-4K / FSH5.813134476e-5K exceeds1e-9.','Noactivecold2Dcontactage; inactivecache consumption/initialstate andactivegroupwitness unresolved. S211count1/S213wrongcell supportattempts preserved.')
unit('R2-C-tail-whole-output','PASS','FORMAL',r'^CHMT-thermal-full(-initcheck)?$',[9,12,14,26,28],'RealNs2original100microsteps initcheck0; ALLCSV/checkpoints/serializedgeometry bytesexact802.','No clockobserver; d611notusedasthisbytebaseline.')
unit('R2-C-cache-freshdata','PASS','FORMAL_OPERATOR',r'^CO-(FIXTURE|FRESH-2|FRESH-10|FRESH-10-INITCHECK)$',[9,12,14,26,28],'Actualempty/single/multiblock/maxOR/NaNbitcontrol,168changingarraycases/ns, sameowner growth/reuse andsecondowner streams, realOOM/retry/destruction.','Hostserializedstreams; no sameBackendconcurrenthostthread claim.')
unit('R2-C-Ns10-control-initcheck','PASS','FORMAL',r'^CHMT-chemistry-control-initcheck$',[9,10,26,28],'ActualoriginalNs10control wholecase initcheck0; alloutputbytes802.','Notfullreactinginputsanitizer.')
unit('R2-C-Ns10-full-initcheck','BLOCKED','FORMAL',r'^CHMT-chemistry-full-initcheck$',[9,10,26,28],'OriginalfullNs10input exceeded300sexternalbudget, finalsanitizersummary absent.','PartialcheckerendtimeFAIL retained; no numeric pass/fail inferred fromtimeout.')
unit('R2-C-restart-vs802','PASS','FORMAL_VS_802',r'^THERMAL-802-SAME-RESTART$',[5,9,12,14,26],'Currentand802fromsameactualcheckpoint/input: all10outputfiles byteexact.','Full-vs-restart strictdifference separatelyretained, no newcacheregression shown.')
unit('R2-C-restart-full-bitparity','FAIL','FORMAL',r'^THERMAL-FORMAL-RESTART$',[5,14,26],'Fullrun-vs-midpointrestart gas/exchange/checkpoint notbitexact; accepted/materialexact.','Sameexistingbehavior802. Unexplainedfirstrestart divergence, cannot closebytol.')
unit('R2-D-clock-lineage','PASS','READONLY_DIAGNOSTIC',r'^TRACE-NATIVE-(tested|formal)$|^TRACE-BASELINE-CHECKPOINT-PAYLOAD-PROOF$',[14,15,21,22,26,27],'Actual3version21400hexrecords: firstclockacceptedstep6, stageclockstep7, firstdt/gas/packetstep50/window; wallphysicalallunchanged;802vsformalalltracebitsidentical.','Baselinecheckpoint identity/checksum changes66bytes; allstatepayload/configotherbytes bitidentical. OriginalwholebyteFAILretained. Approvedcompensatedclockpreserved, notd611wholeparityPASS.')
unit('R2-F-SST22-resource','PASS','FORMAL_REAL_ERROR_OBSERVER',r'^SST22-ACTUAL-OWNER-',[4,12,17,18,26,28],'ActualCHT/FSH22canonicalowners, late/reconfigure, realOOMeveryallocation/copyerror,retryonlymissingowner,nativeinit anddestroy.','Allmodelmigration/concurrency/freefailure types notproved.')
unit('R2-F-SST-RK-net-budget','PASS','FORMAL_READONLY_DIAGNOSTIC',r'^SCOPED-MASS-ENERGY-SST-SST-|^ACTUAL-WEIGHTED-LEDGER-SST-',[8,9,14,26,29],'ActualEuler/RK2/RK3 SSTK/omega transport/source/constraintstockclosure,globalM/E andperrecipientM/E; actualfinalflux weights .5/.5 and1/6/1/6/2/3.','Grossproduction/dissipation andcompletebulkmomentum stillopen; strictmomentumred unchanged.')
unit('R2-F-ALE-chemistry-volumes','PASS','FORMAL_READONLY_DIAGNOSTIC',r'^ACTUAL-ALL-MICROGEOMETRY-|^ACTUAL-KERNELS-moving-chemistry-S206$',[8,10,14,26,29],'All100Ns2/700Ns10acceptedmicrosteps bothgeometry stages; GCLmax5.5511e-17m3 beloworiginal1e-14; all1400actualNs10chemkernels old/newvolume arguments bitexact andsamehalfdt.','Observeroutputsalloriginalbytesunchanged, IOexcludedtiming.')
unit('R2-F-wet-formal','FAIL','FORMAL',r'^EXTRA-wet-(film|evaporation)-S209$',[3,6,13,14,20,21,26,29],'ActualvalidwetThicknessAveraged/noMelting nativeSIGFPE; sourceunconditionallyreadscondensed[-1] forzerophase.','Existing802codebug GPU-F006; exactSIGFPEstacknotresolved. No repeatknownOOB.')
unit('R2-F-wet-heat-diagnostic','PASS','DIAGNOSTIC_D204_SCOPED',r'^SCOPED-MASS-ENERGY-SST-wet-film-S209-D204$',[3,6,13,14,20,21,26,29],'Sameallinputparameters withminimalzero-phase hostdiagnostic+unmodifiedGPUobject completeswholeinterval; global/perrecipientM/E stockclosure.','Fullmomentum strictFAIL(1.03e-13) remains. Diagnosticneverformalpass.')
unit('R2-F-wet-evaporation','BLOCKED','DIAGNOSTIC_D204',r'^EXTRA-wet-evaporation-S209-D204$',[3,6,13,14,20,21,26,29],'Sameevaporationinput afterD204 reached300s cap, no acceptedcompletewindow.','Noendpoint/property/precision/tolchange; keeporiginalSIGFPE andtimeout.')
unit('R2-F-dry-ablation-scoped','PASS','FORMAL',r'^SCOPED-MASS-ENERGY-SST-dry-ablation$|^ACTUAL-WEIGHTED-LEDGER-dry-ablation$',[3,6,11,14,26,29],'ActualC0->S0surfacegas/CoupledRecession, fullM/E stock andpairedM/E, actualfinalflux/rawallmicrogeometry.','CompleteP strictred3.58e-13; bulk/pore/meltingcampaign notcovered.')
unit('R2-F-radiation-singlemapper','PASS','FORMAL',r'^NATIVE-CHT-actual-(coupler|radiation)-S210$|^COMPLETE-SOLID-ALLBOUNDARY-',[18,20,23,26,30,33],'ActualCHT GPUresident+realDOsolidcoupler/oneface/mixeddefaults; profiles1e-9; fullallboundarysolidledger max3.64e-13/2.59e-13J.','Empty/multiple-selectedmapper, fullgas/particleinventory stillopen; wall-onlyincompleteFAILpreserved.')
unit('R2-F-host-codec-sanitizer','PASS','FORMAL_HOST_SANITIZER',r'^HOST-DECODE-.*checkpoint-[012](-DIR)?$|^HOST-DECODE-SST-SSPRK[23]-[012]-DIR$',[5,19,26,28],'Actualcurrentcheckpointdecoder ofrealGPUdata underhostASan/UBSan, expandedactualstock/packet fields.','Onlynamedcodec path; not allwire/mapper/resourceownerhosts.')
unit('R2-performance-resource','PASS','PERFORMANCE_QUALIFIED',r'^NSYS-FORMAL$|^TIMING-',[2,9,16,26,27],'Matched100step Nsight allocation339->140;3000kernels unchanged, stable198Co ranges0malloc/free; peaktracked22281B/final0; nonprofiling3warm+7rawsamples retained.','Memorysubset,certainnonmallocmemoryevents unresolved. Median ratio.99785 withlargevariance; no stablewhole-solverspeedup claim.')
gaps=[
 ('SST22-owner-graph','CLOSED_SCOPED','Actual22owners/resourceretry/reconfigure inbothapps; concurrenthost/freefailure coverageelsewhere.'),
 ('SST-RK-budgets','PARTIAL','NetK/omega andactualRKweights/M/E pass; grossmodels andcompleteP notclosed.'),
 ('moving-GCL','CLOSED','Allmicrogeometry GCL and1400realchemicalold/newvolume input pairs qualified.'),
 ('cold-start-Js','NOT_RUN','Dedicatedcold-startJs firststep oracle/historyCo acrossnewflow inputs notexecuted.'),
 ('chem-native-full-retry-owner','CLOSED_SCOPED','RealNs2owner fullrequestedinterval/refinement/terminal/fullsemantic auditS208;Ns10ordinary/ALEpaths separatelycovered.'),
 ('wet-ablation-material','FAIL_OPEN','GPU-F006zerophaseinvalidmaterialaccess, originalwetSIGFPE;diagnosticwetM/Eonly,evap300sblocked;dryM/E/source/geometryscoped; melting/pore remain.'),
 ('CHT-radiation-mapper','PARTIAL','EnabledDO/singleface/mixedandfullsolidledger pass; empty/multiple-selectedmapper notrun.'),
 ('mixed-wall-coupler','CLOSED_SCOPED','Bothorder/toggle/downloadedvalues/actualcopyretry;actualnativeonecoupled+othersuncoupled;full3appgasbudget elsewhere.'),
 ('FSH-cold-profile','PARTIAL','Real1D/2Ddouble andrestartpass;activefullgroups/fullpairedledger notclosed;16inactivecachestrictred retained.'),
 ('host-sanitizers','PARTIAL','ActualcodecASan/UBSan pass;wire/mapper/resourceownerhostsanitizers notrun.'),
 ('allocation-fault-lifecycle','PARTIAL','Actual22slotOOM/copyretry/coldcopy andCoOOM/destructionpass;otherfree/sync/hostfailure/concurrency effectsnotcovered.'),
 ('resident-full-budgets','FAIL_OPEN','Fullthreeapplications stage/final-limitedflux/allsource/M/E/P inventory notcomplete;actualCHMT/SST/filmPstrictreds kept.'),
 ('native-FP64-clock-lineage','CLOSED_SCOPED','Actualfirstclockstep6/firstphysicaldt-gas-packetstep50; baselinepayloadidentityproof;approvedcompensatedclockunchanged. Fullvsrestartfirstdiffunresolved separately.'),
 ('developer-formal-recheck','PARTIAL','FormalF001/F002/F004/F005 originalspecificinputs rechecked;supportfixture2P2 andnewwetP1/multigroup/fullbudget stillopen.')]
gaprows=[{'gap_id':i,'verdict':v,'observations':s,'historical_unit_id':'NOT-RUN-'+i,'historical_fixed_commit':PATHS['old_delivery_commit']} for i,v,s in gaps]
for g in gaprows:
 if g['verdict']=='NOT_RUN':units.append({'id':'R2-NOT-RUN-'+g['gap_id'],'verdict':'NOT_RUN','variant':'NOT_RUN','test_ids':[],'topics':['R014-015','R014-026'],'scope':g['observations']})
data.update(source_tree=PATHS['source_tree'],record_parent=PATHS['delivery_parent'],acceptance_units=units,remaining_gap_status=gaprows,command_counts=dict(collections.Counter(t['verdict'] for t in tests)),acceptance_unit_counts=dict(collections.Counter(u['verdict'] for u in units)),counts_not_additive_across_topics=True,merge_ready=False,merge_authorized=False,finished_utc=utc(),scope='CommandPASS(build/setup/IO)notGPU numericalPASS. Unitsgroupstrictscope, reusedunitsseparate andnotadded. RawFAILS retained evenlatersupportfixes; diagnosticnotformal.')
(D/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
oldcov=json.loads((OLD/'coverage.json').read_text());oldres=json.loads((OLD/'results.json').read_text());changed=set(git('diff','--name-only',PATHS['old_source'],SOURCE,'--','applications','common','tests').splitlines());reuse=[]
for u in oldres.get('acceptance_units',[]):
 if u['id'].startswith('ACCEPT-') and any(x in u['id'] for x in ['-core','-pressure','-collision','CUDA-AUTO','periodic']):
  reuse.append({'unit_id':u['id'],'fixed_commit':PATHS['old_delivery_commit'],'path':'review/pr14/reviewer-deliveries/20261013T182722Z-fast/results.json','sha256':sha(OLD/'results.json'),'scope':u['scope'],'qualification':'Directcore/pressure/drag/pool/RNG successful/inactivecold callchains have no mathchanges in20filediff; originalcorecoldblock wasexcluded. Native cold/chemistry/cache neweffects expresslynotreused; seeindependentsource-review dependencytable. Originaltestsource remains802, notnew4bexecution.','not_counted_as_fresh':True})
(D/'metadata/reused-evidence.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','old_source':PATHS['old_source'],'old_delivery_commit':PATHS['old_delivery_commit'],'changed_production_support_paths':sorted(changed),'scoped_units':reuse,'support_reuse':['P003 originalinput/serialization purpose','P008 physicalinputpurpose','P009 originaldouble-mean checker arithmetic only','P012 CSVquote-all profiler serialization'],'formal_recheck_excluded':['oldP007/P010diagnosticpass','oldP002cellpass','oldP011tailpass'],'policy':'Unchangeddirectcallchain GPUevidence reusedwithinoriginalscope, neverwholetranslationunit/newrun/all33approval.'},ensure_ascii=False,indent=2)+'\n')
topics=[]
for oldtopic in oldcov['topics']:
 tid=oldtopic['topic_id'];active=[u for u in units if tid in u['topics']];ran=[t for t in tests if tid in t.get('topics',[])];remain=[g for g in gaprows if g['verdict'] not in ['CLOSED','CLOSED_SCOPED'] and any(tid in u['topics'] for u in active)]
 q={'topic_id':tid,'title':oldtopic['title'],'read':True,'source_commit':SOURCE,'source_tree':PATHS['source_tree'],'file_ids':oldtopic.get('file_ids',[]),'targets':oldtopic.get('targets',[]),'current_source_targets':'metadata/source-hashes.json andpertest.spec.json source_targets; historicalfunctionlocators notsilentlyrewritten','executed_test_ids':[x['id'] for x in ran],'acceptance_unit_ids':[x['id'] for x in active],'verdict':'PARTIAL_RECHECK_OPEN_GAPS' if remain or any(u['verdict'] in ['FAIL','BLOCKED'] for u in active) else 'PASS_SCOPED_RECHECK' if active else 'REUSED_OR_ADMINISTRATIVE_SCOPE','remaining_gaps':[g['gap_id'] for g in remain],'reused_unit_ids':[x['unit_id'] for x in reuse if tid in next(u for u in oldres['acceptance_units'] if u['id']==x['unit_id']).get('topics',[])],'not_full_topic_or_merge_approval':True};topics.append(q)
coverage={'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'source_tree':PATHS['source_tree'],'topics':topics,'remaining_gaps':gaprows,'acceptance_unit_counts':data['acceptance_unit_counts'],'reuse_file':'metadata/reused-evidence.json','counts_not_additive_across_topics':True,'merge_ready':False};(D/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
oldfind=json.loads((OLD/'findings.json').read_text());findings=[]
status={'GPU-F001':('FORMAL_RECHECK_PASS_SCOPED','S208allsemanticaudit/fullowner; submittedtest2P2 stillopen'), 'GPU-F002':('FORMAL_ORIGINAL_RECHECK_PASS_SCOPED','Originalcold inputs/config/actualcopyretry andsinglecoupler; fullgasledger elsewhere'), 'GPU-F004':('FORMAL_ORIGINAL_RECHECK_PASS_SCOPED','Originalalltime1e-9/restart/fourphase/nonzerocarry; activefullgroups notclosed'), 'GPU-F005':('FORMAL_RECHECK_PASS_SCOPED','Ns2thermal100 initcheck0/allbytes802; freshNs2/Ns10cachetests; fullNs10sanitizer300sblocked'), 'GPU-F003':('EXPLAINED_APPROVED_CLOCK_CHANGE','Actualfirstclockstep6/stage7/firstdtphysical50;802/formalalltracebitsidentical; no restoration/tolclosure')}
for f in oldfind['findings']:
 q=copy.deepcopy(f);q.update(recheck_source=SOURCE,recheck_status=status[q['id']][0],recheck_observations=status[q['id']][1],old_original_evidence_fixed_commit=PATHS['old_delivery_commit'],old_original_findings_sha256=sha(OLD/'findings.json'));findings.append(q)
findings.extend([
 {'id':'GPU-F006','priority':'P1','classification':'EXISTING_PRODUCTION_DEFECT_NEWLY_EXERCISED','title':'Zero phase transfer reads disabled condensed material index -1','file':'applications/CHMT/materials/CpuMaterialDriver.C','line':461,'source_commit':SOURCE,'same_code_in802':True,'trigger':'Supportedwet ThicknessAveraged, noMelting, defaultphaseCondensed=-1; heat-onlySolidFilmphase.mass0 stillunconditionallydivides bycondensed[-1].rho.','observed':'Formalwet heatand evaporation SIGFPE(-8); originallogs/initialcheckpoints retained. Staticinvalidaccessconfirmed; exactsignalPCnotresolved. KnownOOBnotreexecuted.','diagnostic':'D204zero-massvolume0 bypassunselectedmaterial; originalGPUobjectunchanged. Sameheatinputwholecompletion/M/Escopedpass, fullPstrictred;evap300sblocked.','status':'OPEN_DEVELOPER_FORMAL_REPAIR_REQUIRED','topics':[3,6,13,14,20,21,26,29],'evidence':['metadata/independent-wet-review.md','logs/EXTRA-wet-film-S209.log','logs/EXTRA-wet-evaporation-S209.log','patches/D204-zero-phase-does-not-consume-disabled-material.patch']},
 {'id':'TEST-R2-S1','priority':'P2','classification':'SUBMITTED_TEST_SUPPORT_DEFECT','title':'Chemistryfixture managedhost/deviceowner fails asynchronoushostread onWSL','file':'tests/fixtures/chemistry_trial/chemistry_trial.cu.in','line':19,'status':'OPEN_DEVELOPER_SUPPORT_REPAIR','topics':[10,14,26],'evidence':['metadata/independent-source-review.md','logs/CHEM-OWNER.log','patches/S202-fixture-host-device-owner-split.patch']},
 {'id':'TEST-R2-S2','priority':'P2','classification':'SUBMITTED_TEST_SUPPORT_DEFECT','title':'Hardcodedtwo acceptedchemicalparts conflictswithrealCFL','file':'tests/fixtures/chemistry_trial/chemistry_trial.cu.in','line':126,'status':'OPEN_DEVELOPER_SUPPORT_REPAIR','topics':[10,14,26],'evidence':['metadata/independent-source-review.md','logs/CHEM-OWNER-S202.log','logs/CHEM-OWNER-S208.log']},
 {'id':'OBS-R2-P','classification':'UNRESOLVED_STRICT_BUDGET_OBSERVATION','title':'CompleteCHMT/SST/film momentumcriterion unresolved','status':'OPEN_NOT_CONFIRMED_NEW_PRODUCTION_BUG','observed':'SSTgasP residual1.30e-12..8.03e-12; dry3.58e-13; diagnosticperFilmP1.03e-13. Packetmass-rule testscaling used; originalM/E/SST pass, strictcombinedFAIL kept.','topics':[8,9,14,26,29]},
 {'id':'OBS-R2-COLD16','classification':'UNRESOLVED_NEW_INPUT_SCOPE','title':'16inactivecoldcache profile/currentpT strictmismatch','status':'OPEN_ACTIVE_FULLGROUP_WITNESS_REQUIRED','observed':'All16survivecorrectcellinput, cold2Dagesall0; CHT2.076931505e-4K/FSH5.813134476e-5K >1e-9. Inactivecache/initialrecord purpose unresolved; no finite/tolshortcut.','topics':[18,23,26,33]},
 {'id':'OBS-R2-RESTART','classification':'EXISTING_STRICT_RESTART_DIFFERENCE','title':'Fullvsrestart notbitexact, formalvs802restart exact','status':'OPEN_FIRST_RESTART_DIVERGENCE','topics':[5,14,26],'evidence':['artifacts/chmt/formal-restart-terminal-comparison.json','artifacts/chmt/old802-vs-formal-restart.json']}
]);(D/'findings.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'findings':findings,'merge_ready':False},ensure_ascii=False,indent=2)+'\n')
counts=data['command_counts'];timing=json.loads((D/'artifacts/timing/summary.json').read_text())
report=f'''# PR14 第二轮 FAST 独立真实GPU报告

PR14_REVIEW_ONLY。**本轮不满足整体 merge-ready；不合并。** 原F001/F002/F004/F005的指定正式输入已定向复验，F003实际位差链条已定位；新发现既有wet零相变材料索引缺陷GPU-F006(P1)、两个提交夹具支持缺陷(P2)，多组动态cold/完整动量及其余缺口仍保留。诊断通过不是正式提交通过。

## 固定身份与范围

正式生产修复 `{SOURCE}`，tree `{PATHS['source_tree']}`；本轮起点/开发者交付记录 `{PATHS['delivery_parent']}`。本报告所在提交由最终PR提交历史确定，不把commit写入自身。旧被测 `{PATHS['old_source']}`、旧证据 `{PATHS['old_delivery_commit']}`；原clock baseline `{PATHS['baseline']}`。新目录 `review/pr14/reviewer-deliveries/{D.name}/`。

实际RTX4060 Laptop8GB/sm89、WSL Ubuntu22.04/OpenFOAM10、CUDA13.1、g++11.4。正式Ns2 CHMT/CHT/FSH及Ns10CHMT均完整原生构建和实际GPU运行；Ns10首次编译900s超时保留，计划追加serial1800s后实际成功。正式库/正式安装/生产结果/原始断点未修改；所有工作在授权test工作树。源、环境、object/binary hashes见SOURCE、versions、environment与metadata/source-hashes。

已读取开发者REPORT/SOURCE/VALIDATION/CLOCK_ANALYSIS、旧REPORT/coverage/results/findings、20文件源差分及各调用链、PROTOCOL；先行TEST_PLAN含复用表与所有范围扩展。独立源码席位仅只读审查，报告在metadata；本席位执行实际GPU。源审查不冒充运行验收。死机后114个完整执行记录日志hash全一致，无孤立spec；只做恢复GPU健康观察，未重跑整套，死机原因未知。

## 正式定向复验

|项目|实际结果/限定|
|---|---|
|F001/native Trial|S208真实requested4e-6先StepLimit后缩步：begin8/commit4/rollback4，四个约1e-6acceptedpart精确覆盖原区间；effort耗尽publicterminal对照完成。全部有语义state/audit/status/budget逐字段/位检查，原1e-6native/public成功对照与16backing/metadata/maxSteps1组合保留。S202/S203/S208只修测试host/device布局、前置Co与观察，未改生产CUDA。提交原fixture仍红；默认targetMaxCo实际0.5，原说明0已明确更正。|
|Ns10|正式原chemistry-full/control实际GPU+原checker，全部CSV/checkpoint bytes等于802；700微步非均匀ALE化学另有1400启动参数证据。full initcheck300s超时；原control17.91s initcheck0，不替代full sanitizer。|
|F002/F004|正式原cold2D、1Dwet100、2Drestart，不叠P007/P010；全部写时doublemean+doublecarry与pT按原1e-9K；真实FSHdouble1D/2D和restart均完成。四初始化温区/非零carry、configure/patch双次序、mixed/toggle/真实copyerror重试均实际CUDA。7900/500/15旧CHTtoken仍不消费，默认1800/710/100；FSH有效token单列。|
|多完整cold组|新S214正确16parcel/cell数据每时刻保留16，但无activecold2Dage，strictprofile/currentpT误差CHT2.076931505e-4K、FSH5.813134476e-5K超过1e-9。保留红，未据inactivecache数据关闭动态整组/完整pairedgas账。S211被旧初始文件覆盖为1、S213绑定错误为3均保留支持范围。|
|F005/Co|正式Ns2原thermal100microsteps initcheck0，所有原CSV/checkpoint/内部geometry bytes等于802(不是d611/clockdiag)。实际空/1/31/32/33/513/32768及NaN/max/OR，freshdata同owner容量cycle/第二owner，Ns2/Ns10 bitexact控制；真实OOM/retry和销毁。六个Co前处理保留。|
|restart/cache|正式与802从同本轮中点checkpoint恢复，全部10对文件byteexact；formal-full与formal-restart仍有gas/exchange/checkpoint严格差，accepted/material相同。说明不是新cache回归，第一处restart原因仍待定位，不按容差关闭。|

## actual clock/ALE/SST/资源

三版真实clock各21400hex记录。baseline与正式首次时钟差在window1 accepted微步6：`0x1.f75104d551d69p-15` → `0x1.f75104d551d6ap-15`；stage/sample时钟自7步不同；物理壁面值全部相同。首次dt、gas momentum、packet差在每window第50微步；802与正式全部21400trace bits相同。与旧double累加/补偿累计末步dt解释相符，用户已批准的补偿算式保持。未用容差把严格跨版差变为PASS。baseline observer三个checkpoint仅编译sourceFingerprint及checksum共66bytes不同；物理state payload及其它config/header bytes全一致，两checksum合法；原whole-byte FAIL保留。其它两版observer所有原输出bytes相同。

真实非均匀moving-GCL100微步、movingchemistry700微步全部两阶段GCL最大5.551115123125783e-17m3，小于原absoluteGeometry1e-14；所有1400Ns10实际kernel old/newvolume args逐位对应，half dt一致；诊断原输出bytes不变。旧CUDA入口观察器零记录FAIL保留，S206使用真实CUDA13 kernel handle接口纠正支持，额外D2H/IO不计效率。

实际CHT/FSH SST22 late/reconfigure/all22allocation realOOM/copyerror/retry/native初始化/销毁通过。SST的Euler/RK2/RK3 K/omega实际transport/source/constraint库存闭合；RK实际最终face/packet阶段权重分别1/2+1/2、1/6+1/6+2/3，原输出bytes相同。SST K残差最大约4.93e-16、omega约1.04e-14；质量/能量/recipient M/E通过。完整bulk-P严格核算残差约1.30e-12..8.03e-12未关闭，diagnosticfilm recipient-P约1.03e-13。具体逐点数据、原阈值、通过项和红分开，未改求解器门禁。

enabled Mie DO真实CHT/solid耦合与disabled对照完成，单coupled面+其它uncoupled实际场写出/profile通过。完整固体账累计全部GPU gas/particle/radiation墙热及实际sideWalls300K外部导热，最大残差3.639908641e-13/2.588730305e-13J，原coupler abs/rel1e-8通过。最初wall-only差5.6e-4J漏了约56W外部导热，原不完整账FAIL未删。此结果不是三个resident应用完整gas/particle/momentum库存通过。

## 新red与诊断

GPU-F006(P1)：CpuMaterialDriver.C:461对zero-phase热包无条件读`condensed[phaseCondensed].rho`，合法ThicknessAveraged/noMelting默认index=-1。wet heat/evap原版均SIGFPE(-8)，静态非法访问确认，精确信号PC尚无栈证明；已知OOB不重跑取数。源与802相同，属于既有缺陷新覆盖。D204仅test内zero-mass体积0不消费unselected backing，保留所有原失败、patch/source/originalGPUobject/binary hashes；heat-only同输入全区间完成、global/perrecipient M/E闭合，P仍严格红；evap同输入300s外部timeout。没有开启melting/改终点、物性、精度或阈值换绿。正式修复须开发者。

两个P2提交支持问题：managed host/device owner在本WSL异步host读取SIGSEGV；fixture强制两acceptedpart与真实CFL冲突。S208补全部semantic audit比较、原支持红保留。新S201–S214支持、D201–D204诊断各有源/补丁hash；旧P003/P008/P009/P012仅按原用途复用，旧诊断不算正式修复。

## 实际效率数据

同原thermal100步flags，Nsight相对802：cudaMalloc/free **339→140各**(各少199)，kernel **3000→3000**，memcpy11702、async200、streamSync400、deviceSync400均未减少。200Co归约NVTX ranges：首次transform有1malloc，后198稳定ranges **0malloc/free**。只跟踪cudaMalloc的peak live **22281B**、销毁后0，某些其它memory事件未全部解析，不能当全GPU/context峰值。所有原输出bytes等于802；六前处理kernel未删。

非profiling3warmup+7alternating样本全部保留、所有输出bytes等于802。802 median1.924639985s(range1.870003..2.229358,stdev0.12648)，formal median1.920505083s(range1.769701..3.424826,stdev0.59848)，ratio {timing['ratio_formal_over802']:.8f}；**不能确认稳定整求解器加速**。不从kernel/allocator计数外推整solver。

## 覆盖/复用/尚未关闭

本轮{len(tests)}条已执行记录，command状态{counts}；含编译/setup/环境/分析/支持红，**不当GPU数值PASS计数**。限定验收单元{len(units)}，状态{data['acceptance_unit_counts']}；跨topic引用不叠加、旧复用不加新run数量。33topic矩阵和14旧gap ID各逐项见coverage/results。没有wholePR全功能PASS声明。

chem owner、全部movingGCL/化学体积、actualclock、SST22及mixed配置/实际singlecoupler已有限定闭合；其余仍需：GPU-F006正式修复、化学fixture2P2维护、active完整cold组/FSH完整paired账、cold-startJs/historyCo新输入、empty/multiple-selectedmapper、grossSST模型/完整bulk-P、wet/evap/pore/melting/全库存、三resident全部stage/最终限幅flux/source/M/E/P账、codec之外hostASan/UBSan及其它实际free/sync/host资源恢复。Ns10full sanitizer和evap诊断分别300s阻断；未改变通过标准。

## 交付与审计

SOURCE/TEST_PLAN/coverage/results/findings/commands、environment/versions、完整原始logs/spec、必要新input/逐点data、全部support/diagnosticpatch、SHA256SUMS与复现入口齐备。已有不可变内容只引用固定commit/hash，见metadata/immutable-references；新执行provenance仍由独立新log/spec记录。原版/诊断/未执行/阻断/失败分开，正式生产源码无diff。

按PROTOCOL追加相关R014 reviewer/recheck history、manifest owned_paths/evidence；行政validator结果在logs。正常push前再次fetch核远端，若变化整合并保留他人提交；不force、不写PR Conversation/描述、不改README、不合并。最终record commit/remote verification由PR历史及本地publication-result给出。
'''
(D/'REPORT.md').write_text(report)
reproduce=f'''# PR14_REVIEW_ONLY FAST reproduction

Read SOURCE/TEST_PLAN/REPORT and exact percommand logs/*.spec.json first. Fixedproduction {SOURCE}; oldevidence {PATHS['old_delivery_commit']}; recordparent {PATHS['delivery_parent']}. Allcommands actuallyexecuted in commands.txt. Source/base/variants/flags/binary hashes in eachspec andversions; rawlogs remainraw. Do notreusean ID/overwriteoriginalFAIL. No productionlongrun ormerge.

Use theuserauthorized test directory, freshisolatedcheckout andinstallation; source inputs/env.sh forOpenFOAM10/CUDA13.1/sm89. r2_paths.json contains actual paths used, adjust thoseonly in anewdelivery foranothermachine. Build originalfullapps/Ns10 from fixedproduction. Most helpers requirepriorprepared work/cases andoldimmutable evidence paths; executein TEST_PLAN dependencyorder, notall r2_*.py wholesale. Setup/publication scripts are records, notauthoritytochange formalinstallation.

Main execution scripts: r2_runner(nativebuild), r2_native(originalCHMT/CHT), r2_fsh/r2_fsh_recheck, r2_fixture_builds, r2_owner_complete/S208 r2_owner_audit, r2_co_prepare/r2_more_gpu, r2_clock/r2_analyze_clock/r2_finish_analysis, r2_thin_prepare/build/r2_f_initial, r2_observer_prepare/r2_more_gpu, r2_gcl_prepare/run, r2_ledger_prepare/run, r2_f_prepare/run/repairedinputs/r2_wet_diagnostic, actualASan reader/S207, r2_group_audit and scopedbudgetanalyses. Numericchecks operate actualnativeproducedfields, neverCPUmock. Current portablehelpers retain exactmath; some auditoutercommands computeinside driverandappendreport hashes, so consult driver-scripts andpointJSON, notjust captured placeholderprint. SpeciesNs2/10 andfixturepredicate/hashes explicit.

S202/S203/S208 are fixtureonly; D204 is diagnosishost source only(originalGPUobjectsameSHA). Preserve originalfixture/wetred anddo notpromote diagnosis toformalpass. Added allgeometry/ledger/clock observers are IOonly andexcludedtiming. Reconstruct removedidenticalfiles from metadata/immutable-references.json beforeusingdirectpath helpers; verify SHA. Any referencedlarge nsys-rep/sqlite/binary remainsin authorizedtestwork path with SHA; leanCSV/pointdata contains emittedproof. Newworkflowinstances mustrecord freshcommands/inputs/hashes/output andbudget.
''';(D/'REPRODUCE.md').write_text(reproduce)
print('WRITE_REPORT',len(tests),len(units),data['acceptance_unit_counts'],len(reuse),flush=True)
