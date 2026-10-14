from r3_runner import *
import re
src=json.loads((D/'metadata/source-hashes.json').read_text());oldsrc={q['path']:q for q in json.loads((OLD/'metadata/source-hashes.json').read_text())};changed=[q['path'] for q in src if q['path'] not in oldsrc or q['sha256']!=oldsrc[q['path']]['sha256']]
expected={'applications/CHMT/materials/CpuMaterialDriver.C','tests/fixtures/chemistry_trial/chemistry_trial.cu.in','tests/fixtures/chmt_zero_phase_material/probe.C','tests/test_chemistry_trial_cuda_fixture.py','tests/test_chmt_zero_phase_material.py'};assert set(changed)==expected,changed
dependencies=[q for q in src if q['path'].startswith(('applications/CHMT/gpu/','common/'))];assert all(q['path'] in oldsrc and q['sha256']==oldsrc[q['path']]['sha256'] for q in dependencies)
older=json.loads((OLD/'results.json').read_text());units=older['acceptance_units']
rerun={'R2-A-submitted-fixture','R2-F-wet-formal','R2-F-wet-heat-diagnostic','R2-F-wet-evaporation','R2-C-Ns10-full-initcheck'}
reused=[q for q in units if q['verdict']=='PASS' and q['id'] not in rerun]
reuse={'identity':'PR14_REVIEW_ONLY','fixed_commit':PATHS['old_delivery_commit'],'path':str((OLD/'results.json').relative_to(OLDROOT)),'sha256':sha(OLD/'results.json'),'units':[{'unit_id':q['id'],'scope':q['scope'],'limitations':q.get('limitations'),'verdict':q['verdict'],'not_a_fresh_run':True} for q in reused],'rerun_original_ids':sorted(rerun),'unchanged_direct_source_hashes':dependencies,'inherited_round1_reuse':{'path':str((OLD/'metadata/reused-evidence.json').relative_to(OLDROOT)),'sha256':sha(OLD/'metadata/reused-evidence.json'),'fixed_commit':PATHS['old_delivery_commit']},'limits':'Prior source4b5c480b and actual binaries remain their original identities; exact unaffected math/consumer scopes only. New wet/nonzero branches and new inputs require fresh GPU observations.'}
(D/'metadata/evidence-reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
priorinputs={}
for tid in ['EXTRA-wet-film-S209','EXTRA-wet-evaporation-S209-D204','CHMT-chemistry-full-initcheck']:
 q=next(x for x in older['tests'] if x['id']==tid);priorinputs[tid]={k:q[k] for k in ['argv','input_hashes','binary_sha256','verdict','budget_seconds','log','log_sha256']};priorinputs[tid].update(fixed_commit=PATHS['old_delivery_commit'],result_file_sha256=sha(OLD/'results.json'))
(D/'metadata/prior-required-input-hashes.json').write_text(json.dumps(priorinputs,indent=2)+'\n')
plan=f'''# PR14 第三轮独立真实 GPU FAST 计划

PR14_REVIEW_ONLY。正式 source `{SOURCE}` / tree `{PATHS['source_tree']}`；记录起点 `{PATHS['delivery_parent']}`；上轮证据 `{PATHS['old_delivery_commit']}`，上轮被测 source `{PATHS['old_source']}`。不得合并、force push、改 README/PR Conversation/描述；正式生产源由开发者修复，审查诊断仅隔离在授权 test 内。

已完整读取本轮 REPORT/SOURCE/VALIDATION/EVIDENCE_REUSE/GAP_RECHECK_PLAN、PROTOCOL、上轮报告/findings/coverage/results 及五文件真实差分。本轮 source/dependency/blob/SHA256 全表 metadata/source-hashes.json；复用单位 ID 与原适用范围 metadata/evidence-reuse.json；原失败/两个300s timeout 永久留在固定旧证据。所有范围决策在 GPU 执行前记录，不重跑全套。

## 实际依赖与复用

生产仅 CpuMaterialDriver phase-volume consumer 一处改变；另外四文件为正式测试支持。CHMT CUDA/common 全依赖与旧4b逐路径/blob/hash相同；旧 Ns2/Ns10 Backend.o 在旧版本 versions.json 有实际哈希，本轮再次核验。重生成原 build_info（元数据），用正式全部 host 源编译并最终链接未变 CUDA 对象；编译不是新数值通过。Ns10机制/owner/cold/Co/clock算法未变。CHT/FSH原native binaries可在其完整源码/common哈希相同后复用至 test 内，不宣称重新编译。

|单位或原gap|本轮处理与证据边界|
|---|---|
|R2-A-owner-full-semantic|复用S208原生产链；正式fixture独立生成/nvcc/真实GPU，不能借历史诊断关正式fixture|
|R2-A-submitted-fixture / TEST-R2-S1/S2|原红永久保留；正式fixture不叠S202/S203/S208，16组合、真实Co 0.5/native分段/requested与direct成功、publicfalse、terminal、全部语义字段及host账|
|R2-F-wet-formal / GPU-F006|原wet-film-S209同31实际文件+全部mesh，同0.001终点/100微步设置/noMelting，正式源复验；零质量必须保留热、力与forcing；完整M/E/P和逐recipient账|
|wet restart/continue|本轮正式原wet完整运行checkpoint1恢复后半区间；真实consumer状态/库存/输出逐字段、逐位，保留第一处差异而非有限性绿|
|合法非零phase|以真实有效材料/melting或solidification小输入覆盖该表达式，两个方向各记录；旧4b安全合法控制与本轮同输入/flags比较，保留phase.mass、rho/porosity、volume、heat/force/film、交换与最终库存；不用非法index探测|
|R2-F-wet-evaporation|原D204300s阻断永久保留，正式版同原evap输入900s新尝试，区分进度/异常/资源，未达完整终点不PASS|
|R2-C-Ns10-full-initcheck|原300s阻断保留；同原全reacting输入完整0.0007终点（实际controlDict另核）/原检查/原initcheck flags，1800s新尝试；必须有最终summary0和原checker，control不代替|
|R2-B-*原cold / Co / SST22 / GCL / clock / single radiation / performance|按逐条原范围复用；生产依赖不变。cold整组、mapper其它形状、全部库存、资源家族不能靠复用扩大|
|OBS-R2-COLD16|原S214inactive cache严格红保留；从权威SCHEMA3 ASCII mobile t0入射记录复制合法16个原parcel，实际puz=-5/z5e-7，正确cell/orig_id、x/y，其余原量保留；须真实contact、动态age、两个完整8lane组、active profile doublemean+carry1e-9|
|OBS-R2-RESTART|原thermal8cells/100步及checkpoint1只读首差观察；待验证warm/fresh Tgas seed假设，不修改Tgas或checkpoint语义|
|OBS-R2-P / SST-RK-budgets / resident-full-budgets|先独立复核已有逐阶段finalflux、真实ledgerDt与严格double累计账舍入；原absMass1e-14/relMass1e-10以及严格红不改，物理flux与cumulative audit分别报告；只补未测三应用/film/global/gross源链|
|cold-start-Js/history-Co|至少两组改变流场/dt的真实owner输入，fresh reference、首次Js/已发布lagged与下一stage消费；不人为ready或改策略|
|CHT-radiation-mapper|补empty/multiple-selected实际consumer；single/mixed/DO/完整solid账复用|
|pore/melting、host-sanitizers、allocation-fault-lifecycle|先首要三条证明，按未测source链补合法小输入/codec之外真实host路径/可清理实际free-sync-host资源故障；不危险OOB、不新生产门禁。已通过OOM/copy/retry不重复计数|

## 预算与比较

RTX4060 Laptop8GB/sm89，当前GPU空余约7792MiB、driver592.00、CUDA13.1/OpenFOAM10/g++11.4，单GPU顺序并有外部文件锁。所有install/bin/lib/fixture构建及符号链接目标须resolve在 `/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`；环境捕获实际路径。普通单例300s，host/native link和fixturecompile900s；外部观察不进入求解器。

两个旧timeout的精确原命令、inputSHA256、binary/log哈希见 metadata/prior-required-input-hashes.json。Ns10原完整reacting普通运行57s而sanitizer300s无finalsummary，追加有限1800s以覆盖instrumentation成本；evap此前诊断300s无完整acceptedwindow，正式修复同输入追加有限900s观察。两项均不改网格、步数、终点、物性、精度或阈值，外部每30s保存log字节/accepted/checkpoint进度；不保证完成、不自动再次增预算或转PRODUCTION。

原 M/E 标准 absMass1e-14/relMass1e-10，absEnergy1e-10/relEnergy1e-10；P保持原用途和原严格观察，不偷换阈值。库存用实际finalweightedflux/packets和真实checkpoint，80位 Decimal 从实际double精确转换；全热含film H-pM/rho+KE及实际work/constraint/external渠道，不能用snapshot代完整收支。正式零/非零材料表达式、全语义 chemistry字段采用原bit规则；exclude struct padding/明确hostwallclockseconds only。observer只读、保存补丁与source/object/binSHA，实际原输出必须不受影响；I/O不做效率测量。

## 交付

所有新执行command/UTC/cwd/exit/signal/timeout/types与src/dependency/input/object/binary/GPU/toolchain/flags逐项入results/spec。原版/正式修复/支持/诊断/未执行分开，所有FAIL/BLOCKED/INCONCLUSIVE保留。已有不可变字节只引用固定commit/path/SHA，不上传整套旧数据。同一log只存一次，跨topic复用eid。更新REPORT/coverage/findings/SHA256SUMS与R014响应/manifest，行政validator通过后fetch核远端、保留他人正常push；无合并权限。
'''
(D/'TEST_PLAN.md').write_text(plan)
for p in [P/'r3_runner.py',P/'r3_plan_build.py',P/'r3_paths.json']:shutil.copy2(p,D/'inputs'/p.name)
preflight='set -o pipefail\nsource /opt/openfoam10/etc/bashrc\nset -e\n'
args=cmd('bash','-c','printf "APPBIN=%s\\nLIBBIN=%s\\nCUDA=%s\\nFOAM=%s\\n" "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_HOME" "$WM_PROJECT_DIR"; nvidia-smi; nvcc --version; g++ --version; free -b; uname -a')
t=run('META-ENVIRONMENT',args,[],300,'Environment/install paths only, no numerical acceptance.',kind='ENVIRONMENT');assertion(t,t['exit_code']==0,'ActualsingleGPU/driver/toolchain/RAM/testinstall paths captured.')
area=Path(PATHS['repo']).resolve()
for p in [ROOT,D,ROOT/'work/install/ns2/bin',ROOT/'work/install/ns2/lib',ROOT/'work/install/ns10/bin']:assert p.resolve().is_relative_to(area)
app=ROOT/'applications/CHMT'
for ns in [2,10]:
 build=ROOT/'work/build'/f'CHMT{ns}';build.mkdir(parents=True);gen=build/'generated';oldobj=OLDROOT/'work/build'/f'CHMT{ns}'/'Backend.o';obj=build/'Backend.o'
 version=json.loads((OLD/'versions.json').read_text());key=str(oldobj.relative_to(OLDROOT));assert version['objects'][key]['sha256']==sha(oldobj),key;shutil.copy2(oldobj,obj)
 reuse_receipt={'identity':'PR14_REVIEW_ONLY','ns':ns,'producer_source':PATHS['old_source'],'producer_evidence_commit':PATHS['old_delivery_commit'],'original_object':str(oldobj),'object_sha256':sha(obj),'bytes':obj.stat().st_size,'all_CUDA_common_dependency_files_unchanged':True,'source_dependencies':dependencies,'fresh_host_source':SOURCE,'scope':'Reuse actual native CUDA Backend.o only, compile allformal hostsource and final-link. No compilation is GPU acceptance.'}
 (D/'metadata'/f'cuda-object-reuse-ns{ns}.json').write_text(json.dumps(reuse_receipt,indent=2)+'\n')
 t=run(f'BUILD-INFO-Ns{ns}',cmd('python3',app/'devtools/build_info.py','--species-count',ns,'--arch','sm_89','--nvcc','/usr/local/cuda/bin/nvcc','--output',gen),[1,24,25,26],900,'Originalbuild metadata generation only; sourceidentity separatelyfrozen.',inputs=[app/'devtools/build_info.py'],kind='BUILD');assert t['exit_code']==0
 binpath=ROOT/'work/install'/f'ns{ns}'/'bin/CHMT';binpath.parent.mkdir(parents=True,exist_ok=True)
 flags=['-std=c++17','-O2','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32',f'-DUGKWP_GAS_SPECIES={ns}','-I'+str(app),'-I'+str(ROOT/'common'),'-I'+str(gen),'-I/opt/openfoam10/src/OpenFOAM/lnInclude','-I/opt/openfoam10/src/finiteVolume/lnInclude','-I/opt/openfoam10/src/meshTools/lnInclude','-I/opt/openfoam10/src/OSspecific/POSIX/lnInclude']
 files=[app/x for x in ['CHMT.C','io/MultirateEvolution.C','materials/CpuMaterialDriver.C','materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C','restart/Checkpoint.C']]+[obj]
 libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart']
 t=run(f'BUILD-NATIVE-HOST-LINK-Ns{ns}',cmd('g++',*flags,*files,*libs,'-o',binpath),[3,6,13,14,20,21,24,25,26,29],900,'Compileallformalhostsourceand finalnativeapplicationlink againsthashverifiedunchangedCUDA object.',inputs=files+[gen/'CHMTBuildIdentity.H'],kind='BUILD');assert t['exit_code']==0
 print('NATIVE',ns,sha(binpath),flush=True)
# Fresh formal chemistry generation/compile; no support or diagnostic patches.
fixture=ROOT/'work/fixtures/chemistry-trial-formal';fixture.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHEMISTRY-FORMAL-FIXTURE',cmd('python3',ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py',ROOT,fixture,'--nvcc','/usr/local/cuda/bin/nvcc','--arch','sm_89'),[10,12,14,26],900,'Fresh正式fixture nvcccompile/link; noS202/S203/S208 overlays; run separately.',inputs=[ROOT/'tests/fixtures/chemistry_trial/chemistry_trial.cu.in',ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py'],kind='BUILD')
farchive=D/'inputs/chemistry-fixture';farchive.mkdir();
for n in ['command.json','chemistry_trial.cu','build.log']:
 if (fixture/n).is_file():shutil.copy2(fixture/n,farchive/n)
assert t['exit_code']==0
print('READY formalfixture',sha(fixture/'chemistry_trial'),flush=True)
