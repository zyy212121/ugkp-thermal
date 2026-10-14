from r4_runner import *

read_receipts=[]
for folder,commit,names in [(OLD,PATHS['old_delivery_commit'],['REPORT.md','results.json','findings.json','coverage.json']), (DEV,PATHS['remote_head'],['REPORT.md','SOURCE.json','VALIDATION.json','EVIDENCE_REUSE.md','GAP_RECHECK_PLAN.md','evidence/CURRENT_SCOPE.json','evidence/COPY_RECEIPT.json','PUBLICATION.json','evidence/published-source-object-verification.json'])]:
 for n in names:
  data=(folder/n).read_bytes();entry={'fixed_commit':commit,'repo_path':str((folder/n).relative_to(OLDROOT if folder==OLD else ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'read_complete':True}
  if n.endswith('.json'):
   q=json.loads(data);entry['top_level_keys']=list(q)
   if n=='results.json':entry['units']=[{k:r.get(k) for k in ['id','verdict','scope','limitations']} for r in q.get('acceptance_units',[])]
   if n=='coverage.json':entry['gaps']=q.get('remaining_gaps',q.get('gaps',[]))
  read_receipts.append(entry)
(D/'metadata/prior-evidence-read-receipt.json').write_text(json.dumps(read_receipts,indent=2)+'\n')
plan=f'''# PR14 FAST4 独立审查与真实 GPU 计划

PR14_REVIEW_ONLY / FAST。正式源码 `{SOURCE}`，tree `{PATHS['source_tree']}`；交付记录父提交 `{PATHS['delivery_parent']}`。新目录 `{str(D.relative_to(ROOT))}`。实际机器 RTX4060 Laptop8GB/sm89、CUDA13.1、WSL Ubuntu22.04/OpenFOAM10、FP64；详细环境与原编译flags在每个命令的spec及日志。

完整读取固定开发交付 REPORT/SOURCE/VALIDATION/EVIDENCE_REUSE/GAP_RECHECK_PLAN/新索引、PROTOCOL、第三轮 REPORT/findings/results/coverage；其固定SHA与单位局限在 metadata/prior-evidence-read-receipt.json。本轮139变更与上传414校验和已经逐项核实，完整源/dependency/blob/SHA见metadata/source-hashes.json/source139-verification.json。PUBLICATION.json及SHA256SUMS自身不在上传校验单，另存固定远端对象身份；不改开发原记录false字段。

## 测试前身份与比较

本工作树生产源码与正式固定source逐文件相等，记录文件来自远端已上传head。所有输出/安装 resolve必须在授权test区。输入和实际binary/object SHA在执行前取得；后续每种stage产物单列，不把开发低opt或历史binary算本轮原flags验证。F014/FN014与R014来自当前source-index映射，spec逐命令绑定实际source blob；不称词法index等于全调用链。

用户最新明确授权：资源/用时估计只用于调度和监测。所有构建、fixture、sanitizer和原蒸发端点**无自动时间终止**；旧300/900/1800超时证据只引用，不能把PROTOCOL旧初始预算恢复为硬停。单GPU顺序锁、native构建串行、30s记录进度/进程RSS/GPU/oom计数。真实资源失败保留，不无依据反复重编。诊断I/O不做效率时间，普通测时与profiler分开。

原PASS仅在精确依赖和原适用范围未改变时引用固定commit/path/SHA；旧FAIL/BLOCKED/INCONCLUSIVE/ASan和missing6dd13b/8ed8a永久保留。旧全系统support动量为RETIRED_BY_USER，不写成物理修复或PASS。未采集/reserved0不算守恒零。无CPU CUDA mock替代数值验证；不改输入、端点、FP64、容差；没有生产audit回填或新数值/版本/拒绝/回滚/停算门禁。

## 本轮依赖与14原gap

|原ID/单位|变动/复用边界|本轮必要观测与原标准|
|---|---|---|
|chem-native-full-retry-owner / TEST-R3-NEGATIVE-SEMANTICS|core/native/policy/snapshot与正式fixture改变，旧成功不能全复用|正式generator O3 fmadtrue真实CUDA+memcheck99；16cell/16transaction/32五成员FAIL_STATUS、kind7/15独立replay；四原controls同次、实际acceptedtrace独立reference；逐cell和聚合V/rhoV/rhoEV/3rhoUV/speciesV与全部policy/flags恢复原bit/semantic|
|SST22-owner-graph|shared storage与13audit数组/6D2H删除|新源真实配置/所有权/static+moving/accepted+rollback真实字段；开发149host不算GPU|
|SST-RK-budgets|SST hooks/common/source budget删除|测试侧独立真实Euler源、最终限幅face、RK1/2/3权重、初末库存；不恢复生产audit|
|moving-GCL|新snapshot/common/Backend触及|原短moving输入/GCL、两个chemistry半步实际V与species/rhoE；原volume observer只有其不变ABI/范围复用|
|cold-start-Js|Backend/SST依赖改变|原两flow/dt coldJs/lagged数据和freshCo reference重新编译实际owner；不足selecteddt/SSTscope仍partial|
|wet-ablation-material|signedopenedge与film/material/outerreport删除|native film/curved/lifecycle+actual material受影响检查→原evapGPU输入至.001无自动硬停；17/8/25/32CPU限定证据按receipt复用，不当完整history/GPU|
|CHT-radiation-mapper|physicalradiation loops/report与sharedcommon/wire改变|真实radiation/contact native before/after、empty/single/multiple实际mapper；原implicit_math拒绝/缺refs/非法meshASan分列|
|mixed-wall-coupler|同上shared依赖，coldwall数学本体可限定复用|configure/patch顺序、coupled值、toggle/retry消费者物性原范围对照；不把旧token变有效配置|
|FSH-cold-profile|common/shared/codec变动|实际cold1D/2D及restart，CHT floatcast→已有doublecarry、FSHdouble，多组/温度相区与非零carry；所有写出doublemean+carry逆温差≤原1e-9K|
|host-sanitizers|新hostcodec/storage/material/interface|真实host ASan/UBSan相关边界；旧非法输入失败保留，不算正常物理回归|
|allocation-fault-lifecycle|actualpolicy/arrays/D2H改变|真实consumed species/rho copy失败+rollback/retry/owner销毁；纯auditcopy故障退役；其他free/sync/alloc覆盖按实际证据，不靠check进程退出推断恢复|
|resident-full-budgets|gasUGKP/FSH/CHT shared producer/transfer/consumer变化|三应用真实初末M/P/E/species库存+实际阶段flux/source/particle/wall账；原严格M abs1e-14 rel1e-10、E abs1e-10 rel1e-10、P原同用途标准；完整性不足仍open|
|native-FP64-clock-lineage|生产补偿clock数学本轮未改|固定旧actualhex/首差记录按原范围复用；必要正式新源affected核对，不改旧时间或历史积分|
|developer-formal-recheck|139source/414remote证据已核实，但新O3/GPU原未执行|原nativeNs2/Ns10CHMT与shared gas应用受影响构建、正式fixture原-O3/fmad；低opt不能代替；最终record/source/remote分开|

## 命令依赖顺序（下面为计划，未执行不称PASS）

1. 每种正式源新建native build/install。CHMT Allwmake原-O2、Ns2/Ns10/sm89；gasUGKP/FSH private backend和CHT原build脚本实际common接口。项目现有C++14入口保持，不升级标准掩盖回归。
2. `python3 tests/fixtures/chemistry_trial/prepare_chemistry_trial.py CHECKOUT NEW_REVIEW/fixtures/chemistry-formal --nvcc /usr/local/cuda/bin/nvcc --arch sm_89`；实际binary原GPU；`compute-sanitizer --tool memcheck --error-exitcode 99 binary`。不叠S301/S302或改owner/kernel/CFL/端点，所有比较前真实同步。
3. native film/curved/lifecycle/material、V1旧新端点；真实物理consumer独立观察器只在test副本，保存patch/input/source/binary SHA。Wire原25+2Ns/species25、reserved16..24与config36零，old→new/new→old实际species/heat/boundary，不能只new-new。
4. 正式原thermal100/Ns10、SST static/moving RK1/2/3、原wet-film及原evap完整.001，原合法phase与CHT/FSH/Gas native短case按链补测。新增/超旧范围先追加dependency缺口原因。
5. 因本轮触及restart相关存储/output依赖，原thermal与wet最短full/checkpoint1/resume严格位比较；旧thermal1ULP与wet27差异不能豁免或统一归因。不清cache/coldseed/添迭代。

删除producer→storage/transfer→consumer源码审查并行只读。保留Co/rhoK/omega/sourceNumber、剪切/压力功/热相变、actualdonorclamp、RetryTrial残差和adaptive step。任何诊断/最小支持修复先保留正式红与精确diff，支持绿单列，实际原输出无干扰后才用于观察；正式源码不改。

最终同一目录REPORT/TEST_PLAN/results/coverage/findings/commands/环境/input/source/binary/object/原日志/逐点数据/patch/SHA256SUMS。复用不可变证据只固定引用。独立acceptance单位不重叠加总。R014响应/manifest owned_paths/evidence追加，admin validator后核远端变化正常push；不force、不合并、不改README/Conversation/描述、不创建凭据。
'''
(D/'TEST_PLAN.md').write_text(plan)
for p in [P/'r4_runner.py',P/'r4_plan_build.py',P/'r4_paths.json']:shutil.copy2(p,D/'inputs'/p.name)
t=run('META-ENVIRONMENT',cmd('bash','-c','nvidia-smi; nvcc --version; g++ --version; compute-sanitizer --version; nsys --version; free -b; uname -a'),expected='Actual environment only.',kind='ENVIRONMENT')
assertion(t,t['exit_code']==0,'Actual GPU/toolchain/environment captured; not numerical acceptance.')
for ns in [2,10]:
 build=ROOT/'work/build'/f'CHMT{ns}';install=ROOT/'work/install'/f'ns{ns}'/'bin';install.mkdir(parents=True,exist_ok=True)
 args=cmd('env',f'CHMT_BUILD={build}',f'FOAM_USER_APPBIN={install}',f'UGKWP_GAS_SPECIES={ns}','CHMT_CUDA_ARCH=sm_89','bash',ROOT/'applications/CHMT/Allwmake')
 t=run(f'BUILD-NATIVE-CHMT-Ns{ns}',args,[3,6,8,9,10,13,14,20,21,24,25,26,29],expected='Fresh whole native CHMT with originalO2 flags and finalshared interfaces; numericalrun separate.',inputs=[ROOT/'applications/CHMT/Allwmake',ROOT/'applications/CHMT/gpu/Backend.cu'],kind='BUILD',estimate_seconds=600)
 if t['exit_code']!=0:break
 b=install/'CHMT';amend(t['id'],binary=str(b),binary_sha256=sha(b),object_sha256=sha(build/'Backend.o'))
 print('NATIVE_READY',ns,sha(b),flush=True)
