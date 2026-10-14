import json,pathlib,shutil
from r2_runner import *
plan='''# PR14 第二轮独立 FAST 计划

PR14_REVIEW_ONLY。生产固定4b5c480b572f234ddecf5e89164fe59b0d2e9735；生产树0fa2562f4c83394cdf94e9650de3b22c4fb4bb15；交付父提交75cfd885c028fe80d011a8a081f8bed1f8aeb55e。测试前已读取开发者 REPORT/SOURCE/VALIDATION/CLOCK_ANALYSIS、上轮 REPORT/coverage/results/findings、20文件差分及 PROTOCOL。没有生产长算或合并授权。

## 依赖与证据复用

上轮不可变证据：ae5a59478023c74e23b64dc1a1f388fb805ef565，review/pr14/reviewer-deliveries/20261013T182722Z-fast。旧被测802a11e7、旧比较d6110c02；上轮原失败和diagnostic绿均引用固定路径/哈希，不重复上传。

|范围|依赖变化/缺口|复用与定向执行|
|---|---|---|
|A F001|kernel Trial路由、TrialPolicy运输状态、Integrator失败reaction传播改变|复用原StepLimit红/public/ordinary对照；正式Ns2 kernel复验和完整真实owner retry/terminal/audit/budget夹具，缺backing/reactants组合，补Ns10 affected compile/runtime；不重复原全部化学历史|
|B F002/F004|CHT初始化effusivity与float cast carry改变|原cold2D输入、wet100、2D restart正式版复验；配置次序/mixed/toggle/copy retry真实CUDA；非零carry/相区/多粒子组及FSH double cold/restart缺口补测；原P003/P009支持用途引用，不叠加P007/P010|
|C F005/临时内存|Co尾部、owner持有Thrust allocator改变|正式Ns2 thermal100步initcheck及全部CSV/checkpoint/geometry逐字节对802；补Ns10/single/multiblock/empty及owner cache增长/连续归约/window/restart/可控资源失败销毁；六个前处理kernel保留；Nsight与普通重复计时分开|
|D F003|本轮时间表达式未改；缺actual first divergence|复用已有CSV严格差及标量预测；baseline/tested/formal独立只读clock trace，逐微步hex和原输出等价；trace I/O不得用于计时|
|E support|schema checker和oracle已正式调整|与P001/P005有效支持产物核对；数值阈值和其他冻结函数不变；复用P003/P008/P009/P012各原目的；新增支持差分单列|
|原通过其他范围|无生产依赖修改|复用已验证的operator/pool/pressure/core/AUTO/restart等限定证据；共享修改直接相关cold/chem/CHMT选择补测；不视33topic全功能已通过|
|F 14原缺口|未测证据缺口仍存在|A/B覆盖owner/mixed/FSH不重复；依次补active SST22→Js/historyCo→ALE/GCL/stage volume→wet/film/ablation/surface/porous→radiation/mapper→ASan/UBSan与资源恢复→三应用完整实际stage预算；使用原gap ID，无法实际覆盖如实留空|

## 执行和比较规则

每个case外部预算300秒；native build因先前真实编译约116秒且Ns10更长，预算900秒。为超范围或增预算补测，先追加依赖/缺口原因。小型真实GPU夹具不能用CPU shim或source-text测试冒充通过。GPU单例顺序执行，记录当前RTX4060 Laptop/driver/CUDA/OpenFOAM/flags与src/binary/object/input SHA。

A沿用原kernel对照字段规则，完整owner原requested interval必须完成：失败Trial状态、audit、budget逐字段/字节保持，成功路径原比较规则；实际backing错误区别可继续诊断，maximumSteps1组合。不新增生产阈值停算。

B全写出时刻解码actual profile，double求均值+double carry做原caloric逆，abs1e-9K；非零既有carry不可丢失。restart最后实际原始字段/RNG/profile/carry/history逐字节比较。配置夹具下载实际face effusivity，默认sqrt(1800*710*100)，coupled producer真实值优先。7900/500/15旧token仍未读取。短程预算使用真实stage权重/最终flux/实际库存，有限快照不可代替。

C原100微步所有生产输出逐字节比较802（非d611、非clock诊断）。归约max/OR/NaN表达式不变，empty只测实际可达适用路径。allocator只能缓存临时存储：同owner不同数据必须更新结果。安全资源故障注入位于隔离诊断边界，保留原源与diff/hash，不能当原版无故障数值运行。profiler比较同输入步数flags，导出所有API/kernel/D2H/sync/live/peak和warm/stable划分；保留内存到销毁成本。普通重复3warmup+7sample，保留range/stdev，无whole solver外推。

D按开发者RUN生成每版最小只读observer补丁。实际baseline/tested/formal日志按window/step/occurrence对齐，第一hex差定位到clock→wall→gas→packet；比较全部原输出确认trace无干扰。未解释差不得按容差关闭。

E支持检查结果是support范围，不是生产数值验收。历史失效的schema/oracle失败继续保留。归档仅新增必要输入、原始logs、逐点数据、patch和metadata，不重复旧不可变证据。

F每个缺口关闭必须实际CUDA输出与有意义对照/预算断言；先确认生产调用链。未达到完整收支时仅记partial而不关闭gap。ASan/UBSan是host真实边界检查，不能替代GPU sanitizer。

## 交付

REPORT/TEST_PLAN/coverage/results/findings/commands、源/环境/二进制哈希、原日志、输入/数据、补丁、SHA256SUMS。append R014 reviewer/recheck responses与manifest；validator必须零error。推前fetch/核对remote父，若变化保留并整合他人提交，仅正常push，不force，不改README/生产源码/PR评论描述，不合并。
'''
(D/'TEST_PLAN.md').write_text(plan)
shutil.copy2(pathlib.Path(__file__).parent/'r2_runner.py',D/'inputs/r2_runner.py')
shutil.copy2(pathlib.Path(__file__).parent/'r2_plan.py',D/'inputs/r2_plan.py')
for name,a in [('environment',['bash',str(D/'inputs/env.sh'),'bash','-c','nvidia-smi; nvcc --version; g++ --version; foamVersion; compute-sanitizer --version; nsys --version; uname -a']),('source-diff',['git','-C',str(ROOT),'diff','--stat',PATHS['old_source'],SOURCE])]:
 t=run('META-'+name,a,[],300,'Environment/source capture only.',kind='SETUP');amend(t['id'],verdict='PASS' if t['exit_code']==0 else 'BLOCKED')
c=json.loads((OLD/'coverage.json').read_text());print('coverage keys',list(c))
def walk(x):
 if isinstance(x,dict):
  if x.get('status')=='NOT_RUN' or x.get('verdict')=='NOT_RUN':print(json.dumps(x)[:2000])
  for k,v in x.items():
   if 'gap' in k.lower():print('GAP FIELD',k,json.dumps(v)[:8000])
   else:walk(v)
 elif isinstance(x,list):
  for a in x:walk(a)
walk(c)
print('SOURCE',SOURCE,'D',D)
