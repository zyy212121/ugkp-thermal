# PR14 FAST4 独立审查与真实 GPU 计划

PR14_REVIEW_ONLY / FAST。正式源码 `47034e4e0380fa5b84165b49207a95a9b187a52a`，tree `c932823a00b3ed2f647e22de5002f9b913962fee`；交付记录父提交 `55d3663830ada63077241c528e2c81b53976e2c4`。新目录 `review/pr14/reviewer-deliveries/20261014T190010Z-fast`。实际机器 RTX4060 Laptop8GB/sm89、CUDA13.1、WSL Ubuntu22.04/OpenFOAM10、FP64；详细环境与原编译flags在每个命令的spec及日志。

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

S401支持修正：正式thermal真实GPU/initcheck完成，但原checker所有CSV值强制float转换，在新增明确NOT_COLLECTED元数据列失败。正式FAIL已留 logs/CHECK-ORIGINAL-thermal-full.log；仅在测试sidecar识别该单列并验证标签后排除数值转换，不改任何physical字段/阈值/schema/输入，不删生产数据。支持版实际物理检查与正式原checker红分别记账；不将缺失audit算守恒零。

D401新增原因：生产SST13audit数组/6D2H及sharedbudget删掉后，原GPU端点和host149项不能证明真实Euler/RK/chemhalf全链。只在隔离SOURCE archive副本插入只读实际resident数组/最终flux/old-newV/本地真实SST源/chem半步读值；原算式/控制/调用顺序不改，不恢复productionaudit。正式原所有physical输出bytes必须与observer逐文件相等才用于测量。记录实际ledgerDt与blend权重，测试侧独立逐cell/全域M/P/E/species与K/omega源/face核算；生产labelsreserved0不参与。新增D2H/sync/printf不作效率计时。原SST三RK以及moving/化学依赖改变，逐个原输入/终点复验，不借缩步/缩端点/阈值得到通过。observer声明/补丁及全部SHA单列。

S403 formal原O3/GPU和memcheck均kind7物理field比较exit11（memcheck0errors但不是PASS）。源码跟踪：原faileds保留coefficientCount1/maxRejected0，而旧freshoracle是count6/defaultRejected100；其真实public caloric/rate路径不同，不能按相同数值预期。只改生成的测试sidecar：diagnose模式继续原不等输入对照并打印fieldhex；support模式保留16原失败和四原控制，kind7/15独立freshactual/freshreference均保持各自原metadata/control/interval，执行beginInterval→完整advanceGasTrial(false)→commitInterval且保留originalfailure。补聚合V/P3与snapshot-valid语义，不改生产/kernel/原成功阈值。病态metadata public路径中的NaN/diagnostic传播不按有效case守恒拒绝，也不称有限正确物理结果；支持绿不能变成正式夹具绿。

S404实际checkpoint decoder：复用固定0aa/9d已记录只读完整physical字段源，新编最终Checkpoint.C/HostState接口ASan/UBSan，解析新GPU产物。原格式兼容Budget/ChemistryAudit若打印只标明UNCOLECTED，不作物理预算；完整gas/solid/pore/film/particle/geometry/packet字段从真实checkpoint取。纯hostcodec不替代GPU物理验收。

新增SST64 static/moving ×RK1/2/3原因：原8cells全部贴壁，其omega为代数约束，不能证明内部未约束omega实际source；原moving输入未启SST。只在新独立控制保持原.001/100steps、原物性/FP64/阈值，网格2³改4³得到8内部cells与多face blocks；原所有输入/结果不动。static/moving均使用原PrescribedSinusoidal参数，实际Euler/RK源/最终flux/old-newV及omega约束类别分别观察；D401原全部输出bytes须与对应正式新控制一致。新增规模不是缩旧输入或化学/蒸发终点。

S406实际SST22/冷壁资源恢复增量：canonical DeviceState/storage接口已变，旧owner22证据不能全复用。由固定当前源重新提取完整canonicalprefix/签名，链接本轮原O3的真实CHT/FSH nativearchive；保留原每22真实CUDA allocation错误/retry、实际copy错误、既有ownedpointer保持/reconfigure/destruction与mixed冷壁初始化copy失败。外部fault调用真实cudaMalloc过大请求/cudaMemcpy非法kind，不mock返回值或CUDA算术。追加safe物理资源观察不新增生产门禁；free/sync/hostallocation/concurrency未覆盖不能由普通cleanup泛化。

S407分析工具异常定位：第一次D401分析SIGSEGV/exit139，RSS227340KiB、vmstat oom_kill前后0；不归为GPU生产失败或OOM。保留原partialreports/log/resource，针对moving-chemistry单文件用Pythonfaulthandler原计算诊断，不新GPU重跑或放阈值。wall-constrainedOmega naiveblend旧oracle另记真实约束scope不适配，不能忽略红改整体绿。

D402追加原因：D401真实RK后wall-constrainedomega被已有recoverSstPrimitives重设为代数wall值，naive密度线性blend oracle红保存。只扩充只读host D2H/打印实际wallDistance/gasMu/coefficients/wallTreatment/真实SST与速度梯度/F1F2，独立核算已有低Re壁面omega约束及净/分解源；不改blend/recovery/生产SST代码或阈值。D401原数据/工具异常SIGSEGV/旧oracle失败独立保留；D402逐原SST及新64例原终点/原输入输出bytes再核等价。不以简单忽略wall差关SSTgross/graph范围。

S408分析修正：phase原onecell生成一次discarded outer candidate和一次accepted replay（实际日志one microstep/two material solves），D401初观测聚合错误计入两次压力冲量，原红JSON保留。以实际提交checkpoint gasStages[0].newVolume的原double位值核对最终候选；同time/dt的另一不同geometry候选不属accepted ledger。不改输入/生产重放/阈值。只完成该真实gas库存/通量/packet scope，非globalfilm/support指标。S404默认ModelConfig.physics未解析，不能当实际配置/film KE测量。

正式470原thermal100步/原flags新Nsight资源观察：新audit删除影响API/D2H计数，旧Co allocator细节可限定复用但不能把旧总API当新实测。独立profiler输入原字节，输出全文件对比；只评实际warm/stable allocation/API/transfer/kernel，诊断IO不计性能。非profiling同输入原终点5完整样本及离散程度另记，不用kernel提速外推求解器。

V1双向真实endpoint限定补测：使用实际旧9d交付所验证FSH frontend/backend与本轮470新端点，原16active cold2D输入/full原1e-5终点，原native flags/runtime，所有原物理文件bytes对比。旧前端实际binary与新前端hash相同如实标明（旧frontend生产依赖未改）；backend哈希不同。只闭合该原Legacy模式热/边界/profile路径；不证明未执行的非零多species混合wire/对应gasUGKP双向互通，不把双新host往返当对端验证。
