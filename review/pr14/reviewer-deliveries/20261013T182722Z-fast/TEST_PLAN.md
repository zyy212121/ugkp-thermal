# PR14 独立 FAST 测试计划

PR14_REVIEW_ONLY。计划建立于执行前；原版固定 802a11e7dfe986488a8bf8305e86424dcc94c9e1，tree db2863158139842f6c8221c0da11ba640f381336；生产源码 a73b9635510fb09d7baa5339e14d1aa609f0ff87；基线 d6110c02745a8ad726208a426c4bb3a28233c1fd。33 个主题均已读取，具体文件、blob、SHA256、函数定位与历史证据映射保存在 coverage.json。文档更新不算算法更新。

## 执行约束与记录方法

- 本机 WSL Ubuntu 22.04，RTX 4060 Laptop 8188 MiB、sm_89，驱动592.00、CUDA13.1.115、GCC11.4、Foundation OpenFOAM10、DP/Int32。所有构建/运行使用 test 区隔离 checkout、独立 work/install/bin 与 lib；正式用户 bin/lib 从实际 PATH/LD_LIBRARY_PATH 排除。现有 test 工作保留在原目录并另存 binary patch/未跟踪源码；不修改README。
- 每个测试执行前生成 logs/<ID>.spec.json，记录命令、cwd、源码/依赖/输入SHA256、二进制身份、预期、比较规则、规模和外部预算。执行后保留完整 stdout/stderr、退出码/起止时刻/elapsed 与原始输出；results.json 汇总。计划命令和执行命令分开；新阻断/新风险追加计划。
- 小例预算300秒；sanitizer每次300秒；大TU编译允许900秒且顺序构建，原因是 native device 编译与OpenFOAM链接无法复用本机已存在的匹配产物。预算只在测试进程外使用，不改求解器停算条件。
- 原版失败永远保留。最小诊断源码修复以 patches/ diff交付，修改后的源码指纹/二进制单列；不会计作原版PASS。不存在开发者正式复验版本时该栏为空。
- 可复用的历史日志仅在SHA/依赖适用范围成立时登记；现有CPU/shim/compile不能替代GPU验收。先验证497份已登记证据的字节与628项源码身份；LFS指针与实际产物区别登记。新GPU已覆盖的路径无需重复编译或数值测试。

## 第一阶段：身份、环境与构建

IDENTITY-01：核GitHub API真实head分支/ref、fetch固定源和base、source-index与manifest证据SHA、README diff、final生产源与交付源的区别、CI实际观察。参考Git对象恒等；要求匹配数与失败项逐一报告。环境命令/GPU kernel roundtrip是环境证据，不是物理PASS。

BUILD-CHMT2/10、BUILD-gasUGKP2、BUILD-CHT、BUILD-FSH：使用仓库原生构建入口，sm_89，CHMT O2默认FP64，CHT O3/--fmad=false，gas/FSH O3/--fmad=true，Ns按实际应用支持。记录完整child退出码、最终binary/archive/object SHA、ldd/readelf/CUDA实例与实际执行路径。现存本机旧源码程序不匹配，必须新构建；开发者a164/970/a73编译证据仍保留且不重命名。

## 第二阶段：实际 CUDA 小算子

CUDA-OP-*：优先复用仓库实际CUDA夹具生成器，检查生成源确实include生产共享CUDA/实际operator body，并捕获全字段/最终consumer。首批为 operator_unification/differential.py（三应用）、pressure/limited_flux_balance.py（三应用）、gas_auto/gas_auto_pool.py（三应用）、shared_operators/collision_behavior.py、thermal_workers/direct_dispatch.cu。参考冻结原实现、独立守恒式与解析gas-only冷却。冻结基线先核来源/支持范围，不能概括为整个base版本。

新增 EOS/Ns2/Ns10、chemistry两半步、wall packet、cold1D/2D native wrapper 的小夹具仅在现有真实调用覆盖不足时添加；涵盖producer→storage→consumer→commit、实际温度/焓/phase/history/carry/ledger。纯FP64相同表达式优先逐位；不同host/device transcendental允许事先记录的双精度舍入误差；CHT float profile另有quantization容差且其double residual仍需守恒。负数/NaN/Inf诊断只观察真实计算/发布，不新增准入门禁，不故意越界。

## 第三阶段：原生应用/耦合/重启

CHMT-*：复用 verification/coupled_runtime/make_case.py 生成8 gas+8 solid cells/4 contact faces，实际CHMT native入口。single、thermal/frozen、diffusion及zero-D、Ns10 chemistry及disabled控制；每组 full/split/resumed，适用组H/2和dt/2 refinement。先一次构建Ns2供前三组共用，Ns10独立构建。原checker与独立附加逐点/库存/实际packet汇总保持阈值，必要时修复测试输入适配并保留原版输入/失败。

参考：UA=2 W/K、初始600 W热流；能量误差max(2e-5 J,2e-13*初始总能量)，质量1e-10 kg，EOS/caloric abs1e-7/rel2e-10；元素abs/rel1e-9；restart初始checkpoint逐字节重序列化，最终所有gas/material字段abs1e-10/rel5e-12；热量refinement0.5%+roundoff。两耦合窗口，thermal/single/diffusion H=5e-4、dt<=1e-5；chemistry H=1e-5、dt<=1e-6、终止2e-5。每个实际运行300秒，输入哈希执行前登记。记录stage权重、最终实际通量、HostState与terminal consumer；缺可观察字段标限制，不能以退出0冒充全覆盖。

RESIDENT-*：gasUGKP、CHT、FSH用真实frontend/backend入口、微型闭合网格、固定seed与少量粒子/短步。优先从现有consistency小案例缩小规模与终止时间，不运行生产案例。覆single/mixture、SST、壁面交换/冷profile/压力/拖曳，实际最终场和粒子compact/wire/restart/continue。预期闭合质量/动量/能量库存，并独立累计实际外壁/体源/交换；逐点restart按同binary同输入比较。输入准备完成后追加确切参数与容差，不能用调高阈值掩盖失败。CHT radiation/solid与mapper为空配置或仅消费face0另加安全最小native观测；动边界/GCL适用路径不足则追加或明确未执行。

## 第四阶段：异步/资源与sanitizer

SAN-*：实际native小例与相关CUDA算子运行compute-sanitizer memcheck/initcheck/synccheck/racecheck，error-exitcode=86，实际IPC child使用target-processes all。每次300秒；错误分类cuda/io/storage/数值/环境；sanitizer无报错只能支持其工具覆盖范围。owner配置迁移/alloc/free/graph/restart最终消费检查，host sanitizer仅对可实际构建且必要的host生命周期目标执行，未支持项说明。

## 第五阶段：效率

PERF-*：同GPU/输入/精度/flags/工作量对冻结匹配基线与PR，先warmup至少3次、重复至少7次，报告绝对时间/中位数/离散与PR/base比率。小数组及短吞吐数组；kernel用CUDA event、host/传输同步/整步用同边界wallclock，显存/temporary/resource记录。优先真实operator双实现及实际native窗口同baseline支持实例；基线不支持的批准冷Js变化单列成本不做伪性能对比。没有事件或端到端对照的项写未测，不由少kernel/copy/sync宣称提速。

## 发布和判定

读取每个原始结果并追加33份responses对应记录。新证据用SHA256导出新E014 ID，manifest.evidence/owned_paths登记。运行validate_records.py以及delivery SHA清单/自包含复现核验；不运行cleanup删除模式。提交仅本批目录和必要审查索引，不含正式生产源码诊断改动。推送前fetch实际head，处理新增差异并再验证；普通push到真实PR分支，绝不force或merge。远端读回文件/blob与固定链接验证后才称交付完成。任何FAIL/BLOCKED/NOT_RUN均阻止对其范围宣称merge-ready。

## 追加 P001：失效的独立checkpoint检查器

原版GPU输出先保留。冻结源FieldSchema的独立FNV-1a期待值为11438850135072464462，原checker常量15664811910203657810过时；仅适配该常量，保留全部数值/格式/assertion。预算与输入输出不变，不重跑未失效GPU计算；以原输出新增CHECKER1复验，原失败记录不覆盖。

## 追加 CHEM-RETRY：原成功native内层重试语义

Ns2全backed双组分同MW=.028、linearCp=1000、不可逆S0→S1 A=2；rho28、rhoSpecies[28,0]、momentum0、T700、V=.001、maximumSteps1、initial/maxStep1e-6。调用实际生产advanceGasChemistryKernel，dt2e-6、refining=true，预期d611返回StepLimit且species/audit/budget不发布；current应保留该原成功refinement方法，不应静默terminal完成。补dt1e-6成功控制和false公开terminal控制，maxSteps100正常控制；所有rho/momentum/rhoE/species、status/audit/budget/canary输出。CUDA sm89/O2/--fmad=false，同硬件同输入，每次300s；输入与源码SHA执行前spec登记。只恢复原有refinement入口，不新建整步数值拒绝机制。

## 追加 CHEM-RETRY：原成功native内层重试语义

Ns2全backed双组分同MW=.028、linearCp=1000、不可逆S0→S1 A=2；rho28、rhoSpecies[28,0]、momentum0、T700、V=.001、maximumSteps1、initial/maxStep1e-6。调用实际生产advanceGasChemistryKernel，dt2e-6、refining=true，预期d611返回StepLimit且species/audit/budget不发布；current应保留该原成功refinement方法，不应静默terminal完成。补dt1e-6成功控制和false公开terminal控制，maxSteps100正常控制；所有rho/momentum/rhoE/species、status/audit/budget/canary输出。CUDA sm89/O2/--fmad=false，同硬件同输入，每次300s；输入与源码SHA执行前spec登记。只恢复原有refinement入口，不新建整步数值拒绝机制。

## 追加 RESIDENT 参数

周期dustyBox：4 cells/128 parcels，单位体积，gas U=1、particle U=0、T=300、solid mass=.1kg、seed24681357，10步dt1e-6；gasUGKP/CHT/FSH实际frontend/backend，共用物理输入与各自支持配置。质量/动量abs1e-10+rel1e-10，能量abs1e-7+rel1e-12；同binary restart全字段abs1e-12+rel1e-11。原生cold1D/2D：原singleAluminaDrop物性与10-cell网格，一颗原表示粒子位置z=5e-7、速度z=-5、10步dt1e-6，首步实际撞壁；profile float的quantization通过double残余与完整ledger核算，而非放宽物理库存验收。输入哈希在prepare-resident-inputs.json及执行前spec。各例300秒。

## 追加 Native baseline 窗口效率

建立d611固定独立worktree，sm89/Ns2/O2原生CHMT，同一Ns2 thermal物理输入，实际8+8 cells/两窗口。先验证base/PR全字段与实际库存一致，只有原成功相同工作量才做窗口计时，warmup3/repeats7，各300s。binary与compiled source分离记录；如base输入准入失败则保留失败，不作速度比，kernel性能另用真实匹配共享算子。

## 追加 PHYS1：粒子owner输入修正

准备脚本将schema4/3的cell与orig_id列误写：全部cell0、orig_id重复0–3；与已规划均匀128粒子位置/4-cell库存不一致。保留原候选数据与输出，物理库存判定另用PHYS1：仅cell=i//32、orig_id=i，固定位置/速度/表示质量/seed/所有物性/步长完全不变。它不是生产源修复，旧输出不用于合法均匀闭合案例验收。

## 追加 P005：冻结算子oracle的baseline适用性

原full differential首败源自f03e215旧snapshot，比d611更早；仅补d611已有slip-wall独立thermal branch，不改输入/TFix或memcmp。CORE测试明确不执行旧cold段，其断言仍保留在生成源，原full失败保留；该段的paired gas producer/carry定义过时，不能据此验收cold。新core同flags同512×9输出/原压力拖曳等完整asserts，每应用300s。

追加 native 计时执行细则：同一已生成输入与网格逐字节复制，fresh output；base/PR各一次预检，然后各3次warmup、7次交错重复。外部进程墙钟包含启动、CUDA context、I/O，300s/次。接受实际CSV所有数值列及packet数/accepted_steps一致后才报告速度比；不把启动时间当kernel时间。

计时预检观测：bitwise CSV equality未达成，原版cell momentum最大5.42e-20、packet热量1.87e-16。完整数值阈值使用各版本既有原生checker，不修改容差，结果分别保存；两版本50+50 microsteps/2 material solves和CSV行数相同，可报告相同实际工作量的计时，不能称全字段逐位相等。另需开发者解释这一微小成功路径差异。

追加 native 计时执行细则：同一已生成输入与网格逐字节复制，fresh output；base/PR各一次预检，然后各3次warmup、7次交错重复。外部进程墙钟包含启动、CUDA context、I/O，300s/次。接受实际CSV所有数值列及packet数/accepted_steps一致后才报告速度比；不把启动时间当kernel时间。

追加P006 cold2D首次NaN诊断：完全相同INPUT2、O3/sm89/fmad=false，隔离源仅device printf读取已有计算值，记录face/scalar effusivity、age、wallG、carry。300s native，900s build；原NaN日志和所有输出保留，printf版本不用于速度比较。

追加P007：已先用原版完整field断言复现cold2D NaN（red），P006确认零face effusivity的producer链。仅隔离初始化producer使用现有DeviceState scalar fallback填充，后续真实coupled patch覆盖不动；不加数值门禁，完全相同INPUT2，不改物性输入/容差。原10步1D/2D重跑、full profile/carry/temp/fields有限性及publication检查，300s/例，900s编译；该通过属于补丁版本。

追加真实化学kernel分段效率：same d611/PR actual advanceGasChemistryKernel、sm89/O2/lineinfo/fmad=false，Ns2 1和32768cell，原成功dt1e-6/maxSteps100，3warmup/7sample。所有模型和state字段真实device backing，完整publication逐字段bit hash、独立一阶反应解析species绝差1e-9（局部1e-6积分），密度closure1e-10，保守state不写精确相等；kernel CUDA event、host launch、H2D/reset、D2H、wait与pipeline分别报告，CUDA event float仅计时。900s编译/300s执行，各版本当轮分开运行，运行时无其他GPU测试；相同输入未新增门禁。

追加P009/P010：保留旧checker mean32失败；仅将独立mean改用double，不改1e-9K容差或失败断言。FIELDS2-RED再次记录首次profile cast偏差后，P010隔离补上init producer所丢焓到既有double carry，原float cast及物理law不改，300s/原生例、900s build，复核全部写出time含首contact。P007 producer补丁合用，不能算原PR通过。

追加P008 baseline准入对照：原base相同INPUT2在旧wall字典准入处退出，原PR继续并NaN。删除仅三个PR未消费的旧token并选已有base FP64；同一INPUT3同时运行base/PR，保持所有实际消费物性/初场。仅确定历史失败边界，不作为原成功FP64回归或性能对照。每例300s，原失败不改。

最后追加sanitizer与profile继续：original CHMT Ns2 thermal 8+8两窗口mem/init/sync；P007+P010 cold2D mem/init/sync/race、真实5e-6重启至1e-5全fields比较。P010 cold1D endTime1e-4共100微步让ring0进入wet导热，物性/dt不改，writeInterval1e-5仅输出节奏，300s/例，不算原版通过。

追加Nsight Systems2025.5.2：原base/PR native CHMT相同两thermal窗口，cuda/nvtx/osrt trace+memory usage，禁CPU采样。300s/例。用于实测kernel/API/复制/同步/分配计数与时间、显存临时分配；profiler开销下的时间不作提速比例，正式3warm/7sample窗口墙钟另存。raw .nsys-rep/sqlite本地保留、哈希/生成命令/完整关键tables CSV自包含交付，避免无意义二进制。检查actual CSV对各自未profiling输出一致。

追加P011：原CHMT initcheck100次16byte result D2H未初始化先保留（red）；调查Co transform_reduce double+bool尾部7字节padding，不先认定Co/Js值失效。隔离改metadata flag为uint64 0/1，same16byte size/maximum/OR/遍历；900s Ns2 native build，300s full-run/全CSV+checkpoint逐字节/新fresh initcheck。无需新初始化扫描、数值门禁或物理运算改写。

追加Nsight Systems2025.5.2：原base/PR native CHMT相同两thermal窗口，cuda/nvtx/osrt trace+memory usage，禁CPU采样。300s/例。用于实测kernel/API/复制/同步/分配计数与时间、显存临时分配；profiler开销下的时间不作提速比例，正式3warm/7sample窗口墙钟另存。raw .nsys-rep/sqlite本地保留、哈希/生成命令/完整关键tables CSV自包含交付，避免无意义二进制。检查actual CSV对各自未profiling输出一致。

P011首版u64 metadata槽在CUDA13.1编译报错，独立失败patch/log保留。v2保留bool成员与偏移，将原padding显式化为7个清零uchar metadata成员，仍16byte，既有max/OR与double计算不改；同native100步+全output bytes+initcheck复验，不降验收标准。

## 执行收口与限定范围

296条已执行命令/124个独立验收单元。原版86PASS、9FAIL（4生产/传输、1严格跨版本、4过时支持件），隔离诊断14PASS，旧cold历史基线对照1BLOCKED，14项NOT_RUN；以results.acceptance_units的scope计数，编译/setup不算GPU数值PASS，主题之间不能相加。详细实际矩阵覆盖33项，文件函数registry共享存储。原版4个确认问题及FP64微差仍待正式处置，禁止以诊断通过关闭。

未执行而计划中列出的SST22、SST/RK、移动GCL、独立Js、完整chemistry owner retry、wet/film/产气/烧蚀、enabled radiation/mapper、mixed-wall/FSH cold、host sanitizer/安全资源故障、resident真实收支、clock首分歧与正式修复复验，明确见coverage/results.NOT_RUN；没有生产长算。两个大流量CUDA数组测试仍只10次短kernel。
