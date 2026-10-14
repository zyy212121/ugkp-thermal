# PR14 第三轮独立真实 GPU FAST 计划

PR14_REVIEW_ONLY。正式 source `9d5be6eff69dbced2fc959d422d3ba419a1bc39e` / tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`；记录起点 `a462edd16c20750a5875548f2e4ec690602d05cc`；上轮证据 `f5e6c5e2cbde2693f5fa811d874c200b07bbbc66`，上轮被测 source `4b5c480b572f234ddecf5e89164fe59b0d2e9735`。不得合并、force push、改 README/PR Conversation/描述；正式生产源由开发者修复，审查诊断仅隔离在授权 test 内。

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

S301新增支持观察范围（正式原fixture已actualGPU exit0并保留）：源码复核确认原失败矩阵只比较species/accepted审计/chemistrySpecies budget与cell0粗状态，kind7/15原状态被public重放覆盖。增加独立失败Transaction组合，保持所有原16输入/control与4个成功测试；真实begin/kernel/validate/rollback，打印两cell全部6ChemistryStatus成员，全部cell/face transaction arrays、各budget、accepted Audit与host账不泄漏，rollback恢复语义diagnostics但保留供owner读取的retry/terminal decision flags。kind10正确期望StepLimit+reaction0/1attempt/1reject/1nonlinear，不误要求RateFailure；kind0Newton计数由实际原1e-6direct成功控制获取。源/patch/hash单列，不修改生产或正式fixture，不用marker冒充完整语义。compile900s/run300s。

S301首次nvcc支持编译红保留：ChemistryStatus explicit bool转换不能隐式传check(bool)。S302只加显式bool转换，不改任何assertion、状态/阈值、输入或生产算法；source/patch独立保存，compile900s。

S303真实codec只读字段补充：原dry checker明确只计gas+solid、只允许GasSolid/PoreGas，不能验wet film库存；原checker红完整保留为不适用scope。原restart还缺checker-only restart_directory配置，其红不等于solver恢复失败。扩大原真实checkpoint decoder打印实际FilmAux均速/normal/物性/材料选择/porosity/solidVolume，对新GPU产物算完整film H-pM/rho+KE、vector P和逐recipient账；字段观察不变reader/schema/数值标准，hostASan/UBSan编译/run分列。

D301只读phase/ledger观察：在正式CpuMaterialDriver成功localphase附入ledger位置，仅fprintf实际mass/热/力/forcing/前后film/porosity/density/solidVolume/target。zero mass输出unusedrho0，绝不读取未选材料。GPU用原已验证D203 finalweightedflux/ledger观察object（source算法未改，完整hash/原patch引用），不新增GPU算法或断言。正式wet/合法非零controls逐文件原输出比较必须等价；I/O不计效率。编译900s/run300s。合法新controls独立于原wet：有效phaseCondensed0/phaseFilmY(1 0)/enableMelting+CoupledRecession；保留实际rho/cp/e0/pressure/film，meltTemperature300或400分别拟覆盖positive/negative，明确小区间1e-6一microstep/一个窗口，两旧4b合法控制与正式9d相同输入/flags比。只对新增控制设小终点，原wet/evap0.001保持。

S304只读ownerchronology缺口：原fixture已通过的计数和accepted分段不记录被拒绝attempt的顺序。新独立IO观察只运行原4owner/direct正控制，打印begin/source(dt,native)/validate(stepLimits+decision)/rollback/commit的真实顺序，不改变原assertion/原Co/control或任何GPU调用；16失败矩阵复用正式+S302不在这里重复。观察I/O不计时。保存newsource/patch/bin哈希，completeformalsemanticoutputs按过滤只读行后逐字段核对。compile900/run300。

D301首次host支持编译红保留：当前作用域是driveRecord指针，不是record。D302只纠正观察变量并打印真实消费的p.liquid/condensed物性与phase控制，不动任何物理算式。S303 decoder ModelConfig默认值不代表消费配置：readCheckpoint不恢复完整physics，raw physics0明确记录为UNINITIALIZED_DECODER_CONTEXT，不能当正式liquidRho0或缺backing。完整E计算采用D302实际消费rho（预计原1000；必须实测）、真实storedfilm/pressure/KE，不凭输入token给物性通过。原输出/decoder零记录保留，修订说明metadata/codec-context-correction.json。

S305补足S304观察缺口：S304 currentDuration仅在gravity applySources更新，早期chemistry/Co拒绝前未更新，因此拒绝dt打印0或旧accepted dt，原日志保留为部分chronology，不当attempt duration证明。新隔离copy common/private_backend/gpu，只在真实advanceGasTrial入口fprintf实际dt/time/native；正式源不改，使用同四原owner成功控制、全部原assertion，16失败矩阵不重跑。输出去除IO行应与正式owner stdout bitexact。compile900/run300，正常cleanup，I/O不参与性能。

Ns10扩大预算执行前实际完整controlDict核对：
```
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application CHMT;
startFrom startTime; startTime 0; stopAt endTime; endTime 2.0000000000000002e-05; deltaT 9.9999999999999995e-07;
writeControl timeStep; writeInterval 100; writePrecision 17; runTimeModifiable false;

```
原所有记录输入SHA完全一致，原基准source4b普通运行/802比较复用，1800s外部initcheck新执行须finalsummary+原checker；不改变physicalproblem。先行计划中的0.0007只为占位，完整终点以这里原controlDict为准。

新增等价host-probe配方的一cell真实GPU非零phase控制（不是修改失败8cell输入）：8cell两方向正式/旧版均300s未提交窗口，D302显示localdt约7e-15且真实signedmass，原失败和local诊断不覆盖完整成功。按已存在tests/fixtures/chmt_zero_phase_material/probe.C合法控制独立构造unitcube onecell、gas800/solid700/film800、delta.001、rho1000/cp1000/k2、liquid e0=200000、viscosity1/gas.01，Tm600正方向/900负方向。原wet/evap及已失败8cell输入不改。外部每native/旧safe/readonly300s，不因阻断无限改输入。sameinput/flags bitcompare，全physical字段/收支单列，能否native完成实测。

新增等价host-probe配方的一cell真实GPU非零phase控制（不是修改失败8cell输入）：8cell两方向正式/旧版均300s未提交窗口，D302显示localdt约7e-15且真实signedmass，原失败和local诊断不覆盖完整成功。按已存在tests/fixtures/chmt_zero_phase_material/probe.C合法控制独立构造unitcube onecell、gas800/solid700/film800、delta.001、rho1000/cp1000/k2、liquid e0=200000、viscosity1/gas.01，Tm600正方向/900负方向。原wet/evap及已失败8cell输入不改。外部每native/旧safe/readonly300s，不因阻断无限改输入。sameinput/flags bitcompare，全physical字段/收支单列，能否native完成实测。

D303原thermal full-vs-restart首差诊断：仅新增未证实Tgas seed链观察，使用原8cell/100microsteps及实际checkpoint1后50步，不改物理输入/end/precision/比较。隔离真实Backend在currentCo前后D2H读Tgas/p，打印真实conservative hosttrial/volume/referenceDt/nextDt/lastAcceptedDt/selecteddt/Co及已证明D203最终flux/packet。新增同步/IO不参与效率；只读版原全部输出bytes必须与原formal full/restart各自相等，不将两个本来不等的输出写成绿。还要检查firstmicro差与后续T/p/packet。compile900/link900/run300。

S307实际codec映射字段观察：首个wet离线逐film核算误把surface face index作为packet persistent face ID，同时global P supportImpulse符号与真实budget约定相反。保存原分析源/原错误诊断值和FAIL，不能当物理失败。新增真实codec IO字段surface.persistentId、gas/solid faceIds，依据实际packet namespace逐recipient核对；global P采用Δstock+boundary+support，无虚构reaction。reader/physical input/阈值不变；hostASan/UBSan compile900/run300，S303 physics默认零仍不代表consumedrho。

S308单cell控制support输入坐标错误：模板solid位于[-1,0]，host probe recipe却假定[0,1]，原input gas已[1.001,2.001]造成实际interface gap1.001而非.001，native输入拒绝原log/input/hash完整保留。新独立S308 copy仅将solid points和其blockMeshDict全局平移+1以恢复已声明unitcube控制；温度/厚度/物性/end/数值标准不变、原wet与失败8cell未改。非零GPU验收仍需实际完成/oldlegalcontrol/observer，外部300s，不靠放宽几何阈值。

S309补actualactivecold launch witness：S306已真实16contacted particles、perparcel/timeprofile1e-9、动态age，仅count不独立证明kernel group。外部CUDA13真实cuKernelGetName/launch ABI只打印cold kernel dims，不读取device、不改args、无额外同步；同S306原输入各CHT/FSH再运行≤300s，全部原输出bytes应相等。观察I/O不计时，helper .so源码/hash单列。codec之外host-sanitizer另用真实OpenFOAMmapper/原FSHwire解析，明确host范围不能代GPU验收。

S310 cold-start Js/history-Co缺口：只取现有host-shim test的3个真实unitcube mesh/物理输入配方，独立CUDA编译直接include完整正式Backend.cu及其原native GPU kernels/SharedGasDeviceStorage，不包含shim/stub或host模拟kernel。hostowner与device pointers分开，每次明确D2H/cudaDeviceSynchronize；真实currentCo依次Js.013→.021→.034，每值与新的nativeowner/device fresh reference bit比较实际Co/前处理数组，assert phi=-area*Js与K/omega不改；真实stage发布.034，Co对.055保留原lagged消费，下一实际stage消费.055。两variant改变rho/T/U(1/700/5→.8/900/25)和dt(1e-4→5e-5)。不人为ready，不改K/omega或生产策略。安全copy错误链已证明范围复用，本newfixture不冒称全资源恢复。compile900/run300，保留失败不调阈值。

S310首次构建命令支持失败：nvcc不识别大写Geometry.C后缀；原fatal log和argv保留。S311将同一正式Geometry.C用g++单独编译为host object再链接原S310 .cu，不改输入/source/断言/production。

S312 codec之外host ASan/UBSan：真实OpenFOAM fvMesh和未改GpuSolidPatchMapper.C，空pairlist、single、multiple(left/right)选择，实际scalar/energy两向mapper逐bit identity与无工作empty。这是native host owner/mapper/allocator析构范围，coupler入口的empty dictionary原语义不由这里关闭，真实GPU multi-selectedconsumer另列缺口。不用hostmock代GPU。FSH真实client/协议socket normal10upload frame的ASan/UBSan也单列，不链接mockbackend或宣称GPU验收。compiler900/run300。

D302 原evap900阻断追加原因定位（300s限诊断，不追加完成预算）：原正式900s已acceptedtime.00025，未达.001且无CUDA资源错误；相变8cell观察见非常小localstep，因此对同原evap输入/完整终点做只读phase/finalledger进度定位。每20s外部/proc CPUtime/readwrite统计+nvidia-smi util/memory/logsize与checkpoint时序，不写求解器、不用timing评价效率、不危险gdb attach。期望区分GPU/CPU推进、local accepted候选与macrocommitted，仅部分重现不会算PASS；formal900/block与old300/原SIGFPE永久保留。

S313 S309观察范围补充：cold字符串过滤只证明包含Cold的kernel（含relax），可能漏通用wallThermal group kernel，原result限定保留。取消外部namefilter仅print全部actualCUDA13launch name/dims；原args/kernels/同步不变，同S30616active输入，依据真实group launcher映射完成witness、原输出bytes必须相等。不是重复全套case。compile900/run各300，I/O不计时。

S313 GPU重复取消（compile记录保留，不是GPU完成）：独立源码复核确认S309已捕获的relaxColdWall2DParticlesToResidentGasKernel就是实际thermal group wrapper，8radial lanes/particle、block128与16active directory entries对应16完整groups，内层调用advanceColdWall2DThermalGroup后更新pT/墙热。结合S306全部active/age/profile面积即可形成有效witness；不因函数名误解重复运行全名observer。新增实际GPU只执行S311 Js未覆盖链。

最终收口：86实际命令，19限定单元(12PASS/2FAIL/1BLOCKED/4NOT_RUN)，17旧有效单位按原source/dependency范围复用；这是scope counts，不是全部正式GPU数值通过数量。Ns10full700步finalinitcheck0，原evap900s .00025/.001阻断；wetglobalP/restart严格红/提交fixture完整负例语义和所有旧gap IDs保留。S311ConstantTransport Js范围PARTIAL，S309真thermalwrapper16groups纠正原派生2但保留原JSON，S313onlycompile取消无必要GPU重复。所有1155sourceblob/bytes和86rawlog hashes verified。
