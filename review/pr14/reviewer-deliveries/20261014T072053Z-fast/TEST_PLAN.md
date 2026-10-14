# PR14 第二轮独立 FAST 计划

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

D执行细则：从三固定commit git archive提取applications/common/tools，每版额外Backend observer patch生成并原生构建900s；只记录已有host下载，不增加device copy/算式；同旧thermal-full/同mesh。全部checkpoint/CSV/geometry逐字节对该版未observer输出；baseline/tested使用旧不可变结果，formal使用本轮thermal-full。

D执行细则：从三固定commit git archive提取applications/common/tools，每版额外Backend observer patch生成并原生构建900s；只记录已有host下载，不增加device copy/算式；同旧thermal-full/同mesh。全部checkpoint/CSV/geometry逐字节对该版未observer输出；baseline/tested使用旧不可变结果，formal使用本轮thermal-full。

S202支持修正：原chem fixture把canonical DeviceState也放managed内存，WSL首异步kernel后host读触发SIGSEGV（CUDA-GDB已定位）。仅按真实生产host/device分离owner及同步发布，原kernel/owner/assert保持；300s复验，失败原源/二进制/日志保留。

A新增S203完整owner witness：S202原夹具没有调用实际ComputeGasCourant，targetMaxCo默认0，原强制两accepted assertion失败。保留红；新夹具调用真实native Co preparation，target0.5和原4e-6/物性/几何/maxSteps1均不变，观测全部acceptedpartition（允许真实CFL缩步），必须native完成整区间、逐字节等价显式同partition、public terminal对照和原1e-6chemical单kernel对照。不是扩大数值阈值；300s。

Ns10编译预算变更：第一次完整原生编译因900s外部deadline终止（多大CUDA translation unit并行期间），原完整log/command保留，非数值缺陷。完成其他编译及Ns2非profiling样本后，Ns10单独重复同源/同flags/同入口，外部预算1800s；不改变求解器或数值case预算300s。

D追加同目录baseline无observer对照：原不可变baseline preflight的三个checkpoint hash不匹配，新trace CSV/geometry完全匹配。为区分路径/元数据与observer干扰，保留全部trace输出，将同case原输出重命名后，用既有真实d611原版binary于完全相同case路径/输入重跑一次（300s，不计效率），比较全部bytes并使用该版原checker。

续跑环境确认：用户报告死机后，114条完整结果及日志哈希一致、无孤立spec，无生产源码diff。追加一次既有真实Co二进制健康观察，单独标为环境确认，不增加验收覆盖。此前SST-RK实例实际ddt=Euler，保留原结果；新增仅ddtSchemes默认配置SSPRK2/SSPRK3的两个短程实例，热力参数、终点、阈值与原thermal保持一致，检验真实SST累计审计/实际stage packet及完整库存。Ns10补原chemistry-full initcheck，保留相同输入/步数/flags。均外部300s，GPU串行。

续跑环境确认：用户报告死机后，114条完整结果及日志哈希一致、无孤立spec，无生产源码diff。追加一次既有真实Co二进制健康观察，单独标为环境确认，不增加验收覆盖。此前SST-RK实例实际ddt=Euler，保留原结果；新增仅ddtSchemes默认配置SSPRK2/SSPRK3的两个短程实例，热力参数、终点、阈值与原thermal保持一致，检验真实SST累计审计/实际stage packet及完整库存。Ns10补原chemistry-full initcheck，保留相同输入/步数/flags。均外部300s，GPU串行。

S206诊断支持修正：保留原ACTUAL-MOVING-CHEM-VOLUMES零记录FAIL及全部输出完全相同的结果。nm确认本机CUDA13生成代码实际调用__cudaLaunchKernel(cudaKernel_t,...), 而原观察器只挂接cudaLaunchKernel(constvoid*,...)。使用已安装crt/device_functions.h真实签名和cuda.h/cuKernelGetName记录实际handle对应kernel名字/参数。新增只读8cell化学volume D2H与RKblend参数观察，不改变参数/算式；额外同步/IO不纳入效率。原移动chem输入/步数不变、各版CSV/checkpoint/geometry应全bytes相同。观察器编译900s，实际案例300s。

D checkpoint补充只读核验：实际baseline trace版本与同case原binary的3个checkpoint全bytes比较失败已保存。依据真实checkpoint格式逐段检查header/config/payload，量化仅编译sourceFingerprint及其checksum的变化；禁止重写元数据/原始输出伪造bitwise通过。新限定结果与原全bytes FAIL并列。

S205 Co新增证据：原fixtures不断归约同数据，不能证明临时owner不缓存旧Co。追加同owner跨1→32→513→32768→513→1→empty容量周期、改变每轮实际device Co/status/NaNpayload，并交错另一owner/default与nonblockingstream；每轮对真实CUDA无缓存Thrust归约bit比较。生产operator/header完全相同，无容差。Ns2/Ns10仅编译speciesflags分别记录，initcheck单独观察300s。各合法case独立hostinput及device数组。

Ns10完整chemistry-full initcheck实际300s外部超时，缺少完整sanitizer结尾，标BLOCKED；原5秒chemistry-control输入保持原样另做initcheck作为Ns10受影响Co路径有限证据，不把它冒充full原input sanitizer通过。正式thermal中点checkpoint原生恢复复验：从本轮formal完整100步checkpoint-1恢复至同原end，检查所有terminal CSV/checkpoint bytes；不沿用旧source断点冒充本轮restart。外部300s。

D202动网格证据补充：正式frontend固定retainStageGeometry=false，仅checkpoint保留最后微步两stage，不能据此关闭全部GCL。隔离复制CHMT.C只启用已有gasWindowStageGeometry记录功能、打印原始old/new/evaluation/swept数组，实际GPU Backend.o仍为本轮正式原对象，完整物理源码/headers/flags未改。追加hostlink900s，moving-GCL诊断300s并逐bytes比较所有原输出；在原Ns10 movingchem input也执行同一host诊断以配对S206真实1400化学launch参数。原始profile/性能不使用诊断IO。

正式thermal原生restart全bytes发现：gas/exchange/checkpoint与full不同，accepted/material相同。先保留red；用802原binary从同本轮checkpoint1复验同输入，判定是否这轮Co改动回归。跨版本对照不扩大容差。SSPRK2/3既有D201formal observer binary再运行相同输入，记录真实packet_dt/微步dt阶段权重，所有原输出应全bytes相同；stageCSV聚合stage=1本身不能证明Euler。各300s。

D203实际阶段收支证据：D201 wall observer记录的是applyCoupledFaces的未加权dt，SSPRK观察确实有200/300stage且全部原输出bytes相同，但把该dt当ledgerDt的新增断言失败，保留red。这不表明RK物理路径失败。隔离CUDA Backend在accountCoupledFaces已下载最终限幅face flux后只打印真正ledgerDt、最终flux、实际scaledpacket及gas accepted库存，零新增D2H、零算式替换。结合D202纯hostgeometrymain输出核验实际1/2+1/2与1/6+1/6+2/3和气体/配对库存。仅编译Ns2 Backend900s/link900s，actualcases300s，原输出必须bytes相同；Ns10化学volume仍复用S206，不重复完整化学。

F物理范围新增实例（此前NOT_RUN的原gap IDs）：保持原thermal的8+8cells、原终点0.001/2macro×50micro和原阈值，新增wet-film（rho1000/cp1000/初温350/厚1e-5）、wet-evaporation（HertzKnudsen单species accommodation1e-4/saturation200000，CoupledRecession）、dry-ablation（表面C0→S0，A1e-5/order1，CoupledRecession），分别实际Native GPU气体+原CPU材料/膜owner，不运行CPUshim。用S207实际原checkpointcodec扩展输出film/particle/完整packet及speciesbudget以独立80位库存核算，未改codec和阈值。
CHT enabledradiation/mixedcoupler原缺口：以既有cold2D输入的真实blockMeshDict重新生成原.04×.04×.01十cell流体mesh，不继承历史thinmesh；配置匹配下方solidregion十cell/.001m、真实oneToOneConformal impactWall↔top，其他fluidfaces uncoupled，实际solid rho1000/Cp800/k10。一个disabledradiation对照、一个enabled Mie DO实例（真实table backing，particleCount/源消费和场写出检查）。gas/particle原物性、原终点1e-5和原cold阈值不变。该新增物理mesh不是原F004输入修复或原结果替代。所有nativecases外部300s。empty/多selectedface mapper未据singleface推定覆盖。

独立审查后A说明修正：正式SharedGasTrialFields.H targetMaxCo默认0.5，本计划及S203 metadata中“default0”是审查者说明错误，不是生产事实。真实S203输出target0.5，测得原4e-6 Co约1.087...，一半仍超过0.5，故原强制两个partition不成立；native Co preparation仍应保留。S203 compare只覆盖partialaudit/status，不能声称全部auditbytes；新增S208逐一bit比较speciesMassChange与所有realaudit成员、整数计数，以及所有status语义成员（不比较padding），保持全部原物性/终点/成功规则/16失败组合；真实完整native夹具compile900s/run300s。S203原成功/红记录保留。

F原始setup红保留：新wet-film/evaporation初始gas/solid共面，没有厚1e-5film的真实空间，正式attachCoupledSurface报告不匹配film-offset vertices，未启动GPU；S209只把gasmesh整体沿真实solidoutward+x平移原厚1e-5，保持cells/volumes、全部物性/终点/阈值原样，记录实际pointsdiff/hash；这是新合法测试输入，不能把旧无film空间输入标通过。
新CHTcoupled/radiation缺少实际消费0/thermalExchangeState，原版按IO报错退出；S210按当前真实reader及初始state零ledger语义补入initialState=true、exchangeSequence0、两个clock0的初始dictionary，保留原错误，未修改任何生产reader/门禁/数值。重试各300s。

D204新wet原红与最小诊断：合法ThicknessAveraged、melting=false默认phaseCondensed=-1。正式native湿算例两次SIGFPE；独立源码确认CpuMaterialDriver无条件phase.mass/rho[-1]，即使zero-mass热交换包仍消费不存在材料。该源与802相同，非本轮新增回归。既已知潜在OOB，不再次执行原错误路径来计数或在线gdb；只查现有失败/core。隔离host source将零phase mass的体积设0而不消费unselected backing，不变其它算式、不启melting、不改物性/end/阈值、不新增productiongate。所有原版失败持续FAIL；诊断hostlink900s/同wet输入300s、原GPU Backend.o保持正式hash，修后账目按原标准独立审计。正式源码不改、开发修复未完成。

B多个完整thermal组的证据缺口：原fixture四粒子逐thread初始化不能证明整8lane求解。新增CHT/FSH同原cold2D物性/d=.005/终点1e-5/dt1e-6，在原每cell尺寸(.04,.04,.001)保留的4×4×10真实mesh横向复制成160cells，首层16cells各一个原parcel（原epsilonS与mass），按已有16个独立parcel触发至少两个完整8lane组；其他层粒子库存0，非uniform字段从原首cell逐份复制，不改阈值。原warm相区2373.15沿用，FSH仅对应原有效wall tokens1800/710/100和application。逐输出时刻全部profile doublemean+carry/FSHdouble与pT abs1e-9。新几何是组覆盖输入，不是原fail输入替换。外部300s。

S212独立逐parcel校验补充：旧S201/S204 checker通过case名字coldWall分支选择profile断言；新S211名字cold2D只触发有限性检查。保留原调用及其有限性证据，原native记录不得声称该调用已做profile。新增显式无名字判断的每parcel/每时刻double顺序累加profile、doublecarry（FSH现存double无carry字段按0）、原caloric inverse及1e-9阈值；同时核真实particle_count和动态contactAge/状态，不能仅用planned16计数。CHT真实coupler/DO也使用同物理字段校验，并逐写时积分真实壁面4个heatflux与solid库存。所有只是现有真实GPU产物分析，不新增GPU运行。

S213整组输入校正：S211实际每时刻只有1particle，旧P003提供的初始restart覆盖了新的epsilonS，不能当16组验收。保留该范围失败，S212逐parcel一致性数据仍有效。新独立初始文件通过未变current serializer/P003用途，将原合法单parcel状态复制16份：同mass/d/T/velocity/profile/carry，将各点置于160cellmesh底层16个不同cell中心x/y、同原pz；cell IDs0..15和原seed状态分别记录。没有扩大ε/物性/终点/阈值，显式16个真实输入与场库存相符，原文件/hash保留固定上轮证据。各native300s，要求实际每输出count16及所有profile1e-9；需观察真实contact和完整8lane组，不以输入计数推断运行。

S212独立逐parcel校验补充：旧S201/S204 checker通过case名字coldWall分支选择profile断言；新S211名字cold2D只触发有限性检查。保留原调用及其有限性证据，原native记录不得声称该调用已做profile。新增显式无名字判断的每parcel/每时刻double顺序累加profile、doublecarry（FSH现存double无carry字段按0）、原caloric inverse及1e-9阈值；同时核真实particle_count和动态contactAge/状态，不能仅用planned16计数。CHT真实coupler/DO也使用同物理字段校验，并逐写时积分真实壁面4个heatflux与solid库存。所有只是现有真实GPU产物分析，不新增GPU运行。

S213整组输入校正：S211实际每时刻只有1particle，旧P003提供的初始restart覆盖了新的epsilonS，不能当16组验收。保留该范围失败，S212逐parcel一致性数据仍有效。新独立初始文件通过未变current serializer/P003用途，将原合法单parcel状态复制16份：同mass/d/T/velocity/profile/carry，将各点置于160cellmesh底层16个不同cell中心x/y、同原pz；cell IDs0..15和原seed状态分别记录。没有扩大ε/物性/终点/阈值，显式16个真实输入与场库存相符，原文件/hash保留固定上轮证据。各native300s，要求实际每输出count16及所有profile1e-9；需观察真实contact和完整8lane组，不以输入计数推断运行。

S212独立逐parcel校验补充：旧S201/S204 checker通过case名字coldWall分支选择profile断言；新S211名字cold2D只触发有限性检查。保留原调用及其有限性证据，原native记录不得声称该调用已做profile。新增显式无名字判断的每parcel/每时刻double顺序累加profile、doublecarry（FSH现存double无carry字段按0）、原caloric inverse及1e-9阈值；同时核真实particle_count和动态contactAge/状态，不能仅用planned16计数。CHT真实coupler/DO也使用同物理字段校验，并逐写时积分真实壁面4个heatflux与solid库存。所有只是现有真实GPU产物分析，不新增GPU运行。

S214输入cell/ID实际字段校正：S213旧夹具假定pcell/pCell/pid字段名，真实current schema字段为cell/orig_id，因此实际16个不同位置却都绑定cell0，运行只保留3且无coldcontact，原count/profileFAIL保留为错误输入支持证据，不归为生产F004回归。按真实serialized field_specs改为cell0..15/orig_id1..16；其它所有S213已提交newinput量完全不变。初始profile记录实际取本轮S2111e-6真实GPU产物（不是旧ASCII输入直接解码），已在输入源SHA记录更正。新实际source/mesh/parameters不改、无新增数值条件，300s；要求实际count16和每parcel/time original1e-9及实际contact数。

S212独立逐parcel校验补充：旧S201/S204 checker通过case名字coldWall分支选择profile断言；新S211名字cold2D只触发有限性检查。保留原调用及其有限性证据，原native记录不得声称该调用已做profile。新增显式无名字判断的每parcel/每时刻double顺序累加profile、doublecarry（FSH现存double无carry字段按0）、原caloric inverse及1e-9阈值；同时核真实particle_count和动态contactAge/状态，不能仅用planned16计数。CHT真实coupler/DO也使用同物理字段校验，并逐写时积分真实壁面4个heatflux与solid库存。所有只是现有真实GPU产物分析，不新增GPU运行。

收支收口：保留S207完整momentum严格红（SST约1e-12..8e-12，diagnosticfilm perFilm约1e-13），不扩大阈值或忽略异常。另按既定SST k/omega库存、质量/能量、actualRK stageweight和配对recipient范围建立独立限定验收，不能冒充fullmomentum/三residentapps通过。
CHT wall-only初始账少了真实solid sideWalls固定300K外部导热，保留不完整核算红；新增全边界账以实际每输出solidT/实际300Ksideboundary/原κ10、真实orthogonalcell几何算40sideface conduction（每faceA=.04*.0001、dc=.02），Euler末时刻权重，与四个实际GPUwallheat渠道同时积分至全部时刻。已有energy1e-8绝对/相对标准不变。实际缺项功率约56W对应κ10，可独立核验消费物性。

## 最终收口

本轮221条执行记录，25个限定验收单元(18PASS/4FAIL/2BLOCKED/1NOT_RUN)，跨主题不叠加；18PASS包括明确标识的诊断、host sanitizer和资源统计，不是18项正式GPU数值通过。未受改动影响的36旧单元只按原范围引用；复用表、14旧gap ID、所有范围红/支持红/原版与诊断红完整保留。覆盖矩阵按旧gap实际topics映射，不把无关缺口挂到所有主题。完整生产源码1152路径/所有221原始log hash已校验。没有生产长算。
