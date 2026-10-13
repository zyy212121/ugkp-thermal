# PR #14 独立真实 GPU FAST 审查报告

结论：**目前不建议合并。** 固定原版存在化学 native trial 接线缺陷、冷壁逐面物性 producer 缺口、首次 float profile 初始化焓余量丢失，以及 Co 归约结果尾部未初始化复制。另有微小 FP64 成功路径差异需要开发者解释。隔离诊断补丁的通过不关闭原版发现；没有开发者正式修复版本或正式复验。

本批 `20261013T182722Z-fast`，结果截至 2026-10-13T21:01:08.437905+00:00。执行296条已记录命令，所有33个主题在测试前读取、建立计划/覆盖矩阵；当前矩阵逐项列出运行范围、历史证据、未测项、文件与函数定位。没有生产长算、合并、README修改、PR Conversation评论或PR描述修改。

## 版本、环境与计数口径

实际测试原版：`802a11e7dfe986488a8bf8305e86424dcc94c9e1`，tree `db2863158139842f6c8221c0da11ba640f381336`。最后生产源码 `a73b9635510fb09d7baa5339e14d1aa609f0ff87`；现场确认它至固定head的生产目录无差异。基线 `d6110c02745a8ad726208a426c4bb3a28233c1fd`。实际PR分支 `perf/chmt-first-round-20261013`，推送前fresh API/fetch仍与固定head相同。[版本](versions.json)、[远端核对](publication-prepush.json)、[身份逐路径核对](artifacts/identity-audit.json)。8115个tracked路径的Git对象/模式一致；628个source-index pin、497份历史证据哈希全部匹配。历史CPU/源码/编译证据只按原范围复用，未改称GPU验收。

Windows/WSL Ubuntu22.04，OpenFOAM Foundation10，GCC11.4，CUDA13.1.115，驱动592.00，RTX4060 Laptop 8188MiB/sm89。独立worktree、安装目录、显式应用/backend路径；新建原版CHMT Ns2/Ns10、gasUGKP+backend、FSH+backend、CHT的完整构建/链接/原生运行。gasUGKP此次child退出0，补齐旧wrapper退出2的证据缺口。[环境](environment.json)包含实测工具版本、flags、库路径；[二进制/库哈希](versions.json)。CHMT O2，CHT O3/fmad=false；gas/FSH O3默认fmad；各比较双方采用同flags。

以下为124个**独立、限定范围的验收断言/参数配置**，排除编译/setup、重复计时样本、重复故障尝试。PASS仅表示该条写明的范围；构建通过与GPU数值通过分开。33主题之间共享单元，不能相加。

|原版PASS|原版FAIL|诊断补丁PASS|BLOCKED|NOT_RUN|
|---:|---:|---:|---:|---:|
|86|9|14|1|14|

原版9 FAIL拆为4个生产/传输发现、1个严格跨版本FP64对照、4个过时测试支持单元。4个支持单元是一个checkpoint checker常量问题和三个应用的同一旧冻结oracle问题。并非9个独立生产缺陷。[results.json](results.json)同时保留逐命令状态、原始spec、原始失败和最终验收单元，编译与环境失败未混算物理FAIL。

## 必须正式处理的发现

### GPU-F001 / P1：nativeRefinement 没有路由到既有 Trial

位置：`common/operators/advanceGasChemistryKernel.cuh::advanceGasChemistryCell` → `advanceClosedReactor` / 既有 `advanceClosedReactorTrial`。Ns2实际kernel，完整device thermo/mechanism/state/audit/budget backing；S0→S1、同MW=.028、Cp=1000、rho28、T700、V=.001、dt2e-6、initial/maxStep1e-6、maximumSteps1，native=true。

基线返回StepLimit/code10，cellStatus8，species[28,0]和此前audit哨兵保持；PR原版返回Success，写出两次accepted化学步、species `[27.999888000223997, 0.00011199977600030318]`、integratedTime2e-6及真实budget。真实kernel复现，既有Trial接口在原版无生产caller。其问题是原有native内层refinement语义丢失，不能以public终端完成替代。

证据：[原版红](logs/CHECK-CHEM-RETRY-RED.log)、[基线](logs/CHEM-RETRY-base-step-limit.log)、[原版](logs/CHEM-RETRY-pr-step-limit.log)、[P002](patches/P002-native-chemistry-trial-routing.patch)、[补丁绿](logs/CHECK-CHEM-RETRY-P002.log)。成功控制/ordinary的完整解码数值与整数相同；没有把该JSON比较说成raw bitwise。false公开终端分支保留继续完成。独立计时夹具另有原成功完整publication bit hash对照。

P002仅隔离恢复既有Trial路由。**不是整合就绪修复**：真实不可用thermo/model分类仍需审计（Thermo InvalidModel可能包装成TemperatureFailure，当前最小patch的分类不充分），实际完整owner请求区间retry/rollback序列仍未测。不得新建整步数值拒绝机制。

### GPU-F002 / P1：cold2D逐面壁热惯量没有有效 producer

实际CHT：10 cells、一个5mm热Al2O3 parcel、pm=.00019054628487089897、初温2373.15K、合法域内位置、撞壁face9、dt1e-6、10步；solidRegion/coupling disabled，壁温473.15K。原版退出0，但t2e-6第一次非零接触热步开始pT、64个float焓节点和double carry全NaN。不是缺失存储或CUDA访问错误；原版mem/init/sync均summary0。

P006仅加device printf，观测 face数组effusivity=0、既有scalar fallback=11304.866208850062（device默认1800/710/100），age0、wetArea1.4463267383218867e-7、wallG=NaN。`acquire/initialiseParticleWallEffusivity`分配非空数组并清零；host仅为实际coupled patches上传，disabled场景无上传；consumer只按指针非空选数组，隐藏scalar fallback。sqrt(pi*0)/0导致首NaN，paired mean correction传播至全部64节点。

该输入中的7900/500/15三个旧wall token当前**无host consumer**，不能声称这些物性已到device。原base先在旧token准入处退出；P008删除只在PR未消费的旧token、显式选base已有FP64，同实际消费参数双版本重跑：base在首步后CUDA launch failure，当前继续写NaN。故这是既有producer缺口被新清理暴露，**不是已证明的原成功FP64回归**；不能恢复旧positivity trap来“修复”。

证据：[原始全输出](artifacts/resident/CHT-coldWall2D-INPUT2)、[定位](logs/CHT-COLD2D-P006.log)、[旧版本原生对照](logs/COLD2D-INPUT3-baseline.log)、[当前对照](logs/COLD2D-INPUT3-original.log)、[P007](patches/P007-cold-wall-fallback-producer.patch)。P007初始化producer使用既有scalar fallback，后续coupled patch覆盖保留。它恢复有限输出，但未解决首次carry问题。正式修复需确定uncoupled墙体物性来源及混合face/reconfigure/copy失败恢复契约。

### GPU-F004 / P2：初次profile cast丢失焓余量

`common/operators/clearColdWallParticleState.cuh::initialiseColdWallParticleState` 和 `common/wall/GpuColdWall2DDevice.cuh::initialiseColdWall2DParticleState`。t1e-6，pT2373.1499520863576对应double焓4191779.3873143774；存入float profile4191779.5，却未把差额放入现有double carry。温度与 `mean(double(profile))+carry` 独立反演差7.478092084e-5K，parcel能量偏差约2.1472e-5J。应保留此可量化差异，不能放宽1e-9K一致性阈值掩盖。

初版独立checker误用NumPy float32 mean，又制造了2D后续时刻误差；P009改为double累加，原失败日志保留。修正checker后仍有真实首次cast偏差。[正确的红](logs/FIELDS2-RED-CHT-coldWall1D-INPUT2.log)、[P007仍红](logs/FIELDS2-RED-CHT-coldWall2D-P007.log)。

P010保留原float cast/公式，只把cast损失加到既有double carry，不加数值门禁。[patch](patches/P010-cold-profile-initialization-carry.patch)。P007+P010同输入1D/2D全部时刻的实际profile/carry/pT一致性通过；1D100步进入wet导热、2D真正restart全字段byte equality通过，四san工具0。[1D绿](logs/FIELDS3-CHT-coldWall1D-P010.log)、[2D绿](logs/FIELDS3-CHT-coldWall2D-P010.log)、[wet100](logs/FIELDS3-CHT-coldWall1D-P010-WET100.log)、[restart](logs/COMPARE-RESTART-CHT-coldWall2D-P010.log)。仅限该诊断版本；共享FSH模板/墙气ledger全局收支仍需正式复验。

### GPU-F005 / P2：Co归约结果的7字节padding被未初始化复制

原版CHMT thermal100微步，initcheck退出86、summary100。同一16byte result从device回host未初始化。`Backend.cu::currentGasCourant`的Thrust transform_reduce结果 `{double,bool}` size16，但仅9字节成员被写。只读host/SASS复核：D2H长度0x10，device STG.E.64与STG.E.U8只写value/bool，host只消费8byte value和1byte bool；该日志**不证明Co/Js物理值未初始化**。

[原红](logs/SAN-FINAL-CHMT-thermal-initcheck.log)、[只读复核](artifacts/static-review-followup.md)、[P011-v2](patches/P011-Co-reduction-result-no-padding.patch)。v2保持bool与offset/size/max/OR/遍历，显式初始化原7个尾字节。100步原生initcheck0，所有CSV、checkpoint、几何与原版逐byte一致：[绿](logs/CO-P011-initcheck.log)、[全文件hash/比较](artifacts/Co-P011-initcheck-full-publication-comparison.json)。v1 uint64槽的诊断编译失败保留，未归成原PR失败；其具体编译原因未进一步定性。Ns10及其他配置需要正式复验。

### GPU-F003：正常成功路径的微小FP64差异待解释

相同Ns2 thermal、sm89/O2 flags、50+50微步、两次material solves、全部CSV行数相同。base/PR gas质量/能量/T/p/material相同；momentum最大绝差5.421e-20、packet热量1.869e-16（相对1.25e-13），严格相等检查失败，既有物理checker容差通过。[逐文件差](artifacts/performance/native/preflight-comparison.json)。

静态候选是absolute knot/time → compensated local WindowTime及stage alpha路径，不是已定位的首次分歧。需逐步dt/stamp/alpha hex追踪及开发者明确批准语义；不得笼统以“优化”解释所有变化。本批不把它称为显著精度劣化，也不称全成功FP64路径保持已验收。

## 已执行范围与保留的限制

CHMT17个真实原生小case覆盖single/frozen thermal/diffusion、Ns10化学、完整/分段/重启/细化/禁用控制。气/固各8cells、4接口face，真实packet consumer mask3；质量1e-10kg、能量max(2e-5J,2e-13初始库存)、EOS/caloric abs1e-7/rel2e-10、elements1e-9、restart abs1e-10/rel5e-12、热量细化0.5%等原标准保留。P001只修测试checker fingerprint，未改阈值/断言/生产checkpoint。真实输出[CHMT artifacts](artifacts/chmt)。两个化学window实测11/689微步，不能由CSV就宣称完整两half audit lineage可见。

gasUGKP/FSH/CHT：真实4cell周期输入、128parcels、10步，固定seed。初版cell/orig_id列配置错误与schema/必需字段错误全部保留；PHYS1纠正owner与ID，不改变物性。全部实际粒子成员有限及terminal写出、真实restart最终所有粒子成员/RNG和场/残余/epsilon-history byte equality通过。**完整动量/能量阶段收支没有验收**；已存所有原生字段与观测，不以快照瞬时积分充当实际阶段权重/源/最终通量预算。[resident artifacts](artifacts/resident)、[逐字段restart](logs/COMPARE-RESTART-gasUGKP-periodic-PHYS1.log)。

共享CUDA：三应用core原memcmp对照、压力通量限定后的实际粒子/Eulerian质量动量能量、collision/drag/contact/pool生产与消费、18个AUTO目录/预构建/block配置。旧冻结oracle f03e215早于base d611已含的slip-wall thermal修复c73591f，旧f21rho1.625与真实3.25/500=.0065不适用；P005仅补已有base公式，原memcmp保留。旧cold段仍在源中，CORE明确不执行，不拿它当cold证据；原full失败保留。mock direct_dispatch未运行、不算物理证据。

SAN scope：[原版20条基础san、CHMT3条](results.json)及最新cold诊断4条、P011 initcheck。original CHMT initcheck红之外各原版工具0；original cold2D数值FAIL与san0并存。没有ASan/UBSan主机验收，也没有通过故意坏索引破坏环境。

物理存储检查：实际CHT restart schema9仅8/64焓profile为float32，pT、carry、接触/凝固/history及其他物理字段double；FSH schema8/gas当前粒子物理字段double。timer float及编号类型单列。编译/输出证据不替代所有tracked分支的全行GPU审查。[source review](artifacts/static-review.md)也明确没有声称全426生产文件逐行通过。

## 实测效率和代价

同硬件/输入/work/flags，3warmup、7repeat；编译和profiling不计入重复墙钟时段。笔记本时钟未锁定，WSL/启动/CUDA context/I/O噪声不作工程提速结论。

|测量边界|base中位|PR中位|PR/base|离散/范围|
|---|---:|---:|---:|---|
|Ns2实际化学kernel，1cell|2.731ms|0.833ms|0.305|base2.731–2.747；PR.831–.845ms|
|同kernel，32768cell|34.726ms|11.290ms|0.325|base28.173–37.221；PR11.262–11.311ms|
|32768cell完整reset/launch/wait/full-D2H pipeline|33.624ms|11.658ms|0.347|kernel/event时间和CPU pipeline时钟不同，不相加|
|CHMT两window整进程墙钟|1.868s|1.619s|0.867|base1.717–2.270，σ.203；PR1.167–2.471，σ.398s|

Ns2 kernel全部cell的保守字段/状态/audit/budget/解析一阶反应残差合格，完整publication FNV bit hash一致。源中少重复机制验证、正交基/leftInverse prepare和caloric inversion，可解释方向，不能由调用数推出幅度。32768cell host-launch中位10.176→8.534µs，reset H2D77.003→72.979µs，full D2H1306.412→1248.880µs，wait32052.304→10306.505µs。19个固定夹具分配相同，分配后的device allocator delta8MiB相同；这不是完整峰值VRAM。cuobjdump REG均255，STACK1920→3632byte/thread，临时栈反而增长，不外推Ns10或全部求解器。[全部样本和阶段](artifacts/performance/chemistry/summary.json)、[资源原始日志](logs/resources-perf-chem-original.log)。

原生wall-clock中位低13.34%，但样本离散重叠且启动开销大，**没有建立稳定整求解器提速**。Nsight同case实际轨迹记录更多真实成本（时间受profiler开销，不做速度比）：

|调用/流量/分配|base|PR|
|---|---:|---:|
|kernel launch|2200|3000|
|cudaMalloc / cudaFree|139 /139|339 /339|
|cudaMemcpy|12902|11702|
|cudaMemcpyAsync|800|200|
|cudaStreamSynchronize|800|400|
|cudaDeviceSynchronize|400|400|
|实际D2H copy数 /bytes|7500 /876800|5700 /732000|
|跟踪cudaMalloc活跃字节峰值|22264|22281|

少复制/部分同步，与更多kernel/临时分配并存。当前transform_reduce/reduce临时分配值得用owner内可复用storage审查，不能仅说删状态工作即提速。peak仅是相关API实际跟踪buffer，不含完整driver context、local stack及其他显存。NVTX主要是Thrust/CUB，未给出内部coupled window CPU范围；不按两window把整进程时间机械除二。[完整API/kernel/copy/memory tables](artifacts/performance/nsys)、[成本汇总](artifacts/performance/nsys/cost-comparison.json)、[7样本native计时](artifacts/performance/native/summary.json)。raw trace/sqlite本地保留完整hash/生成命令；交付CSV/JSON足以复核汇总，不上传大profiling二进制。

## 支持补丁、未测和正式复验

源码诊断：P002 native Trial路由；P006只读printf定位；P007 wall默认物性producer；P010初次profile余量；P011显式尾部metadata。P007+P010联合通过只在两份diff合用的隔离版本。其余P001/P003/P005/P008/P009/P012属于checker/input/oracle/serializer支持，绝不说生产改动已合并。所有旧失败、patch版本/diff/hash和新结果均保留在[patches](patches)及[logs](logs)。

14个NOT_RUN条目逐项登记于coverage/results，包括active SST22 owner迁移/配置/资源错误、SST/RK预算、真实动网格GCL、独立cold-start Js、化学完整owner retry和half audit、wet/film/烧蚀/产气、enabled radiation/empty或selected mapper、mixed-wall物性、FSH cold、host sanitizers、安全分配失败恢复/并发、resident完整实际收支、FP64 clock首分歧、正式修复复验。旧源/编译证据不足以关闭这些GPU缺口。

开发者下一步：先正式反审并修复F001/F002/F004/F005（保留原成功路径、无新数值门禁），给出可定位commit；针对四项原红证据及Ns2/Ns10/共享前端影响复验，解释F003 clock差。随后补上述FAST缺口，才评估PRODUCTION小规模起步/基准实验/真实工况；本批没有启动这些长算。

完整复现入口：[REPRODUCE.md](REPRODUCE.md)、[TEST_PLAN.md](TEST_PLAN.md)、[commands.txt](commands.txt)、[SHA256SUMS](SHA256SUMS)。原工作恢复副本留在test/pr-review，未删除原研究结果。发布文件只包含本批及必要审查索引/回复，生产源的诊断修改以patch文件交付。记录提交的最终SHA和远端固定可读链接在本次对话返回；包含自身的commit无法先写入自己的内容，不以自引用伪造发布身份。
