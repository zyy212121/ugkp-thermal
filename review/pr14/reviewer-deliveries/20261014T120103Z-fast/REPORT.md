# PR14 第三轮独立 FAST 源码审查与真实 GPU 复验

PR14_REVIEW_ONLY / FAST。**正式修复的指定范围已取得真实 GPU 证明；整体 FAST 验收仍未满足，不可视为合并就绪。** 原 evaporation 终点未完成、wet 完整物理动量与严格 restart 位一致性仍开放。生产源没有被审查者修改，没有 PRODUCTION、force push、合并、README 或 PR Conversation/描述修改。

## 固定身份与执行范围

正式生产：`9d5be6eff69dbced2fc959d422d3ba419a1bc39e`，tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`。记录父提交：`a462edd16c20750a5875548f2e4ec690602d05cc`，记录所在固定提交在本目录的 Git 历史和最终交付消息中单独给出；不能把记录提交当生产修复。上一轮生产 `4b5c480b572f234ddecf5e89164fe59b0d2e9735`，不可变证据 `f5e6c5e2cbde2693f5fa811d874c200b07bbbc66`。本轮目录 `review/pr14/reviewer-deliveries/20261014T120103Z-fast/`。

五个 source/test 文件实际差分逐 blob/SHA256 核验。生产只改 CpuMaterialDriver 零 phase mass 的体积表达式：零质量不消费 unselected condensed[-1]，热包/力/forcing 与非零原表达式保留。1155 个 source/dependency 文件完整哈希见 metadata/source-hashes.json。CHMT CUDA/common 448 依赖 regions 未变，旧 Ns2/Ns10 CUDA 对象哈希复核后复用；正式全部 host 源重编译、最终 native 链接。CHT/FSH 完整依赖未变，原 4b 实物 binary 复制到本轮 test 安装使用，producer 不改标为 9d。所有路径 resolve 留在授权 test 区。

RTX4060 Laptop 8GB / sm89 / driver592.00 / CUDA13.1 / OpenFOAM10 / g++11.4。普通例 300s；原 Ns10 initcheck 1800s、原 evap 900s 已先写计划，同原输入/完整终点/flags；单 GPU 顺序执行。所有实际 argv/cwd/UTC/exit/timeout/input/binary/source 哈希在 results.json 和 logs/*.spec.json。原始日志单存，跨 topic 复用同 evidence ID。内建 build-info 的 legacy solver_commit/source_fingerprint 报 UNKNOWN，如实保存；固定源身份由外部 blob/hash/实际构建命令及 binary/object 哈希证明，不新增身份门禁。

本轮 86 条实际命令，原始命令状态 {'PASS': 65, 'FAIL': 13, 'BLOCKED': 8}；19 个限定单元 {'PASS': 12, 'FAIL': 2, 'BLOCKED': 1, 'NOT_RUN': 4}。包含编译、host sanitizer、支持/只读诊断、原始支持失败，**这些数不能当正式 GPU 数值通过数量**。17 个未受影响的第二轮单元按原范围固定引用，不重跑全套、不跨主题累加。

## 正式化学夹具与补充证明

正式 fixture 独立生成/新 nvcc/真实 CUDA，无 S202/S203/S208 overlay：16 原失败标记与四类原 owner/terminal/requested1e-6/direct1e-6 控制均完成；positive transaction/audit/status/budget/host ledger 原完整断言通过。真实 targetMaxCo=0.5，原 4e-6 owner 8 begin、4 commit、4 rollback，accepted 为四段而非硬编码两段，总 accepted duration 精确覆盖原 4e-6；最后一段由原剩余量形成，hex 全保留。

S305 在隔离 native advanceGasTrial 入口只读记录实际 rejected dt/time/native，首次真实 StepLimit/rollback 后 4e-6→2e-6→1e-6；Co 继续影响 accepted partition。effort 耗尽的原 public false terminal 完成与普通成功对照均保留，所有原 owner 语义 stdout 去观察行后逐字节一致。S304 在 applySources 更新的 currentDuration 对早期拒绝是 stale/0，原日志保留而不用作拒绝 dt 证明。

**原提交 fixture 的失败矩阵尚非完整负例语义验收**：仅部分 species/budget/audit 与 cell0 粗状态，kind7/15 原状态在 public 重放前未完整保存。新 TEST-R3-NEGATIVE-SEMANTICS 单独列出。S301/S302 仅扩充生成夹具，真实 begin/native kernel/validate/rollback 对全部 16 组合保存两 cell 共32条六成员 ChemistryStatus、全部 cell/face transaction state、完整 Audit/budget/host ledger，失败不泄漏且 rollback 恢复。retry/terminal host 决策标志按原 owner 语义保留。kind10 是原 StepLimit+reaction0/1attempt/1reject/1nonlinear；原成功规则/输入/阈值不改。该支持版通过不提升为提交原版完整矩阵通过，也不声明一般 allocation/free/sync 错误恢复。

## 原 wet、物理库存与 restart

原 EXTRA-wet-film-S209 的全部实际输入哈希匹配，noMelting/phaseCondensed=-1、8gas/8solid/4filmfaces、原 end.001/两窗口/100gas微步：正式 native 完成。8条真实零质量 phase 观察均 target0，仍有非零 heat/force；窗口1每面 phase heat≈−0.02499950001J、force≈22.270882kg·m/s。真实消费 liquid rho=1000；standalone codec 默认 ModelConfig physics 零值不代表消费配置（S303 原字段与纠正说明保留）。D302 全部原 native 输出逐字节不变。

80 位精确转换重算 gas/solid/film 全物理能量（film H−pM/rho+KE）、全部实际 finalized packets/consumer 与逐 film 映射。SolidFilm 用真实 gasMesh face ID，GasFilm 用 surface persistent ID，不能混索引。原 absMass1e-14/relMass1e-10、absEnergy1e-10/relEnergy1e-10 未变：累计 global M 残差 −3.33067e-16 / −2.22045e-16kg，E −3.25931454e-8 / −4.20855050e-8J；全部 global/perrecipient/perfilm M/E/species 范围通过。逐 film E 残差最大约1.36e-13J。旧 dry checker 没有 film 库存，其红和 restart checker-only 缺配置红原样保留，不归为 solver 数值失败。

进一步使用每窗口全部50个实际阶段最终限幅 face flux、真正 ledgerDt 与当前 macro packets 核对完整气体 M/E/vectorP/species；Px 物理残差 −4.10113027e-15 / −4.16827944e-15kg·m/s，原对应阈值1.82417e-14 / 3.71595e-14，均通过。它与原 double cumulative audit 误差是不同量，严格累计红仍保留。

**完整物理 global P 仍未闭合**：Δ(gasP+film mass×actual meanV)+boundaryP+实际 solid-received supportImpulse 的 x 残差 −9.01619622716089e-5 / −1.57820679374845e-4kg·m/s，量级与净 film exchangeP 一致，实际 film mean velocity/stockP 为0。现有 FilmQ 无 momentum 演化、reduced normal trace 为代数约束，IntervalAudit 明确未宣称完整 rigid-solid/reduced-film P。只定位实际缺少完整机械库存/反力通道，不虚构一个平衡项，不改容差、门禁或生产。pressure/stage mismatch 的具体归因仍非唯一证明。OBS-R3-WET-P-PHYSICAL 保持开放；这是既有模型资格限制，非已确认本轮零相变修复回归。

正式原 wet 从实际 checkpoint1 restart 完成.001，但严格 bit parity 失败。exchange-2.csv 首差 row6/packet4/step1/face16/kind1：energy/conductive −0.0024995380314830938 vs −0.0024995380314830904；终态27个语义字段差异全保存。有限终态或很小误差不关闭原位一致性要求。只读 thermal 原100/后50步首差见下，不能把它直接外推为 wet 因果证明。

## 非零相变与扩大预算原输入

两个首拟8cell合法相变控制，旧4b与正式9d都300s未提交完整窗口；只读局部 signed mass/很小 dt 保留为 partial，不把初始文件相同当成功。初次单 cell support 因 solid [-1,0] 与 host-probe recipe [0,1] 坐标不匹配被原入口拒绝，FAIL/input/hash保留。S308 独立副本仅将 solid 实际点全局平移+1恢复已声明单位立方控制，不改原 wet/evap、不改该控制温度/物性/end/阈值。复现用实际 constant/polyMesh，原未消费的生成 dict 与实物坐标差别在输入记录可见。

有效 phaseCondensed0、porosity0、rho1000，host-probe 等价小例 gas800/solid700/film800、thickness.001、liquid e0=200000，Tm600/Tm900：完整1e-6/1gas微步，两材料 corrector candidates 后只有一个最终 phase packet。熔化 +4.002e-6kg、凝固 −2.004e-6kg；target 的实际 hex 与原 −mass/rho/(1−porosity) FP64 逐位相同，真实 recession/solidVolume/film forcing/heat/force/packets/最终库存已记。旧4b安全合法控制↔正式、正式↔D302 四组各10输出文件全部逐字节相同。global M 残差 +3.13083e-14 / −4.74065e-14kg，E −1.22478541e-7 / −1.52056945e-7J，原规则下通过；两个控制实际 global P 三分量残差均0。范围仅这两个合法模型控制，不推广所有 pore/material。

Ns10 原完整 reacting initcheck 的 actual controlDict end **2e-5**，不是计划早期占位0.0007或错误说明.001。原所有输入未改；1800s 内实际473.736s完成，700gas微步/两窗口，最终 ERROR SUMMARY:0 errors，原 checker通过。原300s timeout永久保留，control initcheck不能代替本结果。

原 evaporation 完整end.001正式源900s外部124，已提交checkpoint1/time.00025；没有终点通过。另300s同原物理输入 D302 只读进度观察与已完成原输出全字节相等，window2有17104条phase观察（4276个四面局部候选），dt最小1.1368638508927316e-16s，局部时间只推进约4.8601e-13s。CPU/GPU/IO/日志继续活动，没有观察到CUDA/内存/复制资源错误。属于极小步慢推进，精确 limiter 根因尚未栈证明；不是完全无进展，也不是完成。旧SIGFPE/旧D204300s/本正式900s均分列，不自动增预算或进入长算。下一步需要定向 material/film 步长限制器因果定位，或另行约定有限预算，不能仅盲目延时换绿。

## 其余本轮定向结果与仍缺范围

S306 从不可变权威 SCHEMA3 ASCII t0 mobile 入射复制16行，仅x/y/cell/orig_id/count改变，实际puz=−5、z=5e-7、原d/T/RNG/parcelMass保留。CHT/FSH各10次输出实际16个pStuck3、唯一cell/orig_id、faces384–399；后续 profile/pT 与真实 ring0 age推进，初次接触 age0 正常，未湿 rings age0不等于group没执行。原 double mean+实际CHT carry，FSH真实double schema，最大温差 CHT9.094947017729282e-13K、FSH4.4292391976341605e-10K，均低于原1e-9K。S309实际 global thermal wrapper名称/launch block128与16 active entries，**每粒子8radial lanes，共16完整heat groups**；S306原派生字段16/8=2是记录计算错误，原JSON保留、metadata/cold-group-count-correction.json纠正。所有对应218/238个输出文件原字节相等。S313全名观察只编译，独立源码复核证明S309已足够后取消重复GPU。旧inactivecache strict红仍保留；完整paired gas/particle/wall各stage库存未闭合。

D303对原 thermal8cells/100微步和 checkpoint1后50步，只读 currentCo 前后 Tgas/p、M/E/P/volume、reference/next/last/selecteddt、最终flux/packet。各自与原正式full/restart输出全字节相等。window2首Co前warm Tgas vsfresh0，首Co后cells0/2/4/6温度/压力仍1ULP差（T…2c14 vs…2c13、p…f3c7 vs…f3c6），同一时刻M/E/P/volume一致；全部50 selecteddt/Co记录一致，然后首native最终face energy flux与paired heat/momentum不同。这是实际缓存seed→recovery→flux/packet因果链证据，严格full-vs-restart位差仍开放；不强制清零Tgas、不重写checkpoint或历史时间积分。原补偿时间/clock差异批准范围和R2真实GPUlineage直接复用。

S311直接编译完整正式 Backend.cu/native kernels/storage，无CUDAstub；2组rho/T/U/dt变化、Js.013→.021→.034的6个freshowner参考，实际phi=−area×Js与Co前处理数组逐位一致，K/omega不变。真实 ConstantTransport stage先发布.034，供.055时Co保持既有lagged值，下一wall stage发布.055。此范围通过；未跑ReactingSst、实际copy失败后retry、selected-dt/microstep最终consumer，cold-start-Js仍PARTIAL。不能把referenceDt当已执行selectedDt。

Host ASan/UBSan通过真实当前codec、真实OpenFoamfvMesh/GpuSolidPatchMapper空/single/multiple两向映射与正常析构、真实FSHclient/协议10个POSIX socket上传frames。它们不是GPU mapper/server或一般resource fault验收。旧SST22/CoOOM/copy/retry、GCL/两化学阶段volume、mixed物性/初始化、原cold1D/2D/restart、DO single/mixed与全solid外边界热账、Co六个必要前处理kernels/Nsight+所有非profiling样本，依赖不变直接复用。

14个原 gap ID 全保留逐项追加：SST22-owner-graph、moving-GCL、chem-native-full-retry-owner、mixed-wall-coupler、native-FP64-clock-lineage 的原限定关闭维持。仍缺 gross SST production/dissipation、全部三resident应用final-limitedflux/allsource/M/E/P、真实GPUempty/multiple-selected mapper consumer、未覆盖pore/material→geometry、完整resourceowner/server host sanitizer 与实际free/sync/hostallocation/concurrent失败、Js剩余链；另有上述 strictrestart/globalfilmP/evap和提交fixture完整负例断言缺口。coverage.json逐ID给范围，不重命名、不用支持通过掩盖原失败。

独立 source/证据复核在 metadata/independent-source-review.*、independent-wet-P-review.*、final-evidence-scope-review.*。开发交付旧SST2/3/drygasPx离线重算已独立精确复现46输入hash/42固定旧Gitblob：实际300微步/6宏步物理gasPx在原标准内，按原顺序double累计仍复现5个严格FAIL和dry首窗口PASS；只为算术归因，不算新GPU或完整P闭合。所有原版/支持/诊断/超时/未执行分开，原始失败和支持修正都可追溯。

## 交付与行政核验

所有新原始logs、specs、必要实际输入/逐点数据、patches、源/对象/binary SHA、环境与完整命令统一本目录。已有不可变字节按固定commit/path/SHA引用，不重复上传；新运行raw provenance保留。responses/R014相关追加 reviewer/recheck/history，manifest evidence每SHA一个ID、owned_paths登记，validate_records.py行政校验后正常push同PR。行政验证不代表GPU数值验收；push后再核remote/GitHub PR head/open/unmerged/fixedreportSHA与全部1155生产blob。固定记录commit和远端验证由最后消息给出。**无合并授权，FAST整体不通过。**

支持/观察原失败均保留：S301 ChemistryStatus显式bool编译、D301观察变量作用域编译、S310 nvcc大写Geometry.C命令、初次onecell坐标、首个wet离线packet映射/support符号核算、原dry/restart checker适用范围、build-info双dash/legacy UNKNOWN观察。它们与生产/GPU数值失败分列。正方向首个8cell观察器期间gdb不存在、cuda-gdb只读attach30s无有效stack且可能暂停目标，metadata原尝试/失败保留；该观察器进度不作性能或精确限步根因证明。正式/旧版未观察的300s阻断未受它影响。
