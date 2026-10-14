# PR14 第二轮 FAST 独立只读源码审查

审查者 `/root/source_review`，2026-10-14。按 requesting-code-review/code-reviewer.md 模板独立重新审查；没有将上轮结论直接当作本轮结果，没有派子代理、构建、运行GPU、修改生产源码/index/HEAD或向其它聊天发送消息。唯一写入为本报告。运行结论以下均来自本轮已有记录的只读核对，非本席位执行。

## 固定范围及实际覆盖

- Base：`802a11e7dfe986488a8bf8305e86424dcc94c9e1`。
- 正式生产修复：`4b5c480b572f234ddecf5e89164fe59b0d2e9735`，tree=`0fa2562f4c83394cdf94e9650de3b22c4fb4bb15`。
- 开发者交付head：`75cfd885c028fe80d011a8a081f8bed1f8aeb55e`。
- 仓库根：`/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast`。下文生产file:line为正式生产pin的一基行号；D指该根下 `review/pr14/reviewer-deliveries/20261014T072053Z-fast`。
- 已读开发者 `REPORT.md/SOURCE.json/VALIDATION.json/CLOCK_ANALYSIS.txt/REVIEW_REQUEST.md`、PROTOCOL、本轮TEST_PLAN、formal-production-and-support.diff及results。SOURCE的20项blob内容SHA256逐项与生产pin核对、且与交付head逐字节相同，全部吻合。D正式差分SHA256=`b150590205a0c61e8af9a9e6ec6a4c1480b22679174b3f836fc9132f5dcfebcc`。
- 已读全部新增/修改生产差分：Backend、CourantReductionStorage、CHT owner、SharedGasTrialPolicy、LocalStiffIntegrator、chemistry adapter、1D/2D初始化；另读相关上下游函数与测试夹具。不是全426文件逐行审查。

|范围|本轮直接读过的调用链|覆盖边界|
|---|---|---|
|A|完整LocalStiffIntegrator；chemistry adapter的资源判别、Trial/public调用和volume wrapper；SharedGasTrialPolicy snapshot/status/validate/rollback/target；GpuGasAdvance的两半步、trial与requested-interval owner；ReactorContext、MixtureThermo与reactionInputAvailable关键定义|不宣称全部thermo/kinetics模型、所有刚性尺度、所有native TU逐行正确|
|B|CHT acquire→initialize→publish→patch/configure；1D/2D实际cast/carry初始化；FSH宏与double类型；生产TU cold fixture的物性次序及profile断言|未完整审查所有碰撞/接触/分离调用、非均匀多个thermal warp组和全墙/气物理账|
|C|完整43行缓存、完整58行Buffer/CudaFault；Backend成员生命周期及两种Thrust allocator接口；Co公共准备链、GPU fixture；正常stream同步返回的vendor实现关系|不承诺同Backend多host线程无锁并发安全、所有CUDA错误后恢复或跨device复用|
|D|逐字节比较本轮未改的GasWindowProgram、WindowTime和GpuGasCourant/GpuGasAdvance；Backend时间相关diff；CLOCK_ANALYSIS与新增restart比较，读取HostState上传/下载入口|本席位未独立解析全部clock trace或完整checkpoint状态的首个分歧|
|E|两个schema常量diff，真实FieldSchema重新计算；baseline.json全部键比较并逐条阅读三项差分；chem/cold/Co runner及夹具核心|字符串测试和编译结果不算GPU数值通过；未完整复核所有历史支持补丁|

## Strengths

- 正式native true接已有Trial，false保留public调用；失败类型与实际缺backing区别处理，requested-interval owner和两化学半步没有被换成新的整请求数值拒绝机制。
- CHT初始化使用既有scalar物性并保留pending初始化/发布状态，coupled patch按面覆盖；carry补偿仍受CHT配对宏限定。FSH native TU不定义该宏，原double初始化算式没有进入新增carry分支。
- Co只缓存Thrust临时内存；结果仍是原double+bool/max/OR，7字节成为实际初始化成员。开发者VALIDATION诚实标明compile/link范围与无GPU执行，本轮reviewer记录又分别保留了原夹具红、支持适配和实际运行。

## Issues

### Critical

本轮定向覆盖中未发现新的Critical生产缺陷。不能由此宣称整个PR或全部未测路径通过。

### Important

#### R2-S1 [P2，测试支持] 已提交chemistry owner夹具把host owner与device state放在同一managed对象，异步后host读取在本机失败

**位置**：`tests/fixtures/chemistry_trial/chemistry_trial.cu.in:19–22/27–31`；实际调用至 `common/GpuGasAdvance.cuh:578`附近首个化学launch后的host owner访问。

**触发/证据**：使用提交自带prepare runner构建并运行该真实CUDA夹具。在当前WSL/RTX4060环境，`testMem(s,1)`使用cudaMallocManaged，`s->deviceState=s`，因此host调度对象本身也处于managed区域；首个异步kernel后CPU继续访问它。本轮 `CHEM-OWNER` exit-11，保留的host observer/CUDA-GDB定位在prepareGasTrialTransport首个launch之后的host读取。D `inputs/fixtures/chemistry_trial_host_device_split.cu:29`改用普通host new DeviceState与分离deviceState后，原host访问问题不再是S202/S203的阻断。

**影响**：正式交付宣称的real owner regression入口在支持的实际测试环境无法可靠完成；仅其nvcc/link PASS不能形成运行验收。此问题属于新增测试夹具，不能据此归为生产GpuResidentStrict owner缺陷。

**建议**：正式维护夹具采用真实生产的host/device分离与同步发布，只有所需payload按环境支持方式读取；保留此红及适配diff/hash。不要将S202/S203支持源码悄悄当成提交自带夹具运行成功。不同CUDA环境若支持concurrent managed access可能不复现此host SIGSEGV，不扩大为所有平台必然失败。

#### R2-S2 [P2，测试支持] owner测试硬编码“两次2e-6接受”，与真实CFL分区冲突

**位置**：同夹具 `:121–131`，尤其`:126–129`；参考重放也硬编码两次2e-6。上游实际bound为 `SharedGasTrialPolicy.cuh:327–345`、`GpuGasAdvance.cuh:642–668`。

**触发/证据**：保留同4e-6原请求、V=.001、面面积1及物性，只先修复R2-S1的host/device布局。D `CHEM-OWNER-S202` exit11，原错误为 `owner did not shrink to two accepted native substeps: shared gas post-source timestep bound exceeded`。提交中 `SharedGasTrialFields.H:30`的targetMaxCo默认**.5**，不是TEST_PLAN旧注记中的0。S203真实入口输出targetMaxCo=`0x1p-1`、4e-6 measuredCo=`0x1.166548f19fe31p+0`；其一半仍超过.5。实际成功witness为begins8/commits4/rollbacks4/stepLimits1/terminal0，完整原请求由四个约1e-6接受子步完成。源owner的真实Co准备和限制并未承诺固定两个半步。

**影响**：正确执行原化学缩步再受CFL细分也会被夹具判错；提交的显式reference partition也不是这次owner真正接受的partition。这削弱了可重复的回归入口，但并不证明生产owner缩步错误。

**建议**：正式测试采用D S203思路，按真实生产Co入口准备并保留全部accepted-duration trace，逐段重放同partition，断言requested interval覆盖、一次化学StepLimit失败恢复、terminal对照和原1e-6成功控制。不能提高target/容差、改变V/物性或删掉Co准备以迁就两个子步。TEST_PLAN对默认0的解释应改成与正式pin对应的.5。

### Minor

`tests/fixtures/chemistry_trial/chemistry_trial.cu.in:85–99`的compare对accepted ChemistryAudit仅比较integratedTime和三个计数，对ChemistryStatus仅比较code；未比较speciesMassChange、mass/energy/element residual、maximumLocalError、reaction/thermoCode等完整字段。`cellFields`并不含ChemistryAudit对象（SharedGasTrialPolicy:38–48），所以这些成员也没被前面的double数组memcmp隐式覆盖。D S203 compare:363–377仍有这个范围。已有device预算数组与物理账比较可复用，但“accepted full audits逐字节相同”的称谓需限制，或补对实际所有有语义成员的比较；不要比较未声明padding来制造伪差异。

## 生产源码结论 A–E

### A：F001

`common/operators/advanceGasChemistryKernel.cuh:173–191` native=true选择Trial，false选择原advanceClosedReactor；失败只写试探diagnostic，state/audit/budget在reactorAvailable后才发布。资源分类helper:41–67在失败路径消费相关backing，thermo InvalidModel不自动全部升级InvalidStorage；实际null species/coefficients、不可用模型、basis以及已失败反应的reactants/products/efficiencies被区分。`common/chemistry/LocalStiffIntegrator.H:191–237`保存并传播rejectedDiagnostic到effort/minimum-step返回，解决先前失败反应信息丢失；成功接受后清该临时诊断，数学表达式/步长factor/成功公共路径没有在此diff改变。

`applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh:170–180/284–302`使ChemistryFailure运输标记不否决可重试化学通道；transport与chemistry各自terminal状态组合，不把失败D2H当继续理由。真正Unavailable仍走sticky InvalidStorage，rollback失败仍poison。`common/GpuGasAdvance.cuh:747–797`原requested owner保持：失败缩步、原minimum/effort耗尽转public terminal、整区间完成后commitInterval；两半步:578/:688仍用明确stage volume。本轮没有新增数值/metadata/精度接纳gate。

已只读核对S203日志：正常requested interval完整完成、terminal耗尽与public结果对照、1e-6控制、16类失败资源/metadata组合均有实际PASS行，kind10包括missing reactants+maximumSteps1。它是保留生产源码、修正测试host布局/前置步骤/观察断言后的支持witness，不能替代提交原fixture的可运行性。Ns2两物种witness不能证明所有Ns10反应/两阶段体积/ALE组合；普通Ns10化学native运行和initcheck是另一个有限范围。

### B：F002/F004

`applications/CHT/gpu/GpuResidentStrict.cu:3857–3916/3935–3949/5810/5927–6009`的bit4仅记录初始化未完成，0x800记录指针未发布；新默认填充H2D同步结束才销毁host vector，bad_alloc回实际资源错误并保留bit。patch首先完成初始化/发布，再覆盖所需面；已完成bit不因重复configure/toggle重填，保留既有coupled面。没有按正值/finite重新挑fallback。物性仍1800/710/100，7900/500/15旧token没有因此获得host消费。

1D `common/operators/clearColdWallParticleState.cuh:61–66`和2D `common/wall/GpuColdWall2DDevice.cuh:105–110`在原cast后加其uniform表示损失到已有double carry；原initialiseColdWallProfile构造uniform节点，所以单节点差即mean差。FSH TU不定义GPU_PAIRED_PARTICLE_HEAT，且类型为double；本轮新增分支不修改FSH物理double路径。

真实cold fixture核验两种configure/patch次序、mixed面、重复/toggle及实际下载物性；profile witness四粒子、不同相区和非零carry，但这是四个单线程初始化调用，**不是多个完整8-lane thermal group的动态求解验收**。已读本轮CHT/FSH fixture PASS日志和COLD-ACTUAL-COPY-RETRY实际CUDA错误21/32bytes后重试及mixed面输出。现场CHT原输入/wet100和FSH cold/restart已有有限结果；仍需完整实际paired gas/wall/ledger预算与其它合法资源故障范围，不能由仅表示一致或finite输出推断全收支。

### C：F005/缓存

`applications/CHMT/gpu/Backend.cu:376/380/385/409`保持16字节double+bool布局并初始化显式7字节成员，原max的NaN选择和OR原表达式保留；不存在用数值拒绝掩盖tail。`:33/407/412`为同Backend的Thrust临时allocator，未缓存任何历史Co数值。

`applications/CHMT/gpu/CourantReductionStorage.cuh:24–37`只选闲块；活动块不共享，闲块容量够则复用，否则按需增长，新增Block因vector移动仍保留Buffer实际CUDA指针。类禁止copy/move，CudaFault早于blocks成员声明，逆序销毁期间fault还活着；Buffer RAII释放实际CUDA分配。真实同步归约返回host标量后临时array归还，原stream和Thrust的同步返回保持。异常由Backend catch报告真实操作错误，不能标成数值退步。未发现正常序列使用的确定性生命周期/容量bug。

`common/GpuGasCourant.cuh`逐字节未改，六个Co前处理没有被删除。Buffer本身也未在本轮修改；其cudaFree失败仍清pointer/size的旧语义不能由一般OOM重试成功推断全部free故障可恢复。需实际错误类型/效果审计；不未经证据给此既有行为贴新增缓存回归。

本轮CO-FIXTURE日志包含真实OOM与reuse/equivalence PASS，Co-FRESH-2/10及10-initcheck记录PASS；这些是新增缓存的实际有限证据。原thermal100步initcheck日志summary0，两份formal-vs802全部输出比较artifact均all_files_byte_equal=true。缓存容量与保留显存是效率取舍，需要同时看retained/peak bytes；不从源码少malloc或化学kernel计时推断整solver提速。缓存没有mutex，所谓multiple live requests是序列化host中的不同活动分配，不能推广同Backend并发host访问线程安全；Backend整体也没有该线程安全契约。

### D：时间与新增restart red

WindowTime、GasWindowProgram、GpuGasAdvance/GpuGasCourant均与本轮base逐字节相同，Backend生产差分仅缓存/尾部相关，不含时间数学修改。用户确认保留补偿时间；CLOCK_ANALYSIS标量预测和旧CSV只能支持解释候选，不能冒充本席位已解析的实际首分歧。

新增D `artifacts/chmt/formal-restart-terminal-comparison.json`确认为all_files_byte_equal=false：accepted-2/material-2相同，gas-2/exchange-2/checkpoint-2不同。主代理正在对旧802相同断点真正复验；本席位不认定新缓存回归，也不把exit0或普通checker容差作为bitexact关闭证据。

已追恢复入口 `applications/CHMT/CHMT.C:29–39` readCheckpoint→createBackend→storage.configure；`applications/CHMT/gpu/SharedGasDeviceStorage.cuh:605–620` HostState conserved与device density的除/乘V，`Backend:332–339` restoreGeometry/uploadState以及新owner初始化需要与连续owner对照。`:605–613` uploadState只恢复守恒字段，primitive/cache通过后续recover重新产生；`common/operators/computeSstFaceFluxKernel.cuh:1563–1569`以现有Tgas作热反演guess。重新初始化与已有owner的warm primitive历史是可查候选，相关源码本轮未改。优先对旧802同断点和formal在首次currentCo前/后记录conserved、Tgas guess、stage/nextDt、Co、wall packet的首个hex差；检查真实checkpoint全状态而不只CSV。上述是定位建议，尚未证明差异原因或违反批准语义。

### E：支持修复

两项checker差分只把FieldSchema常量换为11438850135072464462。按正式Checkpoint.C:13的真实FNV-1a seed、:594完整5444字节FieldSchema重新计算结果正是该值；没有schema wire或物理阈值扩改。

baseline.json键集不变，只有gasUGKP/FSH/CHT三项riemannFacePrimitiveForGradient变化，各只在slip速度投影后补入原baseline已有的TFix/T/rho分支；其余函数文本逐字节相同。没有清TFix、放宽memcmp或以当前整体输出重生成oracle。此支持维护合乎已有原生产逻辑；它不证明cold旧differential全部热量/carry消费者适配当前模型。

## 依赖及证据复用判断

|证据|本轮状态/理由|
|---|---|
|上轮immutable原chem StepLimit/public/ordinary|可作原红与历史控制，不能当formal native Trial/owner新绿；adapter/诊断/policy改变，须本轮witness|
|旧P007/P010 cold诊断|仅解释原producer/表示问题，不能叠加到正式pin或重算正式通过；formal CHT/FSH需本轮实际字段/ledger|
|旧Co tail诊断及原100-step输出|原红与802字节基准可用；formal tail/allocator改变必须本轮initcheck和whole publication comparison|
|clock数学/旧CSV|时间依赖未改，可复用已固定输入的预测与对照；真实trace及restart首差尚须独立解释|
|旧schema/frozen face失败|作为历史失效支持证据保留；本轮支持修复使原常量/oracle不能再作为当前预期|
|其余压力/drag/pool/其它旧绿|仅在实际依赖未受这些共享修改影响时按原scope引用；不把本轮源码审查推成全33topic或426文件通过|

此表是本轮重读差分得到的依赖判断，没有重复上传前轮不可变证据，也没有把历史通过累计成新的通过数量。

## Recommendations / 本席位未关闭的测试缺口

1. 正式维护R2-S1/S2的chemistry regression入口；补accepted所有有语义audit/status成员、两半步与multi-cell失败transaction的完整比较，保持原成功/public字段规则。S203现有物理/预算证据保留。
2. cold：多个完整thermal group、非均匀profile及真实stage权重下paired gas/profile/carry/wall/最终账；不能用四线程初始化fixture覆盖warp动态求解。FSH实际double/restart已读记录按其具体范围保留。
3. cache：跨window/restart/reconfigure的数据更新、同owner序列化多个live分配及多个owner/stream、实际copy/sync/free/host allocation故障和销毁保留字节；ordinary Co scalar等价/一次OOM重试不关闭全部生命周期失败。
4. restart：保持当前非bitexact red，等待旧802同断点与formal首个输入/primitive/Co/packet差，不能先归因缓存或通过改变clock/阈值掩盖。
5. D actual trace、F中的active SST/RK完整预算、ALE/GCL/两化学阶段体积、wet/film/ablation/radiation/mapper及host sanitizer，按主代理实际进一步验收；本席位没有逐项复核全部本轮运行的数值闭合。编译、source字符串检查、setup、环境健康观察和profile计数不得合计为GPU数值pass。

## Declined to judge

- 整个426文件逐行正确性、所有33topic闭合：本轮仅A–E及相关调用链，未直接读全树。
- 亲自GPU执行、编译、profiling或重新运行checker：分工禁止；只读已有记录。
- 未知/不安全supplied count/index、负allocation size、故意越界：用户已给边界，不能新增metadata数值拒绝，也不做危险试验。
- NaN/Inf及不收敛的物理准确性：允许真实terminal算术；本轮只核调用/资源/字段/消费者，不新增finite/precision/收敛gate。
- 有效历史Co效率近似与跨架构/compiler的所有位级复现：不是此次改变的已授权通用证明。
- 同Backend并发host调度、跨device迁移、所有cudaFree错误可恢复：缺契约/错误效果证据；不以本缓存无锁为新的普遍并发bug。
- restart red是否是新回归、全部clock trace首差：旧802真正同断点对照与完整解码仍在进行。
- 物性7900/500/15旧token成为有效配置：明确不在本次修复，默认仍原scalar。
- README、正式修复、GitHub评论/PR文档、commit/push/merge：本席位没有这些授权；只写此审查文件。

## Assessment

**Ready to merge? No.**

在已直接审查的正式生产修复调用链中，未发现新的确定性生产数值/资源接线缺陷；但已提交的chemistry测试支持有两个可复现问题，完整audit比较与若干已约定GPU消费者/生命周期/账目范围未被本席位关闭，且最新thermal restart bitexact red尚未解释。应按正式源码及对应支持版本完成这些定向闭合，再由主代理给出整体结论；本报告不授权合并。


## 后续客观验证：同一 checkpoint 的 old802 / formal restart

本节追加于 2026-10-14，保留前文 full-vs-restart 的原红与当时证据边界。

已只读核对 D/results.json 中 THERMAL-802-SAME-RESTART：真实 GPU、old-production pin 802a11e7dfe986488a8bf8305e86424dcc94c9e1、exit 0、无 external timeout；binary SHA256 为 7124cf69b1cd51da033c1360d67129748803a5c66767489f2864c169b85c845d。原日志 logs/THERMAL-802-SAME-RESTART.log 显示从正式 checkpoint-1 恢复到 time=0.001，50 gas microsteps、1 material solve；日志 SHA256 43012a8076f77f64f004c4dd45beac11a27dfaa4bcbe265d976e7bdbc3b48217。

artifacts/chmt/old802-vs-formal-restart.json 的 all_files_byte_equal=true，覆盖 accepted-1/2、exchange-1/2、gas-1/2、material-1/2、checkpoint-1/2/state.chmt 共 10 个文件。比较记录 SHA256 fafa3728a5df0eaab282f4ee04ca7957e81827f6272ebf820e02b25bb1c03aac。本席位另外直接重算 work/cases/chmt/thermal-old802-same-resume/chmtOutput 与 D/artifacts/chmt/thermal-formal-resume 的这 10 对文件 SHA256，全部与记录相符并相同。

此结果说明：在此固定 Ns2 thermal 输入、同一正式 checkpoint 与当前平台下，本轮缓存/尾部改动没有造成 old802 与 formal 的恢复输出差异。它收窄了前文“尚待旧802同断点比较”的缺口，但不解释 full-vs-restart 首差，不证明连续运行与恢复运行应逐位相同。artifacts/chmt/formal-restart-terminal-comparison.json 的 gas-2/exchange-2/checkpoint-2 非 byteexact 原红仍有效；accepted-2/material-2 相同仍有效。不把这一原红认作新缓存回归，不通过改变 clock、阈值或 oracle 关闭它。

本次追加没有执行 GPU、编译或改动生产文件；整体 merge assessment 保持未授权。另一个新增 wet 合法输入的既有缺陷与因果边界单列在 metadata/independent-wet-review.md。
