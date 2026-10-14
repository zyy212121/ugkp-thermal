# PR #14 第二轮独立 wet 源审查追加

## Scope / 固定对象

- 仓库：/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast。
- D：上述仓库/review/pr14/reviewer-deliveries/20261014T072053Z-fast。
- 正式生产：4b5c480b572f234ddecf5e89164fe59b0d2e9735，tree 0fa2562f4c83394cdf94e9650de3b22c4fb4bb15。
- 对照生产：802a11e7dfe986488a8bf8305e86424dcc94c9e1。本追加重新读取本题调用链，未把上一轮结论当本题新证据。
- checkout HEAD：75cfd885c028fe80d011a8a081f8bed1f8aeb55e。读取结束时 git diff --name-only -- applications common 为空。
- 本席位只读源、配置、原始运行日志及比较记录，没有 GPU 执行、编译、危险输入复现、生产修复或子代理。仅写本文件，并在已有 independent-source-review.md 追加同 checkpoint restart 的客观后续。
- 采用 requesting-code-review 的独立 code-reviewer 模板分类；不声称全 426 文件、全 33 topic 或全 wet 物理模型逐行通过。

## Strengths / 已核实边界

S209 对两个原 case 的已有归档输入文件逐个比对，全部 byteequal；新增内容为 geometry-support.json 与已平移 gasmesh-points.dat。27 个 gas 点实际 x 范围为 [1e-5, 1.00001]，y/z 为 [0,1]，面对 solid 的原 x=0 面形成真实 1e-5 m film 空间。gasmesh-points.dat SHA256 851bca5c2ea7d817445920645dabab9c6319348f42d43607895cddf7e6c0dd21，与 geometry-support.json 记录一致。原 blockMesh 字典保留，不能只读其共面坐标便认定实际 S209 mesh 共面。

两种输入均明确使用 enableFilm=true、ThicknessAveraged、350 K UniformFilmTemperature、rho=1000、cp0=1000、cp1=0、e0=100000、conductivity=1、Tmin=100/Tmax=3000、liquidViscosity=.01、thickness=1e-5、surface Y=(1,0)、pressure/liquidReferencePressure=171600。wet-film 未提供 material 子字典；wet-evaporation 仅增加 HertzKnudsen 参数和 CoupledRecession：accommodation=(1e-4,0)、saturationPressureScale=(200000,0)、saturationTemperature=(0,0)。均未启用 melting。物性、endTime=.001 和阈值没有为 S209 修改。

Configuration.H:20–22 的合法默认是 phaseCondensed=-1、enableMelting=false、phaseFilmY 全零；ModelIO.H:77–80 确实读 liquid 表与 viscosity/reference pressure，:88–105 在提供 material 时仍以 -1 为省略 phase 的默认。Reaction.H:210–232 只在 enableMelting 或 ResolvedNormal 时要求显式相变组分和 phaseFilmY；:246–260 的 evaporation 输入检查不要求 melting/phaseCondensed。故此 ThicknessAveraged wet 输入不能仅因没有 melting 的 phase 映射被归为缺输入。InitialConditions.H:57–62 使用正 rho/thickness/area 与显式液体表生成 film 并 recover；本次运行已产出 checkpoint-0。

原 checkpoint-0 的 accepted-0.csv 明确为 time=0、8 gas cells、8 material cells、4 interface faces。gas-0/material-0 的体积与质量有限且正，gas 600 K、solid 300 K、solid porosity=0；exchange-0.csv 无包。初始 film 的全部序列化数值本席位未另行解码，不能由这些 CSV 扩大为全部 film 字段已核验。

## Issues

### Important — R2-W1 [P1，既有缺陷，仍 OPEN]

**零相变质量的 wet 传热包无条件消费默认 -1 相变索引。**

位置：applications/CHMT/materials/CpuMaterialDriver.C:460–461，尤其 :461：

```cpp
target[phase.filmFace]-=phase.mass/p.condensed[p.material.phaseCondensed].rho/(1-material.porosity);
```

触发条件：合法已存在的 positive film + solid、ThicknessAveraged、melting disabled、phaseCondensed 默认 -1；正常 material 子步产生 wet solid/film 的内部传热包，即使该包相变质量为零也到达几何 target 更新。这个条件无需伪造指针、count、非法 mesh 或故意提供越界索引。

已直接读过的调用链：

1. CpuMaterialDriver.C:151–175 的 Implementation::evaluate 调 evaluateCpuSurface，必要时调用既有 completeCpuSurface；:372–383 的 material 子步先恢复 film transport/profile，再评价表面。
2. CpuSurfaceInterface.C:35–51 用实际 face/cell map、film、solid、gas field 构造 SurfacePhysicsInput；未启用 useAcceptedFlux（其默认 false，InterfaceMath.H:268）。
3. InterfaceMath.H:331/342–368：wet 由正 film mass 判定；phaseRate 从 0 开始，仅 enableMelting 分支才计算相变速率。melting disabled 时计算底部传热温度，保持 phaseRate=0。:383/:479 将其传入结果并缩放，保持零。
4. InterfaceMath.H:586–600：phaseAdvective 的固体 phase 表只在 result.phaseMass!=0 时读取；zero phaseMass 仍有真实 solidHeat/phaseConductive/phaseEnergy 和机械作用。这不是无效包。
5. CpuSurfaceInterface.C:57–70 只要 hasSolid && physics.wet 就生成 SolidFilm phase packet，mass=area*dt*phaseMass=0，并记录 conductive/energy/momentum；:62 用安全的 c==phaseCondensed 循环写数组，不读取 -1。
6. CpuMaterialDriver.C:421–432 保留并消费这些 phase 包；CpuFilmDriver.C:320–325 在非 freezing/nonempty 情况直接保留包，不会强制补出 phase mapping。
7. CpuMaterialDriver.C:445–461 计算几何 target。GasSolid 包的组分密度更新有 packet.condensed[c]!=0 保护 (:453–456)；phase 包的 :461 没有相同的 zero-mass 保护，仍读取 p.condensed[-1].rho。

影响：这是确定的 C++ 数组越界读取/未定义行为；零分子不免除分母的求值。可能导致 host 浮点异常、非有限几何或取到无关配置值。严重度 P1 来自合法常见 wet 配置可中断 native 求解或污染其 geometry，且无需相变。

**本次 SIGFPE 的因果边界：**正式原版 Ns2 两例已经实际退出 -8/无 external timeout；logs/EXTRA-wet-film-S209.log 和 logs/EXTRA-wet-evaporation-S209.log 显示 FOAM_SIGFPE 并有未解析 executable stack frames。原 binary SHA256 ea78e2ac08b52915dd6da4f741a791f3251df3a530895455887968586b715599，source pin 4b5c480…，日志 SHA256 分别 8580b0be16ff5104b615b4a994606448d33c8167edea0bf9abefb77c43e6162c / 65ad154659b96ff6e2ad933f5deaf464400440d2b8289484b6bc1454c4067e38。上述源链与 red 条件吻合，但现有日志没有精确 pc/解析栈，没有读到实际 p.condensed[-1] 所对应内存位。本席位不把“越界发生时可能 0/0”写成已观测具体指令或已确认全部 SIGFPE 原因。

**baseline 判断：**CpuMaterialDriver.C 整文件在 802 与 4b 逐字节相同，SHA256 170ae1974fe71d14b093d3f39d93276cfb083cfe430149bbb9c1074a2757b79c；如下相关 producer/config/film 源也相同。这是 PR 原 802 中已有、此次发现的生产缺陷，不能归因为本轮 Co allocator、initialized tail、native chemistry Trial 或 cold carry 改动。本席位未再次运行旧802 wet 来触发已知危险读取。

## Recommendations / 最小修复方向与证据要求

只在确实 nonzero phase.mass 时消费相变密度并更新相变几何；zero phase.mass 不读取未使用的 phase 表。保留 zero-mass SolidFilm 包的真实传热、力、film forcing、双方 ledger/consumer mask。不要通过开启 melting、选择任意 phaseCondensed、修改物性/阈值/时长、丢弃整个 phase 包或添加 finite/metadata 接纳门禁绕开它。真实非零相变时保留原 FP64 除法表达式及原数学次序。

已只读审阅 D/patches/D204-zero-phase-does-not-consume-disabled-material.patch，SHA256 b29e7b8558f13318842152195bccb96e6230427221a25b3dc41d84a0574838fc。实际 diagnostics 源与正式源的唯一差分也是 :461：

```cpp
target[phase.filmFace]-=(phase.mass==0 ? Real(0) : phase.mass/p.condensed[p.material.phaseCondensed].rho)/(1-material.porosity);
```

它在 phase.mass==0 时避免该非法 phase density 读取，未删除 heat packet、未增加收敛或 finite gate；非零相变的原双精度计算保留。保留的 donor EOS recovery 与 1-material.porosity 仍依赖原模型，不据此推广所有 resource/zero-porosity 组合通过。

只读 BUILD-WET-D204-HOST.spec.json 的 argv 确认隔离 host diagnostic 文件参与链接，原 work/build/CHMT2/Backend.o 被复用；本席位没有进行该 build。D204 两个 case 的归档 0/system/constant 输入逐一与原 S209 比对，没有修改。该诊断不写入生产，也不是开发者正式修复。

报告写入前所见 EXTRA-wet-film-S209-D204 已完成：exit 0，diagnostic binary SHA256 0557584e0899f09b104232bfdc530559a0697d96b21b7be3d105e7b9fe1cc641；原日志显示两个窗口各 50 microsteps，最终 time=.001。results.json verdict=INCONCLUSIVE，恰当地区分了执行完成和数值验收。wet-evaporation-S209-D204 当时尚无最终 results 行，只读到运行中的 material solves；不计为 pass。D204 对照支持定位方向，但未提供已完成完整 stock/ledger 审计，不关闭 R2-W1，也不把 unknown stack 变为已解析。

后续可使用已经保留的原 core/日志进行符号解析，或在安全 D204 overlay 中观察 phaseMass=0、enableMelting=false、phaseCondensed=-1、phase heat 与 target 的实际字段。不要重跑已知潜在 OOB 的原分支。正式修复需开发者在固定新 pin 上提供实际 GPU native full/restart、wet heat-only 与 evaporation 的所有 stock/能量/质量/geometry/ledger 消费检查；用真实非零相变控制确认成功表达式和消费者没有改变。编译、exit0、finite 或字符串检查各自均不等于这些数值闭合。

## 依赖证据及非缺陷边界

|本次重读依赖|4b SHA256|802 对照|
|---|---|---|
|CpuMaterialDriver.C|170ae1974fe71d14b093d3f39d93276cfb083cfe430149bbb9c1074a2757b79c|byteequal|
|CpuSurfaceInterface.C|3828e279133094b8a218816fd9dcedf9467e428e46ca47e76c36fe02d5097434|byteequal|
|InterfaceMath.H|423b845f11f1af38d5271b01c2bc442df37774ff0b11dcf0647805e725d72411|byteequal|
|CpuFilmDriver.C|d52c38751124be82e73b3a458bf512dc08a64d2108981d0033787e97ff677e67|byteequal|
|FilmMath.H|daf48f096c6b3262eb5d4ee0f50e56a7e02ad6c45c93d4e198605c310ea73b8f|byteequal|
|InitialConditions.H|5393ce7430d34017a261d78b32b9b359274d5a6f32d3ee23cc4576694b877ea8|byteequal|
|ModelIO.H|6de4c3e3b66ecbad85366f5c96d7c2810167dac388973c0019d2d75f8ef4c016|byteequal|
|Configuration.H|9d9e67cd75061a45ea93dc9c685a6408fcbf1a097ae9a912644eae4147547236|byteequal|
|Reaction.H|dae80474686744541f0aa5a5456e48bf89e52cbd03a3b7da6e1220705c64d274|byteequal|

wet-film-S209 chmtProperties SHA256 8b4576bfba4851bd998c117fcb55d868febcacd7bc478a9fc58ee5a1cebf923d；wet-evaporation-S209 SHA256 eb6bf18f73ad6a4bdae7ab7b32f6f98b6dbda34d3b3c1e93095ed20f3b2da4e6。

- 原共面输入与 finite film 空间的模型冲突已由 S209 几何支持修正并保留原红；不把那一原配置失败混记为本缺陷。
- melting disabled 的 phaseCondensed=-1 本身合法。未使用的 phase 映射不是需要补齐的 backing；真正问题是几何 consumer 读了不应消费的表。
- InterfaceMath 中已读到的 latent/advective/recession phase 表引用受 melting 或 nonzero phaseMass 保护；没有重复报这些 guard 已覆盖的行为。
- zero phase.mass 仍需要 solid-film heat transfer/力包；生产者保留它不是本缺陷。
- CpuFilmDriver phase finalizer 在真正 phase dryout 条件才需要映射；本报告没有证实它导致当前首步 SIGFPE。phaseFilmY 的零默认也不能单独判定这些 zero-mass 包错误。
- 输入的 verification.json 从 thermal 基例继承，不能作为新增 film 模型未来解/预算 oracle；仅复用其原网格、gas/solid 初值范围。
- 原 -8 确认为执行失败，不把 NaN/Inf、不收敛或数值状态改称 MissingStorage。

## Declined to judge / 尚未关闭

- 未解析旧失败 core、原 pc、具体除法 trap 位：原日志不含该证据，已知潜在 OOB 后不重新危险复现。
- wet 全模型合法性的通用证明、所有 film/evaporation 工况/其它边界组合：本次只核指定输入与相关消费链。
- 原/诊断全部 film checkpoint 解码、D204 完整质量能量收支、evaporation 最终运行、非零 melting 控制及 wet restart：由主代理实际验证；本席位未冒充亲自执行或已完成。
- formal actualcoupler/radiation 已 exit0 的物理准确性：主代理负责进一步实际字段/账目审计，本追加没有复核全部消费者。
- full-vs-restart bitexact 原红的首差原因：另文件已追加同 checkpoint old802/formal 全字节相同的客观证据；不能据此认定全部 continuous-vs-restart 语义关闭。
- 修改生产、README、PR 文档、评论、提交、推送、合并或派子代理：无该授权，本席位未执行。

## Assessment

**R2-W1：OPEN，P1 既有生产缺陷。**确定的消费越界与合法输入路径已由固定源码证明；本次正式原红与最小诊断方向相符，原信号精确栈因果仍受上述证据限制。此项不是 802→4b 新改动回归，诊断 overlay 的 exit0 也不等于正式修复/数值验收通过。建议交开发者做最小消费修复并完成同输入实际收支验证；本报告不授权合并。
