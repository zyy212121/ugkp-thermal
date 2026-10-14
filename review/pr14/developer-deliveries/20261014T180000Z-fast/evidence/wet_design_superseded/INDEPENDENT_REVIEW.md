# FAST4 湿膜压力积分设计独立审查

2026-10-14，只读审查。结论：已捕获的静止湿膜失败有可定位的压力时间求积不一致，不需要新增膜动量状态。建议把首轮实现缩小为一个每面 EOS 压力时间积分，保留原有底剪切、压力梯度、重力、几何取样和能量闭合。原 DESIGN 的完整气侧应力直传公式不能作为一般情况下的纯求积替换。

本审查不批准接口扩展，不修改源码、测试或阈值，不把原有 FAIL 改为 PASS。下列实现建议仍需用户批准。这里的“应闭合”是所述离散模型内的物理关系，不另立运行拒绝门槛。

## 审查范围与固定版本

- 生产源码：9d5be6eff69dbced2fc959d422d3ba419a1bc39e。
- 保留证据：0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668，review/pr14/reviewer-deliveries/20261014T120103Z-fast。
- 被审 DESIGN 的 SHA256：bff1689e5fa3c599606316c0f843b4e4a308697c21e61d97d90b35cdc0d6c2fa。原文快照为 REVIEWED_DESIGN.md；作者后续修订不在本次快照结论内。
- 12 个固定源码的 Git blob、SHA256、链接和证据哈希见 metrics.json。审查中共享工作树有其他任务的修改，本文行号与判断以固定生产提交为准。
- 独立重算了 D302 原生观察日志；未启动新的 GPU 求解，也未把其他人的分析结论当作新求解结果。

## 一 真实失败的解释成立

每窗 50 个真实气侧 stage、四个 film face，每面 0.25 m²；几何静止，质量通量、黏性应力、基底速度、膜均速与 G 均为零，所有保留的气侧 scale 均为 1。首窗材料底部冲量等于首个气压乘整个材料时间窗，而气侧冲量使用 50 个实际压力。

独立 Decimal 重算得到：

- 第一窗底部减气侧冲量：9.0161962254198257938e-5 kg m/s。
- 第二窗：6.7658717046281680041e-5 kg m/s。
- 累计：1.5782067930047993798e-4 kg m/s。
- 每面底部冲量与“首 stage 压力 × 面积 × 整材料步”之差仅约 +1.91e-15、−1.20e-15 kg m/s。

这支持原 DESIGN 对该 witness 的归因。无需用全局末态差额、film.exchangeMomentum 或审计残差补入 supportImpulse。反力应由局部面力的正确时间积分形成，再进入既有 SolidFilm 账本。

## 二 法向和当前公式

定义 n_f 为 SurfaceMesh.normal；n_g 为 CPU 公式使用的气面内法向，即 −S_g/|S_g|；n_s 为固体面外法向 S_s/|S_s|。保留 witness 中三者均为 (+1,0,0)，不是气面原始 areaVector 的方向。设 P_a = I − n_a n_aᵀ 为切向投影，A_f 为参考膜面积。

CPU 路径明确给出：

    shear = (A_g/A_f) P_f P_g traction
    bottomShear = P_s (shear − G δ)
    J_b,old = A_s n_s p_EOS Δt − A_f Δt bottomShear

所以，同一输入下的等价展开是：

    J_b,old = Δt [A_s n_s p_EOS − A_g P_s P_f P_g traction
                         + A_f P_s(G δ)]

投影层序不能任意交换。一般不共面时 P_s P_f P_g ≠ P_g P_f P_s。

依据：CpuSurfaceInterface.C:37–46、66–69；InterfaceMath.H:563–566；FilmMath.H:24–28。原始物理包正号为 GasFilm 进入 gas、SolidFilm 进入 film；accountCpuPacket 对 SolidFilm 给 supportImpulse 加 −J_b。因此静止平面 witness 要求 J_b = J_g,stress，符号正确。

还有一处不能被符号简写掩盖：Backend.cu:258 的 LowRe 传给 GasWallInput.normal 的是 n_f；BoundaryLayer 在 :265 改成真实气面内法向。上述 CPU 的 P_g 则来自实际网格。三法向相同的 witness 不受影响；不能据此假定任意非共面几何中的两条 closure 路径完全相同。

## 三 原提案需要收窄的地方

### 完整气侧应力直传会改变法向闭合

GasWallMath.H:91–113 的实际气侧应力冲量使用 p_mechanical n − traction。CPU 底部 :69 使用 p_EOS n_s，只保留底剪切的切向分量，没有传递法向黏性应力。

因此把 J_b 统一设成 J_g,stress + GδAΔt，除了时间求积，还会：

1. 在 BoundaryLayer 中把底部 EOS 压力换成 mechanical pressure。
2. 把气侧法向黏性应力传到底部；当前代码并没有该项。

即使平面共法向也成立。相同切向驱动、无 limiter 缩放且统一求积时，当前闭合允许保留的法向差为：

    J_b − J_g,stress − ∫ A Gδ dt
      = ∫ A [(p_EOS − p_mechanical) + traction·n] n dt

这个项不应为了全动量零残差而强行删除。原 DESIGN 的“使用同一应力迹”须标为额外闭合假设，而非现有代码事实。它提出的 EOSPressureTime 对保持旧法向力而言并不只是可选的热量字段。

### 剪切取样也不是只换求积

CpuSurfaceInterface 调 solveSurfaceInterface，用材料本次算出的 topTemperature 等重新构造 physics.traction。GasIntervalRecord 的 traction 则来自实际 gas stage，使用当时 WallProgram 的边界温度和运动。把 CPU bottomShear 换成实际气史剪切是可讨论的一致化，但可能同时改变驱动取样、膜 profile 和功率关系，不能保证原剪切结果与 FP64 轨迹不变。

### 代表 stage 不是完整 history

Backend.cu:450–453 在微步完成后记录最后执行的代表 stage。Euler 下就是该唯一 stage；RK 下不能把它说成每微步第一个 stage。材料驱动在 CpuMaterialDriver.C:292–304 取当前覆盖 record，并可能跨越许多微步。完整积分必须在 accountCoupledFaces 的每次真实 stage 回调中累积。

## 四 建议的最小一致修复

### 一个标量 保持当前 EOS 压力闭合

给每个 GasIntervalRecord 的每个 film face 增加暂态 host 标量：

    eosPressureTime = Σ_s ledgerDt_s × p_EOS,s       [Pa s]

p_EOS,s 可直接取当次 b.faceRates[i].trace.pressure。LowRe 与 BoundaryLayer 均已存在这个值，后者对应 matching/EOS pressure，不用 mechanicalPressure 替代。无需额外 gas snapshot、CUDA kernel、全域遍历或新的 surface constitutive solve。

材料子步消费到的 I_EOS 进入原有底部包生成公式：

    J_b,new = A_s n_s I_EOS − A_f Δt bottomShear_old

A_s、n_s、bottomShear 使用与原 CPU 面求积相同的状态、同样的公式。可在已有面包构造 helper 接收该步压力积分，而让预测/端点建议调用继续采用原 p_EOS Δt。材料真正消费的 phase packet 必须在现有 finalize、filmForcing 与 accountCpuPacket 前得到最终 momentum；不能到 support 预算处补一个全局差额。

此设计没有 mass==0 分支。蒸发、凝结或相变时也仍计算同一个压力负载；质量、质量携带动量和当前相变能量不被拿来代替压力。它修的是已经证实的 normal pressure 求积，不能声称同时修复所有剪切或相变动量问题。

### limiter scale 的归属必须明确

原实际 gas packet 使用 α_s × ledgerDt_s，而旧 CPU 底部压力力没有乘气侧 α_s。严格的“保留旧闭合，仅改善压力时间取样”应按上式累积 ledgerDt_s × p_EOS,s，不把 limiter 当作热力学压力的缩放因子。

若改成 Σ α_s ledgerDt_s p_EOS,s，则是额外的离散反力规则：α≠1 时底部压力强度也随气侧限幅。它可能有数值一致性动机，但不再等价于原底部公式。需要单列批准与非单位 α 的验证，不能借“accepted history”名称隐含引入。当前真实 witness 的 α 全为 1，现有证据不能区分这两个方案。

对保留旧规则的设计，气侧最终受力与底部 EOS 力在 α≠1 时的差异要照实保留并解释。它不是要求新增求解拒绝条件，也不构成把未闭合总账自动判绿的理由。

### 子步切片和 RK 权重

- 共用 common/GpuGasAdvance.cuh:132–149 的真实 stage 回调。先得到本 stage 最终 flux，再由 accountCoupledFaces 累积；EOS 积分的权重采用 ledgerDt，不重复乘 stagePacketDt。
- Euler 权重为 1；SSPRK2 为 1/2、1/2；SSPRK3 为 1/6、1/6、2/3，见 GpuGasHostPolicy.cuh:38–45、GpuGasAdvance.cuh:328–368。
- IntervalCursor::takeThroughLocal:152–176 的同一遍历对标量做 fraction × 原积分；record 最后一个片段用 原积分 − used 积分。保留候选 cursor，材料 retry 不推进正式消费位置，接受时 :619 才提交。
- 这种切片沿用当前 record 内均匀分摊的数值约定，不是恢复被省略的连续 stage 时间函数。整 record 总量守恒；部分 record 内的高阶时间重建另属设计范围。
- 不强制材料时钟跟随所有气微步；四面、50 records 仅 200 个 doubles，即 1,600 bytes 原始积分数据，另加现有容器/切片开销。没有性能实测，不能据此报运行提速。

### 几何范围

标量 I_EOS 乘材料求积点的 A_s n_s，保留原 CPU 的几何采样。在已证实静态几何下，这是完整的该压力力时间积分修复。移动/转动底面一般有 ∫ p A_s n_s dt ≠ (A_s n_s)_CPU ∫p dt；若要求前者，需要记录向量积分并明确采用 gas 执行轨迹还是最终 CPU 轨迹，会扩大求积变更。现有路径可继续运行，但不能把标量方案宣称为任意移动几何的全 stage 力积分。

## 五 phase front speed 对能量的含义

InterfaceMath.H:423–425 的 solidSpeed 来自消耗体积与 packing，solidWallSpeed 可来自实际几何扫掠。这些是相界面速度，不是固体骨架材料速度。:589–600 同时保留：

    phaseAdvective = j [e_s + p_EOS/ρ_s]
    phasePressureWork = (A_s/A_f) p_EOS w_front
    phaseViscousWork = 0   // stationary rigid skeleton
    u_liquid,bottom·n_s = w_front + j/(solidRatio ρ_liquid)

在非多孔、固定骨架、相变是唯一固体消耗的情形，w_front = −j/(solidRatio ρ_s)，所以压力携带项与前沿压力功恰好相消。不要给 support 再加 J_b·w_front：那会把相界面迁移错误地当成骨架运动，并破坏已有能量分解。外部固定支撑的真实速度为零，支撑冲量可以非零而功为零。

CpuMaterialDriver.C:117–121 把 SolidFilm 的 bottomWorkIncluded 设为 true，CpuFilmDriver.C:297 因此不会另算其通用 bottom supportPower。若未来引入真实运动基底或剪切功，不能只追加一个 supportWork 数字；须同时定义骨架速度、液体 trace、界面速度，以及材料/膜实际能量消费归属。

当前静止、零质量 witness 中，压力力修复可以只改变物理反力与对应账本，原热量、相变、pV 和气侧能量算序均可保留。非零相变或运动情况下，这不等于已验证全部界面功一致。若将 EOS 时间积分用于能量，必须对压力携带项和 front pressure work 成对定义相同离散表示；当前修复不需要这样扩大范围。

## 六 哪些项应闭合 哪些不能以全惯性模型验收

当前 witness 应闭合：实际 gas 压力冲量、局部 SolidFilm 反力及固定 support 的反冲量；由于速度和质量通量全零，没有其他物理项可承接已观察的 1.58e-4 差额。

当前一般薄膜模型仅有代数 Stokes/lubrication profile，FilmQ（Types.H:88）只有质量、焓、组分，没有演化的动量。以下不能由一次压力求积修复宣称成立：

- 任意变剪切下 Δ(M meanVelocity) 等于全过程净冲量；profile 瞬时响应不包含瞬态或对流惯性方程。
- p_EOS 与 p_mechanical 不同或法向黏性非零时，上下完整应力完全相同。
- 非零质量流、相变或孔隙通道中，全部质量携带动量的跳跃条件已经闭合。
- 曲面/非平行面/移动几何中，侧压力和法向体力均已由简单 Gδ 项覆盖。
- 有实际运动基底时，当前 phaseViscousWork=0 已提供正确的机械功。

这些是需独立说明或设计的模型/离散边界。保留它们不等于宣布所有总动量测试合格，也不要求为本 witness 新增完整惯性膜模型。

## 七 实施后的最小证据

1. 用实际原生 helper 验证常压力、跨 record 变压力、Euler/RK2/RK3、一次部分切片与最终余量。测试实现自身，不能用另写 surrogate 算例代替。
2. 验证材料失败 retry、重放和接受一次，压力积分无丢失或重复；原 gas 包数值算序与热/质量操作不被新力元数据改动。
3. 用非零质量包证明压力力计算没有使用 total momentum 代替压力，也没有新的零质量专用分支。
4. 用 p_EOS≠p_mechanical、法向黏性、剪切和 α≠1 对照，验证原来各项的定义仍保留；不要把这些对照的全残差强行置零。
5. 重跑原 stationary wet GPU witness 至 t=0.001，保留真实 stage 压力、材料子步积分和 SolidFilm 反力。现有误差门槛不改，解析预测不是正式 PASS。
6. 移动几何和非零边界速度若未执行上述完整扩展，报告适用范围；本轮无需为得到“通用”结论而改写现有热力学恢复或提高 FP64 迭代次数。

## 文件与复核命令

- REVIEW.md：独立结论。
- REVIEWED_DESIGN.md：被审设计固定快照。
- metrics.json：独立原生证据重算与固定源引用。
- recompute_review_metrics.py：仅读取固定源码/保留证据并写本目录 metrics.json 的复算脚本。
- SHA256SUMS：交付校验。

复算：python recompute_review_metrics.py。核验：sha256sum -c SHA256SUMS。实际代码变更、接口批准和 GPU 验证仍未发生，本审查也未更改 restart strict-bit 的原有结果。

## 修订稿复核

15:27 UTC 追加：已复核作者修订稿第 5–7 节，SHA256 为 bd09f39d84c34e7119e45b0d0fd16be72e896bf642ecb963ca9684e3d0825016，快照见 REVIEWED_DESIGN_REVISION.md。修订稿现已选定本文建议的单标量、无 α、保留 CPU 几何/剪切/能量的方案；上述原始设计中的通用完整应力替换问题已从所选实现路径移除。

该修订的数学和数据路径通过本次只读设计审查，仍不是接口授权、实现完成或 GPU PASS。两个实施注意点已反馈作者并纳入该修订：As/ns 赋值的准确源码行号是 CpuSurfaceInterface.C:45–46；cursor 旧 overload 若丢弃压力输出，也应通过同一内部消费逻辑推进压力 used/remainder，防止混用 overload 重复消费。常压力对照只要求既有数值预期，不承诺新求和与旧乘法逐位一致。
