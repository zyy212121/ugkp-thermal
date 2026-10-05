# 输入、解析参考与比较判据

## 1. 状态、量纲和输出契约

这些判据在任何 CHMT 求解器结果出现前冻结。`cases.json` 中除 TACOT 外使用无量纲数学参数；不能当作真实材料物性。生成器只支持已冻结的解析配置，不是可任意改参数的物性/求解器框架。修改物理配置必须同时审阅公式、容差和生成器，不可只修改未参与某一公式的 JSON 字段。

CSV 的唯一键为 `case,variant,n,time,sample,quantity`，`n` 为每方向单元数；局部代数检查取 0。`time` 为经过时间；压力脉冲的 time 列为压力状态序号，不表示指定的脉冲时间历程。`sample` 为 1-D 单元编号、2-D `i:j`、剖面采样编号或 global/local。

`value` 为观测值；参考的 `scale` 为预设非零归一化尺度；`kind` 必须区分 `ANALYTIC_REFERENCE` 与 `SOLVER_OBSERVATION`。求解器导出可省略 scale，比较器只使用参考尺度。所有字段必须齐全、无重复、有限；缺行、多行、NaN、Inf 均拒绝。几何/场使用指定的单元平均或单元积分，不能拿中心点值代替平均值。三套 n、全部指定时刻都必须导出。

真实运行 run.json 必须含：

```json
{
  "artifact_kind": "SOLVER_OBSERVATION",
  "solver_commit": "实际求解器提交",
  "adapter_commit": "实际适配器提交",
  "build_id": "实际构建标识",
  "input_sha256": "64位小写十六进制输入规范SHA256",
  "precision": "FP64",
  "hardware": "实际GPU型号/架构或该局部契约所用CPU",
  "compiler": "实际工具链及版本",
  "command": "实际完整运行命令",
  "started_utc": "实际UTC时间",
  "time_step": 0.00025,
  "steps": 1000,
  "source_output_sha256": "64位小写十六进制原始求解器输出SHA256"
}
```

这里是字段说明，不是可用于通过检查的结果文件。还须保存实际 case 网格/字典、完整日志、每个分辨率的步长、构建配置及输出散列。CLI 只检查元数据格式，不能认证其真实性，不能代替检查真实文件/命令日志。初始条件均用解析单元平均初始化。

### 本地运行适配器契约

`run_case.py prepare` 只生成中性 case-spec.json、解析 reference.csv 和带 SHA-256 的 request.json。真实 CHMT 尚未实现，因此本包不能提供已验证的 OpenFOAM 字典/可执行适配器；不会猜测尚不存在的求解器字段。

后续适配器须是用户显式指定的本地可执行文件，并支持：

1. `adapter --describe`：输出 JSON，含 `schema_version:1`、`artifact_kind:"CHMT_ADAPTER"`、`supported_cases:[...]`
2. `adapter run --spec /absolute/case-spec.json --solver /absolute/CHMT --output /absolute/run-directory`：把规范转换成真实求解器输入，执行全部变体/分辨率，导出 observations.csv、run.json 及原始输入/网格/日志；不得改写解析 reference.csv 或 case-spec.json
3. run.json 的 input_sha256 必须等于准备后的 case-spec.json 散列；每个实际运行的详细时间步/构建信息都须保存，顶层元数据不能代替分辨率明细

run 入口要求先 prepare，检查真实可执行文件、能力声明、输入一致性和既有结果冲突；保留 adapter.log，失败返回非零。它不会编译新 CHMT、下载软件或提供 mock。run 自动比较字段/预算并生成适用的空间细化报告；不会把返回码 0 解释为完整验证。stage GCL 和独立时间细化仍使用各自入口/报告。

## 2. 独立可核算的公式

### Stefan：一维单相熔化

取热扩散率 1，密度、cp、k、潜热均为 1。固体处于熔点 0，左壁 1，右壁 0；无流动。定义 λ 为

`sqrt(pi)*lambda*exp(lambda^2)*erf(lambda)=1`，由二分法求根，不采用参考网页四舍五入的 0.62。

`lambda≈0.6200626333135955`；`t_abs=t+(0.1/(2*lambda))^2`；前沿 `s=2*lambda*sqrt(t_abs)`。

液区 `T=1-erf(x/(2*sqrt(t_abs)))/erf(lambda)`，固区 T=0。h=液相分数+T；对跨界面单元解析积分，固体熔点的焓零点为 0。积分 erf 的原函数为 `x*erf(x/a)+a*exp(-(x/a)^2)/sqrt(pi)`。

全域 `E=s+integral_0^s T dx`；累计壁面热输入 `Q=2*(sqrt(t_abs)-sqrt(t0))/(sqrt(pi)*erf(lambda))`。用实际导出的各单元 h 和真实接受的壁面累计热量检查 `E(t)-E(0)-Q=0`，不从解析解重构求解器的壁热。该问题采用与 [Stefan 相似解](https://chowland.github.io/AFiD-MuRPhFi/examples/stefan/) 一致的熔化映射。

### 液膜速度、焓与机械功

G=沿面压力梯度−rho*g_t；z=0 为材料基底，z=δ 为顶面：

`u(z)=ub+tau*z/mu+G*(z^2-2*delta*z)/(2*mu)`

`u_mean=ub+tau*delta/(2*mu)-G*delta^2/(3*mu)`

`u_top=ub+tau*delta/mu-G*delta^2/(2*mu)`

`tau_bottom=tau-delta*G`

`Phi=delta*tau^2/mu-delta^2*tau*G/mu+delta^3*G^2/(3*mu)`

独立核算 `Phi=tau*u_top-tau_bottom*ub-delta*u_mean*G`。此式来自乘以 u 的准稳态动量方程积分。质量通量为 rho*δ*u_mean；厚度均匀焓时焓通量为 rho*δ*h*u_mean。不能用 u_top 代替 u_mean。

焓波 H0(x)=0.03+0.002*sin(2πx)：

- `film_plug`：ub=0.25，tau=G=0，`H(x,t)=H0(x-0.25*t)`，Phi=0
- `film_shear`：ub=0，u_mean=0.25，Phi=0.025，`H(x,t)=H0(x-0.25*t)+0.025*t`

CSV 用正弦的单元平均。周期域总质量固定；剪切例 `integral H(t)dx-integral H(0)dx=0.025*t`，等于气侧牵引所做的功。焓方程只加 Phi；不能又加一次顶面剪切功。以上两例 K 不随时间变、周期动能边通量净和为零，因此所省略的惯性项 R_K=0；这不能证明一般变厚/变速膜的完整能量守恒。

压力脉冲使用真实焓：`H=rho*delta*cp*T+delta*(p-p_ref)`，`U=H-p*delta`。rho、δ、T 固定，p 改变时 H 改变而 T/U 不变。若采用参考压力焓，必须同步去掉压力源。

蒸发局部例使用一致质量跳跃恢复两侧密度；完整气侧能量通量为 `F_g=J*(h_g+|u_g|^2/2)+p*w_n`。液侧热输入 `Q=J*(h_l+|u_l|^2/2)+p*w_n-F_g=-1.8`；液体焓方程 RHS=`Q-J*h_l=-2.4`。分别输出 F_g、Q、RHS、两侧密度，发现重复潜热或漏记气侧喷注动能。它只验证代数契约，不验证真实蒸发动力学。

### ALE/GCL 与密度波

使用 [Pan 等的 ALE 验证思想](https://www.math.hkust.edu.hk/~makxu/PAPER/ALE-GKS.pdf)，这里冻结的是自定义光滑周期网格，未复现其高阶方法或表格。网格节点按 cases.json 中的映射移动；最小方向 Jacobian 为 `1-2*pi*0.025>0`。每个单元是随时间伸缩的矩形，二维面积必须实时更新。

自由流 rho=1、u=v=p=1、gamma=1.4、Y=(0.3,0.7)。密度波满足 `rho=1+0.2*sin(2*pi*(x+y-2*t))`；速度 u=v=1 时波速相位为 **2t**，不能照抄成 t。单元平均为中心正弦乘 `sinc(pi*dx)*sinc(pi*dy)`。单元质量、动量、总能量、物种质量全部取积分；`rhoE=p/(gamma-1)+rho*(u^2+v^2)/2`。

必须额外导出每个真正 RK stage 的几何 ledger：

`step,stage,cell,V_old,V_new,sweep_left,sweep_right,sweep_bottom,sweep_top`

四个 sweep 是该 stage 所实际采用的带符号、朝外的累计面扫掠面积；必须与对应 stage 的体积系数/时间权重一致。对一般 RK，不得把不同 stage 的最终体积和瞬时网格速度拼在一起。单位三维厚度下它们等同体积。CLI 检查 `(V_new-V_old)-sum(sweep)`；所有索引从 0 起，`--steps/--stages/--cells` 必须匹配完整导出。直接填 0 残差不能通过该接口；实际四面扫掠量必须来自求解器，不得用体积差倒推出缺失项。这里只审计同一几何离散的闭合，stage 权重/网格速度与守恒通量的一致性仍需代码审阅。

### 守恒 remap

在固定 [0,1] 上一次重分区，物理时间不前进。旧单元内守恒密度为分片常数，密度跳跃固定在 x=0.375，移动后的单元确实跨越该跳跃，能排除逐编号拷贝密度的错误方法。目标单元积分按 `sum_i(q_i*length(old_i intersect new_j))` 精确得到。输出质量、动量、能量、两个物种质量。检查每单元对照、全局总量、均匀场保持、步阶密度界限 [1,2]；单元体积必须正。这里只验证分片常数重映射算子，不能宣称已验证三维 remeshing 或高阶重构。

## 3. 预先冻结的误差预算

FP64 epsilon=2.220446049250313e-16。点/单元误差 `e_i=abs(obs-ref)/scale_i`；加权 L1=`sum(e_i*scale_i)/sum(scale_i)`；Linf=max(e_i)。积分量 scale 含单元体积，因此相应 L1 为体积加权；膜焓波以振幅 0.002 归一化，ALE 波以密度振幅 0.2（乘相应守恒分量系数与体积）归一化，不能用大的常量背景掩盖波形误差。

- 初始精确平均、局部膜代数、压力/相变契约、几何坐标对应的体积：L1/Linf ≤256 epsilon
- 单 stage GCL：abs(ΔV−Σsweep)/max(V_old,V_new,Σabs(sweep)) ≤64 epsilon
- 自由流每个守恒分量：L1/Linf ≤5e−10；适用于规定的 1000 步 FP64 累积预算，不要求逐位相同
- 守恒 remap：局部 ≤256 epsilon；全局总量误差 ≤256*n*epsilon
- ALE 全局各积分量、膜/Stefan 能量预算：归一化残差 ≤5e−10；参考总量或初末能量/累计外部功提供非零尺度
- Stefan n=128：T/h 的 L1≤0.01、Linf≤0.05；前沿与累计壁热相对误差≤0.01；更粗网格限值乘 128/n
- 膜焓波 n=128：振幅归一 L1≤0.005、Linf≤0.02；粗网格限值乘 (128/n)^2
- ALE 波 n=32：振幅归一 L1≤0.03、Linf≤0.08；粗网格限值乘 (32/n)^2

这些截断误差上限是初始验收要求，不是从现有成功结果反推的容差；也不代表误差达到论文相同精度。FP32、不同精度累加、不同网格/算法应另建冻结判据，不共享本预算。

## 4. 收敛与完整验收

三网格误差 CSV：`case,metric,n,error`。error 应取上述归一化 L1（front 可取相对误差）；必须有限，且两个误差都大于 256*epsilon；否则记为不确定，不能建立收敛阶。`p=log(e_coarse/e_fine)/log(n_fine/n_coarse)`，两组相邻细化均须通过：Stefan ≥0.9；膜平滑输运/ALE 波 ≥1.8。它们是拟接入适配器必须达到的最小空间阶要求，不承诺尚不存在的求解器已经二阶。若采用不同阶数，必须在运行前单独审阅并版本化规范。

时间步按 cases.json 缩小以隔离空间误差，并再减半证实时间误差不主导。独立时间收敛至少 3 个 dt、固定足够细网格；目标阶需由实际时间离散在运行前声明。当前 convergence CLI 只自动检查冻结的空间阶，不自动认可时间阶。全零误差不证明收敛。

完整的单例“已验证”需同时有：真实输入/源码/构建/硬件与运行日志、全部观测对照、全局守恒、细化（若适用）、逐 stage GCL（若适用）、独立审阅。局部契约例不需要伪造网格收敛。全耦合还需检查两界面运动、唯一接受账本、相变携能、物种/元素、R_K 与薄层近似余项，本包没有那些运行结果。不能从一个局部匹配推导整个 CHMT 已验证。
