# 气体分子黏度：Sutherland + Wilke

三个应用共用 `common/gasTransport/MolecularViscosity.H`。默认仍为 `const`；物性只影响气体动量输运及原先依赖气体黏度的热输运、SST、壁层和颗粒交换。gasUGKP、CHT、CHMT 的应用分工不变。

## 配置与物理量

在案例 `constant/gasTransportProperties` 中指定：

```foam
FoamFile { version 2.0; format ascii; class dictionary; object gasTransportProperties; }
viscosityModel sutherland;
species (A B);
sutherland
{
    A { muRef 1.8e-5; TRef 300; S 110.4; molarMass 0.028; }
    B { muRef 2.8e-5; TRef 300; S 130.4; molarMass 0.044; }
}
```

以上系数仅用于演示格式，不代表真实 A/B 物种的拟合数据。单位分别为 Pa·s、K、K、kg/mol。所有参数须有限，`muRef`、`TRef`、`molarMass` 必须为正，`S >= 0`。组分名称及顺序须与 `gasModelProperties` 一致，分子量须与热力学表一致。单气体 legacy 模式只配置一个物种；CHMT 使用其选定的 `singleGasSpecies` 名称。三个应用均先经过同一个 OpenFOAM 字典读取入口，支持相对路径 `#include` 和 `$变量` 替换，再交给共享物性解析器。

各组分使用

$$\mu_i(T)=\mu_{i,ref}(T/T_{i,ref})^{3/2}\frac{T_{i,ref}+S_i}{T+S_i}.$$

混合物使用 Wilke 规则：

$$x_i=\frac{Y_i/M_i}{\sum_jY_j/M_j},\quad
\mu_{mix}=\sum_i\frac{x_i\mu_i}{\sum_jx_j\Phi_{ij}},\quad
\Phi_{ij}=\frac{[1+\sqrt{\mu_i/\mu_j}(M_j/M_i)^{1/4}]^2}{\sqrt{8(1+M_i/M_j)}}.$$

实现直接使用未归一化的 `Y/M` 或 `rhoY/M`，利用分子、分母的归一化因子抵消。纯组分端点直接返回该组分的黏度，零浓度组分不参与计算。局部温度非正、浓度非有限/负值或没有有效组分均产生无效物性并进入现有诊断流程；不静默截断物性。

删除此文件，或写 `viscosityModel const;`，继续使用各应用原来的气体常黏度配置，包括原先允许的零黏度路径。常黏度分支保留原有运算顺序，不把常数重新经过混合规则计算。

## 导热率、扩散及颗粒

| 原有选择 | 本实现 |
| --- | --- |
| 气体 `kappa = mu Cp / Pr` | 使用同一局部温度、组分对应的 `mu(T,Y)`、`Cp(T,Y)`，保留配置的分子 Pr |
| 独立配置气体 kappa | 保留直接指定的 kappa，黏度变化不覆盖它 |
| 数值壁层的定 Pr 导热 | 各节点使用自身 mu；区间延续原来的焓差/割线 Cp 守恒离散 |
| 固体导热率、孔隙黏度、液膜黏度 | 保留各自物性 |
| 组分扩散 D | 保留原有独立配置；不由 mu 或 Schmidt 自动改写 |
| 颗粒物性 | 颗粒密度、热容、粒径等不变；仅更新周围气体 mu 参与的 Re、阻力和换热 |

气体黏性面通量和 SST 使用局部气体 mu。壁函数使用气侧壁面温度/组分；数值壁层的每个节点独立计算 mu，并同步用于 SST 混合函数、壁面渐近式、阻尼、应力功、导热和重构通量。CHT/CHMT 传给固体的仍为原有界面交换量。CHMT 喷出气体改变当地组分后，后续气侧计算自然得到新的混合黏度；CPU 界面使用同一主机/设备物性函数。

解析 `constantTransport` 壁层假定常输运系数，不能与 Sutherland 同用；请选择已有数值 `reactingSst` 或受应用支持的 `equilibriumOde`。这项检查避免把变黏度偷偷冻结。数值壁层仍要求已有的显式扩散系数配置。

既有能力限制不变：gasUGKP/CHT 的多组分与颗粒库存耦合尚未实现；本次不启用这个组合。CHMT 的颗粒模型使用其原有直接换热系数，未新增粒子阻力模型。CHMT 的传统 wallFunction / equilibriumOde 枚举组合也不因本次物性扩展而自动获得支持，其已有数值 reacting 壁层路径使用节点局部物性。

## 执行与所有权

- Sutherland 表及 Wilke 分子量配对因子只在初始化时上传。每步在现有 CUDA 算子内用当地 T、Y 求值，无新的黏度主机场、每步上传/下载或物性同步屏障。
- `MolecularTransportDevice` 拥有表，设备视图只借用指针。CHMT 事务存储交换后重新绑定视图，表不随每次状态交换重传。
- 壁层 `LayerPoint` 缓存节点 mu；SST seed 热力学缓存同时保存/恢复 mu，复用已有缓存空间。原有协作线程、批次、图和流调度保持可用。
- gasUGKP 的组分、数值壁层存储/API/适配器抽到 `common/gasTransport/Resident*`；CHT 调用同一份实现。原私有路径保留薄 include，不复制数值求解器。
- CHT 的共用气体阶段统一经过 `advanceChtSharedGas`；该路径不调用会在壁层尚未准备好时执行旧壁面算子的 legacy 启动调优器。没有为测试案例增加专用算子或底层分支。

## 重启

gasUGKP/CHT 写出 `gasTransportIdentity`，对模型、组分顺序及系数做身份校验。已有常黏度重启可不含该文件；Sutherland 非零起始时间必须保留它。组分重启保留已接受的 rho/rhoU/rhoE/rhoY，避免从输出精度下的 T、p 重建保守量；CHT 在耦合输出时也下载并保存组分。

CHMT 写出 schema 9，在配置区保存物性表及身份。schema 6--8 的常黏度物理状态继续可读；不能把旧常黏度检查点直接当作 Sutherland 检查点续算。读取新检查点会拒绝物性变化，并校验格式、维度、长度及校验和。检查失败时保留调用方原状态。未新增其它热力学/化学配置的一致性承诺。

## 复现

需 OpenFOAM 10、CUDA、可运行的 NVIDIA GPU。下列命令在仓库根目录执行；`out` 应指向独立构建目录。默认 Ns=2，算子单测另覆盖 Ns=3。

```bash
source /opt/openfoam10/etc/bashrc
export CUDA_HOME=/usr/local/cuda
export NVCC=$CUDA_HOME/bin/nvcc
export UGKWP_CUDA_ARCH=sm_89 CHMT_CUDA_ARCH=sm_89
export UGKWP_GAS_SPECIES=2
out=/absolute/path/to/isolated-build
export FOAM_USER_APPBIN=$out/bin GAS_UGKP_BUILD=$out/gas CHMT_BUILD=$out/chmt
mkdir -p "$FOAM_USER_APPBIN"
bash tool/gasUGKP/build_cuda_solver.sh
bash tool/CHT/build_cuda_solver.sh
bash applications/CHMT/Allwmake
python3 -m pytest -q validation/tests/gas_transport/test_molecular_viscosity_cuda.py \
  validation/tests/gas_transport/test_molecular_transport_config.py
python3 validation/molecular_transport/native_matrix.py \
  --bin-dir "$FOAM_USER_APPBIN" --work /absolute/path/to/new-results
```

本机完整应用验证额外使用 `--Ofast-compile=max -Xcompiler=-fno-tree-fre`，由 `NVCC` 包装脚本传入，未改默认编译参数；这是原生 CUDA 设备代码，不是 PTX 编译或主机模拟。三个 `*_CUDA_STACK_BYTES` 环境变量均设为 65536，runner 默认设置。默认优化构建在本机被系统终止，详见报告。复现相同编译模式可用：

```bash
cat > "$out/nvcc-validation" <<'EOF'
#!/usr/bin/env bash
exec /usr/local/cuda/bin/nvcc --Ofast-compile=max -Xcompiler=-fno-tree-fre "$@"
EOF
chmod +x "$out/nvcc-validation"
export NVCC="$out/nvcc-validation"
```

基线位于 main `82ea1d90`。在独立基线检出中应用 `baseline-cht-build.patch`，为 CHT 增加与其它应用相同的可选设备栈设置，并允许选择 NVCC 包装脚本；不修改基线数值算子。基线和候选使用相同编译参数、栈大小和输入。

`native_matrix.py` 自带全部初始场/字典；`fixtures.json` 的 `inherits`/`remove` 仅在测试准备阶段复用相同输入，覆盖文件可直接审查。物化后的 19 组输入已逐字节对照已验证的独立案例。脚本通过 blockMesh 建网格，调用真实应用。工作目录必须尚不存在，避免覆盖已有结果或撞到 CHMT 检查点的禁止替换策略。加入 `--baseline-bin-dir` 可做常黏度对照；`--only gas-pure gas-unequal-mass` 可选用例；`--memcheck` 对选定完整应用追加 compute-sanitizer。

`benchmark.py` 使用相同输入，预热一轮后交替运行 baseline / const / Sutherland 五轮，保存全部时间和常黏度物理场对照。计时包含进程启动和正常输出，不含网格生成。应在没有其它计算负载时运行；不能据短小验证网格推断大型生产案例的加速比。

参考：[Wilke 定义与质量/摩尔分数转换](https://idaes-pse.readthedocs.io/en/stable/explanations/components/property_package/general/transport_properties/viscosity_wilke.html)，[Cantera 气体输运实现](https://www.cantera.org/3.1/cxx/d8/d58/classCantera_1_1GasTransport.html)。实测记录见 [RESULTS.md](RESULTS.md)。

## 字典引用与 CHT 壁层输出回归

`io_regressions.py` 运行实际三个应用，比较嵌套相对路径 `#include` / `$变量` 与展开字面值的结果及重启；检查 CHT 的 ordinary/thermal-coupling 输出、独立气体输出分支、reacting/ODE 节点、关闭 profile 和未启用壁层的行为。输出目录中 `gasBoundaryLayer.csv` 为汇总，`gasBoundaryLayerProfiles.csv` 为节点流动/组分，局部 SST 模型另有 `gasBoundaryLayerSstProfiles.csv`（不导出壁面奇异 omega）。诊断下载只在选定写出时刻发生。

```bash
python3 validation/molecular_transport/io_regressions.py \
  --bin-dir /absolute/path/to/native/bin --work /absolute/path/to/new-io-results
```

可用 `--old-bin-dir` 指向修复前 PR26 二进制，重现 CHMT 引用配置启动失败和 CHT 未输出 CSV。
