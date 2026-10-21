# reactingSst 24节点：GPU耗时定位

结论：新版相对2f4a66f增加的GPU计算步耗时，约90%落在壁面BVP求解，约10%落在保守剖面重构。CPU同步等待是在等GPU完成；本次跟踪没有发现数据拷贝或普通气相/SST更新成为主要增量。

## 对照条件

旧数学源码：`2f4a66f82c8405fad65542490c4d88841b2b9cd4`；现有数学源码：`21c6397d7a9981048c193c749ab30abc5ca96c45`。本轮只添加报告、证据和分析脚本，未修改求解算法或重新编译。

复用上一轮20步初测的两个二进制，逐个SHA256核验相同。旧源码只带相同constexpr编译选项和可选CUDA栈设置的支持补丁，旧数值算法未改；补丁随证据包保留。

两端31个初场/constant/system文件SHA256完全相同：reactingSst/24节点，固定dt=2e-7，20步，无喷出，Air=1/CO2=0，maxIterations=60，相对/绝对容差1e-8/1e-10，stretch=2，writeProfiles=false。同一RTX4060 Laptop 8GB、CUDA13.1、sm89、-O3、--fmad=false、--expt-relaxed-constexpr、-Ofc=mid、128KiB线程栈。

算例来自用户指定baseline的原始网格/0场，reactingSst设置由启动manifest的原始fluidProperties哈希恢复；没有使用该目录后来切换的wallFunction配置。此处是纯Air无喷出性能样本，不能代替非零混合组分条件的性能与精度验证。

## Nsight Systems 实测

两端分别完成20步。每个物理步两个RK子步，每端共40次BVP和40次重构核函数。用第3步第一次BVP开始至第20步第一次BVP开始作为对齐区间，包含第3–19步共17个完整计算步；剔除前两步和末步写场。CUDA跟踪确认只有一条执行流，核函数没有时间重叠，可相加归因。

以下为**带跟踪时的平均GPU核函数时间**，单位ms/物理步；增量占比以这17步执行时间差计算。

| 部分 | 核函数 | 旧版 | 新版 | 增量 | 占总耗时增量 |
|---|---|---:|---:|---:|---:|
| 壁面BVP求解 | `evaluateGasBoundaryLayerTeamKernel` | 217.107 | 522.395 | +305.288 | 90.03% |
| 保守剖面重构 | `reconstructConservativeWallPool` | 235.722 | 269.559 | +33.837 | 9.98% |
| 其余GPU核函数合计 | 气相通量、梯度、SST显式更新等 | 12.682 | 12.592 | -0.089 | 约0% |
| 整个对齐区间/17 | 含核函数间隙和必要主机工作 | 467.835 | 806.945 | +339.110 | 100% |

新版BVP占GPU核函数时间64.93%，重构占33.50%，两者合计约98.4%。BVP总时间为旧版的2.406倍，重构为1.144倍。

- 核函数执行占对齐区间：旧99.503%，新99.703%。GPU主要在执行这些核函数，CPU调度/写场不是此区间主要瓶颈。
- 17步全部Host↔Device memcpy的GPU时间：旧3.049ms，新3.057ms；仅占区间约0.04%/0.02%。API端cudaMemcpy总时间约23/24ms，同样很小。
- cudaDeviceSynchronize主机等待：旧7.775491s，新13.599098s；两端各68次。此等待与GPU计算重叠，不能再次加到核函数时间，也不能据此声称CPU隐式计算变慢。
- 两端主要普通核函数时间基本相同，例如内部气相面通量3.853/3.807ms每步、SST显式通量和源项更新约0.414ms每步。

剔除冷启动很重要：新版第一次BVP约5.266s，旧版第一次约0.229s；首次分配/调用等冷路径不能混入稳定区间。后续BVP各步时间仍会随Newton收敛工作量变化，本次是短程定位样本，不是长期统计性生产吞吐测试。

Nsight跟踪本身有开销；上述数值用于增量归因，不能替换未跟踪的效率初测。此前同配置无跟踪20步结果：旧版中位0.421237s/步，新版0.736572s/步，新/旧1.748596倍，即耗时增加74.86%。完整原始记录随包保留。

## 现在能确定与仍未确定的事

动态证据已定位到BVP和重构两个设备核函数。不能把90%的增量进一步直接等同于LU分解开销，也不能直接认定“隐式改显式”就能提速且保持结果。

源码提供的下一层检查位置：

- `common/gasWall/WallCoherentSolver.H`：Engine::solve内的Jacobian构造、残差更新、线搜索和缓存LU求解；首个壁面通量闭合还有嵌套3变量Newton。LU分解和回代由warp leader执行，其余lane同步。这是执行结构证据，并非实测子阶段耗时占比。
- `common/gasWall/WallCoherentNumerics.H`：带状LU分解/回代实现。应先拆分“残差/闭合、Jacobian、factor、backsolve、line-search”的设备计时，区分额外迭代次数和单次成本，再选择优化位置。
- `common/gasWall/reconstruction/ConservativeProfile.H`：fitTeam的标量矩匹配与能量根求解。当前纯Air样本跳过混合组分2×2 Newton，因此本次降速不能归因于该混合分支。
- 两版BVP入口都使用254寄存器/线程；编译日志显示累计设备调用栈由10072B增至30864B。这是资源压力线索，未取得硬件计数前不能宣称已证明内存spill/occupancy导致降速。

尝试用Nsight Compute对第7次BVP调用采硬件计数。第一次因旧运行目录缺少TMPDIR终止；补建临时目录后，工具在后端启动阶段停滞，前端等待IPC、后端futex等待，GPU利用率/分配显存均为0，未进入核函数。已终止该次采集，保留失败日志；**没有可用的NCU硬件计数**。未更改Windows驱动或系统设置。Nsight Systems两端跟踪完整有效。

下一步若做优化，优先拆分BVP内部设备计时，保持同一残差、容差、节点和物理方程；暂不把壁面闭合改为单次显式更新。

## 复现和原始证据

`evidence.zip`包含两端`.nsys-rep`、SQLite、全部CUDA/API/拷贝/OSRT统计CSV、运行日志、输入与二进制哈希、初测原始数据、旧版编译支持补丁、实际命令、源码检查位置和采集/分析脚本。`archive.json`给出证据包SHA256和逐文件哈希。`analyze_gpu_profile.py`直接读取SQLite重算表格，无需GPU。

采集脚本实际使用 `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --wait=all --export=sqlite` 包裹原有run-case.sh。未启用fork前跟踪、统一内存缺页跟踪或额外墙面审计。完整参数和运行环境见`runs.json`。

最终核验无gasUGKP、gasUGKPCudaBackend、nsys、ncu活动进程，无后台对比任务残留。PR保持草稿，不合并。本轮不声称解决性能回退或改变既有数值验收结论。
