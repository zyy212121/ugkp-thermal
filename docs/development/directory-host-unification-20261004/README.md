# 2026-10-04：目录主机流程统一与 gas baseOnly 入口修复

本轮接受评审的三点划分：full/split 是两份主机组织流程，不能据此认定物理算法错误；gas 的 baseOnly 保留；FSH/CHT 不扩展 baseOnly。仅修改 WSL `ugkp-thermal-test`，生产双库和 E 盘不变。上传 GitHub main 的父提交为 `9c3b9e9d1edccf813c634627ffa74e8082d243f3`，现有案例和历史提交保持。

## 复现

旧 gas 的真实 ProbeBinPre 阶段只根据 useSplitPreDirectory 传递 splitBaseAndInjection，忽略 preInjectionSegmentActive。压缩阶段已发布的 baseOnly 任务在下一个非检查步被误判成不同目录而重建；检查步或首次 L1→L2 激活同样发布了错误的目录标签。

新增回归直接提取生产推进函数中的 ProbeCompaction 和 ProbeBinPre 代码，使用实际 CUDA/CUB 目录与任务生产者，再运行实际碰撞池归约。修复前四配置、三目录、四状态、两线程规模的96项中：gas无注入的已有L2非检查步、已有L2检查步、L1启用各两个线程配置，共6项按预期失败，90项通过。CHT32首次编译进程异常退出，未改源码的重试24/24通过；原始日志和合并清单保留，不把编译中断归因于算法。

## 修改

- `common/GpuParticleDirectoryHost.cuh` 是三个应用唯一的 full/split 主机协议：清目录、计数、scan、写游标、scatter、任务失效与准备。目录修改前置 csrTasksReady=0，生产者成功排入 CUDA 流后发布就绪。
- 公共 `prepareParticleDirectoryAndSchedule` 按目录准备 → 当前目录分类 → 公共 auto → 任务可执行的固定顺序返回。三个实际生产入口都调用它，不再自行拼接 runToolB3 与目录枚举。
- `GpuParticleDirectoryHostPolicy.cuh` 仅包含编译期能力、字段和核适配。gas full 为 full；split无注入为 baseOnly；split有注入为 splitBaseAndInjection。FSH/CHT 参数仍是 full/split 布尔值；无注入时仍执行原热应用 split 目录准备。
- gas 独有的 source-free 分支在编译期能力边界内保留，只清零空注入 offsets、发布主机标志，复用压缩阶段的直接 base 任务。非检查步实际 count/materialize 启动次数均为0，描述符原样保留；检查步仍按已有 auto 流程重建。
- 未改任何设备核、物理公式、阈值、随机流、粒子数据布局、归约块几何、FMA 或精度策略。thermal 原先单独的 post-full 包装也集中到 common。适配后的源码结构检查定位共同所有者与新调用链；旧的无关历史检查不以修改预期值方式掩盖。
- 新公共文件登记在受管镜像清单，维护入口更新当前归属和复现方式。没有机械同步冻结的流体库。

## 验证与限制

专项 **171/171通过**：新增生产链96项、原公共 auto 和实际碰撞消费者75项。新增测试涵盖 gasUGKP64、FSH64、CHT64、CHT32 × full缓存缺失回退/无注入/有注入 × L2非检查/L2检查/L1首次启用/L1非检查 × 32/128线程。

测试首先执行实际 survivor 全目录、压缩提交及 base 捕获，然后执行实际下一步入口及消费。测试二进制通过 CUDA launch ABI 的链接包装观察任务构建，不向生产添加计数或计时逻辑。GPU 原子分箱可能重新排列颗粒，L1参照使用提交后的同一颗粒集。逐粒子选择和 RNG 一致；独立 CPU 对照质量、三分量动量、能量、粒径一二阶矩，并验证任务范围、目录类型、重载单元清单和碰撞池计数。三个FP64容差2e-12，CHT32容差2e-5。

**48/48 Compute Sanitizer memcheck通过，0错误**，覆盖四配置、三目录和四调度状态。四种原生 CUDA 后端对象按原O3/sm_89/FMA策略编译成功。受管文件清单234项、四配置字段生命周期及160项策略编译检查通过。生产源文件/链接冻结复核：thermal503项、流体244项均未变化。

完整 pytest 对照：修改前 **608通过、170失败、25错误、38跳过**；修改后原始完整运行 **703通过、171失败、25错误、38跳过**。其中唯一新增失败是 gas split 压力回归的编译进程报 `internal compiler error: Segmentation fault`，没有进入运行阶段。未改任何输入或预期值，单独重跑该项通过，实际有限面通量、颗粒/Eulerian矩、总质量/动量/能量闭合检查均通过。补测合并后 **704通过、170失败、25错误、38跳过**，没有新增可复现失败或错误。原始 [完整测试结果](suite-after-summary.json) 未覆盖改写，另存 [包含明确补测的结果](suite-after-verified-summary.json)；[对照清单](suite-comparison.json) 同时列出原始新增失败及补测通过名称，完整日志和JUnit XML以gzip归档。基线为上轮同一已上传源码的完整测试记录，本轮开始复核上一提交155项交付哈希全部相同。完整仓库仍不是全绿，不能将本轮专项通过写成全工程验收通过。

没有执行效率计时，不声明整体加速比。通过启动追踪证明去除了 gas 非检查步多余构建；保留原检查步任务准备成本。纯主机流程提取不新增性能调参，本次提交不依赖效率结果。

## 复现

`python3 -B -m pytest -q tests/test_production_directory_cuda.py tests/test_shared_auto_schedule.py tests/test_gas_auto_pool_cuda.py tests/test_thermal_auto_pool_cuda.py`

CUDA夹具可用 `UGKWP_CUDA_ARCH` 覆盖sm_89，使用512MiB编译栈。内存检查：`compute-sanitizer --tool memcheck --error-exitcode 99 <production-directory各配置二进制> <目录0/1/2> <状态0/1/2/3> 32`。完整测试集加载OpenFOAM环境并设置 `UGKP_MANAGED_MIRROR_ROOT` 指向测试库，产物写到源码外。

代码补丁 [changes.patch](changes.patch)，输入白名单/哈希 [changed-inputs.json](changed-inputs.json)，修复前 [合并结果](red-combined-summary.json)，专项 [JUnit结果](green.xml)。实际消费者日志在actual-consumers，内存检查及编译日志分别在memcheck和build。按用户授权直接上传main，未替换生产运行文件。
