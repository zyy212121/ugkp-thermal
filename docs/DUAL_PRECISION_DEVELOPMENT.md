# 双精度路径的构建与维护

## 源码和产物

- 根目录数值源码保留原 FP64 路径；原始参照提交为 a3793e59。FP32 差异文件放在 precision/fp32/<原相对路径>，导入来源记录在 precision/fp32-import.json。
- tool/precision/materialize_sources.py 将根源码分别物化到 .build/precision/fp64/source 与 fp32/source，只对后者覆盖 FP32 差异文件。不能用 -I overlay 代替物化：源文件内部的相对 include 会绕过搜索路径。
- 两棵树分别生成 common/PrecisionBuild.H，分别拥有对象、archive、二进制及 source-manifest.json，不复用另一精度的编译产物。两种模式顺序构建，CUDA 编译进程局部软栈目标为 256 MiB，受继承的硬上限限制。
- bash tool/build_local.sh 和根 Allwmake 构建四个 solver 的两种模式。各 solver 的原独立构建脚本也生成该 solver 的双模式。CHMT 的 --frontend-only 和 --host-checks 保留其原用途。
- 公开 bin/gasUGKP、gasUGKPSteady、CHT、CHMT 是薄启动器；独立后端位于 bin/.precision/32/ 和 64/。只有选定的后端被 exec，因此一个求解进程只有一套 resident owner、数据及 kernel/graph 地址。

## 启动和复算

将 tool/precision/precisionProperties.example 复制到算例的 constant/precisionProperties。precision 32; 是默认模式，precision 64; 显式选择原 FP64 路径。选择只在启动时发生，后端不重新读此设置，运行中改配置不会切换精度。启动器支持 OpenFOAM dictionary 的 #include 语法，并原样传递应用参数。

精度元数据由应用实际读写路径处理，不由启动器猜时间目录。OpenFOAM solver 在选定 timePath 检查并写 ugkpPrecision，CHMT 在实际 restartDirectory/checkpoint 目录检查并写。文件记录 schema、computePrecision、restartPrecision、storagePolicy 和 wireEncoding。FP32 计算结果即便按 double wire 写出，restartPrecision 仍为 32。

已明确授权的复算行为：
- 已知保存精度低于选择精度：error 并停止复算。
- 已知保存精度高于选择精度：warning，并按选择的较低精度继续。
- 无元数据或元数据未知：warning，保留原 reader 行为。不从 double wire 或旧 schema 猜测生产精度。旧 FP32 失败实现可能写出与 FP64 相同的 wire schema，无法可靠区分。

写状态之前撤销旧精度记录，所有完整状态文件成功后才原子发布新记录。普通 OpenFOAM 场写入期间，新 stream 的默认精度临时提高至至少 double 的 max_digits10，保留用户更高值，退出写入范围后恢复。既有日志流的格式不改变。不因为用户 writePrecision 较低而新增拒算。

## 后续算子和调度维护

每一项后续算法、算子、调度和参数变动都必须兼顾 FP64 与 FP32：同参数、同功能、同验收覆盖。优先共用经过审核的类型化实现，避免复制算法后两边分别演进。FP32 为稳定性采用不同表达式时，必须明确保留原 FP64 表达式，不把新公式实例化为 double 就称为原 FP64 基线。

原 FP64 参照提交单独保留。新增辐射、稀疏求解、导热及其他 FP32 owner/kernel 都必须纳入同一双路径维护范围，后续 checkpoint 方案预设两种模式的对应覆盖。工具/假后端/文件元数据检查不能代替真实算例的两路径验收；运行真实验收仍按用户选定的 checkpoint 批次执行。

tool/precision/report_overlay_pairs.py 输出只用于提醒的源 hash 配对报告，构建也写 .build/precision/overlay-pairs.json。它列出原 FP64、当前 FP64、初始 FP32 导入与当前 FP32 的 hash，帮助发现只改根文件却忘记 overlay 的情况。报告不是求解门禁，hash 相同或不同都不能证明两模式的功能或数值等价。

## 非数值检查

- bash tool/precision/build_launcher.sh：只编译 host 启动器。
- python3 tool/precision/test_tooling.py：物化与假后端启动测试，不调用真实 solver。
- tool/precision/test_restart_metadata.cpp：仅检查元数据读写与精度比较，可分别用物化后的 32/64 common 头编译。
- 对应用入口和 checkpoint codec 做 host 语法检查不等于完成 CUDA 编译或数值验证。
