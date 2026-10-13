# 固定原版与诊断版本复现

先checkout固定 source802a11e7dfe986488a8bf8305e86424dcc94c9e1的独立测试worktree，或确认生产目录与该source一致；当前交付commit只是文档。历史baseline d611由仓库bundle的可验证对象支持。不要在正式库/正式安装/原生产结果上复现。

本目录中的inputs/native小输入、fixture/generator/checker、patches和完整commands/spec/hash已包含实际测试信息。已执行命令的绝对路径是原现场路径；换位置时替换root并重新生成restart路径，仅调整路径，不改物性、方程、容差或失败断言。输入中少量restart指向原work/cases，必须使用本目录真实artifacts/chmt/checkpoint作为新位置的源。header/codecs与shared source来自固定仓库，支持脚本不替代生产实现。

```bash
export GIT_LFS_SKIP_SMUDGE=1
git worktree add --detach /your/test/pr14-fast 802a11e7dfe986488a8bf8305e86424dcc94c9e1
cd /your/test/pr14-fast
export delivery="$PWD/review/pr14/reviewer-deliveries/20261013T182722Z-fast"
# 若固定802a未带当前交付文件，先把本交付目录复制到同一相对位置。
bash "$delivery/inputs/env.sh" bash applications/CHMT/Allwmake
bash "$delivery/inputs/env.sh" env UGKWP_GAS_SPECIES=10 CHMT_BUILD="$PWD/work/build/CHMT10" FOAM_USER_APPBIN="$PWD/work/install/ns10/bin" bash applications/CHMT/Allwmake
bash "$delivery/inputs/env.sh" bash applications/gasUGKP/private_backend/build_private_backend.sh
bash "$delivery/inputs/env.sh" bash applications/FSH/private_backend/build_private_backend.sh
bash "$delivery/inputs/env.sh" bash applications/CHT/tools/build_cuda_solver.sh
```

CUDA/FOAM路径和sm89只适配本次实机；其他机器改变它们需重新资格化证据。第一次环境wrapper在source bashrc前set-e而退出，原5次日志/spec保留；正确env.sh在source之后set-e。该旧wrapper原字节未恢复，只有原spec SHA和明确失败条件，未编译/未启动GPU，不能把它记为生产编译失败。全部有效/关键失败GPU运行使用已交付正确env.sh。

原生case：复制inputs/native对应子目录到新work/cases（每次fresh output）；运行blockMesh，CHMT再运行solid blockMesh（system/blockMeshSolidDict，-region solid）；使用绝对路径work/install/.../bin，commands.txt记录全部实际参数。输入生成和延续详见prepare_cases.py/chmt_native.py与已保存checkpoint。固定源checker常量过时，第一次失败应保留，再用check_case_current.py验证原输出，阈值/断言不变。

化学红绿：从git archive d611的common编译同一个chemistry_retry.cu（-DBASELINE），原版使用固定common；调用参数 `1 2e-6 1`、`1 1e-6 1`、`0 2e-6 1`、`1 2e-6 100`。完整nvcc命令在对应spec。P002只在隔离common应用patch后编译；核对before/after sha，用同check_chemistry_retry.py，不改red断言。正常JSON值比较没有声称raw bitwise；timed perf_chemistry.cu另外输出完整field bit-stream hash。

冷壁：输入CHT-coldWall1D/2D-INPUT2不改物性，原版2D输出NaN仍保存。P006是printf诊断，不能计速度。P007仅生产现有scalar fallback；P010再保存profile cast余量。两个patch合用的latest诊断以resident_check_fp64_mean.py验证每个timestamp（abs1e-9K），保留原cast；首次场与profile/carry先red后green。FSH共享模板、mixed coupled faces/实际预算不在这项PASS范围内。

Co：P011-v2只将原尾部7padding字节变成显式初始化metadata成员；原double/bool位置、16byte size与最大/OR不变。v1编译失败patch/log单独保留。隔离构建Co100步thermal，compute-sanitizer initcheck error-exitcode86；将所有CSV/checkpoint/geometry输出与原版逐byte比较。latest输出bytes全部一样，initcheck0。不要使用额外过滤/屏蔽sanitizer错误的参数。

diagnostic源码patch默认不整合到PR生产目录。code snapshots/bin不上传，完整sha/版本/构建命令及可生成输入已登记。raw Nsight rep/sqlite保留在现场work/nsys-native，其大小/sha与export命令都在性能summary/spec；交付完整相关CSV tables和聚合，不靠只能本机访问的raw文件作唯一证据。

验收/发布：用results.acceptance_units的scope评价，别把构建、exit0、san0或旧历史证据换成物理PASS。普通validator独立检查文档，sha清单独立检查交付，均不嵌入求解器。Git push只放本目录和必要review索引/append responses，不force、不merge、不发PR Conversation。
