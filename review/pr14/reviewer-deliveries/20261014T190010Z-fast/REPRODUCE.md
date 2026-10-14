# PR14_REVIEW_ONLY / FAST4 复现与证据

固定生产 `47034e4e0380fa5b84165b49207a95a9b187a52a` / tree `c932823a00b3ed2f647e22de5002f9b913962fee`，记录父 `55d3663830ada63077241c528e2c81b53976e2c4`。SOURCE.json记录最初工作树和发布前状态，推送后PUBLICATION.json以独立提交记录实际已验证的主证据提交；不把SHA写入自身。

完整命令是commands.txt和results.json tests[].argv/UTC/cwd/spec，计划不是已执行结果。正式native build原flags如versions.json；输入/source/object/binary哈希在spec和metadata。所有automatic_time_limit=null。工具环境logs/META-ENVIRONMENT.log、inputs/env.sh及initial版本保留。先阅读REPORT/coverage/每个acceptance unit，不能一键将所有生成器当成生产任务。

52个大日志/数据lossless gzip：metadata/lossless-storage.json记录compressed和decompressed SHA/原alias，全部解压后SHA已验证；原始未压缩字节保存在同授权test区work/fast4-raw-cache。results[].log指向实际压缩payload，raw_log_alias/raw_log_sha256保持原记录。原raw日志未加前缀或篡改。已上传同字节旧输入/数据只以metadata/immutable-references.json固定commit/path/SHA引用，无重复上传。

需要离线分析时，在独立授权test输出目录物化，例如`python3 inputs/materialize_evidence.py /home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T190010Z-fast/work/materialized-fast4`，取回固定Git对象并校验解压byte SHA。不要回填旧owned目录。若移动到其他机器，先配置已授权test根；不要指向正式源码、安装、结果或原checkpoint。

inputs/r4_paths.json与drivers保存执行时的绝对路径；重跑须在新的隔离test副本更新这些配置，遵循原输入/端点/flags/阈值。历史脚本仅记录编排，不要在现有delivery重跑覆盖原证据。inputs/r4_stage_audit-initial.py、r4_physical_audit-initial.py和S407版本与原spec哈希完全匹配；S408patch留最终geometry选择修改。S401/S403/S406与D401/D402patch/源单列；S405初编排异常stdout未完整捕获这一限制明确保留。

可执行binary/object不上传，exactSHA/native构建命令、最终源/patch足够复编，所有本轮binary仍在授权test安装/cache。NativeGPU原版、diagnostic/support、host、NOT_RUN分别解释。Nsightrawtrace/tables用于资源计数，非profiling样本独立；新观察器D2H/printf不能用于效率。不得通过恢复productionaudit、改输入/终点/精度/容差或新停算门禁得到通过。

最终formalfixture/strictrestart/旧API测试红保留。14原gap都没有因subsetPASS自动关闭。完整三应用gross/当前selectedowners/混合非零speciesV1/更多实际资源错误仍需开发修复或下一次独立定向证据，当前不merge-ready。
