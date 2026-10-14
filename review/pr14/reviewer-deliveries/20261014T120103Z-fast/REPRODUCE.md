# PR14_REVIEW_ONLY FAST3 复现与证据阅读

固定生产source/tree与recordparent见SOURCE.json，实际命令/UTC/退出/预算见results tests[]和commands.txt，实际source/dependency/input/binary/object SHA见metadata/source-hashes.json、各spec和versions.json。所有输入在已授权testcheckout构造；正式安装/原始断点不改。

新命令与路径是执行实录，不是无条件重跑清单。先按coverage/evidence-reuse检查未变链，已有valid结果引用固定旧证据。原wet/evap输入哈希逐文件same-input元数据给出；evap/end.001和Ns10/end2e-5不缩短。原旧noMelting OOB输入不得为计数重新运行旧4b。old4b只用于已实际安全完成的有效selectedphase0对照。

本轮来源原生二进制/对象保留在SOURCE指定test区；仓库不提交巨大binary，哈希/精确构建argv和freshformalfixture生成源/command/build/run均交付。Ns2/10host链接时复用已核448region的原CUDA对象；从新机器复现可按固定source原native build重编译全CUDA，必须另记新的binary/hash，不能冒称本机原产物。

S308复现用inputs/phase/onecell-*-S308/constant中的实际polyMesh；它包含unitcube校正后的消费点。原生成dict继承模板坐标，若重新blockMesh必须先按actualpoints重建一致坐标，不能直接生成并覆盖实际消费mesh。它是新独立控制，不是对原wet/evap改物理换绿。S306权威ASCII与变更字段在metadata/active16-*-input-provenance.json，禁止复制已wall-bound并置零stuck/velocity冒充入射初值。

S301/S302、S303/S307、D302、S305、D303、S309/S311/S312只在隔离测试/观察源中使用，各scope/原red保留。不要将支持或诊断源推成生产修复。所有保存脚本在inputs/r3_*，按其原运行种类/预算读日志，不自动执行整套驱动；S313只有compile、GPU重复取消。异步host/device所有权、显式同步与正常析构的证明不包括全allocation/free/sync错误恢复。

metadata/immutable-references.json列出内容相同的历史alias。直接git show fixed_commit:repo_path并校验SHA256阅读，或materialize_immutable.py将它们提取到指定独立test目录；不要hydrate旧review目录破坏owned_paths/历史记录。新日志单存、跨topic复用同EID。

性能直接引用原R2Nsight和所有非profiling样本；本轮observer IO/copy/sync、sanitizer耗时不能用于效率结论。最终文档行政验证日志/SHA256SUMS校验本目录；完整FAST及merge状态按REPORT/coverage明确开放范围判定。
