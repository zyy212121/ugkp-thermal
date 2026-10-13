# PR #14 独立只读源码审查补充

日期：2026-10-14；审查者 `/root/source_review`。本补充沿用 requesting-code-review 模板，并按 systematic-debugging 追数据来源。未编译、未运行GPU/CPU数值测试、未修改仓库工作树/index/HEAD、未写GitHub；只读了运行记录、源码、本机CUDA13.1的Thrust头文件，并对已有原版二进制运行只读objdump/cuobjdump。唯一新增写入是此仓库外报告。原 `work/static-review.md` 不重写，SHA256仍为 `06716A11DA82EF34C88850E4A7FDD5D2B174DE1EB7CFB00DB8132E78DC94EABF`。

固定HEAD=`802a11e7dfe986488a8bf8305e86424dcc94c9e1`，tree=`db2863158139842f6c8221c0da11ba640f381336`，base=`d6110c02745a8ad726208a426c4bb3a28233c1fd`。仓库根为 `/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261013T182722Z-fast`；下文file:line均相对此根。Delivery（D）为 `review/pr14/reviewer-deliveries/20261013T182722Z-fast`。本次只是Co reduction传输、P010/P011的定向补充，未扩大为全部426文件逐行审查。

## Strengths

原版失败日志、对应spec/result、诊断v1编译失败均保留。源码、传输实现、原版二进制的地址/写入宽度能相互核对；没有把NaN/Inf改判MissingStorage，没有通过清空Co/Js物理字段或添加数值门禁消除initcheck。

## Issues

### Critical

本次补充未识别新的Critical项；原报告的native chemistry trial接线问题继续有效。

### Important — SR-F1 / GPU-F005 [P2] Co reduction结果尾部7字节未生产，却随整个16字节对象D2H复制

**位置**：`applications/CHMT/gpu/Backend.cu:374`，以及`:378/383/405–409`。

**触发**：原版Ns2 native thermal，常规非空8-cell gas owner；每一microstep调用currentGasCourant的convective transform_reduce。D `logs/SAN-FINAL-CHMT-thermal-initcheck.log:21–31` 首次报告：同一源地址0x606812200、访问16bytes、cudaMemcpy source未初始化，栈经cuMemcpyDtoHAsync_v2→currentGasCourant第二lambda→advanceGasMicrostep。全部100条错误同地址、同宽度、同栈，恰对应两窗口各50microsteps。`:1228` 为 `ERROR SUMMARY: 100 errors`。已读D/results.json：该原版测试exit86/FAIL；同输入memcheck和synccheck为summary0/exit0。进程输出正常接受窗口不能覆盖此真实sanitizer失败。

**来源已定位到结果对象传输，而不是本日志中的Co/Js数值读取**：

1. `CoupledCourantValue`为 `{double value; bool unavailable;}`。当前ABI的成员为offset0的8字节double、offset8的1字节bool，整体16字节，留下7字节尾padding。Read、Reduce和init三处aggregate初始化均明确给两个成员赋值；未给尾padding生产数据。
2. 本机 `/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/thrust/system/cuda/detail/transform_reduce.h:87` 分配sizeof(T)+tmp_size，`:99–105` 将首部ret_ptr作为CUB输出，`:121` 调get_value。`get_value.h:53–56` 默认构造host结果并assign_value；`util.h:147`按 `sizeof(Type)*count`执行cudaMemcpyAsync DeviceToHost，因此拷贝的是整个16字节ret对象，涵盖不消费的padding。不是nCells数组长度误算，也不是拷贝到越界地址。
3. 原版CHMT host反汇编：0x16838c调用实际CUB TransformReduce；0x1683e6设置copy长度0x10，0x1683ee调用cudaMemcpyAsync（kind=2/DtoH）；之后0x1683fd读8字节value，0x168401仅读offset8的1字节bool。与sanitizer调用栈地址0x1683f2相符。
4. 同一原版sm89二进制的CUB `DeviceReduceSingleTileKernel<...CoupledCourantValue,...counting_iterator<int>,...CoupledCourantRead>` 普通非空输出分支，SASS 0x6fa0为 `STG.E.64 [R6.64], R4`，0x6fb0为 `STG.E.U8 [R6.64+0x8], R9`，0x6fc0退出；实际仅写9字节。空输入/其它输出分支可有完整两个64位store，不取消本fixture非空路径的7字节缺口。
5. Co producer先于reduction：`common/GpuGasCourant.cuh:130–144` 在同stream启动convective-by-cell，`:177–178`再做两次max；`common/operators/computeSstFaceFluxKernel.cuh:703–743` 每个正常范围内cell末尾写gasFluxPositivityScale[c]。当前普通静态mesh输入没有nativeRefinement早退。日志也没有8字节Co load或4字节status load的device-kernel uninitialized警告，只有整个结果的Host-API D2H。此fixture输入没有wallModel覆写，CHMT `core/Configuration.H:44–48`默认LowRe，`SharedGasDeviceStorage.cuh:360–368`据此关闭boundary-layer，所以不能将这项归因于其Js数组的缺失消费。

**影响与非缺陷边界**：真实初始化检查失败，阻碍完成已要求的sanitizer验收；当前证据没有表明double最大值或bool unavailable本身未初始化，没有证明Co/Js污染、错误dt、物理数值回归或MissingStorage。合法sizeof对象的字节复制包含padding不等于C++数值运算UB，也不等于越界；应把问题限制为辅助reduction结果的未初始化字节传输。不能把原版结果重标PASS、禁用initcheck或清空物理字段掩盖它。

**基线范围**：d611 Backend没有currentGasCourant或CoupledCourantValue；git blame将当前结构/Read/Reduce定位到8989d628引入。该缺口属于本PR新增的coupled Co结果表示，不是上一报告中基线已有的cold zero-effusivity问题。

**最小修正与复核建议**：在原producer返回类型里为这7字节提供实际初始化成员；保持double、bool、max/OR逻辑及相同遍历，不新增field扫描、scalar门禁或数值条件。当前P011 v2的单行方案为：

```cpp
struct CoupledCourantValue {
    double value;
    bool unavailable;
    unsigned char initializedTail[7]{};
};
```

在current C++17/CUDA ABI下保留value offset0、bool offset8、sizeof16/align8；Read/Reduce/init的两项aggregate initializer会应用第三项default member initializer{}。类型仍standard-layout和trivially-copyable，适合Thrust/CUB对象复制；DMI使默认构造未必trivial，严格意义不能未经说明称它仍为POD。host get_value默认构造不初始化value/bool，但完整D2H随后覆盖它们，关键是device输出16字节必须全部已生产。检查优化后SASS输出宽度、再用完全相同100step initcheck红/绿与原版CSV/checkpoint全输出bytes比较确认。额外覆盖合法empty/single/multiblock输入仅检查该表示/归约边界；不要故意超出backing。无需改vendor Thrust或追加每步scratch memset。

**原始pin**：Backend源码SHA256=`fdec130622e33e1ed3aaea7b7d434e327fe2c17ab363e042214f7b6bf3183bfe`。独立实际读取原bin SHA256=`7124cf69b1cd51da033c1360d67129748803a5c66767489f2864c169b85c845d`（13910824bytes），与原initcheck spec一致。原initcheck log SHA256=`33fc6cf9aff4df49215dabeb1976b66603db35b98ce4dfdcaf9e42ebd3d77053`。本机Thrust transform_reduce.h SHA256=`7847cd9aedb9199c58248ce36486b48a1add91f21fc27889a1280f5ebff08acf`，get_value.h=`b3b6f24a87384889d726f5fa5c39af41117862e5f312984d3409616c12086b37`，util.h=`4ab9020640b3a8fb1ac7ffb89a028eeebbfe5b87aeb4d455bd7586405a8ccb69`；反汇编实际namespace为THRUST/CUB_300104_SM_890，与所读实现一致。

### Minor — P011记录应区分v1编译失败、v2表示与待完成验收

已读 `P011v1-u64-diagnostic-build-failure.patch/json`、`BUILD-CHMT2-P011.log`；v1将bool改成std::uint64_t，实际build exit1，日志337行CUDA13.1 crt/sm_80_rt.h:108:246报expected unqualified-id。这是隔离补丁编译失败，不能归给原PR，也尚无证据把原因归给uint64类型表示本身。

已读现有v2 `P011-Co-reduction-result-no-padding.patch/json`，只增加initializedTail[7]{}；patch SHA256=`e2e08d7f52b55c226fae4e451e2ef648acbf6d18ad7fe5fa5ec0d2a16fdc6a9b`。本次结果snapshot只有v1完成的build记录，v2 build日志在写入，不能仅凭未见error或JSON的scope文字宣称v2 build/运行/全字节比较已经通过。v2 JSON reason仍沿用“explicit8byte 0/1 slot”，应和真实bool+7显式tail表示对应；scope里的initcheck0/bytes equality必须按预期或实际结果清楚标记。

## P010冷壁初始化carry：新增证据与剩余边界

已读 `P010-cold-profile-initialization-carry.patch/json`。它在 `common/operators/clearColdWallParticleState.cuh` 的**initialiseColdWallParticleState**（并非清零clear函数）完成profile cast后，将 `double(nodeEnthalpy[0])-double(storedNode[0])`加到既有double carry；2D在 `initialiseColdWall2DParticleState`执行同样修正。`GpuColdWallSolidification.H:292–308` 的1D构造将同一initialEnthalpy赋给全部8节点，2D `GpuColdWall2DDevice.cuh:81–95`也均匀初始化64节点，因此“单节点的cast损失”恰是本uniform初始化的mean损失。patch保留原cast/热焓公式，没有重新拒绝数值、修改clamp或新增收敛门禁；需要P007补充uncoupled effusivity。

已读真实result记录与三份FIELDS3日志、checker来源。`inputs/resident_check_fp64_mean.py:25–26`先把float节点升为float64，再求mean加double carry、独立caloric求温度，原阈值abs1e-9 K没有放宽。checker SHA256=`8c4d12003e26bb5a44682a0c10ffd7bc730b449596273cb210f4938a83ca600a`。实际1D十时刻误差均0；2D十时刻最大4.547473508864641e-13 K，全部真实粒子物理字段有限；1D WET100的十个written states直到1e-4s误差均0，已越过先前5.848388e-5s的ring0 firstWetAge。D/results.json中的cold2D P010四种sanitizer和restart运行/compare均exit0/PASS。

该证据强于旧短1D未触发wallG控制，但范围仍是这些单粒子CHT案例。checker的mean对整个node数组取值、carry[0]/pT[0]对照，不能直接推广为多粒子逐一caloric验证；只有当前one-particle输入时该比较恰当。未独立证明全wall/gas ledger物理守恒。P010源patch由GPU_PAIRED_PARTICLE_HEAT宏守护且1D文件被FSH共享，正式开发者仍需核准其它app的carry/backing/precision及activation/release/reinitialisation调用边界；本补充不把已批准的CHT cold float存储扩展为其它app的精度或公式豁免。

P010 patch SHA256=`f98d42c5fce4d4b8f2cc842af42f9ccbc9ecc62f6728b100ca3ac74f215ed6db`，metadata SHA256=`cd3d98bb53fa6b2d3755e9bec1473cf9992396aa5cff34abcdf61cd9c0d277cf`。此处是隔离patch与实际运行证据补充，原固定生产源不包含P007/P010正式修复。

## 性能观察补充

主代理新报告原native Nsight trace：PR3000kernels vsbase2200，malloc339 vs139，D2H7500→5700且总hostMemcpy也减少。此处是主代理陈述，本席位尚未独立解析完整trace/clock/counting范围。它说明不能用此前Ns2化学成功kernel计时或某类D2H减少，宣称全native全过程工作均减少。Co的额外计算及Thrust temporary结果会增加部分工作；具体计数与总时长需按同输入/完整trace解释，不从源码调用数推导提速倍率。原报告限定的化学算子成功prefix成本方向和CHMT低位FP64差待解释仍适用。

## Recommendations

保留原initcheck100/exit86和v1编译失败；v2需实测红/绿、原版完整输出bytes及实际SASS写入完整结果。P010现有冷壁证据按单粒子CHT、uniform初始化与真实wet/restart范围陈述；继续披露全ledger、多粒子组和共享app尚未关闭的范围。所有修正仍为诊断交付文件，原生产库只读。

## Declined to judge

未运行GPU、编译或重新执行checker；未证明所有Co/Js消费者、所有Thrust版本/架构、全部426文件、整个solver性能、P011v2已通过或P010正式完整修复。未将这次padding传输错误判为物理NaN/MissingStorage，不建议用数值门禁解决。没有重新审查README、合并、推送或GitHub评论。

## Assessment

**Ready to merge? No.** 原报告的未关闭项继续有效，另有真实Co辅助结果未初始化字节传输。该新项已有源码、vendor实现、原二进制SASS/host copy及initcheck相互支持；P011v2修正方向合理，但需完成实际反证与原输出不变验证。P010新增证据关闭了这些单粒子CHT初始化一致性/wet/restart控制，并未关闭全部物理ledger和共享app范围。本席位的只读补充已完成，不执行合并。
