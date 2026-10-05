# CHMT 未验证测试包

本目录按要求新增在仓库根目录 `test/chmt/`，与已有 `examples/`、`tests/` 平级。仅新增测试资料，不改动旧应用、算子、构建脚本或既有测试。

**所有 CHMT 算例均为 `UNVERIFIED`。当前没有 CHMT 求解器、适配器、求解器运行结果或“成功数据”。** `references.csv` 是解析参考值，不是求解器输出。CPU 自测通过只说明参考生成与比较工具的检查通过。

## 文件

- `cases.json`：12 组冻结输入/待补数据规范，统一标记 `pendingCHMTadapter`
- `run_case.py`：本地 prepare/run 入口；缺真实 CHMT 或适配器时明确拒绝，绝不回退 mock
- `reference.py`：Python 3 标准库解析参考生成器，不推进任何数值求解器
- `references.csv`：确定性解析值，含三个网格级别、初始/终止时刻；每行标记 `ANALYTIC_REFERENCE`
- `compare.py`：字段误差、全局守恒、细化阶数、逐 stage GCL 的独立检查入口
- `criteria.md`：公式、输出契约、预先冻结的容差和判据适用范围
- `external_data.md`：Freno/TACOT 原始来源、版本差异与阻塞项
- `test_tools.py`：CPU 工具自测，含错误输入和故意扰动反例
- `manifest.json`：状态、来源、验证范围与文件 SHA-256

## 可直接准备的输入

Stefan 一维熔化；膜剪切/压差/联合驱动剖面；零剪切塞状流焓平移；有剪切的焓输运与耗散升温；定容膜压力脉冲；蒸发界面携能局部契约；ALE 自由流和光滑密度波；一维守恒重分区映射。

这些是数学验证配置，不是生产材料选择。Stefan 必须使用法向温度可分辨的模型，不能套用厚度均温液膜后宣称验证同一问题。Freno、TACOT1、TACOT2 尚缺完整一致的输入/参考，故没有生成假数据。

## 现在可运行

在仓库根目录：

```sh
python3 -m unittest discover -s test/chmt -p 'test_tools.py' -v
python3 test/chmt/reference.py --output /tmp/chmt-reference.csv
cmp test/chmt/references.csv /tmp/chmt-reference.csv
```

准备单项配置与参考（不运行求解器）：

```sh
python3 test/chmt/run_case.py prepare --case film_shear --output /tmp/chmt-film-shear
```

该命令生成 case-spec.json、reference.csv、request.json，不生成求解结果。

生成单项参考：

```sh
python3 test/chmt/reference.py --case film_shear --output /tmp/film-shear-reference.csv
```

## 接入真实 CHMT 后

实现读取此规范的独立适配器，并输出同一 CSV 键和观测量。不要重命名解析文件来伪装求解器结果。`cases.json` 是中性规范，不是当前可运行的 OpenFOAM case。

```sh
python3 test/chmt/run_case.py run --case film_shear --output /tmp/chmt-film-shear --adapter /path/to/chmt-adapter --solver /path/to/CHMT
python3 test/chmt/compare.py fields --case film_shear --observations /path/to/observations.csv --run /path/to/run.json
python3 test/chmt/compare.py convergence --input /path/to/convergence.csv
python3 test/chmt/compare.py gcl --input /path/to/stage-gcl.csv --steps 1000 --stages 2 --cells 1024
```

上述路径是待接入的真实运行文件，并非本包已有结果。GCL 的 stage 数应填写实际方法值。退出码：0=所请求数值检查匹配，1=不匹配，2=缺失/非法输入。即使返回 0，报告仍保持 `solver_validation_status=UNVERIFIED`，不得自动改写清单。完整验证需补齐实际 CUDA 运行、溯源、时空细化和该模型全部判据。

运行入口会调用适配器并自动做字段/守恒检查，适用时生成空间收敛 CSV 和报告；ALE 的完整 stage GCL 与独立时间细化仍须检查。适配器接口、需要补齐的求解器观测和 run 元数据见 `criteria.md`。不得根据看到的结果放宽容差；如算法阶数或精度改变，应在下一次运行前版本化修改判据并保留旧结果。
