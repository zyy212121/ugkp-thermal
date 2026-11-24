# 大温差壁面函数

在 CHT 或 gasUGKP 的 SST 设置中选择 `wallTreatment wallFunction;`。
配置位置是 `constant/fluid/momentumTransport`（CHT）或气体的 `momentumTransport`，
键位于 `RAS/kOmegaSSTCoeffs`。`compressibleWallFunction` 作为已有案例的同义配置。

模型从贴壁单元和壁面条件取得切向相对速度、温度和物性。法向距离取面法向梯度系数的倒数。
它假设局部平衡、壁层法向压力近似恒定，并在每个壁面冻结分子动力黏度、比热和导热率。
指定壁温时计算进入壁面的热流。绝热壁用同一方程反求恢复壁温。CHMT 的反应壁层有独立的配置入口。

令 \(z=u/u_\tau\)，\(G(z)\) 为 Spalding 的 \(y^+\) 函数。
热子层的交界速度 \(z_i\) 由 \(G(z_i)=y_T^+\) 给出，其中 \(y_T^+\) 沿用 Jayatilleke 热子层判据。
如果贴壁单元位于子层内，积分止于贴壁单元。
子层导热率为 \(\lambda=\mu c_p/\Pr\)，按物理距离积分热阻，并按同一个 Spalding 剖面积分剪切功。
外层以 Jayatilleke 对数热阻定义热扩散。热扩散黏度为
\(\mu_{t,h}=\mu\sigma(\kappa y^+-\Pr_t/\Pr)\)，其中 \(\sigma\) 是同一壁层温度剖面的速度加权密度变换系数。
Jayatilleke 交点取上根，因此外层湍流贡献非负。分子导热与湍流导热组成
\(c_p(\mu/\Pr+\mu_{t,h}/\Pr_t)\)。
冻结 \(\sigma\) 后对外层热阻解析积分。外层恢复项将该热阻按速度增量分配，温度关系和剪切功使用同一份热阻。
定温壁以变换后的速度为求根变量，同时满足摩擦速度和热侧密度变换关系。壁层无内部节点。

常密度且剪切加热趋零时，动量恢复 Spalding，外层热阻与分子段相加恢复 Jayatilleke 对数热阻。

近壁气体热流 \(q_g=(T_P-T_w+\Delta T_{\rm rec})/R_T\) 直接进入宏观能量通量。
