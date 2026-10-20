PR22 热流审查自包含复现包

快速开始（CPU，无需 OpenFOAM 或 GPU）
在 Linux/WSL 下安装 Python 3.9+、NumPy、g++（支持 C++17）、tar/xz。解压后进入包目录：

    python3 reproduce.py

入口从包内文件编译两个 C++ 程序，运行 Ny80、Ny160、UGKP t=0.02305 和最新 t=0.0233 的冻结预算，并核对历史结果。全部输出写入 results/；可用 --output 指定其他目录。不调用 CUDA，不需要 OpenFOAM 库，也不访问历史机器路径。audit/original 和 logs 内的绝对路径只是历史记录，入口不使用这些路径。
用 python3 verify_files.py 核对包内文件清单及 SHA256。results/ 是重新计算输出，不属于静态清单。压缩包本身的 SHA256 另在下载附件和 PR22 中给出。

版本与目录
source/ugkp：计算源码 46dca92cadff3958fdb9956e9b9043d6590bf91b 的 common、applications、tool 和相关测试/fixture，未修改生产源码。
source/matched-of10：产生 Ny80/1282、约 -166.45606 W/m² 的完整参考求解器及 Make/files、Make/options。
source/official-rhoSimpleFoam：本机官方 OpenFOAM 10 基线对应文件；source/matched-vs-official.patch 为可应用的统一补丁。
source/of10-dependencies：实际有效应力、SST 和官方壁函数相关源码，便于审查引用实现。
官方包版本 openfoam10=20230119；原运行日志 Build=10-c4cf895ad8fa。matched 求解器当时是独立目录，没有自己的 Git 提交，不能编造提交号；它由 provenance/versions.json 的逐文件哈希及 matched_source_content_sha256 唯一标识。已与原 build4-command.json 中8个源文件哈希和实际运行二进制哈希逐项相符。
cases/of10-ny80：网格、完整 constant/system、0、原重启722、参考1282及 matched-wall-last.csv。
cases/of10-ny160：对应0、原重启1613、参考2887。
cases/ugkp-wallFunction：完整设置、0、t=0.02305 冻结场、50 μs场、最新累计250 μs终场（t=0.0233）。
data：原热预算使用的原始二进制数组，保持原字节；Ny160数组亦在包内。最新250 μs数组由入口从包内完整场重新导出。
provenance/fields.json：各时刻来源、网格标识和字段哈希；SHA256SUMS：所有静态文件哈希。相同内容以硬链接去重，归档不依赖外部符号链接；不要把硬链接解析成包外路径。
audit/original：原审计、提取和辅助程序；audit/thermal-budget.cpp 是新增详细输出的审查驱动，生产计算头不改。original 版本保留一次，可逐项比较新增输出。
logs：原构建、参考收敛、UGKP续算、已有网格对照记录。audit/historical-results 保留旧原始预算输出；validation/development-run保存组装阶段复算记录。最终压缩包的实际空目录检查记录作为同一Release的clean-directory-verification.json附件提供，其中包含所验证压缩包的SHA256、命令、退出码与输出，避免包内记录自引用其自身哈希。

热物性、边界与模型
两案例均为平板、静止壁面、无吹出（speciesFlux=(0 0)），壁温290.53216418474562 K，壁速(0,0,0)。
入口/顶部固定静温275.90898782976797 K、Ux=179.81233357102013 m/s、p=124094.11053558919 Pa；这是静温条件，不是totalTemperature边界。由cp常数计算的总温见parameters.json（T0=T+|U|²/(2Cp)），只作派生量。
Cp=1004.675 J/(kg K)，原UGKP冻结预算使用R=287.0500676、Cv=717.6249324；UGKP由精确SI常数和摩尔质量得到同一R。实际OpenFOAM10链接库探针给出RR=8314.4700665054497 J/(kmol K)、R=287.05025714824882、Cv=717.62474285175108，两者R约相差6.60e-7（相对值）。保留这个原有差异，没有为使报告吻合而修改数组。实际各框架R和来源详见parameters.json、provenance/of10-runtime-constants.json及完整字典。mu=1.7244421537657099e-5 Pa s，Pr=0.72，Prt=0.9，laminar conductivity=mu*Cp/Pr。
UGKP使用实际 gasModelProperties 的linearCp表：Air/CO2，摩尔质量分别0.028965206821645145/0.0440095 kg/mol，Cp分别1004.675/843.68062566560013，温度范围40–2000 K；形成焓/积分常数为零。不是另造NASA表。固定场为纯Air，保留CO2全部字段、组分扩散表及自定义文件。
OpenFOAM使用 hePsiThermo / pureMixture / const / hConst / perfectGas / sensibleInternalEnergy，Hf=0。
SST与壁函数系数、原始入口k/omega、Spalding、omegaWallFunction和alphat参数均保留在各案例0/constant和原日志；parameters.json提供索引与数值。原alphat类型仍为官方Jayatilleke，实际能量组装时自定义EEqn会覆盖其壁面有效导热系数并加入显式修正，不能只读0/alphat就推断实际热流。

能量变量与实现（审查重点）
matched OF10实际求解e，显内能。hConst字典未指定Tref/Hsref，默认Tref=Tstd=298.15 K、Hsref=0；因此h_s=Cp*(T-298.15)，e_OF=(Cp-R_OF)*T-Cp*298.15。能量偏置为-299543.85125 J/kg。组合输运的是e_OF+|U|²/2，不包含湍动能k。默认值及EOS实现源码已附，不能把Hf=0等同于e_OF=Cv*T。
EEqn左端：
    div(phi,e) + div(phi, |U|²/2 + p/rho)
    + thermophysicalTransport->divq(e)
    + div(turbulence->devTau() & U)
右端为fvModels.source(rho,e)，并在矩阵source中加入matchedHeatSource。
本例没有配置体积质量/动量/总能量源。
官方linearViscousStress的devTau=-rho*nuEff*dev(2*symm(grad U))，不含2/3*rho*k各向同性应力。其负号使上述左端散度等效于右端正的有效黏性应力功散度。离散为Gauss linear的cell-stress×cell-U插值，见fvSchemes。
EEqn没有额外显式rho*epsilon、rho*betaStar*k*omega或独立tau:grad(U)源；SST的产生、耗散、压缩项留在k方程。这里记录实际实现，不宣称这种能量闭合已完成物理正确性验收。
UGKP rhoE为内能（通用组分形成能在内）加平均速度动能，不含rho*k。本例linearCp的e0=0，所以e_UGKP=(Cp-R_UGKP)*T。原导入OF冻结场的程序没有直接复制OF的e，而是从p/T/U重建rho=p/(R_UGKP*T)、rhoE=rho*(Cv_UGKP*T+|U|²/2)，已经转换到UGKP能量零点，同时保留上述微小EOS常数差。原OF rho/T/p等字段仍完整保存。黏性功使用生产面牵引乘重构面速度；本次审查“OF式cell-stress×cell-U”只替换内部面功项，不修改生产代码或壁面热律。

自定义壁面热处理没有独立动态库；完整源码就是matched-of10/EEqn.H中的匹配热律块。
正向定义：qIntoGas>0表示壁面对气体加热。qIntoGas=Keff*(Tw-Towner)*deltaCoeffs - C/Tplus。
第一项是导热；第二项是负的Jayatilleke黏性加热修正（按此符号定义）。OF把-C/Tplus*面面积加入owner能量矩阵source，导热由正的壁面alphaEff承担。UGKP将两项合为同一边界能量通量。
原matched-wall-last.csv是在最后一次SIMPLE迭代的能量求解之前写出，保存了实际组装输入：
    2607.7388458877422 - 2774.1949056580188 = -166.45605977027662 W/m²。
由写出的1282终场按原UGKP冻结导入物性重新计算则约为：
    2607.740269319 - 2774.196419689 = -166.456150370 W/m²。
约9.06e-5 W/m²差涉及迭代内取样时序/最终字段以及上述R/壁密度转换，不能全部归因于单一因素。包同时保留实际OF组装输入和UGKP导入数组，不把它们混作逐位一致的同一输入。

能量预算和输出
results/*-two-laws.csv：从原OF组装输入重新执行matched公式与生产UGKP热律，逐面比较336面，不只是比较四舍五入的净热流。
results/*-wall-decomposition.csv：四组冻结场全部壁面的状态、物性、导热、修正、净热流。summary.json汇总五站。
results/native-case-stations.csv：直接读取完整原生OF/UGKP冻结场的五站状态，分列原生rho与预算导入rho，并使用各自实际R/Cv和能量零点重建内能。
results/{before,after}-*/detailed-faces.csv：五站首层owner的所有6个面（含empty面），面号、邻接、位置、面积及质量、三分量动量、总能量通量。这里是面积积分通量，按网格owner外法向为正，单位kg/s、N、W。
HLLC能量精确分为同一Riemann扇的advected total energy与p_star*u_star*area压力功；动量也列出压力分量。其和与生产无黏能量一致，再加黏性功及导热/壁面修正恢复总能量。入口自动核对每个首层单元全部6面的闭合。
results/*/thermal-faces.csv是原审计格式，按单位面积列出通量，不要与detailed-faces的积分量混用。
results/*/detailed-cells.csv列出状态、体积、守恒量变化率及本例为零的体积源；macro-cell-budget.csv另列SST产生/耗散/压缩等源项。
温度变化率由实际冻结预算按纯Air、恒Cv、组成不变恢复：
    Rrho=-sum(sign*Fmass)/V
    Rm=-sum(sign*Fmomentum)/V
    RE=-sum(sign*Fenergy)/V
    Tdot=[RE - U·Rm + (|U|²/2 - Cv*T)*Rrho]/(rho*Cv)
其中sign对网格owner为+1，neighbour为-1；若有体积源则需先加进对应R，本例为0。
OF式功诊断仅替换RE中的功散度：TdotCellStressWork=(thermal-workReconstructed+workCellStress)/(rho*Cv)。
冻结Ny80场、当前面系数应复现Tdot=-4876.1416473213521 K/s；替换功诊断=-175.69674941592805 K/s。它们都是UGKP算子在OF场上的瞬时预算，不是OF自己的有限体积能量残差，也不是时间推进后的温度变化。
Ny80/Ny160首层物理位置不同，原网格对照不等同于同点网格收敛；保留这一限制。

OpenFOAM构建和可选重新迭代（不属于CPU审查入口）
    bash scripts/build-of10.sh
    bash scripts/run-of10.sh 80
    bash scripts/run-of10.sh 160
默认从/opt/openfoam10/etc/bashrc加载，其他安装可设置OPENFOAM_BASHRC。
这些脚本使用runtime/中的独立程序与案例副本，原包内参考场保持不变。原重启为722/1613，不从已经收敛的1282/2887继续来冒充复现。硬件/编译器/舍入差异可能改变SIMPLE终止迭代号。

GPU构建与50 μs短算（需要OpenFOAM10、CUDA13.1及适配GPU）
    bash scripts/build-ugkp.sh
    bash scripts/run-ugkp-short.sh
可设置CUDA_HOME和UGKWP_CUDA_ARCH。默认Ns=2、sm_89、fmad=true，输出隔离到runtime/，从包内t=0.02305开始至0.0231。本次打包不新增GPU推进；此前250 μs运行日志和源码/二进制哈希已附。

本包的任务是复现与审查，未继续修正热流或宣称工程验收通过。PR22保持草稿。
