from pathlib import Path
import json,csv,hashlib,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/mnt/d/tempt/wallfunction-vs-boundarylayer-20261019')
S=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018')
A=Path('/home/lss/ARC3696-runs/20261017T042008+0800/M0p55/baseline-rh1au35z')
B=Path('/home/lss/ARC3696-runs/pr19-f9e8d775-20261019T185625+0800/M0p55/baseline-lae7a4r6')
rows=list(csv.DictReader((O/'same-input-wall-results.csv').open()));base=[r for r in rows if r['dataset']=='boundaryLayer-old'];cur=[r for r in rows if r['dataset']=='wallFunction-current'];r=base[2];q=cur[2]
check=json.loads((O/'legacy-wallfunction-comparison.json').read_text());metrics=list(csv.DictReader((O/'field-differences.csv').open()))
profiles=list(csv.DictReader((O/'five-station-profiles.csv').open()));inside=list(csv.DictReader((O/'baseline-internal-profiles.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for ax,(key,label,unit)in zip(axes.flat,[('U','Ux','m/s'),('T','Temperature','K'),('k','k','m2/s2'),('nut','nut','m2/s')]):
 for version,color in [('boundaryLayer-old','#2865ad'),('wallFunction-current','#cc4325')]:
  p=sorted([t for t in profiles if abs(float(t['x_station'])-.58)<1e-6 and t['version']==version and float(t['y'])<=.006],key=lambda t:float(t['y']))
  ax.plot([float(t[key])for t in p],[1000*float(t['y'])for t in p],'-o',ms=3,color=color,label=version+' cells')
 if key!='nut':
  p=sorted([t for t in inside if abs(float(t['x_station'])-.58)<1e-6],key=lambda t:float(t['y']))
  ax.plot([float(t['Ux'if key=='U'else key])for t in p],[1000*float(t['y'])for t in p],'--',lw=1.3,color='#2865ad',label='boundaryLayer auxiliary profile')
 ax.set_xlabel(label+' ['+unit+']');ax.set_ylabel('y [mm]');ax.grid(alpha=.25);ax.legend(fontsize=7)
 if key=='nut':ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3));ax.tick_params(axis='x',labelsize=9)
fig.suptitle('Same physical time t=0.018 s, x=0.58087 m; original cases read only')
fig.savefig(O/'station-058-profiles.png',dpi=160);plt.close(fig)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()=='f9e8d77559c9d5f0eb884755203c22d2606aae7e'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=S,text=True).strip()
sources=['common/gasNumerics/OpenFoamWallFunctions.cuh','common/operators/gasFaceSubgridTransportProperties.cuh','common/operators/computeGasPrimitiveGradientsKernel.cuh','common/operators/computeSstFaceFluxKernel.cuh','common/gasTransport/GasBoundaryLayerEvaluation.H','common/gasTransport/GasBoundaryLayerModelState.H','common/gasWall/ReactingWallLayer.H','common/gasWall/WallGeometryBuilder.H']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={'source_sha':'f9e8d77559c9d5f0eb884755203c22d2606aae7e','source_clean':True,'current_case':str(B),'current_mode':'wallFunction','reference_case':str(A),'reference_mode':'boundaryLayer/reactingSst','important_limitation':'The previously accepted f9 boundaryLayer results were removed by the user; current local independent numerical reference is the protected old baseline. The same path was reused for wallFunction at log start 22:12:33 on 2026-10-19. Prior launch metadata alone does not prove selected wall model.','source_hashes':{n:h(S/n)for n in sources},'artifacts':{n:h(B.parents[1]/'bin'/n)for n in ['gasUGKP','gasUGKPCudaBackend']},'input_manifest':'input-comparison.csv','original_cases_read_only':True,'solver_rerun':False,'source_modified':False}
(O/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
report='''壁函数与壁层 baseline 差异诊断，2026-10-19

结论
当前已经证明存在足以产生这类差异的近壁模型及粗网格耦合不等价。wallTreatment 的选择同时改变壁面剪切、热流、首层 omega 和 k 生产项，并非只改变求解器的执行效率。
在同一份冻结的粗网格场上、不推进任何物理时间，调用真实生产算子已得到约 1.8 倍壁剪切和 2.1 倍首层生产项差异。它们能解释壁函数结果速度降低、k/nut 增大的方向和显著幅度。
针对当前输入的检查未发现可以将大差异归因于 Spalding 求根失败、omega 分支错用或本轮热物性优化的证据。但没有完成两个版本 wallFunction 的端到端长算对照，也没有外部物理解验证，不能据此宣布全部实现无错或任一模型必然正确。

一、运行来源与比较对象
源码工作区：/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018
SHA：f9e8d77559c9d5f0eb884755203c22d2606aae7e，开始及结束核验工作树干净。
当前运行：/home/lss/ARC3696-runs/pr19-f9e8d775-20261019T185625+0800/M0p55/baseline-lae7a4r6
本地日志实际开始：2026-10-19 22:12:33，运行至0.018 s正常End；明确打印wallTreatment=wallFunction。
当前目录中的launch元数据沿用了旧启动器。因此本次模型选择以fluidProperties和实际日志交叉核对，不能仅看pr_head或目录名baseline。
冻结gasUGKP/gasUGKPCudaBackend的SHA256与f9报告清单相同，见provenance.json。
对照：保留的旧baseline-rh1au35z，wallTreatment=boundaryLayer，reactingSst，24节点，rtol=1e-8，atol=1e-10。
先前通过验收的f9 boundaryLayer场已由用户删除；当前对照不能伪称该已删除数据。用户给出的统计与本次读回旧baseline的结果接近，但并非全部逐字相同。
31个实际初始输入中仅fluidProperties的壁面模型块和controlDict的endTime不同。网格、初场、黏度、Pr、组分物性、时间步控制、Tw、speciesFlux均相同；speciesFlux=(0 0)。本轮全部原案例只读，没有重跑长算、改参数、改源码或推送。

二、差异位置与量级
t=0.018 s，U单元等权RMS=5.53559 m/s，最大21.5180 m/s；体积加权RMS=2.74803 m/s。两种RMS定义分别保留，体积加权较小不能消除近壁误差。
U最大差在首层cell81，x=0.314115 m，y=0.127389 mm；T最大差2.95041 K也在首层。近壁y<2 mm的U RMS约15.0191 m/s。
k最大绝对差225.410，位于前缘附近首层cell2；nut最大差0.00235421 m²/s，位于x=0.863958 m、y=4.60399 mm。扰动已经传播到壁面以外，不能只当成输出边界值不同。
在x=0.580872 m首层：U=123.481→105.118 m/s，T=284.740→286.731 K，k=59.1271→137.035，omega=476546→409205 s^-1，nut=3.69971e-5→9.99317e-5 m²/s。
箭头从旧boundaryLayer指向当前wallFunction。完整0.0005/0.004/0.018 s统计在field-differences.csv；五站剖面在five-station-profiles.csv和station-058-profiles.png。

三、确切不同的模型路径
1. 采样高度和未知量约束不同。
wallFunction使用首层owner的U/T/k、几何壁距0.127389 mm。
boundaryLayer使用WallGeometryBuilder选出的内侧donor，匹配高度0.386216 mm，并在壁面到匹配点之间求24节点耦合速度/能量/组分/SST边值问题。它发布壁面交换和由辅助z剖面插值得到的ownerOmega。
baseline的辅助剖面并没有被约束等于粗网格owner的U/T/k。在x≈0.58处，该剖面y_owner处U约83.176 m/s，而粗网格owner为123.481 m/s。二者属于不同的数值状态表示，不能直接互换采样后要求同一剪切。
这项不一致不单独证明某个模型实现错误；它明确说明当前两种壁面闭合/粗网格耦合没有等价约束。若设计目标是可无差替换，该目标目前没有实现。
源：GasBoundaryLayerModelState.H:151-170，WallGeometryBuilder.H:407-409，ReactingWallLayer.H:854-856。

2. 首层omega和k生产项都换了。
wallFunction：omegaWallFunctionState用Cmu^(1/4)*sqrt(k)*y/nu选分支，高Re分支给出uStar/(sqrt(Cmu)*kappa*y)，生产项使用OpenFOAM10的速度法向梯度/对数律关系。
boundaryLayer：omega来自已解z=1/sqrt(omega)剖面；粗网格k生产项使用wallShearStress/rho*Cmu^(1/4)*sqrt(k)/(kappa*y)，然后应用共同的生产上限。
因此同属SST不代表首层边界闭合相同，不能将k/nut差异只解释成迭代次数或收敛容差。
源：OpenFoamWallFunctions.cuh:259-311，computeGasPrimitiveGradientsKernel.cuh:283-331，computeSstFaceFluxKernel.cuh:287-365。

四、固定同一粗网格场的真实生产算子重放
这组实验固定同一份粗网格owner和matching donor数据，分别按两个模型的正式采样规则调用，不改方程、不改容差、不推进时间。它不是将两个模型强行设置成同一采样高度。
5个站×旧/当前两份场共10组，evaluateWallModel全都成功，6-9次Newton，残差均满足原判据；未恢复任何旧终端放行。
旧baseline场x≈0.58：
  boundaryLayer：tau=48.6825 Pa，omega_owner=476546 s^-1，Pk=2.58343e6 m²/s³；
  wallFunction：tau=87.2764 Pa，omega_owner=268793 s^-1，Pk=5.52142e6 m²/s³。
这些Pk为壁面生产候选，均未达到共同c1*betaStar*k*omega上限；不是单纯上限截断差异。
最终已经演化后的当前wallFunction场，同站Spalding关系给tau约66.3282 Pa。不得把固定baseline输入的87.2764 Pa冒充当前最终场的剪切。
即使把boundaryLayer辅助剖面在owner高度的U/T/k交给壁函数，同站tau仍为44.6129而非48.6825 Pa，omega约357657而非476546；热流约-117.331而非-241.142 W/m²。这进一步说明两种闭合本身也不等价，采样高度不是唯一原因。
原始输入和结果：same-input-wall-models.csv、same-input-wall-results.csv、cross-model-commands.json、cross-model-build.log。输入由已保存场/剖面重建，并非原生产每个RK阶段的精确capture；未声称重现所有原阶段细节。

五、y+、热流与代码回归检查
当前336个壁面首层距离均约0.127389 mm。基于Spalding摩擦速度的y+在t=0.018为73.06–114.33；基于k的y+为69.96–106.37。没有证据支持“这里其实y+<1却误套高Re分支”的解释。处在对数区并不能证明局部平衡、热流关系或两种耦合等价。
热壁Tw=290.532 K且有显著动能/黏性热修正。当前x≈0.58的真实热函数重放：向外Fourier项约-2869.00 W/m²，动能修正约+3355.11 W/m²，净向外约+486.11（向气体为-486.11）。两大项相减，使温度/速度/摩擦尺度差异被净热流放大。不能只看T全局最大值的1%就推断热流也仅差1%。
baseline同站保存的heatFluxIntoGas约-241.142 W/m²；同一baseline粗网格场代入wallFunction约-438.157 W/m²。两者保留相同法向与正负约定。
此热函数为显式带动能修正的通量形式；并不把它宣称为OpenFOAM10 alphat边界系数整个迭代过程的逐字节等价实现。

另把旧main 7c696b19的真实OpenFoamWallFunctions.cuh取出，在同一CPU程序并列调用旧/新函数，覆盖三个保存时刻各336面共1008个实际输入：
  omega结果、生产项结果及热函数结果逐项完全相同；
  t=0.018新旧Spalding剪切关系最大相对差约0.2976%，另外两时刻约0.0003%；
  新Spalding方程归一化残差最大约7.8e-15；
  由实际保存k重放出的omega与当前保存omega一致。
这不支持“新版壁函数求根算错造成当前十几个百分点速度差”的解释；它也不能替代新旧wallFunction整场长算以完全排除其他路径。
原始数据：legacy-wallfunction-comparison.json，wall-legacy-replay-t*.csv。
壁面根及混合气真实通量/扩散调用专项：7 passed，退出0。测试仅证明所覆盖的实现契约，不是工程物理准确性证明。

六、判断与下一步
现阶段应将主要原因归为已确认的闭合及离散耦合差异，不能把wallFunction当作已验收boundaryLayer的无损提速开关，也不能只凭baseline被接受就断言壁函数的公式错误。
若要求复现已验收的baseline结果，继续使用boundaryLayer。若目标是让wallFunction独立承担工程预测，需要对壁剪切、首层omega/Pk、采样状态和热流关系分别确定一致的模型目标；再用制造解、壁面律及独立网格/实验对照验证，不能对M0.55或某面调经验参数使其贴合。
特别值得单独审查的是：两种路径为何使用不同的首层Pk关系，以及辅助剖面与粗网格owner状态没有一致约束时应如何定义模型交换。统一它们会改变数学模型，需以物理/离散依据决定，当前未擅自修改。
本轮未进行网格收敛、外部实验准确性评判、新旧wallFunction端到端长算，也未证明上述每项对最终差异的独立百分比贡献。没有把局部重放写成完整CFD误差归因。

参考核验：本机OpenFOAM10原始源码与官方对应页面：
https://cpp.openfoam.org/v10/omegaWallFunctionFvPatchScalarField_8C_source.html
https://cpp.openfoam.org/v10/alphatJayatillekeWallFunctionFvPatchScalarField_8C_source.html
'''
(O/'诊断报告.txt').write_text(report)
(O/'进度记录.json').write_text(json.dumps({'stage':'只读诊断及同场生产算子重放完成','source':provenance['source_sha'],'completed':['版本、冻结二进制、31个输入核验','同物理时刻3组场与5站剖面检查','10组生产BVP/壁函数同粗网格输入重放','1008组旧新壁函数并列调用','7个相关专项通过'],'finding':'主要差异定位到模型采样/omega/Pk/热流闭合不等价；尚无大幅求根回归证据','remaining':['新旧wallFunction端到端数值对照未做','外部物理解或网格收敛未做','未修改模型或启动新长算'],'evidence':str(O)},ensure_ascii=False,indent=2))
for p in Path('/mnt/c/Users/25094/Documents/Codex/2026-10-16/1-uniform-work-accepted-source-manifest/work').glob('wallfunction-*'):
 if p.is_file():shutil.copy2(p,O/p.name)
(O/'evidence-sha256.json').write_text(json.dumps({p.relative_to(O).as_posix():h(p)for p in O.rglob('*')if p.is_file()and p.name!='evidence-sha256.json'},indent=2))
print('REPORT',O/'诊断报告.txt');print('SOURCE_CLEAN',True)
