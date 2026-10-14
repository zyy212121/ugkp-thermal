from r4_runner import *
import difflib
first=ROOT/'work/observer-source';second=ROOT/'work/observer-SST-D402';assert not second.exists();shutil.copytree(first,second)
old=(D/'inputs/r4_observer.H').read_text();extra='''
    if(s.sstConfigured){
        std::printf(",\"gasMu\":\"%a\",\"rhoMin\":\"%a\",\"sstKMin\":\"%a\",\"sstOmegaMin\":\"%a\",\"sstWallTreatment\":%d",double(s.gasMu),double(s.rhoMin),double(s.sstKMin),double(s.sstOmegaMin),s.sstWallTreatment);
        R4_CELL(sstWallDistance);R4_CELL(nut);R4_CELL(sstF1);R4_CELL(sstF2);
        R4_CELL(gradKX);R4_CELL(gradKY);R4_CELL(gradKZ);R4_CELL(gradOmegaX);R4_CELL(gradOmegaY);R4_CELL(gradOmegaZ);
        R4_CELL(gradUxX);R4_CELL(gradUxY);R4_CELL(gradUxZ);R4_CELL(gradUyX);R4_CELL(gradUyY);R4_CELL(gradUyZ);R4_CELL(gradUzX);R4_CELL(gradUzY);R4_CELL(gradUzZ);
        R4_FACE(sstBoundaryKMode);R4_FACE(sstBoundaryOmegaMode);
'''
for name in ['alphaK1','alphaK2','alphaOmega1','alphaOmega2','beta1','beta2','betaStar','gamma1','gamma2','a1','b1','c1']:
 extra+='        std::printf(",\\\"coeff_'+name+'\\\":\\\"%a\\\"",double(s.sstCoefficients.'+name+'));\n'
extra+='    }\n'
anchor='    using State=typename std::remove_reference<decltype(s)>::type;';assert old.count(anchor)==1
# Keep literal C++ JSON quote escapes for the multi-line prefix.
extra=extra.replace('std::printf(","gasMu":"%a","rhoMin":"%a","sstKMin":"%a","sstOmegaMin":"%a","sstWallTreatment":%d"','std::printf(",\\\"gasMu\\\":\\\"%a\\\",\\\"rhoMin\\\":\\\"%a\\\",\\\"sstKMin\\\":\\\"%a\\\",\\\"sstOmegaMin\\\":\\\"%a\\\",\\\"sstWallTreatment\\\":%d"')
header=D/'inputs/r4_observer-D402.H';header.write_text(old.replace(anchor,extra+anchor))
p=second/'common/GpuGasAdvance.cuh';s=p.read_text().replace(str(D/'inputs/r4_observer.H'),str(header));p.write_text(s)
patch=D/'patches/D402-actual-SST-constraint-inputs.patch';patch.write_text(''.join(difflib.unified_diff(old.splitlines(True),header.read_text().splitlines(True),fromfile='a/test-only/r4_observer.H',tofile='b/test-only/r4_observer-D402.H')))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD402追加原因：D401真实RK后wall-constrainedomega被已有recoverSstPrimitives重设为代数wall值，naive密度线性blend oracle红保存。只扩充只读host D2H/打印实际wallDistance/gasMu/coefficients/wallTreatment/真实SST与速度梯度/F1F2，独立核算已有低Re壁面omega约束及净/分解源；不改blend/recovery/生产SST代码或阈值。D401原数据/工具异常SIGSEGV/旧oracle失败独立保留；D402逐原SST及新64例原终点/原输入输出bytes再核等价。不以简单忽略wall差关SSTgross/graph范围。\n')
build=ROOT/'work/build/D402-Ns2';install=ROOT/'work/install/diagSST/bin';install.mkdir(parents=True,exist_ok=True)
q=run('BUILD-D402-SST-OBSERVER',cmd('env',f'CHMT_BUILD={build}',f'FOAM_USER_APPBIN={install}','UGKWP_GAS_SPECIES=2','CHMT_CUDA_ARCH=sm_89','bash',second/'applications/CHMT/Allwmake'),[8,9,14,26,29],expected='SameoriginalO2physics/kernels andD401readonlyIO, additionalactualSST scalar/gradient observations only; no wall algorithmchange.',inputs=[header,patch,p],kind='BUILD',variant='DIAGNOSTIC_D402')
assert q['exit_code']==0;b=install/'CHMT';amend(q['id'],binary=str(b),binary_sha256=sha(b))
for name in ['SST-RK','SST-SSPRK2','SST-SSPRK3']+[f'SST64-RK{rk}-{geo}' for rk in [1,2,3] for geo in ['static','moving']]:
 base=ROOT/'work/cases'/name;case=ROOT/'work/cases'/('D402-'+name);case.mkdir()
 for n in ['0','constant','system']:shutil.copytree(base/n,case/n)
 q=run('D402-GPU-'+name,cmd(b,'-case',case),[8,9,14,26,29],expected='ActualSSTsource/wallconstraint inputs withsameformaloriginalcase/full.001; alloriginaloutputbytes invariance.',inputs=[p for n in ['0','constant','system'] for p in (case/n).rglob('*') if p.is_file()],binary=b,variant='DIAGNOSTIC_D402')
 rows=[{'path':str(p.relative_to(base/'chmtOutput')),'formal_sha256':sha(p),'observer_sha256':sha(case/'chmtOutput'/p.relative_to(base/'chmtOutput')),'byte_equal':p.read_bytes()==(case/'chmtOutput'/p.relative_to(base/'chmtOutput')).read_bytes()} for p in (base/'chmtOutput').rglob('*') if p.is_file()]
 artifact=D/'artifacts'/('D402-'+name+'-output-invariance.json');artifact.write_text(json.dumps({'all_original_files_byte_equal':all(r['byte_equal'] for r in rows),'comparisons':rows},indent=2)+'\n');assertion(q,q['exit_code']==0 and all(r['byte_equal'] for r in rows),'AllD402originaloutputbytes='+str(all(r['byte_equal'] for r in rows))+'; independentlymodelwallconstraints next.',artifacts=[str(artifact.relative_to(D))])
