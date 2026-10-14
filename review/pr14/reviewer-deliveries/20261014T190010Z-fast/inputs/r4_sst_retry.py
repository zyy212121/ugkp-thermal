from r4_runner import *
second=ROOT/'work/observer-SST-D402';header=D/'inputs/r4_observer-D402.H'
build=ROOT/'work/build/D402-Ns2';install=ROOT/'work/install/diagSST/bin'
saved={str(p):sha(p) for p in [header,second/'common/GpuGasAdvance.cuh',second/'applications/CHMT/gpu/Backend.cu']}
q=run('BUILD-D402-IDENTICAL-ICE-RETRY',cmd('env',f'CHMT_BUILD={build}',f'FOAM_USER_APPBIN={install}','UGKWP_GAS_SPECIES=2','CHMT_CUDA_ARCH=sm_89','bash',second/'applications/CHMT/Allwmake'),[8,9,14,26,29],expected='One identical source/flag retry after captured GCC internal segmentation fault; no optimization or source reduction.',inputs=[header,second/'common/GpuGasAdvance.cuh',D/'patches/D402-actual-SST-constraint-inputs.patch'],kind='BUILD',variant='DIAGNOSTIC_D402')
assert saved=={n:sha(n) for n in saved};assert q['exit_code']==0
b=install/'CHMT';amend(q['id'],binary=str(b),binary_sha256=sha(b))
for name in ['SST-RK','SST-SSPRK2','SST-SSPRK3']+[f'SST64-RK{rk}-{geo}' for rk in [1,2,3] for geo in ['static','moving']]:
 base=ROOT/'work/cases'/name;case=ROOT/'work/cases'/('D402-'+name);case.mkdir()
 for n in ['0','constant','system']:shutil.copytree(base/n,case/n)
 q=run('D402-GPU-'+name,cmd(b,'-case',case),[8,9,14,26,29],expected='Read-only actual SST values, same formal case endpoint; original output byte comparison.',inputs=[p for n in ['0','constant','system'] for p in (case/n).rglob('*') if p.is_file()],binary=b,variant='DIAGNOSTIC_D402')
 rows=[{'path':str(p.relative_to(base/'chmtOutput')),'formal_sha256':sha(p),'observer_sha256':sha(case/'chmtOutput'/p.relative_to(base/'chmtOutput')),'byte_equal':p.read_bytes()==(case/'chmtOutput'/p.relative_to(base/'chmtOutput')).read_bytes()} for p in (base/'chmtOutput').rglob('*') if p.is_file()]
 artifact=D/'artifacts'/('D402-'+name+'-output-invariance.json');artifact.write_text(json.dumps({'all_original_files_byte_equal':all(r['byte_equal'] for r in rows),'comparisons':rows},indent=2)+'\n');assertion(q,q['exit_code']==0 and all(r['byte_equal'] for r in rows),'Read-only output invariance; separate physical oracle assessment.',artifacts=[str(artifact.relative_to(D))])
