from r3_runner import *
import difflib
overlay=ROOT/'work/thermal-seed-D303';assert not overlay.exists()
for folder in ['common','applications/CHMT']:shutil.copytree(ROOT/folder,overlay/folder)
app=overlay/'applications/CHMT';formal=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text();s=old_read('inputs/Backend-actual-ledger-diagnostic.cu').decode()
anchor='static bool currentGasCourant(Backend& b,Real referenceDt,transport::GasCourantMaxima& result){'
helper=r'''static void r3CoSeed(Backend& b,const char* position,Real referenceDt){
    const auto& v=b.storage.hostView();std::vector<Real> temperature,pressure;
    if(!b.storage.read(v.Tgas,b.nCells,temperature)||!b.storage.read(v.p,b.nCells,pressure)){std::fprintf(stderr,"R3SEED_READ_ERROR,%s,%s\n",position,b.storage.error());return;}
    std::fprintf(stderr,"R3SEED_META,%s,%llu,%zu,%a,%a,%a,%a\n",position,(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,referenceDt,b.trial.nextDt,b.trial.lastAcceptedDt,b.microTime);
    for(int c=0;c<b.nCells;++c){const auto& q=b.trial.gas[c];std::fprintf(stderr,"R3SEED_CELL,%s,%llu,%zu,%d,%a,%a,%a,%a,%a,%a,%a,%a\n",position,(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,c,temperature[c],pressure[c],q.mass,q.energy,q.momentum.x,q.momentum.y,q.momentum.z,b.trial.gasMesh.volumes[c]);}
}
'''
assert s.count(anchor)==1;s=s.replace(anchor,helper+anchor)
anchor='    const int status=transport::computeGasCurrentCourant';assert s.count(anchor)==1;s=s.replace(anchor,'    r3CoSeed(b,"pre",referenceDt);\n'+anchor)
anchor='    // Initial scratch is not a live RK snapshot until advanceGasTrial saves it.';assert s.count(anchor)==1;s=s.replace(anchor,'    r3CoSeed(b,"post",referenceDt);\n'+anchor)
anchor='    const Real dt=selectedDt>0 && b.elapsed+selectedDt!=b.elapsed?selectedDt:referenceDt;';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n    std::fprintf(stderr,"R3DT,%llu,%zu,%a,%a,%a,%a\\n",(unsigned long long)b.program.interval.identity.sequence,b.history.records().size()+1,referenceDt,selectedDt,dt,co.maximum);')
source=app/'gpu/Backend-seed.cu';source.write_text(s);copy=D/'inputs/Backend-seed-observer-D303.cu';copy.write_text(s);patch=D/'patches/D303-readonly-thermal-Co-seed.patch';patch.write_text(''.join(difflib.unified_diff(formal.splitlines(True),s.splitlines(True),fromfile='a/applications/CHMT/gpu/Backend.cu',tofile='b/isolated/Backend-seed.cu')))
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nD303原thermal full-vs-restart首差诊断：仅新增未证实Tgas seed链观察，使用原8cell/100microsteps及实际checkpoint1后50步，不改物理输入/end/precision/比较。隔离真实Backend在currentCo前后D2H读Tgas/p，打印真实conservative hosttrial/volume/referenceDt/nextDt/lastAcceptedDt/selecteddt/Co及已证明D203最终flux/packet。新增同步/IO不参与效率；只读版原全部输出bytes必须与原formal full/restart各自相等，不将两个本来不等的输出写成绿。还要检查firstmicro差与后续T/p/packet。compile900/link900/run300。\n')
obj=ROOT/'work/fixtures/Backend-seed-D303.o';t=run('BUILD-THERMAL-SEED-D303-CUDA',cmd('nvcc','-std=c++17','-O2','-arch=sm_89','-DUGKWP_GAS_SPECIES=2','-I'+str(app),'-I'+str(overlay/'common'),'-I'+str(overlay/'common/gasNumerics'),'-c',source,'-o',obj),[5,9,12,14,26],900,'Readonlyactualseed/finalstage consumer chain; no arithmetic change.',inputs=[copy,patch],kind='BUILD',variant='DIAGNOSTIC_D303');assert t['exit_code']==0
app=ROOT/'applications/CHMT';build=ROOT/'work/build/CHMT2';binary=ROOT/'work/fixtures/CHMT-thermal-seed-D303'
flags=['-std=c++17','-O2','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-DUGKWP_GAS_SPECIES=2','-I'+str(app),'-I'+str(ROOT/'common'),'-I'+str(build/'generated'),'-I/opt/openfoam10/src/OpenFOAM/lnInclude','-I/opt/openfoam10/src/finiteVolume/lnInclude','-I/opt/openfoam10/src/meshTools/lnInclude','-I/opt/openfoam10/src/OSspecific/POSIX/lnInclude']
files=[app/'CHMT.C',*[app/x for x in ['io/MultirateEvolution.C','materials/CpuMaterialDriver.C','materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C','restart/Checkpoint.C']],obj];libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart']
t=run('BUILD-THERMAL-SEED-D303-HOST',cmd('g++',*flags,*files,*libs,'-o',binary),[5,9,12,14,26],900,'Fullformalhost againstreadonlyseed CUDAobject; notnumericvalidation.',inputs=files+[patch],kind='BUILD',variant='DIAGNOSTIC_D303');assert t['exit_code']==0
for name,oldname in [('thermal-full','thermal-full'),('thermal-restart','thermal-formal-resume')]:
 case=ROOT/'work/cases/chmt'/name;assert not case.exists();shutil.copytree(OLDROOT/'work/cases/chmt'/oldname,case,ignore=shutil.ignore_patterns('chmtOutput'))
 if name=='thermal-restart':
  cp=OLDROOT/'work/cases/chmt/thermal-full/chmtOutput/checkpoint-1';assert cp.exists()
  p=case/'constant/chmtProperties';s=p.read_text();import re;s=re.sub(r'restartDirectory\s+[^;]+;', 'restartDirectory "'+str(cp)+'";',s);p.write_text(s)
 shutil.copytree(case,D/'inputs/chmt'/name)
 print('THERMALREADY',name,flush=True)
