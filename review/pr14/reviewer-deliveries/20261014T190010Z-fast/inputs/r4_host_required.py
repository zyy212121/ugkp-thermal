from r4_runner import *
app=ROOT/'applications/CHMT';build=ROOT/'work/host-native';build.mkdir(parents=True,exist_ok=True)
flags=['-O2','-std=c++14','-D_GLIBCXX_ASSERTIONS','-Wall','-Wextra','-Werror','-pedantic','-I'+str(app),'-I'+str(ROOT/'common')]
specs=[('film',['devtools/multirate/test_cpu_film.cpp','film/CpuFilmDriver.C']),('curved',['devtools/multirate/test_cpu_film_curved.cpp','film/CpuFilmDriver.C','mesh/Geometry.C']),('lifecycle',['devtools/multirate/test_cpu_phase_lifecycle.cpp','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','materials/MaterialTransport.C','mesh/Geometry.C'])]
for name,sources in specs:
 b=build/name;files=[app/p for p in sources]
 t=run('BUILD-NATIVE-HOST-'+name,cmd('g++',*flags,*files,'-o',b),[3,6,13,14,20,21,26,29],expected='Originalcurrent nativefilmcomponent C++14/Werror flags; output/testinstallinsideauthorizedtest, noTMPcleanup/noCUDAmock.',inputs=files,kind='BUILD')
 if t['exit_code']==0:
  amend(t['id'],binary=str(b),binary_sha256=sha(b));q=run('HOST-NATIVE-'+name,cmd(b),[3,6,13,14,20,21,26,29],expected='Originalnative CPUfilm/curved/phase lifecycle assertions; standalonecomponent scope, noGPU acceptance.',inputs=files,binary=b,kind='HOST_NATIVE')
  assertion(q,q['exit_code']==0,(D/q['log']).read_text(errors='replace'))
# Reproduce the exact native phase API compile failure without its wrapper's hard timeouts.
b=build/'phase-original';f=['-std=c++17','-O1','-g','-ffp-contract=off','-fsanitize=bounds','-fno-sanitize-recover=bounds','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-DUGKWP_GAS_SPECIES=2','-I'+str(app),'-I'+str(ROOT/'common')]
for inc in ['OpenFOAM','finiteVolume','meshTools','OSspecific/POSIX']:f.append('-I/opt/openfoam10/src/'+inc+'/lnInclude')
files=[ROOT/'tests/fixtures/chmt_zero_phase_material/probe.C']+[app/n for n in ['materials/CpuMaterialDriver.C','materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C']]
libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-Wl,-rpath,/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream']
t=run('BUILD-ORIGINAL-NATIVE-PHASE-FIXTURE',cmd('g++',*f,*files,*libs,'-o',b),[3,6,13,14,20,21,26,29],expected='OriginalactualOF nativephase fixture flags withoutwrapperhardtimeout. StalereportAPI suspected; preserveformalcompile failure, not production numericalFAIL.',inputs=files+[ROOT/'tests/test_chmt_zero_phase_material.py'],kind='BUILD')
if t['exit_code']!=0:amend(t['id'],failure_class='FORMAL_TEST_API_MIGRATION_BLOCKER',observations='Formalphase fixture still referencesremovedfilmSteps,budgetDelta,solidTargetSweeps. Actualcompilefailure preserved. No productionaudit restoration.')
q=run('HOST-ORIGINAL-REACTING-ALE-FIXTURE',cmd('python3','-m','pytest','-q',ROOT/'tests/gas_transport/test_reacting_ale_transport.py','--basetemp',build/'ale-pytest'),[8,10,14,26,29],expected='OnlyreproducestalechemistryAudit fieldcompile failure inoriginalhost helper; noCUDA numerical acceptance orfull-suite claim.',inputs=[ROOT/'tests/gas_transport/test_reacting_ale_transport.py'],kind='HOST_API_REPRO')
if q['exit_code']!=0:amend(q['id'],failure_class='FORMAL_TEST_API_MIGRATION_BLOCKER',observations='OriginalhostALE fixture chemistryAudit reference no longer compiles. NoGPUfailure inferred; physicalnew-source movingchemistry test/independentobserver separate.')
