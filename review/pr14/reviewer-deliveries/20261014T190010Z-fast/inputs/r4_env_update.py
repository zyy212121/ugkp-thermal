from r4_runner import *
p=D/'inputs/env.sh';s=p.read_text();saved=D/'inputs/env.initial.sh';assert not saved.exists();saved.write_text(s)
anchor='export OMP_NUM_THREADS=2\n'
assert s.count(anchor)==1
new=s.replace(anchor,anchor+'''export PATH="$FOAM_USER_APPBIN:$CUDA_HOME/bin:$FOAM_APPBIN:$WM_PROJECT_DIR/wmake:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$CUDA_HOME/lib64:/usr/lib/wsl/lib:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI"
export GAS_UGKP_CUDA_BACKEND="$FOAM_USER_APPBIN/gasUGKPCudaBackend"
export FSH_CUDA_BACKEND="$FOAM_USER_APPBIN/FSHCudaBackend"
export WM_NCOMPPROCS=2
''')
p.write_text(new)
(D/'metadata/environment-revision.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','recorded_utc':utc(),'original_script':'inputs/env.initial.sh','original_sha256':sha(saved),'revised_script':'inputs/env.sh','revised_sha256':sha(p),'reason':'Before anysplit nativegasUGKP/FSH frontendexecution, bindactualchildbackend executablesandPATHexplicitlytoauthorizedtestinstall. PreviousCHMT runs directlylinkedownCUDA, nochildbackend. InitialgasUGKPbuildalreadyrunningusesinitialenvironment, installpath isstilltest. EachcommandrecordsactualenvironmentSHA.','production_paths_written':False},indent=2)+'\n')
print('TEST_INSTALL_ENV_BINDING',sha(p),flush=True)
