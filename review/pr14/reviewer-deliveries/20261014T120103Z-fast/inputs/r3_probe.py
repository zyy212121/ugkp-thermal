from r3_runner import *
for name in ['wet-film-S209','wet-evaporation-S209']:
 c=OLDROOT/'work/cases/extra'/name
 print('CASE',name,'\nPARAMS\n'+(c/'constant/chmtProperties').read_text(),'\nCONTROL\n'+(c/'system/controlDict').read_text());print('INPUT_FILES',[str(p.relative_to(c)) for p in c.rglob('*') if p.is_file() and 'chmtOutput' not in p.parts][:75])
v=json.loads((DEV/'evidence/native_application_build/receipts.json').read_text())
print('DEV_CUDA_COUNTS',[(q['scope'],len(q['unchanged_cuda_region'])) for q in v])
print('BUILD_INFO\n',(ROOT/'applications/CHMT/devtools/build_info.py').read_text())
print('OLD_GENERATED',[(n.name,sha(n)) for n in (OLDROOT/'work/build/CHMT2/generated').iterdir() if n.is_file()])
print('SOURCE_FILES',len(json.loads((D/'metadata/source-hashes.json').read_text())))
print('GPU_HEALTH',subprocess.run(cmd('/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv'),capture_output=True,text=True).stdout)
