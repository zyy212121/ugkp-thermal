from r4_runner import *
import re
names=[('thermal-full',OLDROOT/'work/cases/chmt/thermal-full'),('chemistry-full',OLDROOT/'work/cases/chmt/chemistry-full-initcheck-1800'),('wet-film',OLDROOT/'work/cases/extra/wet-film-formal'),('wet-evaporation',OLDROOT/'work/cases/extra/wet-evaporation-formal-900')]
# Original real SST/static/moving recipes live in the prior R2 test checkout.
r3paths=json.loads((P/'r3_paths.json').read_text());r2=Path(r3paths['old_checkout'])
for n in ['SST-RK','SST-SSPRK2','SST-SSPRK3','moving-GCL','moving-chemistry']:
 candidates=[r2/'work/cases/extra'/n,OLDROOT/'work/cases/extra'/n]
 found=next((p for p in candidates if p.is_dir()),None)
 if found:names.append((n,found))
for a in ['CHT','FSH']:
 n=a+'-active16-ASCII-S306';names.append((n,OLDROOT/'work/cases/resident'/n))
receipt=[]
for name,oldcase in names:
 case=ROOT/'work/cases'/name;assert oldcase.is_dir(),oldcase
 if not case.exists():
  case.mkdir(parents=True)
  for directory in ['0','constant','system']:
   if (oldcase/directory).is_dir():shutil.copytree(oldcase/directory,case/directory,symlinks=False)
 for support in ['verification.json']:
  if (oldcase/support).is_file() and not (case/support).exists():shutil.copy2(oldcase/support,case/support)
 files=[]
 for p in case.rglob('*'):
  if not p.is_file():continue
  assert p.resolve().is_relative_to(Path(PATHS['repo']).resolve())
  oldp=oldcase/p.relative_to(case);assert sha(p)==sha(oldp),str(p)
  files.append({'path':str(p.relative_to(case)),'sha256':sha(p),'original_absolute_path':str(oldp),'bytes':p.stat().st_size})
 out=D/'inputs/cases'/name
 if not out.exists():shutil.copytree(case,out)
 else:
  for support in ['verification.json']:
   if (case/support).is_file() and not (out/support).exists():shutil.copy2(case/support,out/support)
 receipt.append({'name':name,'original_case':str(oldcase),'new_case':str(case),'all_original_input_bytes_equal':True,'files':files})
 print('CASE_READY',name,len(files),(case/'system/controlDict').read_text()[-1000:],flush=True)
(D/'metadata/case-input-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
for p in [P/'r4_case_prepare.py',P/'r4_native_rest.py']:shutil.copy2(p,D/'inputs'/p.name)
