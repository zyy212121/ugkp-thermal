from r3_runner import *
print('ALLWMAKE\n',(ROOT/'applications/CHMT/Allwmake').read_text())
print('DEVELOPER_RECEIPTS\n',(DEV/'evidence/native_application_build/receipts.json').read_text())
print('OLD_INPUTS')
for name in ['wet-film-S209','wet-evaporation-S209']:
 case=OLDROOT/'work/cases/extra'/name;print(name,'EXISTS',case.is_dir());print('FILES',[str(p.relative_to(case)) for p in case.rglob('*') if p.is_file() and 'chmtOutput' not in p.parts][:85]);print('PARAMS\n',(case/'constant/chmtProperties').read_text() if (case/'constant/chmtProperties').is_file() else '')
for name in ['EXTRA-wet-film-S209','EXTRA-wet-evaporation-S209-D204','CHMT-chemistry-full-initcheck']:
 t=next(q for q in json.loads((OLD/'results.json').read_text())['tests'] if q['id']==name);print('PRIORTEST',json.dumps({k:t[k] for k in ['id','argv','budget_seconds','binary','binary_sha256','input_hashes','expected','verdict','observations']},indent=2))
print('CUDA_OBJECTS',[(str(p),sha(p),p.stat().st_size) for p in [OLDROOT/'work/build/CHMT2/Backend.o',OLDROOT/'work/build/CHMT10/Backend.o'] if p.is_file()])
print('FIXTURE_GENERATOR\n',(ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py').read_text())
