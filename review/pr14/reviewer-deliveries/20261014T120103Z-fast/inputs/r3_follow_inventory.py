from r3_runner import *
import re
for base in [ROOT/'applications/gasUGKP', ROOT/'work/cases/extra/wet-film-formal', OLDROOT/'work/install/ns2/bin']:
 print('INVENTORY',base)
 for p in base.rglob('*'):
  if p.is_file() and ('chmtOutput' not in p.parts):
   if base.name=='gasUGKP' and p.name not in ['GpuResidentStrict.cu','GpuGasAdvance.cuh','SharedGasTrialPolicy.cuh']:continue
   print(str(p.relative_to(base)))
for path in ['applications/gasUGKP/private_backend/GpuResidentStrict.cu','work/cases/extra/wet-film-formal/system/blockMeshDict','work/cases/extra/wet-film-formal/system/solid/blockMeshDict','work/cases/extra/wet-film-formal/0/T','work/cases/extra/wet-film-formal/0/solid/T']:
 p=ROOT/path
 if p.is_file():print('FILE',path,p.read_text()[:4500])
print('AUDIT_PATHS')
for p in OLD.rglob('*film*json'):print(p.relative_to(OLD))
print('CODEC_LOG_CONFIG')
p=D/'artifacts/WET-FILM-FORMAL/checkpoint-1-full-decoded.json';s=json.loads(p.read_text());print(s.keys());print({k:v for k,v in s.items() if k in ['film','filmAux','budget','config']})
