from r2_runner import *
for folder in [ROOT/'work/build',ROOT/'applications/CHMT/Make',ROOT/'applications/CHT/Make']:
 print('FOLDER',folder,[(p.name,p.is_dir()) for p in folder.iterdir()] if folder.exists() else 'none')
 for name in ['files','options']:
  for p in folder.rglob(name):
   if p.parent.name=='Make':print('FILE',p,p.read_text()[:7500])
print('objects',*[str(p.relative_to(ROOT)) for p in (ROOT/'work/build').rglob('*.o')],sep='\n')
