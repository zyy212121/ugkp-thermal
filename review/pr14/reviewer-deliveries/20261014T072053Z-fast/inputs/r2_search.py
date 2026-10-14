import re,sys
from r2_runner import *
pattern=re.compile(sys.argv[1].replace(',','|'),re.I);paths=sys.argv[2:]
for name in paths:
 p=ROOT/name
 files=list(p.rglob('*')) if p.is_dir() else [p]
 for q in files:
  if not q.is_file() or q.suffix not in ['.H','.cuh','.cu','.C','.py','.json','.md','.in']:continue
  try:lines=q.read_text().splitlines()
  except UnicodeError:continue
  hits=[i for i,l in enumerate(lines) if pattern.search(l)]
  if hits:
   print('FILE',q.relative_to(ROOT))
   for i in hits[:25]:print(str(i+1)+':'+lines[i])
