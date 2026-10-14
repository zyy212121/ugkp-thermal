from pathlib import Path
import subprocess,json,sys
P=Path(__file__).resolve().parent
paths=json.loads((P/'r4_paths.json').read_text());root=Path(paths['checkout'])
for arg in sys.argv[1:]:
 chunks=arg.rsplit(':',2);p=root/chunks[0]
 if not p.exists():print('MISSING',arg);continue
 if p.is_dir():print('DIRECTORY',arg,'\n'+'\n'.join(str(q.relative_to(root)) for q in p.rglob('*') if q.is_file()));continue
 lines=p.read_text(errors='replace').splitlines()
 if len(chunks)==3:lines=lines[int(chunks[1])-1:int(chunks[2])]
 print('FILE',arg,'\n'+'\n'.join(lines),flush=True)
