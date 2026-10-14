from r2_runner import *
for cmdline in [['ldd',str(ROOT/'work/install/ns10/bin/CHMT')],['nm','-D',str(ROOT/'work/install/ns10/bin/CHMT')]]:
 p=subprocess.run(cmdline,capture_output=True,text=True)
 text=p.stdout
 if cmdline[0]=='nm':text='\n'.join(x for x in text.splitlines() if any(w in x for w in ['cudaLaunch','cudaMalloc','cudaMemcpy','FuncGetName']))
 print(text)
for label in ['baseline','formal']:
 print(label,[(str(p.relative_to(ROOT)),p.read_text()[:3000]) for p in (ROOT/'work/clock-build'/label).rglob('CHMTBuildIdentity.H')])
