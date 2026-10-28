#!/usr/bin/env python3
"""Exercise shared variable transport through CHT and CHMT production owners."""
import argparse,importlib.util,json,re,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];V=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('matrix',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
models=json.loads((m.HERE/'fixtures.json').read_text());report={}
for name in ['cht-mixture','cht-wall','chmt-mixture','chmt-diffusion']:
 c=out/name;meta=models[name];m.prepare(m.HERE/'fixtures'/name,c,meta)
 text=(c/'constant/gasModelProperties').read_text();names=re.search(r'\bspecies\s*\((.*?)\)',text,re.S)[1].split();assert len(names)==2
 law=(V/'transport.dict').read_text();law=re.sub(r'\b(?:Air|CO2)\b',lambda x:names[0] if x[0]=='Air' else names[1],law)
 (c/'constant/gasTransportProperties').write_text(law)
 binary=R/'bin'/meta['solver'];seconds=m.run([binary,'-case',c],c/'run.log')
 restart=m.resume(c,out/(name+'-restart'),meta['solver'],binary)
 changed=out/(name+'-changed');m.copy_input(c,changed)
 if meta['solver']=='CHMT':
  checkpoint=c/'chmtOutput/checkpoint-1'
  config=changed/'constant/chmtProperties';config.write_text(config.read_text()+'\nrestartDirectory "'+str(checkpoint)+'";\n')
 else:
  saved=m.times(c)[0];shutil.copytree(saved,changed/saved.name);m.entry(changed/'system/controlDict','startTime',saved.name)
 q=changed/'constant/gasTransportProperties';q.write_text(q.read_text().replace('temperatureExponent 1.81','temperatureExponent 1.82'))
 m.run([binary,'-case',changed],changed/'run.log',True)
 report[name]=dict(seconds=seconds,restart=restart,changedCoefficientRejected=True)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(name,'PASS',flush=True)
