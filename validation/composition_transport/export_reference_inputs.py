#!/usr/bin/env python3
import argparse,hashlib,json,re,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent;report=[]
for c in sorted(a.references.glob('ARC3696_*')):
 t=max((d for d in c.iterdir() if d.is_dir() and re.fullmatch(r'[\d.]+',d.name)),key=lambda d:float(d.name))
 text=(t/'U').read_text();part=re.search(r'\binjection\s*\{(.*?)(?:\n    \}|\n\})',text,re.S)[1]
 m=re.search(r'massFlux\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)',part,re.S)
 entry='CO2 nonuniform List<scalar> '+m[1]+' ('+m[2]+');' if m else 'CO2 uniform 0;'
 d=a.output/c.name;d.mkdir();shutil.copyfile(here/'transport.dict',d/'gasTransportProperties')
 (d/'gasWallProperties').write_text('FoamFile { version 2.0; format ascii; class dictionary; object gasWallProperties; }\npatches { injection { speciesFluxFields { Air uniform 0; '+entry+' } } }\n')
 report.append(dict(case=c.name,time=t.name,velocitySHA256=hashlib.sha256((t/'U').read_bytes()).hexdigest(),transportSourceSHA256=hashlib.sha256((c/'runtime/source/arc3696TransportI.H').read_bytes()).hexdigest(),faceCount=int(m[1]) if m else None))
assert len(report)==6
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
