#!/usr/bin/env python3
"""Copy original saved inputs and write fixed test controls. Never launch a solver."""
from pathlib import Path
import argparse,hashlib,json,re,shutil

import sys as _scheme_sys
_scheme_sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"validation/solver_controls"))
import migrate_case as scheme_controls

START='0.29500248649130967';END='0.30500348649130967';DT='7.962238329030135e-8'
LABELS=('pr41-fp64','candidate-fp64','candidate-fp32')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(text,key,value):return re.sub(r'(?m)^\s*'+re.escape(key)+r'\s+[^;]+;',key+' '+value+';',text,count=1)
def prepare(original,out):
 out.mkdir(parents=True,exist_ok=False)
 names=('0',START,'constant','system','assets')
 initial={str(f.relative_to(original)):sha(f) for name in names if (original/name).exists() for f in sorted((original/name).rglob('*')) if f.is_file()}
 manifest={'state':'PREPARED_NOT_RUN','source_case':str(original),'start':START,'end':END,'sequence_start':59,'sequences':[60,61],
  'radiation_sequence':60,'previous_radiation_time':0.29000244439008815,'initial_delta_t':DT,'initial_particles':52994,
  'omp_threads':4,'repetitions':1,'seed':'unchanged; saved particle RNG from common original checkpoint',
  'original_sha256':initial,'cases':[],'scope':'Open-domain total mass/energy closure cannot be reconstructed from saved snapshots alone.'}
 for label in LABELS:
  dest=out/('case-'+label);dest.mkdir()
  for name in names:
   if (original/name).exists():shutil.copytree(original/name,dest/name)
  ctrl=dest/'system/controlDict';text=ctrl.read_text()
  for key,value in [('startFrom','startTime'),('startTime',START),('endTime',END),('deltaT',DT)]:text=entry(text,key,value)
  ctrl.write_text(text)
  (dest/'constant/precisionProperties').write_text('FoamFile { version 2.0; format ascii; class dictionary; object precisionProperties; }\nprecision '+('32' if label.endswith('32') else '64')+';\n')
  (dest/START/'ugkpPrecision').write_text('schema 1\ncomputePrecision 64\nrestartPrecision 64\nstoragePolicy legacy64-v1\nwireEncoding existingFieldSchemas\n')
  if label.startswith('candidate-'):scheme_controls.migrate_case(dest,32 if label.endswith('32') else 64)
  manifest['cases'].append({'label':label,'directory':dest.name,'input_sha256':{str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file()},
   'changes':['startTime','endTime','initial deltaT','selected precision','copied checkpoint metadata64 only','candidate numerical controls moved to constant/scheme; PR41 retains old inputs'],
   'original_checkpoint_sha256':{name:sha(dest/name) for name in initial if name.startswith(START+'/')}})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(str(out)+' prepared; no numerical execution')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();prepare(a.original.resolve(),a.output.resolve())
