"""Read existing fixed16 outputs only; never launches a solver."""
from pathlib import Path
import ast,argparse,json,re,struct,math

def scan_outputs(case):
 result={'explicitNonfiniteFiles':[],'textResultFiles':0,'positiveTimeDirectories':0}
 for d in case.iterdir():
  try:positive=d.is_dir() and float(d.name)>0
  except ValueError:positive=False
  if not positive:continue
  result['positiveTimeDirectories']+=1
  for p in d.rglob('*'):
   if not p.is_file():continue
   raw=p.read_bytes()
   if b'\x00' in raw or p.suffix=='.dat':continue
   text=raw.decode(errors='replace')
   if 'internalField' not in text and p.suffix!='.csv':continue
   result['textResultFiles']+=1
   if re.search(r'(?<![\w.])[-+]?(?:nan|inf|infinity)(?![\w.])',text,re.I):result['explicitNonfiniteFiles'].append(str(p.relative_to(case)))
 return result

def review(source,manifest):
 source=Path(source);manifest=Path(manifest);plan=json.loads(manifest.read_text());original=source/'validation/resident_architecture/gas_graph.py';tree=ast.parse(original.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('fields','particle_states')];ns=dict(Path=Path,re=re,struct=struct);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),'exec'),ns)
 records={};pairs=[]
 for row in plan['processes']:
  c=Path(row['case']);item=dict(id=row['id'],outputScan=scan_outputs(c))
  try:
   f=ns['fields'](c);particles=ns['particle_states'](c);item['fields']=f;item['particleStates']=particles;item['finite']=bool(f) and all(bool(a) for a in f.values()) and bool(particles) and all(bool(state) for state in particles.values()) and item['outputScan']['positiveTimeDirectories']>0 and item['outputScan']['textResultFiles']>0 and not item['outputScan']['explicitNonfiniteFiles'] and all(math.isfinite(v) for a in f.values() for v in a) and all(math.isfinite(v) for state in particles.values() for a in state.values() for v in a[:10]);item['particleInventory']={t:dict(count=len(state),mass=sum(v[9] for v in state.values()),thermalEnergy=sum(v[9]*800*v[6] for v in state.values()),kineticEnergy=sum(v[9]*sum(q*q for q in v[3:6])/2 for v in state.values())) for t,state in particles.items()}
  except Exception as exc:item['readError']=repr(exc);item['finite']=False
  records[row['id']]=item
 for row in plan['processes']:
  if row['variant']!='candidate':continue
  left=records[row['id'].replace('-candidate','-baseline')];right=records[row['id']];pair=dict(id=row['id'],baselineFinite=left.get('finite'),candidateFinite=right.get('finite'))
  if 'readError' in left or 'readError' in right:pair['readErrors']=[left.get('readError'),right.get('readError')];pairs.append(pair);continue
  pair['missingFieldKeys']=sorted(set(left['fields'])^set(right['fields']));pair['fieldDifferences']={}
  for key in sorted(set(left['fields'])&set(right['fields'])):
   a,b=left['fields'][key],right['fields'][key]
   if len(a)!=len(b):pair['fieldDifferences'][key]={'shapeMismatch':[len(a),len(b)]};continue
   diff=[x-y for x,y in zip(a,b)];den=sum(x*x for x in a)**.5
   pair['fieldDifferences'][key]={'maxScaledDifference':max((abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(a,b)),default=0),'relativeL2':sum(x*x for x in diff)**.5/den if den else None}
  pair['particleMassRelativeDifferences']={t:(abs(right['particleInventory'][t]['mass']-a['mass'])/abs(a['mass']) if a['mass'] else (0 if right['particleInventory'][t]['mass']==0 else None)) for t,a in left['particleInventory'].items() if t in right['particleInventory']};pair['maximumIndividualMassRelativeDifference']=0;pair['zeroReferenceMassNonzeroCandidate']=False
  pair['particleInventoryBaseline']=left['particleInventory'];pair['particleInventoryCandidate']=right['particleInventory'];pair['particleDiscreteStateEqual']=True;pair['maxParticleScaledDifference']=0
  if left['particleStates'].keys()!=right['particleStates'].keys():pair['particleTimeSetMismatch']=True
  for t,a in left['particleStates'].items():
   b=right['particleStates'].get(t,{})
   if a.keys()!=b.keys():pair['particleIdentityMismatch']=True;continue
   for key,v in a.items():
    w=b[key]
    if v[9]!=0:pair['maximumIndividualMassRelativeDifference']=max(pair['maximumIndividualMassRelativeDifference'],abs(w[9]-v[9])/abs(v[9]))
    elif w[9]!=0:pair['zeroReferenceMassNonzeroCandidate']=True
    pair['particleDiscreteStateEqual']=pair['particleDiscreteStateEqual'] and v[10:]==w[10:];pair['maxParticleScaledDifference']=max(pair['maxParticleScaledDifference'],max(abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(v[:10],w[:10])))
  pairs.append(pair)
 return dict(status='OUTPUT_REVIEW_REQUIRED',runs={k:{n:v for n,v in item.items() if n not in ('fields','particleStates')} for k,item in records.items()},pairs=pairs,scope='Same-precision old/new physical output comparison. Not full-domain conservation proof; short open-boundary flow and finite pseudo-iteration budget.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();Path(a.output).write_text(json.dumps(review(a.source,a.manifest),indent=2)+'\n')
