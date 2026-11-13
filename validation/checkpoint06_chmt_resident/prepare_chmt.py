#!/usr/bin/env python3
"""Prepare CP6's declared 16 CHMT inputs. Never launch a solver or mesh generator."""
import argparse, hashlib, json, math, re, shutil
from pathlib import Path
VARIANTS=('baseline32','candidate32','baseline64','candidate64')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files(root):return {str(p.relative_to(root)):sha(p) for sub in ('0','constant','system') for p in sorted((root/sub).rglob('*')) if p.is_file()}
def clean(s):return re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
def body(p):
 s=clean(p.read_text());m=re.search(r'\b\d+\s*\(',s);assert m,p
 return s[m.end():s.rfind(')')]
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def geometry(mesh):
 points=[tuple(map(float,x.split())) for x in re.findall(r'\(([^()]+)\)',body(mesh/'points'))]
 faces=[list(map(int,x.split())) for x in re.findall(r'\d+\(([^()]+)\)',body(mesh/'faces'))]
 owner=list(map(int,body(mesh/'owner').split()));nei=list(map(int,body(mesh/'neighbour').split()))
 wall=re.search(r'\bwall\s*\{([^}]+)',clean((mesh/'boundary').read_text()))[1]
 wallface=int(re.search(r'startFace\s+(\d+)',wall)[1]);assert owner[wallface]==0
 ids=[i for i,o in enumerate(owner) if o==0 or (i<len(nei) and nei[i]==0)]
 verts={v for f in ids for v in faces[f]};center=tuple(sum(points[v][j] for v in verts)/len(verts) for j in range(3))
 planes=[]
 for f in ids:
  a,b,c=[points[v] for v in faces[f][:3]];n=cross(sub(b,a),sub(c,a));norm=math.sqrt(dot(n,n));assert norm>0
  n=tuple(x/norm for x in n)
  if dot(n,sub(center,a))>0:n=tuple(-x for x in n)
  assert max(abs(dot(n,sub(points[v],a))) for v in faces[f])<1e-12
  planes.append((f,a,n))
 mobile=(.25,.005,.005);attached=(.25,0,.005)
 margins={name:[{'face':f,'outwardSignedDistance':dot(n,sub(p,a))} for f,a,n in planes] for name,p in [('mobile',mobile),('attached',attached)]}
 assert all(x['outwardSignedDistance'] < -1e-12 for x in margins['mobile'])
 assert all(x['outwardSignedDistance'] <= 1e-12 for x in margins['attached'])
 assert abs(next(x['outwardSignedDistance'] for x in margins['attached'] if x['face']==wallface))<1e-12
 return {'method':'offline planar face half-space check from frozen ASCII points/faces/owner/neighbour; not a solver/operator execution', 'cell':0,'wallFace':wallface,'cellVertexBounds':[[min(points[v][j] for v in verts),max(points[v][j] for v in verts)] for j in range(3)],'margins':margins}
def particles(face):return f"""
// CP6 declared independent moving-live fixture; production original cases untouched.
enableParticles true;
particle {{ volumeMode Coupled; capacity 8; rhoSolid 2500;
 dragModel SchillerNaumann; heatTransferModel RanzMarshall; contactModel reboundContact;
 contactCoeffs {{ interfaceResistance 0.00001; }}
}}
particles {{
 mobile {{ id "41"; rng "1234566"; cell 0; position (0.25 0.005 0.005);
 velocity (0 0 0); mass 1e-5; diameter .001; temperature 2500; }}
 attached {{ id "42"; rng "1234567"; cell 0; face {face}; state 2;
 position (0.25 0 0.005); velocity (0 0 0); mass 1e-8; diameter .001; temperature 2500;
 contactAge .0001; contactDuration .001; contactArea 0;
 thermalHistory {{ temperature 2500; savedVelocity (0 1 0); contactMaximumArea 1e-6;
 contactPeakFraction .4; specificEnthalpyResidual .125; }}
 }}
}}
"""
def main():
 a=argparse.ArgumentParser();a.add_argument('--cp5-records',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 assert not x.output.exists(),'Never overwrite existing prepared inputs';x.output.mkdir(parents=True)
 donors={'static':x.cp5_records/'internal-static-repair1/inputs/static-candidate32','moving':x.cp5_records/'chmt-inputs/moving-candidate32'}
 before={k:files(v) for k,v in donors.items()};geo=geometry(donors['moving']/'constant/polyMesh');nodes=[]
 for kind in ('static','moving','static-fallback','moving-live'):
  source=donors['static' if kind.startswith('static') else 'moving'];reference={};common_reference=None
  for variant in VARIANTS:
   bits=int(variant[-2:]);case=x.output/(kind+'-'+variant);case.mkdir()
   for subdir in ('0','constant','system'):
    shutil.copytree(source/subdir,case/subdir,ignore=shutil.ignore_patterns('precisionProperties') if subdir=='constant' else None)
   static=kind.startswith('static')
   template=Path(__file__).resolve().parent/'templates'/('static.scheme' if static else 'moving.scheme')
   laplacian='Gauss linear limited 0.5' if kind=='static-fallback' else 'Gauss linear corrected'
   # Explicit reviewed fixture settings, not an importer or conversion of donor controls.
   wall_relative='1e-4' if bits==32 else '1e-8'
   wall_absolute='9.5367431640625e-7' if bits==32 else '1e-10'
   scheme=template.read_text().replace('@PRECISION@',str(bits)).replace('@SOLID_LAPLACIAN@',laplacian)
   scheme=scheme.replace('@WALL_RELATIVE_TOLERANCE@',wall_relative).replace('@WALL_ABSOLUTE_TOLERANCE@',wall_absolute)
   (case/'constant/scheme').write_text(scheme)
   changes=['hand-authored fixed '+template.name+'; scheme.precision '+str(bits)]
   if not static:changes.append('explicit reviewed wall tolerance for this test: relative '+wall_relative+', absolute '+wall_absolute)
   if kind=='static-fallback':
    changes.append('solidDiscretization.laplacianSchemes.default = Gauss linear limited 0.5 (derived existing static fixture; existing OpenFOAM runtime implementation; NOT previously run input)')
   if kind=='moving-live':
    p=case/'constant/chmtProperties';assert not re.search(r'\bparticles\s*\{',clean(p.read_text()))
    p.write_text(p.read_text()+particles(geo['wallFace']))
    changes.append('native particle dictionary: mobile + original PR31 reboundContact attached history; SchillerNaumann drag and RanzMarshall heat exchange; no 2D cold wall')
   hashes=files(case);same={k:v for k,v in hashes.items() if k!='constant/scheme'}
   common_equal=common_reference is None or same==common_reference
   if common_reference is None:common_reference=same
   pair_equal=bits not in reference or hashes==reference[bits]
   reference.setdefault(bits,hashes)
   # Manifest observations only; these are not solver refusal gates.
   parameter_audit={'samePrecisionOldNewHashesEqual':pair_equal,
     'crossPrecisionNonSchemeHashesEqual':common_equal,
     'declaredCrossPrecisionSchemeDifferences':['precision']+([] if static else ['wallLayer.relativeTolerance','wallLayer.absoluteTolerance']),
     'templateSha256':sha(template),'donorSchemeParsed':False}
   static=kind.startswith('static');name=case.name
   nodes.append({'id':name,'case':str(case.resolve()),'variant':variant,'precision':bits,'source':str(source.resolve()),'changes':changes,'parameterAudit':parameter_audit,'expectedWindows':3 if static else 8,'endTime':.0015 if static else 5.960464477539063e-8,'couplingInterval':.0005 if static else 7.450580596923828e-9,'inputSha256':hashes,'envOverrides':{'OMP_NUM_THREADS':'4','CHMT_CUDA_STACK_BYTES':'65536'},'solverArgvTemplate':['BINARY_'+variant,'-case',str(case.resolve())],'profileArgvTemplate':['/usr/local/bin/nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none','--force-overwrite=false','--output='+str((x.output/(name+'-profile')).resolve()),'BINARY_'+variant,'-case',str(case.resolve())],'log':str((x.output/(name+'.log')).resolve())})
 assert before=={k:files(v) for k,v in donors.items()},'source inputs changed'
 report={'status':'PREPARED_NOT_RUN','baselineCommit':'8be9da68d430e8d95df0fa56f22591fcc31361d5','nodes':nodes,'sourceSha256':before,'geometry':geo,'binaryBinding':'Parent must bind/hash frozen binaries before executing; this script never executes them','remainingParentOwnedRuns':{'count':2,'kind':'original CHT baseline continuation PR41 FP64 versus final candidate FP32','note':'Not prepared by this CHMT script; use original saved 0.29500248649130967, not prior comparison results'},'coverageLimitations':['moving live particles is a declared derived fixture, not a previous accepted calculation','Measure actual exchange, particle activity and geometry change after run; configuration alone does not prove activity','Do not infer conservation from successful exit','ColdWall2D legacy deliberately excluded']}
 (x.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'nodes':len(nodes),'wallFace':geo['wallFace'],'output':str(x.output)}))
if __name__=='__main__':main()
