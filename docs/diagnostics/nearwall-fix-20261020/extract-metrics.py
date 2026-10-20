from pathlib import Path
import re,json,csv,subprocess,numpy as np
r=Path(__file__).resolve().parents[3]
o=r/'work/nearwall-fix-20261020'
w=Path(__file__).resolve().parent
exe=o/'wall-metrics'
if not exe.exists():
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(r),'-I'+str(r/'common'),str(w/'wall_metrics.cpp'),'-o',str(exe)],check=True)
def field(p,vec=False):
    b=p.read_bytes();kind=b'vector' if vec else b'scalar'
    m=re.search(rb'internalField\s+nonuniform\s+List<'+kind+rb'>\s+(\d+)\s*\(',b)
    if m:
        n=int(m[1])
        if re.search(rb'format\s+binary;',b[:1000]):
            a=np.frombuffer(b,dtype='<f8',count=n*(3 if vec else 1),offset=m.end()).copy()
        else:
            end=b.index(b';',m.end());a=np.fromstring(b[m.end():end].replace(b'(',b' ').replace(b')',b' ').decode(),sep=' ')
        return a.reshape((-1,3)) if vec else a
    m=re.search(rb'internalField\s+uniform\s+([^;]+);',b);assert m,p
    a=np.fromstring(m[1].replace(b'(',b' ').replace(b')',b' ').decode(),sep=' ')
    return np.tile(a,(26880,1)) if vec else np.full(26880,a[0])
coords=np.column_stack([field(o/'boundaryLayer-after/0'/('C'+d)) for d in 'xyz'])
indices=[]
for x in [.35,.38,.58,.75,.85]:
    cx=coords[np.argmin(abs(coords[:,0]-x)),0]
    col=np.flatnonzero(abs(coords[:,0]-cx)<1e-9);indices.append(int(col[np.argmin(coords[col,1])]))
records=[];audits={}
for model in ['wallFunction','boundaryLayer']:
    for variant in ['before','after','final']:
        case=o/(model+'-'+variant)
        if not case.exists():continue
        for tp in sorted(case.iterdir()):
            if not tp.is_dir() or not re.fullmatch(r'[0-9.eE+-]+',tp.name) or float(tp.name)==0:continue
            data={k:field(tp/k,k=='U') for k in ['rho','U','T','p','k','omega']}
            if model=='boundaryLayer':
                wp=tp/'uniform/gasBoundaryLayer.csv'
                if not wp.exists():continue
                walls={int(row['owner']):row for row in csv.DictReader(wp.open())}
                wallvals=[[float(walls[c]['tractionX']),float(walls[c]['heatFluxIntoGas'])] for c in indices]
                audits[str(tp.relative_to(o))]={key:max(abs(float(row.get(key,'nan'))) for row in walls.values()) for key in ['residual','heatBalanceResidual','ownerKBalanceResidual']}
                audits[str(tp.relative_to(o))]['iterations_mean']=sum(float(row['iterations']) for row in walls.values())/len(walls)
                audit=audits[str(tp.relative_to(o))]
                audit['wall_count']=len(walls)
                audit['residual_above_1e-8']=sum(float(row['residual'])>1e-8 for row in walls.values())
                audit['nonfinite_residual']=sum(not np.isfinite(float(row['residual'])) for row in walls.values())
                audit['iterations_max']=max(float(row['iterations']) for row in walls.values())
                audit['finite_all_cell_fields']=all(np.isfinite(v).all() for v in data.values())
                audit['minimum_k']=float(data['k'].min())
                audit['minimum_omega']=float(data['omega'].min())
            else:
                inp=''.join(' '.join(str(v) for v in [coords[c,1],data['p'][c],data['T'][c],*data['U'][c],data['k'][c]])+'\n' for c in indices)
                q=subprocess.run([str(exe)],input=inp,capture_output=True,text=True,check=True)
                wallvals=np.array([list(map(float,line.split())) for line in q.stdout.splitlines()])
            for c,wall in zip(indices,wallvals):
                records.append(dict(model=model,variant=variant,time=float(tp.name),owner=c,x=coords[c,0],y=coords[c,1],
                    U=data['U'][c,0],T=data['T'][c],k=data['k'][c],omega=data['omega'][c],tauX=wall[0],qIntoGas=wall[1]))
with (o/'stations.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=records[0],lineterminator='\n');writer.writeheader();writer.writerows(records)
(o/'wall-audits.json').write_text(json.dumps(audits,indent=2))
print(json.dumps([v for v in records if .57<v['x']<.59],indent=2))
print(json.dumps(audits,indent=2))
