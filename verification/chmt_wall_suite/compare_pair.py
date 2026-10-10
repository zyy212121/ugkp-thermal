#!/usr/bin/env python3
"""Report actual high-Re minus low-Re differences, never a hard agreement gate."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from metrics import read_field,final_directory,chmt_endpoints,wall_diagnostics

def linear(x,y,q):
    if len(x)!=len(y) or len(x)<2 or any(b<=a for a,b in zip(x,x[1:])):raise ValueError('invalid interpolation coordinates')
    if q<x[0] or q>x[-1]:raise ValueError('extrapolation is not permitted')
    i=min(len(x)-2,max(0,bisect.bisect_right(x,q)-1));f=(q-x[i])/(x[i+1]-x[i]);return (1-f)*y[i]+f*y[i+1]

def _norm(a,b,weights=None):
    if len(a)!=len(b) or not a:raise ValueError('sample mismatch')
    if any(not math.isfinite(v) for v in a+b):raise ValueError('nonfinite pair sample')
    w=weights or [1.]*len(a);total=sum(w)
    if len(w)!=len(a) or total<=0 or min(w)<=0:raise ValueError('invalid comparison weights')
    d=[u-v for u,v in zip(a,b)];scale=max(math.sqrt(sum(z*v*v for z,v in zip(w,b))/total),1e-30)
    return dict(L1=sum(z*abs(v) for z,v in zip(w,d))/total,L2=math.sqrt(sum(z*v*v for z,v in zip(w,d))/total),Linf=max(map(abs,d)),relative_L2=math.sqrt(sum(z*v*v for z,v in zip(w,d))/total)/scale,reference_rms=scale)

def profile_coordinates(points,case_id):
    import numpy as np
    points=np.asarray(points)
    if case_id.startswith('flatplate_'):
        return points[:,:2]
    if case_id.startswith('mss7_'):
        return np.column_stack((points[:,0],np.hypot(points[:,1],points[:,2])))
    raise ValueError('no declared comparison coordinate system for '+case_id)

def _mesh_hash(case):
    h=hashlib.sha256()
    for name in ('points','faces','owner','neighbour'):
        p=Path(case)/'constant/polyMesh'/name
        if not p.is_file():raise ValueError('actual mesh unavailable')
        h.update(p.read_bytes())
    return h.hexdigest()

def _sample_values(case,spec):
    d=final_directory(case,spec['end_time']);n=spec['cells'];U=read_field(d/'U',n,True)
    values={name:read_field(d/name,n) for name in ('T','rho','p')}
    for i,axis in enumerate('xyz'):values['U'+axis]=[u[i] for u in U]
    for name in ('Y_A','Y_B'):
        if (d/name).exists():values[name]=read_field(d/name,n)
    return values

def compare_pair(low,high):
    low=Path(low);high=Path(high)
    def loaded(case):
        if not (case/'run_status.json').exists():raise ValueError('no actual native run status')
        status=json.loads((case/'run_status.json').read_text())
        if status.get('native_execution')!='COMPLETED':raise ValueError('pair requires two completed native solver runs')
        return json.loads((case/'suite_case.json').read_text()),status
    a,sa=loaded(low);b,sb=loaded(high)
    if a['id']!=b['id'] or a['solver']!=b['solver'] or a.get('wall_family')!='lowRe':raise ValueError('pair must share case physics and use lowRe as reference')
    if a['models']!=b['models'] or a.get('material_card')!=b.get('material_card'):raise ValueError('unlike physical models/materials cannot be paired')
    for key in ('velocity','temperature','pressure','rho','closed'):
        if a['parameters'].get(key)!=b['parameters'].get(key):raise ValueError('pair operating condition differs: '+key)
    if not math.isclose(a['end_time'],b['end_time'],rel_tol=1e-12):raise ValueError('pair final times differ')
    result={'acceptance':'REPORT_ONLY','reference_case':str(low),'candidate_case':str(high),'reference_solver_sha256':sa.get('solver_sha256'),'candidate_solver_sha256':sb.get('solver_sha256'),'physics':a['models'],'profiles':{},'wall_reference':wall_diagnostics(low,a),'wall_candidate':wall_diagnostics(high,b),'yplus_policy':a['yplus_policy']}
    for side,status in [('reference',sa),('candidate',sb)]:
        result[side+'_backend']=status.get('backend',{'status':'NOT_APPLICABLE' if a['solver']=='CHMT' else 'UNAVAILABLE',
            'reason':'CHMT is monolithic' if a['solver']=='CHMT' else 'Run receipt did not record the selected gas backend; frontend SHA is not a substitute.'})
    if a['solver']=='CHMT':
        # Native material output currently has no physical x,y,z; do not compare
        # receding cells at fabricated common coordinates.
        _,_,af=chmt_endpoints(low,a);_,_,bf=chmt_endpoints(high,b)
        for name,index in [('gas',0),('material',1)]:
            def average(rows):return sum(float(r['volume'])*float(r['temperature']) for r in rows)/sum(float(r['volume']) for r in rows)
            result['profiles'][name+'_volume_average_T_K']={'lowRe':average(af[index]),'highRe':average(bf[index]),'difference':average(bf[index])-average(af[index])}
        result['spatial_profile_status']='UNAVAILABLE: receding native output lacks cell/face physical coordinates; provide independently sampled coordinates before field interpolation.'
        return result
    av=_sample_values(low,a);bv=_sample_values(high,b)
    if _mesh_hash(low)==_mesh_hash(high):
        for name in av:result['profiles'][name]=_norm(bv[name],av[name])
        result['weighting']='cell-count weighted on identical physical mesh; use exported V for volume-weighted extension'
        result['spatial_profile_status']='COMPUTED_IDENTICAL_MESH'
    else:
        # Actual stationary mesh centres are written by OpenFOAM postProcess,
        # never inferred from an assumed y+ or from cell numbering.
        if not (low/'0/C').exists() or not (high/'0/C').exists():raise ValueError('different meshes require actual 0/C cell-centre export; run postProcess -func writeCellCentres -time 0')
        try:
            import numpy as np
            from scipy.interpolate import LinearNDInterpolator
        except ImportError:raise ValueError('cross-grid interpolation requires numpy/scipy')
        ac=np.asarray(read_field(low/'0/C',a['cells'],True));bc=np.asarray(read_field(high/'0/C',b['cells'],True))
        xa=profile_coordinates(ac,a['id']);xb=profile_coordinates(bc,b['id'])
        for name in av:
            ref=LinearNDInterpolator(xa,np.asarray(av[name]))(xb)
            if not np.all(np.isfinite(ref)):raise ValueError('candidate centres extend outside reference centre hull; no unreported extrapolation')
            result['profiles'][name]=_norm(bv[name],ref.tolist())
        result['weighting']='candidate cell-count weighted; linear interpolation in Cartesian x/y for flat plates, axial/radial for MSS7'
        result['spatial_profile_status']='COMPUTED_CROSS_MESH'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('low',type=Path);p.add_argument('high',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare_pair(a.low,a.high);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
