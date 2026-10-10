"""Independent reference math and actual-file metrics. No production solver imports."""
import csv
import json
import math
from pathlib import Path
import re

NUM=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'

def read_field(path,count,vector=False):
    text=Path(path).read_text()
    if text.startswith('version https://git-lfs.github.com/spec/v1'):raise ValueError('unresolved Git LFS pointer: '+str(path))
    uniform=re.search(r'internalField\s+uniform\s+([^;]+);',text)
    if uniform:
        x=uniform.group(1).strip();v=tuple(map(float,x.strip('()').split())) if vector else float(x);result=[v]*count
    else:
        match=re.search(r'internalField\s+nonuniform\s+List<'+('vector' if vector else 'scalar')+r'>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
        if not match:raise ValueError('missing ASCII internalField in '+str(path))
        if int(match.group(1))!=count:raise ValueError('field size mismatch in '+str(path))
        result=[tuple(map(float,x.split())) for x in re.findall(r'\(([^()]*)\)',match.group(2))] if vector else list(map(float,match.group(2).split()))
    if len(result)!=count or (vector and any(len(x)!=3 for x in result)):raise ValueError('field count/vector shape mismatch')
    if not all(math.isfinite(y) for x in result for y in (x if vector else [x])):raise ValueError('nonfinite actual field')
    return result

def final_directory(case,end):
    candidates=[]
    for p in Path(case).iterdir():
        if p.is_dir():
            try:t=float(p.name)
            except ValueError:continue
            if math.isfinite(t) and abs(t-end)<=1e-10*max(abs(end),1e-8):candidates.append(p)
    if len(candidates)!=1:raise ValueError('actual requested final-time directory missing or ambiguous')
    return candidates[0]

def wave_average(a,b,t,*,length,velocity,diffusivity,mean=.5,amplitude=.2):
    if b<=a or length<=0 or t<0 or diffusivity<0:raise ValueError('invalid analytic wave domain')
    k=2*math.pi/length
    return mean+amplitude*math.exp(-diffusivity*k*k*t)*(math.cos(k*(a-velocity*t))-math.cos(k*(b-velocity*t)))/(k*(b-a))

def reacting_wave_average(a,b,t,*,rate=2.,**kwargs):
    return math.exp(-rate*t)*wave_average(a,b,t,**kwargs)

def couette_average(a,b,t,H,U,nu):
    if not 0<=a<b<=H or nu<=0 or t<0:raise ValueError('invalid Couette domain')
    if t==0:return 0.
    # u/U=y/H + (2/pi) sum_n (-1)^n sin(n*pi*y/H)/n exp[-nu(n*pi/H)^2*t].
    result=(a+b)/(2*H)
    for n in range(1,10001):
        decay=math.exp(-nu*(n*math.pi/H)**2*t)
        term=2*((-1)**n)*H*(math.cos(n*math.pi*a/H)-math.cos(n*math.pi*b/H))*decay/(n*n*math.pi**2*(b-a))
        result+=term
        if decay<1e-16:break
    return U*result

def _pressure_function(p,r,u,pk,g):
    a=math.sqrt(g*pk/r)
    if p>pk:
        A=2/((g+1)*r);B=(g-1)/(g+1)*pk;q=math.sqrt(A/(p+B))
        return (p-pk)*q,q*(1-.5*(p-pk)/(p+B))
    z=p/pk;return 2*a/(g-1)*(z**((g-1)/(2*g))-1),z**(-(g+1)/(2*g))/(r*a)

def riemann_state(x,t,*,interface,gamma,left,right):
    rl,ul,pl=left;rr,ur,pr=right;g=gamma
    if t<=0:return tuple(left if x<interface else right)
    al=math.sqrt(g*pl/rl);ar=math.sqrt(g*pr/rr);p=max(1e-9,.5*(pl+pr)-.125*(ur-ul)*(rl+rr)*(al+ar))
    for _ in range(100):
        fl,dl=_pressure_function(p,*left,g);fr,dr=_pressure_function(p,*right,g);pn=max(1e-12,p-(fl+fr+ur-ul)/(dl+dr))
        if abs(pn-p)<1e-13*(pn+p):p=pn;break
        p=pn
    fl,_=_pressure_function(p,*left,g);fr,_=_pressure_function(p,*right,g);us=.5*(ul+ur+fr-fl);xi=(x-interface)/t
    if xi<=us:
        if p>pl:
            speed=ul-al*math.sqrt((g+1)/(2*g)*p/pl+(g-1)/(2*g))
            if xi<=speed:return rl,ul,pl
            rs=rl*((p/pl+(g-1)/(g+1))/((g-1)/(g+1)*p/pl+1));return rs,us,p
        astar=al*(p/pl)**((g-1)/(2*g))
        if xi<=ul-al:return rl,ul,pl
        if xi>=us-astar:return rl*(p/pl)**(1/g),us,p
        u=2/(g+1)*(al+(g-1)*ul/2+xi);a=2/(g+1)*(al+(g-1)*(ul-xi)/2)
        return rl*(a/al)**(2/(g-1)),u,pl*(a/al)**(2*g/(g-1))
    if p>pr:
        speed=ur+ar*math.sqrt((g+1)/(2*g)*p/pr+(g-1)/(2*g))
        if xi>=speed:return rr,ur,pr
        rs=rr*((p/pr+(g-1)/(g+1))/((g-1)/(g+1)*p/pr+1));return rs,us,p
    astar=ar*(p/pr)**((g-1)/(2*g))
    if xi>=ur+ar:return rr,ur,pr
    if xi<=us+astar:return rr*(p/pr)**(1/g),us,p
    u=2/(g+1)*(-ar+(g-1)*ur/2+xi);a=2/(g+1)*(ar-(g-1)*(ur-xi)/2)
    return rr*(a/ar)**(2/(g-1)),u,pr*(a/ar)**(2*g/(g-1))

def _norm(errors,scale=1):
    if not errors:raise ValueError('empty actual/reference array')
    return dict(l1=sum(abs(x) for x in errors)/len(errors)/scale,l2=math.sqrt(sum(x*x for x in errors)/len(errors))/scale,linf=max(abs(x) for x in errors)/scale)

def apply_gate(actual,limits,gate):
    if gate not in ('hard','report_only'):raise ValueError('invalid gate')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in actual.values()):raise ValueError('nonfinite/nonnumeric metric')
    missing=[key for key in limits if key not in actual]
    if missing:raise ValueError('required metrics missing: '+', '.join(missing))
    checks={key:dict(value=actual[key],target=lim,within_target=actual[key]<=lim) for key,lim in limits.items()}
    return dict(acceptance=('PASS' if all(v['within_target'] for v in checks.values()) else 'FAIL') if gate=='hard' else 'REPORT_ONLY',metrics=actual,targets=checks)

def _csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def chmt_endpoints(case,spec):
    directory=Path(case)/'chmtOutput';items=[]
    for p in directory.glob('accepted-*.csv'):
        rows=_csv(p)
        if len(rows)!=1:raise ValueError('invalid accepted summary')
        row=rows[0];n=int(row['accepted_steps']);t=float(row['time'])
        if int(row['commit_sequence'])!=n:raise ValueError('noncommitted CHMT output')
        items.append((n,t))
    items.sort()
    if len(items)<2 or items[0][0]!=0:raise ValueError('no completed native CHMT history')
    if any(b[0]!=a[0]+1 or b[1]<=a[1] for a,b in zip(items,items[1:])):raise ValueError('CHMT missing/nonmonotone accepted history')
    if not math.isclose(items[-1][1],spec['end_time'],rel_tol=1e-10,abs_tol=1e-14):raise ValueError('CHMT final time differs from request')
    def inventory(n):
        gas=_csv(directory/f'gas-{n}.csv');solid=_csv(directory/f'material-{n}.csv')
        if len(gas)!=spec['cells'] or not solid:raise ValueError('unexpected native participant sizes')
        for rows in (gas,solid):
            for row in rows:
                for key,v in row.items():
                    if not math.isfinite(float(v)):raise ValueError('nonfinite accepted CSV')
                if float(row['volume'])<=0:raise ValueError('nonpositive volume')
        return gas,solid
    return items,inventory(items[0][0]),inventory(items[-1][0])

def surface_rate_history(times,observed,*,area,mass_flux,density,rate=0.):
    """Closed planar slab: constant surface injection and optional first-order S0 -> S1."""
    if len(times)<2 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('invalid surface-rate history times')
    if area<=0 or mass_flux<=0 or density<=0 or rate<0:
        raise ValueError('invalid independent surface-rate reference parameters')
    for values in observed.values():
        if len(values)!=len(times) or any(not math.isfinite(x) for x in values):
            raise ValueError('missing/nonfinite surface-rate history')
    initial0=observed['gas_S0_mass_kg'][0];initial1=observed['gas_S1_mass_kg'][0]
    injection=area*mass_flux
    expected={key:[] for key in ('removed_mass_kg','volume_loss_m3','gas_S0_mass_kg','gas_S1_mass_kg')}
    for time in times:
        elapsed=time-times[0];added=injection*elapsed
        survival=math.exp(-rate*elapsed)
        gas0=initial0*survival+(-injection*math.expm1(-rate*elapsed)/rate if rate else added)
        expected['removed_mass_kg'].append(added)
        expected['volume_loss_m3'].append(added/density)
        expected['gas_S0_mass_kg'].append(gas0)
        expected['gas_S1_mass_kg'].append(initial0+initial1+added-gas0)
    metrics={};samples={'time_s':times}
    for key,reference in expected.items():
        actual=observed[key]
        scale=max(max(abs(x-reference[0]) for x in reference),
            expected['volume_loss_m3'][-1] if key=='volume_loss_m3' else expected['removed_mass_kg'][-1])
        errors=[x-y for x,y in zip(actual,reference)]
        for norm,value in _norm(errors,scale).items():metrics['surface_rate_'+key+'_'+norm]=value
        samples[key]={'actual':actual,'reference':reference,'norm_scale':scale}
    return metrics,samples

def _chmt_metrics(case,spec,samples=None):
    items,initial,final=chmt_endpoints(case,spec)
    def mass(pair):return sum(float(r['mass']) for r in pair[0])+sum(sum(float(v) for k,v in r.items() if k.startswith('mass_')) for r in pair[1])
    def energy(pair):return sum(float(r['total_energy']) for rows in pair for r in rows)
    m0=mass(initial);e0=energy(initial)
    metrics={'mass_relative_residual':abs(mass(final)-m0)/max(m0,1e-30),'energy_relative_residual':abs(energy(final)-e0)/max(abs(e0),1e-30),'minimum_material_temperature_K':min(float(r['temperature']) for r in final[1]),'maximum_material_temperature_K':max(float(r['temperature']) for r in final[1]),'material_volume_change_m3':sum(float(r['volume']) for r in final[1])-sum(float(r['volume']) for r in initial[1]),'gas_mass_change_kg':sum(float(r['mass']) for r in final[0])-sum(float(r['mass']) for r in initial[0])}
    if spec.get('material_card') and spec['metric']!='contact':
        rho_c=spec['material_card']['condensed'][0]['rho'];removed=sum(float(r['mass_C0']) for r in initial[1])-sum(float(r['mass_C0']) for r in final[1])
        predicted_volume_loss=removed/rho_c;measured_volume_loss=-metrics['material_volume_change_m3']
        metrics['solid_removed_mass_kg']=removed;metrics['mass_predicted_volume_loss_m3']=predicted_volume_loss
        metrics['recession_volume_absolute_error_m3']=abs(measured_volume_loss-predicted_volume_loss)
        metrics['recession_volume_relative_error']=abs(measured_volume_loss-predicted_volume_loss)/max(abs(predicted_volume_loss),1e-20)
    parameters=spec.get('parameters',{})
    if parameters.get('closed') and 'surface_mass_flux_kg_m2_s' in parameters:
        observed={key:[] for key in ('removed_mass_kg','volume_loss_m3','gas_S0_mass_kg','gas_S1_mass_kg')}
        initial_solid=sum(float(row['mass_C0']) for row in initial[1])
        initial_volume=sum(float(row['volume']) for row in initial[1])
        for step,time in items:
            gas=_csv(Path(case)/'chmtOutput'/f'gas-{step}.csv')
            solid=_csv(Path(case)/'chmtOutput'/f'material-{step}.csv')
            if len(gas)!=spec['cells'] or len(solid)!=len(initial[1]):raise ValueError('incomplete surface-rate participant history')
            observed['removed_mass_kg'].append(initial_solid-sum(float(row['mass_C0']) for row in solid))
            observed['volume_loss_m3'].append(initial_volume-sum(float(row['volume']) for row in solid))
            for species in ('S0','S1'):observed['gas_'+species+'_mass_kg'].append(sum(float(row['mass_'+species]) for row in gas))
        rate_metrics,rate_samples=surface_rate_history([time for step,time in items],observed,
            area=parameters['surface_area_m2'],mass_flux=parameters['surface_mass_flux_kg_m2_s'],
            density=spec['material_card']['condensed'][0]['rho'],rate=parameters.get('reaction_rate_per_s',0.))
        metrics.update(rate_metrics)
        if samples is not None:samples['surface_rate_history']=rate_samples
    # Open-flow sums are changes, not conservation residuals without boundary fluxes.
    if spec.get('parameters',{}).get('closed') is False:
        metrics['relative_mass_inventory_change']=metrics.pop('mass_relative_residual');metrics['relative_energy_inventory_change']=metrics.pop('energy_relative_residual')
    if spec['metric']=='contact':
        first=_csv(Path(case)/'chmtOutput'/f'material-{items[1][0]}.csv')
        dE=sum(float(r['total_energy']) for r in first)-sum(float(r['total_energy']) for r in initial[1])
        metrics['initial_heat_rate_W']=dE/(items[1][1]-items[0][1]);metrics['target_initial_heat_rate_W']=600.
        metrics['initial_heat_rate_relative_error']=abs(metrics['initial_heat_rate_W']-600)/600
    return metrics

def compare(case):
    samples={}
    case=Path(case);spec=json.loads((case/'suite_case.json').read_text());kind=spec['metric'];p=spec['parameters'];n=spec['cells'];t=spec['end_time']
    if kind in ('chmt','contact','chmt_profiles'):actual=_chmt_metrics(case,spec,samples)
    elif kind=='reactor':
        # Existing checker compares actual field histories against independently pinned Cantera.
        from foam import module
        checker=module('test/gasUGKP/chemistryReactor/check_case.py')
        result=checker.compare_case(case)
        contract=json.loads((case/'case_contract.json').read_text())
        if result.get('checked_times')!=contract['check_times']:
            raise ValueError('reactor evidence is incomplete: required history was not checked')
        # Numerical disagreement is report-only; missing/invalid evidence is not.
        numerical_metrics=('normalized_history_error','max_relative_density_error',
            'max_relative_energy_inventory_error','max_relative_energy_closure_error',
            'max_relative_pressure_closure_error','max_species_sum_error',
            'max_element_relative_error','max_velocity')
        for failure in result.get('failures',[]):
            numerical=any(failure.startswith(key+' exceeds ') for key in numerical_metrics)
            numerical=numerical or failure=='fine history exceeds the verified common-core error envelope'
            if not numerical:raise ValueError('reactor evidence is invalid: '+failure)
        for key in numerical_metrics:
            if key not in result or not isinstance(result[key],(int,float)) or not math.isfinite(result[key]):
                raise ValueError('reactor evidence metric missing/nonfinite: '+key)
        return {'acceptance':'REPORT_ONLY','reference':'Pinned independent Cantera','raw_metrics':result,'limits_are_objectives':True}
    else:
        d=final_directory(case,t)
        if kind=='couette':
            u=read_field(d/'U',n,True);H=p['height'];exact=[couette_average(i*H/n,(i+1)*H/n,t,H,p['wall_velocity'],p['kinematic_viscosity']) for i in range(n)]
            samples={'velocity':{'actual':[v[0] for v in u],'reference':exact,'cell_edges_y':[i*H/n for i in range(n+1)],'units':'m/s','norm_scale':p['velocity_scale']}}
            norm=_norm([v[0]-e for v,e in zip(u,exact)],p['velocity_scale']);actual={'velocity_'+k:v for k,v in norm.items()};actual['crossflow_linf']=max(abs(v[1])+abs(v[2]) for v in u)
        elif kind in ('wave','reacting_wave'):
            a=read_field(d/'Y_A',n);b=read_field(d/'Y_B',n);kw={k:p[k] for k in ('length','velocity','diffusivity','mean','amplitude')};dx=p['length']/n
            oracle=reacting_wave_average if kind=='reacting_wave' else wave_average
            exact=[oracle(i*dx,(i+1)*dx,t,**kw) for i in range(n)]
            samples={'Y_A':{'actual':a,'reference':exact},'Y_B':{'actual':b,'reference':[1-e for e in exact]},'cell_edges_x':[i*dx for i in range(n+1)]}
            expected_mean=p['mean']*(math.exp(-p['reaction_rate_per_s']*t) if kind=='reacting_wave' else 1.)
            err=[max(abs(x-e),abs(y-(1-e))) for x,y,e in zip(a,b,exact)];actual={'species_'+k:v for k,v in _norm(err).items()}
            actual.update(species_sum_error=max(abs(x+y-1) for x,y in zip(a,b)),species_mass_mean_error=max(abs(sum(a)/n-expected_mean),abs(sum(b)/n-(1-expected_mean))),minimum_species=min(a+b),maximum_species=max(a+b))
            bulk_targets={'rho':p['rho'],'p':p['pressure'],'T':p['temperature'],'rhoE':p['rhoE']}
            for field,target in bulk_targets.items():
                values=read_field(d/field,n);actual.update({field+'_'+k:v for k,v in _norm([x-target for x in values],abs(target)).items()})
            rho=read_field(d/'rho',n);energy=read_field(d/'rhoE',n)
            actual['total_mass_relative_residual']=abs(sum(rho)/n-p['rho'])/p['rho']
            actual['total_energy_relative_residual']=abs(sum(energy)/n-p['rhoE'])/abs(p['rhoE'])
        elif kind=='sod':
            r=read_field(d/'rho',n);pres=read_field(d/'p',n);u=read_field(d/'U',n,True);dx=p['length']/n;kw={k:p[k] for k in ('interface','gamma','left','right')}
            # Independent midpoint quadrature is refined enough to make oracle error visible separately.
            exact=[]
            for i in range(n):
                states=[riemann_state((i+(j+.5)/64)*dx,t,**kw) for j in range(64)]
                rbar=sum(z[0] for z in states)/64;momentum=sum(z[0]*z[1] for z in states)/64;energy=sum(z[2]/(p['gamma']-1)+.5*z[0]*z[1]**2 for z in states)/64
                exact.append((rbar,momentum/rbar,(p['gamma']-1)*(energy-.5*momentum*momentum/rbar)))
            samples={name:{'actual':values,'reference':[e[index] for e in exact]} for name,values,index in [('rho',r,0),('U_x',[v[0] for v in u],1),('p',pres,2)]}
            samples['cell_edges_x']=[i*dx for i in range(n+1)]
            speed=math.sqrt(p['gamma']*p['left'][2]/p['left'][0]);actual={}
            for field,values,index,scale in [('rho',r,0,p['left'][0]),('u',[v[0] for v in u],1,speed),('p',pres,2,p['left'][2])]:
                actual.update({field+'_'+k:v for k,v in _norm([v-e[index] for v,e in zip(values,exact)],scale).items()})
            initial_mass=p['interface']*p['left'][0]+(p['length']-p['interface'])*p['right'][0]
            initial_energy=(p['interface']*p['left'][2]+(p['length']-p['interface'])*p['right'][2])/(p['gamma']-1)
            actual['mass_relative_residual']=abs(dx*sum(r)-initial_mass)/initial_mass
            actual['energy_relative_residual']=abs(dx*sum(v/(p['gamma']-1)+.5*z*sum(w*w for w in vec) for v,z,vec in zip(pres,r,u))-initial_energy)/initial_energy
            actual['momentum_balance_relative_error']=abs(dx*sum(z*v[0] for z,v in zip(r,u))-t*(p['left'][2]-p['right'][2]))/max(t*abs(p['left'][2]-p['right'][2]),1e-30)
        elif kind=='profiles':
            T=read_field(d/'T',n);rho=read_field(d/'rho',n);U=read_field(d/'U',n,True)
            actual=dict(minimum_T_K=min(T),maximum_T_K=max(T),minimum_rho_kg_m3=min(rho),maximum_speed_m_s=max(math.sqrt(sum(x*x for x in v)) for v in U))
        else:raise ValueError('unknown metric '+kind)
    out=apply_gate(actual,spec['limits'],spec['gate'])
    out['reference']=spec['reference'];out['native_output_read']=True
    out['samples']=samples;out['time']=t;out['cells']=n;out['time_controls']=spec.get('time_controls',{})
    out['norm_definition']='L1=mean(abs(error)), L2=sqrt(mean(error^2)), Linf=max(abs(error)), divided by the stated physical scale; no NaN omission.'
    out['wall_flux_diagnostics']=wall_diagnostics(case,spec)
    if spec.get('id')=='wall_constant_transport':
        require_couette_wall_evidence(out['wall_flux_diagnostics'],spec)
    if spec['category']==3:out['paired_comparison']='Use compare_pair.py with the actual lowRe and highRe cases; single-run inventory statistics do not validate wall model accuracy.'
    return out

def require_couette_wall_evidence(diagnostic,spec):
    """A bulk Couette fit alone cannot establish execution of the new wall model."""
    time=diagnostic.get('directory_time')
    if diagnostic.get('status')!='AVAILABLE' or not isinstance(time,(int,float)) or not math.isfinite(time):
        raise ValueError('new wall hard case requires actual gas boundary-layer diagnostics')
    if not math.isclose(time,spec['end_time'],rel_tol=1e-10,abs_tol=1e-14):
        raise ValueError('new wall hard case requires diagnostics at the requested final write')
    keys=('face owner matchingDistance ownerDistance heatFluxIntoGas tractionX tractionY tractionZ '
          'wallKFlux integratedKSource ownerOmega residual iterations traceDensity viscosity stageTime '
          'speciesFlux_A speciesFlux_B reactionIntegral_A reactionIntegral_B').split()
    rows=_csv(diagnostic['file'])
    # The generated Couette mesh has one face on each of its two physical walls.
    if len(rows)!=2 or any(not set(keys).issubset(row) or None in row
            or any(row[key] in (None,'') for key in keys) for row in rows):
        raise ValueError('new wall hard case has incomplete face/schema evidence')
    values=[{key:float(row[key]) for key in keys} for row in rows]
    if any(not all(math.isfinite(value) for value in row.values()) for row in values):
        raise ValueError('new wall hard case contains nonfinite evidence')
    for row in values:
        if any(row[key]<0 or row[key]!=int(row[key]) for key in ('face','owner','iterations')):
            raise ValueError('new wall hard case contains invalid identifiers/iterations')
        if row['owner']>=spec['cells'] or not 0<row['ownerDistance']<row['matchingDistance']:
            raise ValueError('new wall hard case contains invalid owner geometry')
        if row['traceDensity']<=0 or row['viscosity']<=0 or row['residual']<0:
            raise ValueError('new wall hard case contains invalid physical diagnostics')
        if not 0<row['stageTime']<=time+1e-14:
            raise ValueError('new wall hard case lacks a valid evolved stage sample')
    if len({row['face'] for row in values})!=2 or len({row['stageTime'] for row in values})!=1:
        raise ValueError('new wall hard case contains duplicate faces or inconsistent stages')

_BUDGET_CHANNELS=('budgetTransportK','budgetTransportOmega','budgetSourceK',
                  'budgetSourceOmega','budgetConstraintK','budgetConstraintOmega')
_BUDGET_FIELDS=('budgetIntervalStart','budgetIntervalDuration')+_BUDGET_CHANNELS
_BUDGET_COLUMNS=('budgetAuditAvailable',)+_BUDGET_FIELDS

def gas_budget_diagnostics(data):
    """Optional accepted-microstep owner-cell increments, never wall-face fluxes."""
    if not any(key in row for row in data for key in _BUDGET_COLUMNS):return None
    if any(not set(_BUDGET_COLUMNS).issubset(row) or any(row[key] is None for key in _BUDGET_COLUMNS) for row in data):
        raise ValueError('incomplete optional budget diagnostic schema')
    available=[];owners=set()
    for row in data:
        try:flag=float(row['budgetAuditAvailable'])
        except (TypeError,ValueError):raise ValueError('invalid budget availability') from None
        if flag not in (0.,1.):raise ValueError('invalid budget availability')
        if flag==0:
            if any(row[key]!='' for key in _BUDGET_FIELDS):raise ValueError('unavailable budget contains nonblank values')
            continue
        try:values={key:float(row[key]) for key in _BUDGET_FIELDS};owner=float(row['owner'])
        except (KeyError,TypeError,ValueError):raise ValueError('incomplete numeric budget evidence') from None
        if not all(math.isfinite(value) for value in values.values()) or not math.isfinite(owner):
            raise ValueError('nonfinite budget evidence')
        if values['budgetIntervalDuration']<0:raise ValueError('negative budget interval duration')
        if owner<0 or owner!=int(owner) or owner in owners:raise ValueError('invalid or duplicate budget owner')
        owners.add(owner);available.append(values)
    statistics={}
    for key in _BUDGET_CHANNELS if available else ():
        values=[row[key] for row in available]
        statistics[key]={'min':min(values),'max':max(values),'mean_available_owner_rows':sum(values)/len(values)}
    intervals={}
    for row in available:
        key=(row['budgetIntervalStart'],row['budgetIntervalDuration'])
        intervals[key]=intervals.get(key,0)+1
    return {'status':'AVAILABLE' if available else 'UNAVAILABLE','available_rows':len(available),
        'unavailable_rows':len(data)-len(available),'statistics':statistics,
        'intervals':[{'start':start,'duration':duration,'available_owner_rows':count} for (start,duration),count in sorted(intervals.items())],
        'units':{key:('s' if key.startswith('budgetInterval') else 'kg/s' if key.endswith('Omega') else 'J') for key in _BUDGET_FIELDS},
        'semantics':'Last accepted microstep integrated owner-cell increments: inward transport positive, signed sources, and suppressed-row/floor/projection constraints. These are not face fluxes; interval start and duration are explicit.'}

def wall_diagnostics(case,spec):
    if list((Path(case)/'chmtOutput').glob('wall-layer-*.csv')):return chmt_wall_diagnostics(case,spec)
    candidates=[]
    for file in Path(case).glob('*/uniform/gasBoundaryLayer.csv'):
        try:time=float(file.parent.parent.name)
        except ValueError:continue
        candidates.append((time,file))
    if not candidates:return {'status':'UNAVAILABLE','reason':'No actual boundary-layer wall face export. Bulk gradient reconstruction is not substituted for modeled wall flux.','yplus':'UNAVAILABLE'}
    time,path=max(candidates);data=_csv(path)
    if not data:raise ValueError('empty wall diagnostic file')
    output={'status':'AVAILABLE','file':str(path),'directory_time':time,'rows':len(data),'columns':list(data[0]),'statistics':{},'yplus':'UNAVAILABLE'}
    budget=gas_budget_diagnostics(data)
    if budget is not None:output['budget_audit']=budget
    numeric={}
    for key in data[0]:
        if key in _BUDGET_COLUMNS:continue
        try:values=[float(row[key]) for row in data]
        except ValueError:continue
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite actual wall diagnostic '+key)
        numeric[key]=values;output['statistics'][key]={'min':min(values),'max':max(values),'mean_face_count_weighted':sum(values)/len(values)}
    required=['traceDensity','viscosity','tractionX','tractionY','tractionZ','ownerDistance']
    if all(k in numeric for k in required):
        values=[]
        for row in data:
            rho=float(row['traceDensity']);mu=float(row['viscosity']);distance=float(row['ownerDistance'])
            if rho<=0 or mu<=0 or distance<=0:raise ValueError('invalid y+ physical denominator/distance')
            tau=math.sqrt(sum(float(row['traction'+axis])**2 for axis in 'XYZ'))
            values.append(distance*math.sqrt(tau/rho)*rho/mu)
        output['yplus']={'status':'COMPUTED','definition':'ownerDistance*rho*sqrt(norm(tangential traction)/rho)/mu; traction is tangential by production wall output contract','min':min(values),'max':max(values),'mean_face_count_weighted':sum(values)/len(values),'claim':'Measured diagnostic only; no automatic acceptance range.'}
    output['stage_semantics']='stageTime is the exported solver-stage sample, which need not equal directory time; no endpoint refresh is assumed.'
    output['integrated_heat_rate']='UNAVAILABLE without face area export; heatFluxIntoGas remains W/m2, not W.'
    return output


def chmt_wall_diagnostics(case,spec):
    """Read accepted CHMT snapshots; species rows repeat face-level quantities."""
    candidates=[]
    for path in (Path(case)/'chmtOutput').glob('wall-layer-*.csv'):
        data=_csv(path)
        if not data:raise ValueError('empty CHMT wall diagnostic file')
        required=set('accepted_time stage_time face status species wall_temperature wall_density wall_mass_fraction wall_species_flux matching_species_flux reaction_integral species_balance_residual chemistry_mismatch_available chemistry_mismatch conductive_heat_flux traction_x traction_y traction_z wall_k_flux owner_k_source_integral owner_omega iterations residual'.split())
        if any(not required.issubset(row) or None in row or any(row[k] is None for k in required) for row in data):raise ValueError('incomplete CHMT wall diagnostic schema')
        times=[float(row['accepted_time']) for row in data]
        if not all(math.isfinite(t) and t>=0 and t==times[0] for t in times):raise ValueError('inconsistent accepted wall times')
        step=int(path.stem.removeprefix('wall-layer-'))
        candidates.append((times[0],step,path,data))
    time,step,path,data=max(candidates,key=lambda item:item[:2])
    output={'status':'UNAVAILABLE','file':str(path),'accepted_time':time,'step':step,'rows':len(data),'columns':list(data[0]),'yplus':'UNAVAILABLE',
        'yplus_reason':'CHMT wall export does not contain matching viscosity and owner distance.',
        'stage_semantics':'accepted_time is the accepted endpoint; stage_time identifies the exported accepted stage, not an endpoint recomputation.',
        'integrated_heat_rate':'UNAVAILABLE without face area; conductive_heat_flux is W/m2.'}
    statuses={row['status'] for row in data}
    if statuses=={'NOT_AVAILABLE'}:
        if any(any(row[key] for key in row if key not in ('accepted_time','face','status','species')) for row in data):raise ValueError('unavailable CHMT snapshot contains physical values')
        output['reason']='Latest accepted snapshot has no rebuilt wall diagnostics.'
        return output
    if statuses!={'AVAILABLE'}:raise ValueError('partial or unknown CHMT diagnostic status')
    face_fields='wall_temperature wall_density conductive_heat_flux traction_x traction_y traction_z wall_k_flux owner_k_source_integral owner_omega iterations residual'.split()
    species_fields='wall_mass_fraction wall_species_flux matching_species_flux reaction_integral species_balance_residual'.split()
    faces={};species={};seen=set();stages=set();face_species={}
    for row in data:
        face=int(row['face']);name=row['species']
        if not name or (face,name) in seen:raise ValueError('duplicate or unnamed wall species')
        seen.add((face,name));face_species.setdefault(face,set()).add(name)
        stage=float(row['stage_time']);stages.add(stage)
        values={key:float(row[key]) for key in face_fields+species_fields}
        flag=int(row['chemistry_mismatch_available'])
        if flag not in (0,1):raise ValueError('invalid chemistry mismatch availability')
        if flag:values['chemistry_mismatch']=float(row['chemistry_mismatch'])
        elif row['chemistry_mismatch']:raise ValueError('unavailable chemistry mismatch contains value')
        if not math.isfinite(stage) or not 0<=stage<=time or not all(math.isfinite(v) for v in values.values()):raise ValueError('nonfinite or invalid CHMT wall diagnostic')
        physical={key:values[key] for key in face_fields}
        if face in faces and faces[face]!=physical:raise ValueError('inconsistent repeated face diagnostic')
        faces[face]=physical
        species.setdefault(name,[]).append({key:values[key] for key in species_fields+(['chemistry_mismatch'] if flag else [])})
    if len(stages)!=1 or any(names!=set(species) for names in face_species.values()):raise ValueError('incomplete CHMT stage/species snapshot')
    if spec.get('species_count',len(species))!=len(species):raise ValueError('CHMT species count mismatch')
    def stats(rows):
        result={}
        for key in rows[0]:
            if not all(key in row for row in rows):continue
            values=[row[key] for row in rows]
            result[key]={'min':min(values),'max':max(values),'mean_face_count_weighted':sum(values)/len(values)}
        return result
    output.update(status='AVAILABLE',stage_time=stages.pop(),faces=len(faces),statistics=stats(list(faces.values())),species_statistics={name:stats(rows) for name,rows in species.items()})
    return output
