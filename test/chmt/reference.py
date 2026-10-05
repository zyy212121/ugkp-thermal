#!/usr/bin/env python3
"""Deterministic stdlib analytic references. This is NOT a CHMT solver."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ('case','variant','n','time','sample','quantity','value','scale','kind')


def load_cases():
    return json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))


def row_key(row):
    return (row['case'], row['variant'], int(row['n']), float(row['time']), str(row['sample']), row['quantity'])


def stefan_lambda(inverse_Stefan):
    if inverse_Stefan <= 0:
        raise ValueError('inverse_Stefan must be positive')
    lo, hi = 0.0, 1.0
    def f(x):
        return math.sqrt(math.pi)*x*math.exp(x*x)*math.erf(x)
    while f(hi) < inverse_Stefan:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if f(mid) < inverse_Stefan:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def stefan_state(t):
    p = next(c['parameters'] for c in load_cases()['cases'] if c['id']=='stefan')
    lam = stefan_lambda(p['inverse_Stefan'])
    alpha = 1/p['Pe']
    return lam, alpha, t+(p['initial_front']/(2*lam))**2/alpha


def stefan_front(t):
    lam, alpha, ta = stefan_state(t)
    return 2*lam*math.sqrt(alpha*ta)


def stefan_temperature_integral(x, t):
    lam, alpha, ta = stefan_state(t)
    d = 2*math.sqrt(alpha*ta)
    x = min(max(x, 0.0), d*lam)
    # Integral from zero, evaluated with expm1 to limit cancellation near zero.
    return x-(x*math.erf(x/d)+d/math.sqrt(math.pi)*math.expm1(-(x/d)**2))/math.erf(lam)


def stefan_cell(a, b, t):
    if not b > a:
        raise ValueError('positive cell width required')
    temp = (stefan_temperature_integral(b,t)-stefan_temperature_integral(a,t))/(b-a)
    liquid_fraction = max(0.0, min(b,stefan_front(t))-max(a,0.0))/(b-a)
    return temp, temp+liquid_fraction


def stefan_energy(t):
    return stefan_front(t)+stefan_temperature_integral(stefan_front(t),t)


def stefan_heat(t):
    lam, alpha, ta = stefan_state(t)
    _, _, t0 = stefan_state(0)
    return 2*math.sqrt(alpha)*(math.sqrt(ta)-math.sqrt(t0))/(math.sqrt(math.pi)*math.erf(lam))


def film_velocity(z, delta, mu, ub, tau, G):
    return ub+tau*z/mu+G*(z*z-2*delta*z)/(2*mu)


def film_profile(delta, mu, ub, tau, G):
    mean = ub+tau*delta/(2*mu)-G*delta**2/(3*mu)
    top = film_velocity(delta,delta,mu,ub,tau,G)
    bottom = tau-delta*G
    return {'u_mean':mean,'u_top':top,'tau_bottom':bottom,
            'phi':delta*tau*tau/mu-delta**2*tau*G/mu+delta**3*G*G/(3*mu),
            'boundary_work':tau*top-bottom*ub}


def sine_average(a, b, shift):
    return (math.cos(2*math.pi*(a-shift))-math.cos(2*math.pi*(b-shift)))/(2*math.pi*(b-a))


def mesh_node(x, t, amplitude=0.025):
    return x+amplitude*math.sin(2*math.pi*x)*math.sin(2*math.pi*t)


def ale_rho(x, y, t):
    return 1+0.2*math.sin(2*math.pi*(x+y-2*t))


def sinc(x):
    return math.sin(x)/x if x else 1.0


def ale_average(x0,x1,y0,y1,t,mean=1.0,amplitude=0.2,u=1.0,v=1.0):
    return mean+amplitude*math.sin(2*math.pi*((x0+x1+y0+y1)/2-(u+v)*t))*sinc(math.pi*(x1-x0))*sinc(math.pi*(y1-y0))


def overlap_integrals(old, density, new):
    if len(density) != len(old)-1 or old[0] != new[0] or old[-1] != new[-1]:
        raise ValueError('donor values and matching domain endpoints required')
    if any(b <= a for nodes in (old,new) for a,b in zip(nodes,nodes[1:])):
        raise ValueError('strictly increasing meshes required')
    return [math.fsum(q*max(0,min(b,d)-max(a,c)) for c,d,q in zip(old,old[1:],density)) for a,b in zip(new,new[1:])]


def generate(case_filter=None):
    for c in load_cases()['cases']:
        cid = c['id']
        if case_filter and cid != case_filter:
            continue
        if c['reference_status'] != 'ANALYTIC_AVAILABLE':
            continue
        p = c['parameters']
        def row(variant,n,t,sample,quantity,value,scale):
            return dict(zip(FIELDS,(cid,variant,n,t,str(sample),quantity,value,scale,'ANALYTIC_REFERENCE')))
        if cid == 'stefan':
            for n in c['n']:
                for t in c['times']:
                    for i in range(n):
                        T,H = stefan_cell(i/n,(i+1)/n,t)
                        yield row('melting',n,t,i,'T_bar',T,1)
                        yield row('melting',n,t,i,'h_bar',H,2)
                    yield row('melting',n,t,'global','front',stefan_front(t),stefan_front(t))
                    yield row('melting',n,t,'global','wall_heat',stefan_heat(t),stefan_heat(t) if t>0 else stefan_energy(0))
        elif cid == 'film_profile':
            d,mu = p['delta'],p['mu']
            for v in c['variants']:
                values = film_profile(d,mu,v['ub'],v['tau'],v['G'])
                uscale = max(abs(v['ub']),abs(v['tau']*d/mu),abs(v['G']*d*d/mu),1e-12)
                for k,val in values.items():
                    scale = uscale if k.startswith('u_') else max(abs(v['tau']),abs(v['G']*d),1e-12) if k=='tau_bottom' else max(abs(v['tau'])*uscale,abs(v['G'])*d*uscale,1e-12)
                    yield row(v['id'],0,0,'local',k,val,scale)
                for i,z in enumerate(c['z_over_delta']):
                    yield row(v['id'],0,0,i,'u_z',film_velocity(z*d,d,mu,v['ub'],v['tau'],v['G']),uscale)
                yield row(v['id'],0,0,'local','mass_flux',p['rho']*d*values['u_mean'],p['rho']*d*uscale)
                yield row(v['id'],0,0,'local','enthalpy_flux',p['rho']*d*p['h']*values['u_mean'],p['rho']*d*p['h']*uscale)
        elif cid in ('film_plug','film_shear'):
            fp = film_profile(p['delta'],p['mu'],p['ub'],p['tau'],p['G'])
            H0 = p['rho']*p['delta']*p['cp']*p['T_mean']
            amp = p['rho']*p['delta']*p['cp']*p['T_amplitude']
            for n in c['n']:
                for t in c['times']:
                    for i in range(n):
                        H = H0+amp*sine_average(i/n,(i+1)/n,fp['u_mean']*t)+fp['phi']*t
                        yield row('periodic',n,t,i,'H_bar',H,amp)
                    yield row('periodic',n,t,'global','mass',p['rho']*p['delta'],p['rho']*p['delta'])
                    yield row('periodic',n,t,'global','boundary_work',fp['boundary_work']*t,H0+fp['phi']*t)
        elif cid == 'film_pressure':
            for i,pressure in enumerate(p['pressure']):
                H = p['rho']*p['delta']*p['cp']*p['T']+p['delta']*(pressure-p['p_ref'])
                for k,val,scale in [('H',H,1),('T',p['T'],p['T']),('internal_energy',H-pressure*p['delta'],1)]:
                    yield row('pulse',0,i,'local',k,val,scale)
        elif cid == 'film_phase_energy':
            J,hl,hg = p['J'],p['h_liquid'],p['h_gas']
            kl = sum(x*x for x in p['u_liquid'])/2
            kg = sum(x*x for x in p['u_gas'])/2
            Fg = J*(hg+kg)+p['p']*p['w_normal']-p['viscous_work_gas']+p['conductive_flux_gas']
            Q = J*(hl+kl)+p['p']*p['w_normal']-p['viscous_work_liquid']-Fg
            vals = {'F_gas':Fg,'Q_liquid':Q,'enthalpy_rhs':Q-J*hl,
                    'rho_liquid':J/(p['u_liquid'][2]-p['w_normal']),
                    'rho_gas':J/(p['u_gas'][2]-p['w_normal'])}
            for k,val in vals.items():
                yield row('jump',0,0,'local',k,val,max(abs(val),1))
        elif cid in ('ale_free','ale_wave'):
            for n in c['n']:
                for t in c['times']:
                    nodes = [mesh_node(i/n,t,p['mesh_amplitude']) for i in range(n+1)]
                    for j in range(n):
                        for i in range(n):
                            V = (nodes[i+1]-nodes[i])*(nodes[j+1]-nodes[j])
                            rho = ale_average(nodes[i],nodes[i+1],nodes[j],nodes[j+1],t,p['rho_mean'],p['rho_amplitude'],p['u'],p['v'])
                            mass = rho*V
                            amp = p['rho_amplitude'] or p['rho_mean']
                            values = {'volume':(V,V),'mass':(mass,amp*V),'momentum_x':(mass*p['u'],amp*V*abs(p['u'])),
                                      'momentum_y':(mass*p['v'],amp*V*abs(p['v'])),
                                      'energy':((p['p']/(p['gamma']-1)+rho*(p['u']**2+p['v']**2)/2)*V,amp*V*(p['u']**2+p['v']**2)/2)}
                            values.update({f'species_{k}':(mass*y,amp*V*y) for k,y in enumerate(p['Y'])})
                            for k,(val,scale) in values.items():
                                yield row('moving',n,t,f'{i}:{j}',k,val,scale)
        elif cid == 'remap':
            for n in c['n']:
                old = [i/n for i in range(n+1)]
                new = [x+p['mesh_amplitude']*math.sin(2*math.pi*x) for x in old]
                for variant in c['variants']:
                    rho = [1 if variant=='uniform' or i<int(n*p['discontinuity_fraction']) else 2 for i in range(n)]
                    fields = {'mass':rho,'momentum_x':[r*p['u'] for r in rho],
                              'energy':[p['p']/(p['gamma']-1)+r*p['u']**2/2 for r in rho]}
                    fields.update({f'species_{k}':[r*y for r in rho] for k,y in enumerate(p['Y'])})
                    for k,vals in fields.items():
                        for i,q in enumerate(overlap_integrals(old,vals,new)):
                            yield row(variant,n,0,i,k,q,max(abs(v) for v in vals)*(new[i+1]-new[i]))


def write_rows(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            r = dict(r)
            for key in ('time','value','scale'):
                r[key] = format(float(r[key]),'.17g')
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    known = {c['id']:c for c in load_cases()['cases']}
    if args.case and (args.case not in known or known[args.case]['reference_status']!='ANALYTIC_AVAILABLE'):
        parser.error('case is unknown or requires external data; no fabricated reference is produced')
    rows = list(generate(args.case))
    write_rows(args.output,rows)
    print(json.dumps({'artifact_kind':'ANALYTIC_REFERENCE','rows':len(rows),'solver_validation_status':'UNVERIFIED'}))


if __name__ == '__main__':
    main()
