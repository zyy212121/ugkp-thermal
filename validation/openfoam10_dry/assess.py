"""Offline verification assertions; not linked into production runtime."""
from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
r=ROOT/'results'
load=lambda f:json.loads((r/f).read_text())
base=load('pde_summary.json');extra=load('extended_summary.json');surface=load('surface_corrected.json');mixture=load('mixture_summary.json')
f=lambda row,k:float(row[k])
order=lambda errors:[math.log(errors[i]/errors[i+1],2) for i in range(len(errors)-1)]
report={}
for mode in ['conduction','diffusion']:
 spatial=[a for a in extra if a['mode']==mode];temporal=[a for a in base if a['mode']==mode and a['N']=='400']
 so=order([f(a,'L2') for a in spatial]);to=order([f(a,'L2') for a in temporal]);assert all(1.97<x<2.03 for x in so),(mode,so);assert all(.97<x<1.03 for x in to),(mode,to)
 assert max(abs(f(a,'conservation_rel')) for a in spatial+temporal)<2e-11
 report[mode]={'spatial_L2':[f(a,'L2') for a in spatial],'spatial_orders':so,'temporal_orders':to,'max_normalized_budget':max(abs(f(a,'conservation_rel')) for a in spatial+temporal)}
darcy=[a for a in base if a['mode']=='darcy'];do=order([f(a,'relative_flux_error') for a in darcy]);assert all(.95<x<1.05 for x in do)
assert max(f(a,'core_flux_difference') for a in darcy)<1e-14
report['darcy']={'flux_relative_errors':[f(a,'relative_flux_error') for a in darcy],'orders':do,'max_discrete_divergence':max(f(a,'divergence') for a in darcy)}
mo=order([f(a,'L2_cellaverage') for a in mixture]);assert all(x>0 for x in mo) and all(.95<x<1.05 for x in mo[-2:]),mo
assert max(f(a,'zero_sum') for a in mixture)<1e-14 and max(f(a,'core_difference') for a in mixture)<1e-14
report['unequal_diffusion']={'cell_average_orders':mo,'asymptotic_order_grid':'N80/160/320; N20/40 retained as pre-asymptotic','scheme':'Upwind composition for common mass correction; expected first spatial order for unequal D','results':mixture}
for mode in ['isothermal','adiabatic']:
 rows=[a for a in extra if a['mode']==mode];o=order([f(a,'parent_error') for a in rows]);assert all(.99<x<1.03 for x in o)
 assert max(abs(f(a,'mass_residual'))/500.5 for a in rows)<1e-12
 assert max(abs(f(a,'energy_residual'))/350185000 for a in rows)<1e-12
 report[mode]={'mass_time_orders':o,'temperature_time_orders':order([f(a,'temperature_error') for a in rows]) if mode=='adiabatic' else None,'results':rows}
for a in surface:
 assert f(a,'wall_T_error')<1e-7 and f(a,'rate_error')<1e-11 and f(a,'recession_error')<1e-14
 assert abs(f(a,'energy_residual'))<1e-6 and abs(f(a,'element_residual'))<1e-12
report['surface']=surface
coupled=[a for a in extra if a['mode']=='coupled']
assert max(f(a,'mass_relative') for a in coupled)<1e-10 and max(f(a,'energy_relative') for a in coupled)<1e-10
assert min(f(a,'min_pore_mass') for a in coupled)>0
fields=['T','p','phi','solidDensity','Y0']
read=lambda p:np.genfromtxt(p,delimiter=',',names=True)
ref=read(r/'independent_coupled_N20.csv');time=[]
for dt in [4e-5,2e-5,1e-5]:
 a=read(ROOT/f'cases/coupled_N20_dt{dt:g}/profile.csv');time.append({'dt':dt,**{v:float(np.sqrt(np.mean((a[v]-ref[v])**2))) for v in fields}})
time_orders={v:order([a[v] for a in time]) for v in fields}
assert all(.97<o<1.04 for v in time_orders.values() for o in v),time_orders
fine=read(r/'independent_coupled_N640.csv');space=[]
for n in [20,40,80]:
 a=read(ROOT/f'cases/coupled_N{n}_dt1e-06/profile.csv');space.append({'N':n,**{v:float(np.sqrt(np.mean((a[v]-fine[v].reshape(n,-1).mean(1))**2))) for v in fields}})
assert all(space[i+1][v]<space[i][v] for i in range(2) for v in fields)
report['coupled']={'temporal_L2_against_independent_BDF':time,'temporal_orders':time_orders,'spatial_L2_against_restricted_independent_N640':space,'spatial_scope':'Monotonic convergence; coarse thermal layer pre-asymptotic, no universal spatial-order claim. Donor-density Darcy/advection is first order.','budgets':coupled}
threeD=load('threeD_summary.json')
report['three_dimensional']={}
for mode in ['conduction3d','diffusion3d']:
 rows=[a for a in threeD if a['mode']==mode];o=order([f(a,'L2') for a in rows]);assert all(1.93<x<2.03 for x in o),(mode,o)
 assert max(f(a,'conservation_rel') for a in rows)<2e-10
 report['three_dimensional'][mode]={'orders':o,'results':rows}
zero=load('zero_support_summary.json')
for a in zero:
 assert f(a,'initial_new_flux')<0 and f(a,'old_centered_counterexample')>0
 assert f(a,'min_inventory')>=0 and f(a,'species_residual')<1e-12 and f(a,'energy_relative')<1e-12
 assert f(a,'temperature_error')<1e-8 and f(a,'zero_sum_flux')<1e-12
report['three_species_zero_support']=zero
report['scope']='OF10 actual fvMesh/fvScalarMatrix/fvc::div; planar orthogonal unit-area 1D material slab plus true 3D orthogonal product-sine conduction/species diffusion. Constant-cp synthetic verification + separately pinned TACOT kinetics. Not experimental/material universality, gas-CFD, nonorthogonal, dynamic-mesh, or final GPU validation.'
report['status']='PASS within declared scope'
(r/'verification_assessment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
