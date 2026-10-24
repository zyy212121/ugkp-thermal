"""Prescribed-T TACOT kinetic verification, not experimental/energy/PDE validation.

Run with --reference-dir /path/to/pr24-benchmark-sources (or
TACOT_REFERENCE_DIR), containing the frozen source pack and its two extracted
352-row reference CSVs. See provenance.json for pinned source URLs/checksums.
DRY_CORE_ROOT selects the repository whose actual core headers are compiled;
default is this repository. Generated data/binary are in ../results/benchmark.
Requires C++17 compiler, Python 3.10+, numpy, scipy, matplotlib.
No parameter fitting or time-shifted acceptance is performed. The diagnostic
shift remains labelled as a possible reference-output sampling difference.
"""
from pathlib import Path
import subprocess,json,os,argparse,hashlib,shlex
import numpy as np
from scipy.integrate import quad
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir', default=os.environ.get('TACOT_REFERENCE_DIR'),
                    help='Frozen external source pack containing reference CSVs')
parser.add_argument('--output-dir',type=Path,default=HERE.parent/'results'/'benchmark')
args=parser.parse_args()
if not args.reference_dir:
    parser.error('--reference-dir or TACOT_REFERENCE_DIR is required')
S=Path(args.reference_dir).resolve()
B=args.output_dir.resolve(); B.mkdir(parents=True,exist_ok=True)
ROOT=Path(os.environ.get('DRY_CORE_ROOT',str(HERE.parents[2]))).resolve()
# Offline provenance checks, not production runtime admission policy.
provenance=json.loads((HERE/'provenance.json').read_text())
for item in provenance['required_inputs']:
    path=S/item['path']
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=item['sha256']:
        raise ValueError('Reference/source identity mismatch: '+str(path))
command=shlex.split(os.environ.get('CXX','g++'))+[
    '-std=c++17','-O2','-Wall','-Wextra','-Werror',
    '-I'+str(ROOT/'applications'/'CHMT'),str(HERE/'tacot_core.cpp'),
    '-o',str(B/'tacot_core')]
subprocess.run(command,check=True)
(B/'build_command.json').write_text(json.dumps(command,indent=2)+'\n')
cap=np.array([30.,22.8,7.2]); A=np.array([12000.,4.97777e8,4.97777e8])
E=np.array([71130.89,169975.,169975.]); R=8.314471469
onset=(np.array([333.3,555.6,555.6])-300)/3.1
times=np.arange(352.)
# Cubic transformed solution: z=1-xi, d(z^-2)/dt=2*a(t).
# Adaptive Gauss-Kronrod quadrature is independent of core mass-action/Euler code.
remaining=np.empty((352,3)); pi=np.zeros(352); quadrature_error=0.
for i,t in enumerate(times):
    for j in range(3):
        f=lambda x: A[j]*np.exp(-E[j]/(R*(300+3.1*x)))
        I,err=quad(f,onset[j],t,epsabs=1e-11,epsrel=1e-12) if t>onset[j] else (0.,0.)
        quadrature_error=max(quadrature_error,err)
        z=1/np.sqrt(1+2*I); remaining[i,j]=cap[j]*z
        if t>=onset[j]: pi[i]+=cap[j]*f(t)*z**3
rho=220+remaining.sum(axis=1)
mean=np.r_[0.,rho[:-1]-rho[1:]]
np.savetxt(B/'independent_quadrature.csv',np.c_[times,300+3.1*times,rho,pi,mean,remaining],delimiter=',',header='time_s,temperature_K,rho_s,pi_instant,pi_1s_average,volatile1,volatile2,volatile3',comments='')
summary={'core_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'quadrature_max_reported_integral_absolute_error':quadrature_error,'independent_final_density':rho[-1],'refinement':[],'external_reference_comparison':{}}
for dt in [1.,.5,.25,.125,.0625,.03125,.015625]:
    dest=B/f'core_dt_{dt:g}.csv'
    p=subprocess.run([str(B/'tacot_core'),str(dt)],capture_output=True,text=True,check=True)
    dest.write_text(p.stdout); (B/f'core_dt_{dt:g}.log').write_text(p.stderr)
    d=np.genfromtxt(dest,delimiter=',',names=True)
    summary['refinement'].append({'dt_s':dt,'max_abs_rho_error_kg_m3':float(np.max(abs(d['rho_s']-rho))),'max_abs_pi_instant_error_kg_m3_s':float(np.max(abs(d['pi_instant']-pi))),'rho_final':float(d['rho_s'][-1]),'diagnostics':p.stderr.strip()})
fine=d
for name in ['PATO','FIAT']:
    ref=np.genfromtxt(S/f'zeroD_{name}_reference.csv',delimiter=',',names=True)
    rr=ref['solid_bulk_density_kg_m3']; pp=ref['pyrolysis_mass_source_kg_m3_s']
    summary['external_reference_comparison'][name]={
      'max_abs_density_vs_quadrature_kg_m3':float(np.max(abs(rr-rho))),
      'max_density_error_normalized_initial_280':float(np.max(abs(rr-rho))/280),
      'max_density_error_normalized_volatile_60':float(np.max(abs(rr-rho))/60),
      'max_pi_vs_instant_kg_m3_s':float(np.max(abs(pp-pi))),
      'max_pi_vs_one_second_average_kg_m3_s':float(np.max(abs(pp-mean))),
      'max_pi_vs_average_normalized_peak':float(np.max(abs(pp-mean))/max(mean)),
      'reference_final_density':float(rr[-1]),'reference_peak_pi':float(max(pp)),
      'max_abs_density_fine_core_vs_reference':float(np.max(abs(rr-fine['rho_s']))) }
    np.savetxt(B/f'comparison_{name}.csv',np.c_[times,rr,rho,fine['rho_s'],pp,pi,mean,fine['pi_instant'],fine['pi_1s_average']],delimiter=',',header='time_s,reference_rho,quadrature_rho,core_fine_rho,reference_pi,quadrature_pi_instant,quadrature_pi_average,core_pi_instant,core_pi_average',comments='')
# A diagnostic replica of published fixed-dt temperature sampling; still actual core.
p=subprocess.run([str(B/'tacot_core'),'1','1','1'],capture_output=True,text=True,check=True)
(B/'core_PATO_sampling_dt1.csv').write_text(p.stdout);(B/'core_PATO_sampling_dt1.log').write_text(p.stderr)
x=np.genfromtxt(B/'core_PATO_sampling_dt1.csv',delimiter=',',names=True)
r=np.genfromtxt(S/'zeroD_PATO_reference.csv',delimiter=',',names=True)
summary['PATO_dt1_end_temperature_diagnostic']={'max_abs_rho_difference':float(np.max(abs(x['rho_s']-r['solid_bulk_density_kg_m3']))),'max_abs_pi_average_difference':float(np.max(abs(x['pi_1s_average']-r['pyrolysis_mass_source_kg_m3_s'])))}
# Extensive storage scaling test, same bulk chemistry for another cell volume.
p=subprocess.run([str(B/'tacot_core'),'.015625','0.000001'],capture_output=True,text=True,check=True)
(B/'core_fine_volume1e-6.csv').write_text(p.stdout);(B/'core_fine_volume1e-6.log').write_text(p.stderr)
y=np.genfromtxt(B/'core_fine_volume1e-6.csv',delimiter=',',names=True)
summary['volume_scaling_max_density_error']=float(max(abs(y['rho_s']-fine['rho_s'])))
summary['gas_yield_max_absolute_error']=float(max(np.max(abs(fine[n]-np.sum((cap-remaining)*weights,axis=1))) for n,weights in zip(['H2O','A1','A1OH','CO2','CO','CH4','H2'],np.array([[.69,.02,.29,0,0,0,0],[0,0,0,.09,.33,.58,0],[0,0,0,0,0,0,1]]).T)))
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

# Yield map check against actually reacted core pools, not a time-discretization mismatch.
yields=np.array([[.69,.02,.29,0,0,0,0],[0,0,0,.09,.33,.58,0],[0,0,0,0,0,0,1]])
reacted=cap-np.column_stack([fine[f'volatile{j+1}'] for j in range(3)])
expected=reacted@yields
summary['gas_yield_discretization_error_vs_quadrature']=summary.pop('gas_yield_max_absolute_error')
summary['gas_yield_identity_max_error_kg_m3']=float(np.max(abs(expected-np.column_stack([fine[n] for n in ['H2O','A1','A1OH','CO2','CO','CH4','H2']]))))
errs=np.array([r['max_abs_rho_error_kg_m3'] for r in summary['refinement']])
summary['density_refinement_ratios']=(errs[:-1]/errs[1:]).tolist()
coarse=np.genfromtxt(B/'core_dt_1.csv',delimiter=',',names=True)
r=np.genfromtxt(S/'zeroD_PATO_reference.csv',delimiter=',',names=True)
summary['PATO_sampling_diagnostic_NOT_acceptance_shift']={
 'unshifted_dt1_density_max_error':float(max(abs(coarse['rho_s']-r['solid_bulk_density_kg_m3']))),
 'core_t_vs_reference_t_plus_1_density_max_error':float(max(abs(coarse['rho_s'][:-1]-r['solid_bulk_density_kg_m3'][1:]))),
 'reference_rho_discrete_derivative_vs_pi_max_error':float(max(abs(-np.diff(r['solid_bulk_density_kg_m3'])-r['pyrolysis_mass_source_kg_m3_s'][1:])))}
# FIAT has a different rounded input table; compute its own continuum exactly.
fiat_rho=[]
for t in times:
 rem=[]
 for c,a,theta,threshold in [(30,12000,8555.55,333.33),(30,4.48e9/9,20444.44,555.55)]:
  start=(threshold-300)/3.1
  integ=quad(lambda x:a*np.exp(-theta/(300+3.1*x)),start,t,epsabs=1e-11,epsrel=1e-12)[0] if t>start else 0.
  rem.append(c/np.sqrt(1+2*integ))
 fiat_rho.append(220+sum(rem))
summary['FIAT_distinct_input_continuum_max_difference_from_PATO_kg_m3']=float(max(abs(np.array(fiat_rho)-rho)))
np.savetxt(B/'FIAT_distinct_input_continuum.csv',np.c_[times,fiat_rho],delimiter=',',header='time_s,rho_s',comments='')
assert all(1.95<x<2.1 for x in summary['density_refinement_ratios'])
assert summary['volume_scaling_max_density_error']<1e-10
assert summary['gas_yield_identity_max_error_kg_m3']<1e-10
assert errs[-1]<.002
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Plot raw time-aligned external targets; no fitted or shifted curves.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax[0,0].plot(times,rho,label='independent cubic quadrature',color='black')
ax[0,0].plot(times,fine['rho_s'],'--',label='actual core dt=1/64 s')
ax[0,1].plot(times,mean,label='quadrature 1 s mean',color='black')
ax[0,1].plot(times,fine['pi_1s_average'],'--',label='actual core 1 s mean')
for name in ['PATO','FIAT']:
 r=np.genfromtxt(S/f'zeroD_{name}_reference.csv',delimiter=',',names=True)
 ax[0,0].plot(times,r['solid_bulk_density_kg_m3'],':',label=name+' stored reference')
 ax[0,1].plot(times,r['pyrolysis_mass_source_kg_m3_s'],':',label=name+' stored reference')
 ax[1,0].plot(times,r['solid_bulk_density_kg_m3']-rho,label=name+' minus quadrature')
ax[1,1].loglog([r['dt_s'] for r in summary['refinement']],errs,'o-',label='core density max error')
for a in ax.flat:a.grid(True,alpha=.3);a.legend(fontsize=8)
ax[0,0].set(xlabel='t [s]',ylabel='solid bulk density [kg/m3]')
ax[0,1].set(xlabel='t [s]',ylabel='pyrolysis mass source [kg/(m3 s)]')
ax[1,0].set(xlabel='t [s]',ylabel='density difference [kg/m3]')
ax[1,1].set(xlabel='dt [s]',ylabel='max density error [kg/m3]')
fig.suptitle('TACOT prescribed-T kinetic benchmark: code references, not experimental validation')
fig.savefig(B/'tacot_comparison.png',dpi=160)
print(json.dumps(summary,indent=2))
print('Offline convergence/yield/scaling checks PASS')
