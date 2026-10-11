from pathlib import Path
import json
HERE=Path(__file__).parent;OUT=HERE/'results'
contract=json.loads((HERE/'acceptance_contract.json').read_text())
owner=json.loads((OUT/'owner32.json').read_text());a=contract['owner32']
assert owner['time']==a['endpoint_s']
assert owner['outer_dt_cap']==a['outer_max_delta_t_s'] and owner['sst_prediction_target_courant']==a['prediction_target_courant']
assert owner['steps']<a['max_steps'] and owner['max_source_number']<=a['source_limit']
assert owner['minimum_k']>=1e-4
scale=max(abs(owner['source_impulse']),abs(owner['face_impulse']))
assert abs(owner['global_budget_residual'])<a['budget_relative_to_largest_impulse']*scale
ref=json.loads((OUT/'resolved_reference.json').read_text());a=contract['resolved_reference']
assert ref[-1]['nodes']>=a['minimum_nodes']
assert all(x['relative_update']<a['relative_iteration_update'] for x in ref)
assert all(max(x['normalized_discrete_residual'].values())<a['normalized_discrete_residual'] for x in ref)
for field in ['tau','heat']:
 print(field+' grid relative differences:',[ref[i-1][field]/ref[i][field]-1 for i in range(1,len(ref))])
comparison=json.loads((OUT/'couette_comparison.json').read_text());a=contract['couette_comparison']
for row in comparison:
 print('Physical comparison:',row)
 assert row['steady_relative_residual']<a['steady_normalized_residual']
blowing=json.loads((OUT/'blowing.json').read_text());a=contract['blowing']
for row in blowing:
 assert row['time']==a['endpoint_s'] and row['steps']<a['max_steps']
 assert row['max_source_number']<=a['source_limit'] and row['minimum_k']>0
 assert row['wall_k_flux']==row['wall_omega_flux']==0
 assert row['wall_mass_flux_residual']<1e-20 and row['wall_tangent_momentum_residual']<1e-18
 assert row['wall_energy_flux_residual']<a['max_wall_energy_residual_W']
 assert row['aux_species_balance_residual']<a['max_aux_species_balance_kg_m2_s']
 assert abs(row['global_k_budget_residual'])<a['k_budget_relative_to_largest_impulse']*max(abs(row['source_impulse']),abs(row['face_impulse']))
 if row['chemistry']:assert abs(row['aux_reaction_integral'])>1e-6
for nodes in a['nodes']:
 zero=next(r for r in blowing if r['nodes']==nodes and r['mass_flux']==0 and not r['chemistry'])
 seq=[next(r for r in blowing if r['nodes']==nodes and r['mass_flux']==mass and not r['chemistry']) for mass in [1e-9,1e-12,1e-15]]
 for key in ['tau','heat','owner_omega','final_k']:
  errors=[abs(r[key]/zero[key]-1) for r in seq]
  print('Zero-blow limit:',nodes,key,errors)
  assert errors[-1]<a['zero_blow_limit_relative'] and all(e2<e1 for e1,e2 in zip(errors,errors[1:]))
for row in blowing:
 scale=max(abs(row['source_impulse']),abs(row['face_impulse']))
 assert abs(row['audit_source_k']-row['source_impulse'])<1e-11*scale
 assert abs(row['audit_transport_k']-row['face_impulse'])<1e-11*scale
 assert abs(row['audit_constraint_k'])<1e-11*scale
 scale=max(abs(row['audit_source_omega']),abs(row['audit_transport_omega']),abs(row['audit_constraint_omega']))
 assert abs(row['omega_budget_residual'])<1e-11*scale
print('Blowing actual-owner checks:',blowing)
print('PASS: captured owner32 control-volume endpoint, conservative budget, and nonlinear/discrete convergence')
print('PHYSICS: resolved-reference discrepancies are measured above, not converted to a universal PASS')
print('NOT RUN: full 4608-cell gasUGKP endpoint and separate CHMT endpoint, CUDA runtime, GPU throughput')
