"""Case metadata. A catalog entry is not execution evidence."""
import copy
FAMILIES=['lowRe','wallFunction','boundaryLayer_reactingSst']
def cases():
    c={
      'small_couette':dict(category=1,solver='gasUGKP',generator='couette',gate='hard',limits={'velocity_l2':.01,'velocity_linf':.025},models={'gas':'single ideal gas','turbulence':'laminar','geometry':'fixed'},reference='Exact transient impulsively started Couette series, volume averaged.'),
      'small_sod':dict(category=1,solver='gasUGKP',generator='sod',gate='hard',limits={'rho_l1':.03,'p_l1':.03,'u_l1':.03},models={'gas':'single ideal gas','turbulence':'inviscid','geometry':'fixed'},reference='Independent exact Euler Riemann solution, quadrature-averaged over cells.'),
      'small_cht_contact':dict(category=1,solver='CHMT',generator='contact',gate='hard',limits={'energy_relative_residual':2e-10,'initial_heat_rate_relative_error':.05},models={'gas':'frozen two-species','turbulence':'laminar','geometry':'fixed'},reference='UA=2 W/K, initial heat transfer=600 W, closed total formation-inclusive energy.'),
      'wall_constant_transport':dict(category=1,solver='gasUGKP',generator='couette_wall',gate='hard',limits={'velocity_l2':.02,'velocity_linf':.05},models={'gas':'frozen equal-thermo','turbulence':'laminar','geometry':'fixed','wall':'boundaryLayer/constantTransport'},reference='Exact transient Couette series; actual constant-transport production wall closure. Diffusion penetration length must exceed wall-model matching distance.'),
      'gas_species_wave':dict(category=2,solver='gasUGKP',generator='wave',gate='report_only',limits={'species_l1':.01,'species_linf':.02,'species_sum_error':1e-10},models={'gas':'mixtureFrozen','turbulence':'laminar','geometry':'periodic fixed'},reference='Exact advecting-diffusing sine, finite-volume cell average; two independently read species.'),
      'gas_chemistry_reactor':dict(category=2,solver='gasUGKP',generator='reactor',gate='report_only',limits={'history_error':1e-5},models={'gas':'H2/O2 10-species finite-rate','turbulence':'laminar','geometry':'fixed'},reference='Pinned Cantera 3.1.0 constant-volume reactor; does not itself validate spatial transport.'),
      'chmt_receding_slab':dict(category=2,solver='CHMT',generator='receding',gate='report_only',limits={'energy_relative_residual':1e-7,'mass_relative_residual':1e-8},models={'gas':'frozen product/carrier mixture; material finite-rate surface reaction','turbulence':'laminar','geometry':'CoupledRecession'},reference='Controlled synthetic material; independent whole-domain mass/energy and removed-volume accounting; no experimental claim.'),
      'flatplate_fixed':dict(category=3,solver='gasUGKP',generator='flatplate',gate='report_only',limits={},wall_variants=FAMILIES,models={'gas':'frozen equal-thermo mixture','turbulence':'SST','geometry':'fixed'},reference='Same-physics wall-resolved lowRe companion; optional NASA TMR correlation/SST reference with matching operating conditions.'),
      'flatplate_moving':dict(category=3,solver='CHMT',generator='flatplate_moving',gate='report_only',limits={},wall_variants=['lowRe','boundaryLayer_reactingSst'],models={'gas':'frozen mixture; material finite-rate surface reaction','turbulence':'SST','geometry':'CoupledRecession'},reference='Same-physics lowRe companion; controlled synthetic material, not an experiment.'),
      'mss7_fixed':dict(category=3,solver='gasUGKP',generator='mss7',gate='report_only',limits={},wall_variants=FAMILIES,models={'gas':'frozen equal-thermo mixture','turbulence':'SST','geometry':'MSS7 original 3D sector, slip lateral faces'},reference='Geometry-adapted controlled comparison; not the original axisymmetric thermal experiment.'),
      'mss7_moving':dict(category=3,solver='CHMT',generator='mss7_moving',gate='report_only',limits={},wall_variants=['lowRe','boundaryLayer_reactingSst'],models={'gas':'frozen mixture; material finite-rate surface reaction','turbulence':'SST','geometry':'MSS7 3D sector with CoupledRecession and slip lateral faces'},reference='Geometry-adapted controlled lowRe/highRe comparison; not original MSS7 experiment.'),
    }
    c['gas_reacting_wave']=copy.deepcopy(c['gas_species_wave'])
    c['gas_reacting_wave'].update(generator='reacting_wave',reference='Exact simultaneous advection, Fick diffusion and first-order A-to-B decay with equal molecular/caloric properties.')
    c['gas_reacting_wave']['models']['gas']='mixtureChemistry synthetic A -> B, 2/s'
    c['chmt_reacting_receding_slab']=copy.deepcopy(c['chmt_receding_slab'])
    c['chmt_reacting_receding_slab'].update(generator='reacting_receding',reference='Controlled joint gas species transport, finite-rate gas/material surface reactions, CoupledRecession; report-only inventory/refinement comparison.')
    c['chmt_reacting_receding_slab']['models']['gas']='mixtureChemistry synthetic S0 -> S1 plus C0 -> S0 surface conversion'
    c['chmt_laminar_reacting_wall']=copy.deepcopy(c['chmt_reacting_receding_slab'])
    c['chmt_laminar_reacting_wall'].update(generator='laminar_reacting_wall')
    c['chmt_laminar_reacting_wall']['models']['wall']='boundaryLayer/finiteRate, SST disabled'
    for key in ('chmt_receding_slab','chmt_reacting_receding_slab','chmt_laminar_reacting_wall'):
        if key in c:
            for quantity in ('removed_mass_kg','volume_loss_m3','gas_S0_mass_kg','gas_S1_mass_kg'):
                c[key]['limits']['surface_rate_'+quantity+'_linf']=.01
    for key,item in c.items():
        item['id']=key;item['species_count']=10 if item['generator']=='reactor' else 2
        item['wall_variants']=list(item.get('wall_variants',[]))
        item['yplus_policy']='User selects mesh height/grading; report measured y+ only. No automatic y+ tuning or assumed success.'
    return copy.deepcopy(c)
