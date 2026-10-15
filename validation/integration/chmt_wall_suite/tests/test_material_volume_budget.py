"""Independent synthetic CSV fixtures test the inventory reader, not native execution."""
import csv
import math
import pytest
from test_suite import load


def material(cell, masses, densities=(1000., 2000.), porosity=.25):
    volume=math.fsum(m/rho for m,rho in zip(masses,densities))/(1-porosity)
    return dict(cell=cell,volume=volume,total_energy=100.,temperature=700.,porosity=porosity,
                mass_C0=masses[0],mass_C1=masses[1],mass_pore_S0=0.,mass_pore_S1=0.)


def fixture(case, initial, final):
    directory=case/'chmtOutput';directory.mkdir()
    def write(name,rows):
        with (directory/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for step,solid in enumerate((initial,final)):
        write(f'accepted-{step}.csv',[dict(accepted_steps=step,commit_sequence=step,time=step)])
        write(f'material-{step}.csv',solid)
        write(f'gas-{step}.csv',[dict(cell=0,volume=1,mass=1,total_energy=100,temperature=700)])
    return dict(cells=1,end_time=1,metric='chmt',parameters={'closed':True},
                material_card={'condensed_names':['C0','C1'],'condensed':[{'rho':1000.},{'rho':2000.}]})


def test_volume_budget_uses_all_densities_and_actual_cell_porosity(tmp_path):
    initial=[material(0,(30.,40.)),material(1,(10.,20.),porosity=.5)]
    final=[material(1,(9.,18.),porosity=.5),material(0,(27.,36.))]
    spec=fixture(tmp_path,initial,final);samples={}
    actual=load('metrics')._chmt_metrics(tmp_path,spec,samples)
    expected=(3/1000+4/2000)/.75+(1/1000+2/2000)/.5
    assert actual['solid_removed_mass_kg']==pytest.approx(10.)
    assert actual['mass_predicted_volume_loss_m3']==pytest.approx(expected)
    assert actual['recession_volume_relative_error']<1e-13
    assert samples['material_volume_budget']['reference_kind']=='MASS_DENSITY_POROSITY_CONSISTENCY'
    assert samples['material_volume_budget']['accepted_surface_packet_ledger']=='UNAVAILABLE'


def test_changing_porosity_is_not_reported_as_surface_mass_recession(tmp_path):
    initial=[material(0,(30.,40.),porosity=.25)]
    final=[material(0,(30.,40.),porosity=.5)]
    samples={};actual=load('metrics')._chmt_metrics(tmp_path,fixture(tmp_path,initial,final),samples)
    assert actual['solid_removed_mass_kg']==0
    assert actual['material_volume_change_m3']>0
    assert actual['recession_volume_absolute_error_m3']<1e-15
    assert samples['material_volume_budget']['porosity_changed'] is True


@pytest.mark.parametrize('fault',['duplicate_cell','missing_component','unknown_component','invalid_porosity','zero_density','negative_mass'])
def test_material_volume_budget_rejects_invalid_metadata(tmp_path,fault):
    initial=[material(0,(30.,40.)),material(1,(10.,20.))]
    final=[material(0,(27.,36.)),material(1,(9.,18.))]
    if fault=='duplicate_cell':final[1]['cell']=0
    if fault=='missing_component':
        for row in final:del row['mass_C1']
    if fault=='unknown_component':
        for row in final:row['mass_C2']=1.
    if fault=='invalid_porosity':final[0]['porosity']=1.
    if fault=='negative_mass':final[0]['mass_C1']=-1.
    spec=fixture(tmp_path,initial,final)
    if fault=='zero_density':spec['material_card']['condensed'][1]['rho']=0.
    with pytest.raises(ValueError):load('metrics')._chmt_metrics(tmp_path,spec,{})


def test_constant_rate_reference_includes_declared_porosity():
    observed={'removed_mass_kg':[0.,2.],'volume_loss_m3':[0.,.004],
              'gas_S0_mass_kg':[1.,3.],'gas_S1_mass_kg':[0.,0.]}
    metrics,_=load('metrics').surface_rate_history([0.,1.],observed,area=1.,mass_flux=2.,density=1000.,porosity=.5)
    assert metrics['surface_rate_volume_loss_m3_linf']==0.
