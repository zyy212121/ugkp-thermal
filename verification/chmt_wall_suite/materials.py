"""Complete declared synthetic property cards; no empirical-material name switch."""
import copy

def cards():
    base={
      'classification':'controlled_synthetic','source':'Verification definition in this file; not measured material data.',
      'applicability':'100–3000 K synthetic constant-density / linear-cp model, finite-rate surface conversion, dry interface; not a physical ablator.',
      'units':{'rho':'kg/m3','cp0':'J/(kg K)','cp1':'J/(kg K2)','e0':'J/kg','conductivity':'W/(m K)','permeability':'m2','poreViscosity':'Pa s','activationEnergy':'J/mol','surface_A':'mol/(m2 s) for zero-order, zero temperature-power law; condensedNu/gasNu in kg/mol','gas_molar_mass':'kg/mol'},
      'condensed_names':['C0','C1'],
      'condensed':[dict(rho=1000.,cp0=1000.,cp1=0.,e0=0.,conductivity=1.,Tmin=100.,Tmax=3000.),dict(rho=1000.,cp0=1000.,cp1=0.,e0=0.,conductivity=1.,Tmin=100.,Tmax=3000.)],
      'gas_species':[dict(name='S0',molar_mass=.028,cp0=1040.,cp1=0.,e0=0.,Tmin=100.,Tmax=3000.),dict(name='S1',molar_mass=.028,cp0=1040.,cp1=0.,e0=0.,Tmin=100.,Tmax=3000.)],
      'gas_diffusivity_m2_s':.00002,'gas_viscosity_Pa_s':1.8e-5,'gas_conductivity_W_m_K':.03,
      'permeability_m2':0.,'pore_viscosity_Pa_s':1.8e-5,'emissivity':0.,'ambient_temperature_K':300.,'initial_porosity':0.,
      'bulk_reactions':[],
      'surface_reactions':[dict(A=.001/.028,temperaturePower=0.,activationEnergy=0.,condensedNu=[-.028,0.],gasNu=[.028,0.],order=[0.,0.],gasOrder=[0.,0.])],
      'pyrolysis_products':'S0: synthetic product (100% of surface-converted mass); S1: synthetic carrier. Equal elemental mass composition; no real chemistry implied.',
      'element_basis':'Synthetic element X with atom mass 0.028 kg/mol; each gas molecule contains one X and each condensed kg contains 1/0.028 mol X. Mass and this synthetic element are conserved; no real chemical species implied.',
      'gas_phase_reactions':[],'evaporation':'Disabled','melting':False,'film':'disabled','contact_resistances_m2K_W':[0.,0.],
      'energy_equation':'u_c=e0+cp0*T+0.5*cp1*T^2; e_g=e0+(cp0-Ru/W)*T+0.5*cp1*T^2. No additional latent heat source is added.'}
    second=copy.deepcopy(base)
    second['condensed']=[dict(rho=1600.,cp0=700.,cp1=.2,e0=-1e5,conductivity=2.,Tmin=100.,Tmax=3000.),dict(rho=1400.,cp0=850.,cp1=.1,e0=-8e4,conductivity=1.5,Tmin=100.,Tmax=3000.)]
    second['surface_reactions'][0]['A']=.0005/.028
    second['applicability']='100–3000 K second synthetic dense surrogate with variable heat capacity. Same equations and field schema as controlled_v1; no fit to experiments.'
    return {'controlled_v1':base,'controlled_v2':second}

def get(name):
    try:return copy.deepcopy(cards()[name])
    except KeyError:raise ValueError('unknown material card: '+name)
