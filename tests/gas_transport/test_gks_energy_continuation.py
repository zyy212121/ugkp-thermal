"""Actual retained GKS and authoritative energy helpers, no model substitutes."""
import pytest
from test_mixture_state import compile_probe
from test_authoritative_energy_sources import source_fixture

@pytest.mark.parametrize('case',['pivot','gamma','primitive'])
def test_gks_finite_complete_outputs(tmp_path,case):
    bodies={
'pivot':r'''double m[5][5]={},rhs[5],out[5]={-77,-77,-77,-77,-77};for(int k=0;k<5;++k){m[k][k]=k==4?1e-18:1;rhs[k]=m[k][k]*(k+1);}ck(ugkpgks::solveFiveByFive(m,rhs,out),"nonzero epsilon pivot refused");for(int k=0;k<5;++k)ck(out[k]==k+1,"incomplete solve");m[4][4]=0;ck(!ugkpgks::solveFiveByFive(m,rhs,out),"zero pivot fake success");''',
'gamma':r'''ugkpgks::LocalPrimitive p{1,2,3,4,5,6};double d[5]={1,2,3,4,5},out[5]={-77,-77,-77,-77,-77};ck(ugkpgks::analyticMomentCoefficients(p,2.,d,out),"evaluable gamma refused");for(int k=0;k<5;++k)ck(std::isfinite(out[k])&&out[k]!=-77,"coefficient not computed");''',
'primitive':r'''double q[5]={-2,-4,-6,-8,-40};ugkpgks::LocalPrimitive p{-77,-77,-77,-77,-77,-77};ck(ugkpgks::primitiveFromConservative(q,1.4,2.,p),"finite signed primitive refused");ck(p.rho==-2&&p.u==2&&p.v==3&&p.w==4&&p.p<0&&p.T>0,"primitive incomplete");'''}
    compile_probe(tmp_path,'#include "gasNumerics/FullSecondOrderGks.cuh"\nvoid ck(bool v,const char*m){if(!v){std::fprintf(stderr,"%s\\n",m);std::abort();}}\nint main(){'+bodies[case]+'}')

def test_energy_fallback_bound_is_distinct_from_output_finiteness(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){for(double value:{-1.1e300,-1e300,-.9e300,.9e300,1e300,1.1e300}){
ck(ugkwp::gasEnergyFinite(value),"finite output predicate changed");
const bool bounded=value>-1e300&&value<1e300;
ck(ugkwp::gasEnergyFiniteOr(value,0.)==(bounded?value:0.),"legacy fallback bound changed");}
ck(!ugkwp::gasEnergyFinite(std::numeric_limits<double>::infinity())&&!ugkwp::gasEnergyFinite(std::numeric_limits<double>::quiet_NaN()),"nonfinite output admitted");}
''')

def test_mixed_energy_small_positive_pressure_uses_native_storage(tmp_path):
    from test_mixture_transport import fixture
    f=fixture().replace('GasStateView<Real,Time,ugkwp::SstCoefficients>','GasStateView<Real,Time,ugkwp::SstCoefficients,double>').replace('s.rhoE=new Real[8]{};','s.rhoE=new double[8]{};').replace('s.rhoENext=new Real[8]{};','s.rhoENext=new double[8]{};')
    compile_probe(tmp_path,f+r'''
int main(){State s;initialise(s);auto* meta=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);meta[0].molarMass=meta[1].molarMass=Real(1e30);s.rho[0]=Real(1e-30);s.gasSpecies.rho[0]=s.gasSpecies.rho[2]=s.rho[0]*Real(.5);
double partial[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[2]};s.rhoE[0]=ugkwp::mixtureEnergy(partial,700.,s.gasSpecies.thermo);const double energy=s.rhoE[0];s.p[0]=-77;threadIdx.x=0;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[0]==0&&s.p[0]>0&&std::abs(s.p[0]/(s.rho[0]*Real(8.31446261815324)/Real(1e30)*Real(700))-1)<1e-14,"FP64 pressure lost positive equation-of-state result");ck(s.rhoE[0]==energy&&s.Tgas[0]==700&&s.gasSpecies.soundSpeed[0]>0,"underflow cache erased Energy authority");}
''')

def test_retained_gks_sides_and_full_flux_evaluate_gamma_above_five_thirds(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasNumerics/FullSecondOrderGks.cuh"
void ck(bool v,const char*m){if(!v){std::fprintf(stderr,"%s\n",m);std::abort();}}
int main(){ugkpgks::LocalPrimitive p{1,2,3,4,5,6};double grad[3][5]={};ugkpgks::SideFluxAccumulation a;ugkpgks::clearGksSideAccumulation(a);
ck(ugkpgks::accumulateGksSide(p,grad,2.,1,a),"finite GKS side refused");
ck(ugkpgks::accumulateGksSide(p,grad,2.,-1,a),"finite GKS opposite side refused");
auto f=ugkpgks::fullSecondOrderFluxStaged(p,p,grad,grad,2.,1.,.01,.7,.1,.01);
ck(f.valid,"finite full GKS result refused");for(int k=0;k<5;++k)ck(std::isfinite(f.flux[k])&&std::isfinite(f.molecularViscousFlux[k]),"GKS flux incomplete");
ck(std::abs(f.flux[0]-2)<1e-12&&std::abs(f.flux[1]-9)<1e-12,"equal-state GKS flux changed");}
''')
