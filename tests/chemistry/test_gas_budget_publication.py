"""Audit the actual published species, including storage roundoff and zero time."""
import subprocess
import pytest
from test_gas_chemistry_adapter import SOURCE,ROOT


def test_budget_captures_actual_species_publication_and_no_energy_source(tmp_path):
    source=SOURCE.split('int main(){')[0].replace('TEST_REAL','double')
    source=source.replace('GasSpeciesState<Real,2> gasSpecies;', 'GasSpeciesState<Real,2> gasSpecies;GasBudgetAuditState gasBudgetAudit;')
    source+=r'''
int main(){Fixture f;double transport[8]{},initial[8]{},chemical[4]{};
auto&s=f.state;s.gasBudgetAudit={true,2,transport,initial,chemical};
const Real volume=Real(.3);const double before[]={double(f.partial[0])*double(volume),double(f.partial[2])*double(volume)};
const auto energy=f.energy[0];assert(advanceGasChemistryCell(s,0,.01,volume));
for(int k=0;k<2;++k){const double expected=double(f.partial[2*k])*double(volume)-before[k];
 assert(std::abs(chemical[2*k]-expected)<1e-14);}
assert(chemical[0]<0&&chemical[2]>0&&f.energy[0]==energy);
const double old=chemical[0];assert(advanceGasChemistryCell(s,0,0.,volume));assert(chemical[0]==old);
assert(!advanceGasChemistryCell(s,0,.01,Real(0)));assert(chemical[0]==old);
for(double value:transport)assert(value==0);
}
'''
    path=tmp_path/'publication.cpp';path.write_text(source);binary=tmp_path/'publication'
    compiled=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(path),'-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_double_zero_time_chemistry_audits_exact_stored_inventory(tmp_path):
    source=SOURCE.split('int main(){')[0].replace('TEST_REAL','double')
    source=source.replace('GasStateView<Real,double,int>', 'GasStateView<Real,double,int,double>')
    source=source.replace('Real energy[2],volume[2]', 'double energy[2]; Real volume[2]')
    source+=r'''
int main(){
 Fixture f;f.partial[0]=1.1;f.partial[2]=.9;
 const double density[2]={double(f.partial[0]),double(f.partial[2])};
 const double ux=double(f.mx[0])/f.rho[0],uy=double(f.my[0])/f.rho[0],uz=double(f.mz[0])/f.rho[0];
 f.energy[0]=mixtureEnergy(density,700.,f.state.gasSpecies.thermo)
   +.5*double(f.rho[0])*(ux*ux+uy*uy+uz*uz);
 double saved[4];std::memcpy(saved,f.partial,sizeof saved);
 assert(advanceGasChemistryCell(f.state,0,0.,.3));
 assert(std::memcmp(saved,f.partial,sizeof saved)==0);
 assert(f.audit[0].speciesMassChange[0]==0.);
 assert(f.audit[0].speciesMassChange[1]==0.);
 assert(f.audit[0].massResidual==(density[0]*.3+density[1]*.3)-f.rho[0]*.3);
}
'''
    path=tmp_path/'zero.cpp';path.write_text(source);binary=tmp_path/'zero'
    compiled=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(path),'-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
