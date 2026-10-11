"""Audit the actual published species, including storage roundoff and zero time."""
import subprocess
import pytest
from test_gas_chemistry_adapter import SOURCE,ROOT


@pytest.mark.parametrize('real',['float','double'])
def test_budget_captures_actual_species_publication_and_no_energy_source(tmp_path,real):
    source=SOURCE.split('int main(){')[0].replace('TEST_REAL',real)
    source=source.replace('GasSpeciesState<Real,2> gasSpecies;', 'GasSpeciesState<Real,2> gasSpecies;GasBudgetAuditState gasBudgetAudit;')
    source+=r'''
int main(){Fixture f;double transport[8]{},initial[8]{},chemical[4]{};
auto&s=f.state;s.gasBudgetAudit={true,2,transport,initial,chemical};
const Real volume=Real(.3);const double before[]={double(f.partial[0])*double(volume),double(f.partial[2])*double(volume)};
const Real energy=f.energy[0];assert(advanceGasChemistryCell(s,0,.01,volume));
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
