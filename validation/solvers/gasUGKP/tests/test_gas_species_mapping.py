"""Read real ordered chemistry data and compare computed rates, not hashes."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
from test_gas_model_io import mixture
from test_gas_mechanism_io import mechanism
ROOT=SOURCE_ROOT
PROBE=r'''
#include "common/gasTransport/GasMechanismIO.H"
#include "common/chemistry/GasRates.H"
#include <iostream>
#include <iomanip>
#include <iterator>
// Compatibility only for the frozen pre-change red probe.
template<class Thermo> auto read(const std::string& text,const Thermo& thermo,
 const std::vector<std::string>& names,int)
 -> decltype(ugkwp::parseGasMechanismProperties<2>(text,thermo,"test",names))
 {return ugkwp::parseGasMechanismProperties<2>(text,thermo,"test",names);}
template<class Thermo> auto read(const std::string& text,const Thermo& thermo,
 const std::vector<std::string>&,long)
 -> decltype(ugkwp::parseGasMechanismProperties<2>(text,thermo,"test"))
 {return ugkwp::parseGasMechanismProperties<2>(text,thermo,"test");}
int main(){
 const std::string text((std::istreambuf_iterator<char>(std::cin)),{});
 try{
  const auto split=text.find("---");
  auto model=ugkwp::parseGasModelProperties(text.substr(0,split));
  const auto thermo=model.thermoView<2>();
  const auto loaded=read(text.substr(split+3),thermo,model.speciesNames,0);
  const auto view=loaded.mechanismView<2>();
  std::cerr<<"NAMES";for(const auto& name:loaded.speciesNames)std::cerr<<" "<<name;std::cerr<<"\n";
  std::cerr<<"BASIS";for(double value:loaded.stoichiometricBasis)std::cerr<<" "<<value;std::cerr<<"\n";
  std::cerr<<"STORAGE "<<model.speciesNames.size()<<" "<<model.species.size()<<" "<<model.coefficients.size()<<" "<<loaded.speciesNames.size()<<" "<<loaded.stoichiometricBasis.size()<<"\n";
  double c[]={2,7},production[]={-999,-999},progress[32];for(auto&v:progress)v=-999;
  ugkwp::ChemistryStatus status;
  if(!ugkwp::evaluateGasRates(300.,c,thermo,view,production,progress,status))return 3;
  std::cout<<std::setprecision(17)<<view.reactants[0].species<<" "<<view.products[0].species
   <<" "<<view.stoichiometricBasis[0]<<" "<<view.stoichiometricBasis[1]
   <<" "<<production[0]<<" "<<production[1]<<" "<<progress[0];
  if(view.efficiencyCount)std::cout<<" "<<view.efficiencies[0].species;
  std::cout<<" "<<ugkwp::speciesCp(0,300.,thermo)<<" "<<ugkwp::speciesCp(1,300.,thermo)<<"\n";
 }catch(const std::exception& e){std::cerr<<e.what();return 2;}
}
'''
@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    path=tmp_path_factory.mktemp('species_mapping');source=path/'probe.cpp';binary=path/'probe'
    source.write_text(PROBE)
    result=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return binary

def run(probe,model,mechanism_text):return subprocess.run([str(probe)],input=model+'---'+mechanism_text,text=True,capture_output=True)

def reordered(text):
    return text.replace('species (A B)', 'species (B A)').replace('reactants (0 1)', 'reactants (1 1)').replace('products (1 1.0)', 'products (0 1.0)').replace('stoichiometricBasis (-1 1)','stoichiometricBasis (1 -1)').replace('efficiencies (0 2)', 'efficiencies (1 2)')

@pytest.mark.parametrize('third_body',[False,True])
def test_t07_actual_name_mapping_preserves_rates_and_basis_rows(probe,third_body):
    text=mechanism()
    if third_body:text=text.replace('type elementary','type thirdBody').replace('efficiencies ()','efficiencies (0 2)')
    model=mixture().replace(' B { model linearCp; molarMass 0.028;', ' B { model linearCp; molarMass 0.028;').replace('coefficients (1040 0 0); }\n}', 'coefficients (1200 0 0); }\n}')
    ordinary=run(probe,model,text);changed=run(probe,model,reordered(text))
    assert ordinary.returncode==0,ordinary.stderr
    assert changed.returncode==0,changed.stderr
    assert changed.stdout==ordinary.stdout
    expected_rate=22 if third_body else 2
    values=[float(x) for x in changed.stdout.split()]
    assert values[:7]==[0,1,-1,1,-expected_rate,expected_rate,expected_rate]
    assert values[-2:]==[1040,1200]

@pytest.mark.parametrize('mutate',[
    lambda s:s.replace('species (A B)','species (A C)'),
    lambda s:s.replace('reactants (0 1)','reactants (2 1)'),
    lambda s:s.replace('highRate (1 0 0)','highRate (1 0)'),
])
def test_t07_missing_actual_species_or_rate_data_remains_error(probe,mutate):
    result=run(probe,mixture(),mutate(mechanism()))
    assert result.returncode==2,result.stdout+result.stderr


def test_missing_entropy_is_a_real_rate_error_not_an_invented_reference(probe):
    result=run(probe,mixture(),mechanism(reverse=True))
    assert result.returncode==3,result.stdout+result.stderr

def test_surplus_declared_species_and_basis_storage_is_retained_but_not_evolved(probe):
    model=mixture().replace('species (A B)','species (A B C)').replace('\ndiffusion', '\ndiffusion').replace('\n}\ndiffusion','\n C {model linearCp; molarMass .03; minTemperature 100; maxTemperature 4000; coefficients (1300 0 0); }\n}\ndiffusion').replace('coefficients (0.002 0.002)','coefficients (0.002 0.002 0.005)')
    text=mechanism().replace('species (A B)','species (A B C)').replace('stoichiometricBasis (-1 1)','stoichiometricBasis (-1 1 0 99)')
    result=run(probe,model,text)
    assert result.returncode==0,result.stderr
    assert [float(x) for x in result.stdout.split()][:7]==[0,1,-1,1,-2,2,2]
    assert 'STORAGE 3 3 9 3 4' in result.stderr


def test_malformed_unused_elementary_parameters_do_not_change_actual_rates(probe):
    ordinary=run(probe,mixture(),mechanism())
    unused=run(probe,mixture(),mechanism().replace('efficiencies ()','efficiencies (0)').replace('lowRate (0 0 0)','lowRate malformed').replace('troe (0 0 0 0)','troe malformed'))
    assert ordinary.returncode==unused.returncode==0,unused.stderr
    assert unused.stdout==ordinary.stdout
    selected=run(probe,mixture(),mechanism().replace('type elementary','type thirdBody').replace('efficiencies ()','efficiencies (0)'))
    assert selected.returncode==2 and 'pairs' in selected.stderr,selected.stderr


def test_repair_signed_diagnostic_index_keeps_actual_rate_output(probe):
    ordinary=run(probe,mixture(),mechanism())
    negative=run(probe,mixture(),mechanism().replace('sourceIndex 0','sourceIndex -1'))
    assert negative.returncode==0,negative.stderr
    assert negative.stdout==ordinary.stdout


def test_repair_permutation_crossing_compiled_boundary_preserves_all_basis_values(probe):
    model=mixture().replace('species (A B)','species (A B C)').replace('\n}\ndiffusion','\n C {model linearCp; molarMass .03; minTemperature 100; maxTemperature 4000; coefficients (1300 0 0); }\n}\ndiffusion').replace('coefficients (0.002 0.002)','coefficients (0.002 0.002 0.005)')
    text=mechanism().replace('species (A B)','species (C A B)').replace('reactants (0 1)','reactants (1 1)').replace('products (1 1.0)','products (2 1.0)').replace('stoichiometricBasis (-1 1)','stoichiometricBasis (99 -1 1 777)')
    result=run(probe,model,text)
    assert result.returncode==0,result.stderr
    assert [float(x) for x in result.stdout.split()][:7]==[0,1,-1,1,-2,2,2]
    assert 'STORAGE 3 3 9 3 4' in result.stderr
    assert 'BASIS -1 1 99 777\n' in result.stderr
    assert 'NAMES C A B\n' in result.stderr


@pytest.mark.parametrize('names,basis,expected',[
    ('C A B','90 91 -1 10 1 11 777','-1 10 1 11 90 91 777'),
    ('D A B','90 91 -1 10 1 11 777','-1 10 1 11 90 91 777'),
    ('A B C','-1 10 1 11','-1 10 1 11'),
])
def test_repair_multi_column_basis_retains_unused_rows_and_actual_missing_suffix_is_unused(probe,names,basis,expected):
    model=mixture().replace('species (A B)','species (A B C)').replace('\n}\ndiffusion','\n C {model linearCp; molarMass .03; minTemperature 100; maxTemperature 4000; coefficients (1300 0 0); }\n}\ndiffusion').replace('coefficients (0.002 0.002)','coefficients (0.002 0.002 0.005)')
    text=mechanism().replace('species (A B)',f'species ({names})').replace('independentRank 1','independentRank 2').replace('stoichiometricBasis (-1 1)',f'stoichiometricBasis ({basis})')
    if names!='A B C':text=text.replace('reactants (0 1)','reactants (1 1)').replace('products (1 1.0)','products (2 1.0)')
    result=run(probe,model,text)
    assert result.returncode==0,result.stderr
    assert f'BASIS {expected}\n' in result.stderr
    assert [float(x) for x in result.stdout.split()][4:7]==[-2,2,2]


def test_repair_missing_actually_mapped_basis_row_is_specific_error(probe):
    text=mechanism().replace('stoichiometricBasis (-1 1)','stoichiometricBasis (-1)')
    result=run(probe,mixture(),text)
    assert result.returncode==2 and 'stoichiometricBasis' in result.stderr
