"""The wall operator uses FP64 independently of stored gas/table precision."""
import json
import subprocess
from test_wall_model import BASE, ROOT


def test_float_state_strict_coupled_wall_and_published_budget(tmp_path):
    body = r'''
#include <type_traits>
#include <iomanip>
int main(){
 Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.model.mode=GasMode::MixtureChemistry;
 m.coef[2]=100000;m.rx[0].highRate.preExponential=20;
 m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 float y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
 WallModelConfig<float> c;c.enableSst=true;c.nodes=48;c.maxIterations=100;
 WallWorkspace<float,2,48> w;WallOutput<float,2> out;WallStatus status;
 static_assert(sizeof(w.state[0][0])==sizeof(double),"wall arithmetic must not follow outer FP32");
 assert(evaluateWallModel(m.in,c,w,out,status));assert(status.residual<1.01e-8);
 std::cout<<std::setprecision(17)<<"{\"residual\":"<<status.residual<<",\"balance\":[";
 for(int j=0;j<2;++j){
  // Recompute after storing actual public values, not from an FP64-only audit.
  const float matching=out.matchingSpeciesFlux[j],wall=out.wallSpeciesFlux[j],reaction=out.reactionIntegral[j];
  const double balance=double(matching)-double(wall)-double(reaction);
  assert(std::abs(balance)<1e-8);assert(std::abs(out.speciesBalanceResidual[j])<1e-8);
  assert(std::abs(double(out.speciesBalanceResidual[j])-balance)<1e-15);
  if(j)std::cout<<",";
  std::cout<<balance;
 }
 std::cout<<"],\"q\":"<<out.conductiveHeatFlux<<",\"tau\":"<<out.traction[0]<<",\"omega\":"<<out.ownerOmega<<"}\n";
 const auto saved=out;c.relativeTolerance=1e-20f;c.absoluteTolerance=0;
 assert(!evaluateWallModel(m.in,c,w,out,status));assert(status.residual>c.relativeTolerance);
 assert(out.conductiveHeatFlux==saved.conductiveHeatFlux&&out.reactionIntegral[0]==saved.reactionIntegral[0]);
 c.model=BoundaryLayerModel::ConstantTransport;c.enableSst=false;c.relativeTolerance=1e-8f;
 m.in.model.mode=GasMode::MixtureFrozen;m.in.massFlux[0]=m.in.massFlux[1]=0;m.in.model.conductivity=1e38f;
 assert(!evaluateWallModel(m.in,c,w,out,status));assert(status.code==WallCode::InadmissibleState);
 assert(out.conductiveHeatFlux==saved.conductiveHeatFlux);

}
'''
    results = {}
    for bits in (32, 64):
        source=tmp_path/f'internal_{bits}.cpp'
        source.write_text(BASE.replace('double','float')+body)
        exe=source.with_suffix('')
        build=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-DUGKWP_GPU_REAL_BITS={bits}','-I',str(ROOT),'-I',str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
        assert build.returncode==0,build.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        results[bits]=json.loads(run.stdout)
    assert results[32] == results[64]
    print(json.dumps(results, sort_keys=True))


def test_borrowed_float_tables_match_explicit_double_copy(tmp_path):
    source = tmp_path/'borrowed.cpp'
    source.write_text(r'''
#include "common/gasWall/WallModel.H"
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace ugkwp;
int main(){
 GeneratedH2O2<float> original;auto tf=original.thermoView();auto mf=original.mechanismView();
 SpeciesThermoData<double> species[10];double coefficients[140],elements[40],basis[60];
 GasReactionData<double> reactions[29];GasStoichTerm<double> reactants[55],products[50];GasColliderEfficiency<double> efficiencies[19];
 for(int i=0;i<10;++i){const auto& a=original.species[i];species[i]={a.model,a.coefficientOffset,a.molarMass,a.minTemperature,a.midTemperature,a.maxTemperature,a.referencePressure,a.referenceTemperature,a.referenceEntropy,a.hasEntropyReference};}
 for(int i=0;i<140;++i)coefficients[i]=original.coefficients[i];
 for(int i=0;i<40;++i)elements[i]=original.elementComposition[i];
 for(int i=0;i<60;++i)basis[i]=mf.stoichiometricBasis[i];
 for(int i=0;i<29;++i){const auto& a=original.reactions[i];reactions[i]={a.type,a.reversible,a.duplicate,a.sourceIndex,a.reactantOffset,a.reactantCount,a.productOffset,a.productCount,a.efficiencyOffset,a.efficiencyCount,{a.highRate.preExponential,a.highRate.temperatureExponent,a.highRate.activationEnergy},{a.lowRate.preExponential,a.lowRate.temperatureExponent,a.lowRate.activationEnergy},a.defaultEfficiency,{a.troe.alpha,a.troe.T3,a.troe.T1,a.troe.T2,a.troe.hasT2}};}
 for(int i=0;i<55;++i)reactants[i]={original.reactants[i].species,original.reactants[i].coefficient};
 for(int i=0;i<50;++i)products[i]={original.products[i].species,original.products[i].coefficient};
 for(int i=0;i<19;++i)efficiencies[i]={original.efficiencies[i].species,original.efficiencies[i].efficiency};
 SpeciesThermoView<double,10> td{species,coefficients,140,4,elements,tf.speciesOrderHash,tf.thermoHash};
 GasMechanismView<double,10> md{reactions,29,reactants,55,products,50,efficiencies,19,basis,6,double(mf.referencePressure),mf.speciesOrderHash,mf.mechanismHash};
 int checked=0;
 for(double T:{700.,999.99999,1000.,1000.00001,1500.,3000.}){
  for(int i=0;i<10;++i){assert(speciesCp(i,T,tf)==speciesCp(i,T,td));assert(speciesH(i,T,tf)==speciesH(i,T,td));assert(speciesGibbsStandard(i,T,tf)==speciesGibbsStandard(i,T,td));
   const double theta=T-500.;assert(ugkwp::gaswall::detail::speciesEnthalpyIncrement(i,500.,theta,tf)==ugkwp::gaswall::detail::speciesEnthalpyIncrement(i,500.,theta,td));}
  for(int zero:{-1,0,6}){double c[10];for(int i=0;i<10;++i)c[i]=i==zero?0.:.02*(i+1);
   for(int i=0;i<29;++i){double mixed,copy;assert(gasChemistryDetail::progressRate(i,T,c,tf,mf,mixed));assert(gasChemistryDetail::progressRate(i,T,c,td,md,copy));assert(mixed==copy);++checked;}}
 }
 // Independent SST arithmetic ignores the global GpuReal alias, while legacy
 // wrappers continue to call exactly the same scalar-specific formula.
 auto typed=defaultSstCoefficientsT<double>();const double f=sstF1(.5,200.,3e-5,.004,.01,typed);
 assert(std::isfinite(f));auto legacy=defaultSstCoefficients();
 assert(sstF1(GpuReal(.5),GpuReal(200),GpuReal(3e-5),GpuReal(.004),GpuReal(.01),legacy)==sstF1<GpuReal>(GpuReal(.5),GpuReal(200),GpuReal(3e-5),GpuReal(.004),GpuReal(.01),legacy));
 std::cout<<checked<<" mixed-table reaction checks\n";
}
''')
    for bits in (32,64):
        exe=tmp_path/f'borrowed_{bits}'
        run=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror',f'-DUGKWP_GPU_REAL_BITS={bits}','-I',str(ROOT),'-I',str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr


def test_double_wall_ignores_global_precision_alias(tmp_path):
    body=r'''
#include <iomanip>
int main(){Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.model.mode=GasMode::MixtureChemistry;
 m.coef[2]=100000;m.rx[0].highRate.preExponential=20;m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;double y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
 WallModelConfig<double> c;c.enableSst=true;c.nodes=48;c.maxIterations=100;
 WallWorkspace<double,2,48> w;WallOutput<double,2> out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,out,status));
 assert(std::abs(out.speciesBalanceResidual[0])<1e-8&&std::abs(out.speciesBalanceResidual[1])<1e-8);
 std::cout<<std::setprecision(17)<<out.conductiveHeatFlux<<" "<<out.traction[0]<<" "<<out.ownerOmega<<" "<<status.residual<<"\n";
}
'''
    results=[]
    for bits in (32,64):
        source=tmp_path/f'double_{bits}.cpp';source.write_text(BASE+body);exe=source.with_suffix('')
        run=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-DUGKWP_GPU_REAL_BITS={bits}','-I',str(ROOT),'-I',str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        results.append(run.stdout)
    assert results[0]==results[1]
    print(results[0])
