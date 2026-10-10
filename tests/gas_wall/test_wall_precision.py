"""Native GpuReal precision contracts, including explicitly reported noise floors."""
import json
import subprocess
from test_wall_model import BASE, ROOT


def test_native_fp32_fp64_residual_and_explicit_tolerances(tmp_path):
    body = r'''
#include <iomanip>
int main(){
 Model m;WallModelConfig<GpuReal> c;WallWorkspace<GpuReal,2,48> w;WallOutput<GpuReal,2> o;WallStatus s;
 std::cout<<std::setprecision(17);std::cout<<"{";
 c.model=BoundaryLayerModel::ConstantTransport;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(s.residual==0);
 std::cout<<"\"constant_heat\":"<<o.conductiveHeatFlux;
 c.model=BoundaryLayerModel::ReactingSst;c.nodes=32;
 m.in.matching.velocity[0]=0;m.in.temperature=m.in.matching.temperature=500;
 m.in.model.mode=GasMode::MixtureChemistry;m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e-4;
 const bool reactionDefault=evaluateWallModel(m.in,c,w,o,s);
 std::cout<<",\"reaction_default_success\":"<<reactionDefault<<",\"reaction_default_residual\":"<<s.residual;
 c.relativeTolerance=sizeof(GpuReal)==4?GpuReal(1e-4):GpuReal(1e-8);
 assert(evaluateWallModel(m.in,c,w,o,s));
 std::cout<<",\"reaction_tolerance\":"<<c.relativeTolerance<<",\"reaction_residual\":"<<s.residual<<",\"wall_y\":"<<o.traceMassFraction[0];
 m.in.matching.velocity[0]=30;m.in.model.mode=GasMode::MixtureFrozen;
 m.in.matching.k=.5;m.in.matching.omega=200;
 GpuReal distances[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={distances,weights,2,4e-7,1.6e-9};
 c.enableSst=true;c.relativeTolerance=1e-8;c.maxIterations=100;
 const bool sstDefault=evaluateWallModel(m.in,c,w,o,s);
 std::cout<<",\"sst_default_success\":"<<sstDefault<<",\"sst_default_residual\":"<<s.residual;
 c.relativeTolerance=sizeof(GpuReal)==4?GpuReal(1e-2):GpuReal(1e-8);
 assert(evaluateWallModel(m.in,c,w,o,s));
 std::cout<<",\"sst_tolerance\":"<<c.relativeTolerance<<",\"sst_residual\":"<<s.residual<<",\"shear\":"<<o.traction[0]<<",\"owner_omega\":"<<o.ownerOmega<<"}\n";
}
'''
    results = {}
    for bits in (32, 64):
        src = tmp_path / f'precision_{bits}.cpp'
        src.write_text(BASE.replace('double', 'GpuReal') + body)
        exe = src.with_suffix('')
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', f'-DUGKWP_GPU_REAL_BITS={bits}', '-I', str(ROOT), '-I', str(ROOT/'common'), str(src), '-o', str(exe)], check=True, capture_output=True, text=True)
        result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        results[bits] = json.loads(result.stdout)
    print(json.dumps(results, sort_keys=True))
    assert results[64]['reaction_default_success'] == 1
    assert results[64]['sst_default_success'] == 1
    # A failed strict FP32 solve remains a reported failure; no implicit floor
    # or relaxed convergence rule is introduced into the production solver.
    assert results[32]['reaction_default_success'] == 0
    assert results[32]['sst_default_success'] == 0
    assert results[32]['reaction_default_residual'] > 1e-8
    assert results[32]['sst_default_residual'] > 1e-8
    for field in ('constant_heat', 'wall_y', 'shear', 'owner_omega'):
        assert abs(results[32][field] - results[64][field]) < 1e-3 * abs(results[64][field])
    for result in results.values():
        assert result['reaction_residual'] <= result['reaction_tolerance'] + 1e-10
        assert result['sst_residual'] <= result['sst_tolerance'] + 1e-10


def test_conditioned_sst_refinement_and_galilean_invariance(tmp_path):
    body = r'''
#include <iomanip>
int main(){
 static_assert(sizeof(GpuReal)*8==UGKWP_GPU_REAL_BITS,"native precision");
 Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=200;
 GpuReal distances[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={distances,weights,2,4e-7,1.6e-9};
 WallModelConfig<GpuReal> c;c.enableSst=true;c.maxIterations=100;c.relativeTolerance=sizeof(GpuReal)==4?.01:1e-8;
 static WallWorkspace<GpuReal,2,128> w;WallOutput<GpuReal,2> o,shifted;WallStatus s;
 std::cout<<std::setprecision(17)<<"[";bool comma=false;
 for(int n:{24,48,96,128})for(double mass:{0.,1e-9,.001,.01,.1}){
  c.nodes=n;m.in.massFlux[0]=GpuReal(.7*mass);m.in.massFlux[1]=GpuReal(.3*mass);
  m.in.velocity[0]=0;m.in.matching.velocity[0]=30;
  assert(evaluateWallModel(m.in,c,w,o,s));assert(s.residual<=c.relativeTolerance+c.absoluteTolerance);
  m.in.velocity[0]=1000;m.in.matching.velocity[0]=1030;
  assert(evaluateWallModel(m.in,c,w,shifted,s));
  assert(shifted.traction[0]==o.traction[0]&&shifted.conductiveHeatFlux==o.conductiveHeatFlux&&shifted.ownerOmega==o.ownerOmega);
  assert(shifted.traceVelocity[0]==1000);
  if(comma)std::cout<<",";
  comma=true;
  std::cout<<"["<<n<<","<<mass<<","<<o.residual<<","<<o.iterations<<","<<o.traction[0]<<","<<o.conductiveHeatFlux<<","<<o.ownerOmega<<"]";
 }
 std::cout<<"]\n";
}
'''
    results = {}
    for bits in (32, 64):
        src = tmp_path / f'conditioned_{bits}.cpp'
        src.write_text(BASE.replace('double','GpuReal') + body)
        exe=src.with_suffix('')
        subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',f'-DUGKWP_GPU_REAL_BITS={bits}','-I',str(ROOT),'-I',str(ROOT/'common'),str(src),'-o',str(exe)],check=True,capture_output=True,text=True)
        results[bits]=json.loads(subprocess.run([str(exe)],check=True,capture_output=True,text=True).stdout)
    for fp32,fp64 in zip(results[32],results[64]):
        assert fp32[:2] == fp64[:2]
        for a,b in zip(fp32[4:],fp64[4:]):
            assert abs(a-b) <= .005*abs(b)  # predeclared physical precision error
    print(json.dumps(results,sort_keys=True))


def test_exact_zero_creation_ten_species_both_native_precisions(tmp_path):
    body = r'''
int main(){

 SpeciesThermoData<GpuReal> species[10];GpuReal coefficients[30]={},elements[10],basis[10]={};
 GasReactionData<GpuReal> rx;GasStoichTerm<GpuReal> reactant,product;
 WallInput<GpuReal,10> in;for(int j=0;j<10;++j){species[j].coefficientOffset=j*3;species[j].molarMass=.028;species[j].minTemperature=200;species[j].maxTemperature=4000;species[j].referencePressure=101325;coefficients[j*3]=1000;elements[j]=1;}
 in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=30;
 in.model.thermo.elementComposition=elements;in.model.thermo.elementCount=1;in.model.thermo.speciesOrderHash=1;
 rx.reactantCount=1;rx.productCount=1;rx.highRate.preExponential=2;reactant.species=0;reactant.coefficient=1;product.species=9;product.coefficient=1;basis[0]=-1;basis[9]=1;
 auto& mechanism=in.model.mechanism;mechanism.reactions=&rx;mechanism.reactionCount=1;mechanism.reactants=&reactant;mechanism.reactantTermCount=1;mechanism.products=&product;mechanism.productTermCount=1;mechanism.referencePressure=101325;mechanism.speciesOrderHash=1;mechanism.stoichiometricBasis=basis;mechanism.independentRank=1;
 in.pressure=101325;in.temperature=in.matching.temperature=500;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;in.matching.massFraction[0]=1;
 in.model.viscosity=2e-5;in.model.conductivity=.04;for(int j=0;j<10;++j)in.model.diffusivity[j]=1e-4;in.model.mode=GasMode::MixtureChemistry;
 WallModelConfig<GpuReal> c;c.relativeTolerance=sizeof(GpuReal)==4?.01:1e-8;c.nodes=24;WallWorkspace<GpuReal,10,32> w;WallOutput<GpuReal,10> o;WallStatus s;
 if(!evaluateWallModel(in,c,w,o,s)){std::cerr<<int(s.code)<<" iteration "<<s.iteration;return 2;}
 assert(o.traceMassFraction[9]>.1);GpuReal sum=0;for(int j=0;j<10;++j){assert(o.traceMassFraction[j]>=0);sum+=o.traceMassFraction[j];}
 assert(near(sum,1));for(int j=1;j<9;++j)assert(o.traceMassFraction[j]<(sizeof(GpuReal)==4?16*std::numeric_limits<GpuReal>::epsilon():1e-12));
 // At a simplex corner the global dependent reactant is exhausted while
 // eight independent inert species are exactly zero. A tangent basis using
 // the local nonzero species must still yield a valid Jacobian.
 detail::LayerContext<GpuReal,10> ctx;detail::makeContext(in,c,ctx);
 for(int i=0;i<c.nodes;++i){for(int j=0;j<9;++j)w.state[i][3+j]=0;w.state[i][11]=1;}
 GpuReal norm=0;assert(detail::allResidual(in,c,ctx,w.state,w.y,w.residual,norm));
 assert(detail::newtonStep(in,c,ctx,w));
 rx.highRate.preExponential=400;c.nodes=32;c.maxIterations=100;
 assert(evaluateWallModel(in,c,w,o,s));assert(o.traceMassFraction[0]<1e-5);
 for(int j=1;j<9;++j)assert(o.traceMassFraction[j]<(sizeof(GpuReal)==4?16*std::numeric_limits<GpuReal>::epsilon():1e-12));

}
'''
    for bits in (32, 64):
        src=tmp_path / f"chemistry_{bits}.cpp"
        src.write_text(BASE.replace("double","GpuReal")+body)
        exe=src.with_suffix("")
        subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror",f"-DUGKWP_GPU_REAL_BITS={bits}","-I",str(ROOT),"-I",str(ROOT/"common"),str(src),"-o",str(exe)],check=True,capture_output=True,text=True)
        subprocess.run([str(exe)],check=True,capture_output=True,text=True)


def test_simultaneous_reaction_sst_and_nonzero_blowing_both_native_precisions(tmp_path):
    body = r'''
int main(){

 Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.model.mode=GasMode::MixtureChemistry;
 m.coef[2]=100000;m.rx[0].highRate.preExponential=20;
 m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 GpuReal y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
 WallModelConfig<GpuReal> c;c.relativeTolerance=sizeof(GpuReal)==4?.01:1e-8;c.enableSst=true;c.nodes=48;c.maxIterations=100;
 WallWorkspace<GpuReal,2,48> w;WallOutput<GpuReal,2> out;WallStatus status;
 if(!evaluateWallModel(m.in,c,w,out,status)){std::cerr<<"coupled reacting SST blowing failed "<<int(status.code)<<" "<<status.iteration<<" "<<status.residual;return 2;}
 assert(std::isfinite(out.conductiveHeatFlux)&&out.ownerOmega>0&&std::isfinite(out.integratedKSource));
 assert(std::abs(out.reactionIntegral[0])>1e-6);
 for(int j=0;j<2;++j){assert(out.traceMassFraction[j]>=0);assert(std::abs(out.speciesBalanceResidual[j])<1e-8);}
 assert(near(out.matchingSpeciesFlux[0]+out.matchingSpeciesFlux[1],.001,sizeof(GpuReal)==4?1e-5:1e-7));
 detail::LayerContext<GpuReal,2> ctx;detail::makeContext(m.in,c,ctx);detail::LayerFlux<GpuReal,2> first,last;
 assert(detail::intervalFlux(m.in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first));
 assert(detail::intervalFlux(m.in,c,ctx,w.state[c.nodes-2],w.state[c.nodes-1],w.y[c.nodes-2],w.y[c.nodes-1],last));
 assert(std::abs(last.energy-first.energy)<(sizeof(GpuReal)==4?1e-4:1e-7)*ctx.scale[2]);
 GpuReal wallEnergy=out.conductiveHeatFlux+.001*detail::dot(out.traceVelocity,out.traceVelocity)/2-detail::dot(out.traction,out.traceVelocity);
 for(int j=0;j<2;++j)wallEnergy+=m.in.massFlux[j]*speciesH(j,m.in.temperature,m.in.model.thermo);
 assert(near(wallEnergy,first.energy+ctx.mass*ctx.energyReference,sizeof(GpuReal)==4?1e-6:1e-10));

}
'''
    for bits in (32, 64):
        src=tmp_path / f"chemistry_{bits}.cpp"
        src.write_text(BASE.replace("double","GpuReal")+body)
        exe=src.with_suffix("")
        subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror",f"-DUGKWP_GPU_REAL_BITS={bits}","-I",str(ROOT),"-I",str(ROOT/"common"),str(src),"-o",str(exe)],check=True,capture_output=True,text=True)
        subprocess.run([str(exe)],check=True,capture_output=True,text=True)
