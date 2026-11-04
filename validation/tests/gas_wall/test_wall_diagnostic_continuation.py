"""Non-authoritative wall diagnostics never veto complete physical publication."""
import subprocess
import pytest
from test_wall_model import BASE, ROOT
from test_wall_finite_publication import COMPARE

PHYSICS = r'''
template<class R> void physicalCast(const WallOutput<R,2>& out,const WallOutput<double,2>& exact){
#define FIELD(name) assert(out.name==R(exact.name)&&std::isfinite(out.name))
 FIELD(traceTemperature);FIELD(traceDensity);FIELD(conductiveHeatFlux);FIELD(ownerOmega);FIELD(wallKFlux);
#undef FIELD
 for(int d=0;d<3;++d){assert(out.traceVelocity[d]==R(exact.traceVelocity[d])&&std::isfinite(out.traceVelocity[d]));
  assert(out.traction[d]==R(exact.traction[d])&&std::isfinite(out.traction[d]));}
 for(int j=0;j<2;++j){
#define ARRAY(name) assert(out.name[j]==R(exact.name[j])&&std::isfinite(out.name[j]))
  ARRAY(traceMassFraction);ARRAY(wallSpeciesFlux);ARRAY(matchingSpeciesFlux);ARRAY(reactionIntegral);
#undef ARRAY
 }
 assert(out.iterations==exact.iterations);
}
'''


def compile_run(tmp_path, body):
    source=tmp_path/'diagnostic.cpp'
    source.write_text(BASE+COMPARE+PHYSICS+'int main(){\n'+body+'\n}')
    exe=tmp_path/'diagnostic'
    build=subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
        '-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    print(run.stdout,end='')


@pytest.mark.parametrize('iterations',[0,1])
def test_large_double_residual_does_not_publish_unconverged_state(tmp_path,iterations):
    compile_run(tmp_path,r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 m.in.integrateOwnerKSource=false;m.in.model.viscosity=1e20;m.in.model.conductivity=-.04;
 m.in.matching.velocity[0]=1e5;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=LIMIT;
 WallWorkspace<double,2,3> w;WallOutput<double,2> out;WallStatus status,exactStatus;
 auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);WallOutput<double,2> exact;
 if(LIMIT==1){exact.traceTemperature=123;
  assert(!evaluateReactingWallLayerImpl(in,controls,w,exact,exactStatus));
  assert(diagnostic.firstFatal.code==ugkp::diagnostics::Code::ZeroDenominator&&diagnostic.firstFatal.sourceSiteId==2421&&diagnostic.firstFatal.observed==0);
  assert(exact.traceTemperature==123);return 0;}
 exact.traceTemperature=123;out.traceTemperature=124;
 assert(!evaluateReactingWallLayerImpl(in,controls,w,exact,exactStatus));
 assert(exactStatus.code!=WallCode::Success&&exact.traceTemperature==123);
 diagnostic={};assert(!evaluateWallModel(m.in,c,w,out,status));
 assert(status.code!=WallCode::Success&&out.traceTemperature==124);
'''.replace('LIMIT',str(iterations)))


def test_large_species_demand_rejects_unconverged_publication(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.matchingDistance=1;m.in.ownerDistance=.5;m.in.massFlux[0]=3e38;m.in.massFlux[1]=-3e38;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=2e38;
 ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 // Positive heat capacity keeps this a diagnostic-residual test.
 for(int j=0;j<6;++j)m.coef[j]=j%3==0?1000:0;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;WallWorkspace<double,2,3> w;WallOutput<double,2> out;WallStatus status;
 auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);WallOutput<double,2> exact;
 exact.traceTemperature=917;out.traceTemperature=918;
 assert(!evaluateReactingWallLayerImpl(in,controls,w,exact,status));
 assert(exact.traceTemperature==917&&status.code!=WallCode::Success);
 diagnostic={};assert(!evaluateWallModel(m.in,c,w,out,status));
 assert(out.traceTemperature==918&&status.code!=WallCode::Success);
''')


def test_disabled_omega_channel_cannot_veto_complete_physics(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.integrateOwnerKSource=false;m.in.matchingDistance=1e200;m.in.ownerDistance=4e199;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=1;WallWorkspace<double,2,3> w;WallOutput<double,2> out,expected;WallStatus status;
 std::memset(static_cast<void*>(&out),0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,status));
 assert(status.code==WallCode::Success&&!c.enableSst);
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 assert(detail::publishLayer(m.in,c,ctx,w,out.iterations,out.residual,expected));sameOutput(out,expected);
 assert(std::isfinite(out.conductiveHeatFlux)&&out.conductiveHeatFlux<0&&out.traction[0]>0);
 // Disabled transport leaves the unused omega scratch untouched; it must not evaluate inf/inf.
 detail::LayerFlux<double,2> last;last.omega=123;
 assert(detail::intervalFlux(m.in,c,ctx,w.state[1],w.state[2],w.y[1],w.y[2],last));assert(last.omega==123);
 assert(std::isfinite(last.energy)&&std::isfinite(last.k));
 std::cout<<"heat="<<out.conductiveHeatFlux<<" traction="<<out.traction[0]<<'\n';
''')


@pytest.mark.parametrize('rate',[0,20])
def test_optional_nonfinite_chemistry_audit_preserves_physical_profile(tmp_path,rate):
    compile_run(tmp_path,r'''
 Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].highRate.preExponential=RATE;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=100;WallWorkspace<double,2,3> w;
 WallOutput<double,2> plain,out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,plain,status));const auto savedStatus=status;
 double coarse[2]={0,-std::numeric_limits<double>::infinity()};m.in.coarseReactionIntegral=coarse;
 std::memset(static_cast<void*>(&out),0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,status));
 physicalCast(out,plain);assert(status.code==WallCode::Success&&status.residual==savedStatus.residual);
 assert(out.residual==plain.residual&&out.chemistryMismatchAvailable&&!plain.chemistryMismatchAvailable);
 for(int j=0;j<2;++j){assert(out.chemistryMismatch[j]==out.reactionIntegral[j]-coarse[j]);
  assert(out.speciesBalanceResidual[j]==plain.speciesBalanceResidual[j]);}
 assert(std::isinf(out.chemistryMismatch[1])&&out.chemistryMismatch[1]>0);
 // Adapter must preserve the same actual diagnostic even without narrowing.
 WallOutput<double,2> converted;assert(detail::publishWallPrecision(m.in,out,converted));sameOutput(out,converted);
'''.replace('RATE',str(rate)))


def test_large_finite_chemistry_mismatch_preserves_physics(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].highRate.preExponential=20;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=100;WallWorkspace<double,2,3> w;WallOutput<double,2> plain,out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,plain,status));
 double coarse[2]={0,-3e38};m.in.coarseReactionIntegral=coarse;
 auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);WallOutput<double,2> exact;
 assert(evaluateReactingWallLayerImpl(in,controls,w,exact,status));assert(std::isfinite(exact.chemistryMismatch[1]));
 assert(exact.chemistryMismatch[1]==3e38);
 std::memset(static_cast<void*>(&out),0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,status));physicalCast(out,exact);
 assert(out.chemistryMismatchAvailable&&out.chemistryMismatch[1]==exact.chemistryMismatch[1]);
 assert(out.reactionIntegral[0]==plain.reactionIntegral[0]&&out.matchingSpeciesFlux[0]==plain.matchingSpeciesFlux[0]);
 assert(out.conductiveHeatFlux==plain.conductiveHeatFlux);
''')


def test_optional_owner_source_integral_overflow_preserves_physics(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=300;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.integrateOwnerKSource=false;
 WallModelConfig<SCALAR> c;c.enableSst=true;c.nodes=24;c.maxIterations=100;
 WallWorkspace<SCALAR,2,24> w;WallOutput<SCALAR,2> plain,out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,plain,status));
 auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);
 detail::LayerContext<double,2> ctx;detail::makeContext(in,controls,ctx);
 int index=1;double largest=0;
 for(int i=1;i<c.nodes;++i){detail::LayerPoint<double,2> point;assert(detail::evaluatePoint(in,controls,ctx,w.state,w.y,i,point));
  if(std::abs(point.sourceK)>std::abs(largest)){index=i;largest=point.sourceK;}}
 assert(std::abs(largest)>1);
 SCALAR distance=SCALAR(w.y[index]),weight=std::numeric_limits<SCALAR>::max();
 m.in.quadrature={&distance,&weight,1,SCALAR(1),SCALAR(1)};m.in.integrateOwnerKSource=true;
 std::memset(static_cast<void*>(&out),0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,status));
 assert(status.code==WallCode::Success&&!std::isfinite(out.integratedKSource));
 // Optional source work changes no required output or residual trajectory.
 out.integratedKSource=plain.integratedKSource;sameOutput(out,plain);
 in=detail::wallComputationInput(m.in);WallOutput<double,2> exact;
 assert(detail::publishLayer(in,controls,ctx,w,status.iteration,status.residual,exact));
 WallOutput<SCALAR,2> converted;assert(detail::publishWallPrecision(m.in,exact,converted));
 assert(!std::isfinite(converted.integratedKSource));physicalCast(converted,exact);
 assert(converted.integratedKSource==SCALAR(exact.integratedKSource)
  ||(std::isnan(converted.integratedKSource)&&std::isnan(exact.integratedKSource)));
'''.replace('SCALAR','double'))
