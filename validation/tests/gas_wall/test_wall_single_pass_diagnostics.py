"""Single-pass wall solve diagnostics, executable on CPU without a CUDA device."""
from validation.tests.gas_wall.test_wall_model import compile_run


def test_wall_physical_failure_and_sticky_record(tmp_path):
    compile_run(tmp_path, r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;m.in.diagnosticEntity=17;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,32> w;WallOutput<double,2> o;o.traceTemperature=123;WallStatus s;
 m.in.temperature=-1;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.traceTemperature==123);
 assert(diagnostic.firstFatal.code==ugkp::diagnostics::Code::NegativeTemperature);
 assert(diagnostic.firstFatal.entityIndex==17);assert(diagnostic.firstFatal.observed==-1);
 m.in.temperature=300;assert(!evaluateWallModel(m.in,c,w,o,s));
 assert(diagnostic.firstFatal.observed==-1);
 ugkp::diagnostics::DeviceContext second{};m.in.diagnostics=&second;
 m.in.temperature=std::numeric_limits<double>::quiet_NaN();
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(second.firstFatal.code==ugkp::diagnostics::Code::NonFinite);
 ''')


def test_wall_clips_finite_y_and_warns_without_retry(tmp_path):
    compile_run(tmp_path, r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 m.in.matching.massFraction[0]=1.2;m.in.matching.massFraction[1]=-.2;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(!diagnostic.fatalClaim);
 assert(o.traceMassFraction[0]==1&&o.traceMassFraction[1]==0);
 assert(diagnostic.warnings[2].count==2);
 assert(diagnostic.warnings[2].sample.observed==1.2&&diagnostic.warnings[2].sample.corrected==1);
 ''')


def test_wall_iteration_limit_rejects_without_publishing(tmp_path):
    compile_run(tmp_path, r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 WallModelConfig<double> c;c.nodes=12;c.maxIterations=0;
 WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 o.traceTemperature=917;o.conductiveHeatFlux=918;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(s.code!=WallCode::Success);
 assert(o.traceTemperature==917&&o.conductiveHeatFlux==918);
 for(int i=0;i<c.nodes;++i)assert(std::isfinite(w.state[i][2]));
 ''')


def test_wall_nonfinite_y_not_clipped(tmp_path):
    compile_run(tmp_path, r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 WallModelConfig<double> c;WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 m.in.matching.massFraction[0]=std::numeric_limits<double>::quiet_NaN();
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(diagnostic.fatalClaim);
 assert(diagnostic.firstFatal.quantityId==ugkp::diagnostics::Quantity::MassFraction);
 assert(diagnostic.warnings[2].count==0);
 ''')


def test_wall_full_newton_sst_converges_without_fatal(tmp_path):
    compile_run(tmp_path, r'''
 Model m;ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;
 m.in.matching.velocity[0]=30;m.in.temperature=m.in.matching.temperature=500;
 m.in.matching.k=.5;m.in.matching.omega=200;
 double distances[2]={.002,.006},weights[2]={2e-7,2e-7};
 m.in.quadrature.distance=distances;m.in.quadrature.volumeWeight=weights;m.in.quadrature.count=2;
 m.in.quadrature.volume=4e-7;m.in.quadrature.firstMoment=1.6e-9;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=32;c.maxIterations=100;c.relativeTolerance=1e-8;
 WallWorkspace<double,2,48> w;WallOutput<double,2> o;WallStatus s;
 for(int blowing=0;blowing<3;++blowing){
   const double mass=blowing==0?0:(blowing==1?1e-9:1e-3);
   m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
   assert(evaluateWallModel(m.in,c,w,o,s));assert(!ugkp::diagnostics::hasFatal(&diagnostic));
   assert(o.residual<=c.absoluteTolerance+c.relativeTolerance);
   assert(o.iterations>0&&o.iterations<c.maxIterations);
   assert(o.ownerOmega>0&&o.traction[0]>0&&std::isfinite(o.integratedKSource));
 }
 ''')


def test_cooperative_wall_diagnostic_binding_is_leader_owned():
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    text=(root/'common/gasTransport/GasBoundaryLayerModelState.H').read_text()
    binding='in.diagnostics=state.diagnostics;in.diagnosticEntity=face;'
    assert text.count(binding)==1
    slot=text[text.index('UGKWP_GAS_HD inline bool evaluateGasBoundaryLayerSlot'):]
    assert binding not in slot
    assert 'if(team.leader())prepared=prepareGasBoundaryLayerInput' in slot
    assert 'if(!team.broadcast(prepared))return false;' in slot
