"""Actual production wall continuation, complete publication and arithmetic errors."""
from pathlib import Path
import subprocess
import os
import pytest
from test_wall_model import BASE, ROOT
from test_wall_publication_work import BODY

COMPARE = BODY[:BODY.index('int main(){')]
CASES = {
    'terminal': r'''
for(int limit:{1,80}) { Model m;m.in.temperature=m.in.matching.temperature=500;
 m.in.matching.velocity[0]=0;m.in.model.mode=GasMode::MixtureChemistry;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e-4;
 WallModelConfig<double> c;c.nodes=24;c.maxIterations=limit;c.absoluteTolerance=0;c.relativeTolerance=1e-30;
 WallWorkspace<double,2,48> w;WallOutput<double,2> out,expected;WallStatus s;
 std::memset(&out,0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,s));
 assert(s.code==WallCode::Success&&out.residual>c.relativeTolerance);
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 double actual=0;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.trial,actual));
 assert(actual==out.residual&&s.residual==out.residual&&s.iteration==out.iterations);
 for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j)assert(w.trial[i][j]==w.residual[i][j]);
 assert(detail::publishLayer(m.in,c,ctx,w,out.iterations,out.residual,expected));sameOutput(out,expected);
 assert(std::abs(out.reactionIntegral[0])>1e-4&&std::abs(out.matchingSpeciesFlux[0])>1e-4);
 std::cout<<"limit="<<limit<<" iterations="<<out.iterations<<" residual="<<out.residual<<'\n';
}
''',
    'short_nodes': r'''
for(bool sst:{false,true})for(int nodes:{2,3}) {Model m;m.in.integrateOwnerKSource=false;
 m.in.matching.k=.5;m.in.matching.omega=200;
 WallModelConfig<double> c;c.nodes=nodes;c.enableSst=sst;c.maxIterations=1;
 WallWorkspace<double,2,3> w;WallOutput<double,2> out,expected;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,out,s));assert(out.traceTemperature==m.in.temperature);
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 assert(detail::publishLayer(m.in,c,ctx,w,out.iterations,out.residual,expected));sameOutput(out,expected);
 // Supplied out-of-capacity node metadata is caller risk; never run an OOB probe.
 m.in.model.thermo.coefficients=nullptr;assert(!evaluateWallModel(m.in,c,w,out,s));
}
''',
    'signed_point': r'''
Model m;WallModelConfig<double> c;c.enableSst=true;
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 double x[6]={2,3,-200,1.2,2,-.5};detail::LayerPoint<double,2> p;
 assert(!detail::pointState(m.in,c,ctx,x,p));
 p={};assert(detail::pointState(m.in,c,ctx,x,p,true));assert(p.omega==4&&p.Y[ctx.independent[0]]==1.2);
 assert(p.T==100&&p.cp==1000&&std::isfinite(p.rho));
 m.coef[0]=m.coef[3]=-1000;m.in.pressure=-101325;detail::LayerPoint<double,2> signedP;
 assert(!detail::pointState(m.in,c,ctx,x,signedP));
 signedP={};assert(detail::pointState(m.in,c,ctx,x,signedP,true));assert(signedP.cp<0&&signedP.rho<0);
''',
    'tiny_pivot': r'''
double a[1][1]={{std::numeric_limits<double>::min()*2}},u[1][1]={{0}},rhs[1]={a[0][0]};
assert(!detail::eliminateBlock(a,u,rhs));assert(a[0][0]==std::numeric_limits<double>::min()*2&&rhs[0]==a[0][0]);
''',
    'finite_input': r'''
Model m;WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
WallWorkspace<double,2,24> w;WallOutput<double,2> o;WallStatus s;
m.in.temperature=100;m.in.matching.temperature=5000;m.in.normal[1]=2;
m.in.matching.massFraction[0]=1.2;m.in.matching.massFraction[1]=-.1;
m.in.massFlux[0]=-.001;m.in.massFlux[1]=.002;m.coef[1]=.1;
assert(evaluateWallModel(m.in,c,w,o,s));assert(o.traceTemperature==100&&std::isfinite(o.conductiveHeatFlux));
assert(o.traceMassFraction[0]==1.2&&o.traceMassFraction[1]==-.1&&o.wallSpeciesFlux[0]==-.001);
m.in.model.thermo.coefficients=nullptr;auto keep=o;
assert(!evaluateWallModel(m.in,c,w,o,s));sameOutput(o,keep);
''',
    'quadrature': r'''
Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;
m.in.matching.k=.5;m.in.matching.omega=200;
double y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
WallModelConfig<double> c;c.enableSst=true;c.nodes=24;
WallWorkspace<double,2,24> w;WallOutput<double,2> baseline,o;WallStatus s;
assert(evaluateWallModel(m.in,c,w,baseline,s));m.in.quadrature.volume=9;m.in.quadrature.firstMoment=-12;
assert(evaluateWallModel(m.in,c,w,o,s));sameOutput(o,baseline);
weights[0]=-1e-7;assert(evaluateWallModel(m.in,c,w,o,s));assert(std::isfinite(o.integratedKSource));
''',
    'zero_shear': r'''
Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;
m.in.matching.velocity[0]=0;m.in.matching.k=.5;m.in.matching.omega=200;
m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
WallModelConfig<double> c;c.enableSst=true;c.nodes=3;
WallWorkspace<double,2,3> w;WallOutput<double,2> o,expected;WallStatus s;
std::memset(&o,0xA5,sizeof(o));assert(evaluateWallModel(m.in,c,w,o,s));
assert(s.code==WallCode::Success&&std::isnan(o.residual));
// Failed seed must restore its entire initialized profile, not a partial z update.
for(int i=0;i<c.nodes;++i){double f=std::pow(double(i)/(c.nodes-1),c.stretch);
assert(w.state[i][5]==f/std::sqrt(m.in.matching.omega));}
detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
assert(detail::publishLayer(m.in,c,ctx,w,o.iterations,o.residual,expected));sameOutput(o,expected);
assert(o.wallSpeciesFlux[0]==.0007&&std::isfinite(o.ownerOmega)&&std::isfinite(o.wallKFlux));
''',
    'signed_diffusion': r'''
Model m;double y[2]={1.2,-.1},grad[2]={2,-3},d[2]={-.001,.002},flux[2]={123,123},enthalpy=123;
assert(correctedSpeciesDiffusion(y,grad,-2.,d,-.1,-.7,100.,m.in.model.thermo,flux,enthalpy));
double raw[2];for(int j=0;j<2;++j)raw[j]=2*(d[j]+(-.1)/(-.7))*grad[j];
assert(flux[0]==raw[0]-y[0]*(raw[0]+raw[1]));assert(flux[1]==-flux[0]);
assert(enthalpy==speciesH(0,100.,m.in.model.thermo)*flux[0]+speciesH(1,100.,m.in.model.thermo)*flux[1]);
''',
}

@pytest.mark.parametrize('case', CASES)
def test_finite_continuation(tmp_path, case):
    source=tmp_path/'probe.cpp';source.write_text(BASE+COMPARE+'int main(){\n'+CASES[case]+'\n}')
    exe=tmp_path/'probe'
    command=['g++','-std=c++17','-O1','-g','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    if case=='short_nodes': command += ['-fsanitize=address,undefined','-fno-omit-frame-pointer','-no-pie']
    build=subprocess.run(command,capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True,
        env={**os.environ,"ASAN_OPTIONS":"detect_leaks=0"})
    assert run.returncode==0,run.stdout+run.stderr
    print(run.stdout,end='')


def seed_fixture(header):
    """Inject signed Picard results; execute the actual relaxation and next pass."""
    start=header.index('template<class R,int N,int M> UGKWP_GAS_HD inline bool seedScalar(')
    end=header.index('\n}\n',start)
    old='    return true;'
    body=header[start:end]
    body=body[:body.rindex(old)]+r'''
    if(variable==0&&++::seedPass==2)return false;
    if(variable==N+2||variable==N+3)for(int i=1;i<c.nodes-1;++i)
        w.delta[i][variable]=-R(.5)*(variable==N+2?w.state[i][variable]*w.state[i][variable]:w.seed[i][7]);
    return true;'''
    return 'inline int seedPass=0;\n'+header[:start]+body+header[end:]


SEED_BODY=r'''
int main(){Model m;m.in.matching.k=.5;m.in.matching.omega=200;m.in.integrateOwnerKSource=false;
WallModelConfig<double> c;c.enableSst=true;c.nodes=4;
WallWorkspace<double,2,4> w;detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
for(int i=0;i<c.nodes;++i){w.y[i]=.01*i/3;w.state[i][0]=10.*i/3;w.state[i][2]=200.*i/3;
w.state[i][3]=.3;w.state[i][4]=.5;w.state[i][5]=double(i)/3/std::sqrt(200.);}
double oldK=w.state[2][4];detail::LayerPoint<double,2> wall;assert(detail::pointState(m.in,c,ctx,w.state[0],wall));
double slope=std::sqrt(wall.rho*m.in.model.sstCoefficients.beta1/(6*m.in.model.viscosity));
double oldOmega=1/(slope*slope*w.y[2]*w.y[2])+200;
assert(!detail::seedSstLayer(m.in,c,ctx,w)); // Original negative trial rejection.
assert(seedPass==1);assert(w.state[2][4]==oldK);
assert(w.state[2][5]==1/std::sqrt(oldOmega));
}
'''


def test_signed_seed_delta_preserves_original_internal_rejection(tmp_path):
    local=tmp_path/'instrumented';local.mkdir()
    (local/'WallModel.H').write_text((ROOT/'common/gasWall/WallModel.H').read_text())
    (local/'ReactingWallLayer.H').write_text(seed_fixture((ROOT/'common/gasWall/ReactingWallLayer.H').read_text()))
    source=tmp_path/'seed.cpp';source.write_text(BASE.replace('"common/gasWall/WallModel.H"','"instrumented/WallModel.H"')+SEED_BODY)
    exe=tmp_path/'seed'
    build=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),
        '-I'+str(ROOT/'common/gasWall'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_unrepresentable_newton_step_keeps_evaluated_profile(tmp_path):
    from test_wall_model import compile_run
    compile_run(tmp_path,r'''
 Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;
 m.in.matching.velocity[0]=0;m.in.matching.k=0;m.in.matching.omega=200;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=3;c.relativeTolerance=-1;c.absoluteTolerance=0;
 WallWorkspace<double,2,3> w;WallOutput<double,2> out;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,out,s));assert(s.code==WallCode::Success&&out.iterations==0);
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 auto retained=w;assert(!detail::newtonStep(m.in,c,ctx,w));
 for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j)assert(w.state[i][j]==retained.state[i][j]);
 double norm;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));assert(norm==out.residual);
 assert(std::isfinite(out.ownerOmega)&&out.traceTemperature==500);
 ''')


def test_double_zero_shear_preserves_unavailable_diagnostic_and_real_fields(tmp_path):
    source=tmp_path/'double.cpp'
    source.write_text(BASE+COMPARE+r'''
int main(){Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;
m.in.matching.velocity[0]=0;m.in.matching.k=.5;m.in.matching.omega=200;
m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
WallModelConfig<double> c;c.enableSst=true;c.nodes=3;
WallWorkspace<double,2,3> w;WallOutput<double,2> out,expected;WallStatus s;
std::memset(&out,0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,s));
assert(s.code==WallCode::Success&&std::isnan(out.residual));
auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);
detail::LayerContext<double,2> ctx;detail::makeContext(in,controls,ctx);WallOutput<double,2> exact;
assert(detail::publishLayer(in,controls,ctx,w,out.iterations,double(out.residual),exact));
assert(detail::publishWallPrecision(m.in,exact,expected));sameOutput(out,expected);
assert(out.wallSpeciesFlux[0]==m.in.massFlux[0]&&std::isfinite(out.ownerOmega));
}
''')
    exe=tmp_path/'double'
    subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
