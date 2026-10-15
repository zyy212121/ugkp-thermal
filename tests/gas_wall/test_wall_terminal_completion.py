"""Actual terminal wall fields, with strict search and true-input controls."""
import subprocess
import pytest
from test_wall_model import BASE, ROOT

CASES = {
'analytic_trace': r"""
 Model m;m.in.temperature=0;m.in.massFlux[0]=.001;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallOutput<double,2> out;out.conductiveHeatFlux=917;out.matchingSpeciesFlux[1]=918;
 WallStatus status;assert(evaluateConstantTransportWall(m.in,c,out,status));
 assert(std::isinf(out.traceDensity));assert(out.wallSpeciesFlux[0]==.001);
 double cp=1000,a=.001*.01/2e-5,b=.001*cp*.01/.04;
 assert(out.traction[0]==2e-5/.01*detail::expRatio(a)*10);
 assert(out.matchingSpeciesFlux[1]==0&&out.conductiveHeatFlux!=917);
 assert(out.traceTemperature==0);(void)b;
""",
'analytic_heat': r"""
 Model m;m.in.matching.temperature=std::numeric_limits<double>::infinity();
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallOutput<double,2> out;out.conductiveHeatFlux=917;out.matchingSpeciesFlux[1]=918;
 WallStatus status;assert(evaluateConstantTransportWall(m.in,c,out,status));
 assert(out.traceTemperature==300&&out.traction[0]==.02);
 assert(std::isinf(out.conductiveHeatFlux)&&out.conductiveHeatFlux<0);
 assert(out.matchingSpeciesFlux[1]==0&&std::isnan(out.chemistryMismatch[1]));
""",
'point_state': r"""
 Model m;WallModelConfig<double> c;c.enableSst=true;
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 double x[6]={2,3,std::numeric_limits<double>::infinity(),.3,2,.5};
 detail::LayerPoint<double,2> strict;assert(!detail::pointState(m.in,c,ctx,x,strict));
 detail::LayerPoint<double,2> terminal;assert(detail::pointState(m.in,c,ctx,x,terminal,true));
 assert(std::isinf(terminal.T)&&terminal.Y[0]==.7&&terminal.Y[1]==.3);
 assert(terminal.rho==0&&terminal.omega==4&&terminal.u[1]==3);
""",
'point_rate': r"""
 Model m;m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].highRate.preExponential=1e308;
 detail::LayerPoint<double,2> p;p.T=500;p.rho=100;p.Y[0]=.7;p.Y[1]=.3;
 assert(!detail::pointReaction(m.in,p));
 p.source[0]=p.source[1]=0;assert(detail::pointReaction(m.in,p,true));
 assert(std::isinf(p.source[0])&&p.source[0]<0&&std::isinf(p.source[1])&&p.source[1]>0);
""",
'layer_diffusion': r"""
 Model m;WallModelConfig<double> c;detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 detail::LayerPoint<double,2> a,b;a.rho=b.rho=1;a.T=b.T=500;a.cp=b.cp=1000;
 a.Y[0]=.7;a.Y[1]=.3;b.Y[0]=.6;b.Y[1]=.4;b.u[0]=2;
 m.in.model.diffusivity[0]=std::numeric_limits<double>::infinity();
 detail::LayerFlux<double,2> f;assert(!detail::layerFlux(m.in,c,ctx,a,b,.01,f));
 f={};assert(detail::layerFlux(m.in,c,ctx,a,b,.01,f,0.,.01,true));
 assert(std::isnan(f.species[0])&&std::isnan(f.species[1]));
 assert(f.momentum[0]==-.004&&f.k==0&&std::isnan(f.energy));
""",
'bvp_rate_tail': r"""
 Model m;m.in.integrateOwnerKSource=false;m.in.model.mode=GasMode::MixtureChemistry;
 m.rx[0].highRate.preExponential=1e308;m.in.pressure=1e9;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;
 WallWorkspace<double,2,3> w;WallOutput<double,2> out;out.matchingSpeciesFlux[1]=917;out.wallKFlux=918;
 WallStatus status;assert(evaluateWallModel(m.in,c,w,out,status));
 assert(std::isinf(out.reactionIntegral[0])&&out.reactionIntegral[0]<0);
 assert(std::isinf(out.reactionIntegral[1])&&out.reactionIntegral[1]>0);
 assert(std::isinf(out.matchingSpeciesFlux[1])&&std::isnan(out.speciesBalanceResidual[1]));
 assert(out.wallKFlux==0&&out.traceTemperature==300&&out.traction[0]!=0);
 assert(std::isnan(out.residual));
""",
'bvp_trace': r"""
 Model m;m.in.integrateOwnerKSource=false;m.in.pressure=0;m.in.massFlux[0]=.001;
 m.in.model.mode=GasMode::MixtureChemistry;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;WallWorkspace<double,2,3>w;
 WallOutput<double,2> out;out.matchingSpeciesFlux[1]=917;out.wallKFlux=918;out.reactionIntegral[1]=919;
 WallStatus status;assert(evaluateWallModel(m.in,c,w,out,status));
 assert(out.traceDensity==0&&std::isinf(out.traceVelocity[1])&&std::isnan(out.traceVelocity[0]));
 assert(out.traceMassFraction[1]==.3&&out.wallSpeciesFlux[0]==.001);
 assert(out.matchingSpeciesFlux[1]==.001*.3&&out.reactionIntegral[1]==0&&out.wallKFlux==0);
 assert(std::isnan(out.residual)&&out.iterations==0);
""",
'bvp_diffusion': r"""
 Model m;m.in.integrateOwnerKSource=false;m.in.model.diffusivity[0]=INFINITY;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;WallWorkspace<double,2,3>w;
 WallOutput<double,2> out;out.matchingSpeciesFlux[1]=917;out.wallKFlux=918;out.reactionIntegral[1]=919;
 WallStatus status;assert(evaluateWallModel(m.in,c,w,out,status));
 assert(std::isnan(out.matchingSpeciesFlux[0])&&std::isnan(out.matchingSpeciesFlux[1]));
 assert(out.traceTemperature==300&&out.reactionIntegral[1]==0&&out.wallKFlux==0);
 assert(std::isfinite(out.traction[0])&&out.traction[0]!=0&&std::isnan(out.conductiveHeatFlux));
""",
'bvp_missing': r"""
 Model m;m.in.integrateOwnerKSource=false;m.in.model.mode=GasMode::MixtureChemistry;
 m.in.model.mechanism.products=nullptr;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;
 WallWorkspace<double,2,3> w;WallOutput<double,2> out;out.conductiveHeatFlux=917;
 WallStatus status;assert(!evaluateWallModel(m.in,c,w,out,status));assert(out.conductiveHeatFlux==917);
""",
'partial_enthalpy': r"""
 Model m;m.coef[5]=34567;m.in.model.thermo.coefficientCount=5;
 WallModelConfig<double> c;detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 double x[6]={0,0,0,.3,0,1};detail::LayerPoint<double,2> p;
 assert(detail::pointState(m.in,c,ctx,x,p,true));assert(p.h[1]==334567&&p.cp==1000);
 assert(std::isnan(speciesH(1,300.,m.in.model.thermo)));
""",
'precision': r"""
 WallInput<float,2> in;WallOutput<double,2> src;src.traceDensity=1e100;src.traction[0]=2e100;
 src.wallSpeciesFlux[0]=.001;src.matchingSpeciesFlux[0]=.003;src.reactionIntegral[0]=.0002;src.ownerOmega=77;
 WallOutput<float,2> out;out.ownerOmega=917;assert(detail::publishWallPrecision(in,src,out));
 assert(std::isinf(out.traceDensity)&&std::isinf(out.traction[0])&&out.ownerOmega==77);
 assert(out.speciesBalanceResidual[0]==float(double(out.matchingSpeciesFlux[0])-double(out.wallSpeciesFlux[0])-double(out.reactionIntegral[0])));
""",
}

@pytest.mark.parametrize('case', CASES)
def test_terminal_wall_completion(tmp_path, case):
    source=tmp_path/'probe.cpp';source.write_text(BASE+'int main(){'+CASES[case]+'}')
    exe=tmp_path/'probe'
    cmd=['g++','-std=c++17','-O1','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd))
    build=subprocess.run(cmd,capture_output=True,text=True);(tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr

@pytest.mark.parametrize('channel', ['heat', 'velocity', 'traction', 'species', 'omega', 'area', 'partial_h', 'missing_h'])
def test_exchange_completes_all_actual_channels(tmp_path, channel):
    code=BASE+r'''
#include "common/gasTransport/GasBoundaryLayerEvaluation.H"
struct State { GasBoundaryLayerState<double> gasBoundaryLayer; };
int main(){Model m;State state;auto& w=state.gasBoundaryLayer;
GasBoundaryLayerExchange<double> exchange[2];GasBoundaryLayerSstClosure<double> sst[2];int status[2]={};double species[4]={811,812,813,814};
w.enabled=true;w.count=2;w.exchange=exchange;w.sst=sst;w.status=status;w.speciesFlux=species;
WallOutput<double,2> out;out.traceTemperature=500;out.traceVelocity[0]=2;out.traceVelocity[1]=3;out.traceVelocity[2]=4;
out.traction[0]=5;out.traction[1]=6;out.traction[2]=7;out.wallSpeciesFlux[0]=.01;out.wallSpeciesFlux[1]=.02;
out.conductiveHeatFlux=8;out.ownerOmega=9;out.wallKFlux=999;out.integratedKSource=888;
m.in.ownerDistance=.125;m.in.quadrature.volume=.25;double area=2,sweep=.5;
MUTATION
exchange[1].mass=901;exchange[1].energy=902;sst[1].ownerOmega=903;
bool ok=publishGasBoundaryLayerOutput(state,1,m.in,out,area,sweep);
if(MISSING){assert(!ok&&!exchange[1].ready);assert(exchange[1].mass==901&&species[1]==812);return 0;}
assert(ok&&exchange[1].ready&&exchange[1].massReady&&gasBoundaryLayerSlotReady(state,1));
auto same=[](double a,double b){return a==b||(std::isnan(a)&&std::isnan(b));};
const double mass=out.wallSpeciesFlux[0]+out.wallSpeciesFlux[1];double speed=0,work=0,h=0;
for(int d=0;d<3;++d){speed+=out.traceVelocity[d]*out.traceVelocity[d];work+=out.traction[d]*out.traceVelocity[d];}
for(int j=0;j<2;++j){double actual=m.coef[3*j+2]+500*(m.coef[3*j]+.5*500*m.coef[3*j+1]);h+=out.wallSpeciesFlux[j]*actual;assert(same(species[j*2+1],-area*out.wallSpeciesFlux[j]));}
assert(species[0]==811&&species[2]==813);
assert(same(exchange[1].mass,-area*mass));
assert(same(exchange[1].energy,-area*(h+.5*mass*speed+out.conductiveHeatFlux+m.in.pressure*sweep-work)));
assert(same(exchange[1].momentumZ,-area*(mass*out.traceVelocity[2]+m.in.pressure*m.in.normal[2]-out.traction[2])));
assert(same(sst[1].ownerOmega,out.ownerOmega)&&sst[1].volume==.25&&sst[1].ownerDistance==.125);
assert(exchange[1].k==0&&exchange[1].matchingPressure==m.in.pressure);
}
'''
    mutations={'heat':'out.conductiveHeatFlux=INFINITY;', 'velocity':'out.traceVelocity[0]=INFINITY;',
       'traction':'out.traction[0]=INFINITY;', 'species':'out.wallSpeciesFlux[0]=INFINITY;',
       'omega':'out.ownerOmega=INFINITY;', 'area':'area=INFINITY;',
       'partial_h':'m.in.model.thermo.coefficientCount=5;m.coef[5]=34567;',
       'missing_h':'m.in.model.thermo.coefficients=nullptr;'}
    code=code.replace('MUTATION',mutations[channel]).replace('MISSING',str(int(channel=='missing_h')))
    source=tmp_path/'exchange.cpp';source.write_text(code);exe=tmp_path/'exchange'
    cmd=['g++','-std=c++17','-O1','-ffp-contract=off','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)]
    (tmp_path/'command.txt').write_text(' '.join(cmd))
    build=subprocess.run(cmd,capture_output=True,text=True);(tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
