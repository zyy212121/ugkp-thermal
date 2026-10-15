"""Full original-success wall search traces, instrumenting audit copies only."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import shutil
import subprocess
import tarfile
import pytest
from test_wall_model import BASE

ROOT=SOURCE_ROOT
ORIGINAL='d6110c02745a8ad726208a426c4bb3a28233c1fd'

@pytest.fixture(scope='module')
def original_wall_root(tmp_path_factory):
    root=tmp_path_factory.mktemp('original-wall-tree')
    archive=subprocess.check_output(['git','archive',ORIGINAL,'common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:source.extractall(root,filter='data')
    return root

SOURCE=r'''
int main(){unsigned seed=4321;for(int index=0;index<=SELECTED;++index){auto random=[&](){seed=1664525*seed+1013904223;return double(seed)/4294967296.;};Model m;m.in.integrateOwnerKSource=false;
 m.in.matchingDistance=std::pow(10.,-4+3*random());m.in.ownerDistance=.4*m.in.matchingDistance;
 m.in.temperature=250+800*random();m.in.matching.temperature=300+1500*random();m.in.matching.velocity[0]=std::pow(10.,-1+4*random());
 m.in.matching.k=std::pow(10.,-10+12*random());m.in.matching.omega=std::pow(10.,-2+7*random());
 const double mass=std::pow(10.,-8+8*random());m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
 WallModelConfig<double> c;c.nodes=16;c.maxIterations=80;c.relativeTolerance=1e-8;c.enableSst=index%2==0;
 if(!c.enableSst){m.in.model.mode=GasMode::MixtureChemistry;m.coef[2]=1e5+3e5*random();m.rx[0].highRate.preExponential=std::pow(10.,3*random());m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;}
 if(index!=SELECTED)continue;
 for(double v:{m.in.matchingDistance,m.in.ownerDistance,m.in.temperature,m.in.matching.temperature,m.in.matching.velocity[0],m.in.matching.k,m.in.matching.omega,mass,m.coef[2],m.rx[0].highRate.preExponential})std::printf("%a ",v);std::puts("");
 WallWorkspace<double,2,24>w;WallOutput<double,2>o;WallStatus st;bool ok=evaluateWallModel(m.in,c,w,o,st);
 assert(ok);std::printf("FINAL %d %d %d %d %a %d %a %d\n",ok,int(st.code),st.node,st.iteration,st.residual,o.iterations,o.residual,o.chemistryMismatchAvailable);
 for(double v:{o.traceTemperature,o.traceDensity,o.conductiveHeatFlux,o.integratedKSource,o.ownerOmega,o.wallKFlux})std::printf("%a ",v);std::puts("");
 const double*arrays[]={o.traceMassFraction,o.traceVelocity,o.traction,o.wallSpeciesFlux,o.matchingSpeciesFlux,o.reactionIntegral,o.chemistryMismatch,o.speciesBalanceResidual};int count[]={2,3,3,2,2,2,2,2};for(int a=0;a<8;++a){for(int k=0;k<count[a];++k)std::printf("%a ",arrays[a][k]);std::puts("");}
 for(int i=0;i<c.nodes;++i){for(int k=0;k<6;++k)std::printf("%a ",w.state[i][k]);std::puts("");}
}}
'''


def build_run(tmp,label,root,selected):
    copy=tmp/label;shutil.copytree(root/'common',copy/'common')
    p=copy/'common/gasWall/ReactingWallLayer.H';s=p.read_text()
    s=s.replace('if(!valid){w.state[j][col]=old;',
        'std::printf("DERIV %d %d %d %d %a\\n",j,col,attempt,valid,double(h));if(!valid){w.state[j][col]=old;')
    s=s.replace('R next=0;if(allResidual(in,c,ctx,w.trial,w.y,w.residual,next)&&next<norm)',
        'R next=0;const bool validTrial=allResidual(in,c,ctx,w.trial,w.y,w.residual,next);std::printf("LINE %d %d %a %d %a\\n",iteration,line,double(alpha),validTrial,double(next));if(validTrial&&next<norm)')
    s=s.replace('    R norm=0;',
        '    std::puts("SEED");for(int i=0;i<c.nodes;++i){for(int j=0;j<N+4;++j)std::printf("%a ",double(w.state[i][j]));std::puts("");}\n    R norm=0;')
    p.write_text(s)
    body=SOURCE.replace('SELECTED',str(selected))
    src=tmp/(label+'.cpp');exe=tmp/(label+'.exe');src.write_text('#include <cstdio>\n#include <initializer_list>\n'+BASE+body)
    result=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(copy),str(src),'-o',str(exe)],capture_output=True,text=True)
    (tmp/(label+'_build.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=180)
    (tmp/(label+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    return result.stdout

@pytest.mark.parametrize('selected',[11,29,150],ids=['reacting-one-damping','reacting-six-damping','sst-fifty-six-damping'])
def test_original_successful_rejected_trials_and_complete_profile(tmp_path,original_wall_root,selected):
    before=build_run(tmp_path,'original',original_wall_root,selected)
    after=build_run(tmp_path,'corrected',ROOT,selected)
    assert any(line.startswith('LINE ') and int(line.split()[2])>0 for line in before.splitlines())
    assert after==before


def test_constant_wall_original_successful_intermediate_overflow(tmp_path,original_wall_root):


    body=BASE+r'''
int main(){Model m;m.in.integrateOwnerKSource=false;m.in.matchingDistance=1e-20;m.in.ownerDistance=4e-21;
 m.in.temperature=m.in.matching.temperature=300;m.in.matching.velocity[0]=1e-30;m.in.model.viscosity=1e300;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,24>w;WallOutput<double,2>o;WallStatus st;
 bool ok=evaluateWallModel(m.in,c,w,o,st);assert(ok);
 std::printf("%d %d %d %d %a %a %a %a %a %a %a %a\n",ok,int(st.code),st.node,st.iteration,st.residual,o.traceTemperature,o.traceDensity,o.conductiveHeatFlux,o.traction[0],o.traction[1],o.traction[2],o.residual);
 const double*arrays[]={o.traceMassFraction,o.traceVelocity,o.traction,o.wallSpeciesFlux,o.matchingSpeciesFlux,o.reactionIntegral,o.chemistryMismatch,o.speciesBalanceResidual};const int count[]={2,3,3,2,2,2,2,2};
 for(int a=0;a<8;++a){for(int j=0;j<count[a];++j)std::printf("%a ",arrays[a][j]);std::puts("");}
 std::printf("%a %a %a %d %d\n",o.integratedKSource,o.ownerOmega,o.wallKFlux,o.iterations,int(o.chemistryMismatchAvailable));
 assert(std::isfinite(o.conductiveHeatFlux)&&std::isinf(o.traction[0]));
}
'''
    results=[]
    for label,root in [('original',original_wall_root),('corrected',ROOT)]:
        src=tmp_path/(label+'.cpp');exe=tmp_path/label;src.write_text(body)
        build=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(root),str(src),'-o',str(exe)],capture_output=True,text=True)
        (tmp_path/(label+'_build.log')).write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        (tmp_path/(label+'.log')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
        results.append(run.stdout)
    assert results[0]==results[1]
