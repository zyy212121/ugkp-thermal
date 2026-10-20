# Resolve small sqrt(k) derivatives against the current high-precision residual.
# Fixed captured profiles survive changes to the conservative operator; old
# numerical derivatives need not remain valid after a discretization change.
import json
import subprocess
import pytest
from pathlib import Path
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE

CASES=json.loads((Path(__file__).parent/'fixtures/sst_near_zero_k_derivative.json').read_text())['cases']


def input_statements(capture, suffix=''):
    values=capture['input']
    fields=['m.in.matchingDistance','m.in.ownerDistance','m.in.temperature','m.in.matching.temperature','m.in.matching.velocity[0]','m.in.matching.k','m.in.matching.omega']
    statements=[f'{name}={value!r}{suffix};' for name,value in zip(fields,values)]
    statements.extend([f'm.in.massFlux[0]=.7{suffix}*{values[7]!r}{suffix};',f'm.in.massFlux[1]=.3{suffix}*{values[7]!r}{suffix};'])
    return ''.join(statements)


@pytest.mark.parametrize('capture',CASES,ids=lambda x:str(x['case']))
def test_sqrt_k_derivative_has_a_resolvable_physical_scale(tmp_path,capture):
    profile=','.join('{'+','.join(repr(x)for x in row)+'}'for row in capture['profile'])
    precise_profile=','.join('{'+','.join(repr(x)+'L'for x in row)+'}'for row in capture['profile'])
    reference_model=BASE.replace('struct Model {','struct ReferenceModel {').replace(' Model(){',' ReferenceModel(){').replace('near(','referenceNear(').replace('double','long double')
    reference=r'''
long double centralResidualDerivative(long double h){
 ReferenceModel m;m.in.integrateOwnerKSource=false;SETUP
 WallModelConfig<long double>c;c.nodes=16;c.enableSst=true;
 detail::LayerContext<long double,2>ctx;assert(detail::makeContext(m.in,c,ctx));
 const long double data[16][7]={PROFILE};long double state[16][6]{},y[16]{};
 for(int i=0;i<16;++i){y[i]=data[i][0];for(int j=0;j<6;++j)state[i][j]=data[i][j+1];}
 const long double old=state[NODE][4];long double plus[6]{},minus[6]{};
 state[NODE][4]=old+h;assert(detail::layerRow(m.in,c,ctx,state,y,NODE,plus));
 state[NODE][4]=old-h;assert(detail::layerRow(m.in,c,ctx,state,y,NODE,minus));
 return (plus[4]-minus[4])/(2*h);
}
'''.replace('SETUP',input_statements(capture,'L')).replace('PROFILE',precise_profile).replace('NODE',str(capture['node']))
    source=tmp_path/'derivative.cpp'
    source.write_text(BASE+reference_model+reference+"\nint main(){Model m;m.in.integrateOwnerKSource=false;"+input_statements(capture)+r'''
    WallModelConfig<double> c;c.nodes=16;c.enableSst=true;WallWorkspace<double,2,16>w;detail::LayerContext<double,2>ctx;
    assert(detail::makeContext(m.in,c,ctx));const double data[16][7]={PROFILE};
    for(int i=0;i<16;++i){w.y[i]=data[i][0];for(int j=0;j<6;++j)w.state[i][j]=data[i][j+1];}
    double norm=0;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));
    assert(detail::assembleJacobianNode(m.in,c,ctx,w,NODE));
    const double derivative=w.diagonal[NODE][4][4];
    const long double coarse=centralResidualDerivative(1e-5L),fine=centralResidualDerivative(5e-6L);
    // Independent steps must agree before checking the Newton derivative.
    assert(std::isfinite(fine)&&fine!=0&&std::abs(coarse-fine)<1e-5L*std::abs(fine));
    assert(std::isfinite(derivative)&&std::abs(derivative-fine)<.001L*std::abs(fine));
    std::cout.precision(17);std::cout<<"Jacobian="<<derivative<<" high_precision="<<fine<<" reference_step_difference="<<std::abs(coarse-fine)<<'\n';
    }'''.replace('PROFILE',profile).replace('NODE',str(capture['node'])))
    executable=tmp_path/'derivative'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout,end='')


def test_zero_matching_k_retains_nonzero_endpoint_constraint_derivatives(tmp_path):
    source=tmp_path/'zero_matching.cpp'
    source.write_text(BASE+r'''
int main(){Model m;m.in.matching.k=0;m.in.matching.omega=200;m.in.matching.velocity[0]=30;
 WallModelConfig<double>c;c.enableSst=true;c.nodes=24;c.maxIterations=0;
 WallWorkspace<double,2,24>w;WallOutput<double,2>out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,out,status));
 detail::LayerContext<double,2>ctx;assert(detail::makeContext(m.in,c,ctx));double norm=0;
 assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));
 for(int node:{0,23}){assert(w.state[node][4]==0);
  assert(detail::assembleJacobianNode(m.in,c,ctx,w,node));
  // The endpoint residual is exactly sqrt(k)-sqrt(k_boundary), even at zero.
  assert(w.diagonal[node][4][4]==1.0);assert(w.state[node][4]==0);}
 c.maxIterations=60;assert(evaluateWallModel(m.in,c,w,out,status));
 assert(out.residual<=c.absoluteTolerance+c.relativeTolerance);
 assert(w.state[0][4]==0&&w.state[23][4]==0);
}
''')
    binary=source.with_suffix('')
    subprocess.run(['g++','-std=c++17','-O2','-I',str(SOURCE_ROOT),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_real_small_k_column_cannot_be_masked_by_omega_roundoff(tmp_path):
    source=tmp_path/'captured_column.cpp'
    source.write_text(r'''
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cassert>
#include <iostream>
int main(){using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::detail;
 CapturedSingularWall m;auto&in=m.input;auto&c=m.config;LayerContext<double,2>ctx;assert(makeContext(in,c,ctx));
 WallWorkspace<double,2,24>w;loadCapturedSingularIterate(w);double norm=0;
 assert(allResidual(in,c,ctx,w.state,w.y,w.residual,norm));const double before=norm;
 assert(assembleJacobianNode(in,c,ctx,w,20));
 const double derivative=w.diagonal[20][4][4],old=w.state[20][4],h=1e-5;
 double plus[6]{},minus[6]{};w.state[20][4]=old+h;assert(layerRow(in,c,ctx,w.state,w.y,20,plus));
 w.state[20][4]=old-h;assert(layerRow(in,c,ctx,w.state,w.y,20,minus));w.state[20][4]=old;
 const double reference=(plus[4]-minus[4])/(2*h);
 assert(reference<0&&std::abs(derivative-reference)<1e-4*std::abs(reference));
 assert(newtonStep(in,c,ctx,w));double alpha=0;assert(applyLayerNewtonUpdate(in,c,ctx,w,norm,alpha));
 assert(alpha>0&&norm<before);
 std::cout.precision(17);std::cout<<"derivative="<<derivative<<" reference="<<reference<<" residual="<<before<<"->"<<norm<<" alpha="<<alpha<<'\n';
 WallOutput<double,2>out;WallStatus status;const double matchingK=in.matching.k;
 assert(evaluateWallModel(in,c,w,out,status));assert(out.residual<=c.absoluteTolerance+c.relativeTolerance);
 assert(in.matching.k==matchingK&&w.state[23][4]==std::sqrt(matchingK));
 std::cout<<"fresh iterations="<<out.iterations<<" residual="<<out.residual<<'\n';
}
''')
    binary=source.with_suffix('')
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I',str(SOURCE_ROOT),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    print(run.stdout,end='')
