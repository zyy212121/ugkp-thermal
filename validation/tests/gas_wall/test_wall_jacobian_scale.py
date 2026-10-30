"""Small sqrt(k) must not turn a nonzero physical derivative into a zero column."""
import json
import subprocess
import pytest
from pathlib import Path
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE

CASES=json.loads((Path(__file__).parent/'fixtures/sst_near_zero_k_derivative.json').read_text())['cases']

@pytest.mark.parametrize('capture',CASES,ids=lambda x:str(x['case']))
def test_sqrt_k_derivative_has_a_resolvable_physical_scale(tmp_path,capture):
    reference=capture['reference']
    if capture['case']==94:
        from independent_sst_k import derivative
        old_equation=derivative(capture,legacy_face_average=True)
        assert abs(old_equation-reference)<1e-7*abs(reference)
        reference=derivative(capture)
        # ffd64367 changed mean(rho)*mean(alpha)*mean(nut) to
        # mean(rho*alpha*nut). The retained JSON records the former equation.
    values=capture['input'];statements=[]
    fields=['m.in.matchingDistance','m.in.ownerDistance','m.in.temperature','m.in.matching.temperature','m.in.matching.velocity[0]','m.in.matching.k','m.in.matching.omega']
    statements.extend(f'{name}={value!r};' for name,value in zip(fields,values))
    statements.extend([f'm.in.massFlux[0]=.7*{values[7]!r};',f'm.in.massFlux[1]=.3*{values[7]!r};'])
    profile=','.join('{'+','.join(repr(x)for x in row)+'}'for row in capture['profile'])
    source=tmp_path/'derivative.cpp'
    source.write_text(BASE+"\nint main(){Model m;m.in.integrateOwnerKSource=false;"+''.join(statements)+r'''
    WallModelConfig<double> c;c.nodes=16;c.enableSst=true;WallWorkspace<double,2,16>w;detail::LayerContext<double,2>ctx;
    assert(detail::makeContext(m.in,c,ctx));const double data[16][7]={PROFILE};
    for(int i=0;i<16;++i){w.y[i]=data[i][0];for(int j=0;j<6;++j)w.state[i][j]=data[i][j+1];}
    double norm=0;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));
    assert(detail::assembleJacobianNode(m.in,c,ctx,w,NODE));
    const double derivative=w.diagonal[NODE][4][4],reference=REFERENCE;
    assert(std::isfinite(derivative)&&std::abs(derivative-reference)<.001*std::abs(reference));
    }'''.replace('PROFILE',profile).replace('NODE',str(capture['node'])).replace('REFERENCE',repr(reference)))
    executable=tmp_path/'derivative'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
