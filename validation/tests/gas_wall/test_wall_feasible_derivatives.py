"""A valid iterate need not admit the default finite-difference direction."""
import subprocess
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE

def test_valid_base_uses_feasible_temperature_derivative(tmp_path):
    source=tmp_path/'wall.cpp'
    source.write_text(BASE+r'''
int main(){Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;
    m.coef[0]=m.coef[3]=500000001000.;m.coef[1]=m.coef[4]=-1e9;
    ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;m.in.model.thermo.diagnostics=&diagnostic;
    WallModelConfig<double> c;c.nodes=4;WallWorkspace<double,2,4>w;detail::LayerContext<double,2>ctx;
    assert(detail::makeContext(m.in,c,ctx));for(int i=0;i<4;++i){w.y[i]=.01*i/3;w.state[i][0]=10.*i/3;w.state[i][3]=.3;w.state[i][5]=1;}
    double norm=0;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));assert(!diagnostic.fatalClaim);
    const auto before=w;assert(detail::assembleJacobianNode(m.in,c,ctx,w,1));assert(!diagnostic.fatalClaim);
    for(int i=0;i<4;++i)for(int j=0;j<6;++j)assert(w.state[i][j]==before.state[i][j]);
    for(int r=0;r<6;++r)for(int k=0;k<6;++k){assert(std::isfinite(w.upper[0][r][k]));assert(std::isfinite(w.diagonal[1][r][k]));assert(std::isfinite(w.lower[2][r][k]));}
}''')
    executable=tmp_path/'wall'
    subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_missing_selected_thermo_segment_is_not_a_recoverable_probe(tmp_path):
    source=tmp_path/'missing.cpp'
    source.write_text(BASE+r'''
int main(){Model m;m.in.temperature=m.in.matching.temperature=600;m.in.integrateOwnerKSource=false;
    double coefficient[12]{};coefficient[0]=coefficient[6]=1000./(universalGasConstant<double>()/.028);
    m.in.model.thermo.coefficients=coefficient;m.in.model.thermo.coefficientCount=12;
    for(int j=0;j<2;++j){m.sp[j].model=SpeciesThermoModel::NASA7;m.sp[j].coefficientOffset=6*j;m.sp[j].midTemperature=600;}
    ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;m.in.model.thermo.diagnostics=&diagnostic;
    WallModelConfig<double> c;c.nodes=4;WallWorkspace<double,2,4>w;detail::LayerContext<double,2>ctx;
    assert(detail::makeContext(m.in,c,ctx));for(int i=0;i<4;++i){w.y[i]=.01*i/3;w.state[i][0]=10.*i/3;w.state[i][3]=.3;w.state[i][5]=1;}
    double norm=0;assert(detail::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));assert(!diagnostic.fatalClaim);
    assert(!detail::assembleJacobianNode(m.in,c,ctx,w,1));assert(diagnostic.fatalClaim);
    assert(diagnostic.firstFatal.code==ugkp::diagnostics::Code::InvalidIndex);
}''')
    executable=tmp_path/'missing'
    subprocess.run(['g++','-std=c++17','-O2','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
