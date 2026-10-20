'''Wall finite-volume flux: exact thermal integral and coefficient resistance.
Failures identify trapezoidal Cp conduction and arithmetic spatial conductance.
'''
from test_wall_model import compile_run

def test_nasa_cp_heat_flux_uses_exact_enthalpy_increment(tmp_path):
    compile_run(tmp_path,r'''
Model m;double cp[28]={};
for(int j=0;j<2;++j){m.sp[j].model=SpeciesThermoModel::NASA7;m.sp[j].molarMass=universalGasConstant<double>();m.sp[j].coefficientOffset=14*j;m.sp[j].midTemperature=1000;
 for(int branch=0;branch<2;++branch){cp[14*j+7*branch]=3;cp[14*j+7*branch+2]=1e-5;cp[14*j+7*branch+5]=(j?2e6:-1e6);}}
m.in.model.thermo.coefficients=cp;m.in.model.thermo.coefficientCount=28;
m.in.temperature=300;m.in.matching.temperature=900;m.in.model.molecularPrandtl=.8;m.in.model.viscosity=2e-5;m.in.matching.velocity[0]=0;
WallModelConfig<double> c;detail::LayerContext<double,2>ctx;assert(detail::makeContext(m.in,c,ctx));
double xa[6]={0,0,0,.7,0,1},xb[6]={0,0,600,.7,0,1};detail::LayerPoint<double,2>a,b;
assert(detail::pointState(m.in,c,ctx,xa,a)&&detail::pointState(m.in,c,ctx,xb,b));detail::LayerFlux<double,2>flux;
assert(detail::layerFlux(m.in,c,ctx,a,b,.01,flux));
// R=1, Cp=3+1e-5*T^2: integral_300^900 Cp dT=4140 J/kg.
assert(near(flux.energy,-10.35,1e-12));
''')

def test_linear_transport_coefficient_integrates_resistance(tmp_path):
    compile_run(tmp_path,r'''
Model m;m.in.model.viscosity=1;m.in.model.conductivity=1;m.in.matching.velocity[0]=0;
WallModelConfig<double> c;detail::LayerContext<double,2>ctx;assert(detail::makeContext(m.in,c,ctx));
detail::LayerPoint<double,2>a,b;a.rho=b.rho=1;a.cp=b.cp=1000;a.T=b.T=500;a.Y[0]=b.Y[0]=1;a.u[0]=0;b.u[0]=std::log(9.);b.nut=8;
detail::LayerFlux<double,2>flux;assert(detail::layerFlux(m.in,c,ctx,a,b,1.,flux));
// Exact constant-stress solution: mu(y)=1+8*y, u(y)=log(1+8*y).
assert(near(flux.momentum[0],-8.,1e-12));
''')


def test_thermal_work_is_finite_and_covariant_for_decreasing_transport(tmp_path):
    compile_run(tmp_path,r"""
    using detail::thermalWorkVelocityWeight;
    for(auto ratio:{1e20,1e100,1e300}){
        double a=thermalWorkVelocityWeight(ratio,1.,ratio,1.);
        double b=thermalWorkVelocityWeight(1.,ratio,1.,ratio);
        assert(std::isfinite(a)&&std::isfinite(b));assert(near(a,.5,1e-12));assert(near(a+b,1.,1e-12));
    }
    double a=thermalWorkVelocityWeight(1e-300,1e300,1e-300,1e300);
    double b=thermalWorkVelocityWeight(1e300,1e-300,1e300,1e-300);
    assert(std::isfinite(a)&&std::isfinite(b));assert(near(a,.5,1e-12));assert(near(a+b,1.,1e-12));
    """)
