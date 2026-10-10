"""Executable shared wall-model numerical contracts; no source-text assertions."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
BASE = r'''
#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}
'''
def compile_run(tmp_path, body):
    src=tmp_path/'wall.cpp';src.write_text(BASE+'\nint main(){\n'+body+'\n}\n')
    exe=tmp_path/'wall'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-pedantic','-I',str(ROOT),'-I',str(ROOT/'common'),str(src),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)

def test_constant_limits_and_failure_atomicity(tmp_path):
    compile_run(tmp_path,r'''
 Model m;WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,o,s));
 assert(near(o.traction[0],.02));assert(near(o.traction[1],0));
 // Dissipation heats the gas: linear conduction plus half the shear work.
 assert(near(o.conductiveHeatFlux,-800-.1));
 assert(near(o.traceTemperature,300));
 m.in.matching.velocity[0]=0;m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 assert(evaluateWallModel(m.in,c,w,o,s));
 const double pe=.001*1000*.01/.04;
 assert(near(o.conductiveHeatFlux,-.001*1000*200/std::expm1(pe)));
 const double keep=o.conductiveHeatFlux;m.in.temperature=-1;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.conductiveHeatFlux==keep);
 m.in.temperature=300;m.in.massFlux[0]=-.1;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.conductiveHeatFlux==keep);
 m.in.massFlux[0]=.0007;m.in.normal[1]=2;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.conductiveHeatFlux==keep);
 ''')

def test_reacting_diffusion_and_frozen_control(tmp_path):
    compile_run(tmp_path,r'''
 Model m; m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.model.mode=GasMode::MixtureChemistry;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e-4;
 WallModelConfig<double> c;c.nodes=40;c.relativeTolerance=1e-9;c.absoluteTolerance=1e-11;
 WallWorkspace<double,2,48> w;WallOutput<double,2> o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::cerr<<int(s.code)<<" node "<<s.node<<" iteration "<<s.iteration<<"\n";return 2;}
 const double beta=std::sqrt(2/1e-4), expected=.7/std::cosh(beta*.01);
 assert(near(o.traceMassFraction[0],expected,.002));
 assert(near(o.traceMassFraction[0]+o.traceMassFraction[1],1));
 assert(std::abs(o.conductiveHeatFlux)<1e-6);
 assert(near(o.matchingSpeciesFlux[0],o.reactionIntegral[0],1e-7));
 assert(near(o.reactionIntegral[0],-o.reactionIntegral[1],1e-9));
 assert(std::abs(o.reactionIntegral[0])>1e-4);
 m.in.model.mode=GasMode::MixtureFrozen;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(near(o.traceMassFraction[0],.7,1e-8));
 assert(std::abs(o.reactionIntegral[0])<1e-12);
 ''')

def test_sst_zero_positive_blowing_and_owner_volume(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.matching.velocity[0]=30;m.in.temperature=m.in.matching.temperature=500;
 m.in.matching.k=.5;m.in.matching.omega=200;
 double distances[2]={.002,.006},weights[2]={2e-7,2e-7};
 m.in.quadrature.distance=distances;m.in.quadrature.volumeWeight=weights;m.in.quadrature.count=2;
 m.in.quadrature.volume=4e-7;m.in.quadrature.firstMoment=1.6e-9;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=32;c.maxIterations=100;c.relativeTolerance=1e-8;
 WallWorkspace<double,2,48> w;WallOutput<double,2> o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::cerr<<"zero "<<int(s.code)<<" iteration "<<s.iteration<<"\n";return 2;}
 assert(o.ownerOmega>0&&std::isfinite(o.integratedKSource));
 const double omega=o.ownerOmega,shear=o.traction[0];
 m.in.massFlux[0]=7e-10;m.in.massFlux[1]=3e-10;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::cerr<<"tiny "<<int(s.code)<<" iteration "<<s.iteration<<"\n";return 3;}
 assert(near(o.ownerOmega,omega,.01));assert(near(o.traction[0],shear,.01));
 m.in.massFlux[0]=7e-4;m.in.massFlux[1]=3e-4;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::cerr<<"blow "<<int(s.code)<<" iteration "<<s.iteration<<"\n";return 4;}
 assert(o.ownerOmega>0&&o.traction[0]>0);const double keep=o.integratedKSource;
 weights[0]=-1e-7;assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.integratedKSource==keep);
 ''')

def test_thermo_range_and_chemistry_diagnostic_availability(tmp_path):
    compile_run(tmp_path,r'''
 Model m;WallModelConfig<double> c;WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(!o.chemistryMismatchAvailable);
 double coarse[2]={1.,-1.};m.in.coarseReactionIntegral=coarse;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(o.chemistryMismatchAvailable);
 assert(near(o.chemistryMismatch[0],-1.));
 double keep=o.conductiveHeatFlux;m.in.temperature=100;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.conductiveHeatFlux==keep);
 m.in.temperature=300;m.in.matching.temperature=5000;
 assert(!evaluateWallModel(m.in,c,w,o,s));assert(o.conductiveHeatFlux==keep);
 ''')

def test_exothermic_differential_diffusion_independent_bvp(tmp_path):
    import numpy as np
    from scipy.integrate import solve_bvp
    L=.01; pressure=101325.; gasR=8.31446261815324/.028
    D1=1e-4; D2=3e-4; rate=50.; formation=1e5; conductivity=.04
    def ode(y,q):
        Y,j,T,F=q
        rho=pressure/(gasR*T)
        return np.vstack((-j/(rho*(D1*(1-Y)+D2*Y)),-rho*rate*Y,(formation*j-F)/conductivity,np.zeros_like(y)))
    def bc(a,b):
        return np.array([a[1],a[2]-500,b[0]-.7,b[2]-700])
    y=np.linspace(0,L,160);guess=np.vstack((np.full_like(y,.5),-.001*y/L,500+200*y/L,np.full_like(y,-800.)))
    reference=solve_bvp(ode,bc,y,guess,tol=1e-7,max_nodes=10000)
    assert reference.success,reference.message
    wallY=reference.y[0,0];heat=reference.y[3,0];edgeJ=reference.y[1,-1]
    compile_run(tmp_path, f'''
 Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=0;
 m.coef[2]=100000;m.rx[0].highRate.preExponential=50;m.in.model.mode=GasMode::MixtureChemistry;
 m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 WallModelConfig<double> c;c.nodes=96;c.maxIterations=100;c.relativeTolerance=1e-10;
 WallWorkspace<double,2,128> w;WallOutput<double,2> o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s)){{std::cerr<<int(s.code)<<" iteration "<<s.iteration;return 2;}}
 assert(near(o.traceMassFraction[0],{wallY:.17g},.005));
 assert(near(o.conductiveHeatFlux,{heat:.17g},.003));
 assert(near(o.matchingSpeciesFlux[0],{edgeJ:.17g},.003));
 assert(std::abs(o.speciesBalanceResidual[0])<1e-9);
 const double error=std::abs(o.conductiveHeatFlux-({heat:.17g}));c.nodes=24;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(std::abs(o.conductiveHeatFlux-({heat:.17g}))>error*2);
 ''')

def test_exact_zero_creation_ten_species(tmp_path):
    compile_run(tmp_path,r'''
 SpeciesThermoData<double> species[10];double coefficients[30]={},elements[10],basis[10]={};
 GasReactionData<double> rx;GasStoichTerm<double> reactant,product;
 WallInput<double,10> in;for(int j=0;j<10;++j){species[j].coefficientOffset=j*3;species[j].molarMass=.028;species[j].minTemperature=200;species[j].maxTemperature=4000;species[j].referencePressure=101325;coefficients[j*3]=1000;elements[j]=1;}
 in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=30;
 in.model.thermo.elementComposition=elements;in.model.thermo.elementCount=1;in.model.thermo.speciesOrderHash=1;
 rx.reactantCount=1;rx.productCount=1;rx.highRate.preExponential=2;reactant.species=0;reactant.coefficient=1;product.species=9;product.coefficient=1;basis[0]=-1;basis[9]=1;
 auto& mechanism=in.model.mechanism;mechanism.reactions=&rx;mechanism.reactionCount=1;mechanism.reactants=&reactant;mechanism.reactantTermCount=1;mechanism.products=&product;mechanism.productTermCount=1;mechanism.referencePressure=101325;mechanism.speciesOrderHash=1;mechanism.stoichiometricBasis=basis;mechanism.independentRank=1;
 in.pressure=101325;in.temperature=in.matching.temperature=500;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;in.matching.massFraction[0]=1;
 in.model.viscosity=2e-5;in.model.conductivity=.04;for(int j=0;j<10;++j)in.model.diffusivity[j]=1e-4;in.model.mode=GasMode::MixtureChemistry;
 WallModelConfig<double> c;c.nodes=24;WallWorkspace<double,10,32> w;WallOutput<double,10> o;WallStatus s;
 if(!evaluateWallModel(in,c,w,o,s)){std::cerr<<int(s.code)<<" iteration "<<s.iteration;return 2;}
 assert(o.traceMassFraction[9]>.1);double sum=0;for(int j=0;j<10;++j){assert(o.traceMassFraction[j]>=0);sum+=o.traceMassFraction[j];}
 assert(near(sum,1));for(int j=1;j<9;++j)assert(o.traceMassFraction[j]<1e-12);
 // At a simplex corner the global dependent reactant is exhausted while
 // eight independent inert species are exactly zero. A tangent basis using
 // the local nonzero species must still yield a valid Jacobian.
 detail::LayerContext<double,10> ctx;detail::makeContext(in,c,ctx);
 for(int i=0;i<c.nodes;++i){for(int j=0;j<9;++j)w.state[i][3+j]=0;w.state[i][11]=1;}
 double norm=0;assert(detail::allResidual(in,c,ctx,w.state,w.y,w.residual,norm));
 assert(detail::newtonStep(in,c,ctx,w));
 rx.highRate.preExponential=400;c.nodes=32;c.maxIterations=100;
 assert(evaluateWallModel(in,c,w,o,s));assert(o.traceMassFraction[0]<1e-5);
 for(int j=1;j<9;++j)assert(o.traceMassFraction[j]<1e-12);
 ''')

def test_local_cp_transport_conductivity(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.matching.velocity[0]=0;m.coef[1]=m.coef[4]=1;
 m.in.model.molecularPrandtl=.7;
 WallModelConfig<double> c;c.nodes=32;WallWorkspace<double,2,48> w;WallOutput<double,2> o;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,o,s));
 double expected=-2e-5/.7/.01*(1000*(500-300)+.5*(500*500-300*300));
 assert(near(o.conductiveHeatFlux,expected,1e-7));
 ''')

def test_constant_blowing_dissipation_green_function(tmp_path):
    import numpy as np
    from scipy.integrate import quad
    a=.001*.01/2e-5;b=.001*1000*.01/.04
    def integrand(x):
        grad=(a/np.expm1(a))*np.exp(a*x)/.01
        return 2e-5*100*grad*grad*np.expm1(b*(1-x))/np.expm1(b)*.01
    expected=-quad(integrand,0,1,epsabs=1e-14)[0]
    compile_run(tmp_path,f'''
 Model m;m.in.matching.temperature=m.in.temperature;m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,32> w;WallOutput<double,2> o;WallStatus s;
 assert(evaluateWallModel(m.in,c,w,o,s));assert(near(o.conductiveHeatFlux,{expected:.17g},1e-9));
 ''')

def test_near_wall_reconstruction_and_asymptotic_fitting(tmp_path):
    compile_run(tmp_path,r'''
 const double p=.5*(1+std::sqrt(1+24*.09/.075));
 assert(near(detail::centeredDerivative(1.,4.,16.,1.,2.),4.));
 assert(near(detail::nearWallKShape(.01,1.,0.,p,0.),std::pow(.01,p)));
 const double v=detail::nearWallKShape(.01,1.,1e-12,p,0.);
 assert(near(v,std::pow(.01,p),1e-8));
 assert(near(detail::nearWallKShape(.001,1.,100.,p,0.)/.001,detail::nearWallKShape(.0001,1.,100.,p,0.)/.0001,.001));
 const double l=.2,r=.8,mid=.5,A=.03;
 const double fitted=detail::omegaGradientFit(l,r)*(A/(r*r)-A/(l*l))/(r-l);
 assert(near(fitted,-2*A/(mid*mid*mid)));
 const double yl=.1,yc=.2,yr=.35;
 const double integral=detail::omegaDestructionFit(yc,yl,yr)*(yr-yl)*A*A/std::pow(yc,4);
 assert(near(integral,A*A/3*(1/std::pow(yl,3)-1/std::pow(yr,3))));
 ''')

def test_sst_cross_diffusion_uses_physical_omega_gradient(tmp_path):
    compile_run(tmp_path,r'''
 Model m;WallModelConfig<double> c;c.enableSst=true;
 detail::LayerPoint<double,2> a,p,b;
 a.k=.5;p.k=.4;b.k=.3;a.z=.01;p.z=.02;b.z=.1;
 a.omega=1/(a.z*a.z);p.omega=1/(p.z*p.z);b.omega=1/(b.z*b.z);
 a.rho=p.rho=b.rho=.7;
 detail::pointSst(m.in,c,.01,a,b,.003,p,.001);
 const double dk=detail::centeredDerivative(a.k,p.k,b.k,.001,.002);
 const double dw=detail::centeredDerivative(a.omega,p.omega,b.omega,.001,.002);
 const auto& coefficients=m.in.model.sstCoefficients;
 const double cd=sstCrossDiffusion(p.omega,dk*dw,coefficients);
 const double f1=sstF1(p.k,p.omega,m.in.model.viscosity/p.rho,.01,cd,coefficients);
 const double f2=sstF2(p.k,p.omega,m.in.model.viscosity/p.rho,.01,coefficients);
 const double expected=sstOmegaSource(p.rho,p.k,p.omega,0.,0.,0.,f1,f2,cd,coefficients);
 assert(near(p.sourceOmega,expected,1e-12));
 ''')

def test_near_wall_bridge_independent_advection_ode(tmp_path):
    import numpy as np
    from scipy.integrate import solve_bvp
    p=.5*(1+np.sqrt(1+24*.09/.075))
    lines=[]
    # Normalize first-cell distance to one. The dimensionless equation retains
    # advection, unlike the leading Euler reconstruction used in production.
    for pe in (.0005,.05):
        ell=10.
        x=np.linspace(0,1,80)
        solution=solve_bvp(lambda x,q: np.vstack((q[1],pe*q[1]+p*(p-1)*q[0]/(x+ell)**2)),
                           lambda a,b:np.array([a[0],b[0]-1]),x,np.vstack((x,np.ones_like(x))),tol=1e-10)
        assert solution.success
        for xq in (.01,.1,.5,.9):
            expected=solution.sol(xq)[0]
            lines.append(f'assert(near(detail::nearWallKShape({xq},1.,10.,{p:.17g},{pe/2:.17g}),{expected:.17g},{pe*pe/8:.17g}));')
    compile_run(tmp_path,'\n'.join(lines))

def test_owner_source_reconstruction_matches_bvp_node_source(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;
 double y=.004,weight=4e-7;m.in.quadrature={&y,&weight,1,weight,weight*y};
 WallModelConfig<double> c;c.enableSst=true;c.nodes=48;c.maxIterations=100;
 WallWorkspace<double,2,48> w;WallOutput<double,2> out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,out,status));
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
 for(int node:{10,20,30}){y=w.y[node];m.in.quadrature.firstMoment=weight*y;
  detail::LayerPoint<double,2> point;assert(detail::evaluatePoint(m.in,c,ctx,w.state,w.y,node,point));
  assert(detail::publishLayer(m.in,c,ctx,w,status.iteration,status.residual,out));
  assert(near(out.integratedKSource,weight*point.sourceK,1e-10));}
 ''')

def test_rotated_wall_flux_and_sst_covariance(tmp_path):
    compile_run(tmp_path,r'''
 auto rotate=[](const double* a,double* b){b[0]=a[0]/std::sqrt(2.)-a[1]/std::sqrt(6.)+a[2]/std::sqrt(3.);b[1]=a[0]/std::sqrt(2.)+a[1]/std::sqrt(6.)-a[2]/std::sqrt(3.);b[2]=2*a[1]/std::sqrt(6.)+a[2]/std::sqrt(3.);};
 Model m;WallModelConfig<double> c;c.nodes=48;c.maxIterations=100;c.relativeTolerance=1e-10;
 WallWorkspace<double,2,48> w;WallOutput<double,2> base,rotated;WallStatus status;
 for(bool sst:{false,true}){m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;
  m.in.matching.k=.5;m.in.matching.omega=200;c.enableSst=sst;
  double y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
  assert(evaluateWallModel(m.in,c,w,base,status));auto input=m.in;
  rotate(m.in.normal,input.normal);rotate(m.in.velocity,input.velocity);rotate(m.in.matching.velocity,input.matching.velocity);
  assert(evaluateWallModel(input,c,w,rotated,status));double expected[3];rotate(base.traction,expected);
  for(int d=0;d<3;++d)assert(near(rotated.traction[d],expected[d],1e-6));
  assert(near(rotated.conductiveHeatFlux,base.conductiveHeatFlux,1e-6));
  if(sst){assert(near(rotated.ownerOmega,base.ownerOmega,1e-6));assert(near(rotated.integratedKSource,base.integratedKSource,1e-5));}}
 ''')

def test_simultaneous_reaction_sst_and_nonzero_blowing(tmp_path):
    compile_run(tmp_path,r'''
 Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.model.mode=GasMode::MixtureChemistry;
 m.coef[2]=100000;m.rx[0].highRate.preExponential=20;
 m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
 m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;
 double y[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={y,weights,2,4e-7,1.6e-9};
 WallModelConfig<double> c;c.enableSst=true;c.nodes=48;c.maxIterations=100;
 WallWorkspace<double,2,48> w;WallOutput<double,2> out;WallStatus status;
 if(!evaluateWallModel(m.in,c,w,out,status)){std::cerr<<"coupled reacting SST blowing failed "<<int(status.code)<<" "<<status.iteration<<" "<<status.residual;return 2;}
 assert(std::isfinite(out.conductiveHeatFlux)&&out.ownerOmega>0&&std::isfinite(out.integratedKSource));
 assert(std::abs(out.reactionIntegral[0])>1e-6);
 for(int j=0;j<2;++j){assert(out.traceMassFraction[j]>=0);assert(std::abs(out.speciesBalanceResidual[j])<1e-8);}
 assert(near(out.matchingSpeciesFlux[0]+out.matchingSpeciesFlux[1],.001,1e-7));
 detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);detail::LayerFlux<double,2> first,last;
 assert(detail::intervalFlux(m.in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first));
 assert(detail::intervalFlux(m.in,c,ctx,w.state[c.nodes-2],w.state[c.nodes-1],w.y[c.nodes-2],w.y[c.nodes-1],last));
 assert(std::abs(last.energy-first.energy)<1e-7*ctx.scale[2]);
 double wallEnergy=out.conductiveHeatFlux+.001*detail::dot(out.traceVelocity,out.traceVelocity)/2-detail::dot(out.traction,out.traceVelocity);
 for(int j=0;j<2;++j)wallEnergy+=m.in.massFlux[j]*speciesH(j,m.in.temperature,m.in.model.thermo);
 assert(near(wallEnergy,first.energy+ctx.mass*ctx.energyReference,1e-10));
 ''')
