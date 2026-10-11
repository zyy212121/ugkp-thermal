"""Compare production wall solve and actual coarse SST kernels to fine FV SST."""
from pathlib import Path
import json,sys,subprocess,numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'tests/gas_transport'))
from test_legacy_ale_transport import fixture
r=json.loads((OUT/'reference_8192.json').read_text());y=np.asarray(r['y']);T=np.asarray(r['T']);rho=101325/(8314.46261815324/28*T)
cases=[]
for yp in [.00075,.0015,.00225]:
 lo=np.r_[0,y[(y>0)&(y<2*yp)],2*yp]
 def integral(a):return np.trapezoid(np.interp(lo,y,a),lo)/(2*yp)
 rc=integral(rho);kc=integral(rho*np.asarray(r['k']))/rc;uc=integral(rho*np.asarray(r['U']))/rc;tc=integral(rho*T)/rc
 d=[float(np.interp(3*yp,y,r[n])) for n in ['U','T','k','omega']]
 du=float(np.interp(3*yp,y,np.gradient(r['U'],y)));dn=2*np.sqrt(d[2])/(.09*d[3]*(3*yp));dv=500*(2e-5/(101325/(8314.46261815324/28*d[1])))/((3*yp)**2*d[3]);f2=np.tanh(min(max(dn,dv),100)**2);nut=.31*d[2]/max(.31*d[3],abs(du)*f2)
 cases.append([yp,rc,kc,uc,tc,*d,nut,float(np.interp(yp,y,r['omega']))])
casecode='double cases[][11]={'+','.join('{'+','.join(format(a,'.17g') for a in row)+'}' for row in cases)+'};'
body=fixture()+r'''
#include "research/sst_wall_function/owner32_input.hpp"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
#include <iomanip>
int main(){CASES
std::cout<<std::setprecision(17)<<"[";int which=0;
for(const auto& test:cases){Model m;m.in.integrateOwnerKSource=false;const double yp=test[0];m.in.temperature=500;m.in.ownerDistance=yp;m.in.matchingDistance=3*yp;m.in.matching.velocity[0]=test[5];m.in.matching.temperature=test[6];m.in.matching.k=test[7];m.in.matching.omega=test[8];m.in.model.molecularPrandtl=.72;
static double qy[256],qw[256];for(int i=0;i<256;++i){qy[i]=(i+.5)*2*yp/256;qw[i]=2*yp/256;}m.in.quadrature={qy,qw,256,2*yp,2*yp*yp};
WallModelConfig<double> cfg;cfg.enableSst=true;cfg.nodes=128;cfg.maxIterations=100;static WallWorkspace<double,2,128> work;WallOutput<double,2> out;WallStatus status;
if(!evaluateWallModel(m.in,cfg,work,out,status)){std::cerr<<"wall fail "<<int(status.code)<<"\n";return 2;}
State s;initialise(s);s.nFaces=2;s.nInternalFaces=1;s.sstConfigured=1;s.turbulenceModel=3;s.sstWallTreatment=2;s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;s.gasMu=2e-5;s.Rgas=ugkwp::universalGasConstant<Real>()/.028;s.gasPrClamped=.72;
for(int c=0;c<2;++c){s.Cx[c]=0;s.Cy[c]=(1+2*c)*yp;s.Cz[c]=0;s.V[c]=2*yp;s.sstWallDistance[c]=s.Cy[c];s.rho[c]=c==0?test[1]:101325/(s.Rgas*test[6]);s.Tgas[c]=c==0?test[4]:test[6];s.p[c]=101325;s.Ux[c]=c==0?test[3]:test[5];s.Uy[c]=s.Uz[c]=0;s.k[c]=c==0?test[2]/20:test[7];s.omega[c]=c==0?out.ownerOmega:test[8];s.rhoK[c]=s.rho[c]*s.k[c];s.rhoOmega[c]=s.rho[c]*s.omega[c];s.nut[c]=test[9];}
s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=2;s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellPlaneCount[1]=0;
for(int f=0;f<2;++f){s.Sfx[f]=s.Sfz[f]=0;s.Sfy[f]=f==0?1:-1;s.magSf[f]=1;s.faceCx[f]=s.faceCz[f]=0;s.faceCy[f]=f==0?2*yp:0;s.deltaCoeffs[f]=f==0?1/(2*yp):1/yp;s.faceWeight[f]=.5;s.riemannBoundaryKind[f]=f==0?0:2;}
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=new int[2]{-1,0};w.ownerSlot=new int[2]{0,-1};w.status=new int[1]{};w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{};
if(!ugkwp::publishGasBoundaryLayerOutput(s,0,m.in,out,1.,0.))return 3;
double residual=1,time=0;int steps=0;
for(;steps<100000;++steps){threadIdx.x=0;applySstWallFunctionStateKernel(&s);computeGasPrimitiveGradientsKernel(&s);computeGasEddyViscosityKernel(&s);computeSstGradientsKernel(&s);
double dt=1e-4;computeSstStabilityNumberKernel(&s,dt,.5);if(s.sstSourceNumber[0]>.5)dt*=.45/s.sstSourceNumber[0];
for(int f=0;f<2;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
Real sk,so;sstSourcesForCell(s,0,0.,sk,so);const double flux=-s.sstPhiRhoK[0]-s.sstPhiRhoK[1],activity=sstKSourceActivity(s,0,sk,0.);
residual=std::abs(flux/(2*yp)+sk)/std::max(activity,1e-30);if(residual<1e-9)break;
threadIdx.x=0;applySstFluxAndSourceKernel(&s,dt);recoverSstPrimitivesKernel(&s);if(s.gasSpecies.cellStatus[0])return 4;time+=dt;}
if(steps==100000)return 5;
if(which++)std::cout<<",";std::cout<<"{\"y\":"<<yp<<",\"reference_cell_k\":"<<test[2]<<",\"computed_cell_k\":"<<s.k[0]<<",\"k_relative_error\":"<<s.k[0]/test[2]-1<<",\"reference_omega\":"<<test[10]<<",\"computed_omega\":"<<out.ownerOmega<<",\"wall_tau\":"<<out.traction[0]<<",\"heat\":"<<out.conductiveHeatFlux<<",\"steps\":"<<steps<<",\"steady_relative_residual\":"<<residual<<"}";
}std::cout<<"]\n";}
'''.replace('CASES',casecode)
pre=(ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes')[0]
source=OUT/'couette_coarse.cpp';source.write_text(pre+body);exe=OUT/'couette_coarse'
subprocess.run(['g++','-O2','-std=c++17','-I'+str(ROOT),'-I'+str(ROOT/'tests/gas_transport'),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],check=True)
p=subprocess.run([str(exe)],capture_output=True,text=True);print(p.stdout+p.stderr);p.check_returncode();result=json.loads(p.stdout)
for a in result:
 a.update(tau_relative_error=a['wall_tau']/r['tau']-1,heat_relative_error=a['heat']/r['heat']-1,omega_relative_error=a['computed_omega']/a['reference_omega']-1)
 a['y_plus']=a['y']*np.sqrt(r['tau']/(101325/(8314.46261815324/28*500)))/(2e-5/(101325/(8314.46261815324/28*500)))
 a['matching_distance']=3*a['y']
(OUT/'couette_comparison.json').write_text(json.dumps(result,indent=2))
