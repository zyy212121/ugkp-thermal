"""Actual shared SST operators on the captured owner32 six-face control volume.

The surrounding gas/donor state is held at its captured initial value. This is
not a full 4608-cell gasUGKP run or CHMT run. Every step re-solves/publishes the current wall input.
"""
from pathlib import Path
import csv, json, os, re, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/gas_transport'))
from test_legacy_ale_transport import fixture
capture=json.loads((HERE/'owner32_capture.json').read_text())
control=(HERE/'gpu_inputs/gas_flatplate/system/controlDict').read_text()
max_dt=float(re.search(r'\bmaxDeltaT\s+([^;]+);',control).group(1))
max_co=float(re.search(r'\bmaxCo\s+([^;]+);',control).group(1))
rows=capture['faces']; cells={int(k):v for k,v in capture['cells'].items()}
ids=[32,33,128,31]; mp={i:j for j,i in enumerate(ids)}
order=[rows[0],rows[1],rows[5],rows[2],rows[3],rows[4]]
setup=[]
for c,old in enumerate(ids):
 x,y,z,v=cells[old];setup.append(f's.Cx[{c}]={x:.17g};s.Cy[{c}]={y:.17g};s.Cz[{c}]={z:.17g};s.V[{c}]={v:.17g};')
for f,r in enumerate(order):
 oldown=int(float(r['own']));oldnei=int(float(r['nei']));own=mp[oldown];nei=mp.get(oldnei,-1)
 for key,field in [('area','magSf'),('Sfx','Sfx'),('Sfy','Sfy'),('Sfz','Sfz'),('massFlux','gasPhiRho')]: setup.append(f's.{field}[{f}]={float(r[key]):.17g};')
 setup.append(f's.faceOwner[{f}]={own};s.faceNeighbour[{f}]={nei};s.facePeriodicPair[{f}]=-1;s.faceWeight[{f}]=.5;s.riemannBoundaryKind[{f}]={0 if f<3 else 2 if f==3 else 4};')
 if nei>=0:
  import math
  d=math.dist(cells[ids[own]][:3],cells[ids[nei]][:3]);setup.append(f's.deltaCoeffs[{f}]={1/d:.17g};')
 else:setup.append(f's.deltaCoeffs[{f}]=1/m.in.ownerDistance;')
body=fixture()+r'''
#include "research/sst_wall_function/owner32_input.hpp"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
#include <chrono>
#include <sstream>
int main(){Model m;configure(m);m.in.integrateOwnerKSource=std::getenv("SST_WALL_DIAGNOSTICS")!=nullptr;State s;initialise(s);s.nCells=4;s.nFaces=6;s.nInternalFaces=3;
delete[] s.gasSpecies.cellStatus;delete[] s.gasSpecies.faceStatus;s.gasSpecies.cellStatus=new int[4]{};s.gasSpecies.faceStatus=new int[6]{};
s.sstConfigured=1;s.turbulenceModel=3;s.sstWallTreatment=2;s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;
s.sstOmegaMin=1e-6;
s.gasMu=m.in.model.viscosity;s.gasPrClamped=.72;s.gasCp=1040;s.Rgas=ugkwp::universalGasConstant<Real>()/.028;
for(int c=0;c<4;++c){s.rho[c]=m.in.pressure/(s.Rgas*300);s.Tgas[c]=300;s.p[c]=m.in.pressure;s.Ux[c]=70;s.Uy[c]=s.Uz[c]=0;s.k[c]=1e-4;s.omega[c]=c==2?1:47776.201947879716;s.rhoK[c]=s.rho[c]*s.k[c];s.rhoOmega[c]=s.rho[c]*s.omega[c];s.cellPlaneCount[c]=0;s.sstWallDistance[c]=m.in.ownerDistance;s.nut[c]=s.k[c]/s.omega[c];}
SETUP
s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=6;for(int f=0;f<6;++f)s.cellFaceId[f]=f;
auto& wall=s.gasBoundaryLayer;wall.enabled=true;wall.count=1;wall.faceSlot=new int[6]{-1,-1,-1,0,-1,-1};wall.ownerSlot=new int[4]{0,-1,-1,-1};wall.status=new int[1]{};wall.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];wall.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];wall.speciesFlux=new Real[2]{};
WallModelConfig<double> config;config.enableSst=true;WallWorkspace<double,2,48> scratch;WallOutput<double,2> output;WallStatus status;
const double initial=s.rhoK[0]*s.V[0];double sourceIntegral=0,faceIntegral=0,maxBudget=0,minK=s.k[0],minDt=1e30,maxNumber=0,time=0;int steps=0,wallIterations=0;
const auto started=std::chrono::steady_clock::now();
while(time<.2){
 // Unchanged prescribed gas/donor inputs are intentional control-volume BCs.
 if(!evaluateWallModel(m.in,config,scratch,output,status)){std::cerr<<"wall fail "<<int(status.code)<<"\n";return 2;}
 wallIterations+=output.iterations;
 if(!ugkwp::publishGasBoundaryLayerOutput(s,0,m.in,output,s.magSf[3],0.)){return 3;}
 threadIdx.x=0;applySstWallFunctionStateKernel(&s);computeGasPrimitiveGradientsKernel(&s);computeGasEddyViscosityKernel(&s);computeSstGradientsKernel(&s);
 double dt=std::min(CAPTURE_MAX_DT,.2-time);computeSstStabilityNumberKernel(&s,dt,CAPTURE_MAX_CO);
 if(s.sstSourceNumber[0]>CAPTURE_MAX_CO)dt*=(.9*CAPTURE_MAX_CO)/s.sstSourceNumber[0];
 for(int f=0;f<6;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
 double flux=0,div=0;for(int f=0;f<6;++f){double sign=s.faceOwner[f]==0?-1:1;flux+=sign*s.sstPhiRhoK[f];div-=sign*s.gasPhiRho[f]/sstFaceDensity(s,f);}
 Real sk,so;sstSourcesForCell(s,0,div/s.V[0],sk,so);const double previous=s.rhoK[0];threadIdx.x=0;applySstFluxAndSourceKernel(&s,dt);
 if(s.gasSpecies.cellStatus[0]){std::cerr<<"advance fail "<<s.gasSpecies.cellStatus[0]<<" at "<<time<<"\n";return 4;}
 if(s.rhoK[0]<=s.rho[0]*s.sstKMin){std::cerr<<"floor used\n";return 5;}
 sourceIntegral+=dt*s.V[0]*sk;faceIntegral+=dt*flux;
 maxBudget=std::max(maxBudget,std::abs((s.rhoK[0]-previous)*s.V[0]-dt*(s.V[0]*sk+flux)));
 maxNumber=std::max(maxNumber,double(s.sstSourceNumber[0]));minDt=std::min(minDt,dt);recoverSstPrimitivesKernel(&s);
 minK=std::min(minK,double(s.k[0]));time+=dt;++steps;
 if(steps>1000000){std::cerr<<"step explosion\n";return 6;}
}
const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
std::ostringstream diagnostic;diagnostic.precision(17);if(m.in.integrateOwnerKSource)diagnostic<<output.integratedKSource;else diagnostic<<"null";
std::cout.precision(17);std::cout<<"{\"time\":"<<time<<",\"steps\":"<<steps<<",\"wall_solves\":"<<steps<<",\"wall_newton_iterations\":"<<wallIterations<<",\"seconds\":"<<elapsed<<",\"minimum_dt\":"<<minDt<<",\"minimum_k\":"<<minK<<",\"final_k\":"<<s.k[0]<<",\"omega\":"<<s.omega[0]<<",\"wall_tau\":"<<output.traction[0]<<",\"profile_source_diagnostic\":"<<diagnostic.str()<<",\"source_impulse\":"<<sourceIntegral<<",\"face_impulse\":"<<faceIntegral<<",\"global_budget_residual\":"<<s.rhoK[0]*s.V[0]-initial-sourceIntegral-faceIntegral<<",\"max_step_budget_residual\":"<<maxBudget<<",\"max_source_number\":"<<maxNumber<<"}\n";
}
'''.replace('SETUP','\n'.join(setup)).replace('CAPTURE_MAX_DT',format(max_dt,'.17g')).replace('CAPTURE_MAX_CO',format(max_co,'.17g'))
pre=(ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes')[0]
out=Path(os.environ.get('SST_WALL_RESULTS',HERE/'results'));out.mkdir(exist_ok=True)
source=out/'owner32.cpp';source.write_text(pre+body);exe=out/'owner32'
subprocess.run(['g++','-O2','-std=c++17','-I'+str(ROOT),'-I'+str(ROOT/'tests/gas_transport'),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],check=True)
p=subprocess.run([str(exe)],capture_output=True,text=True);(out/'owner32.log').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr);p.check_returncode()
result=json.loads(p.stdout);result['outer_dt_cap']=max_dt;result['sst_prediction_target_courant']=max_co;result['source_diagnostic_enabled']='SST_WALL_DIAGNOSTICS' in os.environ;result['scope']='captured owner32 six-face SST control-volume evolution with fixed gas/donor boundary data; not a full gasUGKP or CHMT run';(out/'owner32.json').write_text(json.dumps(result,indent=2))
