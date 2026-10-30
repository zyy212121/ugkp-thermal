#include "gasTransport/GasWallMatching.H"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
using namespace ugkwp;
struct State {
 int nCells=9;
 double rho[9]{},rhoUx[9]{},rhoUy[9]{},rhoUz[9]{},rhoE[9]{},Tgas[9]{},rhoK[9]{},rhoOmega[9]{};
 double partial[18]{};
 GasSpeciesState<double,2> gasSpecies;
 ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
};
UGKWP_GAS_HD bool probe(int test,double& temperature){
 SpeciesThermoData<double> sp[2];double coef[6]={1004.675,0,0,819.27,0,0};
 for(int j=0;j<2;++j){sp[j].coefficientOffset=3*j;sp[j].molarMass=j?.0440095:.02896520682164515;sp[j].minTemperature=1;sp[j].maxTemperature=5000;}
 State s;s.gasSpecies.thermo.species=sp;s.gasSpecies.thermo.coefficients=coef;s.gasSpecies.thermo.coefficientCount=6;s.gasSpecies.rho=s.partial;
 const double data[9][7]={
  {1.2879923070972847,171.92432662982279,0.16651567115872418,76.873358233212784,-6.167793361816126,7.0078583014221153e-20,38403.080688668488},
  {0.22335413788547431,174.66082846114611,0.18462887227199323,84.242869006831114,-3.6877469153947362,-3.1408232077785461e-20,42397.653491183301},
  {-0.31814115634042028,157.65182370462489,0.24404281809137887,117.82600865560745,-38.438504048556808,3.6224074413527612e-19,59080.642341292303},
  {0.22335413788546962,172.55984410632882,0.16710348002278327,76.589998484210582,-5.6840278022471011,1.6851473486320794e-19,38341.72122936081},
  {0.029439630250025706,168.98839070135054,0.15027902290107692,66.720414907923029,-9.7013327450070275,5.9905666935244392e-19,33348.679841393481},
  {-0.3827793255522402,156.91977454640673,0.17785101537513132,85.754560611675288,-9.1462647728405972,2.7080099717647425e-19,40937.074976901924},
  {0.29011996407661161,153.01613013997718,0.23762183797697423,123.87653918420517,-9.9549973679627435,3.8003923490162001e-19,58590.88942590043},
  {-0.38277932555222738,151.61878852029358,0.43058377327630809,228.18284538224364,-73.407198836235281,8.1757277169544916e-19,113568.61035352635},
  {0.029439630250022164,172.15384774135475,0.23375699121543675,104.23413046648226,-40.688471947928235,3.4076639004603042e-19,55659.385593382249}
 };

 int cells[9];double weights[9];
 for(int i=0;i<9;++i){cells[i]=i;weights[i]=data[i][0];s.Tgas[i]=data[i][1];s.rho[i]=s.partial[i]=data[i][2];s.rhoUx[i]=data[i][3];s.rhoUy[i]=data[i][4];s.rhoUz[i]=data[i][5];s.rhoE[i]=data[i][6];}
 int count=9;
 if(test){
  count=2;weights[0]=2;weights[1]=-1;
  sp[0].minTemperature=sp[1].minTemperature=100;
  if(test==3)s.gasSpecies.thermoControls.minTemperature=200;
  for(int i=0;i<2;++i){
   const double T=test==2?100.:(i==0?300.:(test==1?400.:600.));
   s.rho[i]=s.partial[i]=1;s.rhoUx[i]=50;s.rhoUy[i]=s.rhoUz[i]=0;s.Tgas[i]=T;
   s.rhoE[i]=(coef[0]-universalGasConstant<double>()/sp[0].molarMass)*T+1250;
  }
 }
 const State saved=s;
 gaswall::WallMatchingState<double,2> out;double pressure;
 bool ok=recoverWallMatching(s,cells,weights,count,false,out,pressure);
 temperature=out.temperature;
 for(int i=0;i<9;++i)if(s.rhoE[i]!=saved.rhoE[i]||s.rho[i]!=saved.rho[i]||s.partial[i]!=saved.partial[i])return false;
 double expected=test==0?1.:(test==1?200.:(test==2?100.:200.));
 return ok&&std::isfinite(pressure)&&pressure>0&&std::abs(temperature-expected)<1e-8
     &&out.massFraction[0]==1&&out.massFraction[1]==0;
}
#if defined(__CUDACC__)
__global__ void deviceProbe(int* ok,double* t){int i=threadIdx.x;if(i<4)ok[i]=probe(i,t[i]);}
#endif
int main(){
 for(int i=0;i<4;++i){double t=0;bool ok=probe(i,t);std::cout<<"host case="<<i<<" T="<<t<<" ok="<<ok<<std::endl;assert(ok);}
#if defined(__CUDACC__)
 int* ok;double* t;assert(cudaMallocManaged(&ok,4*sizeof(int))==cudaSuccess);assert(cudaMallocManaged(&t,4*sizeof(double))==cudaSuccess);
 deviceProbe<<<1,4>>>(ok,t);assert(cudaGetLastError()==cudaSuccess);assert(cudaDeviceSynchronize()==cudaSuccess);
 for(int i=0;i<4;++i){std::cout<<"cuda case="<<i<<" T="<<t[i]<<" ok="<<ok[i]<<std::endl;assert(ok[i]);}
 assert(cudaFree(ok)==cudaSuccess);assert(cudaFree(t)==cudaSuccess);
#endif
}
