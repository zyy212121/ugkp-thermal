#include "applications/CHT/gpu/GpuChtParticleRestart.H"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){using namespace Foam::gpuFshRestart; if(argc<3)return 2;MutableView v{};ConstView c{};c.count=v.count=2;
double px[128]={};c.px=v.px=px;
double py[128]={};c.py=v.py=py;
double pz[128]={};c.pz=v.pz=pz;
double pux[128]={};c.pux=v.pux=pux;
double puy[128]={};c.puy=v.puy=puy;
double puz[128]={};c.puz=v.puz=puz;
double pT[128]={};c.pT=v.pT=pT;
double pTheta[128]={};c.pTheta=v.pTheta=pTheta;
double pd[128]={};c.pd=v.pd=pd;
double pm[128]={};c.pm=v.pm=pm;
double pDepositionArea[128]={};c.pDepositionArea=v.pDepositionArea=pDepositionArea;
double pContactDuration[128]={};c.pContactDuration=v.pContactDuration=pContactDuration;
double pContactMaximumArea[128]={};c.pContactMaximumArea=v.pContactMaximumArea=pContactMaximumArea;
double pContactPeakFraction[128]={};c.pContactPeakFraction=v.pContactPeakFraction=pContactPeakFraction;
double pColdRingSolidMass[128]={};c.pColdRingSolidMass=v.pColdRingSolidMass=pColdRingSolidMass;
double pColdFrozenArea[128]={};c.pColdFrozenArea=v.pColdFrozenArea=pColdFrozenArea;
double pColdContactAge[128]={};c.pColdContactAge=v.pColdContactAge=pColdContactAge;
double pCold2DRingContactAge[128]={};c.pCold2DRingContactAge=v.pCold2DRingContactAge=pCold2DRingContactAge;
double pCold2DFrozenArea[128]={};c.pCold2DFrozenArea=v.pCold2DFrozenArea=pCold2DFrozenArea;
double puxOld[128]={};c.puxOld=v.puxOld=puxOld;
double puyOld[128]={};c.puyOld=v.puyOld=puyOld;
double puzOld[128]={};c.puzOld=v.puzOld=puzOld;
double pThermalEnthalpyResidual[128]={};c.pThermalEnthalpyResidual=v.pThermalEnthalpyResidual=pThermalEnthalpyResidual;
float pColdNodeSpecificEnthalpy[128]={};c.pColdNodeSpecificEnthalpy=v.pColdNodeSpecificEnthalpy=pColdNodeSpecificEnthalpy;
float pCold2DNodeSpecificEnthalpy[128]={};c.pCold2DNodeSpecificEnthalpy=v.pCold2DNodeSpecificEnthalpy=pCold2DNodeSpecificEnthalpy;
std::int32_t pCellId[128]={};c.pCellId=v.pCellId=pCellId;
std::int32_t pStatus[128]={};c.pStatus=v.pStatus=pStatus;
std::int32_t pStuckFaceId[128]={};c.pStuckFaceId=v.pStuckFaceId=pStuckFaceId;
std::uint64_t pRng[128]={};c.pRng=v.pRng=pRng;
std::uint64_t pOrigId[128]={};c.pOrigId=v.pOrigId=pOrigId;
std::uint8_t pStuck[128]={};c.pStuck=v.pStuck=pStuck;
if(std::string(argv[1])=="create"){
 if(argc!=7)return 2;for(int i=0;i<2;++i){px[i]=.2+.1*i;py[i]=pz[i]=.25;pT[i]=std::stod(argv[3+i]);pThermalEnthalpyResidual[i]=std::stod(argv[5+i]);pm[i]=i?.7:.3;pd[i]=.001;pCellId[i]=0;pStatus[i]=1;pStuckFaceId[i]=-1;pRng[i]=12345+i;pOrigId[i]=42+i;}
 std::ofstream f(argv[2],std::ios::binary);writeCurrent(f,c);
}else{std::ifstream f(argv[2],std::ios::binary);std::string marker;int count,bound;f>>marker>>count>>bound;if(marker!=currentFormatName||count!=2)return 3;readCurrentPayload(f,bound,v);for(int i=0;i<2;++i)std::cout<<std::setprecision(17)<<pOrigId[i]<<" "<<pT[i]<<" "<<pThermalEnthalpyResidual[i]<<" "<<pm[i]<<"\n";}
}
