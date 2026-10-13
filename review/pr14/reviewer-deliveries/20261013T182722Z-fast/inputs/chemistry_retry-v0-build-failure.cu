#include <cuda_runtime.h>
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
using namespace ugkwp;
#define CU(call) do { auto e=(call);if(e!=cudaSuccess){std::cerr<<"CUDA_ERROR "<<__LINE__<<" "<<cudaGetErrorString(e)<<"\n";std::exit(86);} } while(0)
std::vector<void*> allocations;
template<class T>T* upload(const T* a,int n){T* p=nullptr;CU(cudaMalloc(&p,n*sizeof(T)));CU(cudaMemcpy(p,a,n*sizeof(T),cudaMemcpyHostToDevice));allocations.push_back(p);return p;}
template<class T>T read(T* p){T v;CU(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;GasBudgetAuditState gasBudgetAudit;};
struct Model {
 SpeciesThermoData<double> sp[2];double coef[6]={1000,0,0,1000,0,0},elements[2]={1,1},basis[2]={-1,1};
 GasReactionData<double> rx[1];GasStoichTerm<double> re[1],pr[1];SpeciesThermoView<double,2> t;GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantOffset=0;rx[0].reactantCount=1;rx[0].productOffset=0;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1; }
};
int main(int argc,char**argv){
 const bool refine=argc>1?std::atoi(argv[1])!=0:true;const double dt=argc>2?std::atof(argv[2]):2e-6;const int maxSteps=argc>3?std::atoi(argv[3]):1;
 Model model;State state{};state.nCells=1;state.gasSpecies.mode=GasMode::MixtureChemistry;
 double mass[2]={28,0};double energy=mixtureEnergy(mass,700.,model.t);const double canary=918273.125;
 double rh[2]={28,canary},zero[2]={0,canary},en[2]={energy,canary},species[3]={28,0,canary},vol[2]={.001,canary},budget[3]={0,0,canary};int cell[2]={0,123456};
 state.rho=upload(rh,2);state.rhoUx=upload(zero,2);state.rhoUy=upload(zero,2);state.rhoUz=upload(zero,2);state.rhoE=upload(en,2);
 state.gasSpecies.rho=upload(species,3);state.gasSpecies.cellStatus=upload(cell,2);
 ChemistryStatus initialStatus[2]{};initialStatus[1].reaction=9876;state.gasSpecies.chemistryStatus=upload(initialStatus,2);
 ChemistryAudit<double,2> audit[2]{};audit[0].acceptedSteps=7;audit[0].integratedTime=99;audit[1].acceptedSteps=123456;
 state.gasSpecies.chemistryAudit=upload(audit,2);state.gasSpecies.thermo=model.t;state.gasSpecies.mechanism=model.m;
 auto &t=state.gasSpecies.thermo;auto &m=state.gasSpecies.mechanism;
 t.species=upload(model.sp,2);t.coefficients=upload(model.coef,6);t.elementComposition=upload(model.elements,2);
 m.reactions=upload(model.rx,1);m.reactants=upload(model.re,1);m.products=upload(model.pr,1);m.stoichiometricBasis=upload(model.basis,2);
 state.gasSpecies.chemistryControls.maximumSteps=maxSteps;state.gasSpecies.chemistryControls.initialStep=1e-6;state.gasSpecies.chemistryControls.maximumStep=1e-6;
 state.gasBudgetAudit.enabled=true;state.gasBudgetAudit.speciesCount=2;state.gasBudgetAudit.chemistrySpecies=upload(budget,3);
 State* device=upload(&state,1);double* volume=upload(vol,2);
#if defined(BASELINE)
 advanceGasChemistryKernel<<<1,32>>>(device,dt,volume);
#else
 advanceGasChemistryKernel<<<1,32>>>(device,dt,volume,refine);
#endif
 CU(cudaGetLastError());CU(cudaDeviceSynchronize());
 CU(cudaMemcpy(species,state.gasSpecies.rho,sizeof(species),cudaMemcpyDeviceToHost));CU(cudaMemcpy(audit,state.gasSpecies.chemistryAudit,sizeof(audit),cudaMemcpyDeviceToHost));
 CU(cudaMemcpy(initialStatus,state.gasSpecies.chemistryStatus,sizeof(initialStatus),cudaMemcpyDeviceToHost));CU(cudaMemcpy(cell,state.gasSpecies.cellStatus,sizeof(cell),cudaMemcpyDeviceToHost));CU(cudaMemcpy(budget,state.gasBudgetAudit.chemistrySpecies,sizeof(budget),cudaMemcpyDeviceToHost));
 double rho[2],mx[2],my[2],mz[2],e[2];CU(cudaMemcpy(rho,state.rho,sizeof(rho),cudaMemcpyDeviceToHost));CU(cudaMemcpy(mx,state.rhoUx,sizeof(mx),cudaMemcpyDeviceToHost));CU(cudaMemcpy(my,state.rhoUy,sizeof(my),cudaMemcpyDeviceToHost));CU(cudaMemcpy(mz,state.rhoUz,sizeof(mz),cudaMemcpyDeviceToHost));CU(cudaMemcpy(e,state.rhoE,sizeof(e),cudaMemcpyDeviceToHost));
 bool safe=species[2]==canary&&rho[1]==canary&&mx[1]==canary&&my[1]==canary&&mz[1]==canary&&e[1]==canary&&cell[1]==123456&&audit[1].acceptedSteps==123456&&initialStatus[1].reaction==9876&&budget[2]==canary;
 std::cout<<std::setprecision(17)<<"{\"nativeRefinement\":"<<int(refine)<<",\"dt\":"<<dt<<",\"maximumSteps\":"<<maxSteps<<",\"rho\":"<<rho[0]<<",\"momentum\":["<<mx[0]<<","<<my[0]<<","<<mz[0]<<"],\"rhoE\":"<<e[0]<<",\"rhoSpecies\":["<<species[0]<<","<<species[1]<<"],\"status\":{\"code\":"<<int(initialStatus[0].code)<<",\"thermoCode\":"<<int(initialStatus[0].thermoCode)<<",\"reaction\":"<<initialStatus[0].reaction<<",\"attemptedSteps\":"<<initialStatus[0].attemptedSteps<<",\"rejectedSteps\":"<<initialStatus[0].rejectedSteps<<",\"nonlinearIterations\":"<<initialStatus[0].nonlinearIterations<<"},\"cellStatus\":"<<cell[0]<<",\"audit\":{\"speciesMassChange\":["<<audit[0].speciesMassChange[0]<<","<<audit[0].speciesMassChange[1]<<"],\"massResidual\":"<<audit[0].massResidual<<",\"maximumElementResidual\":"<<audit[0].maximumElementResidual<<",\"energyResidual\":"<<audit[0].energyResidual<<",\"maximumLocalError\":"<<audit[0].maximumLocalError<<",\"integratedTime\":"<<audit[0].integratedTime<<",\"acceptedSteps\":"<<audit[0].acceptedSteps<<",\"rejectedSteps\":"<<audit[0].rejectedSteps<<",\"nonlinearIterations\":"<<audit[0].nonlinearIterations<<"},\"publishedBudget\":["<<budget[0]<<","<<budget[1]<<"],\"canaries_intact\":"<<(safe?"true":"false")<<"}\n";
 for(void* a:allocations)CU(cudaFree(a));return safe?0:2;
}
