
// Appended to the existing gas-state host scaffold and actual fixture() body.
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
int main(int argc,char**argv){
 bool perf=argc>1&&std::string(argv[1])=="perf";double samples[5]={};int checks=0;bool first=true;
 std::cout<<std::setprecision(17);
 if(!perf)std::cout<<"{\"family\":\"face\",\"precision\":"<<8*sizeof(Real)<<",\"mode\":\"correctness\",\"measurements\":{";
 for(int flux:{1,2})for(int limiter:{0,1,2})for(bool native:{false,true}){
 State s;initialise(s);s.gasFluxScheme=flux;s.gasReconstruction=1;s.gasLimiter=limiter;
 auto*meta=const_cast<ugkwp::SpeciesThermoData<double>*>(s.gasSpecies.thermo.species);
 meta[1].molarMass=.044;
 s.riemannBoundaryKind[1]=0;s.riemannBoundaryPFix[1]=1;s.riemannBoundaryTFix[1]=1;
 s.riemannBoundaryP[1]=101325;s.riemannBoundaryT[1]=500;
 s.gasSpecies.compositionBoundaryFixed[1]=1;s.gasSpecies.boundaryMassFraction[1]=.25;s.gasSpecies.boundaryMassFraction[4]=.75;
 s.riemannBoundaryKind[2]=0;
 for(int c=0;c<2;++c){threadIdx.x=c;recoverGasPrimitivesKernel(&s);computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);}
 auto op=[&](){for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,1e-6,native);}};
 op();
 GasPrimDevice owner{s.rho[0],s.Ux[0],s.Uy[0],s.Uz[0],s.p[0],s.Tgas[0]};
 auto boundary=riemannBoundaryState(s,1,owner);
 Real Y[2]={Real(.25),Real(.75)};
 Real expectedR=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);
 ck(std::abs(double(boundary.rho)-101325/(double(expectedR)*500))<(sizeof(Real)==4?5e-4:1e-9),"fixed boundary uses independent composition R");
 if(perf){for(int j=0;j<5;++j){auto a=std::chrono::steady_clock::now();for(int r=0;r<10000;++r)op();auto b=std::chrono::steady_clock::now();samples[j]+=std::chrono::duration<double>(b-a).count();}}
 else{
 std::string key=std::to_string(flux)+"_"+std::to_string(limiter)+"_"+std::to_string(native);
 auto emit=[&](std::string name,double x){ck(std::isfinite(x),"nonfinite face output");if(!first)std::cout<<",";first=false;std::cout<<"\""<<key<<"_"<<name<<"\":"<<x;++checks;};
 emit("boundary_T",boundary.T);emit("boundary_R",expectedR);emit("boundary_rho",boundary.rho);
 for(int c=0;c<2;++c){emit("T"+std::to_string(c),s.Tgas[c]);emit("R"+std::to_string(c),s.gasSpecies.gasConstant[c]);emit("cv"+std::to_string(c),s.gasSpecies.heatCapacity[c]-s.gasSpecies.gasConstant[c]);emit("e"+std::to_string(c),s.rhoE[c]/s.rho[c]);}

 for(int f=0;f<3;++f){
 GasPrimDevice q;Real y[2];bool evaluated=false;
 ck(reconstructGasMixtureFace(s,s.faceOwner[f],f,q,y,native,&evaluated)&&evaluated,"face reconstruction");
 Real faceR=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
 Real faceE=ugkwp::mixtureEnergy(y,q.T,s.gasSpecies.thermo);
 Real faceCv=ugkwp::mixtureHeatCapacity(y,q.T,s.gasSpecies.thermo);
 emit("faceT"+std::to_string(f),q.T);emit("faceR"+std::to_string(f),faceR);
 emit("face_e"+std::to_string(f),faceE);emit("face_cv"+std::to_string(f),faceCv);
 double eos=double(q.rho)*faceR*q.T;
 ck(std::abs(eos-q.p)<(sizeof(Real)==4?5e-4:1e-9)*(1+std::abs(double(q.p))),"reconstructed face EOS");
 }
 for(int f=0;f<3;++f){ck(s.gasSpecies.faceStatus[f]==0,"face status");emit("mass"+std::to_string(f),s.gasPhiRho[f]);emit("energy"+std::to_string(f),s.gasPhiRhoE[f]);emit("species0_"+std::to_string(f),s.gasSpecies.flux[f]);emit("species1_"+std::to_string(f),s.gasSpecies.flux[3+f]);
 double err=double(s.gasSpecies.flux[f])+s.gasSpecies.flux[3+f]-s.gasPhiRho[f];
 ck(std::abs(err)<(sizeof(Real)==4?5e-4:1e-9)*(1+std::abs(double(s.gasPhiRho[f]))),"species mass closure");}
 }
 }

 if(!perf)for(int scheme:{1,2})cp3_refinement::verify(scheme,[&](const char*name,double x){
 ck(std::isfinite(x),"nonfinite refinement error");
 if(!first)std::cout<<",";first=false;std::cout<<"\"refinement_"<<scheme<<"_"<<name<<"\":"<<x;++checks;
 });
 if(perf){std::cout<<"{\"family\":\"face\",\"precision\":"<<8*sizeof(Real)<<",\"mode\":\"perf\",\"backend\":\"host_arithmetic_not_GPU\",\"samples_seconds\":[";for(int j=0;j<5;++j)std::cout<<(j?",":"")<<samples[j];std::cout<<"],\"iterations\":10000,\"worksize\":36,\"setup/reset_excluded\":true,\"cache_bytes\":0}\n";}
 else std::cout<<"},\"checks\":"<<checks<<"}\n";
}
