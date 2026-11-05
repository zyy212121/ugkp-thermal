#include <cuda_runtime.h>
#include "gpu/MaterialReactionKernels.cuh"
#include "tests/TestSupport.H"
#include <cstdio>
#include <vector>
using namespace chmt;
void cudaCheck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* allocate(int n){T* p=nullptr;cudaCheck(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;++i)new(p+i)T{};return p;}
// Original adapter update retained here to distinguish the unchanged interior
// from the explicitly improved common-extent exhaustion representation.
bool originalStep(const SolidQ& base,double volume,const PhysicsConfig& p,
    const std::vector<dry::Reaction<Nc,Ns>>& network,double maximumStep,double cfl,
    SolidQ& output,double& step,bool& active,bool* numerical){
    ReactionStepControl control;
    if(!reactionStepControl(base,volume,p,cfl,control,numerical,dryAdapter::CoreReactionRates{}))return false;
    active=control.active;step=active?minValue(maximumStep,control.dtLimit):maximumStep;
    output=base;if(!active)return true;auto q=dryAdapter::state(base);
    const auto source=dry::reactionSource(q,network,volume,dryAdapter::condensed(p),control.temperature);
    double fraction=1;
    for(int c=0;c<Nc;++c)if(source.condensedMass[c]<0){const double available=q.condensedMass[c]/(-step*source.condensedMass[c]);if(available<fraction)fraction=available;}
    for(int s=0;s<Ns;++s)if(source.poreMass[s]<0){const double available=q.poreMass[s]/(-step*source.poreMass[s]);if(available<fraction)fraction=available;}
    for(int c=0;c<Nc;++c)q.condensedMass[c]+=step*fraction*source.condensedMass[c];
    for(int s=0;s<Ns;++s)q.poreMass[s]+=step*fraction*source.poreMass[s];
    for(int c=0;c<Nc;++c)output.condensed[c]=q.condensedMass[c];
    for(int s=0;s<Ns;++s)output.pore[s]=q.poreMass[s];
    std::array<double,Nc> concentration{};std::array<double,Ns> poreConcentration{};
    for(int c=0;c<Nc;++c)concentration[c]=base.condensed[c]/volume;
    for(std::size_t r=0;r<network.size();++r)output.progress[r]+=step*fraction*volume*dry::reactionRate(network[r],concentration,poreConcentration,control.temperature);
    MaterialPrimitive endpoint;if(!recoverMaterial(output,volume,p,endpoint,nullptr,-1,numerical))return false;
    output.porosity=endpoint.porosity;return true;
}
// Frozen 88bc5434 CPU controller: independent loop, host vector mechanism,
// and the original dry adapter. The production CPU/GPU controller is separate.
MaterialReactionStatus originalCpu(const SolidQ& base,double volume,const PhysicsConfig& p,
    double interval,double maxStep,double cfl,int maximum,SolidQ& out){
    MaterialReactionStatus status;
    if(!(interval>0)||!chmt::finite(interval)||maxStep<0||!chmt::finite(maxStep)||!(chmt::finite(cfl)&&cfl>0)||maximum<=0){status.code=MaterialReactionCode::InvalidControls;return status;}
    SolidQ state=base;MaterialPrimitive material;
    if(!recoverMaterial<false>(state,volume,p,material,nullptr,-1,&status.numericalFailure)){status.code=MaterialReactionCode::BaseRecovery;return status;}
    if(!p.enableReactions){out=base;return status;}
    ReactionStepControl initialControl;
    if(!reactionStepControl(state,volume,p,cfl,initialControl,&status.numericalFailure,dryAdapter::CoreReactionRates{})){status.code=MaterialReactionCode::LocalRecovery;return status;}
    const auto network=dryAdapter::mechanism(p);
    double elapsed=0;std::uint64_t count=0;
    while(elapsed<interval){
        if(count==static_cast<std::uint64_t>(maximum))reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,30900,count,maximum,count);
        if(count==std::numeric_limits<std::uint64_t>::max()){status.code=MaterialReactionCode::CounterOverflow;return status;}
        const double maximumStep=minValue(interval-elapsed,maxStep>0?maxStep:interval);
        SolidQ trial;double dt=0;bool active=false;
        if(!originalStep(state,volume,p,network,maximumStep,cfl,trial,dt,active,&status.numericalFailure)){status.code=MaterialReactionCode::LocalRecovery;return status;}
        if(!active){out=state;status.acceptedSteps=count;return status;}
        if(!chmt::finite(dt)||!(dt>0)||!chmt::finite(elapsed+dt)||elapsed+dt==elapsed){status.code=MaterialReactionCode::NonadvancingStep;return status;}
        state=trial;elapsed+=dt;++count;
        if(interval-elapsed<=8*std::numeric_limits<double>::epsilon()*interval)elapsed=interval;
    }
    out=state;status.acceptedSteps=count;return status;
}
PhysicsConfig config(int variant){auto p=chmt_test::physics();p.enableReactions=variant!=0;p.nReactions=2;
    for(auto& x:p.condensed){x.rho=1000;x.cp0=1000;x.cp1=0;x.e0=0;}
    for(auto& x:p.species){x.R=287;x.cp0=1287;x.cp1=0;x.e0=0;}
    auto& a=p.reactions[0];a.A=2;a.order[0]=1;a.condensedNu[0]=-1;a.condensedNu[1]=1;
    auto& b=p.reactions[1];b.A=1000;b.order[1]=1;b.condensedNu[1]=-1;b.gasNu[0]=1;
    if(variant==2){b={};b.A=7;b.order[0]=1;b.condensedNu[0]=-1;b.gasNu[0]=1;a.A=3;}
    if(variant==3){p.nReactions=1;a.A=1/std::sqrt(.001);a.order[0]=.5;}
    if(variant==4){p.nReactions=1;p.condensed[0].e0=300000;p.condensed[0].cp0=700;p.condensed[0].cp1=.4;
        p.condensed[1].e0=-400000;p.condensed[1].cp0=1200;p.condensed[1].cp1=.15;
        p.species[0].e0=-200000;p.species[0].cp0=1000;p.species[0].cp1=.25;
        a.condensedNu[1]=.65;a.gasNu[0]=.35;a.temperaturePower=.5;a.activationEnergy=55000;
        a.A=.6*std::exp(55000/(universalGasConstant*650))/std::sqrt(650.);}
    if(variant==5){a.A=300;b.A=700;b.gasNu[0]=0;b.condensedNu[0]=1;}
    if(variant==7)p.nReactions=Nr+1;
    return p;
}
using DryState=dry::MaterialState<Nc,Ns>;
__global__ void extentBalanceKernel(DryState* state,const DryState* source,double* fraction){const int i=threadIdx.x;if(i<3)fraction[i]=dry::reactionEuler(state[i],source[i],1);}
void checkExtentBalance(){
    auto* state=allocate<DryState>(3);auto* source=allocate<DryState>(3);auto* fraction=allocate<double>(3);
    // A condensed/pore tie shares exactly one extent; an arbitrarily small
    // nonlimiting inventory is retained. A separate interior case is unchanged.
    state[0].condensedMass[0]=.25;state[0].poreMass[0]=.5;state[0].poreMass[1]=1e-280;state[0].energy=17;
    source[0].condensedMass[0]=-2;source[0].poreMass[0]=-4;source[0].condensedMass[1]=6;
    state[1]=state[0];source[1].condensedMass[0]=-2;source[1].poreMass[0]=-1;source[1].condensedMass[1]=3;
    state[2]=state[0];source[2].condensedMass[0]=-.1;source[2].poreMass[0]=-.2;source[2].condensedMass[1]=.3;
    DryState expected[3];double expectedFraction[3];
    for(int i=0;i<3;++i){expected[i]=state[i];expectedFraction[i]=dry::reactionEuler(expected[i],source[i],1);}
    extentBalanceKernel<<<1,32>>>(state,source,fraction);cudaCheck(cudaDeviceSynchronize());
    for(int i=0;i<3;++i){chmt_test::check(fraction[i]==expectedFraction[i],"unchanged common extent");
        for(int c=0;c<Nc;++c)chmt_test::check(state[i].condensedMass[c]==expected[i].condensedMass[c],"extent donor CPU/CUDA exact");
        for(int s=0;s<Ns;++s)chmt_test::check(state[i].poreMass[s]==expected[i].poreMass[s],"extent pore CPU/CUDA exact");
        chmt_test::check(state[i].poreMass[1]==1e-280&&state[i].energy==17,"nonlimiting tiny mass and formation energy preserved");
        double total=0;for(auto x:state[i].condensedMass)total+=x;for(auto x:state[i].poreMass)total+=x;
        chmt_test::near(total,.75,"common-extent total mass/element balance",0,2e-16);
    }
    chmt_test::check(fraction[0]==.125&&state[0].condensedMass[0]==0&&state[0].poreMass[0]==0,"all tied original limiting donors exhausted");
    chmt_test::check(fraction[1]==.125&&state[1].poreMass[0]==.375,"nonlimiting withdrawal follows original update");
    chmt_test::check(fraction[2]==1&&state[2].condensedMass[0]==.25-.1&&state[2].poreMass[0]==.5-.2,"unrestricted interior unchanged");
    cudaCheck(cudaFree(fraction));cudaCheck(cudaFree(source));cudaCheck(cudaFree(state));
}
int main(){std::setvbuf(stdout,nullptr,_IONBF,0);cudaCheck(cudaDeviceSetLimit(cudaLimitStackSize,65536));checkExtentBalance();constexpr int n=257;
    auto* p=allocate<PhysicsConfig>(1);auto* q=allocate<SolidQ>(n);auto* out=allocate<SolidQ>(n);
    auto* volumes=allocate<double>(n);auto* status=allocate<MaterialReactionStatus>(n);
    auto* diagnostics=allocate<ugkp::diagnostics::DeviceContext>(1);
    for(int variant=0;variant<8;++variant){
        *p=config(variant);*diagnostics={};p->diagnostics=diagnostics;
        ugkp::diagnostics::DeviceContext cpuDiagnostics{};auto cpuPhysics=*p;cpuPhysics.diagnostics=&cpuDiagnostics;
        double interval=variant==3?2:variant==4?.5:variant==5?.1:variant==2?.1:.001;
        if(variant==6)interval=0;
        const double maxStep=variant==2?.003:0,cfl=.1;const int budget=1;
        std::vector<SolidQ> expected(n);std::vector<MaterialReactionStatus> expectedStatus(n);
        for(int c=0;c<n;++c){q[c]={};volumes[c]=.001*(1+.001*c);
            q[c].condensed[0]=variant==4?.72:.8;q[c].condensed[1]=variant==4?.18:(variant==5?.1:(c%2?1e-15:0));q[c].pore[0]=.1;
            // Alternating inactive cells exercise nonuniform independent clocks.
            if(variant==1&&c%7==0){q[c].condensed[0]=q[c].condensed[1]=0;q[c].pore[0]=.9;}
            const double temperature=variant==4?650:700;
            for(int k=0;k<Nc;++k)q[c].energy+=q[c].condensed[k]*condensedE(p->condensed[k],temperature);
            for(int s=0;s<Ns;++s)q[c].energy+=q[c].pore[s]*speciesE(p->species[s],temperature);
            q[c].porosity=.1;out[c]=q[c];out[c].progress[0]=123;expected[c]=out[c];
            expectedStatus[c]=originalCpu(q[c],volumes[c],cpuPhysics,interval,maxStep,cfl,budget,expected[c]);
            SolidQ shared=out[c];const auto host=advanceMaterialReactionsMath(q[c],volumes[c],cpuPhysics,interval,maxStep,cfl,budget,shared);
            if(variant!=3){
            chmt_test::check(host.code==expectedStatus[c].code&&host.acceptedSteps==expectedStatus[c].acceptedSteps,"CPU controller status parity");
            for(int k=0;k<Nc;++k)chmt_test::check(shared.condensed[k]==expected[c].condensed[k],"CPU condensed bitwise parity");
            for(int s=0;s<Ns;++s)chmt_test::check(shared.pore[s]==expected[c].pore[s],"CPU pore bitwise parity");
            for(int r=0;r<Nr;++r)chmt_test::check(shared.progress[r]==expected[c].progress[r],"CPU progress bitwise parity");
            chmt_test::check(shared.energy==expected[c].energy&&shared.porosity==expected[c].porosity,"CPU energy/EOS bitwise parity");
            }else{
                chmt_test::check(host.code==MaterialReactionCode::Success&&shared.condensed[0]==0,"shared common-extent exact exhaustion");
                if(expectedStatus[c].code==MaterialReactionCode::Success){
                    for(int k=0;k<Nc;++k)chmt_test::near(shared.condensed[k],expected[c].condensed[k],"old CPU exhaustion roundoff");
                    for(int k=0;k<Ns;++k)chmt_test::near(shared.pore[k],expected[c].pore[k],"old CPU pore roundoff");
                }
                expected[c]=shared;expectedStatus[c]=host;
            }
        }
        advanceMaterialReactionsKernel<<<(n+127)/128,128>>>(q,volumes,n,p,interval,maxStep,cfl,budget,out,status);
        cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
        for(int c=0;c<n;++c){
            if(status[c].code!=expectedStatus[c].code)std::fprintf(stderr,"status variant=%d cell=%d GPU=%d CPU=%d fatal=%d observed=%.17g\n",variant,c,int(status[c].code),int(expectedStatus[c].code),int(diagnostics->firstFatal.code),diagnostics->firstFatal.observed);
            chmt_test::check(status[c].code==expectedStatus[c].code,"CUDA status parity");
            chmt_test::check(status[c].numericalFailure==expectedStatus[c].numericalFailure,"CUDA failure origin parity");
            if(status[c].acceptedSteps!=expectedStatus[c].acceptedSteps)std::fprintf(stderr,"count variant=%d cell=%d GPU=%llu CPU=%llu\n",variant,c,(unsigned long long)status[c].acceptedSteps,(unsigned long long)expectedStatus[c].acceptedSteps);
            chmt_test::check(status[c].acceptedSteps==expectedStatus[c].acceptedSteps,"CUDA independent reaction step count");
            for(int k=0;k<Nc;++k)chmt_test::near(out[c].condensed[k],expected[c].condensed[k],"CUDA condensed");
            for(int s=0;s<Ns;++s)chmt_test::near(out[c].pore[s],expected[c].pore[s],"CUDA pore");
            for(int r=0;r<Nr;++r)chmt_test::near(out[c].progress[r],expected[c].progress[r],"CUDA progress");
            chmt_test::near(out[c].porosity,expected[c].porosity,"CUDA EOS porosity");
            chmt_test::check(out[c].energy==q[c].energy,"formation-inclusive energy unchanged");
            if(status[c].code==MaterialReactionCode::Success){double initial=0,final=0;
                for(int k=0;k<Nc;++k){initial+=q[c].condensed[k];final+=out[c].condensed[k];chmt_test::check(out[c].condensed[k]>=0,"nonnegative condensed");}
                for(int s=0;s<Ns;++s){initial+=q[c].pore[s];final+=out[c].pore[s];chmt_test::check(out[c].pore[s]>=0,"nonnegative pore");}
                chmt_test::near(final,initial,"reaction mass/element conservation",1e-12,1e-13);
                if(variant==3)chmt_test::check(out[c].condensed[0]==0,"fractional extinction exact");
            }
        }
        if(variant>=1&&variant<=5)chmt_test::check(diagnostics->warnings[1].count>0&&!diagnostics->fatalClaim,"warning budget permits full integration");
        std::printf("PASS resident material reaction variant=%d cells=%d steps=%llu\n",variant,n,(unsigned long long)status[1].acceptedSteps);
    }
    cudaCheck(cudaFree(diagnostics));cudaCheck(cudaFree(status));cudaCheck(cudaFree(volumes));cudaCheck(cudaFree(out));cudaCheck(cudaFree(q));cudaCheck(cudaFree(p));
}
