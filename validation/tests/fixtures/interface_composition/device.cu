// Real production helper GPU test. Build twice with UGKWP_GAS_SPECIES=2 and 10.
#include <cuda_runtime.h>
#include "ablation/InterfaceMath.H"
#include <random>
#include <iostream>
#include <vector>
using namespace chmt;
struct Input { PhysicsConfig p; Real bulk[Ns]{},flux[Ns]{}; int kind=0; };
struct Result { ugkp::diagnostics::DeviceContext context{}; Real out[Ns]{},residual=0; unsigned failure=0; bool ok=false; };
__global__ void exercise(Input* in,Result* result,int count){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
    Input& x=in[k];Result& r=result[k];x.p.diagnostics=&r.context;
    for(int i=0;i<Ns;++i)r.out[i]=123;
    r.ok=interfaceComposition(x.flux,x.bulk,101325,300,.01,x.p,1e-12,r.out);
    if(x.kind==1){if(r.ok||!r.context.fatalClaim||r.context.firstFatal.component!=0||r.out[0]!=123)r.failure|=1;return;}
    if(x.kind==2){if(r.context.fatalClaim)r.failure|=2; if(!r.ok){for(int i=0;i<Ns;++i)if(r.out[i]!=123)r.failure|=4;return;}}
    else if(!r.ok)r.failure|=8;
    if(r.ok){Real F[Ns]{},sum=0,tolerance=1e-12;
        if(!compositionResidual(r.out,x.bulk,x.flux,101325,300,.01,x.p,F,nullptr))r.failure|=16;
        for(int i=0;i<Ns;++i){sum+=r.out[i];tolerance+=x.p.tolerances.relativeMass*absValue(x.flux[i]);r.residual=maxValue(r.residual,absValue(F[i]));if(!chmt::finite(r.out[i])||r.out[i]<0||r.out[i]>1)r.failure|=32;}
        if(absValue(sum-1)>1e-12 || (x.kind==0&&r.residual>tolerance))r.failure|=64;
    }
    if(r.context.fatalClaim)r.failure|=128;
}
int main(){int devices=0;cudaError_t status=cudaGetDeviceCount(&devices);if(status!=cudaSuccess||!devices){std::cerr<<"no CUDA device: "<<cudaGetErrorString(status)<<'\n';return 77;}
    const int count=258;std::vector<Input> host(count);std::mt19937_64 gen(74921);std::uniform_real_distribution<double> uniform(0,1);
    for(int k=0;k<count-2;++k){Input& x=host[k];Real y[Ns]{},sb=0,sy=0,R=0;
        for(int i=0;i<Ns;++i){x.bulk[i]=pow(10.,-5*uniform(gen));y[i]=pow(10.,-5*uniform(gen));sb+=x.bulk[i];sy+=y[i];x.p.species[i].R=pow(10.,1+3*uniform(gen));x.p.gasDiffusivity[i]=pow(10.,-6+5*uniform(gen));}
        for(int i=0;i<Ns;++i){x.bulk[i]/=sb;y[i]/=sy;}
        if(k%4==0){for(int i=0;i<Ns;++i)y[i]=i==k%Ns?1:0;}
        if(k%4==1){x.p.gasDiffusivity[0]=0;}
        for(int i=0;i<Ns;++i)R+=y[i]*x.p.species[i].R;
        Real rho=101325/(R*300),J=pow(10.,-5+5*uniform(gen)),S=0;
        for(int i=0;i<Ns;++i)S+=rho*x.p.gasDiffusivity[i]/.01*(y[i]-x.bulk[i]);
        for(int i=0;i<Ns;++i)x.flux[i]=J*y[i]+rho*x.p.gasDiffusivity[i]/.01*(y[i]-x.bulk[i])-y[i]*S;
    }
    host[count-2]=host[0];host[count-2].kind=1;host[count-2].p.species[0].R=0;
    Input& pole=host[count-1];pole.kind=2;for(int i=0;i<Ns;++i){pole.p.species[i].R=1;pole.p.gasDiffusivity[i]=0;}
    if(Ns>=4){pole.p.gasDiffusivity[1]=pole.p.gasDiffusivity[2]=pole.p.gasDiffusivity[3]=1e-308;pole.p.species[1].R=pole.p.species[3].R=1e-310;pole.bulk[0]=pole.bulk[2]=.5;pole.flux[1]=pole.flux[3]=1.5;pole.flux[2]=-1.5;}
    else {pole.bulk[0]=pole.bulk[1]=.5;pole.flux[0]=1;pole.flux[1]=-1;}
    Input* input=nullptr;Result* output=nullptr;cudaMalloc(&input,sizeof(Input)*count);cudaMalloc(&output,sizeof(Result)*count);cudaMemcpy(input,host.data(),sizeof(Input)*count,cudaMemcpyHostToDevice);cudaMemset(output,0,sizeof(Result)*count);
    exercise<<<(count+31)/32,32>>>(input,output,count);status=cudaDeviceSynchronize();if(status!=cudaSuccess){std::cerr<<cudaGetErrorString(status)<<'\n';return 2;}
    std::vector<Result> result(count);status=cudaMemcpy(result.data(),output,sizeof(Result)*count,cudaMemcpyDeviceToHost);if(status!=cudaSuccess)return 3;int bad=0;Real largest=0;
    for(int k=0;k<count;++k){largest=maxValue(largest,result[k].residual);if(result[k].failure){++bad;std::cerr<<"case="<<k<<" flags="<<result[k].failure<<" ok="<<result[k].ok<<" fatal="<<result[k].context.fatalClaim<<'\n';}}
    cudaFree(output);cudaFree(input);std::cout<<"Ns="<<Ns<<" cases="<<count<<" bad="<<bad<<" maxResidual="<<largest<<'\n';return bad?1:0;
}
