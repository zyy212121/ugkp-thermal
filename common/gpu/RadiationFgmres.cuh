#ifndef UGKP_RADIATION_FGMRES_CUH
#define UGKP_RADIATION_FGMRES_CUH
#include "ExecutionGraph.cuh"
#include <functional>
#include <algorithm>
#include <cmath>
namespace ugkp { namespace gpu {
namespace radiationFgmresDetail {
using R=RadiationReal;
constexpr int restart=16;
__device__ R propagatingMax(R a,R b){return isnan(a)?a:isnan(b)?b:fmax(a,b);}
struct Controls { R tolerance=R(1e-6),relativeTolerance=R(0),small=R(1e-30);int maximumIterations=200; const unsigned long long* availableWork=nullptr;const bool* correctPublishedResidual=nullptr; };
enum class StopReason {NotStarted=0,Converged=1,IterationLimit=2,Breakdown=3,NonFinite=4,Stagnation=5,WorkLimit=6};
struct Report {int iterations=0,residualChecks=0;R initialResidual=0,finalResidual=0,relativeResidual=0;StopReason stopReason=StopReason::NotStarted;bool converged=false;int breakdowns=0;bool nonFinite=false;unsigned long long negativeIntensityBits=0;};
struct State {R beta,initialNorm,target;int used,active,changed;unsigned long long workLimit;Report report;};
__device__ R sum(R a,R* shared){int t=threadIdx.x;shared[t]=a;__syncthreads();for(int s=blockDim.x/2;s;s/=2){if(t<s)shared[t]+=shared[t+s];__syncthreads();}return shared[0];}
__device__ R maximum(R a,R* shared){int t=threadIdx.x;shared[t]=a;__syncthreads();for(int s=blockDim.x/2;s;s/=2){if(t<s)shared[t]=propagatingMax(shared[t],shared[t+s]);__syncthreads();}return shared[0];}
__device__ inline bool affordable(const State& state){
 return 3ULL*state.report.iterations+state.report.residualChecks+4<=state.workLimit;
}
__device__ inline bool canIterate(const State& state,Controls controls){
 return state.report.iterations<controls.maximumIterations&&affordable(state);
}
__device__ bool reachedTarget(R maxr,R norm,R den,const State& state,Controls c,bool first){
 if(first&&c.correctPublishedResidual&&*c.correctPublishedResidual&&norm>0)return false;
 return maxr<=c.tolerance*propagatingMax(den,c.small)||(c.relativeTolerance>0&&norm<=c.relativeTolerance*state.initialNorm);
}
__device__ void reportResidual(State* s,R maxr,R norm,R den,bool done,Controls c){
 ++s->report.residualChecks;
 s->report.nonFinite=s->report.nonFinite||!isfinite(maxr)||!isfinite(norm)||!isfinite(den)||!isfinite(s->report.finalResidual);
 s->report.relativeResidual=s->initialNorm>R(0)?norm/s->initialNorm:(norm==R(0)?R(0):R(INFINITY));
 s->report.converged=done&&!s->report.nonFinite;
 s->report.stopReason=s->report.converged?StopReason::Converged:s->report.nonFinite?StopReason::NonFinite:
     s->report.iterations>=c.maximumIterations?(s->report.breakdowns?StopReason::Breakdown:StopReason::IterationLimit):
     !affordable(*s)?StopReason::WorkLimit:s->report.iterations>0&&!s->changed?StopReason::Stagnation:StopReason::NotStarted;
}
__global__ void initialize(State* s,Controls controls){
 if(!threadIdx.x){*s={};s->workLimit=controls.availableWork?*controls.availableWork:~0ULL;}
}
__global__ void residual(int n,const R* rhs,const R* ax,R* v,State* s,Controls c,bool first,cudaGraphConditionalHandle loop){
 extern __shared__ R tmp[];R ss=0,m=0,scale=0;
 for(int i=threadIdx.x;i<n;i+=blockDim.x){R r=rhs[i]-ax[i];v[i]=r;m=propagatingMax(m,fabs(r));scale=propagatingMax(scale,propagatingMax(fabs(rhs[i]),fabs(ax[i])));}
 R maxr=maximum(m,tmp);__syncthreads();
 for(int i=threadIdx.x;i<n;i+=blockDim.x){R q=maxr>0?v[i]/maxr:R(0);ss+=q*q;}
 R norm=maxr*sqrt(sum(ss,tmp));__syncthreads();R den=maximum(scale,tmp);__syncthreads();
 if(!threadIdx.x){s->beta=norm;s->report.finalResidual=maxr/propagatingMax(den,c.small);
 if(first){s->initialNorm=norm;s->report.initialResidual=s->report.finalResidual;}
 s->target=propagatingMax(c.tolerance*propagatingMax(den,c.small),c.relativeTolerance*s->initialNorm);
 bool done=reachedTarget(maxr,norm,den,*s,c,first);
 reportResidual(s,maxr,norm,den,done,c);
  s->active=!done&&!s->report.nonFinite&&(first||s->changed)&&canIterate(*s,c);
 s->used=0;cudaGraphSetConditional(loop,s->active);}
}
__global__ void begin(int n,R* v,R* h,R* g,R* cs,R* sn,State* s){
 for(int i=threadIdx.x;i<n;i+=blockDim.x)v[i]=s->beta>0?v[i]/s->beta:R(0);
 for(int i=threadIdx.x;i<(restart+1)*restart;i+=blockDim.x)h[i]=0;
 for(int i=threadIdx.x;i<=restart;i+=blockDim.x)g[i]=i?R(0):s->beta;
 for(int i=threadIdx.x;i<restart;i+=blockDim.x){cs[i]=0;sn[i]=0;}
}
__global__ void gate(State* s,Controls c,cudaGraphConditionalHandle handle){if(!threadIdx.x)cudaGraphSetConditional(handle,s->active&&canIterate(*s,c));}
__global__ void arnoldi(int n,int j,R* v,R* w,R* h,R* g,R* cs,R* sn,State* s,Controls c){
 extern __shared__ R tmp[];
 for(int pass=0;pass<2;++pass)for(int k=0;k<=j;++k){
 R local=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)local+=v[k*n+i]*w[i];
 R dot=sum(local,tmp);__syncthreads();if(!threadIdx.x)h[j*(restart+1)+k]+=dot;
 for(int i=threadIdx.x;i<n;i+=blockDim.x)w[i]-=dot*v[k*n+i];__syncthreads();}
 R local=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)local=propagatingMax(local,fabs(w[i]));
 R magnitude=maximum(local,tmp);__syncthreads();local=0;
 for(int i=threadIdx.x;i<n;i+=blockDim.x){R q=magnitude>0?w[i]/magnitude:R(0);local+=q*q;}
 R norm=magnitude*sqrt(sum(local,tmp));__syncthreads();
 for(int i=threadIdx.x;i<n;i+=blockDim.x)v[(j+1)*n+i]=norm>0?w[i]/norm:R(0);
 if(!threadIdx.x){R* col=h+j*(restart+1);col[j+1]=norm;
 for(int k=0;k<j;++k){R a=col[k],b=col[k+1];col[k]=cs[k]*a+sn[k]*b;col[k+1]=-sn[k]*a+cs[k]*b;}
 R a=col[j],b=col[j+1],r=hypot(a,b);cs[j]=r>0?a/r:R(1);sn[j]=r>0?b/r:R(0);col[j]=r;col[j+1]=0;
 R gj=g[j];g[j]=cs[j]*gj;g[j+1]=-sn[j]*gj;s->used=j+1;++s->report.iterations;
 if(!isfinite(r)||!isfinite(norm)||!isfinite(g[j+1]))s->report.nonFinite=true;
 if(!(r>0)||!(norm>0))++s->report.breakdowns;
 s->active=isfinite(r)&&r>0&&norm>0&&fabs(g[j+1])>s->target&&canIterate(*s,c);}
}
__device__ void updateFeasibleIntensity(int n,R* x,const R* z,const R* weights,State* state){
 const int first=blockIdx.x*blockDim.x+threadIdx.x;
 const int stride=gridDim.x*blockDim.x;
 bool changed=false;
 R maximumCorrection=0;
 for(int i=first;i<n;i+=stride){
     R value=x[i];
     for(int k=0;k<state->used;++k)value+=z[k*n+i]*weights[k];
     if(isfinite(value)&&value<0){
         maximumCorrection=fmax(maximumCorrection,-value);
         value=0;
     }
     changed|=value!=x[i];
     x[i]=value;
 }
 if(maximumCorrection>0)
     atomicMax(&state->report.negativeIntensityBits,
         static_cast<unsigned long long>(__double_as_longlong(double(maximumCorrection))));
 const int anyChanged=__syncthreads_or(changed);
 if(!threadIdx.x&&anyChanged)atomicExch(&state->changed,1);
}
__global__ void update(int n,R* x,const R* z,const R* h,R* g,State* s){
 if(!threadIdx.x){s->changed=0;for(int k=s->used-1;k>=0;--k){R q=g[k];for(int j=k+1;j<s->used;++j)q-=h[j*(restart+1)+k]*g[j];R d=h[k*(restart+1)+k];g[k]=d!=0?q/d:R(0);}}
 __syncthreads();
 updateFeasibleIntensity(n,x,z,g,s);
}
__global__ void parallelDots(int n,int j,const R* v,const R* w,R* partial){
 extern __shared__ R tmp[];int k=blockIdx.y;R q=0;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)q+=v[k*n+i]*w[i];
 q=sum(q,tmp);if(!threadIdx.x)partial[k*gridDim.x+blockIdx.x]=q;
}
__global__ void finishDots(int blocks,int j,const R* partial,R* dots,R* h){
 extern __shared__ R tmp[];int k=blockIdx.x;R q=0;
 for(int i=threadIdx.x;i<blocks;i+=blockDim.x)q+=partial[k*blocks+i];q=sum(q,tmp);
 if(!threadIdx.x){dots[k]=q;h[j*(restart+1)+k]+=q;}
}
__global__ void subtractDots(int n,int j,const R* v,R* w,const R* dots){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){R q=w[i];for(int k=0;k<=j;++k)q-=dots[k]*v[k*n+i];w[i]=q;}
}
__global__ void vectorStats(int n,const R* rhs,const R* ax,R* output,R* partial,bool residualMode){
 extern __shared__ R tmp[];R m=0,den=0,q=0;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
 R value=residualMode?rhs[i]-ax[i]:ax[i];if(residualMode)output[i]=value;
 m=propagatingMax(m,fabs(value));if(residualMode)den=propagatingMax(den,propagatingMax(fabs(rhs[i]),fabs(ax[i])));}
 R maximumValue=maximum(m,tmp);__syncthreads();
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){R value=residualMode?output[i]:ax[i];R t=maximumValue>0?value/maximumValue:R(0);q+=t*t;}
 R ss=sum(q,tmp);__syncthreads();R dd=maximum(den,tmp);
 if(!threadIdx.x){partial[blockIdx.x]=maximumValue;partial[gridDim.x+blockIdx.x]=ss;partial[2*gridDim.x+blockIdx.x]=dd;}
}
__global__ void finishStats(int blocks,const R* partial,R* stats){
 extern __shared__ R tmp[];R m=0,den=0,q=0;
 for(int i=threadIdx.x;i<blocks;i+=blockDim.x){m=propagatingMax(m,partial[i]);den=propagatingMax(den,partial[2*blocks+i]);}
 R mm=maximum(m,tmp);__syncthreads();
 for(int i=threadIdx.x;i<blocks;i+=blockDim.x){R t=mm>0?partial[i]/mm:R(0);q+=t*t*partial[blocks+i];}
 R ss=sum(q,tmp);__syncthreads();R dd=maximum(den,tmp);
 if(!threadIdx.x){stats[0]=mm;stats[1]=mm*sqrt(ss);stats[2]=dd;}
}
__global__ void finishResidual(const R* stats,State* s,Controls c,bool first,cudaGraphConditionalHandle loop){
 if(threadIdx.x)return;R maxr=stats[0],norm=stats[1],den=stats[2];s->beta=norm;
 s->report.finalResidual=maxr/propagatingMax(den,c.small);
 if(first){s->initialNorm=norm;s->report.initialResidual=s->report.finalResidual;}
 s->target=propagatingMax(c.tolerance*propagatingMax(den,c.small),c.relativeTolerance*s->initialNorm);
 bool done=reachedTarget(maxr,norm,den,*s,c,first);
 reportResidual(s,maxr,norm,den,done,c);
 s->active=!done&&!s->report.nonFinite&&(first||s->changed)&&canIterate(*s,c);s->used=0;cudaGraphSetConditional(loop,s->active);
}
__global__ void parallelBegin(int n,R* v,R* h,R* g,R* cs,R* sn,State* s){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)v[i]=s->beta>0?v[i]/s->beta:R(0);
 if(blockIdx.x==0){for(int i=threadIdx.x;i<17*16;i+=blockDim.x)h[i]=0;
 for(int i=threadIdx.x;i<=16;i+=blockDim.x)g[i]=i?R(0):s->beta;
 for(int i=threadIdx.x;i<16;i+=blockDim.x){cs[i]=0;sn[i]=0;}}
}
__global__ void finishArnoldi(int j,const R* stats,R* h,R* g,R* cs,R* sn,State* s,Controls c){
 if(threadIdx.x)return;R norm=stats[1];R* col=h+j*17;col[j+1]=norm;
 for(int k=0;k<j;++k){R a=col[k],b=col[k+1];col[k]=cs[k]*a+sn[k]*b;col[k+1]=-sn[k]*a+cs[k]*b;}
 R a=col[j],b=col[j+1],r=hypot(a,b);cs[j]=r>0?a/r:R(1);sn[j]=r>0?b/r:R(0);col[j]=r;col[j+1]=0;
 R gj=g[j];g[j]=cs[j]*gj;g[j+1]=-sn[j]*gj;s->used=j+1;++s->report.iterations;
 if(!isfinite(r)||!isfinite(norm)||!isfinite(g[j+1]))s->report.nonFinite=true;
 if(!(r>0)||!(norm>0))++s->report.breakdowns;
 s->active=isfinite(r)&&r>0&&norm>0&&fabs(g[j+1])>s->target&&canIterate(*s,c);
}
__global__ void normalizeVector(int n,const R* w,R* v,const R* stats){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)v[i]=stats[1]>0?w[i]/stats[1]:R(0);
}
__global__ void triangular(const R* h,R* g,State* s){
 if(threadIdx.x)return;s->changed=0;for(int k=s->used-1;k>=0;--k){R q=g[k];for(int j=k+1;j<s->used;++j)q-=h[j*17+k]*g[j];R d=h[k*17+k];g[k]=d!=0?q/d:R(0);}
}
__global__ void parallelUpdate(int n,R* x,const R* z,const R* g,State* s){
 updateFeasibleIntensity(n,x,z,g,s);
}

}
class RadiationFgmres {
 using R=RadiationReal;using State=radiationFgmresDetail::State;
 int n_=0,blocks_=0;R *partial_=nullptr,*dots_=nullptr,*stats_=nullptr;R *v_=nullptr,*z_=nullptr,*w_=nullptr,*h_=nullptr,*g_=nullptr,*cs_=nullptr,*sn_=nullptr;State* state_=nullptr;
 template<class T> static void allocate(T*& p,size_t n){executionGraphDetail::checked(cudaMalloc(reinterpret_cast<void**>(&p),(n?n:1)*sizeof(T)));}
 void release(){for(R* p:{v_,z_,w_,h_,g_,cs_,sn_,partial_,dots_,stats_})if(p)cudaFree(p);if(state_)cudaFree(state_);v_=z_=w_=h_=g_=cs_=sn_=nullptr;state_=nullptr;partial_=dots_=stats_=nullptr;n_=0;}
 public:
 using Controls=radiationFgmresDetail::Controls;using Report=radiationFgmresDetail::Report;
 using Operation=std::function<ExecutionGraphNode(ExecutionGraph,ExecutionGraphNode,const R*,R*)>;
 RadiationFgmres()=default;~RadiationFgmres(){release();}
 RadiationFgmres(const RadiationFgmres&)=delete;RadiationFgmres& operator=(const RadiationFgmres&)=delete;
 void configure(int n){if(n==n_&&state_)return;release();n_=n;allocate(v_,size_t(n)*17);allocate(z_,size_t(n)*16);allocate(w_,n);allocate(h_,17*16);allocate(g_,17);allocate(cs_,16);allocate(sn_,16);allocate(state_,1);blocks_=std::min(256,(n+127)/128);allocate(partial_,size_t(blocks_)*16);allocate(dots_,16);allocate(stats_,3);}
 const Report* report()const{return &state_->report;}
 ExecutionGraphNode append(ExecutionGraph graph,ExecutionGraphNode dep,void* stream,R* x,const R* rhs,Controls controls,const Operation& apply,const Operation& precondition){
 using namespace radiationFgmresDetail;using namespace executionGraphDetail;auto st=static_cast<cudaStream_t>(stream);
 auto capture=[&](ExecutionGraph gr,ExecutionGraphNode d,auto fn){return appendCaptured(gr,d?std::vector<cudaGraphNode_t>{d}:std::vector<cudaGraphNode_t>{},st,fn);};
 auto norm=[&](bool first,cudaGraphConditionalHandle handle){
 if(n_<512){residual<<<1,128,128*sizeof(R),st>>>(n_,rhs,w_,v_,state_,controls,first,handle);return;}
 vectorStats<<<blocks_,128,128*sizeof(R),st>>>(n_,rhs,w_,v_,partial_,true);
 finishStats<<<1,128,128*sizeof(R),st>>>(blocks_,partial_,stats_);
 finishResidual<<<1,1,0,st>>>(stats_,state_,controls,first,handle);
 };
 cudaGraphConditionalHandle loop;checked(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
 auto node=capture(graph,dep,[&]{initialize<<<1,1,0,st>>>(state_,controls);});
 node=apply(graph,node,x,w_);
 node=capture(graph,node,[&]{norm(true,loop);});
 cudaGraph_t body=nullptr;auto finished=addWhile(graph,loop,{node},body);
 auto inner=capture(body,nullptr,[&]{if(n_<512)begin<<<1,128,0,st>>>(n_,v_,h_,g_,cs_,sn_,state_);else parallelBegin<<<blocks_,128,0,st>>>(n_,v_,h_,g_,cs_,sn_,state_);});
 for(int j=0;j<16;++j){
 cudaGraphConditionalHandle active;checked(cudaGraphConditionalHandleCreate(&active,body,0,cudaGraphCondAssignDefault));
 inner=capture(body,inner,[&]{gate<<<1,1,0,st>>>(state_,controls,active);});
 cudaGraph_t step=nullptr;auto conditional=addIf(body,active,{inner},step);
 auto work=precondition(step,nullptr,v_+size_t(j)*n_,z_+size_t(j)*n_);
 work=apply(step,work,z_+size_t(j)*n_,w_);
 capture(step,work,[&]{if(n_<512)arnoldi<<<1,128,128*sizeof(R),st>>>(n_,j,v_,w_,h_,g_,cs_,sn_,state_,controls);
 else{
 for(int pass=0;pass<2;++pass){
 parallelDots<<<dim3(blocks_,j+1),128,128*sizeof(R),st>>>(n_,j,v_,w_,partial_);
 finishDots<<<j+1,128,128*sizeof(R),st>>>(blocks_,j,partial_,dots_,h_);
 subtractDots<<<blocks_,128,0,st>>>(n_,j,v_,w_,dots_);
 }
 vectorStats<<<blocks_,128,128*sizeof(R),st>>>(n_,nullptr,w_,nullptr,partial_,false);
 finishStats<<<1,128,128*sizeof(R),st>>>(blocks_,partial_,stats_);
 finishArnoldi<<<1,1,0,st>>>(j,stats_,h_,g_,cs_,sn_,state_,controls);
 normalizeVector<<<blocks_,128,0,st>>>(n_,w_,v_+size_t(j+1)*n_,stats_);
 }});
 inner=conditional;
 }
 inner=capture(body,inner,[&]{if(n_<512)update<<<1,128,0,st>>>(n_,x,z_,h_,g_,state_);else{triangular<<<1,1,0,st>>>(h_,g_,state_);parallelUpdate<<<blocks_,128,0,st>>>(n_,x,z_,g_,state_);}});
 inner=apply(body,inner,x,w_);
 capture(body,inner,[&]{norm(false,loop);});
 return finished;
 }
};
}}
#endif
