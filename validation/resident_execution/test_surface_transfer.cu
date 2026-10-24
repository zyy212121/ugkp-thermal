#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/CoupledSurfaceTransfer.cuh"
#include <cassert>
#include <iostream>
#include <new>
#include <vector>
using namespace chmt;
#define CUDA(call) do {const auto result=(call);if(result!=cudaSuccess){std::cerr<<cudaGetErrorString(result)<<'\n';std::abort();}} while(0)
std::vector<void*> allocations;
template<class T> T* managed(int n){T* p=nullptr;CUDA(cudaMallocManaged(&p,n*sizeof(T)));allocations.push_back(p);return p;}
int main(){
    constexpr int nc=257,nf=260,count=129;
    auto* v=managed<SharedGasDeviceView>(1);new(v)SharedGasDeviceView{};
    v->nCells=nc;v->nFaces=nf;v->gammaGas=1.4;v->faceOwner=managed<int>(nf);
    auto* faces=managed<int>(count);auto* samples=managed<CoupledSurfaceSample>(count);auto* flux=managed<GasQ>(count);
    for(int i=0;i<count;++i)faces[i]=2*(count-1-i);
    for(int f=0;f<nf;++f)v->faceOwner[f]=(7*f)%nc;
    auto cell=[&](double base){auto* p=managed<double>(nc);for(int i=0;i<nc;++i)p[i]=base+i*.001;return p;};
    auto face=[&](double base){auto* p=managed<double>(nf);for(int i=0;i<nf;++i)p[i]=base+i*.001;return p;};
    v->rho=cell(2);v->Tgas=cell(400);v->p=cell(100000);v->Ux=cell(1);v->Uy=cell(2);v->Uz=cell(3);
    double** gradients[]={&v->gradRhoX,&v->gradRhoY,&v->gradRhoZ,&v->gradTX,&v->gradTY,&v->gradTZ,&v->gradPx,&v->gradPy,&v->gradPz,&v->gradUxX,&v->gradUxY,&v->gradUxZ,&v->gradUyX,&v->gradUyY,&v->gradUyZ,&v->gradUzX,&v->gradUzY,&v->gradUzZ};
    for(int j=0;j<18;++j)*gradients[j]=cell(10+j);
    v->gasPhiRho=face(30);v->gasPhiRhoUx=face(31);v->gasPhiRhoUy=face(32);v->gasPhiRhoUz=face(33);v->gasPhiRhoE=face(34);
    auto& s=v->gasSpecies;s.rho=managed<double>(Ns*nc);s.soundSpeed=cell(300);s.gradX=managed<double>(Ns*nc);s.gradY=managed<double>(Ns*nc);s.gradZ=managed<double>(Ns*nc);s.flux=managed<double>(Ns*nf);
    for(int k=0;k<Ns;++k){for(int c=0;c<nc;++c){s.rho[k*nc+c]=(k+1)*.1*v->rho[c];s.gradX[k*nc+c]=k+c*.001;s.gradY[k*nc+c]=k+c*.002;s.gradZ[k*nc+c]=k+c*.003;}for(int f=0;f<nf;++f)s.flux[k*nf+f]=100+k+f*.001;}
    for(bool single:{false,true}){
        s.mode=single?ugkwp::GasMode::SingleLegacy:ugkwp::GasMode::MixtureFrozen;
        if(single){s.rho=s.soundSpeed=s.gradX=s.gradY=s.gradZ=s.flux=nullptr;}
        gatherCoupledSurface<<<2,128>>>(v,faces,count,Ns-1,samples);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
        for(int i=0;i<count;++i){const int f=faces[i],c=v->faceOwner[f];const auto& w=samples[i].primitive;const auto& g=samples[i].gradient;
            assert(w.rho==v->rho[c]&&w.temperature==v->Tgas[c]&&w.pressure==v->p[c]);assert(w.velocity.x==v->Ux[c]&&w.velocity.y==v->Uy[c]&&w.velocity.z==v->Uz[c]);
            assert(std::abs(w.soundSpeed-(single?std::sqrt(1.4*v->p[c]/v->rho[c]):s.soundSpeed[c]))<1e-12);
            const Vec3* gs[]={&g.rho,&g.temperature,&g.pressure,&g.velocity[0],&g.velocity[1],&g.velocity[2]};
            for(int j=0;j<6;++j){assert(gs[j]->x==(*gradients[3*j])[c]);assert(gs[j]->y==(*gradients[3*j+1])[c]);assert(gs[j]->z==(*gradients[3*j+2])[c]);}
            for(int k=0;k<Ns;++k){assert(w.Y[k]==(single?double(k==Ns-1):s.rho[k*nc+c]/v->rho[c]));assert(g.Y[k].x==(single?0:s.gradX[k*nc+c]));assert(g.Y[k].y==(single?0:s.gradY[k*nc+c]));assert(g.Y[k].z==(single?0:s.gradZ[k*nc+c]));}
            const auto q=samples[i].flux;assert(q.mass==v->gasPhiRho[f]&&q.energy==v->gasPhiRhoE[f]);assert(q.momentum.x==v->gasPhiRhoUx[f]&&q.momentum.y==v->gasPhiRhoUy[f]&&q.momentum.z==v->gasPhiRhoUz[f]);
            for(int k=0;k<Ns;++k)assert(q.species[k]==(single?(k==Ns-1?q.mass:0):s.flux[k*nf+f]));
            flux[i]=GasQ{};flux[i].mass=-1-i;flux[i].energy=-2-i;flux[i].momentum={-3.-i,-4.-i,-5.-i};for(int k=0;k<Ns;++k)flux[i].species[k]=-6-i-k;
        }
        scatterCoupledFlux<<<2,128>>>(v,faces,count,flux);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
        for(int f=1;f<nf;f+=2)assert(v->gasPhiRho[f]==30+f*.001&&v->gasPhiRhoE[f]==34+f*.001);
        gatherCoupledFlux<<<2,128>>>(v,faces,count,Ns-1,flux);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
        for(int i=0;i<count;++i){assert(flux[i].mass==-1-i&&flux[i].energy==-2-i);assert(flux[i].momentum.x==-3-i&&flux[i].momentum.y==-4-i&&flux[i].momentum.z==-5-i);for(int k=0;k<Ns;++k)assert(flux[i].species[k]==(single?(k==Ns-1?-1-i:0):-6-i-k));}
    }
    for(auto* p:allocations)CUDA(cudaFree(p));std::cout<<"surface transfer Ns="<<Ns<<" PASS\n";
}
