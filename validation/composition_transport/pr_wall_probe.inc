#include <vector>
#define NATIVE NATIVE_MODE
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{
    ugkwp::GasSpeciesState<Real,2> gasSpecies;
};
std::vector<void*> allocations;
template<class T> T* allocate(int n){
    T* p=nullptr;
#if NATIVE
    if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();
#else
    p=static_cast<T*>(std::malloc(sizeof(T)*n));
#endif
    for(int i=0;i<n;++i)new (p+i) T{};
    allocations.push_back(p);return p;
}
void setup(State& s,int config,Real kinetic){
ALLOCATE_FIELDS
    s.nCells=2;s.nFaces=3;s.nInternalFaces=1;
    s.gasFluxScheme=1;s.gasReconstruction=0;s.gasLimiter=0;
    s.gasMu=1.8e-5;s.gasPrClamped=.7;s.gammaGas=1.4;
    s.Rgas=287;s.gasCp=1004.5;s.gasThermalConductivity=-1;
    s.rhoMin=1e-12;s.TgasMin=1e-9;
    s.turbulenceModel=3;s.sstConfigured=1;s.sstWallTreatment=1;
    s.turbulentPrandtl=.9;s.sstCoefficients=ugkwp::defaultSstCoefficients();
    s.sstKMin=1e-12;s.sstOmegaMin=1e-8;
    s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
    s.sstJayatillekeP=ugkpwall::jayatillekeSmoothP(.7/.9);
    s.sstThermalYPlus=ugkpwall::jayatillekeThermalYPlus(.7/.9,.41,9.8);
    auto& g=s.gasSpecies;g.mode=ugkwp::GasMode::MixtureFrozen;
    g.rho=allocate<Real>(4);g.flux=allocate<Real>(6);
    g.gradX=allocate<Real>(4);g.gradY=allocate<Real>(4);g.gradZ=allocate<Real>(4);
    g.boundaryMassFraction=allocate<Real>(6);g.compositionBoundaryFixed=allocate<int>(3);
    g.soundSpeed=allocate<Real>(2);g.heatCapacity=allocate<Real>(2);
    g.gasConstant=allocate<Real>(2);g.cellStatus=allocate<int>(2);g.faceStatus=allocate<int>(3);
    auto* meta=allocate<ugkwp::SpeciesThermoData<Real>>(2);
    auto* coeff=allocate<Real>(6);
    g.thermo.species=meta;g.thermo.coefficients=coeff;g.thermo.coefficientCount=6;
    for(int j=0;j<2;++j){
        meta[j].model=ugkwp::SpeciesThermoModel::LinearCp;
        meta[j].molarMass=j?.04:.028;meta[j].minTemperature=100;
        meta[j].maxTemperature=3000;meta[j].referencePressure=101325;
        meta[j].coefficientOffset=3*j;coeff[3*j]=j?1500:1000;coeff[3*j+1]=.2;
    }
    for(int c=0;c<2;++c){
        s.V[c]=1;s.Cx[c]=c;s.rho[c]=1;s.Tgas[c]=700;
        g.rho[c]=.9;g.rho[2+c]=.1;g.heatCapacity[c]=1190;
        g.gasConstant[c]=ugkwp::universalGasConstant<Real>()*(.9/.028+.1/.04);
        s.p[c]=g.gasConstant[c]*700;
        g.soundSpeed[c]=sqrt(1190/(1190-g.gasConstant[c])*s.p[c]);
        Real partial[2]={.9,.1};
        s.k[c]=kinetic;s.rhoK[c]=s.rho[c]*kinetic;
        s.omega[c]=10;s.rhoOmega[c]=10;
        s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(700),g.thermo)+s.rhoK[c];
        s.sstWallDistance[c]=.5;
        s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;
    }
    s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
    for(int f=0;f<3;++f){
        s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;
        s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;
        s.faceCx[f]=f==0?.5:f==1?-.5:1.5;
        s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;
        s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;
    }
    s.gasBoundaryKind[1]=s.riemannBoundaryKind[1]=2;
    s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=500;
    s.riemannBoundaryUFix[1]=1;
    // 0: independent Pr, differing fixed Y; 1: same face mu, variable-mu control;
    // 2: same Y; 3: zeroGradient; 4: legacy const transport; 5: variable mu, const Pr.
    g.compositionBoundaryFixed[1]=config!=3;
    g.boundaryMassFraction[1]=config==2?.9:.1;
    g.boundaryMassFraction[4]=config==2?.1:.9;
    if(config<4){
        s.gasViscosity.prandtlSpecies=1;
        s.gasViscosity.prandtlCoefficients[0]=.7;
        s.gasViscosity.prandtlCoefficients[1]=.2;
    }
    if(config==1||config==5){
        s.gasViscosity.model=ugkwp::MolecularViscosityModel::MassFractionSutherland;
        s.gasViscosity.compositionSpecies=1;s.gasViscosity.compositionTRef=500;
        s.gasViscosity.muCoefficients[0]=1.8e-5;
        s.gasViscosity.sCoefficients[0]=110;
    }
    s.diagnostics=allocate<ugkp::diagnostics::DeviceContext>(1);
}
__global__ void evaluate(State s,Real* out){
    Real mu=0,kap=0,q=0;int active=0;
    gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,kap,q,active);
    Real mass=0,mx=0,my=0,mz=0,energy=0;
    bool ok=computeRiemannGasFaceFluxDevice<true>(s,1,mass,mx,my,mz,energy);
    Real baseline=s.gasMu*s.gasSpecies.heatCapacity[0]/s.gasPrClamped;
    if(ugkwp::gasVariableViscosity(s)||ugkwp::gasVariablePrandtl(s))
        baseline=ugkwp::gasFaceMolecularViscosity(s,1,0,-1)
            *ugkwp::gasFaceMolecularHeatCapacity(s,1,0,-1)/ugkwp::gasFaceMolecularPrandtl(s,1,0,-1);
    out[0]=q;out[1]=energy;out[2]=mass;out[3]=s.gasSpecies.flux[1];out[4]=s.gasSpecies.flux[4];
    out[5]=s.gasSpecies.faceStatus[1];out[6]=active;out[7]=ok;
    out[8]=ugkp::diagnostics::hasFatal(s.diagnostics);
    out[9]=baseline+kap;out[10]=ugkwp::gasFaceMolecularViscosity(s,1,0,-1);
    out[11]=ugkwp::gasFaceMolecularHeatCapacity(s,1,0,-1);out[12]=ugkwp::gasFaceMolecularPrandtl(s,1,0,-1);
}
int main(){
    for(Real kinetic:{Real(1e-12),Real(.1)})for(int config=0;config<6;++config){
        State s;setup(s,config,kinetic);auto* out=allocate<Real>(13);
#if NATIVE
        evaluate<<<1,1>>>(s,out);
        auto error=cudaGetLastError();if(error==cudaSuccess)error=cudaDeviceSynchronize();
        if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));return 1;}
#else
        evaluate(s,out);
#endif
        std::printf("%d %.17g",config,kinetic);
        for(int i=0;i<13;++i)std::printf(" %.17g",out[i]);std::puts("");
        for(auto* p:allocations){
#if NATIVE
            if(cudaFree(p)!=cudaSuccess)return 2;
#else
            std::free(p);
#endif
        }
        allocations.clear();
    }
}
