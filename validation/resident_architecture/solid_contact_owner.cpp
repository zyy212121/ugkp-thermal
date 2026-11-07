#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuSolidThermalProperties.H"
#include "IStringStream.H"
#include <iomanip>
#include <iostream>
#ifdef UGKP_TEST_RESIDENT_OWNER
#include <cuda_runtime_api.h>
#include "Diagnostics.H"
static void cudaCheck(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
#endif
using namespace Foam;using namespace Foam::gpuThermal;
int main(int argc,char** argv){
    argList::addBoolOption("variable-properties","use nonlinear caloric and conductivity tables");
    argList::addBoolOption("aitken","use nonlinear Aitken relaxation");
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh primaryMesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    fvMesh auxiliaryMesh(IOobject("auxiliary",runTime.timeName(),runTime,IOobject::MUST_READ));
    volScalarField primaryT(IOobject("T",runTime.timeName(),primaryMesh,IOobject::MUST_READ,IOobject::NO_WRITE),primaryMesh);
    volScalarField auxiliaryT(IOobject("T",runTime.timeName(),auxiliaryMesh,IOobject::MUST_READ,IOobject::NO_WRITE),auxiliaryMesh);
    forAll(primaryT,c){const auto& x=primaryMesh.C()[c];primaryT[c]=500+20*std::sin(x.x())+7*x.y();}
    forAll(auxiliaryT,c){const auto& x=auxiliaryMesh.C()[c];auxiliaryT[c]=300+30*std::cos(x.x())+5*x.y();}
    primaryT.correctBoundaryConditions();auxiliaryT.correctBoundaryConditions();
    const label pp=primaryMesh.boundaryMesh().findPatchID("right"),ap=auxiliaryMesh.boundaryMesh().findPatchID("right");
    const label gp=primaryMesh.boundaryMesh().findPatchID("left");const auto& pb=primaryMesh.boundary()[pp];const auto& ab=auxiliaryMesh.boundary()[ap];
    if(pb.size()!=ab.size())return 3;
    IStringStream text(args.optionFound("variable-properties")
        ? "rho {type table; outOfBounds error; values ((100 0.9)(350 1.1)(1000 1.5));} Cp {type table; outOfBounds error; values ((100 1)(300 1.5)(450 2)(1000 3));} kappa {type table; outOfBounds error; values ((100 0.5)(300 1)(500 2)(1000 4));} emissivity {type constant; value 0;}"
        : "rho {type constant; value 1;} Cp {type constant; value 2;} kappa {type constant; value 1;} emissivity {type constant; value 0;}");
    dictionary dict(text);GpuSolidThermalProperties properties(dict);GpuSolidThermalControls controls;
    controls.nonlinearMaxIterations=100;controls.nonlinearRelativeTolerance=1e-11;controls.energyRelativeTolerance=1e-12;controls.energyAbsoluteTolerance=1e-10;
    controls.nonlinearAitken=args.optionFound("aitken");
    labelList primaryPatches(2);primaryPatches[0]=gp;primaryPatches[1]=pp;labelList primaryContact(1,pp),auxiliaryContact(1,ap);
    GpuSolidThermalCandidateSolver primary(primaryMesh,properties,primaryPatches,controls,primaryContact);
    GpuSolidThermalCandidateSolver auxiliary(auxiliaryMesh,properties,auxiliaryContact,controls,auxiliaryContact);
    const double temperatureTolerance=1e-10,energyTolerance=1e-10;const int maximumIterations=200;
#ifdef UGKP_TEST_RESIDENT_OWNER
    double *deviceInterval,*deviceGas;ugkp::diagnostics::DeviceContext* context;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceInterval),sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceGas),primaryMesh.boundary()[gp].size()*sizeof(double)));
    cudaCheck(cudaMemset(deviceGas,0,primaryMesh.boundary()[gp].size()*sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&context),sizeof(*context)));cudaCheck(cudaMemset(context,0,sizeof(*context)));
    ugkp::gpu::SolidThermalContact contact;std::vector<int> mapping(ab.size());forAll(ab,f)mapping[f]=f;
    contact.begin(mapping,std::vector<double>(pb.deltaCoeffs().begin(),pb.deltaCoeffs().end()),
        std::vector<double>(ab.deltaCoeffs().begin(),ab.deltaCoeffs().end()),std::vector<double>(ab.magSf().begin(),ab.magSf().end()));
    if(!primary.prepareResident(primaryT,{true,primaryMesh.boundary()[gp].size(),deviceGas,nullptr,deviceInterval},context,contact.primaryEnergy(deviceInterval))
        ||!auxiliary.prepareResident(auxiliaryT,contact.auxiliaryEnergy(deviceInterval),context))return 4;
    auto& pGraph=primary.prepareResidentContact(pp);auto& aGraph=auxiliary.prepareResidentContact(ap);
    ugkp::gpu::SolidContactIterationGraph graph(pGraph,aGraph,contact,{maximumIterations,temperatureTolerance,energyTolerance},1,deviceInterval);
#endif
    std::cout<<std::setprecision(17);scalar time=0;
    for(int window=1;window<=4;++window){const scalar dt=.00025*window;time+=dt;runTime.setTime(time,window);
        const scalarField oldPrimary(primaryT.primitiveField()),oldAuxiliary(auxiliaryT.primitiveField());int iterations=0;
#ifdef UGKP_TEST_RESIDENT_OWNER
        cudaCheck(cudaMemcpy(deviceInterval,&dt,sizeof(double),cudaMemcpyHostToDevice));graph.enqueue(nullptr);
        const auto report=graph.collect();if(!report.valid||!report.converged)return 5;iterations=report.iteration;
        primary.acceptResidentContact();auxiliary.acceptResidentContact();primary.publishResident(primaryT);auxiliary.publishResident(auxiliaryT);
#else
        scalarField pIter(primaryT.primitiveField()),aIter(auxiliaryT.primitiveField()),previousEnergy(ab.size(),0);
        for(int iteration=0;iteration<maximumIterations;++iteration){
            List<scalarField> pEnergy(2),aEnergy(1);pEnergy[0]=scalarField(primaryMesh.boundary()[gp].size(),0);
            pEnergy[1]=scalarField(pb.size(),0);aEnergy[0]=scalarField(ab.size(),0);
            forAll(ab,f){const double p=pIter[pb.faceCells()[f]],a=aIter[ab.faceCells()[f]];
                aEnergy[0][f]=ab.magSf()[f]*dt*(p-a)/(1/(properties.kappa(p)*pb.deltaCoeffs()[f])+1/(properties.kappa(a)*ab.deltaCoeffs()[f]));
                pEnergy[1][f]=-aEnergy[0][f];}
            auto nextA=auxiliary.solveTemporarySolidCandidate(auxiliaryT,aEnergy,dt);
            auto nextP=primary.solveTemporarySolidCandidate(primaryT,pEnergy,dt);
            double change=0,difference=0,scale=1;
            forAll(ab,f){const int pc=pb.faceCells()[f],ac=ab.faceCells()[f];const double p=nextP.temperature()[pc],a=nextA.temperature()[ac];
                change=std::max(change,std::abs(p-pIter[pc])/std::max(1.,std::max(std::abs(p),std::abs(pIter[pc]))));
                change=std::max(change,std::abs(a-aIter[ac])/std::max(1.,std::max(std::abs(a),std::abs(aIter[ac]))));
                difference+=std::abs(aEnergy[0][f]-previousEnergy[f]);scale+=std::abs(aEnergy[0][f]);}
            pIter=nextP.temperature().primitiveField();aIter=nextA.temperature().primitiveField();previousEnergy=aEnergy[0];iterations=iteration+1;
            if((change<=temperatureTolerance&&difference/scale<=energyTolerance)||iterations==maximumIterations){
                primary.publishSolidCandidate(nextP,primaryT);auxiliary.publishSolidCandidate(nextA,auxiliaryT);break;}
        }
#endif
        long double energy=0;forAll(primaryT,c)energy+=static_cast<long double>(properties.deltaHv(primaryT[c],oldPrimary[c]))*primaryMesh.V()[c];
        forAll(auxiliaryT,c)energy+=static_cast<long double>(properties.deltaHv(auxiliaryT[c],oldAuxiliary[c]))*auxiliaryMesh.V()[c];
        std::cout<<"CONTACT_WINDOW "<<window<<' '<<iterations<<' '<<energy<<'\n';if(!std::isfinite(energy)||std::abs(energy)>2e-8)return 6;
        forAll(primaryT,c)std::cout<<"CONTACT_CELL "<<window<<" primary "<<c<<' '<<primaryT[c]<<'\n';
        forAll(auxiliaryT,c)std::cout<<"CONTACT_CELL "<<window<<" auxiliary "<<c<<' '<<auxiliaryT[c]<<'\n';
    }
    return 0;
}
