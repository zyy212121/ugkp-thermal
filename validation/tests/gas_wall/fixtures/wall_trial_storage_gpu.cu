#include "common/gasWall/WallModel.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace ugkwp;
using namespace ugkwp::gaswall;

struct Fixture {
    SpeciesThermoData<double> species[2]{};
    double coefficients[27]{},elements[2]{1,1},basis[2]{-1,1};
    GasReactionData<double> reactions[1]{};
    GasStoichTerm<double> reactants[1]{},products[1]{};
    WallInput<double,2> input{};
    WallModelConfig<double> config{};
    explicit Fixture(int mode){
        config.nodes=4;
        for(int s=0;s<2;++s){species[s].model=SpeciesThermoModel::NASA7;species[s].coefficientOffset=(mode==0?6:14)*s;species[s].molarMass=.028;species[s].minTemperature=200;species[s].midTemperature=600;species[s].maxTemperature=4000;species[s].referencePressure=101325;species[s].referenceTemperature=300;}
        const double cp=1000./(universalGasConstant<double>()/.028);
        if(mode==0){coefficients[0]=coefficients[6]=cp;input.model.thermo.coefficientCount=12;}
        else {coefficients[0]=coefficients[7]=coefficients[14]=coefficients[21]=cp;input.model.thermo.coefficientCount=27;}
        input.pressure=101325;input.temperature=600;input.normal[1]=1;input.matchingDistance=.01;input.ownerDistance=.004;
        input.matching.temperature=600;input.matching.velocity[0]=10;input.matching.massFraction[0]=.7;input.matching.massFraction[1]=.3;
        input.model.viscosity=2e-5;input.model.conductivity=.04;input.model.diffusivity[0]=input.model.diffusivity[1]=1e-5;
        input.diagnosticEntity=77;input.integrateOwnerKSource=false;
        input.model.mode=mode==2?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
        reactions[0].reversible=mode==2;reactions[0].reactantCount=1;reactions[0].productCount=1;reactions[0].highRate.preExponential=2;
        reactants[0].species=0;reactants[0].coefficient=1;products[0].species=1;products[0].coefficient=1;
        input.model.thermo.elementCount=1;input.model.thermo.speciesOrderHash=1;
        input.model.mechanism.reactionCount=1;input.model.mechanism.reactantTermCount=1;input.model.mechanism.productTermCount=1;input.model.mechanism.referencePressure=101325;input.model.mechanism.speciesOrderHash=1;input.model.mechanism.independentRank=1;
    }
    __device__ void bind(ugkp::diagnostics::DeviceContext* diagnostic){
        input.diagnostics=diagnostic;input.model.thermo.diagnostics=diagnostic;
        input.model.thermo.species=species;input.model.thermo.coefficients=coefficients;input.model.thermo.elementComposition=elements;
        input.model.mechanism.reactions=reactions;input.model.mechanism.reactants=reactants;input.model.mechanism.products=products;input.model.mechanism.stoichiometricBasis=basis;
    }
};
struct Result {int prepared,baseValid,returned,unchanged;double alpha,norm;};
__global__ void evaluate(Fixture* fixture,WallWorkspace<double,2,4>* workspace,
    ugkp::diagnostics::DeviceContext* diagnostic,Result* result,int mode){
    detail::CudaWallTeam team;
    __shared__ detail::LayerContext<double,2> context;
    __shared__ double before[4][6];
    if(team.leader())fixture->bind(diagnostic);
    team.sync();
    bool prepared=false;
    if(team.leader())prepared=detail::makeContext(fixture->input,fixture->config,context);
    prepared=team.broadcast(prepared);
    if(!prepared){if(team.leader())result->prepared=0;return;}
    for(int i=team.rank();i<4;i+=team.size()){
        workspace->y[i]=.01*i/3;workspace->state[i][0]=10.*i/3;
        workspace->state[i][3]=.3;workspace->state[i][5]=1;
    }
    team.sync();
    for(int i=team.rank();i<4;i+=team.size())for(int j=0;j<6;++j)before[i][j]=workspace->state[i][j];
    team.sync();double norm=0,alpha=0;
    const bool valid=detail::allResidual(fixture->input,fixture->config,context,
        workspace->state,workspace->y,workspace->residual,norm,team);
    if(team.leader())workspace->delta[1][2]=mode==3?-1000:1;
    team.sync();
    bool ok=false;
    if(valid)ok=detail::applyLayerNewtonUpdate(fixture->input,fixture->config,context,*workspace,norm,alpha,team);
    team.sync();
    if(team.leader()){
        result->prepared=prepared;result->baseValid=valid;result->returned=ok;result->alpha=alpha;result->norm=norm;result->unchanged=1;
        for(int i=0;i<4;++i)for(int j=0;j<6;++j){const double expected=before[i][j];if(workspace->state[i][j]!=expected)result->unchanged=0;}
    }
}
static void check(cudaError_t code){if(code!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(code));std::exit(2);}}
static void require(bool condition,const char* message){if(!condition){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(3);}}
int main(){
    Fixture* deviceFixture;WallWorkspace<double,2,4>* workspace;
    ugkp::diagnostics::DeviceContext* diagnostic;Result* deviceResult;
    check(cudaMalloc(&deviceFixture,sizeof(Fixture)));check(cudaMalloc(&workspace,sizeof(*workspace)));
    check(cudaMalloc(&diagnostic,sizeof(*diagnostic)));check(cudaMalloc(&deviceResult,sizeof(Result)));
    for(int mode=0;mode<4;++mode){
        Fixture fixture(mode);check(cudaMemcpy(deviceFixture,&fixture,sizeof fixture,cudaMemcpyHostToDevice));
        check(cudaMemset(workspace,0,sizeof(*workspace)));check(cudaMemset(diagnostic,0,sizeof(*diagnostic)));check(cudaMemset(deviceResult,0,sizeof(Result)));
        evaluate<<<1,32>>>(deviceFixture,workspace,diagnostic,deviceResult,mode);check(cudaGetLastError());check(cudaDeviceSynchronize());
        Result result{};ugkp::diagnostics::DeviceContext diagnostics{};
        check(cudaMemcpy(&result,deviceResult,sizeof result,cudaMemcpyDeviceToHost));check(cudaMemcpy(&diagnostics,diagnostic,sizeof diagnostics,cudaMemcpyDeviceToHost));
        const auto& first=diagnostics.firstFatal;
        std::printf("mode=%d prepared=%d base=%d returned=%d unchanged=%d alpha=%.17g fatal=%u site=%u face=%lld node=%u species=%d observed=%.17g\n",mode,result.prepared,result.baseValid,result.returned,result.unchanged,result.alpha,unsigned(diagnostics.fatalClaim),first.sourceSiteId,static_cast<long long>(first.entityIndex),first.stageId,first.component,first.observed);
        require(result.prepared&&result.baseValid&&result.unchanged,"invalid base or changed trial state");
        if(mode==0||mode==2){require(!result.returned&&diagnostics.fatalClaim,"consumed missing storage did not remain fatal");require(first.sourceSiteId==2432&&first.code==ugkp::diagnostics::Code::InvalidIndex&&first.operatorId==23&&first.entityIndex==77&&first.stageId==1,"lost wall storage diagnostic identity");require(first.component==(mode==0?0:1)&&first.observed==(mode==0?5.:6.),"lost actual species/coefficient availability");}
        else require(result.returned&&!diagnostics.fatalClaim&&result.alpha==0,"unused entropy or rejected domain trial became fatal");
    }
    check(cudaFree(deviceResult));check(cudaFree(diagnostic));check(cudaFree(workspace));check(cudaFree(deviceFixture));
}
