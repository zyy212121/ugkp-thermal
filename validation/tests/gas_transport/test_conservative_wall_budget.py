"""Execute production scalar-budget kernels on stationary and ALE inventories."""
from validation.tools.paths import SOURCE_ROOT
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture

def kernels():
    text=(SOURCE_ROOT/"applications/gasUGKP/private_backend/ConservativeWallProjection.cuh").read_text()
    a=text.rfind("template", 0, text.index("void conservativeKFluxBudget"))
    b=text.index("inline int prepareConservativeProjection",a)
    return text[a:b]

def test_budget_uses_absolute_divergence_and_old_inventory(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/GasConservativeWall.H"
struct BudgetState:State{struct{ugkwp::ConservativeWallData conservative;}gasBoundaryLayerModel;};
'''+kernels()+r'''
int main(){BudgetState a,b;initialise(a);initialise(b);a.sstConfigured=b.sstConfigured=1;
a.turbulenceModel=b.turbulenceModel=3;Real scaleA[2]{},scaleB[2]{};
a.gasBoundaryLayerModel.conservative.sourceK=new double[2]{};a.gasBoundaryLayerModel.conservative.sourceOmega=new double[2]{};b.gasBoundaryLayerModel.conservative.sourceK=new double[2]{};b.gasBoundaryLayerModel.conservative.sourceOmega=new double[2]{};
a.gasBoundaryLayerModel.conservative.kFluxScale=scaleA;b.gasBoundaryLayerModel.conservative.kFluxScale=scaleB;
Real oldV[2]={1,1},newV[2]={1.1,.9},sweeps[3]={.1,0,0};double dt=.001;moving(b,oldV,newV,sweeps,dt);
b.V[0]=1.1;b.V[1]=.9;b.gasPhiRho[0]=-b.rho[0]*sweeps[0]/dt;
a.sstPhiRhoK[0]=b.sstPhiRhoK[0]=10000;
for(int c=0;c<2;++c){threadIdx.x=c;conservativeKFluxBudget(&a,dt);conservativeKFluxBudget(&b,dt);near(scaleA[c],scaleB[c],"grid motion changed physical k budget");}
ck(scaleA[0]<1&&scaleA[0]>0,"budget test did not limit outgoing k");
threadIdx.x=0;applyConservativeKFluxBudget(&a);applyConservativeKFluxBudget(&b);near(a.sstPhiRhoK[0],b.sstPhiRhoK[0],"shared ALE k face differs");
}
''')
