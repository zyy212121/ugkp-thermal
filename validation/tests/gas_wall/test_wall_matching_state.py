from test_wall_model import BASE
import test_wall_model

def compile_run(tmp_path, body):
    original=test_wall_model.BASE
    test_wall_model.BASE=original.replace('#include "common/gasWall/WallModel.H"','#include "common/gasWall/WallModel.H"\n#include "common/gasTransport/GasWallMatching.H"')
    try: test_wall_model.compile_run(tmp_path,body)
    finally: test_wall_model.BASE=original

def test_conserved_matching_preserves_energy_and_formation_reference(tmp_path):
    compile_run(tmp_path,r"""
    Model m;struct S:GasStateView<double,double,SstCoefficientsT<double>>{GasSpeciesState<double,2> gasSpecies;};
    S s;s.nCells=3;s.gasSpecies.thermo=m.in.model.thermo;
    double rho[3]={1,1,1},mx[3]={10,20,30},my[3]{},mz[3]{},t[3]={300,300,300};
    double species[6]={.7,.7,.7,.3,.3,.3},e[3],rk[3]={1,2,3},rw[3]={100,200,300};
    s.rho=rho;s.rhoUx=mx;s.rhoUy=my;s.rhoUz=mz;s.rhoE=e;s.Tgas=t;s.rhoK=rk;s.rhoOmega=rw;s.gasSpecies.rho=species;
    double cv=1000-universalGasConstant<double>()/.028;for(int i=0;i<3;++i)e[i]=cv*300+.5*mx[i]*mx[i];
    int cells[3]={0,1,2};double w[3]={.5,.75,-.25};WallMatchingState<double,2> result;double pressure=0;
    assert(recoverWallMatching(s,cells,w,3,true,result,pressure));
    assert(near(result.velocity[0],12.5));
    double expected=(.5*e[0]+.75*e[1]-.25*e[2]-.5*12.5*12.5)/cv;
    assert(near(result.temperature,expected));assert(near(result.k,1.25));assert(near(result.omega,125));
    // Formation energies shift rhoE and the caloric reference together.
    m.coef[2]=4e6;m.coef[5]=-9e6;for(int i=0;i<3;++i)e[i]+=.7*4e6-.3*9e6;
    assert(recoverWallMatching(s,cells,w,3,true,result,pressure));assert(near(result.temperature,expected));
    // A missing species remains exactly absent under signed interpolation.
    for(int i=0;i<3;++i){species[i]=1;species[3+i]=0;e[i]=cv*300+.5*mx[i]*mx[i]+4e6;}
    assert(recoverWallMatching(s,cells,w,3,true,result,pressure));assert(result.massFraction[1]==0);
    """)

def test_matching_uses_one_feasible_scaling_for_all_conserved_components(tmp_path):
    compile_run(tmp_path,r"""
    Model m;struct S:GasStateView<double,double,SstCoefficientsT<double>>{GasSpeciesState<double,2> gasSpecies;};
    S s;s.nCells=2;s.gasSpecies.thermo=m.in.model.thermo;
    double rho[2]={1,1};s.rho=rho;double species[4]={.5,1,.5,0},mx[2]={0,10},zero[2]{},e[2],t[2]={300,300},rk[2]={1,1},rw[2]={100,100};
    double cv=1000-universalGasConstant<double>()/.028;for(int i=0;i<2;++i)e[i]=cv*300+.5*mx[i]*mx[i];
    s.rhoUx=mx;s.rhoUy=zero;s.rhoUz=zero;s.rhoE=e;s.Tgas=t;s.rhoK=rk;s.rhoOmega=rw;s.gasSpecies.rho=species;
    int cells[2]={0,1};double weights[2]={-1,2};WallMatchingState<double,2> result;double p;
    assert(recoverWallMatching(s,cells,weights,2,true,result,p));
    assert(near(result.massFraction[0],1));assert(result.massFraction[1]>=0&&result.massFraction[1]<1e-14);assert(near(result.velocity[0],10));
    // Pure energy overshoot is scaled analytically, without evaluating failed EOS trials.
    species[0]=species[1]=1;species[2]=species[3]=0;mx[1]=1e4;e[1]=cv*300+.5*mx[1]*mx[1];
    assert(recoverWallMatching(s,cells,weights,2,true,result,p));assert(result.temperature>0);assert(result.velocity[0]<2e4);
    // A finite quadratic omega overshoot reaches a strict positive-domain
    // boundary. Do not publish a rounded zero/negative omega to the BVP.
    rw[0]=12345.6789;rw[1]=.123456*rw[0];
    assert(recoverWallMatching(s,cells,weights,2,true,result,p));assert(result.omega>=rw[1]&&result.omega<=rw[0]);

    """)
