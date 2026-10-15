"""Low-k MSS7 wall-input regression; same equations, iteration limit and tolerance."""
from test_wall_model import compile_run

def test_low_matching_k_blowing_wall_converges_without_inventory_change(tmp_path):
    compile_run(tmp_path, r'''

 Model m;m.in.integrateOwnerKSource=false;
 for(int j=0;j<2;++j){m.coef[j*3]=1040;m.sp[j].molarMass=.028000000000000004;m.sp[j].minTemperature=100;m.sp[j].maxTemperature=3000;}
 m.in.pressure=101324.99999999999;m.in.temperature=1193.1640625;m.in.matching.temperature=699.99999999999989;
 m.in.matching.k=9.9999999999999991e-5;m.in.matching.omega=1;
 m.in.matchingDistance=.00038683956872244863;m.in.ownerDistance=.00012457914367287038;
 m.in.massFlux[0]=.0009999999999999998;m.in.massFlux[1]=0;
 m.in.normal[0]=-.51898802547885003;m.in.normal[1]=-.85478150974945899;m.in.normal[2]=-0.;
 m.in.velocity[0]=5.1898802547884996e-7;m.in.velocity[1]=8.5478150974945899e-7;
 m.in.matching.velocity[0]=29.999999999999996;
 m.in.matching.massFraction[0]=.099999999999999992;m.in.matching.massFraction[1]=.90000000000000002;
 m.in.model.viscosity=1.8e-5;m.in.model.conductivity=.03;m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=2e-5;
 m.in.model.mode=GasMode::MixtureFrozen;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=24;c.maxIterations=60;c.relativeTolerance=1e-8;c.absoluteTolerance=1e-10;c.stretch=2;
 WallWorkspace<double,2,24>w;WallOutput<double,2>o;WallStatus st;
 const bool ok=evaluateWallModel(m.in,c,w,o,st);
 std::cout<<ok<<" "<<st.iteration<<" "<<st.residual<<"\n";
 assert(ok);assert(st.residual<=c.relativeTolerance+c.absoluteTolerance);
 assert(o.ownerOmega>0&&std::isfinite(o.conductiveHeatFlux));

''')
