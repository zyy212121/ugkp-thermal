import test_wall_model

def compile_run(tmp_path,body):
    original=test_wall_model.BASE
    test_wall_model.BASE=original.replace('#include "common/gasWall/WallModel.H"','#include "common/gasWall/WallModel.H"\n#include "common/gasTransport/GasBoundaryLayerEvaluation.H"')
    try:test_wall_model.compile_run(tmp_path,body)
    finally:test_wall_model.BASE=original

def test_one_pascal_couette_heat_partition_and_nodal_temperature(tmp_path):
    compile_run(tmp_path,r"""
    Model m;m.in.matching.velocity[0]=500;
    WallModelConfig<double> c;c.nodes=24;c.relativeTolerance=1e-12;c.absoluteTolerance=1e-14;WallWorkspace<double,2,24>w;WallOutput<double,2>o;WallStatus status;
    assert(evaluateWallModel(m.in,c,w,o,status));assert(o.residual<1e-8);
    assert(near(o.traction[0],1,1e-10));assert(near(o.heatConduction,-800,1e-10));assert(near(o.heatViscous,-250,1e-10));
    assert(std::abs(o.heatTransport)<1e-8);assert(near(o.conductiveHeatFlux,-1050,1e-10));assert(std::abs(o.heatBalanceResidual)<1e-7);
    for(int i=0;i<c.nodes;++i){double eta=w.y[i]/m.in.matchingDistance;
        double exact=300+(200+62.5)*eta-62.5*eta*eta;
        assert(near(reactingWallProfileTemperature(m.in,w,i),exact,1e-10));}
    """)

def test_multispecies_blowing_absolute_stress_work_and_ale_pressure_work(tmp_path):
    compile_run(tmp_path,r"""
    struct S{GasBoundaryLayerState<double>gasBoundaryLayer;ugkp::diagnostics::DeviceContext* diagnostics=nullptr;};
    for(double tangential:{0.,50.})for(double normal:{0.,10.})for(double mass:{0.,.001}){
        Model m;m.in.velocity[0]=tangential;m.in.velocity[1]=normal;m.in.matching.velocity[0]=tangential+10;
        m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;m.coef[2]=1e6;m.coef[5]=-2e6;
        WallModelConfig<double>c;c.nodes=24;WallWorkspace<double,2,24>w;WallOutput<double,2>o;WallStatus status;
        assert(evaluateWallModel(m.in,c,w,o,status));assert(o.residual<1.01e-8);
        detail::LayerContext<double,2>ctx;assert(detail::makeContext(m.in,c,ctx));detail::LayerFlux<double,2>first;
        assert(detail::intervalFlux(m.in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first));
        S s;auto& wall=s.gasBoundaryLayer;wall.enabled=true;wall.count=1;int st=0;double js[2];GasBoundaryLayerExchange<double>e;GasBoundaryLayerSstClosure<double>closure;
        wall.status=&st;wall.speciesFlux=js;wall.exchange=&e;wall.sst=&closure;
        assert(publishGasBoundaryLayerOutput(s,0,m.in,o,2.,normal));
        double physical=detail::physicalLayerEnergyFlux(ctx,first)+m.in.pressure*normal;
        assert(near(e.energy,-2*physical,1e-10));
        assert(std::abs(o.heatBalanceResidual)<1e-5);
    }
    """)
