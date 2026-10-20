from test_wall_geometry import BASE as GEOMETRY
import test_wall_model

def test_owner_volume_k_equilibrium_and_finite_omega_moment(tmp_path):
    # Use real geometry quadrature, real converged BVP, and real shared-face
    # interpolant. This tests a volume budget, not a fitted reference value.
    original=test_wall_model.BASE
    test_wall_model.BASE=original+'\n'.join(l for l in GEOMETRY[GEOMETRY.index('using V='):GEOMETRY.index('void checkMoments')].splitlines() if not l.startswith('bool near('))+ '\n#include "common/gasTransport/GasBoundaryLayerState.H"\n'
    test_wall_model.BASE=test_wall_model.BASE.replace('#include "common/gasWall/WallModel.H"','#include "common/gasWall/WallModel.H"\n#include "common/gasWall/WallGeometryBuilder.H"\n#include <map>\n#include <algorithm>')
    try:
        test_wall_model.compile_run(tmp_path,r"""
        Model m;auto mesh=cubes(5);for(auto& p:mesh.points)for(auto&v:p)v*=.001;
        for(auto& p:mesh.cellCentres)for(auto&v:p)v*=.001;for(auto&v:mesh.volumes)v*=1e-9;
        WallGeometryDescriptor<double> d;WallStatus status;WallGeometryOptions<double> options;
        options.profileNodes=24;options.reconstructMatching=true;assert(buildWallGeometry(mesh,0,d,status,options));
        m.in.quadrature=d.quadrature();m.in.matchingDistance=d.matchingDistance;m.in.ownerDistance=d.ownerDistance;
        m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=2000;
        m.in.volumeSstClosure=true;m.in.integrateOwnerKSource=true;
        WallModelConfig<double> c;c.nodes=24;c.enableSst=true;WallWorkspace<double,2,24>w;WallOutput<double,2>o;
        assert(evaluateWallModel(m.in,c,w,o,status));assert(o.residual<1.01e-8);
        assert(near(o.ownerOmega*m.in.model.sstCoefficients.betaStar*o.ownerKContent,o.ownerKDestruction,1e-12));
        GasBoundaryLayerSstClosure<double> closure;
        closure.kFluxCount=o.kFluxCount;closure.kFluxHeight=o.kFluxHeight;closure.kTransportFlux=o.kTransportFlux;
        double area=1e-6,outer=boundaryLayerSstFlux(closure,.001);
        assert(near(o.integratedKSource,area*(outer-o.wallKFlux),1e-9));
        assert(near(o.ownerKBalanceResidual,o.integratedKSource-area*(outer-o.wallKFlux),1e-12));
        // The macro source vanishes with its matching normal-flux divergence
        // at the projected rho*k mean. No separate wall-function P is added.
        double source=o.integratedKSource+o.ownerKDestruction-
            m.in.model.sstCoefficients.betaStar*o.ownerOmega*o.ownerKContent;
        assert(near(source-area*(outer-o.wallKFlux),0.,1e-9));
        assert(std::abs(o.heatBalanceResidual)<1e-5);
        // A valid but unconverged profile must expose its SST equation defect;
        // deriving the source from flux differences would conceal this fault.
        double originalDefect=std::abs(o.integratedKSource-area*(outer-o.wallKFlux));
        detail::LayerContext<double,2>ctx;assert(detail::makeContext(m.in,c,ctx));
        w.state[10][4]*=1.3;WallOutput<double,2>bad;
        assert(detail::publishLayer(m.in,c,ctx,w,0,1.,bad));
        closure.kFluxHeight=bad.kFluxHeight;closure.kTransportFlux=bad.kTransportFlux;
        double defect=std::abs(bad.integratedKSource-area*(boundaryLayerSstFlux(closure,.001)-bad.wallKFlux));
        assert(defect>100*originalDefect+1e-8*bad.ownerKDestruction);

        """)
    finally: test_wall_model.BASE=original

def test_minimum_node_volume_flux_is_finite(tmp_path):
    test_wall_model.compile_run(tmp_path,r"""
        Model m;double y[2]={.00025,.00075},weight[2]={.0005,.0005};
        m.in.matchingDistance=.001;m.in.ownerDistance=.0005;
        m.in.quadrature={y,weight,2,.001,5e-7};m.in.volumeSstClosure=true;m.in.integrateOwnerKSource=true;
        m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=2000;
        WallModelConfig<double> c;c.nodes=2;c.enableSst=true;
        WallWorkspace<double,2,2>w;WallOutput<double,2>o;WallStatus status;
        assert(evaluateWallModel(m.in,c,w,o,status));
        assert(o.kFluxCount==3);
        for(int j=0;j<o.kFluxCount;++j)assert(std::isfinite(o.omegaDiffusiveFlux[j])&&std::isfinite(o.kTransportFlux[j]));
    """)

def test_projection_compression_preserves_real_sst_integrals(tmp_path):
    original=test_wall_model.BASE
    test_wall_model.BASE=original+'\n#include "common/gasWall/WallGeometryBuilder.H"\n'
    try:
        test_wall_model.compile_run(tmp_path,r"""
        Model m;WallGeometryDescriptor<double>d;d.volume=.001;
        std::vector<double>profile;for(int j=0;j<24;++j)profile.push_back(.001*std::pow(double(j)/23,2));
        for(int j=0;j<1000;++j){double y=.001*(j+.5)/1000;
            d.distance.push_back(y);d.volumeWeight.push_back(1e-6);d.quadraturePoints.push_back({0,y,0});}
        for(double y:{(profile[10]+profile[11])/2,(profile[10]+profile[11])/2+(profile[11]-profile[10])/8}){
            d.distance.push_back(y);d.volumeWeight.push_back(1e-4);d.quadraturePoints.push_back({0,y,0});d.volume+=1e-4;}
        m.in.matchingDistance=.001;m.in.ownerDistance=.0005;
        m.in.volumeSstClosure=true;m.in.integrateOwnerKSource=true;
        m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=2000;
        WallModelConfig<double>c;c.nodes=24;c.enableSst=true;WallStatus status;
        WallWorkspace<double,2,24>w;WallOutput<double,2>a,b;
        m.in.quadrature=d.quadrature();assert(evaluateWallModel(m.in,c,w,a,status));
        wallGeometryDetail::compressProjectionQuadrature(d,profile);
        assert(d.distance.size()<500);m.in.quadrature=d.quadrature();
        assert(evaluateWallModel(m.in,c,w,b,status));
        assert(near(a.ownerKContent,b.ownerKContent,1e-13));
        assert(near(a.ownerKDestruction,b.ownerKDestruction,1e-13));
        assert(near(a.integratedKSource,b.integratedKSource,1e-13));
        assert(near(a.ownerKBalanceResidual,b.ownerKBalanceResidual,1e-13));
        assert(near(a.ownerOmega,b.ownerOmega,1e-13));
        assert(a.traction[0]==b.traction[0]&&a.conductiveHeatFlux==b.conductiveHeatFlux);
        """)
    finally: test_wall_model.BASE=original
