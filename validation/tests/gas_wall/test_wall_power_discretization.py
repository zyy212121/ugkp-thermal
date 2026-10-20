from test_wall_conservative_transport import compile_run

def test_omega_log_layer_and_viscous_layer_face_gradients(tmp_path):
    compile_run(tmp_path,r"""
    Model m;WallModelConfig<double> c;c.enableSst=true;detail::LayerContext<double,2> ctx;
    assert(detail::makeContext(m.in,c,ctx));detail::LayerPoint<double,2>a,b;
    a.T=b.T=300;a.rho=b.rho=1;a.cp=b.cp=1000;a.Y[0]=b.Y[0]=1;a.k=b.k=1;
    a.omega=1;b.omega=.25;a.z=1;b.z=2;
    detail::LayerFlux<double,2> flux;
    assert(detail::layerFlux(m.in,c,ctx,a,b,3.,flux,1.,4.));
    assert(near(flux.omega,m.in.model.viscosity/(2.5*2.5),1e-12));
    a.omega=1;b.omega=1./16;a.z=1;b.z=4;
    assert(detail::layerFlux(m.in,c,ctx,a,b,3.,flux,1.,4.));
    assert(near(flux.omega,2*m.in.model.viscosity/std::pow(2.5,3),1e-12));
    """)


def test_first_k_face_uses_the_smooth_wall_sst_asymptote(tmp_path):
    compile_run(tmp_path,r"""
    Model m;WallModelConfig<double> c;c.enableSst=true;detail::LayerContext<double,2>ctx;
    assert(detail::makeContext(m.in,c,ctx));detail::LayerPoint<double,2>a,b;
    a.T=b.T=300;a.rho=b.rho=1;a.cp=b.cp=1000;a.Y[0]=b.Y[0]=1;
    a.k=0;b.k=1;a.z=0;b.z=1;b.omega=1;
    const double power=(1+std::sqrt(1+24*m.in.model.sstCoefficients.betaStar/m.in.model.sstCoefficients.beta1))/2;
    detail::LayerFlux<double,2>flux;assert(detail::layerFlux(m.in,c,ctx,a,b,.001,flux,0.,.001));
    double expected=-m.in.model.viscosity*power*std::pow(.5,power-1)/.001;
    assert(near(flux.k,expected,1e-12));
    """)
