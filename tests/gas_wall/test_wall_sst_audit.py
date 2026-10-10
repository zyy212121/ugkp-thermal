"""Independent full-Jacobian and manufactured conservative-SST regression."""
from test_wall_model import compile_run


def test_shared_flux_jacobian_and_manufactured_omega(tmp_path):
    compile_run(tmp_path, r'''
 for(int n:{48,96,128}){
 namespace d=ugkwp::gaswall::detail;
 Model m; m.in.matching.velocity[0]=30; m.in.temperature=m.in.matching.temperature=500;m.in.matching.k=.5;m.in.matching.omega=200;
 double distances[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature.distance=distances;m.in.quadrature.volumeWeight=weights;m.in.quadrature.count=2;m.in.quadrature.volume=4e-7;m.in.quadrature.firstMoment=1.6e-9;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=n;c.maxIterations=100;
 WallWorkspace<double,2,128> w;WallOutput<double,2> o;WallStatus s;
 if(!evaluateWallModel(m.in,c,w,o,s)){std::cerr<<"FAIL coupled "<<int(s.code)<<"\n";return 1;}
 std::cout<<"N="<<n<<" coupled iterations="<<o.iterations<<" residual="<<o.residual<<" shear="<<o.traction[0]<<" omega="<<o.ownerOmega<<"\n";
 d::LayerContext<double,2> ctx;d::makeContext(m.in,c,ctx);double f[128][6],norm;assert(d::allResidual(m.in,c,ctx,w.state,w.y,f,norm));
 double summed=0,sourceSum=0;for(int i=2;i<c.nodes-1;++i){d::LayerPoint<double,2> p;assert(d::evaluatePoint(m.in,c,ctx,w.state,w.y,i,p));double vol=(w.y[i+1]-w.y[i-1])/2,yl=(w.y[i-1]+w.y[i])/2,yr=(w.y[i]+w.y[i+1])/2;
 double fit=std::pow(w.y[i],4)*(1/std::pow(yl,3)-1/std::pow(yr,3))/(3*vol),des=p.rho*sstBeta(p.f1,m.in.model.sstCoefficients)*p.omega*p.omega;
 sourceSum+=vol*(p.sourceOmega+des*(1-fit));summed+=-f[i][5]*2*vol*ctx.scale[5]/std::pow(p.z,4);}
 d::LayerFlux<double,2> first,last;assert(d::intervalFlux(m.in,c,ctx,w.state[1],w.state[2],w.y[1],w.y[2],first));assert(d::intervalFlux(m.in,c,ctx,w.state[c.nodes-2],w.state[c.nodes-1],w.y[c.nodes-2],w.y[c.nodes-1],last));
 double target=last.omega-first.omega-sourceSum;double rel=std::abs(target-summed)/std::max(1.,std::abs(sourceSum));std::cout<<"telescoping relative="<<rel<<"\n";assert(rel<2e-14);
 for(int j=1;j<c.nodes-1;++j)for(int v=0;v<6;++v){double old=w.state[j][v];w.state[j][v]+=1e-7;double pert[128][6],pn;assert(d::allResidual(m.in,c,ctx,w.state,w.y,pert,pn));w.state[j][v]=old;for(int i=0;i<c.nodes;++i)if(std::abs(i-j)>1)for(int q=0;q<6;++q)assert(pert[i][q]==f[i][q]);}
 std::cout<<"block bandwidth support exact: pass\n";

 for(int i=1;i<c.nodes-1;++i){w.state[i][0]*=1.001;w.state[i][2]+=.01*std::sin(i);w.state[i][4]*=1.001;w.state[i][5]*=1.001;}
 assert(d::allResidual(m.in,c,ctx,w.state,w.y,w.residual,norm));
 static double denseJ[768][768]{};for(int j=0;j<c.nodes;++j)for(int v=0;v<6;++v){double old=w.state[j][v],re=std::sqrt(std::numeric_limits<double>::epsilon()),vs=v==2?100.:1.;if(v==4)vs=std::sqrt(m.in.matching.k)*re;if(v==5)vs=re/std::sqrt(m.in.matching.omega);double h=re*std::max(std::abs(old),vs);w.state[j][v]=old+h;h=w.state[j][v]-old;double full[128][6],pn;assert(d::allResidual(m.in,c,ctx,w.state,w.y,full,pn));w.state[j][v]=old;for(int i=0;i<c.nodes;++i)for(int q=0;q<6;++q)denseJ[i*6+q][j*6+v]=(full[i][q]-w.residual[i][q])/h;}
 assert(d::newtonStep(m.in,c,ctx,w));double linearDefect=0;for(int i=0;i<c.nodes;++i)for(int q=0;q<6;++q){double r=w.residual[i][q];for(int j=0;j<c.nodes;++j)for(int v=0;v<6;++v)r+=denseJ[i*6+q][j*6+v]*w.delta[j][v];linearDefect=std::max(linearDefect,std::abs(r));}
 std::cout<<"full dense Jacobian Newton relative defect="<<linearDefect/norm<<"\n";assert(linearDefect/norm<1e-10);
 // Exact manufactured omega diffusion/destruction with k=0; equal beta disables F1 dependence.
 m.in.matching.velocity[0]=0;m.in.matching.k=0;m.in.model.sstCoefficients.gamma1=m.in.model.sstCoefficients.gamma2=0;m.in.model.sstCoefficients.beta2=m.in.model.sstCoefficients.beta1;
 d::LayerPoint<double,2> wall;assert(d::pointState(m.in,c,ctx,w.state[0],wall));double A=6*m.in.model.viscosity/(wall.rho*m.in.model.sstCoefficients.beta1);m.in.matching.omega=A/(.01*.01);d::makeContext(m.in,c,ctx);
 for(int i=0;i<c.nodes;++i){w.state[i][0]=w.state[i][1]=w.state[i][4]=0;w.state[i][2]=0;w.state[i][3]=.3;w.state[i][5]=w.y[i]/std::sqrt(A);}
 assert(d::allResidual(m.in,c,ctx,w.state,w.y,f,norm));double maxz=0;for(int i=2;i<c.nodes-1;++i)maxz=std::max(maxz,std::abs(f[i][5]));std::cout<<"manufactured A/y^2 omega residual="<<maxz<<"\n";double maxRoundoffFraction=0;for(int i=2;i<c.nodes-1;++i){d::LayerFlux<double,2> fl,fr;assert(d::intervalFlux(m.in,c,ctx,w.state[i-1],w.state[i],w.y[i-1],w.y[i],fl));assert(d::intervalFlux(m.in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],fr));double z=w.state[i][5],vol=(w.y[i+1]-w.y[i-1])/2,yl=(w.y[i-1]+w.y[i])/2,yr=(w.y[i]+w.y[i+1])/2;double integratedDestruction=vol*d::omegaDestructionFit(w.y[i],yl,yr)*wall.rho*m.in.model.sstCoefficients.beta1/std::pow(z,4);double bound=64*std::numeric_limits<double>::epsilon()*(std::abs(fl.omega)+std::abs(fr.omega)+std::abs(integratedDestruction))*std::pow(z,4)/(2*vol*ctx.scale[5]);maxRoundoffFraction=std::max(maxRoundoffFraction,std::abs(f[i][5])/bound);assert(std::abs(f[i][5])<bound);}std::cout<<"manufactured residual / arithmetic roundoff bound="<<maxRoundoffFraction<<"\n";
 for(int i=1;i<c.nodes-1;++i){d::LayerFlux<double,2> face;assert(d::intervalFlux(m.in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],face));double ym=(w.y[i]+w.y[i+1])/2,exact=2*m.in.model.viscosity*A/std::pow(ym,3);double wa=A/(w.y[i]*w.y[i]),wb=A/(w.y[i+1]*w.y[i+1]);double bound=64*std::numeric_limits<double>::epsilon()*(1+(wa+wb)/std::abs(wb-wa));assert(std::abs(face.omega/exact-1)<bound);}
 std::cout<<"manufactured exact shared face flux: pass\n";
}


    ''')
