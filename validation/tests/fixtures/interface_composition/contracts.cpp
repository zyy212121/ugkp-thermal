#include "ablation/InterfaceMath.H"
#include <cfenv>
#include <fenv.h>
#include <iostream>
#include <limits>
#include <cstdlib>
using namespace chmt;
void req(bool x,const char*m){if(!x){std::cerr<<m;std::exit(2);}}
int main(int argc,char**argv){feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);int kind=argc>1?std::atoi(argv[1]):0;const int mask=fegetexcept();ugkp::diagnostics::DeviceContext ctx{};PhysicsConfig p;p.diagnostics=&ctx;double Y[Ns]{},b[Ns]{},j[Ns]{},out[Ns]{};for(int i=0;i<Ns;++i){Y[i]=b[i]=1./Ns;p.species[i].R=300+i;p.gasDiffusivity[i]=.01;out[i]=123;}
if(kind==0){p.species[0].R=0;req(!interfaceComposition(j,b,101325,300,.01,p,1e-12,out),"missing R accepted");req(ctx.fatalClaim&&out[0]==123,"missing R not reported or output overwritten");}
if(kind==1){double d[Ns]{},u[Ns]{},v[Ns]{},r[Ns]{};for(int i=0;i<Ns;++i){d[i]=0;u[i]=1;r[i]=.5;}bool recovered=composition_detail::recoverRoot(1e-200,d,u,v,r,Y);for(int i=0;i<Ns;++i)req(!recovered||std::isfinite(Y[i]),"private root became nonfinite");req(!ctx.fatalClaim,"trial contaminated fatal");}
if(kind==2){double m[Ns][Ns+1]{};for(int i=0;i<Ns;++i){m[i][i]=1e-300;m[i][Ns]=1e300;}req(!composition_detail::linear(m,Y),"unrepresentable Newton correction accepted");}
if(kind==3){for(int i=0;i<Ns;++i)j[i]=1e308;req(!interfaceComposition(j,b,101325,300,.01,p,1e-12,out),"unrepresentable total flux accepted");req(ctx.fatalClaim&&out[0]==123,"true physical overflow hidden");}
if(kind==4){p.species[0].R=std::numeric_limits<double>::quiet_NaN();req(!interfaceComposition(j,b,101325,300,.01,p,1e-12,out),"NaN config accepted");req(ctx.fatalClaim,"NaN config no diagnostic");}
if(kind==5){for(int i=0;i<Ns;++i)p.gasDiffusivity[i]=0;j[0]=1;j[1]=-1;composition_detail::Info info;double copy[Ns]{};req(composition_detail::linearComposition(j,b,0,101325,300,.01,p,1e-12,out,&info),"finite linear approximation rejected");req(info.state==composition_detail::SolveState::AdmissibleApproximation&&info.actualResidual==1,"nonconverged linear result mislabeled");req(!ctx.fatalClaim,"finite precision became fatal");bool warned=false;for(auto& w:ctx.warnings)warned=warned||w.count;req(warned,"actual residual not recorded");p.diagnostics=nullptr;req(interfaceComposition(j,b,101325,300,.01,p,1e-12,copy),"output disabled changed behavior");for(int i=0;i<Ns;++i)req(copy[i]==out[i],"diagnostics changed output");}
if(kind==6){for(int i=0;i<Ns;++i){p.species[i].R=1;p.gasDiffusivity[i]=1;}j[0]=1;j[1]=-1;composition_detail::Info info;req(!composition_detail::scalarSolve(j,b,1,1,1,p,1e-12,out,&info),"unverified candidate published");req(info.state==composition_detail::SolveState::NoCertifiedCandidate&&!ctx.fatalClaim&&out[0]==123,"not-found mislabeled physical fatal or output mutated");}
if(kind==7){composition_detail::Info info;info.candidates=info.ambiguous=info.refine=999;info.state=composition_detail::SolveState::AdmissibleApproximation;req(composition_detail::scalarSolve(j,b,101325,300,.01,p,1e-12,out,&info),"zero exact state failed");req(info.state==composition_detail::SolveState::Converged&&!info.candidates&&!info.ambiguous&&!info.refine,"Info reused stale counts");}
if(kind==8){p.gasDiffusivity[0]=std::numeric_limits<double>::infinity();req(!interfaceComposition(j,b,101325,300,.01,p,1e-12,out),"infinite diffusion accepted");req(ctx.fatalClaim&&ctx.firstFatal.observed==p.gasDiffusivity[0]&&ctx.firstFatal.component==0&&ctx.firstFatal.code==ugkp::diagnostics::Code::NonFinite,"actual invalid diffusion provenance lost");}
req(fegetexcept()==mask&&std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW)==0,"mask/flags changed");std::cout<<"PASS "<<kind<<'\n';}
