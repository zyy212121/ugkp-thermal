// Generic-species/precision coverage of the shared reduced-energy identity.
#include "common/gasWall/reconstruction/GeneralEnergyIdentity.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
template<class R,int N> void check(){
 using namespace ugkwp;using namespace ugkwp::gaswall;
 SpeciesThermoData<R> sp[N]{};R coefficients[3*N]{};
 WallInput<R,N> in{};WallModelConfig<R> cfg{};detail::LayerContext<R,N> ctx{};
 in.model.thermo.species=sp;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=3*N;
 in.temperature=R(300);in.normal[1]=1;in.model.viscosity=R(2e-5);in.model.molecularPrandtl=R(.72);cfg.turbulentPrandtl=R(.9);cfg.turbulentSchmidt=R(.7);
 ctx.mass=R(.03);ctx.energyReference=R(10000);ctx.wallU[0]=2;ctx.wallU[1]=-1;ctx.tangent[0][0]=1;ctx.tangent[1][2]=1;
 physical::Primitive<R,N> v{};v.T=330;v.rho=R(1.2);v.U[0]=12;v.U[1]=R(.025);v.U[2]=-3;R ysum=R(N*(N+1)/2),flux[N]{},cp=0;
 for(int j=0;j<N;++j){sp[j].coefficientOffset=3*j;sp[j].molarMass=R(.028)+R(.001)*j;sp[j].minTemperature=40;sp[j].maxTemperature=2000;coefficients[3*j]=R(1000+10*j);v.Y[j]=R(j+1)/ysum;in.massFlux[j]=ctx.mass*v.Y[j];in.model.diffusivity[j]=R(1e-5)*(j+1);ctx.wallEnthalpy[j]=coefficients[3*j]*in.temperature;ctx.enthalpyJump[j]=coefficients[3*j]*R(50);flux[j]=in.massFlux[j]+(j==0?R(.002):j==N-1?R(-.002):R(0));v.gradY[j][1]=j==0?R(3):j==N-1?R(-3):R(0);cp+=v.Y[j]*coefficients[3*j];}
 const R momentum[2]={ctx.mass*R(10)-R(.3),ctx.mass*R(-2)-R(.2)},nut=R(1e-4);
 auto t=coherent_energy::terms(in,cfg,ctx,v,nut,momentum,flux);
 R adv=0;for(int j=0;j<N;++j)adv+=flux[j]*(coefficients[3*j]*v.T-ctx.energyReference);adv+=ctx.mass*(v.U[1]*v.U[1]+R(100)+R(4))/2;
 R stress=-R(.3)*R(10)-R(.2)*R(-2),conductivity=in.model.viscosity*cp/in.model.molecularPrandtl+v.rho*nut*cp/cfg.turbulentPrandtl;
 auto close=[](R a,R b){return std::abs(a-b)<=R(64)*std::numeric_limits<R>::epsilon()*(R(1)+std::abs(b));};
 assert(close(t.advective,adv));assert(close(t.stress,stress));assert(close(t.conductivity,conductivity));
 coherent_energy::Integral<R> integral;integral.add(R(.25),t);integral.add(R(.75),t);assert(close(integral.resistance,R(1)/conductivity));assert(close(integral.energy(R(2)),adv+stress-R(2)*conductivity));
 R physicalFlux[N]{},diff[N]{},gradient[N]{},enthalpy=0;bool available=false;for(int j=0;j<N;++j)gradient[j]=v.gradY[j][1];bool ok=correctedSpeciesDiffusion(v.Y,gradient,v.rho,in.model.diffusivity,nut,cfg.turbulentSchmidt,v.T,in.model.thermo,diff,enthalpy,true,&available);assert(ok&&available);assert(coherent_energy::physicalSpecies(in,cfg,ctx,v,nut,physicalFlux));for(int j=0;j<N;++j)assert(physicalFlux[j]==ctx.mass*v.Y[j]+diff[j]);
 R wall=ctx.mass*v.U[1]*v.U[1]/2;for(int j=0;j<N;++j)wall+=in.massFlux[j]*(ctx.wallEnthalpy[j]-ctx.energyReference);assert(coherent_energy::wallTransport(in,ctx,v.U[1])==wall);
}
int main(){check<float,2>();check<float,3>();check<float,5>();check<double,2>();check<double,3>();check<double,5>();std::cout<<"6 precision/species energy-identity cases passed\n";}
