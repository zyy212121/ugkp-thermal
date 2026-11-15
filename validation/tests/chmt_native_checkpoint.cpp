#include "restart/Checkpoint.C"
#include <cassert>
#include <iostream>
int main(int argc,char**argv){chmt::ModelConfig m;chmt::HostState s;std::string error;m.physics.enableParticles=true;m.particleContact.model=3;
if(std::string(argv[1])=="write"){
 for(int row=0;row<3;++row){
 chmt::ParticleQ p{};p.state=row%2;p.mass=1+row;p.thermal.initialized=true;p.thermal.status=row%2;
 for(int n=0;n<8;++n)p.thermal.coldNodeSpecificEnthalpy[n]=1000.123456789+row*17.125+n*.25;
 s.particles.push_back(p);}
 assert(chmt::writeCheckpoint(argv[2],m,s,error));
}else{
 const bool ok=chmt::readCheckpoint(argv[2],m,s,error);std::cout<<ok<<" "<<error<<"\n";
 if(std::string(argv[1])=="reject")return ok?1:0;
 if(!ok)return 2;assert(s.particles.size()==3);
 for(int row=0;row<3;++row){assert(s.particles[row].state==row%2);assert(s.particles[row].mass==1+row);assert(s.particles[row].thermal.status==row%2);
 for(int n=0;n<8;++n){using T=std::remove_reference<decltype(s.particles[row].thermal.coldNodeSpecificEnthalpy[n])>::type;assert(s.particles[row].thermal.coldNodeSpecificEnthalpy[n]==T(1000.123456789+row*17.125+n*.25));}
} } }
