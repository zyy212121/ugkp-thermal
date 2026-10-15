
int main(){
 DeviceState s;const double tiny=sizeof(PressureReal)==8?1e-301:1e-31;
 for(double dt:{tiny,1.,2.}){auto b=pressureLocalConvexScale(s,0,dt);printf("step %a %a %a %a\n",dt,double(b),double(s.solidPressurePhiMomX[0]*b),double(-dt*s.solidPressurePhiMomX[0]*b));}
 const PressureReal cutoff=sizeof(PressureReal)==8?PressureReal(1e300):PressureReal(1e30);
 for(int field=0;field<13;++field)for(PressureReal value:{std::nextafter(cutoff,PressureReal(0)),cutoff,std::nextafter(cutoff,PressureReal(INFINITY))}){
  DeviceState p;PressureReal* fields[]={p.momRhoP,p.momRhoUPx,p.momRhoUPy,p.momRhoUPz,p.momRhoEP,p.V,p.cellLength,&p.thetaMin,&p.pressureKickFraction,&p.epsSMin,&p.rhoSolid,p.V,&p.pressureKickFraction};
  *fields[field]=value;const double dt=field==11?double(cutoff):1.;
  printf("fallback %d %a %a\n",field,double(value),double(pressureLocalConvexScale(p,0,dt)));
 }
 return 0;
}
