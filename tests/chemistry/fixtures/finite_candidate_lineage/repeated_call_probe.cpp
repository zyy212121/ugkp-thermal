#define main lineage_probe_original_main
#include "lineage_probe.cpp"
#undef main
int main(){
 Model m;ClosedReactorInput<double,2> b;b.speciesMass[0]=b.totalMass=.028;b.volume=.001;b.internalEnergy=mixtureEnergy(b.speciesMass,700.,m.t);ChemistryControls<double> c;c.relativeTolerance=.05;
 ClosedReactorResult<double,2> a,d,z;ChemistryAudit<double,2> audit;independent::reset();
 auto first=advanceClosedReactor(b,.1,m.t,m.m,c,a,audit);int count1=independent::counters.prepare,basis1=independent::counters.basis;
 auto zero=advanceClosedReactor(b,0.,m.t,m.m,c,z,audit);CHECK(independent::counters.prepare==count1);CHECK(independent::counters.basis==basis1);
 m.basis[0]=-2;m.basis[1]=2;m.rx[0].highRate.preExponential=.14;
 auto second=advanceClosedReactor(b,.1,m.t,m.m,c,d,audit);CHECK(first&&zero&&second);CHECK(independent::counters.prepare==2);CHECK(independent::counters.basis==2);CHECK(d.speciesMass[0]<a.speciesMass[0]);CHECK(z.speciesMass[0]==b.speciesMass[0]);
 std::cout<<"first_mass="<<std::setprecision(21)<<a.speciesMass[0]<<" changed_table_mass="<<d.speciesMass[0]<<" prepare="<<independent::counters.prepare<<" basis="<<independent::counters.basis<<"\nRESULT failures="<<failures<<"\n";return failures?1:0;
}
