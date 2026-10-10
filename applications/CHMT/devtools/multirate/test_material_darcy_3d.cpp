// Host-only mathematical verification of the actual production transport operator.
// Fixed manufactured states, not an OpenFOAM/transient coupled simulation.
#include "materials/MaterialTransport.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace chmt;
namespace {
constexpr double T=600, P0=120000, phi=0.3, pi=3.14159265358979323846;
const Vec3 slope={0.5,-0.35,0.2};
const Vec3 linearGradient={23000,17000,-13000};
void require(bool ok,const std::string& msg){if(!ok)throw std::runtime_error(msg);}
PhysicsConfig physics(int order){PhysicsConfig p;p.spatialOrder=order;p.permeability=2e-11;p.poreViscosity=1.8e-5;
 for(int c=0;c<Nc;++c){auto& t=p.condensed[c];t.rho=1200+200*c;t.cp0=900+100*c;t.conductivity=1;t.Tmin=100;t.Tmax=2000;}
 for(int s=0;s<Ns;++s){auto& t=p.species[s];t.R=280+60*s;t.cp0=1050+100*s;t.Tmin=100;t.Tmax=2000;}
 return p;}
double Y(int s){return s==0?.7:(s==1?.3:0);}
double gasR(const PhysicsConfig& p){double v=0;for(int s=0;s<Ns;++s)v+=Y(s)*p.species[s].R;return v;}
double gasH(const PhysicsConfig& p){double v=0;for(int s=0;s<Ns;++s)v+=Y(s)*p.species[s].cp0*T;return v;}
int idx(int i,int j,int k,int n){return (k*n+j)*n+i;}
bool interior(int c,int n,int margin){int i=c%n,j=c/n%n,k=c/n/n;return i>=margin&&j>=margin&&k>=margin&&i<n-margin&&j<n-margin&&k<n-margin;}
Vec3 transform(Vec3 x,double shear){return {x.x+shear*x.y,x.y,x.z};}
HostMesh cube(int n,double shear=0,bool periodic=false,bool stretched=false){
 HostMesh m;const double h=1./n;
 auto edge=[&](int i){return i*h+(stretched?.035*std::sin(2*pi*i*h):0);};
 auto centre=[&](int i){return .5*(edge(i)+edge(i+1));};
 auto width=[&](int i){return edge(i+1)-edge(i);};std::vector<std::vector<std::pair<int,int>>> entries(n*n*n);
 for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){m.cellCentres.push_back(transform({centre(i),centre(j),centre(k)},shear));m.volumes.push_back(width(i)*width(j)*width(k));}
 auto face=[&](int o,int r,Vec3 fc,Vec3 S,BoundaryKind kind){int f=m.owner.size();m.owner.push_back(o);m.neighbour.push_back(r);m.faceCentres.push_back(fc);m.areaVectors.push_back(S);m.boundaryKind.push_back(kind);m.periodicPartner.push_back(-1);entries[o].push_back({f,1});if(r>=0)entries[r].push_back({f,-1});return f;};
 for(int axis=0;axis<3;++axis)for(int b=0;b<n;++b)for(int a=0;a<n;++a){int low=-1,high=-1;
  for(int t=0;t<=n;++t){int xyz[3];xyz[axis]=t;xyz[(axis+1)%3]=a;xyz[(axis+2)%3]=b;
   Vec3 fc={axis==0?edge(t):centre(xyz[0]),axis==1?edge(t):centre(xyz[1]),axis==2?edge(t):centre(xyz[2])};
   const double area=width(a)*width(b);
   Vec3 S=axis==0?Vec3{area,-shear*area,0}:(axis==1?Vec3{0,area,0}:Vec3{0,0,area});
   int l[3]={xyz[0],xyz[1],xyz[2]},r[3]={xyz[0],xyz[1],xyz[2]};l[axis]=t-1;
   int f;if(t==0){r[axis]=0;f=face(idx(r[0],r[1],r[2],n),-1,transform(fc,shear),S*(-1),periodic?BoundaryKind::Periodic:BoundaryKind::NoSlip);low=f;}
   else if(t==n){f=face(idx(l[0],l[1],l[2],n),-1,transform(fc,shear),S,periodic?BoundaryKind::Periodic:BoundaryKind::NoSlip);high=f;}
   else f=face(idx(l[0],l[1],l[2],n),idx(r[0],r[1],r[2],n),transform(fc,shear),S,BoundaryKind::Internal);
  }
  if(periodic){m.periodicPartner[low]=high;m.periodicPartner[high]=low;}
 }
 m.cellFaceOffsets.push_back(0);for(const auto& list:entries){for(auto e:list){m.cellFaces.push_back(e.first);m.cellFaceSigns.push_back(e.second);}m.cellFaceOffsets.push_back(m.cellFaces.size());}
 for(std::size_t c=0;c<m.volumes.size();++c){Vec3 sum{};require(m.cellFaceOffsets[c+1]-m.cellFaceOffsets[c]==6,"mesh has six faces per cell");for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e)sum+=m.areaVectors[m.cellFaces[e]]*m.cellFaceSigns[e];require(mag(sum)<1e-14,"mesh closure");}
 return m;
}
enum class Field { AffineSquared,LinearPressure,PeriodicSquared,Uniform,Hydrostatic };
double pressure(Field f,Vec3 x,Vec3 g,double R){if(f==Field::AffineSquared)return P0*std::sqrt(1+dot(slope,x));if(f==Field::LinearPressure)return P0+dot(linearGradient,x);if(f==Field::PeriodicSquared)return P0*std::sqrt(1+.1*(std::sin(2*pi*x.x)+.7*std::sin(2*pi*x.y)+.4*std::sin(2*pi*x.z)));if(f==Field::Hydrostatic)return P0*std::exp(dot(g,x)/(R*T));return P0;}
std::vector<SolidQ> state(const HostMesh& m,Field f,const PhysicsConfig& p){std::vector<SolidQ> q(m.volumes.size());double R=gasR(p);
 for(std::size_t c=0;c<q.size();++c){double V=m.volumes[c],rho=pressure(f,m.cellCentres[c],p.gravity,R)/(R*T);q[c].condensed[0]=(1-phi)*p.condensed[0].rho*V;q[c].porosity=phi;q[c].progress[0]=.015*V;q[c].energy=q[c].condensed[0]*p.condensed[0].cp0*T;
  for(int s=0;s<Ns;++s){q[c].pore[s]=Y(s)*phi*rho*V;q[c].energy+=q[c].pore[s]*(p.species[s].cp0-p.species[s].R)*T;}}
 return q;
}
double mass(const SolidQ& q){double x=0;for(int s=0;s<Ns;++s)x+=q.pore[s];return x;}
struct Norm {long double error=0,reference=0;double maxError=0;int count=0;void add(double v,double r,double weight=1){error+=weight*(v-r)*(v-r);reference+=weight*r*r;maxError=std::max(maxError,std::abs(v-r));++count;}double relative()const{return std::sqrt(double(error/reference));}double rms()const{return std::sqrt(double(error/count));}};
void conservation(const HostMesh& m,const MaterialTransportResult& out,double& mr,double& er){long double dm=0,de=0,ma=0,ea=0;
 for(const auto& q:out.rates){dm+=mass(q);de+=q.energy;ma+=std::abs(mass(q));ea+=std::abs(q.energy);}for(std::size_t f=0;f<m.owner.size();++f)if(m.neighbour[f]<0&&m.boundaryKind[f]!=BoundaryKind::Periodic){dm+=mass(out.faceFlux[f]);de+=out.faceFlux[f].energy;}de-=out.bodyPower;mr=double(std::abs(dm)/std::max((long double)1e-30,ma));er=double(std::abs(de)/std::max((long double)1e-30,ea));require(mr<2e-12&&er<2e-12,"global conservative balance");}
struct Row {double flux=0,rhs=0,velocity=0;};
Row manufactured(Field field,int n,int order,double shear=0,ReconstructionMode mode=ReconstructionMode::LimitedLinear,bool stretched=false){bool periodic=field==Field::PeriodicSquared;auto m=cube(n,shear,periodic,stretched);auto p=physics(order);p.reconstruction=mode;auto q=state(m,field,p);MaterialTransportResult out;std::string error;require(evaluateMaterialTransport(m,q,p,{},1,.4,out,error),error);double R=gasR(p),K=p.permeability/p.poreViscosity/(R*T),h=1./n;Norm flux,rhs,velocity;double enthalpyError=0;
 for(std::size_t f=0;f<m.owner.size();++f){int l=m.owner[f],r=m.neighbour[f];if(m.boundaryKind[f]==BoundaryKind::Periodic){r=m.owner[m.periodicPartner[f]];if(static_cast<int>(f)>m.periodicPartner[f])continue;}if(r<0||(!periodic&&(!interior(l,n,1)||!interior(r,n,1))))continue;Vec3 S=m.areaVectors[f],x=m.faceCentres[f];double exact=0;
  if(field==Field::AffineSquared)exact=-.5*K*P0*P0*dot(slope,S);
  else if(field==Field::LinearPressure)exact=-K*pressure(field,x,{},R)*dot(linearGradient,S);
  else exact=-K*P0*P0*.1*pi*dot({std::cos(2*pi*x.x),.7*std::cos(2*pi*x.y),.4*std::cos(2*pi*x.z)},S);
  flux.add(mass(out.faceFlux[f])/mag(S),exact/mag(S),mag(S));enthalpyError=std::max(enthalpyError,std::abs(out.faceFlux[f].energy-gasH(p)*mass(out.faceFlux[f]))/std::max(1.,std::abs(out.faceFlux[f].energy)));
 }
 for(std::size_t c=0;c<q.size();++c){if(!periodic&&!interior(c,n,2))continue;Vec3 x=m.cellCentres[c];double exact=0;Vec3 grad{};double pr=pressure(field,x,{},R);
  if(field==Field::AffineSquared)grad=slope*(.5*P0*P0/pr);
  else if(field==Field::LinearPressure){exact=K*dot(linearGradient,linearGradient);grad=linearGradient;}
  else {exact=-K*P0*P0*.1*2*pi*pi*std::sin(pi*h)/(pi*h)*(std::sin(2*pi*x.x)+.7*std::sin(2*pi*x.y)+.4*std::sin(2*pi*x.z));grad={std::cos(2*pi*x.x),.7*std::cos(2*pi*x.y),.4*std::cos(2*pi*x.z)};grad=grad*(P0*P0*.1*pi/pr);}
  rhs.add(mass(out.rates[c])/m.volumes[c],exact);Vec3 expected=grad*(-p.permeability/(p.poreViscosity*phi));velocity.add(mag(out.poreVelocity[c]-expected),0);velocity.reference+=dot(expected,expected);
 }
 double mr,er;conservation(m,out,mr,er);require(enthalpyError<5e-12,"isothermal mixture enthalpy flux");require(flux.count>0&&rhs.count>0,"nonempty error regions");
 double rhsMetric=field==Field::AffineSquared?rhs.rms()/(K*P0*P0):rhs.relative();Row row{flux.relative(),rhsMetric,velocity.relative()};
 const char* name=field==Field::AffineSquared?"affine_p2":(field==Field::LinearPressure?"linear_p":"periodic_p2");
 std::cout<<name<<(stretched?"_stretched":"")<<','<<n<<','<<order<<','<<shear<<','<<(mode==ReconstructionMode::LimitedLinear?"limited":"smooth")<<','<<row.flux<<','<<row.rhs<<','<<row.velocity<<','<<mr<<','<<er<<','<<out.dtLimit<<'\n';return row;
}
Row gravity(int n,int order,bool hydro,double shear=0){auto m=cube(n,shear);auto p=physics(order);double R=gasR(p);p.gravity=hydro?Vec3{.12*R*T,-.18*R*T,.23*R*T}:Vec3{3,-7,-9};auto q=state(m,hydro?Field::Hydrostatic:Field::Uniform,p);MaterialTransportResult out;std::string error;require(evaluateMaterialTransport(m,q,p,{},1,.4,out,error),error);Norm f,u,rhs;double rho=P0/(R*T),M=p.permeability/p.poreViscosity*rho*rho;double scale=M*mag(p.gravity);
 for(std::size_t face=0;face<m.owner.size();++face){int l=m.owner[face],r=m.neighbour[face];if(r<0||!interior(l,n,1)||!interior(r,n,1))continue;double exact=hydro?0:M*dot(p.gravity,m.areaVectors[face]);f.add((mass(out.faceFlux[face])-exact)/mag(m.areaVectors[face])/scale,0);}
 for(std::size_t c=0;c<q.size();++c){if(!interior(c,n,2))continue;Vec3 exact=hydro?Vec3{}:p.gravity*(p.permeability/p.poreViscosity*rho/phi);u.add(mag(out.poreVelocity[c]-exact)/(p.permeability/p.poreViscosity*rho*mag(p.gravity)/phi),0);rhs.add(mass(out.rates[c])/m.volumes[c]/scale,0);}
 double mr,er;conservation(m,out,mr,er);if(!hydro){require(f.rms()<2e-10&&u.rms()<2e-10,"uniform gravity exact flux/velocity");require(std::abs(out.bodyPower-M*dot(p.gravity,p.gravity))<1e-12,"uniform gravity body power");}
 std::cout<<(hydro?"hydrostatic":"uniform_gravity")<<','<<n<<','<<order<<','<<shear<<",limited,"<<f.rms()<<','<<rhs.rms()<<','<<u.rms()<<','<<mr<<','<<er<<','<<out.dtLimit<<'\n';return Row{f.rms(),rhs.rms(),u.rms()};
}
void ale(int n,int order){auto m=cube(n,.4);auto p=physics(order);p.permeability=0;auto q=state(m,Field::Uniform,p);double dt=.002;std::vector<double>sweep;Vec3 shift={.2,-.3,.1};Vec3 rate={.04,.07,.03};double divergence=rate.x+rate.y+rate.z;
 for(std::size_t f=0;f<m.owner.size();++f){Vec3 x=m.faceCentres[f],v=shift+Vec3{rate.x*x.x,rate.y*x.y,rate.z*x.z};sweep.push_back(dt*dot(v,m.areaVectors[f]));}
 MaterialTransportResult out;std::string error;require(evaluateMaterialTransport(m,q,p,sweep,dt,.4,out,error),error);double maximum=0;
 for(std::size_t c=0;c<q.size();++c){double V=m.volumes[c],newV=V*(1+dt*divergence);auto check=[&](double value,double rateValue){double final=(value+dt*rateValue)/newV,initial=value/V;maximum=std::max(maximum,std::abs(final-initial)/std::max(1.,std::abs(initial)));};for(int k=0;k<Nc;++k)check(q[c].condensed[k],out.rates[c].condensed[k]);for(int s=0;s<Ns;++s)check(q[c].pore[s],out.rates[c].pore[s]);for(int r=0;r<Nr;++r)check(q[c].progress[r],out.rates[c].progress[r]);check(q[c].energy,out.rates[c].energy);}
 double mr,er;conservation(m,out,mr,er);require(maximum<2e-13,"ALE uniform state/GCL balance");std::cout<<"ale_freestream,"<<n<<','<<order<<",0.4,limited,"<<maximum<<",0,0,"<<mr<<','<<er<<','<<out.dtLimit<<'\n';
}
}
// These are operator derivatives of independently prepared isothermal states,
// not an isothermal transient obtained by resetting evolved temperatures.
void pressureJacobian(int n,int order,double shear) {
    auto p=physics(order);auto m=cube(n,shear,true);auto q=state(m,Field::Uniform,p);
    MaterialTransportResult base;std::string error;
    require(evaluateMaterialTransport(m,q,p,{},1,.4,base,error),error);
    const int cells=q.size();const double delta=.12;
    const double storage=phi*m.volumes[0]/(gasR(p)*T);
    std::vector<double> jacobian(cells*cells);
    for(int column=0;column<cells;++column){
        auto plus=q,minus=q;
        for(int species=0;species<Ns;++species){
            const double dm=q[column].pore[species]*delta/P0;
            const double de=dm*(p.species[species].cp0-p.species[species].R)*T;
            plus[column].pore[species]+=dm;minus[column].pore[species]-=dm;
            plus[column].energy+=de;minus[column].energy-=de;
        }
        MaterialTransportResult forward,backward;
        require(evaluateMaterialTransport(m,plus,p,{},1,.4,forward,error),error);
        require(evaluateMaterialTransport(m,minus,p,{},1,.4,backward,error),error);
        for(int row=0;row<cells;++row)
            jacobian[row*cells+column]=(mass(forward.rates[row])-mass(backward.rates[row]))/(2*delta*storage);
    }
    double maxAbsoluteRow=0,maxEntry=0,maxAsymmetry=0;
    for(int row=0;row<cells;++row){double sum=0;
        for(int col=0;col<cells;++col){const double value=jacobian[row*cells+col];
            sum+=std::abs(value);maxEntry=std::max(maxEntry,std::abs(value));
            maxAsymmetry=std::max(maxAsymmetry,std::abs(value-jacobian[col*cells+row]));
        }
        maxAbsoluteRow=std::max(maxAbsoluteRow,sum);
    }
    require(maxAsymmetry<1e-6*maxEntry,"periodic pressure Jacobian symmetry");
    require(base.dtLimit*maxAbsoluteRow<=.8*(1+2e-6),"corrected pressure stencil timestep row bound");
    // Every Fourier mode diagonalizes this uniform periodic 3-D stencil.
    double maxAmplification=0,minEigenvalue=0,maxEigenvalue=0,maxImaginary=0;
    for(int kz=0;kz<n;++kz)for(int ky=0;ky<n;++ky)for(int kx=0;kx<n;++kx){
        double real=0,imaginary=0;
        for(int col=0;col<cells;++col){
            const int i=col%n,j=col/n%n,k=col/n/n;
            const double phase=2*pi*(kx*i+ky*j+kz*k)/n;
            real+=jacobian[col]*std::cos(phase);imaginary+=jacobian[col]*std::sin(phase);
        }
        minEigenvalue=std::min(minEigenvalue,real);maxEigenvalue=std::max(maxEigenvalue,real);
        maxImaginary=std::max(maxImaginary,std::abs(imaginary));
        const double amplification=std::hypot(1+base.dtLimit*real,base.dtLimit*imaginary);
        maxAmplification=std::max(maxAmplification,amplification);
    }
    require(maxImaginary<1e-6*maxEntry,"periodic pressure modes have real eigenvalues");
    require(maxEigenvalue<1e-6*maxEntry,"periodic pressure operator is dissipative");
    require(maxAmplification<=1+2e-6,"explicit pressure modes stable at returned timestep");
    std::cerr<<"pressure_jacobian,n="<<n<<",order="<<order<<",shear="<<shear
        <<",dt="<<base.dtLimit<<",dt_times_row_l1="<<base.dtLimit*maxAbsoluteRow
        <<",max_amplification="<<maxAmplification<<",min_eigenvalue="<<minEigenvalue
        <<",max_eigenvalue="<<maxEigenvalue<<",relative_asymmetry="<<maxAsymmetry/maxEntry<<'\n';
}

void gradientContracts(){
    const double constant=840,ulp=std::nextafter(constant,1000.)-constant;
    require(materialReconstructionLimiter(constant,constant,constant+ulp,-ulp)==1,"unresolved constant-component roundoff does not limit other fields");
    require(materialReconstructionLimiter(0,0,1e-30,-1e-30)==0,"no absolute small-inventory limiter floor");
    require(materialReconstructionLimiter(1,0,2,2)==.5,"resolved increments retain Barth-Jespersen limiting");
    const Vec3 d={.2,.3,-.1},normal={1,0,0},gradient={7,-3,2};
    double value=-999,reverse=0,inverseLength=0;
    require(darcyNormalPressureGradient(d,normal,dot(gradient,d),gradient,gradient,value),"valid nonorthogonal gradient");
    require(std::abs(value-dot(gradient,normal))<2e-14,"linear normal derivative exactness");
    require(darcyNormalPressureGradient(d*(-1),normal*(-1),-dot(gradient,d),gradient,gradient,reverse),"reversed gradient");
    require(value==-reverse,"face derivative antisymmetry");
    require(darcyPressureStencilInverseLength(d,normal,12,17,inverseLength),"valid stencil bound");
    require(inverseLength>1/dot(d,normal),"tangential sensitivity tightens timestep");
    value=123;
    require(!darcyNormalPressureGradient({0,1,0},normal,1,gradient,gradient,value)&&value==123,"zero projected separation rejected transactionally");
    require(!darcyNormalPressureGradient(d*(-1),normal,1,gradient,gradient,value)&&value==123,"negative projected separation rejected");
    require(!darcyNormalPressureGradient(d,normal,std::numeric_limits<double>::quiet_NaN(),gradient,gradient,value),"nonfinite difference rejected");
    require(!darcyPressureStencilInverseLength(d,normal,-1,0,inverseLength),"negative gradient bound rejected");
    auto p=physics(2);auto mesh=cube(4,.6);auto q=state(mesh,Field::Uniform,p);
    MaterialTransportResult result;std::string error;
    result.bodyPower=123;
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]>=0){mesh.areaVectors[f]=mesh.areaVectors[f]*(-1);break;}
    require(!evaluateMaterialTransport(mesh,q,p,{},1,.4,result,error)&&result.bodyPower==123,"invalid face orientation rejects without publishing output");
    std::cerr<<"PASS: corrected-gradient symmetry, exactness, sensitivity and invalid-input contracts\n";
}

void converges(const Row& coarse,const Row& fine,int order,const std::string& label){
    require(coarse.flux/fine.flux>(order==1?1.8:3.5),label+" flux convergence");
    require(coarse.rhs/fine.rhs>(order==1?1.8:3.4),label+" RHS convergence");
}

int main(){try{
    std::cout<<std::scientific<<std::setprecision(12)
        <<"case,n,order,shear,reconstruction,flux_error,rhs_error,velocity_error,mass_balance_relative,energy_balance_relative,dt_limit\n";
    gradientContracts();
    for(int order:{1,2}){
        for(double shear:{0.,.6}){
            Row previousAffine{},previousHydro{};
            for(int n:{8,16,32}){
                const Row affine=manufactured(Field::AffineSquared,n,order,shear);
                const Row linear=manufactured(Field::LinearPressure,n,order,shear);
                const Row hydro=gravity(n,order,true,shear);
                require(linear.rhs<1e-8,"linear pressure has exact finite-volume divergence");
                if(order==2)require(linear.flux<1e-10,"linear pressure exact face flux on nonorthogonal mesh");
                if(n>8){converges(previousAffine,affine,order,"affine p-squared");converges(previousHydro,hydro,2,"hydrostatic");}
                previousAffine=affine;previousHydro=hydro;
            }
        }
        if(order==2)for(int n:{8,16,32}){
            const Row linear=manufactured(Field::LinearPressure,n,order,.6,ReconstructionMode::LimitedLinear,true);
            require(linear.flux<1e-10&&linear.rhs<1e-8,"linear pressure on stretched nonorthogonal cells");
        }
        Row previousPeriodic{};
        for(int n:{8,16,32}){
            const Row periodic=manufactured(Field::PeriodicSquared,n,order);
            if(order==2){
                const Row smooth=manufactured(Field::PeriodicSquared,n,order,0,ReconstructionMode::SmoothVerification);
                if(n>8)converges(previousPeriodic,periodic,2,"periodic pressure-squared");
                require(std::abs(periodic.flux-smooth.flux)<1e-12,"smooth periodic limiter equivalence");
            }
            previousPeriodic=periodic;
        }
        gravity(8,order,false);ale(8,order);
        for(double shear:{0.,.6,1.5,3.})pressureJacobian(4,order,shear);
    }
    std::cerr<<"PASS: actual production Darcy 3-D analytic flux/RHS, orthogonal/nonorthogonal convergence, periodic/gravity/ALE conservation and pressure-Jacobian timestep checks. HOST_HELPERS_ONLY.\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}}
