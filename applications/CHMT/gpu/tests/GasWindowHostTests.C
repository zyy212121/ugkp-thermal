#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "gpu/GasStability.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
static PhysicsConfig physics(){
    PhysicsConfig p;
    for(int s=0;s<Ns;++s){p.species[s].R=287;p.species[s].cp0=1005;p.species[s].Tmin=100;p.species[s].Tmax=5000;}
    p.gasConductivity=2;p.gasViscosity=0;p.liquid.rho=1000;
    return p;
}
static GasWallInput input(){
    GasWallInput in;in.bulk.Y[0]=1;in.bulk.temperature=400;in.bulk.pressure=100000;
    in.bulk.rho=100000/(287.0*400);in.bulk.soundSpeed=400;
    in.temperature=300;in.gasDistance=.1;in.area=1;in.gasArea=1;in.normal={1,0,0};in.dt=.001;
    return in;
}
static SurfacePacketIdentity identity(){
    SurfacePacketIdentity id;id.step=1;id.stage=1;id.geometry=1;id.face=42;id.gasCell=0;id.solidCell=0;id.filmFace=0;return id;
}
static void wallTests(){
    ExchangePacket counterflow;counterflow.kind=ExchangeKind::PoreGas;counterflow.poreSweep[0]=.1;
    assert(gasPacketSpeciesWithdrawal(counterflow,0)==.1);
    counterflow.pore[0]=counterflow.species[0]=-.025;assert(gasPacketSpeciesWithdrawal(counterflow,0)==.125);
    const auto p=physics();auto in=input();const auto id=identity();GasWallResult out;
    assert(evaluateGasWall(in,id,p,out));assert(out.primary.mass==0);assert(out.primary.energy==-2);
    assert(out.primary.momentum.x==100);assert(out.trace.temperature==300);
    in.speciesRate[0]=.2;in.condensedRate[0]=.2;
    assert(evaluateGasWall(in,id,p,out));assert(out.primary.mass==.0002);assert(out.pore.mass==0);
    assert(out.primary.energy==out.primary.advective+out.primary.conductive+out.primary.pressureWork+out.primary.viscousWork);
    in.poreRate[0]=.1;in.poreSweepRate[0]=.025;
    assert(evaluateGasWall(in,id,p,out));assert(out.pore.mass==.0001);assert(out.pore.poreSweep[0]==.000025);
    assert(out.pore.energy==out.pore.advective);
    in.sweptVolume=.00003125;in.normalSpeed=in.sweptVolume/(in.dt*in.gasArea);
    assert(evaluateGasWall(in,id,p,out));assert(out.primary.pressureWork==in.bulk.pressure*in.sweptVolume);
    in.temperature=-1;assert(!evaluateGasWall(in,id,p,out));
    in=input();in.speciesRate[0]=1;assert(!evaluateGasWall(in,id,p,out)); // no matching material donor
    in=input();in.primaryKind=ExchangeKind::GasFilm;in.velocity={0,2,0};in.speciesRate[0]=.2;
    assert(evaluateGasWall(in,id,p,out));assert(out.primary.condensed[0]==0);assert(out.primary.liquidKineticAdvection>0);
}
static WallProgram program(HostState& base){
    base.time=2;base.gasMesh.points={{0,0,0}};base.solidMesh.points={{0,0,0}};
    base.surface.area={1};base.surface.gasFace={0};base.surface.solidFace={0};base.surface.solidCell={0};base.surface.persistentId={42};
    WallProgram p;p.interval.begin=2;p.interval.end=4;p.surface=base.surface;
    WallKnot a;a.time=2;a.gasPoints=base.gasMesh.points;a.solidPoints=base.solidMesh.points;
    WallFaceSample face;face.temperature=300;a.faces.push_back(face);WallKnot b=a;b.time=4;b.faces[0].temperature=500;b.faces[0].speciesRate[0]=2;b.faces[0].condensedRate[0]=2;
    b.gasPoints[0]={2,4,6};b.solidPoints[0]={4,2,6};p.knots={a,b};return p;
}
static void programTests(){
    HostState base;auto p=program(base);std::string error;assert(validateGasWallProgram(p,base,error));
    WallKnot mid;assert(sampleGasWallProgram(p,3,mid,error));assert(mid.faces[0].temperature==400);assert(mid.faces[0].speciesRate[0]==1);
    assert(mid.gasPoints[0].x==1&&mid.gasPoints[0].y==2&&mid.gasPoints[0].z==3);assert(nextGasWallKnot(p,2)==4);
    WallKnot light;assert(sampleGasWallProgram(p,3,light,error,false));assert(light.gasPoints.empty()&&light.solidPoints.empty());assert(light.faces[0].temperature==400);
    assert(!sampleGasWallProgram(p,4.1,mid,error));assert(!sampleGasWallProgram(p,1.9,mid,error));
    p.knots.back().faces[0].primaryKind=ExchangeKind::GasFilm;assert(!validateGasWallProgram(p,base,error));
    p=program(base);p.knots.front().gasPoints[0].x=1;assert(!validateGasWallProgram(p,base,error));
    p=program(base);p.knots.back().time=2;assert(!validateGasWallProgram(p,base,error));
}
static void gasStabilityTests(){
    auto p=physics();p.maxDt=1;p.cfl=.5;GasPrimitive w=input().bulk;
    GasView gas;gas.nCells=1;gas.primitive=&w;
    GeometryView geometry;int offsets[]={0,1},faces[]={0};BoundaryKind kind[]={BoundaryKind::Interface};
    Vec3 area[]={{1,0,0}},face[]={{.5,0,0}},cell[]={{0,0,0}};Real sweep[]={0},volume[]={1};
    geometry.cellFaceOffsets=offsets;geometry.cellFaces=faces;geometry.boundaryKind=kind;geometry.areaVector=area;
    geometry.faceCentre=face;geometry.cellCentre=cell;geometry.sweptVolume=sweep;geometry.evaluationVolume=volume;
    const Real initial=gasOnlyStableStepLimit(gas,geometry,p,.01);assert(initial>0);
    p.permeability=1e30;p.liquid.conductivity=1e30;p.enableReactions=true;p.nReactions=1;p.reactions[0].A=1e30;
    for(int c=0;c<Nc;++c)p.condensed[c].conductivity=1e30;
    assert(gasOnlyStableStepLimit(gas,geometry,p,.01)==initial);
    p.gasConductivity=1e12;assert(gasOnlyStableStepLimit(gas,geometry,p,.01)<initial);
}
int main(){wallTests();programTests();gasStabilityTests();std::cout<<"gas window wall, waveform, donor and independent CFL host tests passed\n";}
