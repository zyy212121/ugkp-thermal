#include "materials/MaterialTransport.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
PhysicsConfig config(){PhysicsConfig p; p.spatialOrder=1;p.permeability=1e-14;p.poreViscosity=1e-5;
 for(int c=0;c<Nc;++c){auto&t=p.condensed[c];t.rho=1000;t.cp0=1000;t.conductivity=1;t.Tmin=100;t.Tmax=3000;}
 for(int s=0;s<Ns;++s){auto&t=p.species[s];t.R=287;t.cp0=1000;t.Tmin=100;t.Tmax=3000;}return p;}
int main(){auto p=config();HostMesh m;m.volumes.assign(4,.001);m.cellCentres={{0,0,0},{.1,0,0},{0,.1,0},{0,0,.1}};
 m.owner={0,0,0};m.neighbour={1,2,3};m.areaVectors={{.01,0,0},{0,.01,0},{0,0,.01}};m.faceCentres={{.05,0,0},{0,.05,0},{0,0,.05}};
 m.periodicPartner.assign(3,-1);m.boundaryKind.assign(3,BoundaryKind::Internal);
 m.cellFaceOffsets={0,3,4,5,6};m.cellFaces={0,1,2,0,1,2};m.cellFaceSigns={1,1,1,-1,-1,-1};
 std::vector<SolidQ> q(4);for(int c=0;c<4;++c){q[c].condensed[0]=.9;q[c].pore[0]=c?1e-4:2e-4;q[c].porosity=.1;q[c].energy=.9*condensedE(p.condensed[0],600)+q[c].pore[0]*speciesE(p.species[0],600);}
 MaterialTransportResult out;std::string error;assert(evaluateMaterialTransport(m,q,p,{},1,.4,out,error));
 assert(out.faceFlux[0].pore[0]>0&&out.faceFlux[1].pore[0]>0&&out.faceFlux[2].pore[0]>0);
 double dm=0,de=0;for(auto&r:out.rates){dm+=r.pore[0];de+=r.energy;}assert(std::abs(dm)<1e-18);assert(std::abs(de)<1e-10);
 assert(out.dtLimit>0&&out.dtLimit<1e10);assert(out.poreVelocity[0].x>0&&out.poreVelocity[0].y>0&&out.poreVelocity[0].z>0);
 p.permeability=0;q[1].energy=.9*condensedE(p.condensed[0],900)+q[1].pore[0]*speciesE(p.species[0],900);
 assert(evaluateMaterialTransport(m,q,p,{},1,.4,out,error));for(auto&r:out.rates)assert(r.energy==0); // no duplicate explicit conduction
 p.enableReactions=true;p.nReactions=1;p.reactions[0].A=10;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;p.reactions[0].order[0]=1;
 SolidQ next;std::uint64_t count=0;assert(advanceMaterialReactions(q[0],.001,p,.1,0,.1,10000,next,count,error));assert(count>1);assert(next.energy==q[0].energy);assert(next.condensed[0]<q[0].condensed[0]);assert(std::abs(next.condensed[0]+next.condensed[1]-.9)<1e-14);
 SolidQ unchanged=next;assert(!advanceMaterialReactions(q[0],.001,p,.1,0,.1,1,next,count,error));assert(next.condensed[0]==unchanged.condensed[0]);
 HostMesh translated=m;translated.points={{1e9,0,0}};std::vector<Vec3> shifted={{1e9+.05,0,0}};Tolerances tol;
 assert(!materialPointsClose(translated,shifted,tol));translated.points={{0,0,0}};shifted={{.05,0,0}};assert(!materialPointsClose(translated,shifted,tol));
 SurfaceMesh surface;surface.solidFace={0};assert(materialSweepsCompatible(m,surface,{-.0001},{-.0001,0,0},tol));assert(!materialSweepsCompatible(m,surface,{-.0001},{-.0002,0,0},tol));
 HostMesh dense=m;dense.volumes[0]=4.1666063293320483e-9;
 const Real tinyTarget=-1e-14,overfill=3.427009277807e-20;
 assert(!materialSweepsCompatible(dense,surface,{tinyTarget},{tinyTarget-overfill,0,0},tol));
 assert(materialSweepsCompatible(dense,surface,{tinyTarget},{tinyTarget,0,0},tol));
 // A cell with two constrained faces shares one inventory error allowance;
 // individually accepting a full per-cell allowance on both would double it.
 dense.owner={0,0,0};surface.solidFace={0,1};const Real epsV=std::numeric_limits<Real>::epsilon()*dense.volumes[0];
 assert(!materialSweepsCompatible(dense,surface,{0,0},{6*epsV,6*epsV,0},tol));
 assert(materialSweepsCompatible(dense,surface,{0,0},{2*epsV,-2*epsV,0},tol));
 std::vector<SolidQ> donor(1);donor[0].condensed[0]=.3;donor[0].condensed[1]=.7;std::vector<ExchangePacket> withdrawals(3);
 for(auto& packet:withdrawals){packet.kind=ExchangeKind::GasSolid;packet.solidCell=0;packet.mass=.1;packet.condensed[0]=.1;packet.species[0]=.1;}
 std::vector<SolidQ> applied;Real correction=0;assert(applyMaterialPacketBatch(donor,withdrawals,applied,correction,error));assert(applied[0].condensed[0]==0&&correction>0&&correction<1e-15);
 withdrawals[2].condensed[0]=.11;withdrawals[2].mass=.11;withdrawals[2].species[0]=.11;assert(!applyMaterialPacketBatch(donor,withdrawals,applied,correction,error));
 std::cout<<"CPU three-axis Darcy and reaction tests passed\n";
}
