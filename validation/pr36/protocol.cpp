#include "applications/FSH/gpu/GpuBackendProtocol.H"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){using namespace ugkwpGpuIpc;
 static_assert(protocolMajor==10&&protocolMinor==0,"RR wire semantics need a new protocol version");
 CreateArgs a{},b{};a.particleDiameterN=2.5;a.nCells=17;a.particleDiameterFallback=1e-4;
 unsigned char bytes[WireCodec<CreateArgs>::size];WireCodec<CreateArgs>::encode(bytes,a);WireCodec<CreateArgs>::decode(bytes,b);
 assert(b.particleDiameterN==2.5&&b.nCells==17&&b.particleDiameterFallback==1e-4);
 ParticleStuckConfigArgs c{},d{};c.sommerfeldThreshold=INFINITY;c.nFaces=19;c.heatTransferEnabled=1;
 unsigned char wall[WireCodec<ParticleStuckConfigArgs>::size];WireCodec<ParticleStuckConfigArgs>::encode(wall,c);WireCodec<ParticleStuckConfigArgs>::decode(wall,d);
 assert(std::isinf(d.sommerfeldThreshold)&&d.sommerfeldThreshold>0&&d.nFaces==19&&d.heatTransferEnabled==1);
 std::cout<<"PASS FSH RR shape and no-deposition mode wire roundtrip\n";
}
