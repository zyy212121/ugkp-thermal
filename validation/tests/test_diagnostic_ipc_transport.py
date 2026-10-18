#!/usr/bin/env python3
"""Exercise actual central IPC receive paths for both independent protocols."""
import pathlib
import subprocess
import tempfile
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
class DiagnosticIpcTransport(unittest.TestCase):
    def test_both_clients(self):
        for application in ('gasUGKP','FSH'):
            with self.subTest(application=application), tempfile.TemporaryDirectory() as tmp:
                code=r'''
#include "applications/APP/gpu/GpuBackendClient.C"
#include <cassert>
static unsigned warnings=0;
static void warning(void*,const ugkp::diagnostics::WarningSummary& value,const ugkp::diagnostics::SourceDescriptor*) {warnings+=value.count;}
int main() {
 int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
 ClientState client;client.fd=sockets[0];client.diagnosticSink.warning=warning;
 assert(ugkwpGpuResidentStrictSetDiagnosticOperation(&client,19,12,0.75)==0);
 assert(startRequest(&client,Op::advance,0));
 RequestHeader request{};assert(receiveObject(sockets[1],request));
 assert(WireCodec<RequestHeader>::size==48&&request.diagnosticSequence==19&&request.diagnosticStep==12&&request.diagnosticTime==0.75);
 ugkp::diagnostics::DeviceContext context{};ugkp::diagnostics::reset(context,12,34);
 auto w=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::AccuracyResidual,123);
 ugkp::diagnostics::recordWarning(&context,w);
 auto snapshot=ugkp::diagnostics::takeSnapshot(context);
 auto envelope=ugkp::diagnostics::encode(snapshot);
 ResponseHeader header{magic,protocolMajor,protocolMinor,0,0,sizeof(double),envelope.size()};
 auto encoded=encodeWire(header);assert(encoded.size()==32);
 ResponseHeader decoded{};WireCodec<ResponseHeader>::decode(encoded.data(),decoded);
 assert(decoded.diagnosticBytes==envelope.size());
 double sent=42,received=0;
 assert(writeAll(sockets[1],encoded.data(),encoded.size()));
 assert(writeAll(sockets[1],envelope.data(),envelope.size()));
 assert(writeAll(sockets[1],&sent,sizeof(sent)));
 std::uint64_t bytes=0;assert(receiveResponse(&client,bytes,true,sizeof(double))==0);
 assert(warnings==1 && bytes==sizeof(double));
 assert(receiveArray(client.fd,&received,1,&client));assert(received==42);assert(discardResponse(&client));
 // Success with no payload still delivers warnings before returning.
 header.payloadBytes=0;encoded=encodeWire(header);
 assert(writeAll(sockets[1],encoded.data(),encoded.size()));assert(writeAll(sockets[1],envelope.data(),envelope.size()));
 assert(finishNoPayload(&client)==0 && warnings==2);
 // Fatal remains failure even if the outer response status was mistakenly 0.
 ugkp::diagnostics::recordFatal(&context,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::NonFinite,124));
 snapshot=ugkp::diagnostics::takeSnapshot(context);envelope=ugkp::diagnostics::encode(snapshot);
 header.diagnosticBytes=envelope.size();encoded=encodeWire(header);
 assert(writeAll(sockets[1],encoded.data(),encoded.size()));assert(writeAll(sockets[1],envelope.data(),envelope.size()));
 client.diagnosticSink={};assert(finishNoPayload(&client)!=0);assert(client.terminalDiagnostic);
 assert(!startRequest(&client,Op::advance,0));
 close(sockets[0]);close(sockets[1]);
 // Reject incompatible layout after its 8-byte prefix, before payload reads.
 assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);client.fd=sockets[0];
 header.major=protocolMajor-1;encoded=encodeWire(header);
 assert(writeAll(sockets[1],encoded.data(),8));
 assert(receiveResponse(&client,bytes,false,0)!=0&&lastError.find("mismatch")!=std::string::npos);
 close(sockets[0]);close(sockets[1]);
 // A truncated same-version frame fails through ordinary transport errors.
 assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);client.fd=sockets[0];
 header.major=protocolMajor;encoded=encodeWire(header);
 assert(writeAll(sockets[1],encoded.data(),11));shutdown(sockets[1],SHUT_WR);
 assert(receiveResponse(&client,bytes,false,0)!=0);
 close(sockets[0]);close(sockets[1]);
}
'''.replace('APP',application)
                source=pathlib.Path(tmp)/'test.cpp';binary=pathlib.Path(tmp)/'test';source.write_text(code)
                subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(binary)],check=True)
                subprocess.run([str(binary)],check=True)
    def test_terminal_release_reply(self):
        for application in ('gasUGKP','FSH'):
            with self.subTest(application=application), tempfile.TemporaryDirectory() as tmp:
                code=r'''#define main backend_main
#include "applications/APP/private_backend/GpuBackendServer.C"
#undef main
#include <sys/socket.h>
#include <cassert>
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "test";}
extern "C" int ugkwpGpuResidentStrictCollectDiagnostics(void*,ugkp::diagnostics::Snapshot*){return 0;}
int main(){
 int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
 ServerState s; s.finalDiagnostics.hasFatal=true;
 s.finalDiagnostics.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::NonFinite,123);
 assert(!responseHeader(sockets[0],s,0,0));
 ResponseHeader header{};assert(readObject(sockets[1],header));assert(header.status!=0&&header.payloadBytes==0&&header.diagnosticBytes>0);
 std::vector<char> error;assert(readVector(sockets[1],error,header.errorBytes));
 std::vector<std::uint8_t> encoded;assert(readVector(sockets[1],encoded,header.diagnosticBytes));
 ugkp::diagnostics::Snapshot decoded;std::string failure;
 assert(ugkp::diagnostics::decode(encoded.data(),encoded.size(),decoded,failure));assert(decoded.hasFatal&&decoded.fatal.sourceSiteId==123);
 close(sockets[0]);close(sockets[1]);
 assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
 RequestHeader request{magic,static_cast<std::uint16_t>(protocolMajor-1),protocolMinor,1,0,0};
 const auto bytes=encodeWire(request);assert(writeAll(sockets[1],bytes.data(),8));
 RequestHeader received{};assert(!readObject(sockets[0],received));assert(frameReadError.find("mismatch")!=std::string::npos);
 close(sockets[0]);close(sockets[1]);
 assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);frameReadError.clear();request.major=protocolMajor;
 const auto shortBytes=encodeWire(request);assert(writeAll(sockets[1],shortBytes.data(),11));shutdown(sockets[1],SHUT_WR);
 assert(!readObject(sockets[0],received));close(sockets[0]);close(sockets[1]);
}
'''.replace('APP',application)
                source=pathlib.Path(tmp)/'test.cpp';binary=pathlib.Path(tmp)/'test';source.write_text(code)
                subprocess.run(['g++','-std=c++17','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I'+str(ROOT),'-I'+str(ROOT/'applications'/application/'gpu'),str(source),'-o',str(binary)],check=True)
                subprocess.run([str(binary)],check=True)
if __name__=='__main__':unittest.main()
