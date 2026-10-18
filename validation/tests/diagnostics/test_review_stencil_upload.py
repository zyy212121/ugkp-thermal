import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_upload_range_prevents_invalid_device_access(tmp_path):
    src=tmp_path/'range.cpp';exe=tmp_path/'range'
    src.write_text(r'''#include "gasTransport/GasMeshUploadValidation.H"
#include <cassert>
#include <climits>
int main(){ugkp::diagnostics::Record r{};int a[2]={0,4},n[2]={4,4};
assert(ugkwp::gasMeshStencilUploadValid(2,8,a,n,r));
a[0]=8;n[0]=1;assert(!ugkwp::gasMeshStencilUploadValid(2,8,a,n,r));assert(r.entityIndex==0 && r.observed==8);
a[0]=INT_MAX;n[0]=INT_MAX;assert(!ugkwp::gasMeshStencilUploadValid(2,8,a,n,r));
a[0]=0;n[0]=-1;assert(!ugkwp::gasMeshStencilUploadValid(2,8,a,n,r));
a[0]=-1;n[0]=0;assert(ugkwp::gasMeshStencilUploadValid(2,8,a,n,r));
assert(!ugkwp::gasMeshStencilUploadValid(2,8,nullptr,n,r));
}''')
    subprocess.run(['g++','-std=c++17','-fsanitize=address,undefined','-I'+str(SOURCE_ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
    for app,folder in [('gasUGKP','private_backend'),('FSH','private_backend'),('CHT','gpu')]:
        text=(SOURCE_ROOT/f'applications/{app}/{folder}/GpuResidentStrict.cu').read_text()
        upload=text[text.index('"mesh upload"'):]
        assert upload.index('gasMeshStencilUploadValid') < upload.index('copyToDevice')
        assert 'recordHostFatal(topologyFailure)' in upload
