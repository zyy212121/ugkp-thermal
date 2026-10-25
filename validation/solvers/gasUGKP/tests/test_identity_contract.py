from validation.tools.paths import SOURCE_ROOT
ROOT=SOURCE_ROOT/'applications/gasUGKP'
def test_resident_solver_build_and_identity():
    build=(ROOT/'../../tool/gasUGKP/build_cuda_solver.sh').read_text()
    assert 'EXE = $(FOAM_USER_APPBIN)/gasUGKP' in (ROOT/'Make/files').read_text()
    assert '$(UGKWP_CUDA_EXE_LIBS)' in (ROOT/'Make/options').read_text()
    assert 'libugkwp_cuda_backend.a' in build and '-lcudart' in build
    assert 'GAS_UGKP_FMAD must be 0 or 1' in build
    assert '#error gasUGKP must be built with UGKWP_USE_CUDA' in (ROOT/'diluteUgkwpFoam.C').read_text()
    for f in ('gpu/GpuBackendClient.C','gpu/GpuBackendProtocol.H','private_backend/GpuBackendServer.C'):
        assert not (ROOT/f).exists()
    for wrapper in ('build_public_frontend.sh','build_private_backend.sh'):
        assert 'build_cuda_solver.sh' in (ROOT/'../../tool/gasUGKP'/wrapper).read_text()
