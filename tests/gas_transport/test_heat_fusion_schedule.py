"""Source-order contract complements numeric actual-correction fusion tests."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_pairing_has_no_new_clear_or_particle_traversal():
    source=(ROOT/'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    begin=source.index('extern "C" int ugkwpGpuResidentStrictAdvance')
    source=source[begin:source.index('extern "C"',begin+1)]
    assert 'clearPairedParticleHeatKernel<<<' not in source
    assert 'reducePairedParticleHeatKernel<<<' not in source
    keys=['clearPoissonThermalPoolKernel<<<','samplePoissonPoolParticlesKernel<<<',
          'correctPoissonThermalizedMobileParticlesKernel<true>','correctPoissonThermalizedStuckParticlesKernel<true>',
          'acceptPairedParticleHeatKernel<<<','relaxColdWall1DParticlesToResidentGasKernel',
          'settlePairedParticleHeatKernel<<<','applyMobilePackingProjection','trackParticlesLocalFaceWalkKernel']
    positions=[source.index(x) for x in keys]
    assert positions==sorted(positions)
    for name in keys[2:4]:
        call=source[source.index(name):]
        assert 'dt' in call[:call.index(');')]

def test_fused_clear_and_partition_have_local_guards():
    clear=(ROOT/'common/operators/clearPoissonThermalPoolKernel.cuh').read_text()
    assert clear.index('if (c >= s.nCells)')<clear.index('clearPairedParticleHeatCell(s, c)')
    correction=(ROOT/'common/operators/correctOnePoissonThermalizedParticlePath.cuh').read_text()
    mobile=correction[correction.index('__global__ void correctPoissonThermalizedMobileParticlesKernel'):correction.index('__global__ void correctPoissonThermalizedStuckParticlesKernel')]
    assert mobile.index('const bool deferredHeatCandidate')<mobile.index('correctOnePoissonThermalizedMobileParticle')<mobile.index('accumulatePairedParticleHeatCandidate')
    assert 's.pStuck[i] != 0 && s.pStatus[i] == 2' in mobile
    assert 'if (!deferredHeatCandidate)' in mobile
    assert 'if constexpr (PairedHeat)' in mobile
    assert 'template<bool PairedHeat = false>' in correction
    stuck=correction[correction.index('__global__ void correctPoissonThermalizedStuckParticlesKernel'):]
    assert stuck.index('correctOnePoissonThermalizedStuckParticle')<stuck.index('accumulatePairedParticleHeatCandidate')
    paired=(ROOT/'common/GpuPairedParticleHeat.cuh').read_text()
    helper=paired[paired.index('__device__ __forceinline__ void accumulatePairedParticleHeatCandidate'):]
    assert helper.index('s.particleHeatCoefficient==nullptr')<helper.index('pairedParticleHeatCandidate(s,i,dt)')
