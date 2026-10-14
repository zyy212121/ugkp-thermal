"""Regression for removal of production-only diagnostic allocation and passes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_common_operators_have_no_diagnostic_ledger_hooks():
    paths = [
        'common/GpuGasAdvance.cuh',
        'common/GpuGasOperatorDependencies.cuh',
        'common/gasTransport/GasStateView.H',
        'common/operators/computeSstFaceFluxKernel.cuh',
        'common/operators/computeGasPrimitiveGradientsKernel.cuh',
        'common/operators/advanceGasChemistryKernel.cuh',
    ]
    for relative in paths:
        source = (ROOT / relative).read_text()
        for token in ('GasBudgetAudit', 'GasSstAudit', 'gasBudgetFlux', 'gasBudgetChemistry', 'gasSstAudit'):
            assert token not in source, (relative, token)
    assert not (ROOT / 'common/gasTransport/GasBudgetAudit.H').exists()
    assert not (ROOT / 'common/gasTransport/GasSstAudit.H').exists()


def test_native_backend_has_no_budget_resources_or_environment_switch():
    directory = ROOT / 'applications/gasUGKP/private_backend'
    assert not (directory / 'GasBudgetRuntime.cuh').exists()
    for name in ('GpuResidentStrict.cu', 'SharedGasAdapter.cuh', 'SharedGasTrialPolicy.cuh',
                 'SharedGasTrialFields.H', 'BoundaryLayerStorage.cuh', 'BoundaryLayerAdapter.cuh'):
        source = (directory / name).read_text()
        for token in ('GasBudgetRuntime', 'gasBudgetAudit', 'gasBudgetHost', 'gasSstAudit', 'UGKP_GAS_BUDGET_OUTPUT'):
            assert token not in source, (name, token)
    policy = (directory / 'SharedGasTrialPolicy.cuh').read_text()
    assert 'static int commitInterval(DeviceState* s){return s->gasModelPoisoned?1:0;}' in policy


def test_wall_wire_and_csv_remove_only_audit_columns():
    protocol = (ROOT / 'applications/gasUGKP/gpu/GpuBackendProtocol.H').read_text()
    writer = (ROOT / 'applications/gasUGKP/gpu/BoundaryLayerInput.H').read_text()
    producer = (ROOT / 'applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh').read_text()
    assert 'boundaryLayerDiagnosticScalars=25' in protocol
    assert 'reservedAudit=0' in protocol
    assert 'budgetAuditAvailable' not in writer
    assert 'budgetAudit' not in writer
    assert 'if(col>=16&&col<25)continue;' in writer
    assert 'r[25+k]=o.wallSpeciesFlux[k]' in producer
    assert 'r[25+species+k]=o.reactionIntegral[k]' in producer


def test_chemistry_runtime_keeps_control_residuals_without_audit_output():
    integrator = (ROOT / 'common/chemistry/LocalStiffIntegrator.H').read_text()
    adapter = (ROOT / 'common/operators/advanceGasChemistryKernel.cuh').read_text()
    for source in (integrator, adapter):
        for token in ('ChemistryAudit', 'localAudit', 'maximumLocalError', 'speciesMassChange', 'nonlinearIterations'):
            assert token not in source
    assert 'if(RetryTrial)' in integrator
    for token in ('massResidual', 'energyResidual', 'controls.absoluteElementTolerance',
                  'status.attemptedSteps', 'status.rejectedSteps', 'covered+=h', 'stepError('):
        assert token in integrator
    view = (ROOT / 'common/gasTransport/GasStateView.H').read_text()
    assert 'chemistryAudit' not in view
    assert 'chemistryStatus' in view
    policy = (ROOT / 'applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh').read_text()
    assert 'chemistryAudit' not in policy
    assert 'snapshot.chemistryStatus' in policy
    assert 'captureChemistryAudit' not in (ROOT / 'common/GpuGasAdvance.cuh').read_text()


def test_chmt_runtime_has_no_unconsumed_report_accumulators():
    app = ROOT / 'applications/CHMT'
    for relative in ('materials/CpuMaterialDriver.C', 'io/MultirateEvolution.C',
                     'film/CpuFilmDriver.C'):
        source = (app / relative).read_text()
        for token in ('budgetDelta', 'numericalMassRoundoff', 'materialRhsEvaluations',
                      'boundaryDriveSamples', 'reactionSteps', 'filmSteps',
                      'FilmRateBudget', 'e.budget', 'e.edgeFlux'):
            assert token not in source, (relative, token)
    material = (app / 'materials/CpuMaterialDriver.C').read_text()
    assert 'accumulated.materialSteps>=' in material
    assert 'physicalSweepSum[f].value()-actualSweepSum[f].value()' in material
    film = (app / 'film/CpuFilmDriver.C').read_text()
    assert 'e.rhs[f].enthalpy+=bodyPower+supportPower+prescribedTopPower+localRadiation' in film
    assert 'addRate(e.rhs[owner],flux,-1)' in film
    evolution = (app / 'io/MultirateEvolution.C').read_text()
    assert 'candidate.nextDt=report.nextGasDt' in evolution
    assert 'report.acceptedGasMicrosteps=history.records().size()' in evolution
    assert 'compareWallPrograms(' in evolution
    assert 'auditCoupledCandidate(' not in evolution
