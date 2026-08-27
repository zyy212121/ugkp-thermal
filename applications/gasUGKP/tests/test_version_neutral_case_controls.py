from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_control_dictionary_and_legacy_migration_contract() -> None:
    source = (ROOT / "readGpuGasConfiguration.H").read_text(encoding="utf-8")
    assert 'fvSolutionDict.found("UGKP")' in source
    assert 'fvSolutionDict.subDict("UGKP")' in source
    assert 'fvSolutionDict.found("UGKP")' in source
    assert 'fvSolutionDict.found("UGKP")' in source
    assert 'fvSolutionDict.found("UGKP")' in source


def test_private_source_case_documents_automatic_build() -> None:
    readme = (
        ROOT.parents[1]
        / "examples/consistency/windSandShockTube/README.md"
    ).read_text(encoding="utf-8")
    assert "./Allrun" in readme
    assert "build_validation_solver.sh" in readme


def test_current_examples_have_no_versioned_solver_control_block() -> None:
    examples = ROOT / "examples"
    for path in examples.rglob("fvSolution"):
        text = path.read_text(encoding="utf-8")
        assert not any(
            line.strip() in {"UGKP", "UGKP", "UGKP"}
            for line in text.splitlines()
        ), path
