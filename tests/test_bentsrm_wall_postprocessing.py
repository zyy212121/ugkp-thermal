import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "examples/thermal/bentSRM_coldWall/assets/postprocessing/"
    "bentsrm_wall_model_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("bentsrm_wall_model_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_no_wall_contact_temperatures_writes_empty_outputs_and_returns(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    monkeypatch.setattr(MODULE, "DATA", data)
    monkeypatch.setattr(MODULE, "FIGURES", figures)
    monkeypatch.setattr(MODULE, "coupled_wall_particle_temperature_records", lambda: ([], None))

    MODULE.coupled_wall_particle_temperature_results()

    csv_path = data / "bentSRM_wall_model_coupled_wall_particle_temperature.csv"
    figure_path = figures / "bentSRM_wall_model_coupled_wall_particle_temperature.png"
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "temperature_component,time_s,particle_count,temperature_median_K,temperature_mean_K,temperature_p25_K,temperature_p75_K"
    ]
    assert figure_path.is_file()
    assert "No wall-contact particle temperatures yet" in capsys.readouterr().out