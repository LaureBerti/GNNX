"""Task 14 — full plumbing on the mock roster, zero real compute."""
import json

from gnnxc.cli import main


def test_run_mock_ba2motifs_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main([
        "run",
        "explainers=[mock]",
        "datasets=[ba2motifs]",
        "seeds.S=5",
        "seeds.floor=3",
        "data.n_graphs=12",
        "k_sweep=[5,10]",
    ])
    assert rc == 0
    report = json.load(open(tmp_path / "outputs" / "report.json"))
    # all fields present; one point per k in the sweep
    assert report["config"]["S"] == 5
    assert report["config"]["k_sweep"] == [5, 10]
    assert len(report["points"]) == 2  # one per k
    pt = report["points"][0]
    assert pt["explainer"] == "mock" and pt["dataset"] == "ba2motifs" and "k" in pt
    assert 0.0 <= pt["stat"] <= 1.0 and 0.0 <= pt["logic"] <= 1.0
    # deterministic mock ⇒ logical self-consistency == 1 at every k through the CLI
    assert all(p["logic"] == 1.0 for p in report["points"])
    cert = report["certificates"][0]
    assert cert["controls"]["apparatus_valid"] is True
    assert "off_diagonal_proportion" in report


def test_write_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["--write-config", str(tmp_path / "conf" / "config.yaml")])
    assert rc == 0
    assert (tmp_path / "conf" / "config.yaml").exists()
