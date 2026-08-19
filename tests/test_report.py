import json

from gnnxc.report.certificate import certificate, save_certificates
from gnnxc.report.dissociation import (
    DANGEROUS,
    off_diagonal_proportion,
    plane,
    save_plane_figure,
)


def test_plane_classifies_four_cells():
    pts = plane([
        {"explainer": "a", "dataset": "d", "stat": 0.9, "logic": 0.9},  # consistent
        {"explainer": "b", "dataset": "d", "stat": 0.9, "logic": 0.1},  # DANGEROUS
        {"explainer": "c", "dataset": "d", "stat": 0.1, "logic": 0.9},  # benign-divergent
        {"explainer": "e", "dataset": "d", "stat": 0.1, "logic": 0.1},  # inconsistent
    ])
    cells = {p.explainer: p.cell for p in pts}
    assert cells["b"] == DANGEROUS
    assert off_diagonal_proportion(pts) == 0.5  # b and c


def test_certificate_flags_apparatus_validity():
    good = certificate("mock", "ba2motifs", 10, 1.0, 1.0, (0.9, 1.0), 30, False, ceiling=1.0)
    bad = certificate("mock", "ba2motifs", 10, 1.0, 0.8, (0.6, 0.9), 30, False, ceiling=0.8)
    assert good["controls"]["apparatus_valid"] is True
    assert bad["controls"]["apparatus_valid"] is False


def test_save_certificates_and_figure(tmp_path):
    recs = [certificate("mock", "d", 10, 1.0, 1.0, (0.9, 1.0), 30, False, ceiling=1.0)]
    p = save_certificates(recs, str(tmp_path / "certs.json"))
    assert json.load(open(p))["certificates"][0]["logical_consistency"] == 1.0
    pts = plane([{"explainer": "a", "dataset": "d", "stat": 0.9, "logic": 0.2}])
    fig = save_plane_figure(pts, str(tmp_path / "plane.pdf"))
    assert fig.endswith(".pdf")
