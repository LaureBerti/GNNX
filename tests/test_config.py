from gnnxc.config import default_config, write_config
from omegaconf import OmegaConf


def test_loads_default_config():
    cfg = default_config()
    assert cfg.seeds.S == 30
    assert list(cfg.datasets) == ["ba2motifs", "bamultishapes", "mutag"]
    assert len(cfg.k_sweep) > 0


def test_dot_notation_override():
    cfg = default_config()
    cfg2 = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["seeds.S=5", "k_sweep=[5,10]"]))
    assert cfg2.seeds.S == 5
    assert list(cfg2.k_sweep) == [5, 10]


def test_write_config(tmp_path):
    p = write_config(str(tmp_path / "conf" / "config.yaml"))
    loaded = OmegaConf.load(p)
    assert loaded.seeds.S == 30
