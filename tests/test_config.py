from spacerec.config import Config, DepthCfg, VizCfg
from spacerec.device import select_torch_device


def test_backend_model_falls_back_to_live_model():
    cfg = DepthCfg()
    assert cfg.backend_model_resolved == cfg.model
    assert cfg.backend_process_res_resolved == cfg.process_res


def test_backend_model_override():
    cfg = DepthCfg(backend_model="depth-anything/DA3-LARGE-1.1",
                   backend_process_res=672)
    assert cfg.backend_model_resolved == "depth-anything/DA3-LARGE-1.1"
    assert cfg.backend_process_res_resolved == 672
    # 라이브 설정은 영향 받지 않는다
    assert cfg.model == "depth-anything/DA3-SMALL"
    assert cfg.process_res == 504


def test_load_reads_utf8_config_comments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(
        b"# \xed\x95\x9c\xea\xb8\x80 \xec\xa3\xbc\xec\x84\x9d\n"
        b"source: sources/sample_720p.mp4\n"
        b"realtime: false\n"
    )

    cfg = Config.load(path)

    assert cfg.source == "sources/sample_720p.mp4"
    assert cfg.realtime is False


def test_gaussian_cfg_defaults_off_and_loads_from_yaml(tmp_path):
    # 기본은 비활성 (비하락 보장) — yaml로만 켠다
    assert Config().gaussian.enabled is False

    path = tmp_path / "config.yaml"
    path.write_text(
        "gaussian:\n  enabled: true\n  period_s: 20\n  max_gaussians: 100000\n",
        encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.gaussian.enabled is True
    assert cfg.gaussian.period_s == 20
    assert cfg.gaussian.max_gaussians == 100_000
    assert cfg.gaussian.opt_steps == 150  # 미지정 필드는 기본값


def test_viz_cfg_defaults_and_loads_from_yaml(tmp_path):
    defaults = VizCfg()
    assert defaults.show_map_points is True
    assert defaults.show_live_preview is True
    assert defaults.show_gaussians is True
    assert defaults.map_recent_epochs == 0

    path = tmp_path / "config.yaml"
    path.write_text(
        "viz:\n"
        "  memory_limit: 2GB\n"
        "  point_subsample: 8\n"
        "  show_map_points: false\n"
        "  show_live_preview: false\n"
        "  show_gaussians: false\n"
        "  map_recent_epochs: 3\n",
        encoding="utf-8")

    cfg = Config.load(path)

    assert cfg.viz.memory_limit == "2GB"
    assert cfg.viz.point_subsample == 8
    assert cfg.viz.show_map_points is False
    assert cfg.viz.show_live_preview is False
    assert cfg.viz.show_gaussians is False
    assert cfg.viz.map_recent_epochs == 3


def test_select_torch_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert select_torch_device() == "cuda"


def test_select_torch_device_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert select_torch_device() == "mps"


def test_select_torch_device_honors_explicit_device(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert select_torch_device("cpu") == "cpu"
