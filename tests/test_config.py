from spacerec.config import Config
from spacerec.device import select_torch_device


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
