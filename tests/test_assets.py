from stable_audio_wanderer import assets


def test_checkout_web_precedes_packaged_copy(tmp_path, monkeypatch):
    package = tmp_path / "stable_audio_wanderer"
    package.mkdir()
    checkout = tmp_path / "web"
    checkout.mkdir()
    (checkout / "index.html").write_text("current")
    packaged = package / "resources" / "web"
    packaged.mkdir(parents=True)
    (packaged / "index.html").write_text("old")
    monkeypatch.setattr(assets, "__file__", str(package / "assets.py"))
    monkeypatch.setattr(assets.resources, "files", lambda _: package / "resources")

    assert assets.web_directory() == checkout


def test_installed_web_uses_packaged_assets(tmp_path, monkeypatch):
    package = tmp_path / "stable_audio_wanderer"
    packaged = package / "resources" / "web"
    packaged.mkdir(parents=True)
    (packaged / "index.html").write_text("installed")
    monkeypatch.setattr(assets, "__file__", str(package / "assets.py"))
    monkeypatch.setattr(assets.resources, "files", lambda _: package / "resources")

    assert assets.web_directory() == packaged
