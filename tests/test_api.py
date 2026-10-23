import io
import importlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(tiny_checkpoint, monkeypatch_module):
    monkeypatch_module.setenv("CHECKPOINT_PATH", tiny_checkpoint)
    import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def jpeg(color="red", size=(64, 48), noise=False):
    img = Image.new("RGB", size, color)
    if noise:
        img = Image.effect_noise(size, 80).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["encoder"] == "resnet50"


def test_predict_returns_caption_attention_and_warnings(client):
    r = client.post("/predict?num_captions=2", files={"file": ("x.jpg", jpeg(noise=True), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["caption"], str) and body["caption"]
    assert 1 <= len(body["captions"]) <= 2
    att = body["attention"]
    assert att["grid"] == 7 and att["crop"] == "full"
    assert len(att["maps"]) == len(att["words"]) == len(att["word_probs"])
    assert all(len(m) == 49 for m in att["maps"])
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["inference_ms"] >= 0


def test_blank_image_is_flagged(client):
    body = client.post("/predict", files={"file": ("x.jpg", jpeg("white"), "image/jpeg")}).json()
    assert any("little visual detail" in w for w in body["warnings"])


@pytest.mark.parametrize("kind,status", [("garbage", 400), ("too_large", 413)])
def test_bad_uploads(client, kind, status):
    data = b"not an image" if kind == "garbage" else b"0" * (11 * 1024 * 1024)
    assert client.post("/predict", files={"file": ("x.jpg", data, "image/jpeg")}).status_code == status


def test_beam_size_is_validated(client):
    r = client.post("/predict?beam_size=50", files={"file": ("x.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422
