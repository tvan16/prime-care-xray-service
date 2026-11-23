from app.detector import Detector, detector, enrich
from tests.conftest import TEST_SECRET, auth, make_token, png_bytes


def post_image(client, headers, data=None, files=None):
    files = files or {"file": ("phim.png", png_bytes(), "image/png")}
    return client.post("/xray/analyze", headers=headers, data=data or {}, files=files)


# ---------- phân quyền: chỉ bác sĩ ----------

def test_no_token_is_401(client):
    r = post_image(client, {})
    assert r.status_code == 401 and r.json()["code"] == 1401


def test_invalid_or_expired_token_is_401(client):
    for token in ("abc", make_token(secret="x" * 64), make_token(exp_offset=-10)):
        r = post_image(client, {"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


def test_patient_and_admin_are_forbidden(client, fake):
    for role in ("PATIENT", "USER", "ADMIN", None):
        r = post_image(client, auth(role))
        assert r.status_code == 403, role
        assert "bác sĩ" in r.json()["message"]
    assert fake.calls == []  # không chạy mô hình cho người không có quyền


def test_models_list_also_doctor_only(client):
    assert client.get("/xray/models", headers=auth("PATIENT")).status_code == 403
    r = client.get("/xray/models", headers=auth())
    body = r.json()["result"]
    assert r.status_code == 200 and body["default_model"] == "v4_lite"
    assert {m["name"] for m in body["models"] if m["loaded"]} == {"v4_lite", "v3_fast"}


def test_health_is_public_and_hides_nothing_sensitive(client):
    r = client.get("/xray/health")
    assert r.status_code == 200 and r.json()["result"]["status"] == "UP"
    assert TEST_SECRET not in r.text


# ---------- phân tích ----------

def test_doctor_gets_enriched_result(client, fake):
    r = post_image(client, auth())
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["model"] == "v4_lite" and res["image_width"] == 1991 and res["image_height"] == 1127
    assert res["warnings"] == []
    assert res["num_detections"] == 2
    top = res["detections"][0]
    assert top["name_vi"] == "Sâu răng" and top["group"] == "pathology" and top["bbox_xyxy"] == [620.3, 568.0, 694.9, 652.9]
    assert len(res["summary"]) == 9
    decay = next(s for s in res["summary"] if s["class_id"] == 1)
    assert decay == {"class_id": 1, "class": "Decay", "name_vi": "Sâu răng", "group": "pathology", "count": 1, "max_score": 0.8696}
    # mô hình nhận ảnh BGR 3 kênh, đúng kích thước gốc, ngưỡng mặc định 0.3
    assert fake.calls == [{"model": "model-v4", "shape": (1127, 1991, 3), "score_thr": 0.3}]


def test_doctor_can_choose_model_and_threshold(client, fake):
    r = post_image(client, auth(), data={"model": "v3_fast", "score_thr": "0.5"})
    res = r.json()["result"]
    assert r.status_code == 200 and res["model"] == "v3_fast" and res["num_detections"] == 1
    assert fake.calls[-1]["model"] == "model-v3" and fake.calls[-1]["score_thr"] == 0.5


def test_unknown_model_and_bad_threshold_rejected(client, fake):
    assert post_image(client, auth(), data={"model": "yolo"}).status_code == 400
    for thr in ("0", "1", "-0.2", "0.99"):
        assert post_image(client, auth(), data={"score_thr": thr}).status_code == 400, thr
    assert fake.calls == []


def test_non_image_files_rejected(client, fake):
    cases = [
        ("a.txt", b"hello", "text/plain"),           # sai loại
        ("a.png", b"", "image/png"),                 # rỗng
        ("a.png", b"not really a png", "image/png"),  # đuôi đúng nhưng không giải mã được
    ]
    for name, data, ctype in cases:
        r = post_image(client, auth(), files={"file": (name, data, ctype)})
        assert r.status_code == 400, name
    assert fake.calls == []


def test_too_large_file_rejected(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.001)  # ~1 KB
    r = post_image(client, auth())
    assert r.status_code == 413


def test_warning_for_non_panoramic_image(client):
    r = post_image(client, auth(), files={"file": ("chup.jpg", png_bytes(400, 400), "image/jpeg")})
    w = r.json()["result"]["warnings"]
    assert r.status_code == 200 and len(w) == 2  # tỉ lệ vuông và độ phân giải thấp


def test_model_not_loaded_gives_503(client):
    detector.set_fake({"v4_lite": "m"}, lambda *a: {})  # v3_fast không được nạp
    r = post_image(client, auth(), data={"model": "v3_fast"})
    assert r.status_code == 503


def test_model_crash_gives_500_without_details(client):
    def boom(*a):
        raise RuntimeError("CUDA out of memory at /secret/path")

    detector.set_fake({"v4_lite": "m", "v3_fast": "m"}, boom)
    r = post_image(client, auth())
    assert r.status_code == 500 and "secret" not in r.text


# ---------- nạp mô hình ----------

def test_missing_bundle_reports_error_instead_of_crashing(tmp_path):
    d = Detector()
    d.load(["v4_lite"], str(tmp_path))
    st = d.status()
    assert st[0]["name"] == "v4_lite" and st[0]["loaded"] is False and "infer.py" in st[0]["error"]


def test_enrich_keeps_all_nine_classes():
    out = enrich({"score_thr": 0.3, "inference_ms": 1, "detections": []})
    assert out["num_detections"] == 0 and [s["class_id"] for s in out["summary"]] == list(range(9))
    assert all(s["count"] == 0 and s["max_score"] == 0.0 for s in out["summary"])
