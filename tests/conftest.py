"""Test chạy không cần GPU hay PyTorch: không nạp mô hình thật, thay bằng mô hình giả."""
import os
import time

# Đặt trước khi Settings được tạo; biến môi trường thắng file .env
TEST_SECRET = "t" * 64
os.environ["JWT_SECRET_KEY"] = TEST_SECRET
os.environ["LOAD_MODELS"] = "false"
os.environ["EUREKA_ENABLED"] = "false"

import cv2  # noqa: E402
import jwt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.detector import detector  # noqa: E402
from app.main import app  # noqa: E402


def make_token(role="DOCTOR", secret=TEST_SECRET, exp_offset=600):
    payload = {"sub": "u-1", "id": 1, "role": role, "exp": int(time.time()) + exp_offset}
    return jwt.encode(payload, secret.encode(), algorithm="HS512")


def auth(role="DOCTOR"):
    return {"Authorization": f"Bearer {make_token(role)}"}


def png_bytes(width=1991, height=1127):
    ok, buf = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class FakePredict:
    """Giả lập infer.predict: trả 2 box (sâu răng 0.87, mão sứ 0.45) và ghi lại tham số đã nhận."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, image, score_thr):
        self.calls.append({"model": model, "shape": image.shape, "score_thr": score_thr})
        dets = [
            {"class": "Decay", "class_id": 1, "score": 0.8696, "bbox_xyxy": [620.3, 568.0, 694.9, 652.9]},
            {"class": "Porcelain Crown", "class_id": 7, "score": 0.45, "bbox_xyxy": [100.0, 200.0, 180.0, 260.0]},
        ]
        dets = [d for d in dets if d["score"] >= score_thr]
        return {"image": None, "score_thr": score_thr, "inference_ms": 12.3,
                "num_detections": len(dets), "summary": {}, "detections": dets}


@pytest.fixture
def fake():
    f = FakePredict()
    detector.set_fake({"v4_lite": "model-v4", "v3_fast": "model-v3"}, f)
    return f


@pytest.fixture
def client(fake):
    with TestClient(app) as c:
        yield c
