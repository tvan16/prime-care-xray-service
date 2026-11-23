"""Nạp và chạy mô hình phát hiện tổn thương (gói oralxrays9_detector).

- Nạp các mô hình MỘT lần lúc khởi động (mỗi lần nạp mất vài giây).
- Chỉ chạy một ảnh tại một thời điểm (khóa), vì GPU 6 GB không chịu được nhiều ảnh song song.
- torch/mmcv chỉ được import khi nạp mô hình, để test chạy được mà không cần GPU hay PyTorch.
"""
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("xray")

# Thông tin hiển thị cho bác sĩ chọn mô hình (số liệu lấy từ HUONG_DAN của gói)
MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "v4_lite": {"label": "Chính xác (v4_lite)", "ap": 73.16,
                "description": "Độ chính xác cao nhất, chậm hơn một chút."},
    "v3_fast": {"label": "Nhanh (v3_fast)", "ap": 72.35,
                "description": "Nhanh hơn khoảng 2,4 lần, kém hơn chút ở sâu răng và viêm quanh chóp."},
}

# class_id -> (tên tiếng Anh như trong mô hình, tên tiếng Việt, nhóm). Nhóm dùng để frontend tô màu: bệnh lý nổi bật hơn.
CLASS_INFO: Dict[int, Dict[str, str]] = {
    0: {"class": "Apical Periodontitis", "name_vi": "Viêm quanh chóp", "group": "pathology"},
    1: {"class": "Decay", "name_vi": "Sâu răng", "group": "pathology"},
    2: {"class": "Wisdom Tooth", "name_vi": "Răng khôn", "group": "structure"},
    3: {"class": "Missing Tooth", "name_vi": "Mất răng", "group": "structure"},
    4: {"class": "Dental Filling", "name_vi": "Miếng trám", "group": "restoration"},
    5: {"class": "Root Canal Filling", "name_vi": "Điều trị tủy", "group": "restoration"},
    6: {"class": "Implant", "name_vi": "Implant", "group": "restoration"},
    7: {"class": "Porcelain Crown", "name_vi": "Mão sứ", "group": "restoration"},
    8: {"class": "Ceramic Bridge", "name_vi": "Cầu răng sứ", "group": "restoration"},
}


class ModelNotReady(Exception):
    pass


class Detector:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._predict: Optional[Callable[..., Dict[str, Any]]] = None
        self._lock = threading.Lock()
        self.device: Optional[str] = None

    # ---------- nạp mô hình ----------
    def load(self, names: List[str], detector_dir: str, device: str = "") -> None:
        root = Path(detector_dir).resolve()
        if not (root / "infer.py").is_file():
            msg = f"Không thấy {root / 'infer.py'}. Hãy giải nén gói oralxrays9_detector vào đúng thư mục."
            for n in names:
                self._errors[n] = msg
            log.error(msg)
            return
        # Dùng bản mmdet/projects đi kèm gói, không dùng bản cài bằng pip
        sys.path.insert(0, str(root))
        try:
            import infer  # noqa: WPS433  (import muộn: kéo theo torch, mmcv)
        except Exception as e:  # noqa: BLE001
            for n in names:
                self._errors[n] = f"Không import được gói mô hình: {e}"
            log.exception("Không import được infer.py")
            return
        self._predict = infer.predict
        for n in names:
            if n not in infer.MODELS:
                self._errors[n] = "Tên mô hình không có trong gói"
                continue
            try:
                log.info("Đang nạp mô hình %s ...", n)
                self._models[n] = infer.load_model(n, device or None)
                self.device = str(next(self._models[n].parameters()).device)
                log.info("Đã nạp %s trên %s", n, self.device)
            except Exception as e:  # noqa: BLE001 - một mô hình lỗi không làm hỏng mô hình còn lại
                self._errors[n] = str(e)
                log.exception("Nạp mô hình %s thất bại", n)

    # ---------- dùng trong test ----------
    def set_fake(self, models: Dict[str, Any], predict: Callable[..., Dict[str, Any]], device: str = "cpu") -> None:
        self._models, self._errors, self._predict, self.device = dict(models), {}, predict, device

    # ---------- trạng thái ----------
    def loaded(self) -> List[str]:
        return list(self._models)

    def status(self) -> List[Dict[str, Any]]:
        names = list(dict.fromkeys(list(self._models) + list(self._errors)))
        return [
            {"name": n, **MODEL_INFO.get(n, {"label": n}), "loaded": n in self._models,
             **({"error": self._errors[n]} if n in self._errors else {})}
            for n in names
        ]

    # ---------- suy luận ----------
    def predict(self, name: str, image: Any, score_thr: float) -> Dict[str, Any]:
        if name not in self._models or self._predict is None:
            raise ModelNotReady(self._errors.get(name, "Mô hình chưa được nạp"))
        with self._lock:
            return self._predict(self._models[name], image, score_thr)


detector = Detector()


def enrich(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Thêm tên tiếng Việt và nhóm cho từng box; đổi summary thành danh sách theo thứ tự lớp."""
    detections = []
    for d in raw.get("detections", []):
        info = CLASS_INFO.get(d.get("class_id"), {"class": d.get("class"), "name_vi": d.get("class"), "group": "other"})
        detections.append({**d, **info})
    summary = []
    for cid, info in CLASS_INFO.items():
        scores = [d["score"] for d in detections if d.get("class_id") == cid]
        summary.append({"class_id": cid, **info, "count": len(scores), "max_score": max(scores, default=0.0)})
    return {
        "score_thr": raw.get("score_thr"),
        "inference_ms": raw.get("inference_ms"),
        "num_detections": len(detections),
        "summary": summary,
        "detections": detections,
    }
