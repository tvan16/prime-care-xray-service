"""PrimeCare X-ray service: phát hiện tổn thương trên phim X-quang răng toàn cảnh. Chỉ bác sĩ được dùng."""
import logging
import socket
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.auth import CurrentUser, require_doctor
from app.config import get_settings
from app.detector import ModelNotReady, detector, enrich

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("xray")

ALLOWED_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")


def detect_instance_host(eureka_url: str) -> str:
    """IP của card mạng dùng để tới Eureka (Eureka ở localhost thì ra 127.0.0.1)."""
    parsed = urlparse(eureka_url)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((parsed.hostname or "localhost", parsed.port or 8761))  # UDP: không gửi gói nào
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if s.load_models:
        # Nạp trong threadpool để không chặn event loop; service vẫn lên nếu nạp lỗi (health sẽ báo lỗi)
        await run_in_threadpool(detector.load, s.models, s.detector_dir, s.device)
    eureka_started = False
    if s.eureka_enabled:
        try:
            from py_eureka_client import eureka_client

            host = s.eureka_instance_host or detect_instance_host(s.eureka_url)
            log.info("Đăng ký Eureka với địa chỉ %s:%s", host, s.xray_service_port)
            await eureka_client.init_async(
                eureka_server=s.eureka_url,
                app_name="xray-service",
                instance_port=s.xray_service_port,
                instance_host=host,
                instance_ip=host,
            )
            eureka_started = True
        except Exception as e:  # noqa: BLE001 - lỗi Eureka không được làm sập service
            log.error("Không đăng ký được Eureka (%s). Gateway lb://xray-service sẽ không thấy service.", e)
    yield
    if eureka_started:
        try:
            await eureka_client.stop_async()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="PrimeCare X-ray Service", lifespan=lifespan)


def ok(result: Any = None) -> Dict[str, Any]:
    return {"code": 1000, "result": result}


@app.exception_handler(HTTPException)
async def http_exc_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"code": 1000 + exc.status_code, "message": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"code": 1422, "message": "Dữ liệu gửi lên không hợp lệ"})


def decode_image(data: bytes) -> np.ndarray:
    import cv2  # import muộn để test phần xác thực không cần OpenCV

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR, đúng định dạng predict() cần
    if img is None:
        raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ")
    return img


def image_warnings(width: int, height: int) -> List[str]:
    """Mô hình học trên phim toàn cảnh (~1991x1127, tỉ lệ ~1.77). Ảnh khác hẳn thì kết quả kém tin cậy."""
    warnings = []
    ratio = width / height if height else 0
    if not 1.4 <= ratio <= 2.6:
        warnings.append("Tỉ lệ ảnh không giống phim X-quang toàn cảnh (panoramic); kết quả có thể không đáng tin.")
    if min(width, height) < 500:
        warnings.append("Ảnh có độ phân giải thấp; mô hình có thể bỏ sót tổn thương nhỏ.")
    return warnings


@app.get("/xray/health")
async def health():
    return ok({"status": "UP", "device": detector.device, "models": detector.status()})


@app.get("/xray/models")
async def list_models(_: CurrentUser = Depends(require_doctor)):
    s = get_settings()
    return ok({"default_model": s.default_model, "default_score_thr": s.default_score_thr, "models": detector.status()})


@app.post("/xray/analyze")
async def analyze(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    score_thr: Optional[float] = Form(default=None),
    user: CurrentUser = Depends(require_doctor),
):
    s = get_settings()
    name = model or s.default_model
    if name not in s.models:
        raise HTTPException(status_code=400, detail=f"Mô hình không hợp lệ. Chọn một trong: {', '.join(s.models)}")
    thr = s.default_score_thr if score_thr is None else score_thr
    if not s.min_score_thr <= thr <= s.max_score_thr:
        raise HTTPException(status_code=400, detail=f"Ngưỡng điểm phải từ {s.min_score_thr} đến {s.max_score_thr}")

    filename = (file.filename or "").lower()
    if file.content_type not in ALLOWED_TYPES and not filename.endswith(ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail="Chỉ nhận ảnh JPG hoặc PNG")
    max_bytes = int(s.max_upload_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Ảnh vượt quá {s.max_upload_mb:g} MB")

    img = decode_image(data)
    height, width = img.shape[:2]
    try:
        raw = await run_in_threadpool(detector.predict, name, img, thr)
    except ModelNotReady as e:
        raise HTTPException(status_code=503, detail=f"Mô hình {name} chưa sẵn sàng: {e}")
    except Exception:  # noqa: BLE001
        log.exception("Lỗi khi chạy mô hình %s", name)
        raise HTTPException(status_code=500, detail="Lỗi khi phân tích ảnh, vui lòng thử lại")

    # Không lưu ảnh và không ghi tên file vào log: đây là dữ liệu y tế
    log.info("analyze user=%s model=%s %dx%d -> %d box, %.0f ms",
             user.user_id, name, width, height, raw.get("num_detections", 0), raw.get("inference_ms") or 0)
    return ok({
        "model": name,
        "device": detector.device,
        "image_width": width,
        "image_height": height,
        "warnings": image_warnings(width, height),
        **enrich(raw),
    })


if __name__ == "__main__":
    import uvicorn

    # 1 worker: mỗi worker nạp một bản mô hình riêng, nhiều worker sẽ hết bộ nhớ GPU
    uvicorn.run("app.main:app", host="0.0.0.0", port=get_settings().xray_service_port, workers=1)
