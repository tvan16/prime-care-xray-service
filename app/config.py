from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Service
    xray_service_port: int = 9005

    # JWT (HS512, dùng chung với user-service). repr=False để khoá không lộ ra log hay lỗi.
    jwt_secret_key: str = Field(default="", repr=False)
    # Vai trò được dùng tính năng (claim `role` trong JWT)
    allowed_roles: List[str] = ["DOCTOR"]

    # Mô hình
    load_models: bool = True                       # tắt khi chạy test
    models: List[str] = ["v4_lite", "v3_fast"]     # nạp sẵn cả hai lúc khởi động, bác sĩ chọn khi phân tích
    default_model: str = "v4_lite"
    device: str = ""                               # để trống = cuda:0 nếu có GPU, ngược lại cpu
    detector_dir: str = "oralxrays9_detector"      # thư mục chứa infer.py, configs/, weights/, mmdet/, projects/

    # Ngưỡng điểm: service trả mọi box >= score_thr; frontend tự chia nét liền (>= 0.5) / nét đứt
    default_score_thr: float = 0.3
    min_score_thr: float = 0.05
    max_score_thr: float = 0.95

    # Ảnh tải lên
    max_upload_mb: float = 10.0

    # Eureka
    eureka_enabled: bool = True
    eureka_url: str = "http://localhost:8761/eureka/"
    eureka_instance_host: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
