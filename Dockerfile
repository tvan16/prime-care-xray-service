# PrimeCare X-ray service. Trọng số mô hình KHÔNG nằm trong image: gắn thư mục weights/ vào lúc chạy (xem README.md).
# Có GPU NVIDIA thì chạy trên GPU; không có thì tự chạy CPU (đặt DEVICE=cpu, nên dùng MODELS=["v3_fast"]).
FROM python:3.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Ho_Chi_Minh

# OpenCV cần libGL và glib
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bản pip của PyTorch đã kèm thư viện CUDA, image không cần cài CUDA riêng (chỉ cần driver NVIDIA trên máy chủ)
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113
# Bản dựng sẵn khớp torch 1.11 + CUDA 11.3; sai phiên bản thì pip sẽ tự biên dịch rất lâu
RUN pip install mmcv-full==1.7.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY oralxrays9_detector ./oralxrays9_detector

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 9005

# Nạp hai mô hình mất khoảng 20-60 giây nên cho thời gian khởi động dài
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:9005/xray/health', timeout=4).status==200 else 1)"

# 1 worker: mỗi worker nạp một bản mô hình riêng, nhiều worker sẽ hết bộ nhớ GPU
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9005", "--workers", "1"]
