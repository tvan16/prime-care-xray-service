# PrimeCare X-ray Service

Phát hiện tổn thương trên phim X-quang răng toàn cảnh (9 lớp). Chỉ tài khoản **bác sĩ** được dùng.
FastAPI, cổng **9005**, đăng ký Eureka tên `xray-service`, gateway chuyển `/xray/**` tới đây.

## Chạy bằng Docker (khuyên dùng)

Không cần cài Python, PyTorch hay mmcv: tất cả nằm trong image. Chỉ cần Docker Desktop và driver NVIDIA.

1. **Trọng số** (không có trong git): tải `v4_lite.pth` và `v3_fast.pth`, chép vào
   `oralxrays9_detector/weights/`.
2. **Cấu hình:** chép `.env.example` thành `.env`.
3. **Thêm vào file docker compose** của dự án, trong mục `services:`. Ví dụ khi tất cả service cùng chạy
   trong Docker (đổi `service-registry` cho đúng tên service Eureka trong file của bạn):

   ```yaml
     xray-service:
       build: ./prime-care-xray-service
       container_name: datn-xray-service
       restart: unless-stopped
       ports:
         - "9005:9005"
       env_file:
         - path: ./prime-care-xray-service/.env
           required: false
       environment:
         EUREKA_URL: http://service-registry:8761/eureka/
         EUREKA_INSTANCE_HOST: xray-service
       volumes:
         - ./prime-care-xray-service/oralxrays9_detector/weights:/app/oralxrays9_detector/weights:ro
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

   Nếu các service Java chạy trên máy thật (không trong Docker): đặt
   `EUREKA_URL: http://host.docker.internal:8761/eureka/`, `EUREKA_INSTANCE_HOST: host.docker.internal`
   và thêm `extra_hosts: ["host.docker.internal:host-gateway"]`.

   **Máy không có GPU NVIDIA:** bỏ khối `deploy`, thêm vào `environment` hai dòng
   `DEVICE: cpu` và `MODELS: '["v3_fast"]'` (khoảng 1,5 giây/ảnh).
4. **Chạy:** `docker compose up -d --build xray-service`. Lần build đầu tải khoảng 2 GB.
   Xem log: `docker logs -f datn-xray-service`, phải có `Đã nạp v4_lite trên cuda:0`.

Gateway cần route `/xray/**` → `lb://xray-service` (đã có trong repo api-gateway).

## Cài đặt không dùng Docker (Linux hoặc WSL2, có GPU NVIDIA)

Trên Windows bật Smart App Control, PyTorch bị chặn nên phải chạy trong WSL2: dùng `run-wsl.sh`.

1. **Trọng số mô hình** (không có trong git vì mỗi file ~250 MB): chép `v4_lite.pth` và `v3_fast.pth` vào
   `oralxrays9_detector/weights/`.
2. **Môi trường** (bắt buộc Python 3.8):

   ```
   conda create -n primecare-xray python=3.8 -y
   conda activate primecare-xray
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
   pip install -r requirements.txt
   ```

   - Không chạy `pip install mmdet`: service dùng bản mmdet đi kèm trong `oralxrays9_detector/`.
   - Nếu pip bắt đầu "Building wheel for mmcv-full" và chạy rất lâu, là sai phiên bản torch hoặc Python.
   - Máy không có GPU: thay `cu113` bằng `cpu` ở hai lệnh đầu và đặt `DEVICE=cpu` trong `.env`.
3. **Cấu hình:** chép `.env.example` thành `.env`. `JWT_SECRET_KEY` phải giống user-service.

## Kiểm tra mô hình

```
cd oralxrays9_detector
python infer.py samples/026330.jpg --model v4_lite
python infer.py samples/026330.jpg --model v3_fast
```

Đúng khi mỗi lệnh ra `"num_detections": 6`: 1 `Decay` (score ~0.87, bbox ~[620, 568, 695, 653]),
4 `Wisdom Tooth`, 1 `Porcelain Crown`.

## Chạy

```
conda activate primecare-xray
python -m app.main
```

Log phải có `Đã nạp v4_lite trên cuda:0` và `Đã nạp v3_fast trên cuda:0`.
Kiểm tra: `http://localhost:9005/xray/health` (mỗi mô hình `"loaded": true`).

## API (qua gateway, cần token của bác sĩ)

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/xray/models` | Danh sách mô hình, mô hình và ngưỡng mặc định |
| POST | `/xray/analyze` | multipart: `file` (JPG/PNG ≤ 10 MB), `model` (`v4_lite`/`v3_fast`, tùy chọn), `score_thr` (tùy chọn, mặc định 0.3) |
| GET | `/xray/health` | Trạng thái nạp mô hình, GPU hay CPU |

Token không phải của bác sĩ: 403. Ảnh không được lưu lại.

## Test

Không cần GPU hay PyTorch (dùng mô hình giả):

```
python -m pytest -q
```
