# Phát hiện tổn thương trên phim X-quang răng toàn cảnh (OralXrays-9)

Gói chạy độc lập, chỉ để **suy luận** (không train). Phát hiện 9 lớp: Apical Periodontitis
(viêm quanh chóp), Decay (sâu răng), Wisdom Tooth, Missing Tooth, Dental Filling,
Root Canal Filling, Implant, Porcelain Crown, Ceramic Bridge.

## Hai mô hình

| Tên | AP (test OralXrays-9) | Thời gian mô hình (RTX 5880, GPU rảnh) | CPU 24 nhân | Khi nào dùng |
|---|---:|---:|---:|---|
| `v4_lite` | 73.16 (bằng MOQAM gốc 73.22) | ~62 ms | ~4.2 s | Ưu tiên độ chính xác |
| `v3_fast` | 72.35 | ~26 ms | ~1.4 s | GPU yếu / không có GPU / cần nhanh |

Thời gian đầu-cuối (đọc ảnh 2 MP + tiền xử lý + mô hình) dài hơn số trên vài trăm ms.

## Cấu trúc

```
infer.py            ← API Python + dòng lệnh, trả về JSON
configs/            ← config đã gộp, không phụ thuộc repo gốc
weights/            ← v4_lite.pth, v3_fast.pth (đã bỏ trạng thái optimizer)
mmdet/ projects/ mmcv_custom/   ← code mô hình đi kèm (đừng xoá, đừng thay bằng mmdet từ pip)
samples/026330.jpg  ← ảnh mẫu để kiểm tra cài đặt
requirements.txt
```

## Cài đặt (Python 3.8, GPU NVIDIA)

```bash
conda create -n dental python=3.8 -y && conda activate dental
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install -r requirements.txt
```

- `mmcv-full` **bắt buộc đúng bản cho torch 1.11 + CUDA 11.3** (có phép toán deformable attention viết
  bằng CUDA). Nếu pip tự build từ source mất rất lâu hoặc lỗi → sai URL/phiên bản.
- **Không** `pip install mmdet`: gói dùng bản mmdet đi kèm.
- GPU đời mới (RTX 50xx) có thể không chạy được với PyTorch 1.11 → dùng CPU, hoặc báo lại để xử lý.

## Kiểm tra cài đặt

```bash
python infer.py samples/026330.jpg --model v4_lite
```
Kết quả đúng: `num_detections: 6` — Decay 1 (score ≈ 0.87), Wisdom Tooth 4, Porcelain Crown 1.

## Dùng trong web (ví dụ FastAPI)

```python
import sys; sys.path.insert(0, '/duong/dan/toi/oralxrays9_detector')
import cv2, numpy as np
from fastapi import FastAPI, UploadFile
from infer import load_model, predict

app = FastAPI()
model = load_model('v4_lite')          # nạp MỘT LẦN khi khởi động (có sẵn bước warm-up)

@app.post('/detect')
async def detect(file: UploadFile, score_thr: float = 0.3):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    return predict(model, img, score_thr)
```
Mô hình xử lý từng ảnh một; nếu nhiều request đồng thời, cho chúng xếp hàng (một worker GPU).

## Đầu ra

```json
{
  "image": "026330.jpg", "score_thr": 0.3, "inference_ms": 568.7, "num_detections": 6,
  "summary": { "Decay": {"count": 1, "max_score": 0.8696}, "...": "..." },
  "detections": [
    {"class": "Decay", "class_id": 1, "score": 0.8696, "bbox_xyxy": [620.3, 568.0, 694.9, 652.9]}
  ]
}
```
- `bbox_xyxy`: pixel trên ảnh gốc (trái, trên, phải, dưới) → vẽ khung lên ảnh.
- `score`: **độ tin cậy để xếp hạng, KHÔNG phải xác suất mắc bệnh.** Hiển thị là "độ tin cậy".

## Chọn ngưỡng `score_thr` (đo trên 2,688 ảnh test, mô hình v4_lite, IoU ≥ 0.5)

| Ngưỡng | Precision | Recall | Box sai / ảnh | Sâu răng P / R | Viêm quanh chóp P / R |
|---|---:|---:|---:|---|---|
| 0.3 | 0.81 | 0.96 | 1.62 | 0.57 / 0.89 | 0.55 / 0.86 |
| 0.4 | 0.89 | 0.95 | 0.80 | 0.71 / 0.86 | 0.73 / 0.78 |
| 0.5 | 0.95 | 0.91 | 0.35 | 0.82 / 0.79 | 0.85 / 0.64 |

Sàng lọc (ít bỏ sót) → 0.3. Kết quả gọn (ít báo nhầm) → 0.5. Bảng này đo cho `v4_lite`;
`v3_fast` kém hơn một chút ở sâu răng và viêm quanh chóp.

## Lưu ý

- Công cụ nghiên cứu, **không thay thế chẩn đoán của bác sĩ**.
- Mô hình học trên OralXrays-9 (phim toàn cảnh, ~1991×1127). Ảnh khác loại (phim quanh chóp,
  cephalometric, ảnh chụp màn hình) cho kết quả không đáng tin.
- Kiểm tra điều khoản sử dụng của bộ dữ liệu OralXrays-9 trước khi dùng cho mục đích thương mại;
  ảnh bệnh nhân phải xử lý theo quy định bảo mật dữ liệu y tế.
