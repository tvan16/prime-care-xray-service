#!/usr/bin/env bash
# Chạy xray-service trong WSL2 (Ubuntu). Windows bật Smart App Control chặn DLL của PyTorch nên không chạy trực tiếp trên Windows được.
#   wsl -d Ubuntu -- bash run-wsl.sh     (chạy từ thư mục prime-care-xray-service trên Windows)
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate primecare-xray
# Eureka chạy trên Windows: từ WSL (chế độ mạng NAT) phải gọi qua IP của Windows, IP này đổi sau mỗi lần khởi động lại
WIN_HOST=$(ip route | awk '/^default/ {print $3; exit}')
export EUREKA_URL="${WSL_EUREKA_URL:-http://$WIN_HOST:8761/eureka/}"
# Gateway trên Windows gọi localhost:9005, WSL tự chuyển cổng này vào trong Ubuntu
export EUREKA_INSTANCE_HOST="${WSL_INSTANCE_HOST:-localhost}"
echo "Eureka: $EUREKA_URL | dang ky voi dia chi: $EUREKA_INSTANCE_HOST:9005"
exec python -m app.main
