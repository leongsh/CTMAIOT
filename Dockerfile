# Fruit Freshness App — Dockerfile
# 使用 CPU-only PyTorch 以縮小映像大小

FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（Pillow 需要 libjpeg）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements，利用 Docker 快取層
COPY requirements.txt .

# 安裝 CPU-only torch（縮小映像）
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安裝所有其他依賴（含 jose 認證模組）
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    paho-mqtt \
    Pillow \
    numpy \
    scikit-learn \
    joblib \
    python-multipart \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    cryptography \
    bcrypt

# 複製程式碼與模型檔
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
