# Fruit Freshness App — Dockerfile
# 使用 CPU-only PyTorch 以縮小映像大小

FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（Pillow 需要 libjpeg，wget 用於分段下載 PyTorch）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 libgl1 wget \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements，利用 Docker 快取層
COPY requirements.txt .

# 安裝 CPU-only torch（使用 wget 分段下載 + 重試，避免 pip 大檔案下載中斷）
# torch CPU wheel 約 180MB，torchvision 約 20MB（cpu 版本比 cuda 版小很多）
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    || (echo "Retry torch install..." && \
        pip install --no-cache-dir --retries 5 --timeout 180 \
        torch==2.2.2+cpu torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu)

# 安裝所有其他依賴（含 jose 認證模組）
RUN pip install --no-cache-dir --retries 3 --timeout 60 \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    paho-mqtt \
    Pillow \
    numpy \
    scikit-learn \
    joblib \
    python-multipart \
    requests \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    cryptography \
    bcrypt \
    psycopg2-binary

# 複製程式碼與模型檔
COPY . .

# 支援 Fly.io（8080）和 Render（動態 PORT）
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
