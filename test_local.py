"""
test_local.py  ─  本機快速測試腳本 (Updated with fixes)
確認模型能正常載入並對假資料做推理
"""

import torch
import numpy as np
import joblib
from torchvision import transforms
from PIL import Image
from hybrid_model import HybridModel

MODEL_PATH  = "./spoilage_model_rnn.pth"
SCALER_PATH = "./sensor_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

try:
    # 1. 載入模型
    model = HybridModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("✅ 模型載入成功")

    # 2. 載入 Scaler
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler 載入成功")

    # 3. 模擬推理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 假圖像 (全黑 128x128)
    fake_img = Image.new("RGB", (128, 128), color=(120, 80, 50))
    img_tensor = transform(fake_img).unsqueeze(0).to(DEVICE)

    # 假感測器 (溫度=25, 濕度=60%, 儲存天數=5天)
    temp = 25.0
    hum = 60.0
    storage_days = 5.0

    # --- FIX LOGIC (Same as app.py) ---
    model_hum = hum / 100.0 if hum > 1.0 else hum
    storage_hours = storage_days * 24.0

    print(f"Input: Temp={temp}, Hum={hum}%, Storage={storage_days} days")
    print(f"Model Input: Temp={temp}, Hum={model_hum}, Hours={storage_hours}")

    raw = np.array([[temp, model_hum, storage_hours]], dtype=np.float32)
    scaled = scaler.transform(raw)
    sensor_tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor, sensor_tensor).item()

    print(f"✅ 推理成功，預測腐敗值 = {pred:.4f}")

except Exception as e:
    print(f"❌ Error: {e}")
