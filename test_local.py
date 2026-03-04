"""
test_local.py  ─  本機快速測試腳本
確認模型能正常載入並對假資料做推理
在正式部署前執行這個腳本確認環境正確
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

# 假感測器 (溫度=25, 濕度=60, 儲存天數=5)
raw = np.array([[25.0, 60.0, 5.0]], dtype=np.float32)
scaled = scaler.transform(raw)
sensor_tensor = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    pred = model(img_tensor, sensor_tensor).item()

print(f"✅ 推理成功，預測腐敗值 = {pred:.4f}")
print("\n所有測試通過，可以部署！ 🎉")
