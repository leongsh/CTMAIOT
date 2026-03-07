"""
hybrid_model.py
模型架構定義 - 與訓練時完全一致，供推理時載入使用
"""
import torch
import torch.nn as nn

IMG_HEIGHT = 128
IMG_WIDTH  = 128

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        # --- CNN 分支 (處理圖像) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            # 128 -> 64 -> 32 -> 16 (經過3次 MaxPool2d)
            # 16 * 16 * 128 = 32768
            nn.Linear(16 * 16 * 128, 128),
            nn.ReLU()
        )

        # --- LSTM 分支 (處理感測器數值) ---
        self.lstm    = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.lstm_fc  = nn.Linear(64, 32)
        self.lstm_relu = nn.ReLU()

        # --- 融合層 ---
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)   # 回歸輸出: spoilage_level
        )

    def forward(self, img, sensor):
        # CNN
        x1 = self.cnn(img)

        # LSTM: (batch, 3) -> (batch, 1, 3)
        sensor = sensor.unsqueeze(1)
        lstm_out, _ = self.lstm(sensor)
        x2 = lstm_out[:, -1, :]
        x2 = self.lstm_fc(x2)
        x2 = self.lstm_relu(x2)

        # Concat & 融合
        combined = torch.cat((x1, x2), dim=1)
        out = self.fusion(combined)
        return out
