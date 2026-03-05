
"""
train_model.py ─ Standalone Training Script
功能：
  1. 讀取 CSV 數據 (格式: image_path, temp, humidity, storage_time, spoilage_label)
  2. 訓練 HybridModel (CNN + RNN)
  3. 儲存更新後的模型權重 (.pth) 與標準化器 (.pkl)

使用方式：
  python train_model.py --data data.csv --epochs 20
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib

# Import the model structure (must match app.py)
from hybrid_model import HybridModel

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpoilageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: Path to the csv file with annotations.
        root_dir: Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Pre-process sensor data for scaling
        self.sensor_data = self.data[['temperature', 'humidity', 'storage_time']].values.astype(np.float32)
        self.scaler = StandardScaler()
        self.sensor_data = self.scaler.fit_transform(self.sensor_data)
        
        self.labels = self.data['spoilage_level'].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a blank image in case of error (or handle differently)
            image = Image.new('RGB', (IMG_HEIGHT, IMG_WIDTH))

        if self.transform:
            image = self.transform(image)

        sensor = self.sensor_data[idx]
        label = self.labels[idx]

        return image, sensor, label

    def get_scaler(self):
        return self.scaler

def train(args):
    print(f"🚀 Starting training on {DEVICE}...")
    
    # 1. Data Preparation
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(), # Normalizes to [0, 1]
    ])

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file '{args.data}' not found.")
        # Create a dummy CSV for demonstration if it doesn't exist
        print("⚠️ Creating dummy data.csv for demonstration...")
        df = pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg'], 
            'temperature': [25.0, 26.0], 
            'humidity': [0.60, 0.65], 
            'storage_time': [24.0, 48.0], # Hours
            'spoilage_level': [5.0, 15.0]
        })
        df.to_csv(args.data, index=False)
        # Create dummy images
        Image.new('RGB', (128, 128), color='red').save('img1.jpg')
        Image.new('RGB', (128, 128), color='brown').save('img2.jpg')

    dataset = SpoilageDataset(csv_file=args.data, root_dir=args.img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model Initialization
    model = HybridModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (imgs, sensors, labels) in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            sensors = sensors.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1) # (Batch, 1)

            optimizer.zero_grad()
            outputs = model(imgs, sensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # 4. Save Artifacts
    print("💾 Saving model and scaler...")
    torch.save(model.state_dict(), "spoilage_model_rnn.pth")
    joblib.dump(dataset.get_scaler(), "sensor_scaler.pkl")
    print("✅ Done! Files 'spoilage_model_rnn.pth' and 'sensor_scaler.pkl' updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Food Spoilage Model')
    parser.add_argument('--data', type=str, default='data.csv', help='Path to CSV data file')
    parser.add_argument('--img_dir', type=str, default='.', help='Directory containing images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train(args)
