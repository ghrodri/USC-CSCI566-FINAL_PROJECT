# Train a video emotion recognition model using frame sequences (CNN + LSTM)

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
SEQUENCE_LENGTH = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class VideoSequenceDataset(Dataset):
    def __init__(self, csv_file, frame_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.frame_dir = frame_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = self.data.iloc[idx]['video_id']
        label = self.data.iloc[idx]['label']
        frames = []
        for i in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(self.frame_dir, f"{video_id}_frame{i}.jpg")
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        video_tensor = torch.stack(frames)
        return video_tensor, int(label)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        cnn_features = []
        for t in range(T):
            out = self.cnn(x[:, t, :, :, :])
            cnn_features.append(out)
        cnn_seq = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_seq)
        return self.fc(lstm_out[:, -1, :])

# Prepare data
df = pd.read_csv("data/video_data/labels.csv")
X_train, X_val = train_test_split(df, test_size=0.1)
X_train.to_csv("data/video_data/train.csv", index=False)
X_val.to_csv("data/video_data/val.csv", index=False)

train_dataset = VideoSequenceDataset("data/video_data/train.csv", "data/video_data/frames", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNLSTM(num_classes=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for videos, labels in tqdm(train_loader):
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "checkpoints/video_model.pt")

