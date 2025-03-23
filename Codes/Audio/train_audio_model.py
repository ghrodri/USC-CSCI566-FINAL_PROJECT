# Train an audio emotion classifier using CNN and extracted features (MFCC, pitch, energy)

import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SAMPLE_RATE = 16000
NUM_MFCC = 40
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

class AudioFeatureDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=NUM_MFCC)

    def __len__(self):
        return len(self.filepaths)

    def extract_features(self, waveform, sr):
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        mfcc = self.mfcc(waveform).squeeze(0)[:, :200]
        if mfcc.shape[1] < 200:
            mfcc = nn.functional.pad(mfcc, (0, 200 - mfcc.shape[1]))
        pitch, _ = torchaudio.functional.detect_pitch_frequency(waveform, SAMPLE_RATE)
        pitch = pitch[:200]
        if pitch.shape[0] < 200:
            pitch = nn.functional.pad(pitch, (0, 200 - pitch.shape[0]))
        energy = waveform.pow(2).mean(dim=1).repeat(200)
        return torch.stack([mfcc.mean(dim=0), pitch, energy], dim=0)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sr = torchaudio.load(filepath)
        features = self.extract_features(waveform, sr)
        label = torch.tensor(self.labels[idx])
        return features, label

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 50, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.cnn(x))

# Load audio dataset (RAVDESS)
all_files, all_labels = [], []
for root, _, files in os.walk("data/audio_data/ravdess"):
    for file in files:
        if file.endswith(".wav"):
            emotion_id = file.split("-")[2]
            if emotion_id in LABELS:
                all_files.append(os.path.join(root, file))
                all_labels.append(int(emotion_id) - 1)

X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.1)
train_loader = DataLoader(AudioFeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

model = CNNEmotionClassifier(num_classes=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for features, labels in tqdm(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "checkpoints/audio_model.pt")

