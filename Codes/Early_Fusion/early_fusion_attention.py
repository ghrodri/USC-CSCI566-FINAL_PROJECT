import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load serialized audio-text fusion features ===
DATA_PATH = "/home1/ggrimald/FINAL_PROJECT/Dataset"

def load_features(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

train_data = load_features(os.path.join(DATA_PATH, "train_audio_features.pkl"))
test_data = load_features(os.path.join(DATA_PATH, "test_audio_features.pkl"))

# === Load BERT tokenizer and model ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

# === Determine max lengths for padding ===
max_text_len = max([len(tokenizer.encode(sample["utterance"], truncation=False)) for sample in train_data + test_data])
max_audio_len = max([len(sample["features"]) for sample in train_data + test_data])

# === Text encoding: use [CLS] from BERT ===
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_text_len)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# === Pad audio feature vectors ===
def pad_audio(audio_feat, max_len):
    if len(audio_feat) >= max_len:
        return audio_feat[:max_len]
    else:
        return np.pad(audio_feat, (0, max_len - len(audio_feat)), mode="constant")

# === Build input matrices for model ===
def prepare_data(samples):
    text_vecs, audio_vecs, labels = [], [], []
    for sample in samples:
        text_feat = get_text_embedding(sample["utterance"])
        audio_feat = pad_audio(sample["features"], max_audio_len)
        text_vecs.append(text_feat)
        audio_vecs.append(audio_feat)
        labels.append(sample["label"])
    return np.array(text_vecs), np.array(audio_vecs), np.array(labels)

text_train, audio_train, y_train = prepare_data(train_data)
text_test, audio_test, y_test = prepare_data(test_data)

# === Encode emotion labels ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# === Convert to tensors ===
text_train_tensor = torch.tensor(text_train, dtype=torch.float32)
audio_train_tensor = torch.tensor(audio_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)

text_test_tensor = torch.tensor(text_test, dtype=torch.float32)
audio_test_tensor = torch.tensor(audio_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

# === Split training set into train/val ===
full_dataset = TensorDataset(text_train_tensor, audio_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(TensorDataset(text_test_tensor, audio_test_tensor, y_test_tensor), batch_size=32)

# === Attention-based Early Fusion Model ===
class EarlyFusionAttention(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_feat, audio_feat):
        t = self.text_proj(text_feat).unsqueeze(1)   # [B, 1, H]
        a = self.audio_proj(audio_feat).unsqueeze(1) # [B, 1, H]
        x = torch.cat([t, a], dim=1)  # [B, 2, H]
        attn_out, _ = self.attn(x, x, x)
        pooled = attn_out.mean(dim=1)  # average over the 2 modalities
        return self.classifier(pooled)

# Instantiate model
model = EarlyFusionAttention(text_dim=text_train.shape[1], audio_dim=audio_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training loop ===
for epoch in range(18):
    model.train()
    total_loss = 0
    for tb, ab, yb in train_loader:
        optimizer.zero_grad()
        output = model(tb, ab)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for tb, ab, yb in val_loader:
            output = model(tb, ab)
            loss = criterion(output, yb)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# === Final Evaluation ===
model.eval()
y_pred = []
with torch.no_grad():
    for tb, ab, _ in test_loader:
        preds = model(tb, ab).argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)

# === Report and Confusion Matrix ===
print("Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Early Fusion Attention")
plt.tight_layout()
plt.savefig("confusion_matrix_attention.png")
print("Confusion matrix saved as confusion_matrix_attention.png")
