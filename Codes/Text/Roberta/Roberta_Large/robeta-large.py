import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_scheduler
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import re
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Hyperparameters
MODEL_NAME = "roberta-large"
MAX_LEN = 150
BATCH_SIZE = 16
EPOCHS = 50
LR = 8e-6
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_roberta2.pt"
CONF_MATRIX_PATH = "confusion_matrix_roberta_large.png"

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Emotion mapping
emotion_mapping = {
    "admiration": "positive", "amusement": "positive", "approval": "positive", "caring": "positive",
    "curiosity": "positive", "desire": "positive", "excitement": "positive", "gratitude": "positive",
    "joy": "positive", "love": "positive", "optimism": "positive", "pride": "positive",
    "realization": "positive", "relief": "positive", "surprise": "positive",
    "anger": "negative", "annoyance": "negative", "confusion": "negative", "disappointment": "negative",
    "disapproval": "negative", "disgust": "negative", "embarrassment": "negative", "fear": "negative",
    "grief": "negative", "nervousness": "negative", "remorse": "negative", "sadness": "negative",
    "neutral": "neutral"
}

def preprocess_dataset(split):
    data = load_dataset("go_emotions", split=split)
    label_names = data.features["labels"].feature.names
    texts, labels = [], []
    for example in data:
        text = re.sub(r"[^\x00-\x7F]+", "", example["text"])
        label_ids = example["labels"]
        if not label_ids:
            continue
        emotions = [label_names[i] for i in label_ids]
        mapped_labels = list(set(emotion_mapping[e] for e in emotions))
        texts.append(text)
        labels.append(mapped_labels)
    return texts, labels

train_texts, train_labels = preprocess_dataset("train")
val_texts, val_labels = preprocess_dataset("validation")
test_texts, test_labels = preprocess_dataset("test")

# Multi-label binarization
mlb = MultiLabelBinarizer()
train_labels_enc = mlb.fit_transform(train_labels)
val_labels_enc = mlb.transform(val_labels)
test_labels_enc = mlb.transform(test_labels)

class EmotionsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = EmotionsDataset(train_texts, train_labels_enc)
val_dataset = EmotionsDataset(val_texts, val_labels_enc)
test_dataset = EmotionsDataset(test_texts, test_labels_enc)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled_output))

model = RoBERTaClassifier(num_labels=len(mlb.classes_)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

best_val_loss = float("inf")
early_stop_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    
    accuracy = correct / total
    avg_val_loss = sum(criterion(model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)), batch["labels"].to(DEVICE)).item() for batch in val_loader) / len(val_loader)
    
    print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping activated.")
            break

print("Training completed.")
# Load best model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids, attention_mask)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Set Classification Report:")
print(classification_report(all_labels, all_preds, target_names=mlb.classes_))

cm = confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1))
ConfusionMatrixDisplay(cm, display_labels=mlb.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix - RoBERTa Emotion Classifier")
plt.savefig(CONF_MATRIX_PATH)
print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")
