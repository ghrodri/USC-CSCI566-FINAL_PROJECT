import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_scheduler
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
MODEL_NAME = "roberta-base"
MAX_LEN = 60
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    texts, main_labels = [], []
    for example in data:
        text = re.sub(r"[^\x00-\x7F]+", "", example["text"])
        label_ids = example["labels"]
        if not label_ids:
            continue
        emotions = [label_names[i] for i in label_ids]
        mapped = [emotion_mapping[e] for e in emotions]
        main = "negative" if "negative" in mapped else "positive" if "positive" in mapped else "neutral"
        texts.append(text)
        main_labels.append(main)
    return texts, main_labels

train_texts, train_labels = preprocess_dataset("train")
val_texts, val_labels = preprocess_dataset("validation")
test_texts, test_labels = preprocess_dataset("test")

label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_labels)
val_labels_enc = label_encoder.transform(val_labels)
test_labels_enc = label_encoder.transform(test_labels)

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
            "labels": torch.tensor(self.labels[idx])
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

model = RoBERTaClassifier(num_labels=3).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
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

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            val_loss += criterion(outputs, labels).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model_roberta.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping activated.")
            break

print("Training completed.")

# Evaluation
model.load_state_dict(torch.load("best_model_roberta.pt"))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Set Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix - RoBERTa Emotion Classifier")
plt.savefig("confusion_matrix_roberta.png")
