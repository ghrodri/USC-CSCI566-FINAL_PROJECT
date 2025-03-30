import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_scheduler, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# General Configuration
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 10  # Reduce epochs, LoRA es eficiente
LR = 2e-5
PATIENCE = 5
MAX_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Emotion Mapping to General Categories
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

# Load and Preprocess Dataset
def preprocess_dataset(split):
    data = load_dataset("go_emotions", split=split)
    label_names = data.features["labels"].feature.names
    
    def process(example):
        text = re.sub(r"[^\x00-\x7F]+", "", example["text"])
        label_ids = example["labels"]
        if not label_ids:
            return None
        emotions = [label_names[i] for i in label_ids]
        mapped = [emotion_mapping[e] for e in emotions]
        main = "negative" if "negative" in mapped else "positive" if "positive" in mapped else "neutral"
        return {"text": text, "label": main}
    
    processed_data = data.map(process, remove_columns=data.column_names)
    return processed_data.filter(lambda x: x is not None)

print("Loading GoEmotions dataset")
train_data = preprocess_dataset("train")
val_data = preprocess_dataset("validation")
test_data = preprocess_dataset("test")

# Encode Labels
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_data["label"])
val_labels_enc = label_encoder.transform(val_data["label"])
test_labels_enc = label_encoder.transform(test_data["label"])

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create Datasets and Dataloaders
train_dataset = GoEmotionsDataset(train_data["text"], train_labels_enc)
val_dataset = GoEmotionsDataset(val_data["text"], val_labels_enc)
test_dataset = GoEmotionsDataset(test_data["text"], test_labels_enc)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load Base Model and Apply LoRA
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Apply only to attention layers
)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.to(DEVICE)

# Training Loop
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(train_loader))

best_val_loss = float("inf")
early_stopping_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "bert_lora_best_model.pt")
        print("Best model saved.")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Load Best Model and Evaluate
model.load_state_dict(torch.load("bert_lora_best_model.pt"))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask).logits
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report and Confusion Matrix
print("\nClassification Report on Test Set:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_).plot(cmap="Blues")
plt.title("Confusion Matrix: Positive vs Negative vs Neutral")
plt.tight_layout()
plt.savefig("confusion_matrix_bert_lora.png")
print("Confusion Matrix saved.")
