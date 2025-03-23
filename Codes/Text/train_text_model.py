# Train a text emotion classifier using BERT and the GoEmotions dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

class BERTEmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.out(self.dropout(pooled_output))

# Load dataset
df = pd.read_csv("data/text_data/goemotions.csv")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'].apply(eval))
X_train, X_val, y_train, y_val = train_test_split(df['text'], y, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = GoEmotionsDataset(X_train.tolist(), y_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BERTEmotionClassifier(num_labels=y.shape[1]).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Train
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "checkpoints/text_model.pt")

