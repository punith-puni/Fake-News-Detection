import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# -------------------------
# 1️⃣ Load Data
# -------------------------
data = pd.read_csv("train.csv")
data = data.sample(n=2000, random_state=42)  # use 2000 rows

# Ensure labels are integers (0 and 1)
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({"FAKE": 0, "REAL": 1})

texts = data['text'].tolist()
labels = data['label'].tolist()

# -------------------------
# 2️⃣ Tokenizer
# -------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 128

# -------------------------
# 3️⃣ Dataset
# -------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------------------------
# 4️⃣ Train/Validation Split
# -------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -------------------------
# 5️⃣ Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------------
# 6️⃣ Training Loop
# -------------------------
from tqdm import tqdm

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f}")

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Validation Accuracy: {val_acc:.4f}")

# -------------------------
# 7️⃣ Save Model
# -------------------------
model.save_pretrained("fake_news_model")
tokenizer.save_pretrained("fake_news_model")
print("Model and tokenizer saved successfully!")