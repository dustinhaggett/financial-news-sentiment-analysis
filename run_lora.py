"""
Standalone LoRA training script — binary classification for stock direction.
Run: python3 run_lora.py
"""
import os, pickle, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BertModel, BertTokenizer

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# Load data
print("Loading data...")
news_df = pd.read_csv('news_w_flang_sentiment.csv')
news_df['publication_datetime'] = pd.to_datetime(news_df['publication_datetime'])
price_df = pd.read_csv('price.csv')
price_df['Date'] = pd.to_datetime(price_df['Date'])

# Feature engineering
price_df['log_return'] = price_df.groupby('ticker')['close'].transform(lambda x: np.log(x / x.shift(1)))
price_df['lag1_return'] = price_df.groupby('ticker')['log_return'].shift(1)
price_df['lag5_return'] = price_df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(5).sum().shift(1))
price_df['vol_20d'] = price_df.groupby('ticker')['log_return'].transform(lambda x: x.rolling(20).std().shift(1))

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100

price_df['rsi_14'] = price_df.groupby('ticker')['close'].transform(lambda x: compute_rsi(x, 14)).shift(1)

def compute_macd_signal(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    return (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

price_df['macd_hist'] = price_df.groupby('ticker').apply(
    lambda g: compute_macd_signal(g['close']) / g['close']
).reset_index(level=0, drop=True).shift(1)
price_df = price_df.dropna()

NUM_FEATURES = 5
return_map = price_df.set_index(['Date', 'ticker'])['log_return'].to_dict()
lag1_map = price_df.set_index(['Date', 'ticker'])['lag1_return'].to_dict()
lag5_map = price_df.set_index(['Date', 'ticker'])['lag5_return'].to_dict()
vol_map = price_df.set_index(['Date', 'ticker'])['vol_20d'].to_dict()
rsi_map = price_df.set_index(['Date', 'ticker'])['rsi_14'].to_dict()
macd_map = price_df.set_index(['Date', 'ticker'])['macd_hist'].to_dict()

# Build datasets
def build_dataset(news_subset):
    texts, labels, num_feats, valid_idx = [], [], [], []
    for idx in range(len(news_subset)):
        row = news_subset.iloc[idx]
        date, ticker = row['publication_datetime'], row['tickers']
        next_days = price_df[(price_df['Date'] > date) & (price_df['ticker'] == ticker)]['Date'].unique()
        if len(next_days) == 0:
            continue
        next_date = min(next_days)
        vals = [return_map.get((next_date, ticker)),
                lag1_map.get((next_date, ticker)),
                lag5_map.get((next_date, ticker)),
                vol_map.get((next_date, ticker)),
                rsi_map.get((next_date, ticker)),
                macd_map.get((next_date, ticker))]
        if any(v is None for v in vals):
            continue
        ret = vals[0]
        texts.append(row['body'] if pd.notna(row['body']) else '')
        labels.append(1 if ret > 0 else 0)
        num_feats.append(vals[1:])
        valid_idx.append(idx)
    return texts, np.array(labels), np.array(num_feats), valid_idx

print("Building datasets...")
train_news = news_df[news_df['publication_datetime'] < '2019-01-01'].reset_index(drop=True)
val_news = news_df[(news_df['publication_datetime'] >= '2019-01-01') &
                   (news_df['publication_datetime'] < '2020-01-01')].reset_index(drop=True)
test_news = news_df[news_df['publication_datetime'] >= '2020-01-01'].reset_index(drop=True)

tr_texts, tr_labels, tr_num, _ = build_dataset(train_news)
vl_texts, vl_labels, vl_num, _ = build_dataset(val_news)
te_texts, te_labels, te_num, te_idx = build_dataset(test_news)

tr_pos_rate = tr_labels.mean()
print(f"Train: {len(tr_texts)} ({tr_pos_rate:.1%} up), Val: {len(vl_texts)}, Test: {len(te_texts)}")

# Normalize numerical features
num_mean, num_std = tr_num.mean(axis=0), tr_num.std(axis=0)
tr_num_z = (tr_num - num_mean) / num_std
vl_num_z = (vl_num - num_mean) / num_std
te_num_z = (te_num - num_mean) / num_std

# Model
print("Loading FinBERT + LoRA...")
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_encoder = BertModel.from_pretrained('yiyanghkust/finbert-tone')

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "value"],
)
finbert_lora = get_peft_model(finbert_encoder, lora_config)
trainable = sum(p.numel() for p in finbert_lora.parameters() if p.requires_grad)
total = sum(p.numel() for p in finbert_lora.parameters())
print(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

class FinBERTLoRAClassifier(nn.Module):
    def __init__(self, encoder, num_features_dim=5, hidden_size=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(768 + num_features_dim, hidden_size),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 2),
        )
    def forward(self, input_ids, attention_mask, num_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        mean_emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        x = torch.cat([mean_emb, num_features], dim=-1)
        return self.head(x)

model = FinBERTLoRAClassifier(finbert_lora).to(device)

# Training setup
class_weights = torch.FloatTensor([tr_pos_rate, 1 - tr_pos_rate]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

lora_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if 'head' in n]
optimizer = torch.optim.AdamW([
    {'params': lora_params, 'lr': 2e-5, 'weight_decay': 0.01},
    {'params': head_params, 'lr': 5e-4, 'weight_decay': 1e-4},
])

BATCH_SIZE = 64
EPOCHS = 10
total_steps = (len(tr_texts) // BATCH_SIZE) * EPOCHS
warmup_steps = total_steps // 10

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Train
print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE}...")
train_losses, val_losses, val_accs = [], [], []
best_val_acc = 0
best_state = None
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, n_correct, n_total = 0, 0, 0
    indices = np.random.permutation(len(tr_texts))

    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        batch_texts = [tr_texts[j] for j in batch_idx]
        batch_num = torch.FloatTensor(tr_num_z[batch_idx]).to(device)
        batch_labels = torch.LongTensor(tr_labels[batch_idx]).to(device)

        tokens = finbert_tokenizer(batch_texts, return_tensors='pt', max_length=512,
                                   padding='max_length', truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        optimizer.zero_grad()
        logits = model(tokens['input_ids'], tokens['attention_mask'], batch_num)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        n_total += len(batch_labels)

    train_acc = n_correct / n_total
    train_losses.append(epoch_loss / (len(indices) // BATCH_SIZE))

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(vl_texts), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(vl_texts))
            batch_texts = vl_texts[i:end]
            batch_num = torch.FloatTensor(vl_num_z[i:end]).to(device)
            batch_labels = torch.LongTensor(vl_labels[i:end]).to(device)
            tokens = finbert_tokenizer(batch_texts, return_tensors='pt', max_length=512,
                                       padding='max_length', truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(tokens['input_ids'], tokens['attention_mask'], batch_num)
            val_loss += criterion(logits, batch_labels).item()
            val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            val_total += len(batch_labels)

    val_acc = val_correct / val_total
    val_losses.append(val_loss / (len(vl_texts) // BATCH_SIZE))
    val_accs.append(val_acc)

    marker = ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = ' *best*'

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_losses[-1]:.4f}, "
          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}{marker}  [{elapsed:.0f}s]")

# Test
print(f"\nEvaluating best model (val_acc={best_val_acc:.4f})...")
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

test_probs = []
with torch.no_grad():
    for i in range(0, len(te_texts), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(te_texts))
        batch_texts = te_texts[i:end]
        batch_num = torch.FloatTensor(te_num_z[i:end]).to(device)
        tokens = finbert_tokenizer(batch_texts, return_tensors='pt', max_length=512,
                                   padding='max_length', truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logits = model(tokens['input_ids'], tokens['attention_mask'], batch_num)
        probs = torch.softmax(logits, dim=1)[:, 1]
        test_probs.extend(probs.cpu().numpy())

test_probs = np.array(test_probs)
pred_bin = (test_probs > 0.5).astype(int)
acc = accuracy_score(te_labels, pred_bin)
f1 = f1_score(te_labels, pred_bin)

print(f"\n{'='*50}")
print(f"LoRA FinBERT Binary Classification — TEST RESULTS")
print(f"{'='*50}")
print(f"Accuracy:    {acc:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"% positive:  {pred_bin.mean():.4f}")
print(f"Best val acc: {best_val_acc:.4f}")
print()
print(classification_report(te_labels, pred_bin, target_names=['Down', 'Up']))

# Save cache for notebook
with open('lora_results_cache.pkl', 'wb') as f:
    pickle.dump({
        'test_probs': test_probs,
        'test_labels': te_labels,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_idx': te_idx,
    }, f)
print("Results cached to lora_results_cache.pkl")
