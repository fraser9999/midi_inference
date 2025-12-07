import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import glob
import argparse
from tqdm import tqdm

# -------------------------------
# Dataset
# -------------------------------
class MidiDataset(Dataset):
    def __init__(self, token_folder, seq_len=256):
        self.seq_len = seq_len
        self.files = glob.glob(os.path.join(token_folder, "*.pt"))
        self.data = []
        for f in self.files:
            tokens = torch.load(f)
            if len(tokens) >= seq_len:
                self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        start = torch.randint(0, len(tokens)-self.seq_len, (1,)).item()
        seq = tokens[start:start+self.seq_len]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

# -------------------------------
# Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds PE max_len {self.pe.size(0)}")
        return x + self.pe[:seq_len, :].unsqueeze(0)

# -------------------------------
# GPTLike
# -------------------------------
class GPTLike(nn.Module):
    def __init__(self, vocab_size=273, d_model=512, n_layers=6, n_heads=8, dropout=0.1, max_len=10000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=4*d_model, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        emb = self.token_emb(x)
        emb = self.pos_enc(emb)
        emb = self.transformer(emb.transpose(0,1)).transpose(0,1)
        emb = self.ln_f(emb)
        logits = self.head(emb)
        return logits

# -------------------------------
# Training Loop
# -------------------------------
def train(args):
    device = args.device
    train_dataset = MidiDataset(args.train_folder, seq_len=args.seq_len)
    val_dataset = MidiDataset(args.val_folder, seq_len=args.seq_len) if args.val_folder else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset else None

    model = GPTLike(vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads, dropout=args.dropout, max_len=args.max_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss/len(val_loader):.4f}")

        # Checkpoint speichern
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({'model_state': model.state_dict()}, checkpoint_path)
        print(f"Checkpoint gespeichert: {checkpoint_path}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, required=True)
    parser.add_argument("--val_folder", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=273)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
