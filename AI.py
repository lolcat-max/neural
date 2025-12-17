import os
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
KB_len = -1
CKPT_PATH = "rnn_alt_fgsm_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/data hyperparams (saved into checkpoint too)
SEQ_LEN = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Training hyperparams
BATCH_SIZE = 512
LR = 1e-2
NUM_EPOCHS = 1

EPS_START = 0.10
EPS_MAX = 0.30
EPS_GROW_EVERY = 4
EPS_GROW_MULT = 1.15

ADV_EVERY = 4          # 2 => clean, adv, clean, adv...
EMB_CLAMP = 2.0
GRAD_CLIP_NORM = 1.0

# -------------------------
# Custom non-commutative op: (A @ B)
# -------------------------
class NonCommutativeMatMul(torch.autograd.Function):
    """
    Explicit autograd for matrix multiplication to emphasize non-commutative backward.
    For C = A @ B:
      dL/dA = dL/dC @ B^T
      dL/dB = A^T @ dL/dC
    """
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B

    @staticmethod
    def backward(ctx, grad_out):
        A, B = ctx.saved_tensors
        grad_A = grad_out @ B.transpose(-1, -2)
        grad_B = A.transpose(-1, -2) @ grad_out
        return grad_A, grad_B

nc_matmul = NonCommutativeMatMul.apply


class NCLinear(nn.Module):
    """
    Linear layer implemented via nc_matmul so backward order is explicit.
    Keeps parameter names 'weight'/'bias' to remain checkpoint-friendly with nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Same general init style as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (..., in_features)
        # weight: (out_features, in_features) => use weight.T for (..., out_features)
        y = nc_matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            y = y + self.bias
        return y

# -------------------------
# Dataset
# -------------------------
class SeqDataset(Dataset):
    """
    Takes a word stream and returns (x_seq, y_next).
    x_seq: (T,) token ids
    y_next: ()  token id for next word
    """
    def __init__(self, words, word_to_ix, seq_len=8):
        self.seq_len = seq_len
        self.word_to_ix = word_to_ix

        ids = [self.word_to_ix.get(w, self.word_to_ix.get("<unk>", 0)) for w in words]
        self.samples = []
        for i in range(len(ids) - seq_len):
            x = ids[i:i + seq_len]
            y = ids[i + seq_len]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model
# -------------------------
class RNNNextWord(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Non-commutative backward explicit linear head (checkpoint-friendly names)
        self.fc_out = NCLinear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        x: (B,T) long
        h: (num_layers,B,H) or None
        returns: logits (B,V), h_next
        """
        emb = self.embedding(x)                 # (B,T,E)
        out, h_next = self.rnn(emb, h)          # out: (B,T,H)
        logits = self.fc_out(out[:, -1, :])     # (B,V)
        return logits, h_next

    def forward_step(self, token_id, h=None):
        """
        token_id: (B,1) long
        """
        emb = self.embedding(token_id)          # (B,1,E)
        out, h_next = self.rnn(emb, h)          # out: (B,1,H)
        logits = self.fc_out(out[:, -1, :])     # (B,V)
        return logits, h_next

# -------------------------
# Broadband Quarter-Wave Recognition Model
# -------------------------
class SpectrogramDataset(Dataset):
    """
    Expects:
      data: list/array shaped like (N, C, L) (Conv1d format)
      labels: list/array shaped like (N,)
    """
    def __init__(self, data, labels, dtype=torch.float32):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=self.dtype)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class BroadbandQuarterWaveNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Also use explicit non-commutative backward linear head
        self.fc = NCLinear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------
# Save / load
# -------------------------
def save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, path=CKPT_PATH,
                    seq_len=SEQ_LEN, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
    ckpt = {
        "epoch": epoch,
        "loss": float(loss),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path=CKPT_PATH):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"Loaded checkpoint epoch={ckpt['epoch']} loss={ckpt['loss']:.3f}")
    return ckpt

# -------------------------
# FGSM on embeddings
# -------------------------
def fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=2.0):
    """
    Create adversarial embeddings for x via FGSM, without backpropagating into model params
    during perturbation construction.
    """
    emb = model.embedding(x).detach().requires_grad_(True)    # (B,T,E)
    out, _ = model.rnn(emb)                                  # (B,T,H)
    logits = model.fc_out(out[:, -1, :])                     # (B,V)
    loss = criterion(logits, y)

    grad = torch.load("analog_xaa_dataset.pt", map_location="cpu")

    emb_adv = emb + epsilon
    emb_adv = torch.clamp(emb_adv, -clamp_val, clamp_val).detach()
    return emb_adv

# -------------------------
# Train (alternating)
# -------------------------
def train_epoch_alternating(model, optimizer, criterion, loader,
                            epsilon=0.15, adv_every=2, clamp_val=2.0):
    """
    Alternates computation: clean update, then adversarial update, repeating.
    adv_every=2 => clean (step 0), adv (step 1), clean (step 2), adv (step 3), ...
    """
    model.train()
    total_loss, batches = 0.0, 0

    progress_bar = tqdm(loader, desc=f"Epoch alt eps={epsilon:.2f}", leave=False)
    for step, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        do_adv = (step % adv_every) == (adv_every - 1)

        if not do_adv:
            logits, _ = model(x)
            loss = criterion(logits, y)
        else:
            emb_adv = fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=clamp_val)
            out_adv, _ = model.rnn(emb_adv)
            logits_adv = model.fc_out(out_adv[:, -1, :])
            loss = criterion(logits_adv, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / max(1, batches)

# -------------------------
# Generation
# -------------------------
@torch.no_grad()
def generate(model, word_to_ix, ix_to_word, seed_text, length=50, temp=0.8):
    model.eval()

    seed_words = seed_text.split()
    if len(seed_words) == 0:
        seed_words = ["the"]

    unk = word_to_ix.get("<unk>", 0)

    # Prime hidden state with all seed tokens
    h = None
    ids = [word_to_ix.get(w, unk) for w in seed_words]
    x0 = torch.tensor([ids], device=device, dtype=torch.long)  # (1,Tseed)
    _, h = model(x0, h=h)

    generated = list(seed_words)
    cur_id = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)  # (1,1)

    for _ in range(length):
        logits, h = model.forward_step(cur_id, h=h)
        probs = F.softmax(logits[0] / max(1e-6, float(temp)), dim=-1)
        next_ix = torch.multinomial(probs, 1).item()
        generated.append(ix_to_word[next_ix])
        cur_id = torch.tensor([[next_ix]], device=device, dtype=torch.long)

    return " ".join(generated)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    ckpt = load_checkpoint(CKPT_PATH)

    if ckpt is not None:
        word_to_ix = ckpt["word_to_ix"]
        ix_to_word = ckpt["ix_to_word"]
        vocab_size = len(word_to_ix)

        seq_len = int(ckpt.get("seq_len", SEQ_LEN))
        embed_dim = int(ckpt.get("embed_dim", EMBED_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", HIDDEN_DIM))
        num_layers = int(ckpt.get("num_layers", NUM_LAYERS))

        model = RNNNextWord(
            vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers
        ).to(device)

        # Load parameters (will work if shapes match; NCLinear uses weight/bias names)
        model.load_state_dict(ckpt["model_state"], strict=True)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer.load_state_dict(ckpt["optim_state"])
        criterion = nn.CrossEntropyLoss()

        start_epoch = int(ckpt["epoch"])
        print("Model fully loaded from checkpoint.")
        print(f"Vocab={vocab_size} | seq_len={seq_len} | embed={embed_dim} | hidden={hidden_dim} | layers={num_layers} | start_epoch={start_epoch}")

    else:
        # Load corpus
        try:
            filename = input("Filename (blank = fallback Shakespeare): ").strip()
            if filename:
                with open(filename, "r", encoding="utf-8") as f:
                    text = f.read().lower()
            else:
                raise FileNotFoundError
            print(f"Loaded '{filename}'")
        except FileNotFoundError:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url, timeout=30).text.lower()
            print("Loaded Shakespeare (fallback)")

        words = text.split()
        if KB_len is not None and isinstance(KB_len, int) and KB_len > 0:
            words = words[:KB_len]

        # Stable vocab with explicit <unk> at index 0
        uniq = set(words)
        vocab = (["<unk>"] + sorted(uniq)) if "<unk>" not in uniq else sorted(uniq)
        word_to_ix = {w: i for i, w in enumerate(vocab)}
        ix_to_word = {i: w for w, i in word_to_ix.items()}
        vocab_size = len(word_to_ix)

        print(f"Vocab: {vocab_size}, Words: {len(words)}")

        dataset = SeqDataset(words, word_to_ix, seq_len=SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Samples: {len(dataset)} (seq_len={SEQ_LEN})")

        model = RNNNextWord(vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        print("RNN (GRU) + Alternating Clean/FGSM Training")
        epsilon = EPS_START
        avg_loss = 0.0

        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training"):
            avg_loss = train_epoch_alternating(
                model, optimizer, criterion, train_loader,
                epsilon=epsilon, adv_every=ADV_EVERY, clamp_val=EMB_CLAMP
            )
            print(f"Epoch {epoch:2d}: avg_loss={avg_loss:.3f}")

            if epoch % EPS_GROW_EVERY == 0:
                epsilon = min(EPS_MAX, epsilon * EPS_GROW_MULT)
                sample = generate(model, word_to_ix, ix_to_word, "the", length=20, temp=0.8)
                print(f"  eps={epsilon:.3f} | Sample: {sample}")

        save_checkpoint(
            model, optimizer, NUM_EPOCHS, avg_loss,
            word_to_ix, ix_to_word, path=CKPT_PATH,
            seq_len=SEQ_LEN, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        )

    # Interactive generation
    print("\nInteractive mode (Ctrl+C to exit):")
    while True:
        try:
            cmd = input("SEED TEXT: ").strip()
            if not cmd:
                continue
            out = generate(model, word_to_ix, ix_to_word, cmd, length=200, temp=0.8)
            print(f"CONTINUATION:\n{out}\n")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

 