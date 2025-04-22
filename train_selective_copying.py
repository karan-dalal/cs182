import math, random, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Hyper‑parameters ----------
SEQ_LEN       = 4096
VOCAB_SIZE    = 16         # token 0 == noise
N_TARGETS     = 16
D_MODEL       = 64
N_LAYERS      = 2
N_HEADS       = 8
BATCH_SIZE    = 64
LR            = 1e-4
TOTAL_STEPS   = 400_000
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX  = -100       # for masked CE loss
# -------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred   = logits.argmax(dim=-1)           # (B, L)

        mask_true = y != IGNORE_INDEX            # where a target exists
        mask_pred = pred != 0                    # we treat non‑noise as “copied”

        tp += ((pred == y) & mask_true).sum().item()
        fp += ((pred != 0) & ~mask_true).sum().item()
        fn += ((pred == 0) &  mask_true).sum().item()

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

class SelectiveCopyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Randomly select a target
        target = random.randint(1, N_TARGETS)
        # Randomly select a sequence of tokens
        x = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        # Create the target sequence with noise
        y = torch.full((SEQ_LEN,), 0)  # fill with noise
        y[:SEQ_LEN // 2] = target      # copy the target to the first half
        return x, y
    
def make_loader(size, shuffle=True):
    dataset = SelectiveCopyDataset(size)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train():
    model = MambaModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    train_loader = make_loader(TOTAL_STEPS, shuffle=True)
    val_loader   = make_loader(512, shuffle=False)   # quick dev set

    t0 = time.time()
    for step, (x, y) in enumerate(train_loader, 1):
        model.train()
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)                    # (B, L, V)
        loss   = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- Logging / Eval every 5 000 steps ----
        if step % 5_000 == 0 or step == 1:
            p, r, f1 = evaluate(model, val_loader)
            elapsed  = (time.time() - t0) / 60
            print(f"[{step:>6}/{TOTAL_STEPS}] "
                  f"loss={loss.item():.4f}  "
                  f"P={p:.3f} R={r:.3f} F1={f1:.3f}  "
                  f"time={elapsed:.1f} min")

        if step >= TOTAL_STEPS:
            break

    # -------- Save final checkpoint ----------
    torch.save(model.state_dict(), "selective_copy.pt")
    print("Finished! Model saved to selective_copy.pt")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    torch.manual_seed(42)
    train()
