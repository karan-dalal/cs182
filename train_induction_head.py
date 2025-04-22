import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import InductionHeadsDataset
from model import InductionHeadsModel

# ---------- Hyper‑parameters ----------
VOCAB_SIZE    = 16         # including noise token 0
BATCH_SIZE    = 8
EPOCH_SIZE    = 8192      # steps per epoch
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX  = -100       # for masked CE loss
VAL_SET_SIZE  = 512        # number of sequences per validation loader
# -------------------------------------

MODEL_CONFIG = {
    'mha_abs':  {'epochs': 25, 'lr': 2e-4},
    'mamba':    {'epochs': 25, 'lr': 1e-3},
    'mha_rope': {'epochs': 50, 'lr': 2e-4},
    'mha_xpos': {'epochs': 50, 'lr': 2e-4},
    'h3':       {'epochs': 10, 'lr': 2e-4},
    'hyena':    {'epochs': 10, 'lr': 2e-4},
}

@torch.no_grad()
def evaluate(model, val_loaders):
    model.eval()
    results = {}
    for seq_len, loader in val_loaders.items():
        tp = fp = fn = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred   = logits.argmax(dim=-1)

            mask_true = y != IGNORE_INDEX
            mask_pred = pred != 0

            tp += ((pred == y) & mask_true).sum().item()
            fp += (mask_pred & ~mask_true).sum().item()
            fn += ((pred == 0) & mask_true).sum().item()

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        results[seq_len] = {'P': precision, 'R': recall, 'F1': f1}
    return results


def make_loader(size, seq_len, shuffle=True):
    # Dataset generates data randomly each step
    dataset = InductionHeadsDataset(size=size, seq_len=seq_len)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description="Train induction heads task.")
    parser.add_argument('--model_type', choices=MODEL_CONFIG.keys(), required=True,
                        help="Model variant to train (affects epochs and LR)")
    args = parser.parse_args()

    config = MODEL_CONFIG[args.model_type]
    epochs = config['epochs']
    lr     = config['lr']

    # Initialize model
    model = InductionHeadsModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Prepare validation loaders for sequence lengths 2^6..2^20
    val_seq_lens = [2**i for i in range(6, 21)]
    val_loaders = {L: make_loader(VAL_SET_SIZE, seq_len=L, shuffle=False)
                   for L in val_seq_lens}

    # Training loop
    total_steps = epochs * EPOCH_SIZE
    global_step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loader = make_loader(EPOCH_SIZE, seq_len=SEQ_LEN, shuffle=True)
        for step, (x, y) in enumerate(train_loader, 1):
            global_step += 1
            model.train()
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate at end of each epoch
        metrics = evaluate(model, val_loaders)
        elapsed = (time.time() - t0) / 60
        print(f"Epoch {epoch}/{epochs} - step {global_step}/{total_steps} - time {elapsed:.1f} min")
        for L, m in metrics.items():
            print(f"  SeqLen={L:<7} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}")

    # Save checkpoint
    ckpt = f"induction_heads_{args.model_type}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Training complete. Model saved to {ckpt}")


if __name__ == "__main__":
    torch.manual_seed(42)
    # default training sequence length
    SEQ_LEN = 4096
    main()
