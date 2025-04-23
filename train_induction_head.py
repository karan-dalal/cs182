import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import csv
from pathlib import Path
from tqdm.auto import tqdm
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs182.datasets.induction_head_data import InductionHeadsDataset
from cs182.models.mamba import MambaModel
from cs182.models.deltanet import DeltaNetModel

# ---------- Hyper‑parameters ----------
VOCAB_SIZE    = 16         # including noise token 0
BATCH_SIZE    = 8
EPOCH_SIZE    = 8192      # steps per epoch
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX  = -1       # for masked CE loss
TRAIN_SEQ_LEN = 256       # fixed training sequence length
VAL_SET_SIZE  = 512        # number of sequences per validation loader
# -------------------------------------

MODEL_CONFIG = {
    'mamba':    {'epochs': 8, 'lr': 1e-3},
    'deltanet': {'epochs': 5, 'lr': 1e-3}
}

@torch.no_grad()
def evaluate(model, val_loaders):
    model.eval()
    results = {}
    # Create progress bar for sequence lengths
    for seq_len, loader in tqdm(val_loaders.items(), desc="Evaluating", leave=False):
        tp = fp = fn = 0
        # Create progress bar for batches within each sequence length
        for x, y in tqdm(loader, desc=f"SeqLen {seq_len}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            
            # Only consider positions where we expect specific predictions
            mask_true = y != IGNORE_INDEX
            
            # True positives: correct predictions at positions we care about
            tp += ((pred == y) & mask_true).sum().item()
            
            # False positives: wrong predictions at positions we care about
            fp += ((pred != y) & mask_true).sum().item()
            
            # False negatives: we don't need this for this task since
            # every position we care about will either be a true positive
            # or false positive
            fn = fp  # Every wrong prediction is a false negative

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        results[seq_len] = {'P': precision, 'R': recall, 'F1': f1}
    return results


def make_loader(size, seq_len, shuffle=True, split="train"):
    """
    Utility to build a DataLoader for the induction-heads task.
    """
    dataset = InductionHeadsDataset(
        split=split,
        size=size,
        seq_length=seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train induction heads task.")
    parser.add_argument('--model_type', choices=MODEL_CONFIG.keys(), required=True,
                        help="Model variant to train (affects epochs and LR)")
    args = parser.parse_args()

    # Get training config
    config = MODEL_CONFIG[args.model_type]
    epochs, lr = config['epochs'], config['lr']

    # Initialize model and training components
    if args.model_type == 'deltanet':
        model = DeltaNetModel(vocab_size=VOCAB_SIZE).to(DEVICE).to(torch.bfloat16)
    elif args.model_type == "mamba":
        model = MambaModel(vocab_size=VOCAB_SIZE).to(DEVICE).to(torch.bfloat16)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Training loop
    global_step = 0
    t0 = time.time()
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", position=0)    
    for epoch in epoch_pbar:
        train_loader = make_loader(EPOCH_SIZE, seq_len=TRAIN_SEQ_LEN, shuffle=True, split="train")
        model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for step, (x, y) in enumerate(train_loader, 1):
            global_step += 1
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            
            # Use bfloat16 for forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            loss.backward()

            # Calculate accuracy on non-ignored positions
            pred = logits.argmax(dim=-1)
            mask = y != IGNORE_INDEX
            correct = ((pred == y) & mask).sum().item()
            total = mask.sum().item()
            
            epoch_correct += correct
            epoch_total += total
            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 100 == 0:  # Log every 100 steps
                avg_loss = epoch_loss / step
                avg_acc = epoch_correct / (epoch_total + 1e-12) * 100
                epoch_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    acc=f"{avg_acc:.2f}%",
                    time=f"{(time.time() - t0) / 60:.1f}min"
                )

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_correct / (epoch_total + 1e-12) * 100
        print(f"\nEpoch {epoch}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_epoch_acc:.2f}%")

    print("\nTraining complete. Running validation...")

    # Create validation loaders for sequence lengths 2^6 (64) to 2^20
    val_seq_lens = [2**i for i in range(6, 16)]
    val_loaders = {L: make_loader(VAL_SET_SIZE, seq_len=L, shuffle=False, split="val")
                for L in val_seq_lens}
    
    metrics = evaluate(model, val_loaders)
    elapsed = (time.time() - t0) / 60

    # Print metrics    
    print(f"\nFinal Results - Total time {elapsed:.1f} min")
    for L, m in metrics.items():
        print(f"  SeqLen={L:<7} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
