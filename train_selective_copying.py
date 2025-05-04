import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm.auto import tqdm

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dataset functions
from datasets.selective_copying_data import (
    SelectiveCopyingDataset, create_datasets, get_dataloaders, 
    save_datasets, load_datasets
)
# Import the models
from models.mamba import MambaModel
from models.deltanet import DeltaNetModel

# ---------- Hyper‑parameters ----------
VOCAB_SIZE    = 16         # including noise token 0
BATCH_SIZE    = 8
EPOCH_SIZE    = 8192      # steps per epoch
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TOKENS_TO_COPY = 16   # Number of tokens to copy (as in the paper)
VAL_SET_SIZE  = 512        # number of sequences per validation loader
# -------------------------------------

MODEL_CONFIG = {
    'mamba':    {'epochs': 8, 'lr': 1e-4},
    'deltanet': {'epochs': 5, 'lr': 1e-4}
}

@torch.no_grad()
def evaluate(model, val_loaders):
    model.eval()
    results = {}
    # Create progress bar for sequence lengths
    for seq_len, loader in tqdm(val_loaders.items(), desc="Evaluating", leave=False):
        total_correct = 0
        total_tokens = 0
        
        # Create progress bar for batches within each sequence length
        for inputs, targets in tqdm(loader, desc=f"SeqLen {seq_len}", leave=False):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass through the model
            logits = model(inputs)
            
            # Get the predictions for the positions where marker tokens are
            # The marker tokens are at the end of the sequence
            marker_positions = torch.arange(inputs.size(1) - NUM_TOKENS_TO_COPY, inputs.size(1), device=DEVICE)
            
            # Use gather to properly extract the logits at marker positions
            # This handles batching correctly
            marker_logits = logits.gather(1, marker_positions.unsqueeze(0).unsqueeze(-1).expand(
                inputs.size(0), NUM_TOKENS_TO_COPY, VOCAB_SIZE))
            
            # Get the predicted tokens
            predictions = marker_logits.view(-1, VOCAB_SIZE).argmax(dim=-1)
            targets_flat = targets.view(-1)
            
            # Calculate accuracy
            correct = (predictions == targets_flat).sum().item()
            total_correct += correct
            total_tokens += targets_flat.numel()
        
        # Calculate metrics
        accuracy = total_correct / total_tokens
        results[seq_len] = {'accuracy': accuracy}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train selective copying task.")
    parser.add_argument('--model_type', choices=MODEL_CONFIG.keys(), required=True,
                        help="Model variant to train (affects epochs and LR)")
    parser.add_argument('--seq_len', type=int, default=1024,
                        help="Training sequence length")
    parser.add_argument('--use_saved_datasets', action='store_true',
                        help="Use previously saved datasets instead of creating new ones")
    args = parser.parse_args()

    # Get training config
    config = MODEL_CONFIG[args.model_type]
    epochs, lr = config['epochs'], config['lr']
    train_seq_len = args.seq_len

    # Initialize model and training components
    if args.model_type == 'deltanet':
        model = DeltaNetModel(vocab_size=VOCAB_SIZE).to(DEVICE).to(torch.bfloat16)
    elif args.model_type == "mamba":
        model = MambaModel(vocab_size=VOCAB_SIZE).to(DEVICE).to(torch.bfloat16)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    global_step = 0
    t0 = time.time()
    
    # Get training dataset and dataloader
    if args.use_saved_datasets:
        print("Loading saved datasets...")
        training_datasets, _ = load_datasets()
        if training_datasets is None:
            print("No saved datasets found. Creating new datasets...")
            training_datasets, _ = create_datasets(max_seq_length=train_seq_len*2)
            save_datasets(training_datasets, {})
    else:
        print("Creating new datasets...")
        training_datasets, _ = create_datasets(max_seq_length=train_seq_len*2)
        save_datasets(training_datasets, {})
    
    # Create dataloader for the specific sequence length
    if train_seq_len not in training_datasets:
        closest_length = min(training_datasets.keys(), key=lambda x: abs(x - train_seq_len))
        print(f"Warning: Sequence length {train_seq_len} not found in datasets. Using closest length: {closest_length}")
        train_seq_len = closest_length
    
    train_dataset = training_datasets[train_seq_len]
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", position=0)    
    for epoch in epoch_pbar:
        model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        step_pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
        for step, (inputs, targets) in step_pbar:
            global_step += 1
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use bfloat16 for forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                
                # We only care about the predictions at marker positions
                # The last NUM_TOKENS_TO_COPY positions contain the marker tokens
                marker_positions = torch.arange(inputs.size(1) - NUM_TOKENS_TO_COPY, inputs.size(1), device=DEVICE)
                marker_logits = logits.gather(1, marker_positions.unsqueeze(0).unsqueeze(-1).expand(
                    inputs.size(0), NUM_TOKENS_TO_COPY, VOCAB_SIZE))
                
                # Reshape for cross entropy loss
                marker_logits = marker_logits.view(-1, VOCAB_SIZE)
                targets_flat = targets.view(-1)
                
                # Calculate loss
                loss = criterion(marker_logits, targets_flat)
            
            loss.backward()

            # Calculate accuracy on marker positions
            predictions = marker_logits.argmax(dim=-1)
            correct = (predictions == targets_flat).sum().item()
            total = targets_flat.numel()
            
            epoch_correct += correct
            epoch_total += total
            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update progress bar
            step_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total*100:.2f}%"
            )
            
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_correct / (epoch_total + 1e-12) * 100
        epoch_pbar.set_postfix(
            loss=f"{avg_epoch_loss:.4f}",
            acc=f"{avg_epoch_acc:.2f}%",
            time=f"{(time.time() - t0) / 60:.1f}min"
        )
        print(f"\nEpoch {epoch}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_epoch_acc:.2f}%")

    print("\nTraining complete. Running validation...")

    # Create validation loaders for different sequence lengths using the dataset creation utilities
    print("Creating validation datasets for different sequence lengths...")
    _, validation_datasets = create_datasets(max_seq_length=2**15)  # Up to 32768
    
    # Create batch sizes dictionary that scales with sequence length
    batch_sizes = {}
    for length in validation_datasets.keys():
        # Scale batch size inversely with sequence length
        batch_sizes[length] = max(1, min(BATCH_SIZE, int(BATCH_SIZE * 1024 / length)))
    
    # Get validation dataloaders
    _, validation_loaders = get_dataloaders(
        training_datasets={},  # Empty as we don't need training loaders here
        validation_datasets=validation_datasets,
        batch_sizes=batch_sizes,
        num_workers=0  # Use 0 workers to avoid multiprocessing issues
    )
    
    # Evaluate on each sequence length
    metrics = evaluate(model, validation_loaders)
    elapsed = (time.time() - t0) / 60

    # Print metrics    
    print(f"\nFinal Results - Total time {elapsed:.1f} min")
    for L, m in metrics.items():
        print(f"  SeqLen={L:<7} Accuracy={m['accuracy']:.3f}")
    
    # Save the model
    save_path = f"selective_copying_{args.model_type}_{train_seq_len}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()