import torch
import os
from induction_head_data import create_datasets, get_dataloaders
import torch.multiprocessing as mp

def main():
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Create datasets up to sequence length 2^12 = 4096 (reduced from 2^16 for faster execution)
    print("Creating datasets...")
    training_datasets, validation_datasets = create_datasets(max_seq_length=4096)

    # Print information about the datasets
    print("\n--- Dataset Information ---")
    print(f"Available sequence lengths: {sorted(list(training_datasets.keys()))}")

    # Print details for each sequence length
    for seq_len in sorted(list(training_datasets.keys())):
        train_dataset = training_datasets[seq_len]
        val_dataset = validation_datasets[seq_len]
        
        print(f"\nSequence Length: {seq_len}")
        print(f"  Training dataset size: {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")
        
        # Sample data from each dataset
        if seq_len <= 256:  # Only show samples for smaller sequences to avoid cluttering output
            # Get a sample from training set
            train_sample_x, train_sample_y = train_dataset[0]
            print(f"  Sample training sequence shape: {train_sample_x.shape}")
            print(f"  Sample training sequence: {train_sample_x[:10].tolist()} ... (first 10 tokens)")
            
            # Find positions where labels are not -1 (expected predictions)
            prediction_positions = [i for i, label in enumerate(train_sample_y.tolist()) if label != -1]
            if prediction_positions:
                print(f"  Training sample prediction positions: {prediction_positions}")
                for pos in prediction_positions[:3]:  # Limit to 3 positions to avoid clutter
                    print(f"    Position {pos}: Expected token {train_sample_y[pos]}")
            
            # Get a sample from validation set
            val_sample_x, val_sample_y = val_dataset[0]
            print(f"  Sample validation sequence shape: {val_sample_x.shape}")
            
            # Find positions where labels are not -1 (expected predictions)
            prediction_positions = [i for i, label in enumerate(val_sample_y.tolist()) if label != -1]
            if prediction_positions:
                print(f"  Validation sample prediction positions: {prediction_positions}")
                for pos in prediction_positions[:3]:  # Limit to 3 positions to avoid clutter
                    print(f"    Position {pos}: Expected token {val_sample_y[pos]}")

    # Save the datasets
    print("\nSaving datasets...")
    torch.save(training_datasets, 'data/training_datasets.pt')
    torch.save(validation_datasets, 'data/validation_datasets.pt')

    print("Datasets saved successfully!")

    # You can also create and save dataloaders if needed
    print("\nCreating dataloaders...")
    
    # Create custom batch sizes dictionary with single worker for all dataloaders
    batch_sizes = {}
    for length in training_datasets.keys():
        # Scale batch size inversely with sequence length, base is 8 at length 256
        batch_sizes[length] = max(1, min(8, int(8 * 256 / length)))
    
    # Force num_workers=0 to avoid multiprocessing issues
    training_loaders, validation_loaders = get_dataloaders(
        training_datasets, 
        validation_datasets,
        batch_sizes=batch_sizes
    )

    # Print batch information from a dataloader
    print("\n--- Dataloader Information ---")
    train_loader_256 = training_loaders[256]
    
    # Safely get the first batch
    batch_x, batch_y = next(iter(train_loader_256))
    print(f"Sample batch from train_loader_256: {batch_x.shape}")
    print(f"Batch size: {batch_x.shape[0]}")
    print(f"Sequence length: {batch_x.shape[1]}")

    # Count non-masked positions in the labels
    non_masked = (batch_y != -1).sum().item()
    print(f"Number of prediction positions in batch: {non_masked}")
    print(f"Average predictions per sequence: {non_masked / batch_x.shape[0]:.2f}")

    print("\nDataloaders created. Ready for training.")

if __name__ == "__main__":
    # Add multiprocessing support for Windows
    mp.freeze_support()
    main() 