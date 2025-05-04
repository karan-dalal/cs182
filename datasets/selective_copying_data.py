import torch
import random
import math
import os
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

class SelectiveCopyingDataset(Dataset):
    """
    Dataset for the selective copying task from the Mamba paper.
    Generates sequences with tokens to memorize scattered randomly throughout
    a noise sequence, followed by marker tokens that should trigger recall.
    """
    def __init__(self,
                 split='train',
                 seq_length=4096,
                 num_tokens_to_copy=16,
                 vocab_size=16,
                 size=None,
                 fixed_seed=None):
        """
        Initialize the dataset.

        Args:
            split: 'train' or 'val' to indicate dataset type
            seq_length: Length of the noise/padding sequence
            num_tokens_to_copy: Number of tokens to memorize and copy
            vocab_size: Size of the vocabulary
            size: Number of examples in the dataset
            fixed_seed: Seed for reproducibility (used for validation)
        """
        self.seq_length = seq_length
        self.num_tokens_to_copy = num_tokens_to_copy
        self.vocab_size = vocab_size
        self.split = split
        self.fixed_seed = fixed_seed

        # Ensure vocab size is large enough for tokens and markers
        assert vocab_size > 2, "Vocab size must be at least 3 (0 for padding, 1+ for tokens, last for marker)"

        # Default size based on similar logic to induction heads
        if size is None:
            if split == 'train':
                self.size = 10000  # Large enough for training
            else:
                self.size = 1000   # Reasonable validation size
        else:
            self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Generate a sequence and target for the given index"""
        # Set seed for reproducibility if in validation mode
        if self.fixed_seed is not None:
            rand_state = random.getstate()
            random.seed(self.fixed_seed + idx)

        sequence, targets = self.generate_sequence()

        # Restore random state if needed
        if self.fixed_seed is not None:
            random.setstate(rand_state)

        return torch.tensor(sequence, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def generate_sequence(self):
        """Generate a sequence for the selective copying task"""
        # Generate random tokens to memorize (values between 1 and vocab_size-2)
        tokens_to_copy = [random.randint(1, self.vocab_size-2) for _ in range(self.num_tokens_to_copy)]

        # Create a sequence filled with noise (zeros)
        sequence = [0] * self.seq_length

        # Randomly choose positions for the tokens to be copied
        # We'll place them in the first half of the sequence
        first_half_end = self.seq_length // 2
        positions = random.sample(range(first_half_end), self.num_tokens_to_copy)
        positions.sort()  # Sort positions to maintain order

        # Place tokens at the chosen positions
        for i, pos in enumerate(positions):
            sequence[pos] = tokens_to_copy[i]

        # Add marker tokens at the end (value = vocab_size-1)
        marker_token = self.vocab_size - 1
        marker_sequence = [marker_token] * self.num_tokens_to_copy

        # Combine the main sequence with marker tokens
        inputs = sequence + marker_sequence

        # The target is the tokens to be copied (these should be predicted at marker positions)
        targets = tokens_to_copy

        return inputs, targets


def create_datasets(max_seq_length=1048576):
    """
    Create training and validation datasets for the selective copying task.

    Args:
        max_seq_length: Maximum sequence length to consider

    Returns:
        training_datasets: Dict mapping sequence lengths to training datasets
        validation_datasets: Dict mapping sequence lengths to validation datasets
    """
    training_datasets = {}
    validation_datasets = {}

    # Generate sequence lengths (powers of 2)
    seq_lengths = [2**i for i in range(10, int(math.log2(max_seq_length))+1)]

    # Number of tokens to copy is fixed at 16 as in the paper
    num_tokens_to_copy = 16

    # Vocabulary size is 16 as in the paper
    vocab_size = 16

    for length in seq_lengths:
        # Scale training examples: more for short sequences, fewer for long ones
        scaling_factor = min(1.0, 4096 / length)  # Use 4096 as base length
        train_size = max(1000, int(10000 * scaling_factor))

        # Fixed number of validation examples
        val_size = 100

        # Create datasets
        training_datasets[length] = SelectiveCopyingDataset(
            split='train',
            seq_length=length,
            num_tokens_to_copy=num_tokens_to_copy,
            vocab_size=vocab_size,
            size=train_size
        )

        validation_datasets[length] = SelectiveCopyingDataset(
            split='val',
            seq_length=length,
            num_tokens_to_copy=num_tokens_to_copy,
            vocab_size=vocab_size,
            size=val_size,
            fixed_seed=42  # Fixed seed for reproducibility
        )

    return training_datasets, validation_datasets


def get_dataloaders(training_datasets, validation_datasets=None, batch_sizes=None, num_workers=2):
    """
    Create PyTorch DataLoaders for the training and optionally validation datasets.

    Args:
        training_datasets: Dict mapping sequence lengths to training datasets
        validation_datasets: Dict mapping sequence lengths to validation datasets (optional)
        batch_sizes: Dict mapping sequence lengths to batch sizes
                    (default: 64 for length 4096, scaled for others)
        num_workers: Number of worker processes for data loading

    Returns:
        training_loaders: Dict mapping sequence lengths to training DataLoaders
        validation_loaders: Dict mapping sequence lengths to validation DataLoaders (if validation_datasets provided)
    """
    training_loaders = {}

    if batch_sizes is None:
        batch_sizes = {}
        for length in training_datasets.keys():
            # Scale batch size inversely with sequence length, base is 64 at length 4096
            batch_sizes[length] = max(1, min(64, int(64 * 4096 / length)))

    for length, dataset in training_datasets.items():
        batch_size = batch_sizes.get(length, 1)
        training_loaders[length] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # Only create validation loaders if validation datasets are provided
    validation_loaders = {}
    if validation_datasets:
        for length, dataset in validation_datasets.items():
            batch_size = batch_sizes.get(length, 1)
            validation_loaders[length] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return training_loaders, validation_loaders


def save_datasets(training_datasets, validation_datasets, save_path='data'):
    """Save datasets to disk"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(training_datasets, os.path.join(save_path, 'selective_copying_train_datasets.pt'))
    torch.save(validation_datasets, os.path.join(save_path, 'selective_copying_val_datasets.pt'))
    print(f"Datasets saved to {save_path}")


def load_datasets(save_path='data'):
    """Load previously saved datasets"""
    try:
        train_path = os.path.join(save_path, 'selective_copying_train_datasets.pt')
        val_path = os.path.join(save_path, 'selective_copying_val_datasets.pt')
        
        training_datasets = torch.load(train_path)
        validation_datasets = torch.load(val_path)
        
        print(f"Datasets loaded successfully from {save_path}")
        return training_datasets, validation_datasets
    except FileNotFoundError:
        print(f"Dataset files not found in {save_path}")
        return None, None


def main():
    """Main function to create, display, and save datasets"""
    # Create datasets
    print(f"Creating datasets...")
    training_datasets, validation_datasets = create_datasets(max_seq_length=1048576)

    # Print information about the datasets
    print("\n--- Dataset Information ---")
    print(f"Available sequence lengths: {sorted(list(training_datasets.keys()))}")

    for seq_len in sorted(list(training_datasets.keys())):
        train_dataset = training_datasets[seq_len]
        val_dataset = validation_datasets[seq_len]

        print(f"\nSequence Length: {seq_len}")
        print(f"  Training dataset size: {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")

        # Only show a sample for a small sequence length to avoid clutter
        if seq_len == 1024:  # Just show for one length as an example
            # Get a sample from training set
            train_sample_x, train_sample_y = train_dataset[0]
            print(f"  Sample input shape: {train_sample_x.shape}")
            print(f"  Sample target shape: {train_sample_y.shape}")

            # Count marker tokens in input
            marker_token = train_dataset.vocab_size - 1
            marker_count = (train_sample_x == marker_token).sum().item()
            print(f"  Number of marker tokens: {marker_count}")

            # Show a few of the tokens to copy
            print(f"  Target tokens to copy: {train_sample_y[:5].tolist()}...")

    # Save the datasets
    save_datasets(training_datasets, validation_datasets)

    # Create training dataloaders (don't save them)
    print("\nCreating dataloaders...")
    training_loaders, _ = get_dataloaders(training_datasets, validation_datasets, num_workers=0)

    # Show example of accessing a dataloader
    print("\n--- Example Usage ---")
    print("# Load datasets")
    print("training_datasets, validation_datasets = load_datasets()")
    print("# Create dataloaders on the fly")
    print("training_loaders, validation_loaders = get_dataloaders(training_datasets, validation_datasets)")
    print("# Get a specific dataloader")
    print("train_loader_4096 = training_loaders[4096]")
    print("# Use in training loop")
    print("for inputs, targets in train_loader_4096:")
    print("    # Your training code here")
    print("    pass")

    print("\nAll done! Ready for training.")


if __name__ == "__main__":
    # Add multiprocessing support for Windows
    mp.freeze_support()
    main()