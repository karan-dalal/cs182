import torch
import random
import math
from torch.utils.data import Dataset, DataLoader

class InductionHeadsDataset(Dataset):
    """
    Dataset for the induction heads task from the Mamba paper.
    Generates sequences with token pairs in the first half and triggers in the second half.
    """
    def __init__(self, 
                 split='train',
                 seq_length=256, 
                 vocab_size=16, 
                 num_patterns=4, 
                 size=None, 
                 fixed_seed=None):
        """
        Initialize the dataset.
        
        Args:
            split: 'train' or 'val' to indicate dataset type
            seq_length: Length of generated sequences
            vocab_size: Size of the vocabulary
            num_patterns: Number of token pairs to introduce
            size: Number of examples in the dataset
            fixed_seed: Seed for reproducibility (used for validation)
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_patterns = num_patterns
        self.split = split
        self.fixed_seed = fixed_seed
        
        # Default size based on Mamba paper (8192 steps * batch_size 8)
        if size is None:
            if split == 'train':
                self.size = 8192 * 8  # Default from paper
            else:
                self.size = 1000  # Reasonable validation size
        else:
            self.size = size
            
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Generate a sequence and its corresponding labels for the given index"""
        # Set seed for reproducibility if in validation mode
        if self.fixed_seed is not None:
            rand_state = random.getstate()
            random.seed(self.fixed_seed + idx)
        
        sequence, labels = self.generate_sequence()
        
        # Restore random state if needed
        if self.fixed_seed is not None:
            random.setstate(rand_state)
            
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def generate_sequence(self):
        """Generate a sequence with patterns and triggers for induction heads testing"""
        # Initialize sequence with zeros (will be filled with random tokens later)
        sequence = [0] * self.seq_length
        
        # Define pattern and trigger regions
        pattern_end = self.seq_length // 2 - 1
        
        # Generate token pairs (A→B) where A and B are different tokens
        patterns = []
        for _ in range(self.num_patterns):
            token_a = random.randint(1, self.vocab_size-1)  # Avoid using token 0
            token_b = random.randint(1, self.vocab_size-1)
            while token_b == token_a:  # Ensure A and B are different
                token_b = random.randint(1, self.vocab_size-1)
            patterns.append((token_a, token_b))
        
        # Place patterns in first half of sequence with spacing between them
        first_half_positions = random.sample(range(pattern_end), len(patterns))
        first_half_positions.sort()  # Sort positions to maintain order
        
        for i, pos in enumerate(first_half_positions):
            sequence[pos] = patterns[i][0]    # Place A
            sequence[pos+1] = patterns[i][1]  # Place B next to A
        
        # Create labels array (-1 indicates no specific prediction)
        labels = [-1] * self.seq_length
        
        # Place triggers in second half and mark labels
        # Select a subset of patterns to test
        triggers = random.sample(patterns, min(len(patterns), 2))
        second_half_positions = random.sample(range(pattern_end + 1, self.seq_length - 1), len(triggers))
        
        for i, pos in enumerate(second_half_positions):
            trigger_token = triggers[i][0]  # Token A
            expected_token = triggers[i][1]  # Token B that should follow
            
            sequence[pos] = trigger_token
            
            # Mark the next position as requiring the specific token
            labels[pos+1] = expected_token
            
            # For training, also place the expected token B in the sequence
            if self.split == 'train':
                sequence[pos+1] = expected_token
        
        # Fill remaining positions with random tokens
        for i in range(self.seq_length):
            if sequence[i] == 0:  # If position still has a placeholder
                sequence[i] = random.randint(1, self.vocab_size-1)
        
        return sequence, labels

def create_datasets(max_seq_length=1048576):
    """
    Create training and validation datasets for the induction heads task.
    
    Args:
        max_seq_length: Maximum sequence length to consider
    
    Returns:
        training_datasets: Dict mapping sequence lengths to training datasets
        validation_datasets: Dict mapping sequence lengths to validation datasets
    """
    # Calculate number of training examples based on sequence length
    # More examples for shorter sequences, fewer for longer ones
    training_datasets = {}
    validation_datasets = {}
    
    # Generate sequence lengths from 2^6 to log2(max_seq_length)
    seq_lengths = [2**i for i in range(6, int(math.log2(max_seq_length))+1)]
    
    # Training dataset scaling factor - inversely proportional to sequence length
    base_train_size = 8192 * 8  # From Mamba paper: 8192 steps with batch size 8
    
    for length in seq_lengths:
        # Scale training examples: more for short sequences, fewer for long ones
        # For the base length (256), use the full size; scale inversely with length
        scaling_factor = min(1.0, 256 / length)
        train_size = int(base_train_size * scaling_factor)
        
        train_size = max(train_size, 1000)
        
        # Fixed number of validation examples per length
        val_size = 100  
        
        # Create datasets
        training_datasets[length] = InductionHeadsDataset(
            split='train',
            seq_length=length,
            vocab_size=16,
            num_patterns=4,
            size=train_size
        )
        
        validation_datasets[length] = InductionHeadsDataset(
            split='val',
            seq_length=length,
            vocab_size=16,
            num_patterns=4,
            size=val_size,
            fixed_seed=42  
        )
    
    return training_datasets, validation_datasets

def get_dataloaders(training_datasets, validation_datasets, batch_sizes=None):
    """
    Create PyTorch DataLoaders for the training and validation datasets.
    
    Args:
        training_datasets: Dict mapping sequence lengths to training datasets
        validation_datasets: Dict mapping sequence lengths to validation datasets
        batch_sizes: Dict mapping sequence lengths to batch sizes
                    (default: 8 for length 256, scaled for others)
    
    Returns:
        training_loaders: Dict mapping sequence lengths to training DataLoaders
        validation_loaders: Dict mapping sequence lengths to validation DataLoaders
    """
    training_loaders = {}
    validation_loaders = {}
    
    if batch_sizes is None:
        batch_sizes = {}
        for length in training_datasets.keys():
            # Scale batch size inversely with sequence length, base is 8 at length 256
            batch_sizes[length] = max(1, min(8, int(8 * 256 / length)))
    
    for length, dataset in training_datasets.items():
        batch_size = batch_sizes.get(length, 1)
        training_loaders[length] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    for length, dataset in validation_datasets.items():
        batch_size = batch_sizes.get(length, 1)
        validation_loaders[length] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    return training_loaders, validation_loaders


# Create datasets up to sequence length 2^16 = 65536
training_datasets, validation_datasets = create_datasets(max_seq_length=65536)

# Create dataloaders with default batch sizing
training_loaders, validation_loaders = get_dataloaders(training_datasets, validation_datasets)

# # Access the standard training loader (length 256)
# train_loader_256 = training_loaders[256]
