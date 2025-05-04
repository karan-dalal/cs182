import torch
import random
import math
from torch.utils.data import Dataset, DataLoader

class InductionHeadsDataset(Dataset):
    """
    Dataset for the induction heads task from the Mamba paper.
    Generates sequences with token pairs in the first half and triggers in the second half.
    Each sample has:
        • first ½:   k‑v pairs  (A→B)
        • second ½: triggers k   whose target is   v  at the next position
    Non‑target positions get label -1 so they can be masked out.
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
        max_pairs = (seq_length // 2 - 1) // 2
        num_patterns = min(max_pairs, num_patterns)
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
        
        rng = random if self.fixed_seed is None else random.Random(self.fixed_seed + idx)
        sequence, labels = self.generate_sequence(rng)
            
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def generate_sequence(self, rng: random.Random):
        """Generate a sequence with patterns and triggers for induction heads testing"""
        L          = self.seq_length
        half       = L // 2
        pattern_end = half - 1                    # last idx allowed for *A* so that B is ≤ half‑1
        seq    = [0] * L                          # 0 = noise token
        labels = [-1] * L                         # -1 ➟ will be masked out

        # -------------------------------------------------- #
        # 1. sample distinct key‑value pairs
        pairs = []
        while len(pairs) < self.num_patterns:
            k = rng.randint(1, self.vocab_size - 1)
            v = rng.randint(1, self.vocab_size - 1)
            if v != k and (k, v) not in pairs:
                pairs.append((k, v))

        # -------------------------------------------------- #
        # 2. place each (A,B) non‑overlapping in the 1st half
        occupied = set()
        for k, v in pairs:
            while True:
                pos = rng.randint(0, pattern_end - 1)  # leave room for +1
                if pos     not in occupied and \
                   pos + 1 not in occupied:
                    break
            seq[pos]     = k
            seq[pos + 1] = v
            occupied.update({pos, pos + 1})

        # -------------------------------------------------- #
        # 3. choose a subset of pairs to trigger in 2nd half
        n_trig  = min(2, len(pairs))
        triggers = rng.sample(pairs, n_trig)
        for k, v in triggers:
            while True:
                pos = rng.randint(half, L - 2)   # also leave room for +1
                if seq[pos] == 0 and seq[pos + 1] == 0:
                    break
            seq[pos]     = k          # key only
            seq[pos + 1] = v          # correct value
            labels[pos + 1] = v       # compute loss *only* here

        # -------------------------------------------------- #
        # 4. fill remaining blanks with random noise (≠0 optional)
        remaining = [i for i, tok in enumerate(seq) if tok == 0]
        if remaining:
            seq_tensor = torch.randint(1, self.vocab_size, (len(remaining),))
            for i, sidx in enumerate(remaining):
                seq[sidx] = int(seq_tensor[i])

        return seq, labels