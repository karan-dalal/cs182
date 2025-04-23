import torch
from mamba_ssm import Mamba

class MambaModel(torch.nn.Module):
    def __init__(self, vocab_size=16, d_model=64, n_layer=2, d_state=16):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        
        self.layers = torch.nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            for _ in range(n_layer)
        ])
        
        self.norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits