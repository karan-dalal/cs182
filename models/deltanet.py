import torch
from fla.layers import DeltaNet

class DeltaNetModel(torch.nn.Module):
    def __init__(self, vocab_size=16, d_model=16, n_layer=2):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        
        self.layers = torch.nn.ModuleList([
            DeltaNet(
                d_model=d_model,
                num_heads=1,
                use_short_conv=False,
            )
            for _ in range(n_layer)
        ])
        
        self.norm = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x , _, _ = layer(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits