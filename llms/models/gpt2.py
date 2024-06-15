"""
This file contains the GPT2 model.
"""
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from dataclasses import dataclass


@dataclass
class MyGPT2Config:
    seq_len: int = 1024
    vocab_size: int = 50257


class GPT2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(
            vocab_size=MyGPT2Config.vocab_size,
            n_positions=MyGPT2Config.seq_len,
        )
    
    def forward(self, x):
        return GPT2Model(self.config)(x)