# temporal_encoder.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025

import torch
from mamba_ssm import Mamba2
from deepseek_inference import OptimizedEngine
from utils.encryption import encrypt_if_enabled

class TemporalStateEncoder(torch.nn.Module):
    def __init__(self, embed_dim=512, beta0=0.1, gamma=0.01, epsilon=0.01):
        super().__init__()
        self.mamba = OptimizedEngine(model="mamba2", qar=True, d_model=embed_dim)
        self.beta0 = beta0
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, tokens, iter_k=0, error_times=None, y_base=None):
        tokens_enc = encrypt_if_enabled(tokens)
        states = self.mamba.infer(tokens_enc)
        probs = torch.softmax(states, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

        if error_times is None:
            error_times = torch.zeros(states.size(0), device=states.device)
        deviation = torch.norm(states - y_base, dim=-1)
        error_times += (deviation > self.epsilon).float()
        grad_error = error_times.diff(dim=0) / (iter_k + 1e-6) if iter_k > 0 else torch.zeros_like(error_times)

        beta_k = self.beta0 / (1 + self.gamma * iter_k)
        v_visual = torch.tensor([1.0 if "image" in str(type(tokens)) else 0.5 for _ in range(states.size(0))], device=states.device)
        w = torch.exp(-beta_k * torch.abs(grad_error) * v_visual.unsqueeze(-1))

        temp_loss = w * (-torch.log(probs + 1e-10)).mean() + 0.1 * entropy
        return states, temp_loss, error_times
