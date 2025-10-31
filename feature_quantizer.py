# feature_quantizer.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025
# Zenodo: https://doi.org/10.5281/zenodo.17429702

import torch
import torch.nn as nn
import sympy
from deepseek_inference import OptimizedEngine
from utils.encryption import encrypt_if_enabled

class FeatureQuantizer(nn.Module):
    def __init__(self, embed_dim=512, num_tokens=512, beta0=0.1, gamma=0.01, epsilon=0.01, max_rule_complexity=100):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_tokens, embed_dim))
        self.llm_rule_gen = OptimizedEngine(model="llama3", qar=True)
        self.beta0 = beta0
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_rule_complexity = max_rule_complexity
        self.rule_cache = {}

    def auto_derive_rules(self, symbols):
        key = tuple(symbols)
        if key in self.rule_cache:
            return self.rule_cache[key]
        rules = [sympy.Eq(sympy.Symbol(s), sympy.Symbol('invariant')) for s in symbols]
        rules = [r for r in rules if sympy.count_ops(r) < self.max_rule_complexity]
        self.rule_cache[key] = rules
        return rules

    def forward(self, z, iter_k=0, error_times=None, y_base=None, baseline_symbols=["A", "B"]):
        z_enc = encrypt_if_enabled(z)
        distances = torch.cdist(z_enc, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook[indices]
        probs = torch.softmax(-distances, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

        if error_times is None:
            error_times = torch.zeros(indices.size(0), device=z.device)
        deviation = torch.norm(quantized - y_base, dim=-1)
        error_times += (deviation > self.epsilon).float()
        grad_error = error_times.diff(dim=0) / (iter_k + 1e-6) if iter_k > 0 else torch.zeros_like(error_times)

        beta_k = self.beta0 / (1 + self.gamma * iter_k)
        v_visual = torch.tensor([1.0 if "image" in str(type(z)) else 0.5 for _ in range(z.size(0))], device=z.device)
        w = torch.exp(-beta_k * torch.abs(grad_error) * v_visual.unsqueeze(-1))

        rules = self.auto_derive_rules(baseline_symbols)
        valid_mask = torch.ones(len(rules), dtype=torch.bool, device=z.device)
        quantized = quantized[valid_mask]

        quant_loss = w * (z_enc - quantized).norm(dim=-1).mean() + 0.1 * entropy
        return quantized, quant_loss, error_times
