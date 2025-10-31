# output_projector.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025

import torch
from transformers import AutoModelForCausalLM
from deepseek_inference import OptimizedEngine
from utils.encryption import encrypt_if_enabled

class OutputProjector(torch.nn.Module):
    def __init__(self, model_name="gpt2", state_dim=768, beta0=0.1, gamma=0.01, epsilon=0.01):
        super().__init__()
        self.decoder = OptimizedEngine(model=model_name, qar=True)
        self.projector = torch.nn.Linear(state_dim, self.decoder.config.hidden_size)
        self.beta0 = beta0
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, states, iter_k=0, error_times=None, y_base=None):
        states_enc = encrypt_if_enabled(states)
        proj_states = self.projector(states_enc)
        output = self.decoder.infer(inputs_embeds=proj_states).logits

        if error_times is None:
            error_times = torch.zeros(states.size(0), device=states.device)
        deviation = torch.norm(states_enc - y_base, dim=-1)
        error_times += (deviation > self.epsilon).float()
        grad_error = error_times.diff(dim=0) / (iter_k + 1e-6) if iter_k > 0 else torch.zeros_like(error_times)

        beta_k = self.beta0 / (1 + self.gamma * iter_k)
        v_visual = torch.ones(states.size(0), device=states.device) * 0.5
        w = torch.exp(-beta_k * torch.abs(grad_error) * v_visual.unsqueeze(-1))

        boundary_loss = w * torch.norm(torch.autograd.grad(output, states, create_graph=True)[0]).mean()
        return output, boundary_loss, error_times
