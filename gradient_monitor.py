# gradient_monitor.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025

import torch

class GradientMonitor:
    def __init__(self, theta=0.1):
        self.theta = theta

    def forward(self, y_t, error_times, prev_state=None, iter_k=0):
        grad_error = error_times.diff(dim=0) / (iter_k + 1e-6) if iter_k > 0 else torch.zeros_like(error_times)
        avg_grad = torch.abs(grad_error).mean()
        if avg_grad > self.theta and prev_state is not None:
            y_t = prev_state  # Rollback
        return y_t, avg_grad
