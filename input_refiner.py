# input_refiner.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025
# Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX
# License: CC-BY-SA 4.0

import torch
import torch.nn as nn
import torch.fft
from transformers import AutoTokenizer
from deepseek_inference import OptimizedEngine
from whisper import Whisper
from utils.encryption import encrypt_if_enabled

class InputRefiner(nn.Module):
    def __init__(
        self,
        llm_model_name="meta-llama/Llama-3-8b",
        vision_model_name="resnet50",
        audio_model_name="whisper-base",
        freq_scale=0.1,
        beta0=0.1,
        gamma=0.01,
        epsilon=0.01
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = OptimizedEngine(model=llm_model_name, qar=True)
        self.vision = torch.hub.load("pytorch/vision", vision_model_name, pretrained=True)
        self.audio = Whisper(audio_model_name)
        self.fusion = nn.Linear(768 + 768 + 512, 512)
        self.freq_scale = freq_scale
        self.beta0 = beta0
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, text, image, audio, prev_z=None, iter_k=0, error_times=None, y_base=None):
        # Embed multimodal inputs
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        z_text = self.llm.infer(text_inputs)

        z_image = self.vision(image).mean(dim=1) if image is not None else torch.zeros(z_text.size(0), 768, device=z_text.device)
        z_audio = self.audio.transcribe(audio)["embedding"] if audio is not None else torch.zeros(z_text.size(0), 512, device=z_text.device)

        z = torch.cat([z_text, z_image, z_audio], dim=-1)
        z = self.fusion(z)

        # Optional FHE encryption
        z_enc = encrypt_if_enabled(z)

        # Spectral regularization
        freq_domain = torch.fft.fftn(z_enc)
        spec_loss = torch.norm(freq_domain - self.freq_scale * freq_domain.mean(dim=-1, keepdim=True), dim=-1).mean()

        # TEGW
        if error_times is None:
            error_times = torch.zeros(z_enc.size(0), device=z_enc.device)
        deviation = torch.norm(z_enc - y_base, dim=-1)
        error_times += (deviation > self.epsilon).float()
        grad_error = error_times.diff(dim=0) / (iter_k + 1e-6) if iter_k > 0 else torch.zeros_like(error_times)

        beta_k = self.beta0 / (1 + self.gamma * iter_k)
        v_visual = torch.tensor([1.0 if image is not None or audio is not None else 0.5 for _ in text], device=z.device)
        w = torch.exp(-beta_k * torch.abs(grad_error) * v_visual.unsqueeze(-1))
        z_weighted = w * z_enc

        if prev_z is not None:
            spec_loss += 0.1 * torch.norm(z_weighted - prev_z, dim=-1).mean()

        return z_weighted, spec_loss, error_times
