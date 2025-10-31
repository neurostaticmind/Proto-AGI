# demo.py
# Proto-AGI v0.1 Demo — Justin Thurmond
# Run: python demo.py

import torch
from refine_loop import refine_loop
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data = {
    "text": ["Design a super-strong, lightweight alloy using quantum principles"],
    "image": [torch.randn(3, 224, 224)],  # Replace with real image
    "audio": None
}

print("Starting Proto-AGI refinement...")
results, seed = refine_loop(data, config, domain="materials_science")
print(f"Done! Final stability: {results[-1][1]:.3f}")
