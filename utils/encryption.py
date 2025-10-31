# utils/encryption.py
# Proto-AGI v0.1 — Justin Thurmond, October 2025

import yaml
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

def encrypt_if_enabled(data):
    if config["fhe_params"]["use_seal"]:
        from seal import SEALContext, CKKSEncoder
        context = SEALContext(poly_degree=config["fhe_params"]["poly_degree"])
        encoder = CKKSEncoder(context)
        return encoder.encode(data)
    return data
