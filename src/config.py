import os
import yaml
from typing import Any, Dict, Optional
import torch


def get_default_device() -> str:
    """Auto-detect the best available device for training/inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(config_path: str = "configs/doorkey_config.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file and apply any runtime overrides.
    """
    if not os.path.exists(config_path):
        # Fallback to legacy config names if new ones are queried or vice versa
        if config_path == "configs/doorkey_config.yaml" and os.path.exists("configs/config.yaml"):
            config_path = "configs/config.yaml"
        elif config_path == "configs/config.yaml" and os.path.exists("configs/doorkey_config.yaml"):
            config_path = "configs/doorkey_config.yaml"
        elif config_path == "configs/doorkey_rm_ppo.yaml" and os.path.exists("configs/rm_ppo.yaml"):
            config_path = "configs/rm_ppo.yaml"
        elif config_path == "configs/rm_ppo.yaml" and os.path.exists("configs/doorkey_rm_ppo.yaml"):
            config_path = "configs/doorkey_rm_ppo.yaml"
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                parts = key.split(".")
                d = config
                for part in parts[:-1]:
                    if part not in d or not isinstance(d[part], dict):
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = val

    # Resolve device if set to 'auto'
    device = config.get("experiment", {}).get("device", "auto")
    if device == "auto":
        resolved_device = get_default_device()
        if "experiment" not in config:
            config["experiment"] = {}
        config["experiment"]["device"] = resolved_device

    return config
