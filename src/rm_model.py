import os
import zipfile
import logging
from typing import Dict, Optional, Tuple, Union
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_LABEL2ID = {'-3.5': 0, '-2.5': 1, '-0.1': 2, '2.0': 3, '3.0': 4, '10.0': 5}
DEFAULT_ID2LABEL = {0: '-3.5', 1: '-2.5', 2: '-0.1', 3: '2.0', 4: '3.0', 5: '10.0'}


def ensure_lora_unzipped(lora_path: str, zip_path: str = "reward model/reward-model-lora.zip") -> str:
    """
    Ensure the LoRA model directory exists. If not, auto-extract from the zip archive.
    """
    if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        return lora_path

    # Check alternative common locations
    candidates = [
        lora_path,
        "reward-model-lora-doorkey",
        "./reward-model-lora-doorkey",
        "reward-model-lora-putnear",
        "./reward-model-lora-putnear",
        "reward-model-lora-v2",
        "./reward-model-lora-v2",
        "reward model/reward-model-lora",
        "./reward-model-lora"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "adapter_model.safetensors")):
            return c

    # Attempt to unzip from zip_path if available
    zip_candidates = [
        zip_path,
        "reward model/reward-model-lora.zip",
        "./reward-model-lora.zip"
    ]
    found_zip = None
    for z in zip_candidates:
        if os.path.exists(z):
            found_zip = z
            break

    if found_zip:
        extract_target = "reward model/reward-model-lora"
        logger.info(f"Extracting LoRA model from '{found_zip}' to '{extract_target}'...")
        os.makedirs(extract_target, exist_ok=True)
        with zipfile.ZipFile(found_zip, "r") as archive:
            archive.extractall(extract_target)
        logger.info("Extraction complete.")
        return extract_target

    raise FileNotFoundError(
        f"Could not find LoRA weights directory '{lora_path}' or zip archive '{zip_path}'."
    )


class ActionLevelRewardModel:
    """
    Wrapper around the fine-tuned DeBERTa-v3-small sequence classification model.
    Provides fast, cached inference for state-action description strings.
    """
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        lora_path: str = "reward model/reward-model-lora",
        device: str = "auto",
        use_cache: bool = True,
        cache_size: int = 10000,
        max_length: int = 128,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None
    ):
        if isinstance(model_name, dict):
            cfg = model_name
            rm_cfg = cfg.get("reward_model", {})
            model_name = rm_cfg.get("model_name", "microsoft/deberta-v3-small")
            lora_path = rm_cfg.get("lora_path", lora_path)
            device = cfg.get("experiment", {}).get("device", device)
            use_cache = rm_cfg.get("use_cache", use_cache)
            cache_size = rm_cfg.get("cache_size", cache_size)
            max_length = rm_cfg.get("max_length", max_length)

        self.model_name = model_name
        self.label2id = label2id or DEFAULT_LABEL2ID
        self.id2label = id2label or DEFAULT_ID2LABEL
        self.use_cache = use_cache
        self.cache: Dict[str, float] = {}
        self.cache_size = cache_size
        self.max_length = max_length

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        lora_dir = ensure_lora_unzipped(lora_path)

        # Auto-detect label mapping if present in lora_dir
        mapping_file = os.path.join(lora_dir, "label_mapping.json")
        if os.path.exists(mapping_file):
            import json
            try:
                with open(mapping_file, "r") as f:
                    mapping = json.load(f)
                    if not label2id and "label2id" in mapping:
                        self.label2id = mapping["label2id"]
                    if not id2label and "id2label" in mapping:
                        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
                logger.info(f"Loaded dynamic label mapping from '{mapping_file}': {self.label2id}")
            except Exception as e:
                logger.warning(f"Could not load '{mapping_file}': {e}")

        logger.info(f"Loading tokenizer for '{model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        merged_dir = os.path.join(lora_dir, "merged")
        if os.path.exists(merged_dir) and (os.path.exists(os.path.join(merged_dir, "model.safetensors")) or os.path.exists(os.path.join(merged_dir, "pytorch_model.bin"))):
            logger.info(f"Loading merged standalone sequence classification model from '{merged_dir}'...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                merged_dir,
                torch_dtype=torch.float32
            )
        else:
            logger.info(f"Loading base sequence classification model '{model_name}'...")
            try:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.label2id),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )
            except Exception:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.label2id),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    torch_dtype=torch.float32,
                    ignore_mismatched_sizes=True
                )

            logger.info(f"Loading PEFT LoRA adapter from '{lora_dir}'...")
            self.model = PeftModel.from_pretrained(base_model, lora_dir)

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Action-Level RM successfully loaded onto {self.device}!")

    def predict_reward(self, text: str) -> float:
        """
        Predict the scalar reward value corresponding to the predicted reward tier.
        Uses in-memory caching to eliminate redundant forward passes.
        """
        if self.use_cache and text in self.cache:
            return self.cache[text]

        inputs = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = int(torch.argmax(outputs.logits, dim=-1).item())
            pred_label = self.id2label.get(pred_id, "0.0")
            reward_val = float(pred_label)

        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                # Evict oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[text] = reward_val

        return reward_val

    # Convenience aliases
    predict = predict_reward
    get_reward = predict_reward
