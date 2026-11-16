"""
Standalone Training Script for Action-Level Reward Model (DeBERTa-v3-small with LoRA).
Supports Mac (MPS / CPU) and Linux/Cloud (CUDA).
"""

import os
import time
import json
import argparse
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
import transformers.utils.logging as hf_logging
from peft import LoraConfig, get_peft_model, TaskType

from src.config import load_config, get_default_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """
    Live terminal dashboard for tracking LoRA fine-tuning stages and metrics.
    """
    def __init__(self, log, train_size: int, eval_size: int, unique_rewards: list, early_stopping_cb=None):
        self.log = log
        self.train_size = train_size
        self.eval_size = eval_size
        self.unique_rewards = unique_rewards
        self.early_stopping_cb = early_stopping_cb
        self.best_macro_f1 = -1.0
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.log.info("=" * 75)
        self.log.info("🚀 STARTING DEBERTA-V3 REWARD MODEL LORA FINE-TUNING")
        self.log.info(f"   Train Split:         {self.train_size:,} samples")
        self.log.info(f"   Validation Split:    {self.eval_size:,} samples")
        self.log.info(f"   Reward Classes:      {len(self.unique_rewards)} classes {self.unique_rewards}")
        self.log.info(f"   Target Epochs:       {int(args.num_train_epochs)}")
        if self.early_stopping_cb:
            self.log.info(f"   Early Stopping:      patience = {self.early_stopping_cb.early_stopping_patience} epochs")
        self.log.info(f"   Batch Size:          {args.per_device_train_batch_size} (Eval: {args.per_device_eval_batch_size})")
        self.log.info(f"   Peak Learning Rate:  {args.learning_rate:.2e}")
        self.log.info(f"   Total Planned Steps: {state.max_steps:,}")
        self.log.info("=" * 75)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        current_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        total_epochs = int(args.num_train_epochs)
        self.log.info(f"\n▶ [STAGE] Starting Epoch {current_epoch}/{total_epochs}...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            max_steps = state.max_steps
            pct = (step / max_steps) * 100 if max_steps else 0.0
            loss = logs.get("loss", 0.0)
            lr_val = logs.get("learning_rate", 0.0)
            epoch = state.epoch if state.epoch is not None else 0.0

            elapsed = time.time() - self.train_start_time if self.train_start_time else 0.0
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            rem_steps = max(0, max_steps - step)
            eta_sec = rem_steps / steps_per_sec if steps_per_sec > 0 else 0.0
            eta_min = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)

            self.log.info(
                f"  [Step {step:4d}/{max_steps} ({pct:4.1f}%)] "
                f"Epoch: {epoch:4.2f} | Loss: {loss:.4f} | LR: {lr_val:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s | ETA: {eta_min:02d}m{eta_s:02d}s"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = int(round(state.epoch)) if state.epoch is not None else 1
            loss = metrics.get("eval_loss", 0.0)
            acc = metrics.get("eval_accuracy", 0.0) * 100
            macro_f1 = metrics.get("eval_f1", 0.0) * 100
            weighted_f1 = metrics.get("eval_weighted_f1", 0.0) * 100
            duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0

            is_best = False
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                is_best = True

            patience_info = ""
            if self.early_stopping_cb is not None:
                cnt = getattr(self.early_stopping_cb, "early_stopping_patience_counter", 0)
                pat = getattr(self.early_stopping_cb, "early_stopping_patience", 10)
                rem = max(0, pat - cnt)
                patience_info = f"   • Early Stopping:     {cnt}/{pat} epochs without improvement (patience remaining: {rem})"

            self.log.info("\n" + "=" * 75)
            self.log.info(f"📊 VALIDATION REPORT: Epoch {epoch}/{int(args.num_train_epochs)} (completed in {duration:.1f}s)")
            self.log.info(f"   • Validation Loss:    {loss:.4f}")
            self.log.info(f"   • Accuracy:           {acc:.2f}%")
            self.log.info(f"   • Macro F1-Score:     {macro_f1:.2f}%" + (" 🌟 [NEW BEST]" if is_best else ""))
            self.log.info(f"   • Weighted F1-Score:  {weighted_f1:.2f}%")
            if patience_info:
                self.log.info(patience_info)
            self.log.info("=" * 75 + "\n")


def _train_reward_model_impl(config: Dict[str, Any], log_dir: str, output_dir: str):
    ds_cfg = config.get("dataset", {})
    train_cfg = config.get("rm_training", {})
    rm_cfg = config.get("reward_model", {})

    data_path = ds_cfg.get("data_path", "data/doorkey_reward_dataset_balanced.csv")
    model_name = rm_cfg.get("model_name", "microsoft/deberta-v3-small")
    epochs = train_cfg.get("epochs", 20)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
    save_total_limit = train_cfg.get("save_total_limit", 3)
    batch_size = train_cfg.get("batch_size", 32)
    lr = float(train_cfg.get("learning_rate", 2e-4))
    max_length = train_cfg.get("max_length", 96)
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    target_modules = train_cfg.get("target_modules", ["value_proj", "query_proj"])
    lora_r = train_cfg.get("lora_r", 8)
    lora_alpha = train_cfg.get("lora_alpha", 16)
    lora_dropout = train_cfg.get("lora_dropout", 0.1)

    # 1. Load and process dataset
    logger.info(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found. Please verify the dataset path.")

    df = pd.read_csv(data_path)
    if 'reward' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'reward': 'label'})

    logger.info(f"Loaded dataset with {len(df)} rows across classes: {sorted(df['label'].unique())}")

    if ds_cfg.get("balanced_downsampling", False):
        logger.info("Performing balanced downsampling across classes...")
        min_class_size = int(df['label'].value_counts().min())
        sampled_dfs = [group.sample(n=min_class_size, random_state=42) for _, group in df.groupby('label')]
        df = pd.concat(sampled_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Balanced dataset shape: {df.shape}")

    sample_size = ds_cfg.get("sample_size")
    if sample_size and sample_size < len(df):
        logger.info(f"Subsampling to {sample_size} records...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    unique_rewards = sorted(list(set(df["label"])))
    num_classes = len(unique_rewards)
    label2id = {str(float(reward)): i for i, reward in enumerate(unique_rewards)}
    id2label = {i: str(float(reward)) for i, reward in enumerate(unique_rewards)}
    logger.info(f"Class mapping ({num_classes} classes): {label2id}")

    # 2. Tokenize and Split Dataset
    logger.info(f"Loading tokenizer '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        if "label" in examples:
            tokenized["labels"] = [int(label2id[str(float(l))]) for l in examples["label"]]
        return tokenized

    train_split = ds_cfg.get("train_split", 0.8)
    unique_prompt_split = ds_cfg.get("unique_prompt_split", True)

    if unique_prompt_split:
        logger.info("Performing STRICT DISJOINT PROMPT SPLIT (ensuring test set has 100% unique unseen prompts)...")
        unique_texts_df = df.drop_duplicates(subset=["text"]).copy()
        _, test_unique_df = train_test_split(
            unique_texts_df,
            test_size=1.0 - train_split,
            random_state=42,
            stratify=unique_texts_df["label"]
        )
        test_prompts_set = set(test_unique_df["text"])
        train_df = df[~df["text"].isin(test_prompts_set)].copy()
        test_df = test_unique_df.copy()

        logger.info(
            f"Disjoint Split: Train={len(train_df)} rows ({train_df['text'].nunique()} unique), "
            f"Test={len(test_df)} rows (100% unique, zero overlap: {len(set(train_df['text']) & set(test_df['text']))})"
        )

        # Save dedicated test dataset for transparency / reproducibility
        disjoint_test_path = os.path.join(log_dir, "test_dataset_unique_disjoint.csv")
        test_df.to_csv(disjoint_test_path, index=False)
        logger.info(f"Saved strictly disjoint unique test set to: {disjoint_test_path}")

        train_cols = [c for c in train_df.columns if c not in ["text", "label"]] + (["text", "label"] if "text" in train_df.columns else [])
        train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True, remove_columns=list(train_df.columns))
        test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True, remove_columns=list(test_df.columns))
        splits = DatasetDict({"train": train_ds, "test": test_ds})
    else:
        dataset = Dataset.from_pandas(df)
        logger.info("Tokenizing dataset...")
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        splits = tokenized.train_test_split(test_size=1.0 - train_split, seed=42)

    # 3. Model setup with LoRA
    logger.info(f"Initializing base model '{model_name}'...")
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            label2id=label2id,
            id2label=id2label,
            torch_dtype=torch.float32,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
    except Exception:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            label2id=label2id,
            id2label=id2label,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True
        )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=train_cfg.get("modules_to_save", ["classifier", "pooler", "score"])
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 4. Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": float(acc),
            "f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        }

    # Precision settings: CUDA supports bf16; Mac (MPS/CPU) is more stable in fp32
    is_cuda = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        logging_steps=train_cfg.get("logging_steps", 25),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        warmup_steps=10,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=False,
        report_to=["tensorboard"]
    )

    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
    progress_cb = TrainingProgressCallback(
        logger,
        train_size=len(splits["train"]),
        eval_size=len(splits["test"]),
        unique_rewards=unique_rewards,
        early_stopping_cb=early_stopping_cb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_cb, progress_cb]
    )

    logger.info("Starting Reward Model LoRA fine-tuning...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    logger.info(f"Fine-tuning complete in {duration:.2f} seconds ({duration/60:.2f} minutes)!")

    # Best Model Verification & Summary
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    logger.info("=" * 75)
    logger.info("🏆 BEST MODEL CHECKPOINT STATUS:")
    logger.info(f"   • Checkpoint Path:   {best_ckpt}")
    if best_metric is not None:
        logger.info(f"   • Peak Macro F1:     {best_metric * 100:.2f}%")
    logger.info(f"   • Total Epochs Run:  {trainer.state.epoch:.2f} / {epochs}")
    logger.info("   • Automatic Reload:  Best checkpoint weights successfully restored into model.")
    logger.info("=" * 75)

    best_info = {
        "best_model_checkpoint": best_ckpt,
        "best_macro_f1": float(best_metric) if best_metric is not None else None,
        "total_epochs_trained": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
        "global_step": int(trainer.state.global_step),
        "target_epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "best_checkpoint_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    with open(os.path.join(log_dir, "best_checkpoint_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # 5. Evaluate and save
    eval_results = trainer.evaluate()
    logger.info("Final Evaluation Results on Loaded Best Checkpoint:")
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"Saving fine-tuned LoRA model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    try:
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Saved merged standalone model to: {merged_dir}")
    except Exception as e:
        logger.warning(f"Could not export merged model: {e}")

    label_map_file = os.path.join(output_dir, "label_mapping.json")
    with open(label_map_file, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    logger.info(f"Saved class mapping to: {label_map_file}")

    # 6. Extract and Save Structured Training History
    history_records = []
    for entry in trainer.state.log_history:
        record = {}
        if "step" in entry: record["step"] = entry["step"]
        if "epoch" in entry: record["epoch"] = entry["epoch"]
        if "loss" in entry: record["train_loss"] = entry["loss"]
        if "learning_rate" in entry: record["learning_rate"] = entry["learning_rate"]
        if "eval_loss" in entry: record["eval_loss"] = entry["eval_loss"]
        if "eval_accuracy" in entry: record["eval_accuracy"] = entry["eval_accuracy"]
        if "eval_f1" in entry: record["eval_macro_f1"] = entry["eval_f1"]
        if "eval_weighted_f1" in entry: record["eval_weighted_f1"] = entry["eval_weighted_f1"]
        if record:
            history_records.append(record)

    history_df = pd.DataFrame(history_records)
    history_csv = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    history_df.to_csv(os.path.join(log_dir, "training_history.csv"), index=False)
    logger.info(f"Saved training history metrics log to: {history_csv} and {log_dir}")

    # 7. Plot Training Curves (Loss, Eval Loss, Accuracy/F1, LR)
    train_points = history_df.dropna(subset=["train_loss"]).sort_values("step") if ("train_loss" in history_df.columns and "step" in history_df.columns) else pd.DataFrame()
    eval_points = history_df.dropna(subset=["eval_loss"]).sort_values("epoch") if ("eval_loss" in history_df.columns and "epoch" in history_df.columns) else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    if not train_points.empty:
        axes[0, 0].plot(train_points["step"], train_points["train_loss"], color="#3498db", alpha=0.45, label="Step Loss")
        smooth = train_points["train_loss"].rolling(window=min(5, len(train_points)), min_periods=1).mean()
        axes[0, 0].plot(train_points["step"], smooth, color="#1a5276", linewidth=2, label="Smoothed Trend")
        axes[0, 0].set_title("Training Loss vs Global Step", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Cross-Entropy Loss")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)
        axes[0, 0].legend()

    if not eval_points.empty:
        axes[0, 1].plot(eval_points["epoch"], eval_points["eval_loss"], marker="o", color="#e74c3c", linewidth=2)
        axes[0, 1].set_title("Validation Loss vs Epoch", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Eval Loss")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        axes[1, 0].plot(eval_points["epoch"], eval_points["eval_accuracy"] * 100, marker="s", color="#27ae60", linewidth=2, label="Accuracy (%)")
        axes[1, 0].plot(eval_points["epoch"], eval_points["eval_macro_f1"] * 100, marker="^", color="#8e44ad", linewidth=2, label="Macro F1 (%)")
        axes[1, 0].set_title("Validation Metrics vs Epoch", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score (%)")
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)
        axes[1, 0].legend()

    if not train_points.empty and "learning_rate" in train_points.columns:
        axes[1, 1].plot(train_points["step"], train_points["learning_rate"], color="#d35400", linewidth=2)
        axes[1, 1].set_title("Learning Rate Decay", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves plot to: {curves_path}")

    # 8. Detailed Classification Report & Per-Class Bar Chart
    logger.info("Generating predictions for classification report and confusion matrix...")
    predictions = trainer.predict(splits["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    present_labels = sorted(list(set(labels) | set(preds)))
    display_labels = [id2label[i] for i in present_labels if i in id2label]

    report_dict = classification_report(labels, preds, labels=present_labels, target_names=display_labels, output_dict=True, zero_division=0)
    report_text = classification_report(labels, preds, labels=present_labels, target_names=display_labels, zero_division=0)

    report_json_path = os.path.join(output_dir, "classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    report_txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(report_text)
    with open(os.path.join(log_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)
    logger.info(f"\nClassification Report:\n{report_text}")

    # Per-class Precision, Recall, F1 Bar Chart
    prec_vals = [report_dict[lbl]["precision"] for lbl in display_labels if lbl in report_dict]
    rec_vals = [report_dict[lbl]["recall"] for lbl in display_labels if lbl in report_dict]
    f1_vals = [report_dict[lbl]["f1-score"] for lbl in display_labels if lbl in report_dict]

    y_pos = np.arange(len(display_labels))
    bar_height = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_pos - bar_height, prec_vals, bar_height, label="Precision", color="#3498db")
    ax.barh(y_pos, rec_vals, bar_height, label="Recall", color="#2ecc71")
    ax.barh(y_pos + bar_height, f1_vals, bar_height, label="F1-Score", color="#9b59b6")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Reward: {l}" for l in display_labels], fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score", fontweight="bold")
    ax.set_title("Per-Class Precision, Recall, and F1-Score", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    per_class_path = os.path.join(output_dir, "per_class_metrics.png")
    plt.savefig(per_class_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-class performance plot to: {per_class_path}")

    # 9. Confusion Matrix & Plotting (Dual-Panel: Absolute Counts & Normalized Recall)
    all_class_indices = list(range(num_classes))
    all_display_labels = [id2label[i] for i in all_class_indices]

    cm = confusion_matrix(labels, preds, labels=all_class_indices)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = confusion_matrix(labels, preds, labels=all_class_indices, normalize='true')
        cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_display_labels)
    disp_counts.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=axes[0])
    axes[0].set_title('Absolute Counts', fontsize=12, fontweight="bold")

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=all_display_labels)
    disp_norm.plot(cmap=plt.cm.Greens, xticks_rotation=45, ax=axes[1], values_format='.2f')
    axes[1].set_title('Normalized Recall (Per Class)', fontsize=12, fontweight="bold")

    plt.suptitle('Confusion Matrix: Action-Level DeBERTa Reward Model', fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to: {cm_plot_path}")


def train_reward_model(config: Dict[str, Any]):
    train_cfg = config.get("rm_training", {})
    output_dir = train_cfg.get("output_dir", "./reward-model-lora-doorkey")
    log_dir = train_cfg.get("log_dir", "./logs/rm_training")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamped_log = os.path.join(log_dir, f"train_rm_{timestamp}.log")
    latest_log = os.path.join(log_dir, "train_rm_latest.log")
    output_log = os.path.join(output_dir, "training.log")

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handlers = [
        logging.FileHandler(timestamped_log, mode="a", encoding="utf-8"),
        logging.FileHandler(latest_log, mode="w", encoding="utf-8"),
        logging.FileHandler(output_log, mode="w", encoding="utf-8")
    ]
    for h in file_handlers:
        h.setLevel(logging.INFO)
        h.setFormatter(formatter)
        logger.addHandler(h)
        logging.getLogger().addHandler(h)

    hf_logging.set_verbosity_info()
    for h in file_handlers:
        hf_logging.add_handler(h)

    logger.info("=" * 75)
    logger.info("Logging initialized. Telemetry will be saved to:")
    logger.info(f"  • Run Log:    {timestamped_log}")
    logger.info(f"  • Latest Log: {latest_log}")
    logger.info(f"  • Model Log:  {output_log}")
    logger.info("=" * 75)

    try:
        _train_reward_model_impl(config, log_dir=log_dir, output_dir=output_dir)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during DeBERTa fine-tuning: {str(e)}", exc_info=True)
        err_file = os.path.join(log_dir, "error_summary.txt")
        try:
            with open(err_file, "w") as f:
                f.write(f"DeBERTa Fine-Tuning Failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n\nTraceback:\n")
                import traceback
                f.write(traceback.format_exc())
            logger.info(f"Error details saved to: {err_file}")
        except Exception:
            pass
        raise e
    finally:
        for h in file_handlers:
            try:
                h.flush()
                h.close()
                logger.removeHandler(h)
                logging.getLogger().removeHandler(h)
                hf_logging.remove_handler(h)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeBERTa Action-Level Reward Model")
    default_cfg = "configs/doorkey_config.yaml" if os.path.exists("configs/doorkey_config.yaml") else "configs/config.yaml"
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to config YAML")
    parser.add_argument("--data-path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory (e.g. ./logs/rm_training)")
    parser.add_argument("--model-name", type=str, default=None, help="Override base model name or path (e.g. microsoft/deberta-v3-small)")
    parser.add_argument("--sample-size", type=int, default=None, help="Override dataset sample size limit")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Override maximum saved checkpoints limit")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Override maximum sequence length")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    overrides = {}
    if args.data_path:
        overrides["dataset.data_path"] = args.data_path
    if args.model_name:
        overrides["reward_model.model_name"] = args.model_name
    if args.sample_size:
        overrides["dataset.sample_size"] = args.sample_size
    if args.output_dir:
        overrides["rm_training.output_dir"] = args.output_dir
    if args.log_dir:
        overrides["rm_training.log_dir"] = args.log_dir
    if args.epochs is not None:
        overrides["rm_training.epochs"] = args.epochs
    if args.early_stopping_patience is not None:
        overrides["rm_training.early_stopping_patience"] = args.early_stopping_patience
    if args.save_total_limit is not None:
        overrides["rm_training.save_total_limit"] = args.save_total_limit
    if args.batch_size is not None:
        overrides["rm_training.batch_size"] = args.batch_size
    if args.max_length is not None:
        overrides["rm_training.max_length"] = args.max_length
    if args.lr is not None:
        overrides["rm_training.learning_rate"] = args.lr

    cfg = load_config(args.config, overrides=overrides)
    train_reward_model(cfg)
