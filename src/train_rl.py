"""
Standalone RL Training Script for PPO on MiniGrid-DoorKey-8x8.
Supports:
  - rm_ppo:              Action-Level DeBERTa Reward Model
  - baseline_ppo:        Sparse Environment Reward Only
  - intrinsic_ppo:       Count-Based Novelty Reward
  - reward_shaping_ppo:  Heuristic Milestone Reward Shaping Baseline
"""

import os
import time
import argparse
import logging
from typing import Dict, Any, Callable

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch

from src.config import load_config
from src.features_extractor import (
    MinigridFeaturesExtractor,
    MinigridSemanticSlotExtractor,
    MinigridMultimodalFiLMExtractor,
    MinigridMultimodalConcatExtractor
)
from src.rm_model import ActionLevelRewardModel
from src.wrappers import (
    DebertaRewardWrapper,
    IntrinsicMotivationWrapper,
    RewardShapingWrapper
)
from src.wrappers_putnear import (
    PutNearDebertaRewardWrapper,
    PutNearRewardShapingWrapper,
    PutNearMultimodalWrapper,
    PutNearTokenizer
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float, min_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate decay schedule from initial_value down to min_value.
    progress_remaining starts at 1.0 (start of training) and anneals to 0.0 (end of training).
    """
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)
    return func


def build_environment(config: Dict[str, Any]) -> gym.Env:
    agent_type = config.get("agent", {}).get("type", "rm_ppo").lower()
    env_cfg = config.get("environment", {})
    rl_cfg = config.get("rl_training", {})

    env_id = env_cfg.get("env_id", "MiniGrid-DoorKey-8x8-v0")
    max_steps = env_cfg.get("max_episode_steps", 200)
    render_mode = env_cfg.get("render_mode", "rgb_array")
    log_dir = rl_cfg.get("log_dir", f"./logs/{agent_type}")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Creating base Gymnasium environment '{env_id}' (max_steps={max_steps})...")
    raw_env = gym.make(env_id, render_mode=render_mode, max_steps=max_steps, max_episode_steps=max_steps)

    is_putnear = "putnear" in env_id.lower()

    if agent_type == "rm_ppo":
        rm_cfg = config.get("reward_model", {})
        device = config.get("experiment", {}).get("device", "auto")
        logger.info(f"Initializing DeBERTa Action-Level Reward Model (device={device})...")
        rm = ActionLevelRewardModel(
            model_name=rm_cfg.get("model_name", "microsoft/deberta-v3-small"),
            lora_path=rm_cfg.get("lora_path", "reward-model-lora-doorkey"),
            device=device,
            use_cache=rm_cfg.get("use_cache", True),
            cache_size=rm_cfg.get("cache_size", 10000)
        )
        if is_putnear:
            logger.info("Applying PutNearDebertaRewardWrapper...")
            env = PutNearDebertaRewardWrapper(
                raw_env,
                reward_model=rm,
                step_penalty=rm_cfg.get("step_penalty", -0.02),
                collision_penalty=rm_cfg.get("collision_penalty", -0.25),
                ineffective_penalty=rm_cfg.get("ineffective_penalty", -0.25),
                anti_farming_penalty=rm_cfg.get("anti_farming_penalty", -2.50),
                wrong_pickup_penalty=rm_cfg.get("wrong_pickup_penalty", -3.50),
                active_rejection_bonus=rm_cfg.get("active_rejection_bonus", 0.05),
                max_stall_steps=rm_cfg.get("max_stall_steps", 0),
                selective_eval=rm_cfg.get("selective_eval", False),
                early_term_on_penalty=rm_cfg.get("early_term_on_penalty", False),
                include_env_oracle=rm_cfg.get("include_env_oracle", False)
            )
        else:
            env = DebertaRewardWrapper(
                raw_env,
                reward_model=rm,
                step_penalty=rm_cfg.get("step_penalty", -0.02),
                collision_penalty=rm_cfg.get("collision_penalty", -0.25),
                ineffective_penalty=rm_cfg.get("ineffective_penalty", -0.25),
                max_stall_steps=rm_cfg.get("max_stall_steps", 0),
                selective_eval=rm_cfg.get("selective_eval", False),
                early_term_on_penalty=rm_cfg.get("early_term_on_penalty", False),
                include_env_oracle=rm_cfg.get("include_env_oracle", False)
            )

    elif agent_type == "intrinsic_ppo":
        im_cfg = config.get("intrinsic_motivation", {})
        beta = float(im_cfg.get("beta", 0.05))
        count_type = im_cfg.get("count_type", "state_action")
        logger.info(f"Applying IntrinsicMotivationWrapper (beta={beta}, count_type={count_type})...")
        env = IntrinsicMotivationWrapper(raw_env, beta=beta, count_type=count_type)

    elif agent_type == "reward_shaping_ppo":
        rs_cfg = config.get("reward_shaping", {})
        logger.info("Applying heuristic RewardShapingWrapper baseline...")
        if is_putnear:
            env = PutNearRewardShapingWrapper(
                raw_env,
                pickup_reward=rs_cfg.get("key_pickup_reward", 2.0),
                goal_reward=rs_cfg.get("goal_reward", 10.0),
                step_penalty=rs_cfg.get("step_penalty", -0.02),
                collision_penalty=rs_cfg.get("collision_penalty", -0.25),
                anti_farming_penalty=rs_cfg.get("anti_farming_penalty", -2.50),
                wrong_pickup_penalty=rs_cfg.get("wrong_pickup_penalty", -3.50),
                active_rejection_bonus=rs_cfg.get("active_rejection_bonus", 0.05),
                ineffective_penalty=rs_cfg.get("ineffective_penalty", -0.25),
                max_stall_steps=rs_cfg.get("max_stall_steps", 0),
                enable_anti_farming=rs_cfg.get("enable_anti_farming", False)
            )
        else:
            env = RewardShapingWrapper(
                raw_env,
                key_pickup_reward=rs_cfg.get("key_pickup_reward", 2.0),
                door_open_reward=rs_cfg.get("door_open_reward", 3.0),
                goal_reward=rs_cfg.get("goal_reward", 10.0),
                step_penalty=rs_cfg.get("step_penalty", -0.02),
                collision_penalty=rs_cfg.get("collision_penalty", -0.25),
                anti_farming_penalty=rs_cfg.get("anti_farming_penalty", -2.5),
                ineffective_penalty=rs_cfg.get("ineffective_penalty", -0.25),
                max_stall_steps=rs_cfg.get("max_stall_steps", 0),
                enable_anti_farming=rs_cfg.get("enable_anti_farming", False)
            )

    elif agent_type == "baseline_ppo":
        logger.info("Using baseline sparse reward (no shaping wrapper).")
        env = raw_env

    else:
        raise ValueError(f"Unknown agent type '{agent_type}'. Choose from: rm_ppo, baseline_ppo, intrinsic_ppo, reward_shaping_ppo")

    # Observation wrapping: PutNear uses multimodal (image + mission), DoorKey uses ImgObsWrapper
    policy_cfg = config.get("policy", {})
    is_multimodal = policy_cfg.get("is_multimodal", is_putnear)
    extractor_type = policy_cfg.get("extractor", "slot").lower()

    if is_multimodal:
        mode = "tokens" if extractor_type in ["film", "concat"] else "slots"
        logger.info(f"Applying PutNearMultimodalWrapper (mode={mode})...")
        env = PutNearMultimodalWrapper(env, mode=mode)
    else:
        logger.info("Applying ImgObsWrapper (Box observation: image only)...")
        env = ImgObsWrapper(env)

    env = Monitor(env, log_dir)
    return env


def train_rl_agent(config: Dict[str, Any]):
    agent_type = config.get("agent", {}).get("type", "rm_ppo")
    env_cfg = config.get("environment", {})
    rl_cfg = config.get("rl_training", {})
    exp_cfg = config.get("experiment", {})
    policy_cfg = config.get("policy", {})

    env_id = env_cfg.get("env_id", "MiniGrid-DoorKey-8x8-v0")
    is_putnear = "putnear" in env_id.lower()
    is_multimodal = policy_cfg.get("is_multimodal", is_putnear)

    total_timesteps = int(rl_cfg.get("total_timesteps", 1000000))
    learning_rate = float(rl_cfg.get("learning_rate", 3e-4))
    min_learning_rate = float(rl_cfg.get("min_learning_rate", 1e-4))
    lr_schedule = rl_cfg.get("lr_schedule", "linear")
    gamma = float(rl_cfg.get("gamma", 0.999))
    n_steps = int(rl_cfg.get("n_steps", 2048))
    batch_size = int(rl_cfg.get("batch_size", 256))
    n_epochs = int(rl_cfg.get("n_epochs", 10))
    clip_range = float(rl_cfg.get("clip_range", 0.2))
    ent_coef = float(rl_cfg.get("ent_coef", 0.01))
    features_dim = int(rl_cfg.get("features_dim", 128))
    log_dir = rl_cfg.get("log_dir", f"./logs/{agent_type}")
    save_path = rl_cfg.get("save_model_path", f"./models/{agent_type}_agent")
    seed = exp_cfg.get("seed", 42)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = build_environment(config)

    if is_multimodal:
        policy_name = policy_cfg.get("policy_type", "MultiInputPolicy")
        extractor_type = policy_cfg.get("extractor", "slot").lower()

        if extractor_type == "slot":
            extractor_cls = MinigridSemanticSlotExtractor
            extractor_kwargs = dict(features_dim=features_dim)
        elif extractor_type == "film":
            extractor_cls = MinigridMultimodalFiLMExtractor
            text_dim = int(policy_cfg.get("text_dim", 64))
            extractor_kwargs = dict(features_dim=features_dim, text_dim=text_dim)
        elif extractor_type == "concat":
            extractor_cls = MinigridMultimodalConcatExtractor
            text_dim = int(policy_cfg.get("text_dim", 64))
            extractor_kwargs = dict(features_dim=features_dim, text_dim=text_dim)
        else:
            raise ValueError(f"Unknown extractor '{extractor_type}'. Choose 'slot', 'film', or 'concat'.")

        policy_kwargs = dict(
            features_extractor_class=extractor_cls,
            features_extractor_kwargs=extractor_kwargs,
            normalize_images=False,
        )
        logger.info(f"Policy: {policy_name} | Extractor: {extractor_type} (features_dim={features_dim})")
    else:
        policy_name = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            normalize_images=False,
        )
        logger.info(f"Policy: {policy_name} | Extractor: MinigridFeaturesExtractor (features_dim={features_dim})")

    if lr_schedule == "linear":
        lr_value = linear_schedule(learning_rate, min_value=min_learning_rate)
    else:
        lr_value = learning_rate

    logger.info("Initializing PPO agent...")
    model = PPO(
        policy_name,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        ent_coef=ent_coef,
        learning_rate=lr_value,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        seed=seed,
        device=exp_cfg.get("device", "auto")
    )

    logger.info(f"Starting training for {total_timesteps:,} timesteps on {agent_type}...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)!")

    model.save(save_path)
    logger.info(f"Saved trained agent to: {save_path}.zip")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent on MiniGrid DoorKey")
    default_cfg = "configs/doorkey_config.yaml" if os.path.exists("configs/doorkey_config.yaml") else "configs/config.yaml"
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to config YAML")
    parser.add_argument("--agent", type=str, default=None, choices=["rm_ppo", "baseline_ppo", "intrinsic_ppo", "reward_shaping_ppo"], help="Override agent type")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--save-path", type=str, default=None, help="Override saved model path")
    parser.add_argument("--lora-path", type=str, default=None, help="Override LoRA adapter path for RM")
    parser.add_argument("--model-name", type=str, default=None, help="Override base RM model name")
    parser.add_argument("--device", type=str, default=None, help="Override device (auto, mps, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--min-lr", type=float, default=None, help="Override minimum learning rate threshold")
    parser.add_argument("--ent-coef", type=float, default=None, help="Override entropy coefficient")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    overrides = {}
    if args.agent:
        overrides["agent.type"] = args.agent
        if not args.log_dir and args.config in ["configs/config.yaml", "configs/doorkey_config.yaml"]:
            overrides["rl_training.log_dir"] = f"./logs/{args.agent}"
        if not args.save_path and args.config in ["configs/config.yaml", "configs/doorkey_config.yaml"]:
            overrides["rl_training.save_model_path"] = f"./models/{args.agent}_agent"
    if args.timesteps:
        overrides["rl_training.total_timesteps"] = args.timesteps
    if args.lr is not None:
        overrides["rl_training.learning_rate"] = args.lr
    if args.min_lr is not None:
        overrides["rl_training.min_learning_rate"] = args.min_lr
    if args.ent_coef is not None:
        overrides["rl_training.ent_coef"] = args.ent_coef
    if args.save_path:
        overrides["rl_training.save_model_path"] = args.save_path
    if args.lora_path:
        overrides["reward_model.lora_path"] = args.lora_path
    if args.model_name:
        overrides["reward_model.model_name"] = args.model_name
    if args.device:
        overrides["experiment.device"] = args.device
    if args.seed:
        overrides["experiment.seed"] = args.seed
    if args.log_dir:
        overrides["rl_training.log_dir"] = args.log_dir

    cfg = load_config(args.config, overrides=overrides)
    train_rl_agent(cfg)
