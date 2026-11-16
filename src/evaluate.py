"""
Evaluation and Visualization Script for Trained Agents.
Reproduces:
  1. Mean reward +/- std reward evaluation
  2. Action behavior statistics and anti-farming exploit analysis
  3. Trajectory visualization plots (Figure 6 in paper)
  4. Agent visit heatmap
  5. Training convergence curves from monitor.csv
"""

import os
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.config import load_config
from src.features_extractor import (
    MinigridFeaturesExtractor,
    MinigridSemanticSlotExtractor,
    MinigridMultimodalFiLMExtractor,
    MinigridMultimodalConcatExtractor
)
from src.wrappers_putnear import PutNearMultimodalWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_agent(config: Dict[str, Any]):
    eval_cfg = config.get("evaluation", {})
    env_cfg = config.get("environment", {})
    rl_cfg = config.get("rl_training", {})
    agent_type = config.get("agent", {}).get("type", "rm_ppo")

    model_path = eval_cfg.get("model_path", rl_cfg.get("save_model_path", f"./models/{agent_type}_agent"))
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        model_path = model_path + ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    env_id = env_cfg.get("env_id", "MiniGrid-DoorKey-8x8-v0")
    eval_max_steps = env_cfg.get("max_episode_steps", 200)
    output_dir = eval_cfg.get("output_dir", f"./eval_results/{agent_type}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading trained PPO model from: {model_path}...")
    model = PPO.load(model_path)

    # Base evaluation environment
    eval_env = gym.make(env_id, render_mode="rgb_array", max_steps=eval_max_steps, max_episode_steps=eval_max_steps)
    if isinstance(model.observation_space, gym.spaces.Dict):
        mission_shape = model.observation_space["mission"].shape
        mode = "slots" if mission_shape in [(4,), (5,)] else "tokens"
        logger.info(f"Model requires Dict observation space: applying PutNearMultimodalWrapper (mode={mode})...")
        eval_env = PutNearMultimodalWrapper(eval_env, mode=mode)
    else:
        logger.info("Model requires Box observation space: applying ImgObsWrapper...")
        eval_env = ImgObsWrapper(eval_env)

    # 1. Quantitative Evaluation
    n_eval_episodes = eval_cfg.get("num_eval_episodes", 10)
    logger.info(f"Evaluating policy over {n_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    logger.info(f"Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # 2. Behavioral Analysis (Exploit Detection)
    n_behavior_episodes = eval_cfg.get("num_behavior_analysis_episodes", 50)
    start_seed = eval_cfg.get("start_seed", None)
    logger.info(f"Analyzing agent behavior over {n_behavior_episodes} episodes (start_seed={start_seed})...")
    total_pick_ups = 0
    total_drops = 0
    total_toggles = 0
    episodes_with_exploit = 0

    for ep in range(n_behavior_episodes):
        reset_kwargs = {"seed": start_seed + ep} if start_seed is not None else {}
        obs, _ = eval_env.reset(**reset_kwargs)
        done = False
        pick_ups_this_ep = 0
        drops_this_ep = 0
        toggles_this_ep = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            act_val = int(action)
            if act_val == 3: pick_ups_this_ep += 1
            if act_val == 4: drops_this_ep += 1
            if act_val == 5: toggles_this_ep += 1

        is_putnear = "putnear" in env_id.lower()
        if is_putnear:
            if pick_ups_this_ep > 1 or drops_this_ep > 1 or toggles_this_ep > 0:
                episodes_with_exploit += 1
        else:
            if pick_ups_this_ep > 1 or toggles_this_ep > 1 or drops_this_ep > 0:
                episodes_with_exploit += 1

        total_pick_ups += pick_ups_this_ep
        total_drops += drops_this_ep
        total_toggles += toggles_this_ep

    logger.info("=" * 50)
    logger.info("BEHAVIOR STATISTICS:")
    logger.info(f"Total 'pick up' actions: {total_pick_ups} (avg: {total_pick_ups / n_behavior_episodes:.2f})")
    logger.info(f"Total 'drop' actions:    {total_drops} (avg: {total_drops / n_behavior_episodes:.2f})")
    logger.info(f"Total 'toggle' actions:  {total_toggles} (avg: {total_toggles / n_behavior_episodes:.2f})")
    logger.info(f"Episodes with farming pattern detected: {episodes_with_exploit}/{n_behavior_episodes}")
    logger.info("=" * 50)

    # 3. Qualitative Trajectory Plots (Figure 6 in paper)
    n_plots = eval_cfg.get("num_trajectory_plots", 5)
    logger.info(f"Generating {n_plots} qualitative trajectory plots...")
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    tile_size = 32
    for i in range(n_plots):
        obs, info = eval_env.reset(seed=100 + i)
        agent_path = [tuple(eval_env.unwrapped.agent_pos)]
        rewards = 0.0

        for _ in range(eval_max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            rewards += reward
            agent_path.append(tuple(eval_env.unwrapped.agent_pos))
            if terminated or truncated:
                break

        final_img = eval_env.render()
        ax = axes[i]
        ax.imshow(final_img)
        ax.set_title(f"Episode {i+1}\nReward: {rewards:.2f}")
        ax.axis("off")

        pixel_x = [p[0] * tile_size + tile_size / 2 for p in agent_path]
        pixel_y = [p[1] * tile_size + tile_size / 2 for p in agent_path]

        ax.plot(pixel_x, pixel_y, color='red', linestyle='-', linewidth=2, marker='o', markersize=3, alpha=0.8)
        ax.scatter(pixel_x[0], pixel_y[0], color='green', marker='X', s=70, label="Start" if i == 0 else "")
        ax.scatter(pixel_x[-1], pixel_y[-1], color='blue', marker='o', s=70, label="End" if i == 0 else "")

    if n_plots > 0:
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.tight_layout()
    traj_plot_path = os.path.join(output_dir, "trajectories.png")
    plt.savefig(traj_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved trajectory visualization to: {traj_plot_path}")

    # 4. Visit Heatmap
    n_heatmap = eval_cfg.get("num_heatmap_simulations", 100)
    logger.info(f"Running {n_heatmap} simulations for agent visit heatmap...")
    env_w = eval_env.unwrapped.width
    env_h = eval_env.unwrapped.height
    visit_counts = np.zeros((env_h, env_w), dtype=int)

    for i in range(n_heatmap):
        obs, _ = eval_env.reset(seed=200 + i)
        pos = eval_env.unwrapped.agent_pos
        visit_counts[pos[1], pos[0]] += 1

        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            pos = eval_env.unwrapped.agent_pos
            visit_counts[pos[1], pos[0]] += 1
            if terminated or truncated:
                break

    plt.figure(figsize=(7, 7))
    sns.heatmap(visit_counts, cmap='hot_r', annot=False, linewidths=0.5, linecolor='black')
    plt.title(f'Agent Visit Heatmap ({n_heatmap} Simulations)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    heatmap_path = os.path.join(output_dir, "visit_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    logger.info(f"Saved visit heatmap to: {heatmap_path}")

    # 5. Training Curves from monitor.csv (if exists)
    log_file = os.path.join(rl_cfg.get("log_dir", f"./logs/{agent_type}"), "monitor.csv")
    if os.path.exists(log_file):
        logger.info(f"Plotting training curves from '{log_file}'...")
        df_mon = pd.read_csv(log_file, skiprows=1)
        rolling_window = 50

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(df_mon['l'].cumsum(), df_mon['r'], alpha=0.3, color='steelblue', label='Episode Reward')
        if len(df_mon) >= rolling_window:
            rolling_r = df_mon['r'].rolling(window=rolling_window).mean()
            ax1.plot(df_mon['l'].cumsum(), rolling_r, color='darkblue', linewidth=2, label=f'{rolling_window}-Ep Rolling Mean')
        ax1.set_title('Episode Rewards During Training')
        ax1.set_xlabel('Cumulative Timesteps')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df_mon['l'].cumsum(), df_mon['l'], alpha=0.3, color='coral', label='Episode Length')
        if len(df_mon) >= rolling_window:
            rolling_l = df_mon['l'].rolling(window=rolling_window).mean()
            ax2.plot(df_mon['l'].cumsum(), rolling_l, color='darkred', linewidth=2, label=f'{rolling_window}-Ep Rolling Mean')
        ax2.set_title('Episode Lengths During Training')
        ax2.set_xlabel('Cumulative Timesteps')
        ax2.set_ylabel('Episode Length')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curves_path, dpi=300)
        plt.close()
        logger.info(f"Saved training curves to: {curves_path}")

    eval_env.close()
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Agent")
    default_cfg = "configs/doorkey_config.yaml" if os.path.exists("configs/doorkey_config.yaml") else "configs/config.yaml"
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to config YAML")
    parser.add_argument("--agent", type=str, default=None, choices=["rm_ppo", "baseline_ppo", "intrinsic_ppo", "reward_shaping_ppo"], help="Override agent type")
    parser.add_argument("--model-path", type=str, default=None, help="Override trained model path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seeds", "--episodes", dest="episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--start-seed", type=int, default=None, help="Starting random seed for evaluation rollouts")
    args = parser.parse_args()

    overrides = {}
    if args.agent:
        overrides["agent.type"] = args.agent
        if not args.model_path and args.config in ["configs/config.yaml", "configs/doorkey_config.yaml"]:
            overrides["evaluation.model_path"] = f"./models/{args.agent}_agent"
        if not args.output_dir and args.config in ["configs/config.yaml", "configs/doorkey_config.yaml"]:
            overrides["evaluation.output_dir"] = f"./eval_results/{args.agent}"
    if args.model_path:
        overrides["evaluation.model_path"] = args.model_path
    if args.output_dir:
        overrides["evaluation.output_dir"] = args.output_dir
    if args.episodes is not None:
        overrides["evaluation.num_eval_episodes"] = args.episodes
        overrides["evaluation.num_behavior_analysis_episodes"] = args.episodes
    if args.start_seed is not None:
        overrides["evaluation.start_seed"] = args.start_seed

    cfg = load_config(args.config, overrides=overrides)
    evaluate_agent(cfg)
