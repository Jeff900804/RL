#!/usr/bin/env python

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect friction dataset using a trained RSL-RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect friction dataset with an RSL-RL agent.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# dataset related args
parser.add_argument(
    "--output",
    type=str,
    default="friction_dataset.npz",
    help="Output .npz file path for collected dataset.",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=200,
    help="How many outer loops to run for data collection (不是 env 真正 episode，只是分段記數用)。",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Max steps per outer loop in data collection.",
)
parser.add_argument(
    "--history_len",
    type=int,
    default=25,
    help="Estimator 使用的歷史長度 K（步數）。",
)

# append RSL-RL cli arguments (checkpoint / experiment_name / load_run / load_checkpoint ...)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args (headless, renderer 等)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# 啟 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion import mdp


def get_base_env(env):
    """從 wrapper 逐層往下拿出 ManagerBasedRLEnv."""
    base_env = env
    for attr in ("env", "unwrapped"):
        if hasattr(base_env, attr):
            base_env = getattr(base_env, attr)
    return base_env


def collect_dataset(
    env: RslRlVecEnvWrapper,
    policy,
    output_path: str,
    num_episodes: int,
    max_steps_per_episode: int,
    history_len: int,
):
    """實際做資料收集的主程式。

    X: (N, history_len * feat_dim)
        feat_t = [ obs["estimator"], last_action ]
    Y: (N, 12)
        每一隻腳 (mu_s, mu_d, e)
    """
    # 取得底層 IsaacLab env
    base_env = get_base_env(env)

    # 先拿一次 RL obs，看型態（直接照 play 的用法）
    if version("rsl-rl-lib").startswith("2.3."):
        obs_rl, _ = env.get_observations()
    else:
        obs_rl = env.get_observations()

    # estimator 用的 obs（dict，有 "estimator" 這個 key）
    obs_all = base_env.observation_manager.compute()
    if "estimator" not in obs_all:
        raise RuntimeError(
            "obs_all 中找不到 key 'estimator'。\n"
            "請確認你有在 ObservationsCfg 裡定義 estimator 這個 ObsGroup，"
            "且裡面不要包含 foot_friction，因為 foot_friction 要當 label。"
        )

    device = base_env.device
    num_envs = base_env.num_envs

    est_obs = obs_all["estimator"]
    D_obs = est_obs.shape[-1]

    # action 維度從 env 的 action_space 拿
    if hasattr(env.action_space, "shape"):
        action_dim = env.action_space.shape[-1]
    else:
        raise RuntimeError("無法從 env.action_space 取得 action 維度")

    feat_dim = D_obs + action_dim
    print(f"[INFO] num_envs      = {num_envs}")
    print(f"[INFO] D_obs         = {D_obs}")
    print(f"[INFO] action_dim    = {action_dim}")
    print(f"[INFO] feat_dim      = {feat_dim} (= D_obs + action_dim)")
    print(f"[INFO] history_len K = {history_len} → X 單筆維度 = {history_len * feat_dim}")

    # 上一步 action（初始 0）
    last_action = torch.zeros(num_envs, action_dim, device=device, dtype=torch.float32)

    # 每個 env 各自的 K 步 feature history
    history = torch.zeros(num_envs, history_len, feat_dim, device=device, dtype=torch.float32)

    X_list = []
    Y_list = []
    total_samples = 0

    for ep in range(num_episodes):
        print(f"\n[INFO] ===== Collect Episode {ep + 1}/{num_episodes} =====")
        # 每一個 outer loop，把 history & last_action 清空
        history.zero_()
        last_action = torch.zeros(num_envs, action_dim, device=device, dtype=torch.float32)

        for t in range(max_steps_per_episode):
            # ====================== 1) 準備 estimator feature ======================
            # 這裡用 observation_manager.compute() 拿 "estimator"，
            # 會依照你在 EstimatorCfg 裡的 noise / clip / scale 做 corruption
            obs_all = base_env.observation_manager.compute()
            est_obs = obs_all["estimator"].to(device=device, dtype=torch.float32)  # (N, D_obs)

            # 單步 feature = [estimator obs, last_action]
            feat_t = torch.cat([est_obs, last_action], dim=-1)  # (N, feat_dim)

            # history 往前 shift 一格，把最新的放在最後一格
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = feat_t

            # history_length 之前先暖機，不收資料
            if t >= history_len:
                # X_t: K 步 flatten 成一條
                X_t = history.reshape(num_envs, history_len * feat_dim).detach().cpu().numpy()

                # Y_t: 當前每腳摩擦真值 (mu_s, mu_d, e)
                with torch.inference_mode():
                    friction = mdp.foot_friction_4legs(base_env)  # (N, 12)
                    Y_t = friction.detach().cpu().numpy()

                X_list.append(X_t)
                Y_list.append(Y_t)
                total_samples += num_envs

            # ====================== 2) RL policy 推 action ======================
            # **這裡非常關鍵：完全照 play.py 的方式，把 obs_rl 原封不動丟給 policy**
            with torch.inference_mode():
                actions = policy(obs_rl)  # 不要自己去拆 ["policy"]，交給 rsl-rl 處理

            # env.step：用 wrapper 的 env 來 step，維持跟 train/play 一致
            obs_rl, reward, terminated, truncated = env.step(actions)

            # 更新 last_action
            last_action = actions.clone()

        print(f"[INFO] Episode {ep + 1}: 累積樣本數 = {total_samples}")

    # ====================== 3) 全部疊起來存檔 ======================
    if len(X_list) == 0:
        raise RuntimeError("X_list 是空的，表示沒有任何樣本被收集到，請檢查 history_len / max_steps 等設定。")

    X = np.concatenate(X_list, axis=0)  # (N, history_len * feat_dim)
    Y = np.concatenate(Y_list, axis=0)  # (N, 12)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, X=X, Y=Y)

    print(f"\n[INFO] 儲存 dataset 到: {output_path}")
    print(f"[INFO] X 形狀: {X.shape} (N, {history_len * feat_dim})")
    print(f"[INFO] Y 形狀: {Y.shape} (N, 12)")
    print("[INFO] 資料收集完成。")


def main():
    """Main: 建立 env + 載入 checkpoint + 收集資料。"""
    # 解析 env config（沿用 play.py 的寫法，用 play_env_cfg_entry_point）
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 決定 checkpoint 路徑
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # 建立 Isaac 環境（跟 play.py 一樣）
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # 若是 multi-agent，就轉 single agent
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包成 RslRlVecEnvWrapper，讓 runner 可以直接吃
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 建立 runner 並載入學好的 policy
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)

    # 取得 inference 用的 policy 函式（已包 normalizer）
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 小小 print 一下 config（可有可無）
    print_dict(agent_cfg.to_dict(), nesting=4)

    # 進入資料收集
    collect_dataset(
        env=env,
        policy=policy,
        output_path=args_cli.output,
        num_episodes=args_cli.episodes,
        max_steps_per_episode=args_cli.max_steps,
        history_len=args_cli.history_len,
    )

    # 關閉環境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

