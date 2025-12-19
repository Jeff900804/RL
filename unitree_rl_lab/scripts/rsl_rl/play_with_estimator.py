import argparse
import os
import time
from importlib.metadata import version

from isaaclab.app import AppLauncher
import cli_args  # 這個沒問題，純 Python
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import Counter

# =========================
# 1) 先處理 argparse + 啟動 Isaac Sim
# =========================
parser = argparse.ArgumentParser(description="Play with RSL-RL agent and friction estimator.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)

parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# RSL-RL args
cli_args.add_rsl_rl_args(parser)
# AppLauncher args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# ✅ 一定要先啟動 AppLauncher，omni.* 才會存在
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================
# 2) 這裡之後才可以 import 會用到 omni / isaaclab 的東西
# =========================
import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
# ---- Estimator MLP 定義（跟你 train 的那隻一致） ----

class MLPFrictionEstimator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 12, hidden_dims=(512, 512, 256)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- 線上推論用的 Estimator runtime wrapper ----

class FrictionEstimatorRuntime:
    def __init__(
        self,
        ckpt_path: str,
        num_envs: int,
        obs_no_fric_dim: int = 45,   # obs 前 45 維（不含 foot_friction）
        action_dim: int = 12,        # policy action 維度
        history_len: int = 50,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs
        self.obs_no_fric_dim = obs_no_fric_dim
        self.action_dim = action_dim
        self.feat_dim = obs_no_fric_dim + action_dim  # 每一步 estimator feature 維度
        self.history_len = history_len

        # 讀取 estimator ckpt
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        input_dim = ckpt["input_dim"]    # 應該是 2850 (= history_len * feat_dim)
        output_dim = ckpt["output_dim"]  # 12
        self.mean = torch.as_tensor(ckpt["mean"], device=self.device, dtype=torch.float32)
        self.std = torch.as_tensor(ckpt["std"], device=self.device, dtype=torch.float32)

        self.model = MLPFrictionEstimator(input_dim=input_dim, output_dim=output_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # history buffer: (num_envs, K, feat_dim)
        self.hist = torch.zeros(num_envs, history_len, self.feat_dim, device=self.device)
        self.step_count = 0  # 用來判斷 history 是否已經填滿

    def reset(self, env_ids=None):
        """在 env reset 時重置對應 env 的 history。"""
        if env_ids is None:
            self.hist.zero_()
            self.step_count = 0
        else:
            self.hist[env_ids] = 0.0
            # 簡化做法：直接把 step_count 歸零，或維持全局也可以
            self.step_count = 0

    @torch.no_grad()
    def update_and_predict(self, obs_no_fric: torch.Tensor, last_action: torch.Tensor):
        """
        obs_no_fric: (num_envs, 45)
        last_action: (num_envs, 12)
        return:
            None (history 未滿) 或 (num_envs, 12) 的 friction estimate
        """
        # 檢查維度
        assert obs_no_fric.shape[0] == self.num_envs
        assert last_action.shape[0] == self.num_envs

        feat = torch.cat([obs_no_fric, last_action], dim=-1)  # (num_envs, 57)

        # 滾動 history：可以用 roll 或手動 index
        if self.step_count < self.history_len:
            # 還在填滿階段，直接寫在對應時間步
            self.hist[:, self.step_count, :] = feat
        else:
            # 已經滿了，往前推一格
            self.hist[:, :-1, :] = self.hist[:, 1:, :]
            self.hist[:, -1, :] = feat

        self.step_count += 1

        if self.step_count < self.history_len:
            # history 還沒滿，不做估計
            return None

        # flatten: (num_envs, K*feat_dim) = (num_envs, 2850)
        X = self.hist.reshape(self.num_envs, -1)
        # normalization
        X_norm = (X - self.mean) / self.std
        # 丟進 model
        mu_hat = self.model(X_norm)
        return mu_hat  # (num_envs, 12)

def get_policy_obs_tensor(obs):
    """把 env.get_observations() 回傳的東西，轉成給 policy 用的 torch.Tensor。

    env.get_observations() 可能是：
    - 直接就是 torch.Tensor  (舊版/某些 wrapper)
    - TensorDict / dict，裡面含有 'policy' 或 'obs' 之類的 key
    """
    if isinstance(obs, torch.Tensor):
        return obs

    # TensorDict / dict / 其他 mapping
    if hasattr(obs, "keys"):
        keys = list(obs.keys())
        # 優先找 'policy'
        if "policy" in keys:
            x = obs["policy"]
        elif "obs" in keys:
            x = obs["obs"]
        else:
            # 如果只有一個 key，就直接拿那個 value
            if len(keys) == 1:
                x = obs[keys[0]]
            else:
                raise RuntimeError(f"無法從 obs 中推 policy obs，keys={keys}")

        # 如果還是 TensorDict，就再檢查一次
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "keys"):
            # 再 fallback 一次：若有 'obs' key
            if "obs" in x.keys():
                return x["obs"]
            else:
                raise RuntimeError(f"Nested obs 結構太複雜，type={type(x)}, keys={list(x.keys())}")
        else:
            raise RuntimeError(f"未知 obs 子結構 type={type(x)}")

    raise TypeError(f"未知 obs 類型: {type(obs)}")
from rsl_rl.runners import OnPolicyRunner  # 確定上面有這行

class VecEpisodeCSVLoggerV3:
    """
    依照 unitree_rl_lab / isaaclab task 的 infos['log'] 格式：
      - Episode_Reward/*：每個 episode 的累積回報（通常只在 done 那一步有效）
      - Episode_Termination/*：每個 env 本回合是否因該原因結束（done 那一步有效）
      - Metrics/*：像 tracking error 這種可每 step 觀察的 metric（done/非 done 都可能有值）
    """

    def __init__(self, log_dir: str, dt: float, num_envs: int, reward_keys=None, metric_keys=None, term_keys=None):
        os.makedirs(log_dir, exist_ok=True)
        self.dt = float(dt)
        self.num_envs = int(num_envs)

        self.step_path = os.path.join(log_dir, "step_metrics.csv")
        self.ep_path   = os.path.join(log_dir, "episode_metrics.csv")

        self.step_f = open(self.step_path, "w", newline="")
        self.ep_f   = open(self.ep_path, "w", newline="")
        self.step_w = csv.writer(self.step_f)
        self.ep_w   = csv.writer(self.ep_f)

        # 你這個 task 的 keys（也可自行傳入客製）
        self.reward_keys = reward_keys or [
            "Episode_Reward/track_lin_vel_xy",
            "Episode_Reward/track_ang_vel_z",
            "Episode_Reward/feet_slide",
            "Episode_Reward/energy",
            "Episode_Reward/action_rate",
            "Episode_Reward/flat_orientation_l2",
            "Episode_Reward/undesired_contacts",
        ]
        self.metric_keys = metric_keys or [
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
        ]
        self.term_keys = term_keys or [
            "Episode_Termination/time_out",
            "Episode_Termination/base_contact",
            "Episode_Termination/bad_orientation",
        ]

        # per-env running
        self.ep_return = torch.zeros(self.num_envs, dtype=torch.float32)
        self.ep_len    = torch.zeros(self.num_envs, dtype=torch.int64)

        # header: step metrics（每步跨 env 平均）
        self.step_w.writerow([
            "wall_time", "step",
            "reward_mean", "reward_std", "reward_min", "reward_max",
            "reward_per_sec_mean",
            "done_count",
            "estimator_active_rate",
            "mu_abs_mean", "mu_abs_std",
            "mu_gt_abs_mean", "mu_gt_abs_std",
            "mu_l1_err_mean",
            # ===== NEW =====
            "mu_s_gt_mean", "mu_d_gt_mean", "e_gt_mean",
            "mu_s_hat_mean", "mu_d_hat_mean", "e_hat_mean",
            # =============
            *[k.replace("/", "_") + "_mean" for k in self.metric_keys],
            *[k.replace("/", "_") + "_std"  for k in self.metric_keys],
        ])

        # header: episode metrics（每次有 env done，把那批 episodes 統計寫一筆）
        self.ep_w.writerow([
            "wall_time", "step",
            "ended_env_count",
            "episode_return_mean", "episode_return_std", "episode_return_min", "episode_return_max",
            "episode_len_mean", "episode_len_std", "episode_len_min", "episode_len_max",
            "return_per_step_mean", "return_per_step_std",
            "return_per_sec_mean",  "return_per_sec_std",
            *[k.replace("/", "_") + "_mean" for k in self.reward_keys],
            *[k.replace("/", "_") + "_std"  for k in self.reward_keys],
            *[k.replace("/", "_") + "_count" for k in self.term_keys],
        ])

        self.step_f.flush()
        self.ep_f.flush()

    def _to_cpu_1d(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().flatten().to("cpu")
        return torch.as_tensor(x).flatten().to("cpu")

    def _get_log_tensor(self, infos, key: str):
        if not (isinstance(infos, dict) and "log" in infos and key in infos["log"]):
            return None

        v = infos["log"][key]

        # 轉成 1D CPU tensor
        if isinstance(v, torch.Tensor):
            t = v.detach().flatten().to("cpu")
        else:
            t = torch.as_tensor(np.array(v)).flatten().to("cpu")

        # ✅ 若是 scalar 或 (1,)：代表已聚合，broadcast 成 (num_envs,)
        if t.numel() == 1:
            t = t.repeat(self.num_envs)

        # ✅ 若維度不是 num_envs：保守處理（避免再炸）
        if t.numel() != self.num_envs:
            # 你也可以改成 raise，讓你直接知道是哪個 key 不合
            # raise RuntimeError(f"log[{key}] has shape {t.shape}, expected ({self.num_envs},)")
            t = t[:1].repeat(self.num_envs)

        return t

    def log_step(
        self,
        step: int,
        rewards,
        dones,
        infos,
        estimator_active=None,
        mu_hat=None,
        mu_gt=None,

        # ===== NEW =====
        mu_s_gt_mean=0.0,
        mu_d_gt_mean=0.0,
        e_gt_mean=0.0,
        mu_s_hat_mean=0.0,
        mu_d_hat_mean=0.0,
        e_hat_mean=0.0,
        # =============
    ):
        wall = time.time()

        r = self._to_cpu_1d(rewards).float()          # (N,)
        d = self._to_cpu_1d(dones).bool()             # (N,)
        done_count = int(d.sum().item())

        # update running episode stats（用 env.step 回來的總 reward）
        self.ep_return += r
        self.ep_len += 1

        # ===== step metrics =====
        reward_mean = float(r.mean().item())
        reward_std  = float(r.std(unbiased=False).item())
        reward_min  = float(r.min().item())
        reward_max  = float(r.max().item())
        reward_per_sec_mean = reward_mean / self.dt if self.dt > 0 else 0.0

        # estimator active rate
        if estimator_active is None:
            active_rate = 0.0
        elif isinstance(estimator_active, torch.Tensor):
            active_rate = float(estimator_active.detach().flatten().to("cpu").float().mean().item())
        else:
            active_rate = float(estimator_active)

        # mu stats（跨 env 平均）
        def mu_stats(mu):
            if (mu is None) or (not isinstance(mu, torch.Tensor)):
                return (0.0, 0.0)
            x = mu.detach().to("cpu").abs().mean(dim=-1)  # (N,) 每隻狗的 |mu| 平均
            return float(x.mean().item()), float(x.std(unbiased=False).item())

        mu_abs_mean, mu_abs_std = mu_stats(mu_hat)
        mu_gt_abs_mean, mu_gt_abs_std = mu_stats(mu_gt)

        # mu l1 error（如果兩者都有）
        mu_l1_err_mean = 0.0
        if isinstance(mu_hat, torch.Tensor) and isinstance(mu_gt, torch.Tensor):
            err = (mu_hat.detach().to("cpu") - mu_gt.detach().to("cpu")).abs().mean(dim=-1)  # (N,)
            mu_l1_err_mean = float(err.mean().item())

        # metrics (error_vel_xy / error_vel_yaw)
        metric_means = []
        metric_stds  = []
        for k in self.metric_keys:
            v = self._get_log_tensor(infos, k)
            if v is None or v.numel() == 0:
                metric_means.append(0.0)
                metric_stds.append(0.0)
            else:
                metric_means.append(float(v.mean().item()))
                metric_stds.append(float(v.std(unbiased=False).item()))

        self.step_w.writerow([
            wall, int(step),
            reward_mean, reward_std, reward_min, reward_max,
            reward_per_sec_mean,
            done_count,
            active_rate,
            mu_abs_mean, mu_abs_std,
            mu_gt_abs_mean, mu_gt_abs_std,
            mu_l1_err_mean,
            # ===== NEW (寫你從 main 傳進來的數值) =====
            mu_s_gt_mean, mu_d_gt_mean, e_gt_mean,
            mu_s_hat_mean, mu_d_hat_mean, e_hat_mean,
            # ==========================================
            *metric_means,
            *metric_stds,
        ])

        # ===== episode metrics（只在 done 的 env 上取 Episode_Reward/ & Episode_Termination/）=====
        if done_count > 0:
            done_idx = d.nonzero(as_tuple=False).squeeze(-1)  # CPU indices

            ended_returns = self.ep_return[done_idx].clone().numpy()
            ended_lens    = self.ep_len[done_idx].clone().numpy()

            # return_per_step / per_sec
            rps = ended_returns / np.maximum(ended_lens, 1)
            rsec = ended_returns / np.maximum(ended_lens * self.dt, 1e-9)

            # reward breakdown：用 infos['log']['Episode_Reward/...'] 在 done env 上取值
            rb_means = []
            rb_stds  = []
            for k in self.reward_keys:
                v = self._get_log_tensor(infos, k)
                if v is None:
                    rb_means.append(0.0); rb_stds.append(0.0)
                    continue
                vv = v[done_idx].numpy()
                rb_means.append(float(vv.mean()))
                rb_stds.append(float(vv.std(ddof=0)))

            # term counts：直接用 Episode_Termination/* 在 done env 上計數
            term_counts = []
            for k in self.term_keys:
                v = self._get_log_tensor(infos, k)
                if v is None:
                    term_counts.append(0)
                    continue
                # 只統計 done 的那些 env
                vv = v[done_idx]
                term_counts.append(int((vv > 0.5).sum().item()))

            self.ep_w.writerow([
                wall, int(step),
                int(done_count),
                float(ended_returns.mean()), float(ended_returns.std(ddof=0)), float(ended_returns.min()), float(ended_returns.max()),
                float(ended_lens.mean()),    float(ended_lens.std(ddof=0)),    int(ended_lens.min()),     int(ended_lens.max()),
                float(rps.mean()), float(rps.std(ddof=0)),
                float(rsec.mean()), float(rsec.std(ddof=0)),
                *rb_means,
                *rb_stds,
                *term_counts,
            ])

            # reset finished env buffers
            self.ep_return[done_idx] = 0.0
            self.ep_len[done_idx] = 0

        # flush
        if (step % 200) == 0:
            self.step_f.flush()
            self.ep_f.flush()

    def close(self):
        try:
            self.step_f.flush()
            self.ep_f.flush()
        except Exception:
            pass
        self.step_f.close()
        self.ep_f.close()
                
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] No published checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_with_estimator"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    num_envs = env.num_envs
    print(f"[INFO] num_envs = {num_envs}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = float(getattr(env_cfg, "sim", None).dt * env_cfg.decimation) if hasattr(env_cfg, "decimation") else float(env.unwrapped.step_dt)

    # reset / get obs
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    # quick check
    print("infos keys preview will be printed at step=0")

    action_dim = 12
    # logger
    # csv_dir = os.path.join(log_dir, "csv")
    # logger = VecEpisodeCSVLoggerV3(log_dir=csv_dir, dt=dt, num_envs=num_envs)
    # print(f"[INFO] CSV logging to: {csv_dir}")

    # estimator
    friction_ckpt_path = os.path.join("friction_estimator_ckpts", "best_model.pt")
    print(f"[INFO] Loading friction estimator from: {friction_ckpt_path}")
    estimator = FrictionEstimatorRuntime(
        ckpt_path=friction_ckpt_path,
        num_envs=num_envs,
        obs_no_fric_dim=45,
        action_dim=action_dim,
        history_len=50,
        device=args_cli.device,
    )

    last_action = torch.zeros(num_envs, action_dim, device=env.unwrapped.device)

    step = 0
    timestep = 0

    while simulation_app.is_running():
        obs_policy = obs["policy"]                # (N,57)
        obs_no_fric = obs_policy[:, :45]          # (N,45)
        mu_gt = obs_policy[:, 45:]                # (N,12) 這是你 env 裡的 gt friction

        mu_hat = estimator.update_and_predict(obs_no_fric, last_action)  # (N,12) or None
        estimator_active = 1.0 if (mu_hat is not None) else 0.0
        
        # ===== NEW: mu_gt / mu_hat 拆成 (mu_s, mu_d, e) 的 mean =====
        # mu_gt: (N,12) -> (N,4,3)
        mu_gt_legs = mu_gt.view(-1, 4, 3)
        mu_s_gt_mean = float(mu_gt_legs[:, :, 0].mean().item())
        mu_d_gt_mean = float(mu_gt_legs[:, :, 1].mean().item())
        e_gt_mean    = float(mu_gt_legs[:, :, 2].mean().item())

        if mu_hat is not None:
            mu_hat_legs = mu_hat.view(-1, 4, 3)
            mu_s_hat_mean = float(mu_hat_legs[:, :, 0].mean().item())
            mu_d_hat_mean = float(mu_hat_legs[:, :, 1].mean().item())
            e_hat_mean    = float(mu_hat_legs[:, :, 2].mean().item())
        else:
            mu_s_hat_mean = 0.0
            mu_d_hat_mean = 0.0
            e_hat_mean    = 0.0
        # ============================================================

        if mu_hat is not None:
            obs_for_policy = obs.clone()
            obs_for_policy["policy"] = torch.cat([obs_no_fric, mu_hat], dim=-1)
        else:
            obs_for_policy = obs

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs_for_policy)
            last_action = actions.clone()
            obs, rewards, dones, infos = env.step(actions)

        if step == 0:
            print("infos keys:", list(infos.keys()) if isinstance(infos, dict) else type(infos))
            if isinstance(infos, dict) and "log" in infos:
                print("infos['log'] keys:", list(infos["log"].keys()))

        # ✅ log（重點：Episode_Reward/ & Episode_Termination/ 只在 done 那批 env 取）
        # logger.log_step(
        #     step=step,
        #     rewards=rewards,
        #     dones=dones,
        #     infos=infos,
        #     estimator_active=estimator_active,
        #     mu_hat=mu_hat,
        #     mu_gt=mu_gt,

        #     # ===== NEW =====
        #     mu_s_gt_mean=mu_s_gt_mean,
        #     mu_d_gt_mean=mu_d_gt_mean,
        #     e_gt_mean=e_gt_mean,
        #     mu_s_hat_mean=mu_s_hat_mean,
        #     mu_d_hat_mean=mu_d_hat_mean,
        #     e_hat_mean=e_hat_mean,
        #     # =============
        # )

        step += 1

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    # logger.close()
    simulation_app.close()
    


if __name__ == "__main__":
    main()

