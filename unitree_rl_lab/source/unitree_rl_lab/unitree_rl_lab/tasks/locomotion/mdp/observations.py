from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def foot_friction_4legs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """回傳每個 env 四隻腳各自的 [mu_s, mu_d, e]，最後攤平成 (N, 12)。

    預期 robot 的 body_names 類似：
    - "FL_foot"
    - "FR_foot"
    - "RL_foot"
    - "RR_foot"
    如果名稱不同，可以把分組邏輯改掉（例如用 contains "front_left_foot" 等等）。
    """
    device = env.device
    num_envs = env.num_envs

    # ===== 第一次呼叫時：建立「哪幾個 shape 屬於哪隻腳」的索引 =====
    if not hasattr(env, "_foot_shape_indices_by_leg"):
        robot = env.scene["robot"]

        # 1) 取得 body_names，找出四隻腳的 body_id
        try:
            body_names = list(robot.data.body_names)
        except AttributeError:
            body_names = list(robot.root_physx_view.body_names)

        # 將 *_foot 依照 "FL/FR/RL/RR" 分組
        leg_to_body_ids = {"FL": [], "FR": [], "RL": [], "RR": []}
        for i, name in enumerate(body_names):
            if not name.endswith("_foot"):
                continue
            # 取前綴當成腳的 label，例如 "FL_foot" -> "FL"
            leg_label = name.split("_")[0]
            if leg_label in leg_to_body_ids:
                leg_to_body_ids[leg_label].append(i)

        # 確認有抓到四隻腳
        for leg in ["FL", "FR", "RL", "RR"]:
            if len(leg_to_body_ids[leg]) == 0:
                raise RuntimeError(
                    f"[foot_friction_4legs] 找不到 {leg}_foot 對應的 body，實際 body_names = {body_names}"
                )

        # 2) 計算每個 body 有幾個 shapes（抄 IsaacLab randomize_rigid_body_material 的作法）
        num_shapes_per_body: list[int] = []
        for link_path in robot.root_physx_view.link_paths[0]:
            link_physx_view = robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)

        total_num_shapes = sum(num_shapes_per_body)
        expected_shapes = robot.root_physx_view.max_shapes
        if total_num_shapes != expected_shapes:
            raise RuntimeError(
                "[foot_friction_4legs] 解析 shape 數量不一致："
                f" sum(num_shapes_per_body)={total_num_shapes}, "
                f"root_physx_view.max_shapes={expected_shapes}"
            )

        # 3) 對每一隻腳，收集它名下所有 body 對應的 shape indices
        foot_shape_indices_by_leg = {}
        for leg, body_ids in leg_to_body_ids.items():
            indices = []
            for body_id in body_ids:
                start_idx = sum(num_shapes_per_body[:body_id])
                end_idx = start_idx + num_shapes_per_body[body_id]
                indices.extend(range(start_idx, end_idx))
            if len(indices) == 0:
                raise RuntimeError(f"[foot_friction_4legs] {leg} 沒有任何 shape")
            foot_shape_indices_by_leg[leg] = torch.as_tensor(
                indices, dtype=torch.long, device="cpu"
            )

        # cache 起來，之後就不用再算一次
        env._foot_shape_indices_by_leg = foot_shape_indices_by_leg

    # ===== 每次呼叫：從 PhysX 讀材質 → 只取腳底 → 分別算四隻腳平均 =====
    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()  # (N_env, max_shapes, 3) 在 CPU
    if not isinstance(materials, torch.Tensor):
        materials = torch.from_numpy(materials)

    feats_per_leg = []
    for leg in ["FL", "FR", "RL", "RR"]:
        idx = env._foot_shape_indices_by_leg[leg]     # shape_indices for this leg (CPU)
        leg_mats = materials[:, idx, :]               # (N_env, N_shapes_leg, 3)

        mu_s = leg_mats[:, :, 0].mean(dim=1).to(device)
        mu_d = leg_mats[:, :, 1].mean(dim=1).to(device)
        e    = leg_mats[:, :, 2].mean(dim=1).to(device)

        feats_per_leg.append(mu_s)
        feats_per_leg.append(mu_d)
        feats_per_leg.append(e)

    # feats_per_leg: list of 12 tensors，每個 shape = (N_env,)
    # → stack → (12, N_env) 再 transpose → (N_env, 12)
    feet_feat = torch.stack(feats_per_leg, dim=0).T
    return feet_feat  # (num_envs, 12)
