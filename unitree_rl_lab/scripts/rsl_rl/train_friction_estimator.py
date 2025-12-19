# train_friction_estimator_multi.py
#
# 使用收集好的 friction_dataset_0~4.npz（或單一 npz 檔）
# 來訓練一個 MLP，輸入 X (history feature)，輸出 Y (4 腳的 [mu_s, mu_d, e])

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# -----------------------------
# Dataset：支援多個 npz shard
# -----------------------------


class MultiShardFrictionDataset(Dataset):
    """多檔案 (shard) 合併的 Dataset。

    X_files: list of memmap arrays, 每個 shape: (Ni, D_in)
    Y_files: list of memmap arrays, 每個 shape: (Ni, 12)
    """

    def __init__(
        self,
        X_list: List[np.ndarray],
        Y_list: List[np.ndarray],
        mean: np.ndarray = None,
        std: np.ndarray = None,
        use_normalize: bool = True,
    ):
        assert len(X_list) == len(Y_list), "X_list, Y_list 長度不一致"

        self.X_list = X_list
        self.Y_list = Y_list
        self.use_normalize = use_normalize

        if use_normalize:
            assert mean is not None and std is not None, "使用 normalize 時必須提供 mean/std"
        self.mean = mean
        self.std = std

        # 計算每個 shard 的 offset，方便用 global index 對應
        lengths = [x.shape[0] for x in X_list]  # 每個 shard 的樣本數 Ni
        self.shard_offsets = np.cumsum([0] + lengths)  # [0, N0, N0+N1, ...]
        self.total_len = self.shard_offsets[-1]

    def __len__(self):
        return self.total_len

    def _locate_shard(self, idx: int) -> Tuple[int, int]:
        """把 global index 轉成 (shard_id, local_idx)。"""
        # searchsorted: 找到第一個 > idx 的 offset，再 -1 得到所在 shard
        shard_id = int(np.searchsorted(self.shard_offsets, idx, side="right") - 1)
        local_idx = idx - self.shard_offsets[shard_id]
        return shard_id, int(local_idx)

    def __getitem__(self, idx: int):
        shard_id, local_idx = self._locate_shard(idx)
        x = self.X_list[shard_id][local_idx]  # (D_in,)
        y = self.Y_list[shard_id][local_idx]  # (12,)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.use_normalize:
            x = (x - self.mean[0]) / self.std[0]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


# -----------------------------
# MLP 模型
# -----------------------------


class MLPFrictionEstimator(nn.Module):
    """簡單的 MLP：輸入 flatten 後的 history feature，輸出 12 維 friction 參數。"""

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


# -----------------------------
# Loss：三種參數分開算
# -----------------------------


def compute_group_losses(pred: torch.Tensor, target: torch.Tensor):
    """
    pred, target: (B, 12)，順序為 4 腳 * [mu_s, mu_d, e]

    回傳：
      total_loss: 三個 group MSE 的平均
      (loss_mu_s, loss_mu_d, loss_e): 各自的 MSE
    """
    B, D = pred.shape
    assert D == 12, f"輸出維度應為 12，實際為 {D}"
    pred_3 = pred.view(B, 4, 3)   # (B, 4, 3)
    tgt_3 = target.view(B, 4, 3)  # (B, 4, 3)

    mse = nn.MSELoss()

    loss_mu_s = mse(pred_3[:, :, 0], tgt_3[:, :, 0])
    loss_mu_d = mse(pred_3[:, :, 1], tgt_3[:, :, 1])
    loss_e = mse(pred_3[:, :, 2], tgt_3[:, :, 2])

    total_loss = (loss_mu_s + loss_mu_d + loss_e) / 3.0
    return total_loss, (loss_mu_s, loss_mu_d, loss_e)


# -----------------------------
# 輔助函式：載入多個 npz
# -----------------------------


def load_sharded_npz(data: str, data_prefix: str, num_shards: int):
    """
    如果 args.data 非空，則只載入單一檔案。
    否則依照 data_prefix + {0..num_shards-1}.npz 載入多個。
    """
    X_list = []
    Y_list = []

    if data:  # 單一檔案模式
        paths = [data]
    else:     # 多 shard 模式
        paths = [f"{data_prefix}{i}.npz" for i in range(num_shards)]

    print("[INFO] Loading datasets:")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到檔案: {p}")
        print(f"  - {p}")
        d = np.load(p, mmap_mode="r")
        X = d["X"]  # memmap
        Y = d["Y"]  # memmap (較小)
        X_list.append(X)
        Y_list.append(Y)

    return X_list, Y_list


def compute_mean_std_over_shards(X_list: List[np.ndarray], use_normalize: bool = True):
    """對所有 shard 的 X 做 streaming mean/std，避免一次吃爆記憶體。"""
    if not use_normalize:
        return None, None

    D_in = X_list[0].shape[1]
    sum_x = np.zeros(D_in, dtype=np.float64)
    sum_x2 = np.zeros(D_in, dtype=np.float64)
    total_N = 0

    block_size = 4096  # 一次讀幾筆樣本，可視 RAM 調整

    print("[INFO] Computing normalization stats (mean/std) over all shards...")
    for shard_idx, X in enumerate(X_list):
        N = X.shape[0]
        print(f"  - shard {shard_idx}: N = {N}")
        for start in range(0, N, block_size):
            end = min(N, start + block_size)
            block = X[start:end].astype(np.float64)  # (B, D_in)
            sum_x += block.sum(axis=0)
            sum_x2 += (block ** 2).sum(axis=0)
            total_N += (end - start)

    mean = (sum_x / total_N).astype(np.float32)[np.newaxis, :]   # (1, D_in)
    var = sum_x2 / total_N - mean[0].astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)[np.newaxis, :]  # (1, D_in)

    print("[INFO] Done. Example mean/std[0:5]:")
    print("  mean[0:5] =", mean[0, :5])
    print("  std[0:5]  =", std[0, :5])

    return mean, std


# -----------------------------
# Eval：在 val set 上算各 group 的 RMSE / MAE
# -----------------------------


def evaluate_on_val(model, val_loader, device):
    model.eval()
    mse_s = mse_d = mse_e = 0.0
    mae_s = mae_d = mae_e = 0.0
    cnt = 0  # 總的「腳數」= batch_size * 4

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)  # (B, 12)

            pred = model(xb)  # (B, 12)
            B = pred.shape[0]

            pred_3 = pred.view(B, 4, 3)
            tgt_3 = yb.view(B, 4, 3)

            diff = pred_3 - tgt_3
            # 每個 group 都是 (B, 4)
            diff_s = diff[:, :, 0]
            diff_d = diff[:, :, 1]
            diff_e = diff[:, :, 2]

            mse_s += (diff_s ** 2).sum().item()
            mse_d += (diff_d ** 2).sum().item()
            mse_e += (diff_e ** 2).sum().item()

            mae_s += diff_s.abs().sum().item()
            mae_d += diff_d.abs().sum().item()
            mae_e += diff_e.abs().sum().item()

            cnt += B * 4

    import math

    rmse_s = math.sqrt(mse_s / cnt)
    rmse_d = math.sqrt(mse_d / cnt)
    rmse_e = math.sqrt(mse_e / cnt)

    mae_s /= cnt
    mae_d /= cnt
    mae_e /= cnt

    print("\n[VAL] Per-parameter error (over all legs & samples):")
    print(f"  mu_s: RMSE = {rmse_s:.4f}, MAE = {mae_s:.4f}")
    print(f"  mu_d: RMSE = {rmse_d:.4f}, MAE = {mae_d:.4f}")
    print(f"  e   : RMSE = {rmse_e:.4f}, MAE = {mae_e:.4f}\n")


# -----------------------------
# Argparse
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train friction estimator from collected multi-shard dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="若提供，使用單一 npz 檔 (含 X, Y)。若留空，則使用 --data_prefix + [0..num_shards-1].npz。",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default="friction_dataset_",
        help="多 shard 資料前綴，例如 'friction_dataset_' 對應 0~num_shards-1。",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=5,
        help="shard 數量，使用 data_prefix0.npz ~ data_prefix(num_shards-1).npz。",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable input normalization.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="friction_estimator_ckpts",
        help="Directory to save model and stats.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device.")
    return parser.parse_args()


# -----------------------------
# main
# -----------------------------


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) 載入多個 npz 檔（X, Y 都用 memmap）
    X_list, Y_list = load_sharded_npz(args.data, args.data_prefix, args.num_shards)

    # 基本資訊
    D_in = X_list[0].shape[1]
    N_total = sum(x.shape[0] for x in X_list)
    print(f"[INFO] Total #samples = {N_total}, Input dim = {D_in}, Output dim = 12")

    # 2) 計算 normalization (mean/std)
    use_normalize = not args.no_normalize
    X_mean, X_std = compute_mean_std_over_shards(X_list, use_normalize=use_normalize)

    # 3) 建 Dataset + train/val split
    full_dataset = MultiShardFrictionDataset(
        X_list,
        Y_list,
        mean=X_mean,
        std=X_std,
        use_normalize=use_normalize,
    )

    indices = np.arange(N_total)
    np.random.shuffle(indices)
    split = int(N_total * (1.0 - args.val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    print(f"[INFO] #train = {len(train_idx)}, #val = {len(val_idx)}")

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4) 建 model + optimizer
    model = MLPFrictionEstimator(input_dim=D_in, output_dim=12, hidden_dims=(512, 512, 256))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print(f"[INFO] Total params: {sum(p.numel() for p in model.parameters())}")

    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    # 5) 訓練 loop
    for epoch in range(1, args.epochs + 1):
        # ---------- Train ----------
        model.train()
        train_loss_sum = 0.0
        train_ls_s = train_ls_d = train_ls_e = 0.0
        num_train_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss, (ls_s, ls_d, ls_e) = compute_group_losses(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_ls_s += ls_s.item()
            train_ls_d += ls_d.item()
            train_ls_e += ls_e.item()
            num_train_batches += 1

        train_loss = train_loss_sum / max(1, num_train_batches)
        train_ls_s /= max(1, num_train_batches)
        train_ls_d /= max(1, num_train_batches)
        train_ls_e /= max(1, num_train_batches)

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        val_ls_s = val_ls_d = val_ls_e = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss, (ls_s, ls_d, ls_e) = compute_group_losses(pred, yb)

                val_loss_sum += loss.item()
                val_ls_s += ls_s.item()
                val_ls_d += ls_d.item()
                val_ls_e += ls_e.item()
                num_val_batches += 1

        val_loss = val_loss_sum / max(1, num_val_batches)
        val_ls_s /= max(1, num_val_batches)
        val_ls_d /= max(1, num_val_batches)
        val_ls_e /= max(1, num_val_batches)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss = {train_loss:.6f} "
            f"(mu_s={train_ls_s:.6f}, mu_d={train_ls_d:.6f}, e={train_ls_e:.6f}); "
            f"val_loss = {val_loss:.6f} "
            f"(mu_s={val_ls_s:.6f}, mu_d={val_ls_d:.6f}, e={val_ls_e:.6f})"
        )

        # 存最佳 model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": D_in,
                    "output_dim": 12,
                    "mean": X_mean,
                    "std": X_std,
                },
                ckpt_path,
            )
            print(f"[INFO] Saved best model to: {ckpt_path} (val_loss={val_loss:.6f})")

    print(f"[INFO] Training finished. Best val_loss = {best_val_loss:.6f}")

    # 6) 用 best model 在 val set 上做一次更詳細的誤差統計
    if os.path.exists(ckpt_path):
        print(f"[INFO] Reload best model from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    evaluate_on_val(model, val_loader, device)


if __name__ == "__main__":
    main()

