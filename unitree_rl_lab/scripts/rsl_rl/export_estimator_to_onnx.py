# scripts/rsl_rl/export_estimator_to_onnx.py

import os
import argparse

import torch
import torch.nn as nn
import numpy as np

class MLPFrictionEstimator(nn.Module):
    """跟訓練時一樣的結構：input_dim -> 512 -> 512 -> 256 -> 12"""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Export friction estimator MLP to ONNX.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="friction_estimator_ckpts/best_model.pt",
        help="Path to trained estimator checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="friction_estimator_ckpts",
        help="Directory to save ONNX file.",
    )
    parser.add_argument(
        "--onnx_name",
        type=str,
        default="friction_estimator.onnx",
        help="ONNX file name.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = args.ckpt
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")

    # ⚠️ 這裡一定要 weights_only=False，不然就會出現你看到的 UnpicklingError
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = ckpt["mean"]   # shape: (input_dim,)
    std  = ckpt["std"]    # shape: (input_dim,)

    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std, dtype=np.float32)

    mean_path = os.path.join(args.out_dir, "estimator_mean.bin")
    std_path  = os.path.join(args.out_dir, "estimator_std.bin")
    mean.tofile(mean_path)
    std.tofile(std_path)

    print(f"[INFO] Saved mean to: {mean_path}")
    print(f"[INFO] Saved std  to: {std_path}")

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    print(f"[INFO] input_dim = {input_dim}, output_dim = {output_dim}")

    model = MLPFrictionEstimator(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 建一個假的輸入給 ONNX 匯出用
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    onnx_path = os.path.join(args.out_dir, args.onnx_name)
    print(f"[INFO] Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["X"],
        output_names=["Y"],
        dynamic_axes={
            "X": {0: "batch_size"},
            "Y": {0: "batch_size"},
        },
        opset_version=args.opset,
    )

    print("[INFO] Export finished.")
    print(f"[INFO] ONNX saved to: {onnx_path}")


if __name__ == "__main__":
    main()

