import numpy as np

path = "friction_dataset.npz"  # 換成你實際檔名
data = np.load(path)

print("keys in file:", list(data.keys()))

X = data["X"]
Y = data["Y"]

print("X shape:", X.shape)  # (N, 2850) 之類
print("Y shape:", Y.shape)  # (N, 12)
print()

print("X dtype:", X.dtype)
print("Y dtype:", Y.dtype)


# 先取前幾筆來看
n_show = 5

print("First few Y rows (flat):")
print(Y[:n_show])

# 轉成 (N, 4, 3)： N 筆樣本、4 隻腳、每隻腳 3 個數值 (mu_s, mu_d, e)
Y_legs = Y.reshape(-1, 4, 3)

print("\nFirst sample per-leg friction (shape:", Y_legs[0].shape, ")")
print("格式： [ [mu_s, mu_d, e] for 4 legs ]")
print(Y_legs[0])      # 第 0 筆：4 隻腳 × 3 參數

print("\n第二筆：")
print(Y_legs[1])

# 看一下整體統計
mu_s_all = Y_legs[:, :, 0]
mu_d_all = Y_legs[:, :, 1]
e_all    = Y_legs[:, :, 2]

print("\n=== Stats over entire dataset ===")
print("mu_s: min = %.3f, max = %.3f, mean = %.3f" % (mu_s_all.min(), mu_s_all.max(), mu_s_all.mean()))
print("mu_d: min = %.3f, max = %.3f, mean = %.3f" % (mu_d_all.min(), mu_d_all.max(), mu_d_all.mean()))
print("e   : min = %.3f, max = %.3f, mean = %.3f" % (e_all.min(), e_all.max(), e_all.mean()))

