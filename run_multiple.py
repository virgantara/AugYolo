import subprocess
import numpy as np
from scipy.stats import ttest_rel

seeds = [42, 43, 44, 45, 46]
baseline_scores = []
our_scores = []


# Run BeyondRPC
print("\nRunning Baseline Yolov8 model...")
for seed in seeds:
    result = subprocess.check_output([
        "python", "main_cls.py",
        "--exp_name", f"exp_yolov8_ep300_600x600_sceG_{seed}",
        "--img_size", "600",
        "--model_name", "yolo",
        "--seed", str(seed),
        "--pretrain_path", "pretrain/yolov8n-cls.pt",
        "--epochs", "300",
        "--batch_size","16",
        "--test_batch_size","16"
    ])

    print(f"Yolobaseline_seed{seed} Result: ",result)
    # acc = float(result.decode().split("acc:")[-1].split(",")[0].strip())
    # beyondrpc_scores.append(acc)

print("\nRunning Our model...")
for seed in seeds:
    result = subprocess.check_output([
        "python", "main_cls.py",
        "--exp_name", f"exp_yolov8_ep300_600x600_sceG_{seed}",
        "--img_size", "600",
        "--model_name", "yolo",
        "--seed", str(seed),
        "--pretrain_path", "pretrain/yolov8n-cls.pt",
        "--epochs", "300",
        "--batch_size","16",
        "--test_batch_size","16",
        "--scenario","G",
        "--use_clahe",
        "--clahe_p","0.25",
        "--use_wavelet",
        "--wavelet_name","db2",
        "--wavelet_level","1",
        "--wavelet_p","1.0",
        "--use_unsharp",
        "--unsharp_amount","0.5",
        "--unsharp_radius","0.8",
        "--unsharp_threshold","2",
        "--unsharp_p","1.0"
        
    ])

    print(f"Ourmodel_seed{seed} Result: ",result)
# Convert to NumPy arrays
# rpc_scores = np.array(rpc_scores)
# beyondrpc_scores = np.array(beyondrpc_scores)

# # Run paired t-test
# t_stat, p_val = ttest_rel(rpc_scores, beyondrpc_scores)

# print("\n=== Paired t-test ===")
# print(f"RPC mean acc: {rpc_scores.mean():.4f}")
# print(f"BeyondRPC mean acc: {beyondrpc_scores.mean():.4f}")
# print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

# if p_val < 0.05:
#     print(" Statistically significant difference (p < 0.05)")
# else:
#     print(" No statistically significant difference")