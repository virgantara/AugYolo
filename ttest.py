from scipy.stats import ttest_rel
import numpy as np
baseline_scores = np.array([
	0.813,0.8452,0.8278,0.8658,0.8339
	])
our_scores = np.array([
	0.8557,0.8557,0.8557,0.8356,0.8557
	])

t_stat, p_val = ttest_rel(baseline_scores, our_scores)
n_repeats = 5

std_baseline_scores = baseline_scores.std()
ci95_baseline = 1.96 * (std_baseline_scores / np.sqrt(n_repeats))

std_our_scores = our_scores.std()
ci95_our = 1.96 * (std_our_scores / np.sqrt(n_repeats))

print("\n=== Paired t-test ===")
print(f"Baseline mean: {baseline_scores.mean():.4f}")
print(f"Baseline std: {baseline_scores.std():.4f}")
print(f"CI95: {ci95_baseline:.4f}")
print(f"AugYolo mean: {our_scores.mean():.4f}")
print(f"AugYolo Std: {our_scores.std():.4f}")
print(f"CI95: {ci95_our:.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

if p_val < 0.05:
    print(" Statistically significant difference (p < 0.05)")
else:
    print(" No statistically significant difference")