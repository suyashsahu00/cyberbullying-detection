import os
import sys
import json
import numpy as np
from scipy.stats import binomtest

with open('scratch/human_flips_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Models available
# keys: Archive_prod_safe, Run 0 (Seed 42)_prod_safe, Run 0b (Seed 43)_prod_safe,
# Ablation B (Seed 42)_prod_safe, Ablation B (Seed 43)_prod_safe,
# Ablation C (Seed 42)_prod_safe, Ablation C (Seed 43)_prod_safe

def get_contingency(ctrl_bool, treat_bool):
    a = sum(1 for c, t in zip(ctrl_bool, treat_bool) if c and t)
    b = sum(1 for c, t in zip(ctrl_bool, treat_bool) if c and not t) # Degraded
    c = sum(1 for c, t in zip(ctrl_bool, treat_bool) if not c and t) # Improved
    d = sum(1 for c, t in zip(ctrl_bool, treat_bool) if not c and not t)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        p = binomtest(b, n, 0.5, alternative='two-sided').pvalue
    return {'a': a, 'b': b, 'c': c, 'd': d, 'n': n, 'p': p}

def holm_adjust(p_values, m=None):
    # p_values: list of float
    # if m is None, m = len(p_values)
    k = len(p_values)
    if m is None:
        m = k
    
    # Sort with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * k
    running_max = 0.0
    
    for rank, (orig_idx, p) in enumerate(sorted_pairs):
        multiplier = m - rank
        raw_adj = p * multiplier
        running_max = max(running_max, raw_adj)
        adj[orig_idx] = min(1.0, running_max)
        
    return adj

# Candidate same-seed comparisons:
# Let's inspect the 8 natural same-seed tests:
tests_8 = [
    ("Ablation B (s42) vs Run 0 (s42)", "Run 0 (Seed 42)_prod_safe", "Ablation B (Seed 42)_prod_safe"),
    ("Ablation C (s42) vs Run 0 (s42)", "Run 0 (Seed 42)_prod_safe", "Ablation C (Seed 42)_prod_safe"),
    ("Ablation B (s43) vs Run 0b (s43)", "Run 0b (Seed 43)_prod_safe", "Ablation B (Seed 43)_prod_safe"),
    ("Ablation C (s43) vs Run 0b (s43)", "Run 0b (Seed 43)_prod_safe", "Ablation C (Seed 43)_prod_safe"),
    ("Ablation B (s42) vs Archive", "Archive_prod_safe", "Ablation B (Seed 42)_prod_safe"),
    ("Ablation C (s42) vs Archive", "Archive_prod_safe", "Ablation C (Seed 42)_prod_safe"),
    ("Run 0 (s42) vs Archive", "Archive_prod_safe", "Run 0 (Seed 42)_prod_safe"),
    ("Run 0b (s43) vs Archive", "Archive_prod_safe", "Run 0b (Seed 43)_prod_safe"),
]

results_8 = []
for name, c_key, t_key in tests_8:
    ct = get_contingency(data[c_key], data[t_key])
    results_8.append({
        'test': name,
        'a': ct['a'],
        'b': ct['b'],
        'c': ct['c'],
        'd': ct['d'],
        'n': ct['n'],
        'p_raw': ct['p']
    })

p_raw_list = [r['p_raw'] for r in results_8]
p_holm_8 = holm_adjust(p_raw_list, m=8)

for r, ph in zip(results_8, p_holm_8):
    r['p_holm'] = ph

print("=" * 115)
print("EXACT MCNEMAR TESTS WITH HOLM-BONFERRONI CORRECTION (8 Same-Seed Comparisons, m=8)")
print("=" * 115)
print(f"{'Comparison':<35} | {'a (Both)':<8} | {'b (Deg)':<7} | {'c (Imp)':<7} | {'d (Fail)':<8} | {'Discordant':<10} | {'p (raw)':<10} | {'p (Holm m=8)':<12}")
print("-" * 115)
for r in results_8:
    print(f"{r['test']:<35} | {r['a']:>8} | {r['b']:>7} | {r['c']:>7} | {r['d']:>8} | {r['n']:>10} | {r['p_raw']:>10.5f} | {r['p_holm']:>12.5f}")

# Also check 4 core treatment-vs-control same-seed tests:
tests_4 = [
    ("Ablation B (s42) vs Run 0 (s42)", "Run 0 (Seed 42)_prod_safe", "Ablation B (Seed 42)_prod_safe"),
    ("Ablation C (s42) vs Run 0 (s42)", "Run 0 (Seed 42)_prod_safe", "Ablation C (Seed 42)_prod_safe"),
    ("Ablation B (s43) vs Run 0b (s43)", "Run 0b (Seed 43)_prod_safe", "Ablation B (Seed 43)_prod_safe"),
    ("Ablation C (s43) vs Run 0b (s43)", "Run 0b (Seed 43)_prod_safe", "Ablation C (Seed 43)_prod_safe"),
]
p_raw_4 = [get_contingency(data[ck], data[tk])['p'] for _, ck, tk in tests_4]
p_holm_m4 = holm_adjust(p_raw_4, m=4)
p_holm_m8 = holm_adjust(p_raw_4, m=8)

print("\n" + "=" * 115)
print("CORE SAME-SEED TREATMENT-VS-CONTROL PAIRS (N=4)")
print("=" * 115)
print(f"{'Comparison':<35} | {'p (raw)':<10} | {'p (Holm m=4)':<14} | {'p (Holm m=8)':<14}")
print("-" * 115)
for (name, _, _), pr, ph4, ph8 in zip(tests_4, p_raw_4, p_holm_m4, p_holm_m8):
    print(f"{name:<35} | {pr:>10.5f} | {ph4:>14.5f} | {ph8:>14.5f}")

