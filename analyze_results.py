import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

with open('test_cases_evaluation.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
groups = {}
for d in data:
    g = d['language_group']
    groups.setdefault(g, {'total': 0, 'match': 0, 'fp': 0, 'fn': 0})
    groups[g]['total'] += 1
    if d['eval_status'] == 'MATCH':
        groups[g]['match'] += 1
    elif d['eval_status'] == 'FALSE_POSITIVE':
        groups[g]['fp'] += 1
    elif d['eval_status'] == 'FALSE_NEGATIVE':
        groups[g]['fn'] += 1

total_match = sum(g['match'] for g in groups.values())
total_all = len(data)
print(f"Overall Accuracy: {total_match}/{total_all} ({total_match/total_all*100:.1f}%)")

print("\nGroup Breakdown:")
for g, s in groups.items():
    acc = s['match'] / s['total'] * 100
    print(f"  {g:12s}: Total={s['total']}, Match={s['match']} ({acc:.1f}%), False Positives={s['fp']}, False Negatives={s['fn']}")

print("\n=== FALSE NEGATIVES (Bullying missed by model) ===")
for d in data:
    if d['eval_status'] == 'FALSE_NEGATIVE':
        p_safe = d['muril']['all_probabilities'].get('not_cyberbullying', 'N/A')
        print(f"  [{d['id']:8s}] [{d['language_group']:10s}] \"{d['text']}\" -> Safe Prob: {p_safe}% | Baseline: {d['baseline']['verdict']}")

print("\n=== FALSE POSITIVES (Safe flagged as Bullying) ===")
for d in data:
    if d['eval_status'] == 'FALSE_POSITIVE':
        p_bully = 100 - d['muril']['all_probabilities'].get('not_cyberbullying', 0)
        print(f"  [{d['id']:8s}] [{d['language_group']:10s}] \"{d['text']}\" -> Pred: {d['muril']['category']} ({d['muril']['confidence']}%) | Baseline: {d['baseline']['verdict']}")

print("\n=== CATEGORY ANALYSIS (For MATCH items) ===")
for d in data:
    if d['eval_status'] == 'MATCH' and d['expected_is_bully']:
        pred_cat = d['muril']['category']
        exp_cat = d['expected_category']
        print(f"  [{d['id']:8s}] Text: \"{d['text'][:35]:35s}\" | Pred: {pred_cat:10s} | Expected: {exp_cat}")
