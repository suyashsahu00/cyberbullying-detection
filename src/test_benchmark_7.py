import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import get_classifier

test_inputs = [
    "you fucking idiot / motherfucker",
    "suck my dick, go fuck yourself",
    "your mom is a whore",
    "madarchod",
    "bhosdiwala traffic",
    "you are so helpful, thank you!",
    "bohot samajhdar ho aap, dimaag mat use karna"
]

classifier = get_classifier()

print("=" * 80)
print("BENCHMARK TEST SUITE: 7 REQUIRED INPUT EVALUATIONS")
print("=" * 80)

results = []

for idx, text in enumerate(test_inputs, 1):
    print(f"\n[{idx}] INPUT: \"{text}\"")
    print("-" * 80)
    
    # 1. MuRIL v2
    muril_res = classifier.predict(text, model_choice="muril")
    print(f"  [MuRIL v2 (Transformer)]")
    print(f"    Verdict       : {muril_res['verdict']}")
    print(f"    Is Bullying   : {muril_res['is_cyberbullying']}")
    print(f"    Category      : {muril_res['category']}")
    print(f"    Confidence    : {muril_res['confidence']}%")
    print(f"    Language      : {muril_res['language']}")
    print(f"    Probabilities : {muril_res['all_probabilities']}")
    print(f"    Explain Method: {muril_res['explainability_method']}")
    print(f"    Top Attributed Tokens: {muril_res['explainability'].get('top_tokens', [])}")
    print(f"    Raw Attributions: {muril_res['explainability'].get('raw_attributions', [])}")
    
    # 2. Baseline SVM
    base_res = classifier.predict(text, model_choice="baseline")
    print(f"  [Baseline (Linear SVM)]")
    print(f"    Verdict       : {base_res['verdict']}")
    print(f"    Is Bullying   : {base_res['is_cyberbullying']}")
    print(f"    Category      : {base_res['category']}")
    print(f"    Confidence    : {base_res['confidence']}%")
    print(f"    Explain Method: {base_res['explainability_method']}")
    print(f"    Trigger Words : {base_res['explainability'].get('trigger_words', [])}")
    
    results.append({
        "input": text,
        "muril_output": muril_res,
        "baseline_output": base_res
    })

# Save results to json
with open(os.path.join(ROOT_DIR, "models", "benchmark_7_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("Benchmark test complete! Saved full raw JSON output to models/benchmark_7_results.json")
print("=" * 80)
