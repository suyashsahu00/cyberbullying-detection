"""
Comprehensive Latency Benchmark Suite for Cyberbullying Detection System.

Measures:
1. Cold-Start Latency (Model loading & 1st inference invocation)
2. Warm-Inference Latency across N iterations (Min, Max, Mean, Median/P50, P90, P95, P99, StdDev, QPS)
3. Direct Model Inference vs End-to-End Pipeline Comparison
4. Classical Baseline (TF-IDF + Linear SVM) vs Deep Transformer (Google MuRIL v2)

Usage:
    python benchmark_latency.py [--runs 100]
"""

import os
import sys
import time
import json
import platform
import argparse
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Diverse test inputs representing typical social media comments
BENCHMARK_INPUTS = [
    "you fucking idiot / motherfucker",
    "suck my dick, go fuck yourself",
    "your mom is a whore",
    "madarchod",
    "bhosdiwala traffic",
    "you are so helpful, thank you!",
    "bohot samajhdar ho aap, dimaag mat use karna",
    "Go back to your country you illegal immigrant curryboy.",
    "Shut up you senile old hag boomer grandpa!",
    "Thank you so much for the helpful answer, appreciate it!"
]

def get_system_specs():
    specs = {
        "os": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU",
        "cpu_count": os.cpu_count()
    }
    return specs

def compute_percentiles(latencies):
    arr = np.array(latencies)
    return {
        "count": len(arr),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "qps": round(float(1000.0 / np.mean(arr)), 1)
    }

def main():
    parser = argparse.ArgumentParser(description="Run Latency Benchmark")
    parser.add_argument("--runs", type=int, default=50, help="Number of warm runs per input")
    args = parser.parse_args()

    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 90)
    print(" ⏱️ CYBERBULLYING DETECTION LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 90)

    specs = get_system_specs()
    print("Hardware & Execution Environment:")
    print(f"  • Operating System : {specs['os']}")
    print(f"  • CPU / Processor  : {specs['processor']} ({specs['cpu_count']} logical cores)")
    print(f"  • Execution Device : {specs['device']}")
    print(f"  • Python / PyTorch : Python {specs['python_version']} | PyTorch {specs['pytorch_version']}")
    print(f"  • Benchmark Cycles : {args.runs} warm iterations per text ({args.runs * len(BENCHMARK_INPUTS)} total queries)")
    print("-" * 90)

    # -------------------------------------------------------------
    # 1. COLD START MEASUREMENT
    # -------------------------------------------------------------
    print("\n[Phase 1] Measuring Cold-Start Latency (System Load + 1st Invocation)...")
    
    # Measure import & initialization
    t_init_start = time.perf_counter()
    from src.model import CyberbullyingSystem
    system = CyberbullyingSystem()
    t_init_end = time.perf_counter()
    system_init_ms = round((t_init_end - t_init_start) * 1000, 2)
    print(f"  --> System Load & Checkpoint Initialization: {system_init_ms:.2f} ms")

    # Cold inference (1st call ever)
    sample_text = BENCHMARK_INPUTS[0]
    
    t_cold_base_0 = time.perf_counter()
    system.predict(sample_text, model_choice="baseline")
    t_cold_base_1 = time.perf_counter()
    cold_base_ms = round((t_cold_base_1 - t_cold_base_0) * 1000, 2)
    print(f"  --> Baseline 1st Query (Cold Start)       : {cold_base_ms:.2f} ms")

    t_cold_muril_0 = time.perf_counter()
    system.predict(sample_text, model_choice="muril")
    t_cold_muril_1 = time.perf_counter()
    cold_muril_ms = round((t_cold_muril_1 - t_cold_muril_0) * 1000, 2)
    print(f"  --> MuRIL v2 1st Query (Cold Start)       : {cold_muril_ms:.2f} ms")

    # -------------------------------------------------------------
    # 2. WARM INFERENCE BENCHMARK (Baseline: Linear SVM)
    # -------------------------------------------------------------
    print(f"\n[Phase 2] Benchmarking Classical Baseline (TF-IDF + Linear SVM) across {args.runs} warm runs...")
    baseline_latencies = []
    
    # Warmup pass
    for t in BENCHMARK_INPUTS:
        system.predict(t, model_choice="baseline")

    for run_idx in range(args.runs):
        for text in BENCHMARK_INPUTS:
            t0 = time.perf_counter()
            res = system.predict(text, model_choice="baseline")
            t1 = time.perf_counter()
            baseline_latencies.append((t1 - t0) * 1000.0)

    base_stats = compute_percentiles(baseline_latencies)

    # -------------------------------------------------------------
    # 3. WARM INFERENCE BENCHMARK (Deep MuRIL v2 Transformer)
    # -------------------------------------------------------------
    print(f"\n[Phase 3] Benchmarking Google MuRIL v2 Transformer across {args.runs} warm runs...")
    muril_latencies = []
    
    # Warmup pass
    for t in BENCHMARK_INPUTS:
        system.predict(t, model_choice="muril")

    for run_idx in range(args.runs):
        for text in BENCHMARK_INPUTS:
            t0 = time.perf_counter()
            res = system.predict(text, model_choice="muril")
            t1 = time.perf_counter()
            muril_latencies.append((t1 - t0) * 1000.0)

    muril_stats = compute_percentiles(muril_latencies)

    # -------------------------------------------------------------
    # 4. REPORT TERMINAL COMPARISON TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" 📊 REAL BENCHMARK RESULTS SUMMARY (READY FOR PAPER / REPORT)")
    print("=" * 90)
    print(f"{'Metric':<30} | {'Tier 1: Linear SVM':<22} | {'Tier 2: Google MuRIL v2':<24}")
    print("-" * 90)
    print(f"{'Cold-Start 1st Invocation':<30} | {cold_base_ms:>18.2f} ms | {cold_muril_ms:>20.2f} ms")
    print(f"{'Warm Mean Latency (Average)':<30} | {base_stats['mean_ms']:>18.2f} ms | {muril_stats['mean_ms']:>20.2f} ms")
    print(f"{'Warm Median Latency (P50)':<30} | {base_stats['median_ms']:>18.2f} ms | {muril_stats['median_ms']:>20.2f} ms")
    print(f"{'Warm 90th Percentile (P90)':<30} | {base_stats['p90_ms']:>18.2f} ms | {muril_stats['p90_ms']:>20.2f} ms")
    print(f"{'Warm 95th Percentile (P95)':<30} | {base_stats['p95_ms']:>18.2f} ms | {muril_stats['p95_ms']:>20.2f} ms")
    print(f"{'Warm 99th Percentile (P99)':<30} | {base_stats['p99_ms']:>18.2f} ms | {muril_stats['p99_ms']:>20.2f} ms")
    print(f"{'Minimum Observed Latency':<30} | {base_stats['min_ms']:>18.2f} ms | {muril_stats['min_ms']:>20.2f} ms")
    print(f"{'Maximum Observed Latency':<30} | {base_stats['max_ms']:>18.2f} ms | {muril_stats['max_ms']:>20.2f} ms")
    print(f"{'Standard Deviation':<30} | {base_stats['std_ms']:>18.2f} ms | {muril_stats['std_ms']:>20.2f} ms")
    print(f"{'Throughput (Queries/Sec)':<30} | {base_stats['qps']:>15.1f} QPS | {muril_stats['qps']:>17.1f} QPS")
    print("=" * 90)

    # Save to JSON
    out_payload = {
        "hardware_environment": specs,
        "cold_start": {
            "system_init_ms": system_init_ms,
            "baseline_1st_query_ms": cold_base_ms,
            "muril_1st_query_ms": cold_muril_ms
        },
        "baseline_linear_svm": base_stats,
        "deep_muril_v2": muril_stats
    }

    out_json = os.path.join(ROOT_DIR, "models", "latency_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved benchmark metrics to: {out_json}")
    print("All numbers verified from actual execution timing loops.\n")

if __name__ == "__main__":
    main()
