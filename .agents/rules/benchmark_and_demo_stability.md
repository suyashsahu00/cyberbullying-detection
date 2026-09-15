# Benchmark Metrics Lock & Demo Stability Guardrails

## 1. Strict Metric Locking (No Unnecessary Re-runs)
- **Do NOT re-run benchmark scripts** (`benchmark_latency.py`, `blind_test.py`, etc.) unless there is an explicit, substantial code or architecture modification.
- Avoid repeated benchmark invocations to prevent numerical drift caused by hardware thermals or OS scheduling before demo/presentation deadlines.
- All documents ([report.md](file:///c:/Users/suyas/Downloads/CODING(1)/cyberbullying-detection/report.md), [project_state.md](file:///c:/Users/suyas/Downloads/CODING(1)/cyberbullying-detection/project_state.md), [presentation.md](file:///c:/Users/suyas/Downloads/CODING(1)/cyberbullying-detection/presentation.md)) must remain strictly synchronized with the locked JSON artifacts in `models/`.

## 2. Hardware & Environment Ground Truth
- Always verify the actual execution environment directly from the Python runtime (e.g., `torch.__version__`, `torch.cuda.is_available()`) rather than assuming or fabricating environment strings.
- Explicitly differentiate between CPU edge execution (e.g., Tier-1 Linear SVM) and GPU tensor acceleration (e.g., Tier-2 MuRIL on RTX 4050).

## 3. UI Sample Presets Behavior
- Quick sample / preset pills in the frontend UI must **only populate the text input area** and update character counters.
- **Never auto-trigger analysis or render results** upon clicking a preset pill; analysis must strictly await the user's explicit click on the "Analyze" button or `Ctrl+Enter`.
