DEFAULT_MAX_NEW_TOKENS = 10

MODEL_NAME = "sshleifer/tiny-gpt2"

BENCHMARK_BASELINE_CASES = "benchmark_cases.jsonl"
BENCHMARK_BASELINE_RESULTS = "benchmark_results.jsonl"
BENCHMARK_BASELINE_SUMMARY = "benchmark_summary.jsonl"

BENCHMARK_BASELINE_FIG_1 = "avg_latency_vs_output_tokens.png"
BENCHMARK_BASELINE_FIG_2 = "avg_ttft_vs_prompt_length.png"
BENCHMARK_BASELINE_FIG_3 = "avg_tps_vs_output_tokens.png"
BENCHMARK_BASELINE_FIG_4 = "avg_tps_vs_prompt_length.png"

RUNTIME_TRACE_LOG_FILE = "logs/runtime_trace.jsonl"