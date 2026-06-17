
# read benchmark_cases
# per case: 
# get prompt
# call generate_with_cache
# consume trace
# generate output
# 1. calculate avg, 50th and 95th overall latency among max_new_token distribution (10, 50, 100)
# 2. calculate avg, 50th and 95th ttft among categories distribution (short, medium, long)
# 3. throughput (TPS: token/sec) in:
#    a) max_new_token distribution (prove: throughput stable while output token increase)
#    b) prompt category distribution (prove: throughput stable while input token increase)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT)) # model-serving-from-scratch/

from generation import generate_with_cache
import json
import config
from statistics import mean, median
import numpy as np
from collections import defaultdict

CURRENT_DIR = Path(__file__).resolve().parent # model-serving-from-scratch/benchmarks/baseline/
BASELINE_CASE_FILE = CURRENT_DIR / config.BENCHMARK_BASELINE_CASES
BASELINE_RESULT_FILE = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_RESULTS
BASELINE_SUMMARY_FILE = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_SUMMARY

def reset_output():
    with open(BASELINE_RESULT_FILE, "w"):
        pass
    with open(BASELINE_SUMMARY_FILE, "w"):
        pass


def write_to_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(content, ensure_ascii=False) + "\n")


def calculate_metrics(values):
    return {
        "avg": round(mean(values), 2),
        "p50": round(median(values), 2),
        "p95": round(np.percentile(values, 95), 2),
    }


def generate_summary_report():
    summary = {}
    latencies = []
    ttfts = []
    tpss = []

    latency_by_max_new_tokens = defaultdict(list)
    ttft_by_category = defaultdict(list)

    tps_by_max_new_tokens = defaultdict(list)
    tps_by_category = defaultdict(list)

    with open(BASELINE_RESULT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            category = case["category"]
            max_new_tokens = str(case["max_new_tokens"])
            trace = case.get("trace")

            if trace is None:
                continue

            latency = trace["end_to_end_latency_ms"]
            ttft = trace["ttft_ms"]
            tps = trace["tokens_per_second"]

            latencies.append(latency)
            ttfts.append(ttft)
            tpss.append(tps)

            latency_by_max_new_tokens[max_new_tokens].append(latency)
            ttft_by_category[category].append(ttft)

            tps_by_max_new_tokens[max_new_tokens].append(tps)
            tps_by_category[category].append(tps)

    overall_summary = {
        "latency": calculate_metrics(latencies),
        "ttft": calculate_metrics(ttfts),
        "tps": calculate_metrics(tpss)
    }

    by_max_new_tokens = {}
    for key, values in latency_by_max_new_tokens.items():
        by_max_new_tokens[key] = {
            "latency": calculate_metrics(values),
            "tps": calculate_metrics(tps_by_max_new_tokens[key])
        }
    
    by_category = {}
    for category, ttfts in ttft_by_category.items():
        by_category[category] = {
            "ttft": calculate_metrics(ttfts),
            "tps": calculate_metrics(tps_by_category[category])
        }

    summary_report = {
        "overall_summary": overall_summary,
        "by_max_new_tokens": by_max_new_tokens,
        "by_prompt_token_cat": by_category
    }

    write_to_file(BASELINE_SUMMARY_FILE, summary_report)


def main():
    reset_output()

    with open(BASELINE_CASE_FILE) as f:
        for line in f:
            case = json.loads(line)
            id = case["id"]
            prompt = case["prompt"]
            category = case["category"]
            max_new_tokens = case["max_new_tokens"]

            response, trace = generate_with_cache(prompt, True, max_new_tokens)

            result = {
                "id": id,
                "category": category,
                "prompt": prompt,
                "response": response,
                "max_new_tokens": max_new_tokens,
                "trace": trace
            }

            write_to_file(BASELINE_RESULT_FILE, result)

    # aggregate to summary

    generate_summary_report()


if __name__ == "__main__":
    main()



