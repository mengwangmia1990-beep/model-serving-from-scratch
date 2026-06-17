from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT)) # model-serving-from-scratch/

import matplotlib.pyplot as plt
import config
import json

CURRENT_DIR = Path(__file__).resolve().parent
SUMMARY_FILE = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_SUMMARY
FIG1_OUTPUT = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_FIG_1
FIG2_OUTPUT = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_FIG_2
FIG3_OUTPUT = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_FIG_3
FIG4_OUTPUT = CURRENT_DIR / "results" / config.BENCHMARK_BASELINE_FIG_4

def draw_ttft_vs_prompt_length(data):
    x = ["short", "medium", "long"]
    y = [
        data["short"]["ttft"]["avg"],
        data["medium"]["ttft"]["avg"],
        data["long"]["ttft"]["avg"]
    ]

    plt.figure(figsize=(6,4))
    plt.bar(x, y)

    plt.xlabel("Prompt token length")
    plt.ylabel("TTFT (ms)")
    plt.title("TTFT vs Prompt Token Length")
    plt.savefig(FIG2_OUTPUT)

    plt.close()

def draw_latency_vs_max_new_tokens(data):
    x = [10, 50, 100]
    y = [
        data["10"]["latency"]["avg"],
        data["50"]["latency"]["avg"],
        data["100"]["latency"]["avg"]
    ]

    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker="o")

    plt.xlabel("Max New Tokens")
    plt.ylabel("Average Latency (ms)")
    plt.title("Latency vs Output Length")

    plt.grid(True)

    plt.savefig(FIG1_OUTPUT)
    plt.close()


def draw_throughput_vs_max_new_tokens(data):
    x = [10, 50, 100]
    y = [
        data["10"]["tps"]["avg"],
        data["50"]["tps"]["avg"],
        data["100"]["tps"]["avg"]
    ]

    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker="o")
    plt.ylim(bottom=0)

    plt.xlabel("Max New Tokens")
    plt.ylabel("Throughput (Tokens/sec)")
    plt.title("Throughput vs Output Length")

    plt.grid(True)

    plt.savefig(FIG3_OUTPUT)
    plt.close()


def draw_throughput_vs_prompt_length(data):
    x = ["short", "medium", "long"]
    y = [
        data["short"]["tps"]["avg"],
        data["medium"]["tps"]["avg"],
        data["long"]["tps"]["avg"]
    ]

    plt.figure(figsize=(6,4))
    plt.bar(x, y)

    plt.xlabel("Prompt token length")
    plt.ylabel("Throughput (Tokens/sec)")
    plt.title("Throughput vs Prompt Token Length")
    plt.savefig(FIG4_OUTPUT)

    plt.close()


def main():
    with open(SUMMARY_FILE, "r") as f:
        report = json.load(f)

        draw_latency_vs_max_new_tokens(report["by_max_new_tokens"])
        draw_ttft_vs_prompt_length(report["by_prompt_token_cat"])
        draw_throughput_vs_max_new_tokens(report["by_max_new_tokens"])
        draw_throughput_vs_prompt_length(report["by_prompt_token_cat"])

if __name__ == "__main__":
    main()




