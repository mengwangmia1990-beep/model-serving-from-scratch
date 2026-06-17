import json
from pathlib import Path
import config

CURRENT_DIR = Path(__file__).parent

OUTPUT_PATH = CURRENT_DIR / Path(config.BENCHMARK_BASELINE_CASES)

MAX_NEW_TOKENS_LIST = [10, 50, 100]

PROMPTS = {
    "short": [
        "Where is Seattle?",
        "What is KV cache?",
        "Explain GPU memory briefly.",
        "What is model serving?",
        "Define latency.",
        "Define throughput.",
        "What is FastAPI?",
        "What is a tokenizer?",
        "What is batching?",
        "What is model inference?"
    ],
    "medium": [
        "Explain how KV cache improves transformer inference performance.",
        "Describe the difference between latency and throughput in model serving.",
        "Explain why batching can improve GPU utilization.",
        "Describe what happens during autoregressive decoding.",
        "Explain the role of attention masks during generation.",
        "Describe how FastAPI can expose a model serving endpoint.",
        "Explain why TTFT is important for user experience.",
        "Describe the difference between prefill and decode stages.",
        "Explain how tokenization converts text into model inputs.",
        "Describe why runtime tracing is useful in model serving."
    ],
    "long": [
        "Explain the full request flow of a minimal LLM serving system, from receiving an HTTP request to returning generated text to the user.",
        "Compare normal autoregressive decoding and KV-cache-based decoding. Explain what computation is reused and what still needs to be computed.",
        "Describe how a production model serving system might handle many concurrent user requests, including request queues, batching, streaming responses, and observability metrics.",
        "Explain how prompt length and generated token count affect different latency metrics in an LLM serving system.",
        "Describe the relationship between prefill latency, decode latency, TTFT, end-to-end latency, and tokens per second.",
        "Explain why GPU utilization matters in LLM serving and how batching can improve throughput.",
        "Describe the purpose of runtime tracing in a model serving system and what metrics should be collected for debugging and benchmarking.",
        "Explain how a simple FastAPI-based model serving system can evolve toward a more production-like serving architecture.",
        "Compare single-request serving, request batching, and continuous batching in terms of latency and throughput tradeoffs.",
        "Describe how KV cache memory grows during generation and why memory management becomes important in high-throughput serving systems."
    ]
}


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_cases = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for category, prompts in PROMPTS.items():
            for prompt_index, prompt in enumerate(prompts, start=0):
                for max_new_tokens in MAX_NEW_TOKENS_LIST:
                    case = {
                        "id": f"{category}_{prompt_index:03d}_t{max_new_tokens}",
                        "category": category,
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens
                    }

                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
                    total_cases += 1

    print(f"Generated {total_cases} benchmark cases.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()