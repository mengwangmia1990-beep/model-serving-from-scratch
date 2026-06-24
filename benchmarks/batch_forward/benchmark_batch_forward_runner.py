from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT)) # model-serving-from-scratch/

import json
import statistics
from collections import deque
from generation import run_scheduler, run_scheduler_batch
from models.request_state import RequestState
import uuid
import config

RUNTIME_TRACE_FILE = PROJECT_ROOT / config.RUNTIME_TRACE_LOG_FILE

def build_batch_requests(run_id: str):
    return [
        RequestState(f"batch-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"batch-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"batch-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"batch-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"batch-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
    ]

def build_sequential_requests(run_id: str):
    return [
        RequestState(f"sequential-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"sequential-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"sequential-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"sequential-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
        RequestState(f"sequential-{run_id}-{str(uuid.uuid4())}", "where is seattle?", 2, True),
    ]

def load_trace_file(path: str):
    traces = []

    with open(path, "r") as f:
        for line in f:
            traces.append(json.loads(line))

    return traces


def filter_traces_by_prefix(traces, prefix: str):
    return [
        t for t in traces
        if t["request_id"].startswith(prefix)
    ]

def print_summary(traces):
    summary = {
        "num_requests": len(traces),
        "avg_latency_ms": statistics.mean(
            t["end_to_end_latency_ms"] for t in traces
        ),
        "avg_ttft_ms": statistics.mean(
            t["ttft_ms"] for t in traces
        ),
        "avg_tokens_per_second": statistics.mean(
            t["tokens_per_second"] for t in traces
        ),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    max_active_requests = 5
    run_id = str(uuid.uuid4())[:8]

    sequential_outputs = run_scheduler(
        build_sequential_requests(run_id), 
        pending_queue=deque(), 
        max_active_requests=max_active_requests,
    )

    batched_outputs = run_scheduler_batch(
        build_batch_requests(run_id),
        pending_queue=deque(),
        max_active_requests=max_active_requests,
    )

    traces = load_trace_file(RUNTIME_TRACE_FILE)
    sequential_traces = filter_traces_by_prefix(traces, f"sequential-{run_id}-")
    batch_traces = filter_traces_by_prefix(traces, f"batch-{run_id}-")

    print("\n=== Sequential Summary ===")
    print_summary(sequential_traces)

    print("\n=== Batched Summary ===")
    print_summary(batch_traces)

    print("\n=== Outputs ===")
    print("Sequential:")
    print(json.dumps(sequential_outputs, indent=2, ensure_ascii=False))

    print("Batched:")
    print(json.dumps(batched_outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()