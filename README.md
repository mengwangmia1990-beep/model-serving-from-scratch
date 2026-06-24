# LLM Inference & Serving System
A minimal LLM serving system built with PyTorch, Hugging Face Transformers, KV Cache, FastAPI, streaming generation, runtime tracing, benchmarking, and a toy continuous batching scheduler.

The goal of this project is to understand how modern LLM inference works under the hood by implementing the decoding loop manually instead of relying on high-level generation APIs.  

## Features
- Token-by-token autoregressive decoding
- KV Cache based inference using `past_key_values`
- EOS token based early stopping
- FastAPI serving endpoint
- Streaming generation with incremental token delivery
- Request state based generation lifecycle management
- Toy continuous batching scheduler with active and pending request queues
- Sequential and batched-forward scheduling modes
- Runtime tracing and request correlation via `request_id`
- Automated benchmarking and performance visualization

## Example API
### Request
```json
{ 
    "query": "where is seattle located at?"
}
```
### Response
```json
{ 
    "status": "success",
    "request_id": "c0e50044-3148-47ee-9872-5e4cfaed94b9",
    "response": "where is seattle? stairs stairs stairs..." 
}
```

## Architecture
```text
 Requests
    |
Pending Queue
    |
Scheduler
    |
Active Requests
    |
Batched Forward
    |
KV Cache Merge
    |
Model Forward
    |
KV Cache Split
    |
Request State Update
    |
Runtime Trace
    |
Response
```

## Request Lifecycle
Each request is represented by a `RequestState` object.  

Generation state is maintained across decoding steps:  
- Input token IDs
- Attention mask
- KV Cache
- Generated Tokens
- Runtime Metrics
- Request Metadata

This allows the scheduler to continuously advance active requests while dynamically admitting new requests.

## Continuous Batching Scheduler
A toy continuous batching scheduler was implemented to simulate how modern LLM serving systems manage multiple concurrent requests.

The scheduler maintains:
- Pending request queue
- Active request list
- Request lifecycle management
- Dynamic admission control
- Request completion handling

### Scheduling Modes
Two execution modes are implemented:

#### Sequential Scheduling
Each active request advances independently. One model forward pass is executed per request.
```text
Request A -> Forward Pass
Request B -> Forward Pass
Request C -> Forward Pass
```

#### Batched Forward Scheduling
Multiple compatible requests are grouped into a single model forward pass.

```text
Request A
Request B  ---> Batched Forward Pass
Request C
```

The scheduler:
1. Pads request inputs into a batch
2. Merges KV caches
3. Executes a single model forward pass
4. Splits KV caches back into individual requests
5. Updates each request state independently

This reduces the number of forward passes required when multiple requests can be processed together.

#### Current Limitation
This toy batched-forward implementation only batches requests with identical KV-cache sequence lengths during decoding.  

If requests have different KV-cache sequence lengths during decoding, the scheduler automatically falls back to sequential processing.  

This keeps the implementation simple while demonstrating the core batching mechanics.

## Streaming Generation
Streaming generation returns tokens incrementally as they are produced by the model.
```text
User Request
      |
      v
Generate Token
      |
      v
Stream Token To Client
      |
      v
Generate Next Token
```
This reduces perceived latency compared to waiting for the entire response to finish.

## Runtime Trace
Generation metrics are persisted to a runtime trace log and correlated with requests using `request_id`.  
```jsonl
{
    "request_id": "c0e50044-3148-47ee-9872-5e4cfaed94b9",
    "prompt_tokens": 6,
    "generated_tokens": 10,
    "total_tokens": 16,
    "max_new_tokens": 10,
    "hit_eos": false,
    "end_to_end_latency_ms": 14.5,
    "ttft_ms": 1.74,
    "tokens_per_second": 692,
}
```

## Benchmarking
### Single-Request Baseline
The baseline benchmark evaluates single-request serving performance across different prompt lengths and output lengths.

Collected metrics:  
- TTFT (Time To First Token)
- End-to-End Latency in ms
- Tokens Per Second (TPS)

Benchmark visualizations include:
- Latency vs Output Length
- Throughput vs Output Length
- TTFT vs Prompt Length
- Throughput vs Prompt Length

### Key Findings
- End-to-end latency scales approximately linearly with generated output length.
- Longer prompts increase TTFT due to higher prefill cost.
- Decode throughput remains relatively stable across different prompt lengths and output lengths.

#### Latency vs Output Length
![alt text](image.png)

#### TTFT vs Prompt Length
![alt text](image-1.png)

#### Throughput vs Output Length
![alt text](image-2.png)

---

### Sequential Scheduling vs Batched Forward Scheduling
A simple benchmark was conducted to compare sequential scheduling and batched-forward scheduling under the same workload.
```
=== Sequential Summary ===
{
  "num_requests": 5,
  "avg_latency_ms": 18.745920015498996,
  "avg_ttft_ms": 9.275939967483282,
  "avg_tokens_per_second": 108.91488376173365
}

=== Batched Summary ===
{
  "num_requests": 5,
  "avg_latency_ms": 6.430420046672225,
  "avg_ttft_ms": 3.245260054245591,
  "avg_tokens_per_second": 313.7067001068591
}
```

### Key Findings
Batched forward scheduling reduced average latency and TTFT while **significantly improving throughput by sharing model forward passes** across compatible requests.


## Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```
### SwaggerUI
```text
http://127.0.0.1:8000/docs
```

## Next Steps
- KV cache padding for heterogeneous decode batches
- Paged KV Cache (vLLM-inspired)
