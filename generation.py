from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import config
import time
import json
from pathlib import Path

model_name = config.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

CURRENT_DIR = Path(__file__).resolve().parent
RUNTIME_TRACE_FILE = CURRENT_DIR / config.RUNTIME_TRACE_LOG_FILE

def write_trace(trace):
    RUNTIME_TRACE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(RUNTIME_TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

def set_trace(
        runtime_trace: bool,
        request_id: str | None,
        prompt_tokens: int,
        generated_tokens: int,
        request_start: float,
        request_end: float,
        first_token_time: float,
        max_new_tokens: int,
        hit_eos: bool
        ):
    
    if runtime_trace is False:
        return None
    
    total_tokens = prompt_tokens + generated_tokens
    end_to_end_latency_ms = (request_end - request_start) * 1000
    ttft_ms = (first_token_time - request_start) * 1000

    duration = max(request_end - request_start, 1e-6)
    tokens_per_second = generated_tokens / duration

    trace = {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "max_new_tokens": max_new_tokens,
        "hit_eos": hit_eos,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_second": tokens_per_second,
    }
    return trace


def generate_token_ids_with_cache(
    prompt: str,
    runtime_trace: bool = True,
    max_new_tokens: int | None = None,
    request_id: str | None = None,
):
    if max_new_tokens is None:
        max_new_tokens = config.DEFAULT_MAX_NEW_TOKENS
    
    request_start = time.perf_counter()
    first_token_time = None
    generated_tokens = 0
    hit_eos = False

    inputs = tokenizer(
        prompt, 
        return_tensors="pt"
    )

    prompt_tokens = inputs["input_ids"].shape[1]

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    past_key_values = None
    current_input_ids = input_ids

    try:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            # collect kv cache
            past_key_values = outputs.past_key_values

            # pick next token
            last_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True) # token Id is logits index

            if first_token_time is None:
                first_token_time = time.perf_counter()

            if next_token_id.item() == tokenizer.eos_token_id:
                hit_eos = True
                break

            current_input_ids = next_token_id
            next_attention = torch.ones_like(next_token_id)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

            generated_tokens += 1

            yield(next_token_id.item())
    
    finally:
        request_end = time.perf_counter()

        # set runtime trace
        trace = set_trace(
            runtime_trace, 
            request_id,
            prompt_tokens, 
            generated_tokens, 
            request_start, 
            request_end,
            first_token_time,
            max_new_tokens,
            hit_eos
        )

        if trace is not None:
            write_trace(trace)


def generate_with_cache(
        prompt: str, 
        runtime_trace: bool = True,
        max_new_tokens: int | None = None,
        request_id: str | None = None) -> str:
    
    generated_token_ids = []

    for token_id in generate_token_ids_with_cache(prompt, runtime_trace, max_new_tokens, request_id):
        generated_token_ids.append(token_id)

    # decode
    result = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return result


def stream_with_cache(
        prompt: str,
        runtime_trace: bool = True,
        max_new_tokens: int | None = None,
        request_id: str | None = None
):
    for token_id in generate_token_ids_with_cache(prompt, runtime_trace, max_new_tokens, request_id):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        # time.sleep(0.5)
        yield(token_text)

