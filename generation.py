from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import config
import time
import json
from pathlib import Path
from models.request_state import RequestState
from collections import deque
from transformers.cache_utils import DynamicCache

model_name = config.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

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

# keep this method for baseline benchmark_runner
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

def finish_request(request: RequestState):
    request.finished = True
    request.request_end_time = time.perf_counter()
    trace = set_trace(
        request.runtime_trace,
        request.request_id,
        request.prompt_tokens,
        request.generated_tokens,
        request.request_start_time,
        request.request_end_time,
        request.first_token_time,
        request.max_new_tokens,
        request.hit_eos,
    )
    if trace:
        write_trace(trace)

def dynamic_cache_to_tuple(cache):
    return tuple(
        (layer.keys, layer.values)
        for layer in cache.layers
    )

def tuple_to_dynamic_cache(past_key_values):
    if past_key_values is None:
        return None

    cache = DynamicCache()

    for layer_idx, (key, value) in enumerate(past_key_values):
        cache.update(
            key_states=key,
            value_states=value,
            layer_idx=layer_idx,
        )

    return cache

def to_dynamic_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


# generate one token per each request for scheduler
def step_request(request: RequestState) -> int | None:
    if request.finished:
        return None
    
    with torch.no_grad():
        outputs = model(
            input_ids=request.current_input_ids,
            attention_mask=request.attention_mask,
            past_key_values=tuple_to_dynamic_cache(request.past_key_values),
            use_cache=True
        )

    # collect kv cache
    request.past_key_values = dynamic_cache_to_tuple(outputs.past_key_values)

    # pick next token
    last_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True) # token Id is logits index

    if request.first_token_time is None:
        request.first_token_time = time.perf_counter()

    token_id = next_token_id.item()

    # hit eos --> stop
    if token_id == tokenizer.eos_token_id:
        request.hit_eos = True
        finish_request(request)
        return None

    request.current_input_ids = next_token_id
    next_attention = torch.ones_like(next_token_id)
    request.attention_mask = torch.cat([request.attention_mask, next_attention], dim=1)

    request.generated_tokens += 1
    request.generated_token_ids.append(token_id)

    # generated tokens reach max new token limit --> stop
    if request.generated_tokens >= request.max_new_tokens:
        finish_request(request)

    return token_id


def step_batched_requests(requests: list[RequestState]) -> dict[str, int | None]:
    active = [r for r in requests if not r.finished]
    if not active:
        return {}
    
    # 先只 batch 当前 past_key_values 兼容的一组
    # 最小 toy 版：要求它们 past length 一样
    # 否则先 fallback 成单个处理，避免复杂 padding KV cache
    past_lens = []
    for r in active:
        if r.past_key_values is None:
            past_lens.append(None) # prefill stage can batch
        else:
            past_lens.append(r.past_key_values[0][0].shape[2])
            # kv cache structure: [batch_size, head_num, seq_len, head_dim]
            # e.g. [1, 2, 5, 64]
            # 1 -> 1 request per batch
            # 2 -> 2 heads
            # 5 -> current processed total token count (prompt + predicted token)
            # 64 -> head dimention (64 dimention vector)
            # r.past_key_values[0][0].shape[2] --> 0th layer, key, seq_len
            # r.past_key_values[0][1].shape[2] --> 0th layer, value, seq_len
    
    if len(set(past_lens)) > 1:
        results = {}
        for r in active:
            results[r.request_id] = step_request(r)
        return results
    
    # pad current_input_ids
    batched_current_input_ids = torch.nn.utils.rnn.pad_sequence(
        [r.current_input_ids.squeeze(0) for r in active],
        batch_first=True,
        padding_value=tokenizer.pad_token_id, # PAD
    )
    # pad attention_mask
    batched_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [r.attention_mask.squeeze(0) for r in active],
        batch_first=True,
        padding_value=0
    )
    #==================================================================================================#
    # prefill stage input ids and attention_mask padding
    # input_ids =
    # [
    # [t1,t2,t3,t4],
    # [t1,t2,PAD,PAD],
    # [t1,t2,t3,PAD],
    # ]

    # attention_mask =
    # [
    # [1,1,1,1],
    # [1,1,0,0],
    # [1,1,1,0],
    # ]
    #==================================================================================================#
    # decode stage input ids and attention_mask padding
    # input_ids = 
    # [
    # [t5],
    # [t3],
    # [t4],
    # ]

    # attention_mask =
    # [
    # [1,1,1,1,1],
    # [1,1,1,PAD,PAD],
    # [1,1,1,1,PAD],
    # ]
    #==================================================================================================#

    # batch past_key_values
    if active[0].past_key_values is None:
        batched_past = None
    else:
        batched_past = []
        num_layers = len(active[0].past_key_values)

        for layer_idx in range(num_layers):
            keys = torch.cat(
                [r.past_key_values[layer_idx][0] for r in active],
                dim=0,
            )
            values = torch.cat(
                [r.past_key_values[layer_idx][1] for r in active],
                dim=0,
            )
            batched_past.append((keys, values))

        batched_past = tuple_to_dynamic_cache(tuple(batched_past))

    with torch.no_grad():
        outputs = model(
            input_ids=batched_current_input_ids,
            attention_mask=batched_attention_mask,
            past_key_values=batched_past, # model needs dynamic cache
            use_cache=True,
        )

    # next token ids for all requests in the same batch
    next_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    outputs_past = dynamic_cache_to_tuple(outputs.past_key_values)
    # print("output_past[0][0] shape")
    # print(outputs_past[0][0].shape) # [3, 2, 6, 1] after prefill, 6 is the max padded seq_len.

    # outputs_past KV Cache structure:
    # layer_0
    #     Key Tensor
    #         Request0
    #             ├── Head0: Key
    #             └── Head1: Key
    #         Request1
    #             ├── Head0: Key
    #             └── Head1: Key        
    #         Request2
    #             ├── Head0: Key
    #             └── Head1: Key
    #     Value Tensor
    #         Request0
    #             ├── Head0: Value
    #             └── Head1: Value
    #         Request1
    #             ├── Head0: Value
    #             └── Head1: Value        
    #         Request2
    #             ├── Head0: Value
    #             └── Head1: Value
    # layer_1:
    #      Key Tensor
    #      Value Tensor
    
    results = {}

    # post processing --> for each active request
    for i, r in enumerate(active):
        # split kv cache back to each request
        new_past = []
        for layer_key, layer_value in outputs_past:
            request_key = layer_key[i:i+1]
            request_value = layer_value[i:i+1]
            new_past.append(
                (
                    request_key,
                    request_value
                )
            )
        r.past_key_values = tuple(new_past)
        # print(r.past_key_values[0][0].shape)

        # log first token time
        if r.first_token_time is None:
            r.first_token_time = time.perf_counter()

        token_id = next_token_ids[i].item()
        if token_id == tokenizer.eos_token_id:
            r.hit_eos = True
            finish_request(r)
            results[r.request_id] = None
            continue

        # update current_input_ids and attention_mask
        next_token_id = next_token_ids[i].view(1, 1)
        r.current_input_ids = next_token_id
        r.attention_mask = torch.cat(
            [r.attention_mask, torch.ones_like(next_token_id)],
            dim=1,
        )

        r.generated_tokens += 1
        r.generated_token_ids.append(token_id)

        if r.generated_tokens >= r.max_new_tokens:
            finish_request(r)

        # save result: request_id : token_id
        results[r.request_id] = token_id

    return results


def run_scheduler(
    initial_requests: list[RequestState],
    pending_queue: deque[RequestState] | None = None,
    max_active_requests: int = 3,
):
    active_requests = []
    all_requests = []

    if pending_queue is None:
        pending_queue = deque()

    pending_queue = deque(
        list(initial_requests) + list(pending_queue)
    )

    while active_requests or pending_queue:
        # fill upcoming requests from pending queue into active request list when there are available slots
        # initial fill
        # during generation fill
        while pending_queue and len(active_requests) < max_active_requests:
            req = pending_queue.popleft()
            
            active_requests.append(req)
            all_requests.append(req)
            print("admitted:", req.request_id)

        print("active requests: ", [req.request_id for req in active_requests])
        
        for request in active_requests:
            token_id = step_request(request)

            if token_id is not None:
                print(request.request_id, tokenizer.decode([token_id]))
        
        active_requests = [
            r for r in active_requests
            if not r.finished
        ]

    return {
        r.request_id: tokenizer.decode(r.generated_token_ids, skip_special_tokens=True)
        for r in all_requests
    }


def run_scheduler_batch(
    initial_requests: list[RequestState],
    pending_queue: deque[RequestState] | None = None,
    max_active_requests: int = 3,
):
    active_requests = []
    all_requests = []

    if pending_queue is None:
        pending_queue = deque()

    pending_queue = deque(
        list(initial_requests) + list(pending_queue)
    )

    while active_requests or pending_queue:
        while pending_queue and len(active_requests) < max_active_requests:
            req = pending_queue.popleft()
            
            active_requests.append(req)
            all_requests.append(req)
            print("admitted:", req.request_id)

        print("active requests: ", [req.request_id for req in active_requests])

        results = step_batched_requests(active_requests)

        # # simulate streaming mode: return next token id to current client
        # for request_id, token_id in results.items():
        #     print(request_id, tokenizer.decode([token_id]))

        active_requests = [
            r for r in active_requests
            if not r.finished
        ]
    
    final_outputs = {}
    for r in all_requests:
        final_outputs[r.request_id] = tokenizer.decode(
            r.generated_token_ids,
            skip_special_tokens=True
        )
    print("final_outputs:")
    print(final_outputs)
    return final_outputs

