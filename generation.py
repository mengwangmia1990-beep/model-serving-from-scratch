from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import config
import time

model_name = config.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
max_new_tokens = config.MAX_NEW_TOKENS

def generate_with_cache(prompt: str, runtime_trace: bool = True) -> str:
    trace = {}

    request_start = time.perf_counter()
    first_token_id = None
    generated_tokens = 0
    hit_eos = False

    inputs = tokenizer(
        prompt, 
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    prompt_tokens = input_ids.shape[1]

    attention_mask = inputs["attention_mask"]

    past_key_values = None

    current_input_ids = input_ids
    generated_token_ids = input_ids
        
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

        if first_token_id is None:
            first_token_time = time.perf_counter()
            first_token_id = next_token_id

        if next_token_id.item() == tokenizer.eos_token_id:
            hit_eos = True
            break

        # append next token to generated token list
        generated_token_ids = torch.cat([generated_token_ids, next_token_id], dim=1)
        
        current_input_ids = next_token_id
        
        next_attention = torch.ones_like(next_token_id)
        attention_mask = torch.cat([attention_mask, next_attention], dim=1)

        generated_tokens += 1

    # decode
    result = tokenizer.decode(generated_token_ids[0])

    request_end = time.perf_counter()

    if runtime_trace is True:
        total_tokens = prompt_tokens + generated_tokens
        end_to_end_latency_ms = (request_end - request_start) * 1000
        ttft_ms = (first_token_time - request_start) * 1000
        tokens_per_second = generated_tokens / (request_end - request_start)

        trace = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
            "max_new_tokens": max_new_tokens,
            "hit_eos": hit_eos,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "ttft_ms": ttft_ms,
            "tokens_per_second": tokens_per_second,
        }
    else:
        trace = None

    return result, trace