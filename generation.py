from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import config

model_name = config.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
max_new_tokens = config.MAX_NEW_TOKENS

def generate_with_cache(prompt: str) -> str:
    inputs = tokenizer(
        prompt, 
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
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

        # layer0 = past_key_values.layers[0]
        # print(layer0.__dict__)

        # pick next token
        last_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True) # token Id is logits index

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # append next token to generated token list
        generated_token_ids = torch.cat([generated_token_ids, next_token_id], dim=1)
        
        current_input_ids = next_token_id
        
        next_attention = torch.ones_like(next_token_id)
        attention_mask = torch.cat([attention_mask, next_attention], dim=1)

    result = tokenizer.decode(generated_token_ids[0])
    # print(result)

    return result