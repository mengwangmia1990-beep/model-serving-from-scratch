from transformers import AutoTokenizer
import config
import time

model_name = config.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)

class RequestState:
    def __init__(
        self,
        request_id: str,
        prompt: str,
        max_new_tokens: int,
        runtime_trace: bool = True,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.runtime_trace = runtime_trace

        inputs = tokenizer(prompt, return_tensors="pt")
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

        self.prompt_tokens = self.input_ids.shape[1]

        self.current_input_ids = self.input_ids
        self.past_key_values = None

        self.generated_token_ids = []
        self.generated_tokens = 0

        self.finished = False
        self.hit_eos = False

        self.request_start_time = time.perf_counter()
        self.first_token_time = None
        self.request_end_time = None
