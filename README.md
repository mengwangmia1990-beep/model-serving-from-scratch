# Model Serving From Scratch

A minimal LLM model serving project built with PyTorch, Hugging Face Transformers, KV Cache, and FastAPI.

The goal of this project is to understand how modern LLM inference works under the hood by implementing the decoding loop manually instead of relying on high-level generation APIs.  

## Features
- Token-by-token autoregressive decoding
- KV Cache based inference using `past_key_values`
- EOS token based early stopping
- FastAPI serving endpoint

## Example API
### Request
```json
{ 
    "query": 
    "where is seattle?"
}
```
### Response
```json
{ 
    "status": "success",
    "response": "where is seattle? stairs stairs stairs..." 
}
```

### Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```
### SwaggerUI
```text
http://127.0.0.1:8000/docs
```

## Next Steps
- Latency benchmarking
- Streaming generation
- Request batching
- Continuous batching
- vLLM-inspired optimizations
