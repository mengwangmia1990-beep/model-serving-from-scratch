import requests

response = requests.post(
    "http://127.0.0.1:8000/stream",
    json={
        "query": "where is seattle located?",
        "max_new_tokens": 80
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode(), end="", flush=True)
