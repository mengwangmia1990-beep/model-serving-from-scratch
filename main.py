from generation import generate_with_cache, stream_with_cache, run_scheduler
import uuid
from models.request_state import RequestState
from collections import deque

def test_scheduler():
    initial_requests = [
        RequestState("r1", "where is seattle?", 1, True),
        RequestState("r2", "what is AI?", 2, True),
        RequestState("r3", "diff between cat and dog?", 2, True),
        RequestState("r4", "how big is the universe?", 1, True)
    ]

    pending_queue = deque([
        RequestState("r5", "please briefly explain kv cache", 1, True),
        RequestState("r6", "is san juan island a good place for whale watching?", 2, True),
    ])

    max_active_requests = 3
    response = run_scheduler(initial_requests, pending_queue, max_active_requests)
    print(response)

def test_stream():
    query = "where is seattle located at?"
    request_id = str(uuid.uuid4())

    for text in stream_with_cache(query, True, 5, request_id):
        print(text, end="", flush=True)
        print()

def main():
    test_scheduler()


if __name__ == "__main__":
    main()