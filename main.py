from generation import generate_with_cache, stream_with_cache
import uuid


def main():
    query = "where is seattle located at?"
    request_id = str(uuid.uuid4())

    for text in stream_with_cache(query, True, 5, request_id):
        print(text, end="", flush=True)
        print()


if __name__ == "__main__":
    main()