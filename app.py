from fastapi import FastAPI
from pydantic import BaseModel
from generation import generate_with_cache, stream_with_cache
import uuid
from fastapi.responses import StreamingResponse

class UserRequest(BaseModel):
    query: str
    max_new_tokens: int

app = FastAPI(
    title = "model serving version 1",
    version = "0.1.0"
)

@app.post("/generate")
def generate(request: UserRequest):
    query = request.query.strip()

    request_id = str(uuid.uuid4())

    if not query:
        return {
            "status": "failed",
            "request_id": request_id,
            "response": "User input cannot be empty."
        }
    
    response = generate_with_cache(
        query,
        runtime_trace=True,
        max_new_tokens=request.max_new_tokens,
        request_id=request_id)

    return {
        "status": "success",
        "request_id": request_id,
        "response": response
    }


@app.post("/stream")
def stream(request: UserRequest):
    request_id = str(uuid.uuid4())
    query = request.query.strip()
    
    if not query:
        return {
            "status": "failed",
            "request_id": request_id,
            "response": "User input cannot be empty."
        }
    
    return StreamingResponse(
        stream_with_cache(
            query,
            runtime_trace=True,
            max_new_tokens=request.max_new_tokens,
            request_id=request_id),
        media_type="text/plain",
        headers={"X-Request-Id": request_id}
    )