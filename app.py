from fastapi import FastAPI
from pydantic import BaseModel
from generation import generate_with_cache

class UserRequest(BaseModel):
    query: str

app = FastAPI(
    title = "model serving version 1",
    version = "0.1.0"
)

@app.post("/generate")
def generate(request: UserRequest):
    query = request.query.strip()

    if not query:
        return {
            "status": "failed",
            "response": "User input cannot be empty.",
            "runtime_trace": None
        }
    
    response, trace = generate_with_cache(query)

    return {
        "status": "success",
        "response": response,
        "runtime_trace": trace
    }