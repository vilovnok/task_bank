"""
Mapping and resolution for model endpoint routing
"""

ENDPOINT_MAP = {
    # Gemini family (Google)
    "google/gemini": "chat",
    "google/gemini-3.6": "chat",
    "google/gemini-3.7": "chat",
    "google/gemini-3.6-flash": "chat",
    "google/gemini-3.8-flash": "chat",
    "qwen/qwen3.7-flash": "chat",
    # Terra variants (OpenAI) - default to chat, unless known otherwise
    "openai/gpt-5.6-terra": "responses",
    # OpenAI GPT-5.6 family should route to /responses (web-search via /responses)
    "openai/gpt-5.6": "responses",
    "openai/gpt-5.6-sol": "responses",
    # Add more mappings as needed
}

def resolve_model_endpoint(model: str) -> str:
    if not model:
        return "chat"
    m = model.lower()
    for key, endpoint in ENDPOINT_MAP.items():
        if key in m:
            return endpoint
    return "chat"

def get_model_endpoint(model: str) -> str:
    return resolve_model_endpoint(model)
