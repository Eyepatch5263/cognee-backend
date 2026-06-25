import httpx
import asyncio

async def test():
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "user", "content": "hello"}
        ],
        "temperature": 0.2,
        "max_tokens": 50,
        "stream": False
    }
    print("Sending request to NVIDIA API...", flush=True)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            print(f"Status: {response.status_code}", flush=True)
            print(f"Response: {response.text}", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)

asyncio.run(test())
