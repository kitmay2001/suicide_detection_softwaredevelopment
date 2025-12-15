import time
import os
import json
from huggingface_hub import InferenceClient

# 1. Initialize Client
token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not found. API calls might fail.")

client = InferenceClient(token=token)

# 2. Define the Prompt (Chat Format)
# For Llama-3-Instruct, we use a list of messages instead of a raw string
dummy_post = "I feel a bit sad today, but I am going to watch a movie to cheer myself up."

messages = [
    {
        "role": "user", 
        "content": (
            f"Evaluate this post for suicide severity (0-6). Post: {dummy_post}\n"
            "Respond with ONLY JSON: {\"severity\": int}"
        )
    }
]

# 3. Measure Latency
print("Sending request to Hugging Face API (Chat Mode)...")
start_time = time.time()

try:
    # UPDATED: Use chat_completion instead of text_generation
    response = client.chat_completion(
        messages=messages,
        model="meta-llama/Llama-3.1-8B-Instruct", 
        max_tokens=100
    )
    
    end_time = time.time()
    
    # Extract content
    # The response object structure is similar to OpenAI's API
    content = response.choices[0].message.content
    
    # Calculate duration
    latency = end_time - start_time
    print(f"Success! Response received in {latency:.2f} seconds.")
    print(f"Raw Output: {content}")

    if latency > 10.0:
        raise TimeoutError(f"API is too slow! Took {latency:.2f}s (Threshold: 10s)")

except Exception as e:
    print(f"Inference Failed: {e}")
    exit(1)
