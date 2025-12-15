import time
import os
import json
import re
from huggingface_hub import InferenceClient

# 1. Initialize Client
# Ensure HF_TOKEN is set in your environment (or hardcode it for local testing)
token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not found. API calls might fail or be rate-limited.")

client = InferenceClient(token=token)

# 2. Define a Dummy Prompt (Same structure as your main script)
# We use a shorter text to test latency without wasting too many tokens.
dummy_post = "I feel a bit sad today, but I am going to watch a movie to cheer myself up."

prompt = (
    f"Evaluate this post for suicide severity (0-6). Post: {dummy_post}\n"
    "Respond with ONLY JSON: {\"severity\": int}"
)

# 3. Measure Latency
print("Sending request to Hugging Face API...")
start_time = time.time()

try:
    response = client.text_generation(
        prompt, 
        model="meta-llama/Llama-3.1-8B-Instruct", 
        max_new_tokens=50
    )
    end_time = time.time()
    
    # Calculate duration
    latency = end_time - start_time
    print(f"Success! Response received in {latency:.2f} seconds.")
    print(f"Raw Output: {response.strip()}")

    # 4. Fail if too slow (Optional assertion for CI)
    # If API takes longer than 10 seconds, fail the test
    if latency > 10.0:
        raise TimeoutError(f"API is too slow! Took {latency:.2f}s (Threshold: 10s)")

except Exception as e:
    print(f"Inference Failed: {e}")
    exit(1)
