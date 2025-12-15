import json
import re
import os
from huggingface_hub import InferenceClient

# Initialize the client (ensure HF_TOKEN is set in your environment variables)
# You can also pass token="hf_..." directly here if testing locally
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

# Required JSON keys
TEMPLATE_KEYS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "severity"]

def validate_json(data):
    """Check if all required keys exist and are non-empty/valid."""
    for key in TEMPLATE_KEYS:
        if key not in data or data[key] in ["", None]:
            return False
    try:
        sev = int(data["severity"])
        if not (0 <= sev <= 6):
            return False
    except Exception:
        return False
    return True

def get_assessment(post):
    p = ("\n Evaluate this post step by step and assign them severity on a scale of 0-6, "
         "where 0 means the post is not related to suicide, "
         "and 1-6 depending on the metrics above where 1 is for the first question and 6 for the last. \n")

    q = ('COLUMBIA-SUICIDE SEVERITY RATING SCALE metrics:\n'
         '1) Have you wished you were dead or wished you could go to sleep and not wake up? Severity: 1\n'
         '2) Have you actually had any thoughts about killing yourself? Severity: 2\n'
         '3) Have you been thinking about how you might do this? Severity: 3\n'
         '4) Have you had these thoughts and had some intention of acting on them? Severity: 4\n'
         '5) Have you started to work out or worked out the details of how to kill yourself? Did you intend to carry out this plan? Severity: 5\n'
         '6) Have you done anything, started to do anything, or prepared to do anything to end your life? Severity: 6\n')

    r = (
        "\n\n"
        "Respond with ONLY the completed JSON object in the following template:\n\n"
        "{"
        "\n  \"Q1\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"Q2\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"Q3\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"Q4\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"Q5\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"Q6\": \"Reasoning that fulfills this criteria or \\\"N/A\\\"\","
        "\n  \"severity\": integer (0-6)"
        "\n}\n"
        "---\n"
    )

    full_prompt = p + '\n' + q + '\n' + r + '\n\n post: ' + post + '\n---\n'
    
    try:
        # Call API
        response = client.text_generation(
            full_prompt, 
            model="meta-llama/Llama-3.1-8B-Instruct", 
            max_new_tokens=512
        )
        # Extract JSON from response
        matches = re.findall(r"\{(?:[^{}]|\n)*?\}", response, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load dataset
posts = ["I don't know what to do anymore, I just feel empty and hopeless I feel alone while I'm not really alone, I feel humiliated while no one humiliates me, I'm getting sick of this world day by day, even though this world hasn't really been bad for me. I got into this boring cycle, a cycle that may be normal and not that terrible, but it's definitely boring because there's nothing special about it except for the absurdity and emptiness. Pray for me so that I can escape from this cycle, whether the way to escape is life or death."]
post_ids = [0] # Created dummy ID to match the single post above

filename = "output.jsonl" # Simplified path for testing
failed_filename = filename.replace(".jsonl", "_failed.jsonl")

os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

with open(filename, "w") as f, open(failed_filename, "w") as f_failed:
    for i, post in enumerate(posts):
        data = None
        retries = 0
        max_retries = 5

        while retries < max_retries:
            output_data = get_assessment(post)
            if output_data and validate_json(output_data):
                data = output_data
                break
            else:
                retries += 1
                print(f"[Retry {retries}] post_index: {post_ids[i]} - invalid response")

        if data:  # valid -> save
            data["post_index"] = post_ids[i]
            f.write(json.dumps(data) + "\n")
            f.flush()
        else:  # failed after max retries -> log separately
            fail_entry = {
                "post_index": post_ids[i],
                "content": post,
                "raw_outputs": output_data if output_data else None
            }
            f_failed.write(json.dumps(fail_entry) + "\n")
            f_failed.flush()

print(f"Saved valid assessments to {filename}")
print(f"Saved failed samples to {failed_filename}")
