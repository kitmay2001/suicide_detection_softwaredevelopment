from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import json
import re
import pandas as pd
import os

set_seed(0)

# Load tokenizer and model (adjust path or model name as needed)
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # or your local model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Create pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)


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

    output = llm(full_prompt, max_new_tokens=512)[0]['generated_text']

    matches = re.findall(r"\{(?:[^{}]|\n)*?\}", output, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


# Load dataset
dataset_df = pd.read_csv("c-ssrs.csv")
posts = dataset_df["content"].tolist()
post_ids = dataset_df["id"].tolist()

filename = f"output.jsonl"
failed_filename = filename.replace(".jsonl", "_failed.jsonl")

os.makedirs(os.path.dirname(filename), exist_ok=True)

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

        if data:  # valid → save
            data["post_index"] = post_ids[i]
            f.write(json.dumps(data) + "\n")
            f.flush()
        else:  # failed after max retries → log separately
            fail_entry = {
                "post_index": post_ids[i],
                "content": post,
                "raw_outputs": output_data if output_data else None
            }
            f_failed.write(json.dumps(fail_entry) + "\n")
            f_failed.flush()

print(f"Saved valid assessments to {filename}")
print(f"Saved failed samples to {failed_filename}")
