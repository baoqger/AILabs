import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]

# Your Foundry resource endpoint (same as curl)
ENDPOINT = endpoint

# The curl used: Authorization: Bearer $AZURE_API_KEY
AZURE_API_KEY = subscription_key

payload = {
    "prompt": "A photograph of a red fox in an autumn forest",
    "width": 1024,
    "height": 1024,
    "n": 1,
    "model": "FLUX.2-pro",
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AZURE_API_KEY}",
}

resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=180)
resp.raise_for_status()

data = resp.json()
b64_img = data["data"][0]["b64_json"]  # same jq path: .data[0].b64_json
img_bytes = base64.b64decode(b64_img)

out_path = "generated_image.png"
with open(out_path, "wb") as f:
    f.write(img_bytes)

print(f"Saved: {out_path}")