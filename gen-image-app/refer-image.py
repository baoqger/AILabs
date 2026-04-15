import os
import base64
import mimetypes
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_KEY"]
model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# Your Foundry resource endpoint (same as curl)
ENDPOINT = endpoint

# The curl used: Authorization: Bearer $AZURE_API_KEY
AZURE_API_KEY = subscription_key
API_VERSION = api_version


def file_to_b64(path: str) -> str:
    data = Path(path).read_bytes()
    filestr = base64.b64encode(data).decode("utf-8")
    return filestr


def save_bytes_as_image(content: bytes, out_path: str, content_type: str | None):
    ext = ".jpg"
    if content_type:
        if "png" in content_type.lower():
            ext = ".png"
        elif "jpeg" in content_type.lower() or "jpg" in content_type.lower():
            ext = ".jpg"

    out = Path(out_path)
    if out.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        out = out.with_suffix(ext)

    out.write_bytes(content)
    print(f"✅ Saved: {out.resolve()}")


def call_flux2pro_two_images(
    image_base_path: str,
    image_ref_path: str,
    prompt: str,
    out_path: str = "flux2pro_out.jpg",
):
    
    url = f"{ENDPOINT}?api-version={API_VERSION}"

    payload = {
        "model": model,
        "prompt": prompt,
        "input_image": file_to_b64(image_base_path),
        "input_image_2": file_to_b64(image_ref_path),
        #"safety_tolerance": 4, 
        "output_format": "jpeg",
        # "seed": 42,
        # "resolution": "1 MP",
        # "aspect_ratio": 1,
        # "width": 1024,
        # "height": 1024,
    }

    headers = {
        "Authorization": f"Bearer {AZURE_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "*/*",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    if not resp.ok:
        try:
            print("❌ Error JSON:", resp.json())
        except Exception:
            print("❌ Error Text:", resp.text[:2000])
        resp.raise_for_status()


    ct = resp.headers.get("Content-Type", "")
    if ct.startswith("image/"):
        save_bytes_as_image(resp.content, out_path, ct)
        return

    data = resp.json()
    b64_img = (
        data.get("image")
        or data.get("b64_image")
        or (data.get("data", [{}])[0].get("b64_json") if isinstance(data.get("data"), list) else None)
    )
    if not b64_img:
        raise RuntimeError(f"Unexpected response shape. Content-Type={ct}. Keys={list(data.keys())}")

    img_bytes = base64.b64decode(b64_img)
    save_bytes_as_image(img_bytes, out_path, "image/jpeg")


if __name__ == "__main__":
    
    prompt = (
        "Use Image 1 as the base (the room interior). Use Image 2 as the exact artwork reference (the horse image). Task: Place the horse image from Image 2 into the room in Image 1 as a realistic framed wall painting."
    )

    call_flux2pro_two_images(
        image_base_path=r"./room.png",
        image_ref_path=r"./horse.png",
        prompt=prompt,
        out_path="flux2pro_result.jpg",
    ) 