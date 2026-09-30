import urllib.request
import json
import io
from pathlib import Path

target_path = Path("evaluation/visualizations/full_diffusion_test/01_input_with_gray_mask.png")
mask_path = Path("evaluation/visualizations/full_diffusion_test/02_binary_mask.png")
ref1_path = Path("user_tests/test1/ref1.jpg")

target_bytes = target_path.read_bytes()
mask_bytes = mask_path.read_bytes()
ref1_bytes = ref1_path.read_bytes()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()

def add_file(field, filename, data, ctype="image/png"):
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode("utf-8"))
    body.write(data)
    body.write(b"\r\n")

def add_field(field, val):
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{val}\r\n".encode("utf-8"))

add_file("target", "target.png", target_bytes)
add_file("mask", "mask.png", mask_bytes)
add_file("ref_0", "ref1.jpg", ref1_bytes, "image/jpeg")
add_field("execution_mode", "preview")
body.write(f"--{boundary}--\r\n".encode("utf-8"))

req = urllib.request.Request("http://127.0.0.1:5000/api/mode1/restore", data=body.getvalue())
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode())

print("Preview Status:", data["status"])
print("Status Badge:", data["status_badge"])
print("Is Preview:", data["is_preview"])
print("Masked Pixels:", data["masked_pixels"])
print("References Used:", data["references_used"])
print("Mode Routed:", data["mode_routed"])
