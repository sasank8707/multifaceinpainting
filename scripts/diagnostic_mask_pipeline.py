"""
Mask Pipeline Diagnostic Script.
Traces mask from browser/API upload through Flask, FaceInpaintingPipeline, and diffusion entry point.
Compares:
A) Direct Python call using user_tests/test1/target.jpg and user_tests/test1/mask.png
B) Web API call using the exact same files
"""

import io
import json
import urllib.request
import base64
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from PIL import Image
import cv2

# 1. Load exact files
target_path = Path("user_tests/test1/target.jpg")
mask_path = Path("user_tests/test1/mask.png")
ref1_path = Path("user_tests/test1/ref1.jpg")
ref2_path = Path("user_tests/test1/ref2.jpg")
ref3_path = Path("user_tests/test1/ref3.jpg")

target_bytes = target_path.read_bytes()
mask_bytes = mask_path.read_bytes()
ref1_bytes = ref1_path.read_bytes()
ref2_bytes = ref2_path.read_bytes()
ref3_bytes = ref3_path.read_bytes()

print("=" * 80)
print("STAGE 1: BROWSER / FILE SYSTEM INSPECTION")
print("=" * 80)
print(f"Mask filename:      {mask_path.name}")
print(f"Mask file byte size:{len(mask_bytes):,} bytes")
raw_pil = Image.open(io.BytesIO(mask_bytes))
print(f"PIL Image format:   {raw_pil.format}")
print(f"PIL Image mode:     {raw_pil.mode}")
print(f"PIL Image size:     {raw_pil.size} (Width={raw_pil.width}, Height={raw_pil.height})")

raw_arr = np.array(raw_pil)
print(f"NumPy shape:        {raw_arr.shape}")
print(f"Unique values:      {np.unique(raw_arr)}")
white_px = int(np.sum(raw_arr > 127))
print(f"White pixels (>127):{white_px:,}")
ys, xs = np.where(raw_arr > 127)
bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
print(f"Bounding box:       {bbox} (WxH = {bbox[2]-bbox[0]+1} x {bbox[3]-bbox[1]+1})")

print("\n" + "=" * 80)
print("STAGE 2: DIRECT PIPELINE MASK PROCESSING")
print("=" * 80)
from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from models.diffusion.diffusion_adapter import AspectRatioPreserver

target_np = np.array(Image.open(io.BytesIO(target_bytes)).convert("RGB"))
direct_mask_bin = (np.array(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 127).astype(np.uint8) * 255
h_t, w_t = target_np.shape[:2]
print(f"Target dimensions:  Width={w_t}, Height={h_t}")

# Pipeline validate_and_normalize_mask
pipeline = FaceInpaintingPipeline(mock_mode=True)
normalized_mask = pipeline.validate_and_normalize_mask(direct_mask_bin, (h_t, w_t))
print(f"Validated mask shape: {normalized_mask.shape}")
print(f"Validated white px:   {np.sum(normalized_mask > 0):,}")

# Letterbox before diffusion
masked_target = target_np.copy()
masked_target[normalized_mask > 0] = 0
pil_img, pil_mask, transform_info = AspectRatioPreserver.letterbox(
    image=masked_target,
    mask=normalized_mask,
    target_size=(512, 512)
)
letterbox_mask_np = np.array(pil_mask)
print(f"Letterbox 512x512 mask shape:    {letterbox_mask_np.shape}")
print(f"Letterbox mask white px (512x512):{np.sum(letterbox_mask_np > 127):,}")

# Unletterbox back
unletterboxed_mask = AspectRatioPreserver.unletterbox(
    padded_candidate=np.repeat(letterbox_mask_np[:, :, np.newaxis], 3, axis=-1),
    original_image=target_np,
    original_mask=normalized_mask,
    transform_info=transform_info
)[:, :, 0]
print(f"Unletterboxed mask white px:     {np.sum(unletterboxed_mask > 127):,}")

print("\n" + "=" * 80)
print("STAGE 3: WEB API CALL (PREVIEW ENDPOINT)")
print("=" * 80)
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()

def add_file(field, filename, data, ctype="image/png"):
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode("utf-8"))
    body.write(data)
    body.write(b"\r\n")

def add_field(field, val):
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{val}\r\n".encode("utf-8"))

add_file("target", "target.jpg", target_bytes, "image/jpeg")
add_file("mask", "mask.png", mask_bytes, "image/png")
add_file("ref_0", "ref1.jpg", ref1_bytes, "image/jpeg")
add_file("ref_1", "ref2.jpg", ref2_bytes, "image/jpeg")
add_file("ref_2", "ref3.jpg", ref3_bytes, "image/jpeg")
add_field("execution_mode", "preview")
body.write(f"--{boundary}--\r\n".encode("utf-8"))

req = urllib.request.Request("http://127.0.0.1:5000/api/mode1/restore", data=body.getvalue())
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
resp = urllib.request.urlopen(req)
api_data = json.loads(resp.read().decode())

print(f"Web API Status:            {api_data['status']}")
print(f"Web API Reported Mask Px:  {api_data['masked_pixels']:,}")

web_mask_bytes = base64.b64decode(api_data["mask_base64"].split(",")[1])
web_mask_np = np.array(Image.open(io.BytesIO(web_mask_bytes)))
print(f"Web API Returned Mask Shape:{web_mask_np.shape}")
print(f"Web API Returned White Px:  {np.sum(web_mask_np > 127):,}")

print("\n" + "=" * 80)
print("STAGE 4: PIXEL-BY-PIXEL COMPARISON (DIRECT vs WEB API)")
print("=" * 80)
direct_bin = (normalized_mask > 0)
web_bin = (web_mask_np > 127)

direct_px = int(np.sum(direct_bin))
web_px = int(np.sum(web_bin))
intersection = int(np.sum(direct_bin & web_bin))
only_direct = int(np.sum(direct_bin & ~web_bin))
only_web = int(np.sum(~direct_bin & web_bin))
bit_exact = (direct_px == web_px == intersection) and (only_direct == 0) and (only_web == 0)

print(f"Direct pipeline mask pixels: {direct_px:,}")
print(f"Web API mask pixels:         {web_px:,}")
print(f"Intersection:                {intersection:,}")
print(f"Only in direct mask:         {only_direct}")
print(f"Only in web mask:            {only_web}")
print(f"Are direct and web masks bit-exact identical?: {'YES' if bit_exact else 'NO'}")
print("=" * 80)
