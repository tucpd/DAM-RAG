"""
Test DAM inference đơn giản với ảnh local
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.dam import DAMInference
import torch
from PIL import Image

print("="*80)
print("TEST DAM-3B INFERENCE (Simple)")
print("="*80)

# 1. Load DAM
print("\n[1] Đang load DAM model...")
dam = DAMInference(
    model_path="nvidia/DAM-3B",
    conv_mode="v1",
    prompt_mode="full+focal_crop",
    device="cuda"
)

# 2. Load ảnh test
print("\n[2] Load ảnh test...")
image_path = "data/images/test_simple.jpg"
image = Image.open(image_path)
print(f"✓ Đã load ảnh: {image.size}")

# 3. Test 1 - Mô tả toàn ảnh
print("\n[3] Test 1: Mô tả toàn bộ ảnh...")
caption1 = dam.generate_caption(
    image=image,
    max_new_tokens=150,
    temperature=0.2
)

print("\n" + "="*80)
print("CAPTION (Full Image):")
print("="*80)
print(caption1)
print("="*80)

# 4. Test 2 - Mô tả vùng bên trái (mặt trời)
print("\n[4] Test 2: Mô tả vùng có mặt trời (box: top-left)...")
w, h = image.size
box_sun = (0, 0, int(w*0.3), int(h*0.4))

caption2 = dam.generate_caption(
    image=image,
    box=box_sun,
    max_new_tokens=150,
    temperature=0.2
)

print("\n" + "="*80)
print(f"CAPTION (Sun region {box_sun}):")
print("="*80)
print(caption2)
print("="*80)

# 5. Test 3 - Mô tả vùng giữa (ngôi nhà)
print("\n[5] Test 3: Mô tả vùng ở giữa (ngôi nhà)...")
box_house = (int(w*0.3), int(h*0.2), int(w*0.7), int(h*0.8))

caption3 = dam.generate_caption(
    image=image,
    box=box_house,
    max_new_tokens=150,
    temperature=0.2
)

print("\n" + "="*80)
print(f"CAPTION (House region {box_house}):")
print("="*80)
print(caption3)
print("="*80)

print("\n✓ Test hoàn tất thành công!")
print("✓ DAM-3B có thể sinh caption tiếng Anh cho các vùng khác nhau trong ảnh")
