"""
Script test DAM inference
Test với ảnh mẫu từ internet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dam import DAMInference
import torch
import requests
from PIL import Image
from io import BytesIO

def download_sample_image(url: str) -> Image.Image:
    """Download ảnh mẫu từ URL"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    response = requests.get(url, verify=False)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def main():
    print("=" * 80)
    print("TEST DAM-3B INFERENCE")
    print("=" * 80)
    
    # 1. Khởi tạo DAM model
    print("\n[1] Đang khởi tạo DAM model...")
    dam = DAMInference(
        model_path="nvidia/DAM-3B",
        conv_mode="v1",
        prompt_mode="full+focal_crop",
        device="cuda"
    )
    
    # 2. Download ảnh mẫu (Angkor Wat - temple nổi tiếng)
    print("\n[2] Đang tải ảnh mẫu...")
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Angkor_Wat.jpg/800px-Angkor_Wat.jpg"
    
    try:
        image = download_sample_image(sample_url)
        print(f"✓ Đã tải ảnh: {image.size}")
        
        # Lưu ảnh
        os.makedirs("data/images", exist_ok=True)
        image.save("data/images/test_angkor.jpg")
        print("✓ Đã lưu ảnh tại: data/images/test_angkor.jpg")
        
    except Exception as e:
        print(f"✗ Lỗi khi tải ảnh: {e}")
        print("Sử dụng ảnh local thay thế...")
        return
    
    # 3. Test inference - mô tả toàn bộ ảnh
    print("\n[3] Đang sinh caption cho toàn bộ ảnh...")
    caption_full = dam.generate_caption(
        image=image,
        max_new_tokens=200,
        temperature=0.2
    )
    
    print("\n" + "="*80)
    print("CAPTION (Full image):")
    print("="*80)
    print(caption_full)
    print("="*80)
    
    # 4. Test với bounding box (trung tâm ngôi đền)
    print("\n[4] Đang sinh caption cho vùng trung tâm (box)...")
    w, h = image.size
    box = (int(w*0.3), int(h*0.2), int(w*0.7), int(h*0.8))
    
    caption_box = dam.generate_caption(
        image=image,
        box=box,
        max_new_tokens=200,
        temperature=0.2
    )
    
    print("\n" + "="*80)
    print(f"CAPTION (Box region: {box}):")
    print("="*80)
    print(caption_box)
    print("="*80)
    
    print("\n✓ Test hoàn tất!")
    print("\nLưu ý: Caption được sinh bằng TIẾNG ANH vì DAM được train trên dữ liệu tiếng Anh.")

if __name__ == "__main__":
    main()
