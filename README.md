# DAM-RAG Travel Captioner

Hệ thống tự động sinh caption du lịch chi tiết dựa trên **vùng ảnh người dùng chọn**, kết hợp mô tả hình ảnh chính xác (từ DAM-3B) + thông tin thực tế từ knowledge base (Wikimedia Commons).

## Đặc điểm chính

- **Ngôn ngữ chính: Tiếng Anh** - Tận dụng tối đa chất lượng của DAM-3B và dữ liệu Wikimedia Commons
- **Không cần text query** - Chỉ cần chọn vùng ảnh
- **Knowledge base quốc tế** - Địa danh, di tích, kiến trúc trên toàn thế giới

## Kiến trúc

### Module 1: DAM (Describe Anything Model) ✓ Hoàn thành

- Mô hình: DAM-3B (nvidia/DAM-3B)
- Chức năng: Sinh caption chi tiết tiếng Anh cho vùng được chọn
- Hỗ trợ: mask, box, full image
- Status: Đã test thành công

```python
from modules.dam import DAMInference

# Khởi tạo
dam = DAMInference(
    model_path="nvidia/DAM-3B",
    device="cuda"
)

# Sinh caption
caption = dam.generate_caption(
    image="path/to/image.jpg",
    box=(x1, y1, x2, y2),  # optional
    max_new_tokens=200,
    temperature=0.2
)
```

### Module 2: Visual Embedding & Retrieval (Đang phát triển)

- Embed vùng ảnh được chọn
- Retrieval từ Wikimedia Commons
- Lấy thông tin về địa danh, kiến trúc, lịch sử

### Module 3: Synthesis LLM (Đang phát triển)

- Kết hợp caption DAM + knowledge snippets
- Output: Caption phong cách hướng dẫn du lịch (tiếng Anh)

## Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd DAM-RAG

# Tạo conda environment
conda create -n dr python=3.10 -y
conda activate dr

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt DAM package
pip install git+https://github.com/NVlabs/describe-anything
```

## Test

```bash
python test_dam_simple.py
```

## Requirements

- GPU: RTX 4080 16GB VRAM (hoặc tương đương)
- Python: 3.10
- CUDA: 12.x
- Conda environment: `dr`

## Tiến độ

- [x] Module 1: DAM inference
- [ ] Module 2: RAG retrieval
- [ ] Module 3: LLM synthesis
- [ ] Demo Gradio/FastAPI