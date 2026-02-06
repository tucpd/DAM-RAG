---
applyTo: '**/*.py, **/*.md, '
---
## Tên dự án tạm thời: DAM-RAG Travel Captioner
Mục tiêu chính: Xây dựng hệ thống tự động sinh caption du lịch chi tiết dựa trên **vùng ảnh người dùng chọn**, kết hợp mô tả hình ảnh chính xác + thông tin thực tế (lịch sử, kiến trúc, văn hóa, đặc điểm địa lý…) mà **KHÔNG cần người dùng nhập text query**.

## Đặc điểm quan trọng hiện tại của dự án
- Chúng ta **sử dụng DAM-3B** (Describe Anything Model) làm core để sinh mô tả chi tiết vùng ảnh.
- DAM hiện tại chỉ được huấn luyện chủ yếu trên dữ liệu **tiếng Anh** → output caption mặc định sẽ bằng **tiếng Anh**.
- Dữ liệu knowledge base sẽ được crawl chủ yếu từ **Wikimedia Commons** → bao gồm rất nhiều địa danh, di tích, cảnh quan, công trình kiến trúc **nước ngoài** (không chỉ Việt Nam).
- Vì vậy hệ thống hiện tại sẽ ưu tiên:
  - Caption mô tả bằng **tiếng Anh** (chất lượng cao nhất từ DAM)
  - Knowledge snippets cũng bằng **tiếng Anh** (từ Wikimedia Commons)
  - Synthesis LLM có thể giữ nguyên tiếng Anh hoặc dịch sang ngôn ngữ khác nếu cần (nhưng giai đoạn đầu ưu tiên giữ tiếng Anh để đảm bảo độ chính xác)

## Kiến trúc tổng thể (Agentic Pipeline – các module độc lập)

1. Input: Một ảnh + một vùng được chỉ định (region)
   - Region có thể là: point (click), box, scribble, hoặc mask
   - Sẽ dùng SAM / SAM 2 để chuyển point/box/scribble thành mask nếu cần

2. Module 1 – DAM (Describe Anything Model)
   - Sử dụng mô hình DAM-3B đã release (không train lại, inference zero-shot)
   - Repo chính thức: https://github.com/NVlabs/describe-anything
   - Paper: https://arxiv.org/abs/2504.16072
   - Nhiệm vụ: Sinh **detailed localized caption bằng tiếng Anh** cho vùng được chọn
   - Output mong muốn:
     - Text: caption chi tiết bằng tiếng Anh (ví dụ: "A majestic ancient stone temple with intricate carvings, multi-tiered roofs, and golden spires, surrounded by lush tropical greenery...")
     - (Tùy chọn) Regional visual features (từ localized vision backbone, sau gated cross-attention) để làm embedding cho RAG

3. Module 2 – Visual Embedding & Retrieval (RAG dựa trên hình ảnh)
   - Chỉ embed **vùng ảnh được chọn** (focal crop hoặc regional features từ DAM)
   - KHÔNG dùng text query từ người dùng
   - Cách embed ưu tiên:
     - Lấy regional features trực tiếp từ DAM (z' sau gated cross-attention) → projector xuống 512/768 dim
     - Hoặc: crop vùng theo focal prompt → embed bằng SigLIP / CLIP image encoder
   - Retrieval: tìm Top-K thông tin liên quan từ knowledge base (Wikimedia Commons)
   - Knowledge base chứa: thông tin về di tích, công trình kiến trúc, cảnh quan, địa danh trên thế giới (lịch sử xây dựng, kiến trúc sư, phong cách nghệ thuật, ý nghĩa văn hóa, năm xây dựng…)

4. Module 3 – Synthesis (LLM cuối)
   - Input:
     - Caption chi tiết bằng tiếng Anh từ DAM
     - Top-K knowledge snippets bằng tiếng Anh từ RAG
   - Output: Caption cuối cùng phong cách hướng dẫn du lịch, **ưu tiên giữ tiếng Anh** để đảm bảo độ chính xác và mạch lạc
   - Ví dụ mong muốn:
     "This is the Angkor Wat temple complex in Cambodia, the largest religious monument in the world, built in the early 12th century during the Khmer Empire. The central tower rises 65 meters, surrounded by intricate bas-reliefs depicting Hindu epics. Best viewed at sunrise when the five towers are reflected in the moat."

## Yêu cầu quan trọng khi Copilot hỗ trợ code
- Bạn phải sử dụng tiếng Việt để phản hồi cho tôi trong suốt dự án này.
- Nếu bạn cần chạy câu lệnh trong terminal, hãy kích hoạt môi trường conda: conda activate dr
- Tôi đang dùng **RTX 4080 16GB VRAM** → ưu tiên giữ FP16/BF16, không cần 4-bit trừ khi VRAM thực sự thiếu
- Sử dụng **DAM-3B** inference (không train lại, không LoRA)
- Tối ưu inference: torch.compile, torch.inference_mode, flash-attention nếu có
- Synthesis LLM: ưu tiên dùng API như Gemini để thử nghiệm trước, sau khi thành công thì dùng model nhỏ chạy local (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B)
- Code cần modular: tách riêng các agent/module (DAM inference, embedding, retrieval, synthesis)
- Khi viết code, hãy ưu tiên:
  - Dùng huggingface transformers để load DAM
  - Hỗ trợ input mask / box / point
  - Xử lý batch size = 1 trước, sau đó mở rộng nếu cần
  - Có comment rõ ràng giải thích từng bước
  - Lưu ý rằng output của DAM hiện tại là tiếng Anh
- Sau mỗi module, mỗi bước hoặc 1 phần code quan trọng, bạn hãy giải thích các bước bạn đã làm, nhưng phần đã làm được, sau đó viết lệnh commit và push lên nhánh GitHub, bạn chỉ gợi ý câu lệnh còn việc commit và push để tôi tự thực hiện.
- Trường hợp bạn viết file test, sau khi test xong hãy xóa file test đó đi để giữ repo gọn gàng
- Hạn chết tối đa việc tạo ra các file .md không cần thiết trong repo, chỉ cần file readme chính và các file hướng dẫn quan trọng
- Trong code/ comment, file readme hoặc file hướng dẫn tuyệt đối không được thêm icons

## Phong cách caption mong muốn (giai đoạn đầu)
- Chi tiết, sinh động, chính xác về hình ảnh (từ DAM)
- Bổ sung thông tin thực tế, hữu ích cho du khách (từ Wikimedia Commons)
- Ngắn gọn đến trung bình (80–200 từ)
- Giọng điệu chuyên nghiệp, thông tin, hấp dẫn
- Ngôn ngữ chính: **tiếng Anh** (để tận dụng tối đa chất lượng của DAM và dữ liệu Wikimedia Commons)
- Sau này có thể thêm bước dịch sang tiếng Việt hoặc các ngôn ngữ khác nếu cần

## Các bước triển khai tôi đang hướng tới
1. Chạy được DAM-3B inference + sinh caption vùng bằng tiếng Anh trên máy local
2. Crawl và xây retrieval index từ **Wikimedia Commons** (ảnh + caption + metadata địa danh, di tích, kiến trúc)
3. Kết nối DAM caption + RAG → synthesis bằng tiếng Anh
4. Làm Gradio / FastAPI demo để test trên các ảnh du lịch quốc tế (sẽ thực hiện sau khi hoàn thiện 3 bước trên)

Khi tôi hỏi code hoặc debug, hãy dựa vào kiến trúc này để trả lời.
Cảm ơn Copilot!