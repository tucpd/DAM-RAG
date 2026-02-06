# DAM-RAG Travel Captioner

Automated travel caption generation system based on **user-selected image regions**, combining accurate visual descriptions (from DAM-3B) with factual knowledge from Wikimedia Commons.

## Key Features

- **English-first approach** - Leveraging DAM-3B and Wikimedia Commons data quality
- **No text query required** - Simply select an image region
- **Global knowledge base** - Landmarks, monuments, and architecture worldwide
- **Three-stage pipeline** - DAM captioning → Visual retrieval → DAM synthesis
- **Optimized performance** - ~10s per image with 7GB VRAM

## Project Structure

```
DAM-RAG/
├── modules/
│   ├── dam/
│   │   ├── inference.py          # DAM-3B wrapper for region captioning
│   │   └── DAM/                  # DAM-3B model package
│   ├── retrieval/
│   │   ├── crawler.py            # Wikimedia Commons crawler (10 landmarks)
│   │   ├── embedder.py           # CLIP visual embedder (768-dim)
│   │   ├── retriever.py          # FAISS vector similarity search
│   │   └── build_vector_index.py # Build FAISS index from images
│   └── synthesis/
│       ├── local_synthesizer.py  # Qwen2.5-7B caption synthesizer
│       └── llm_synthesizer.py    # (Deprecated) Gemini API version
├── data/
│   ├── knowledge_base/           # Crawled images by landmark
│   │   ├── Taj_Mahal/
│   │   ├── Eiffel_Tower/
│   │   └── ...                   # 10 landmarks total
│   └── vector_index/             # FAISS index + metadata
├── test_pipeline_e2e.py          # End-to-end pipeline test
└── test_dam_simple.py            # Simple DAM test

```

## Module Details

### Module 1: DAM (Describe Anything Model)

**Status:** ✅ Complete

Generate detailed English captions for image regions using NVIDIA's DAM-3B model.

**File:** [`modules/dam/inference.py`](modules/dam/inference.py)

**Key Class:** `DAMInference`

**Usage:**
```python
from modules.dam.inference import DAMInference

dam = DAMInference(device="cuda")
caption = dam.generate_caption(
    image=image,           # PIL Image
    mask=None,            # Optional: binary mask
    box=(x1, y1, x2, y2)  # Optional: bounding box
)
```

### Module 2: Visual Embedding & Retrieval

**Status:** ✅ Complete

Retrieve relevant landmark information using CLIP embeddings and FAISS vector search.

#### 2.1 Wikimedia Crawler

**File:** [`modules/retrieval/crawler.py`](modules/retrieval/crawler.py)

**Key Class:** `WikimediaCommonsCrawler`

**Features:**
- Crawl images from 10 famous landmarks
- Retry mechanism with exponential backoff
- Rate limiting handling (429 errors)
- Metadata extraction (name, description, year, style)

**Landmarks:**
- Taj Mahal, Great Wall of China, Eiffel Tower, Colosseum
- Statue of Liberty, Machu Picchu, Angkor Wat, Burj Khalifa
- Sydney Opera House, Ha Long Bay

**Run crawler:**
```bash
python modules/retrieval/crawler.py
```

#### 2.2 Visual Embedder

**File:** [`modules/retrieval/embedder.py`](modules/retrieval/embedder.py)

**Key Class:** `VisualEmbedder`

**Model:** CLIP ViT-Large-Patch14 (openai/clip-vit-large-patch14)

**Features:**
- Embed images to 768-dim vectors
- Support batch processing
- Normalized L2 embeddings

**Usage:**
```python
from modules.retrieval.embedder import VisualEmbedder

embedder = VisualEmbedder(device="cuda")
vector = embedder.embed_image(image)  # Returns: (768,) numpy array
```

#### 2.3 Vector Retriever

**File:** [`modules/retrieval/retriever.py`](modules/retrieval/retriever.py)

**Key Class:** `VectorRetriever`

**Backend:** FAISS-GPU with IndexFlatIP (inner product similarity)

**Features:**
- Fast similarity search on GPU
- Persistent index storage
- Metadata management

**Usage:**
```python
from modules.retrieval.retriever import VectorRetriever

# Load existing index
retriever = VectorRetriever.load("data/vector_index", use_gpu=True)

# Search
distances, metadata = retriever.search(query_vector, top_k=5)
```

#### 2.4 Build Vector Index

**File:** [`modules/retrieval/build_vector_index.py`](modules/retrieval/build_vector_index.py)

**Purpose:** Build FAISS index from all crawled images

**Features:**
- Collect images from all landmark folders
- Batch embedding with CLIP (batch_size=16)
- Save FAISS index + metadata

**Run index builder:**
```bash
python modules/retrieval/build_vector_index.py
```

### Module 3: LLM Synthesis

**Status:** ✅ Complete (Optimized)

Synthesize final travel captions using DAM's built-in LLM (Llama-3.2-3B).

**File:** [`modules/dam/inference.py`](modules/dam/inference.py)

**Key Method:** `DAMInference.synthesize_with_knowledge()`

**Model:** Llama-3.2-3B (integrated in DAM-3B)

**Features:**
- Combines DAM visual understanding + retrieved knowledge
- Multiple styles: informative, casual
- 80-200 words output
- **Optimized:** No need to load separate LLM, saves ~14GB VRAM
- **Fast:** ~2s synthesis time (60x faster than Qwen)

**Usage:**
```python
from modules.dam.inference import DAMInference

dam = DAMInference(device="cuda")
caption = dam.synthesize_with_knowledge(
    image=image,
    mask=None,
    knowledge_items=top_k_results,
    style="informative"
)
```

## Installation

### Prerequisites

- **GPU:** NVIDIA RTX 4080 16GB or equivalent (CUDA 12.x)
- **Python:** 3.10
- **Conda:** Recommended for environment management

### Setup Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd DAM-RAG

# 2. Create conda environment
conda create -n dr python=3.10 -y
conda activate dr

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install DAM package
pip install git+https://github.com/NVlabs/describe-anything
```

### Verify Installation

```bash
python test_dam_simple.py
```

## Pipeline Workflow

### Step 1: Crawl Landmark Data

Crawl images famous landmarks on Wikimedia Commons.

```bash
# Run crawler
python modules/retrieval/crawler.py
```

**Output:**
- `data/knowledge_base/{landmark_name}/*.jpg` - Downloaded images
- `data/knowledge_base/{landmark_name}/metadata.json` - Metadata for each landmark

**Expected structure:**
```
data/knowledge_base/
├── Taj_Mahal/
│   ├── Taj_Mahal_12345.jpg
│   ├── Taj_Mahal_67890.jpg
│   └── metadata.json
├── Eiffel_Tower/
│   └── ...
└── (8 more landmarks)
```

### Step 2: Build Vector Index

Build FAISS index from all crawled images using CLIP embeddings.

```bash
# Build index
python modules/retrieval/build_vector_index.py
```

**Output:**
- `data/vector_index/faiss.index` - FAISS index with 768-dim vectors
- `data/vector_index/metadata.json` - Metadata for each vector

**Expected output:**
```
Collecting images from knowledge base...
Found 244 images across 10 landmarks

Building FAISS index...
Processing: 100%|██████████| 16/16 [00:04<00:00]
Successfully built index with 244 vectors
```

### Step 3: Run End-to-End Pipeline

Test the complete pipeline: DAM → Retrieval → Synthesis.

```bash
# Run end-to-end test
python test_pipeline_e2e.py
```

**Pipeline stages:**

1. **DAM Captioning** - Generate detailed English description of image region
2. **CLIP Embedding** - Convert image to 768-dim vector
3. **FAISS Retrieval** - Find top-5 similar landmarks
4. **LLM Synthesis** - Generate travel caption combining visual + knowledge

**Expected output:**
```
[STEP 1] DAM - Generating detailed caption...
DAM Caption: A majestic stone temple with intricate carvings...

[STEP 2] Embedding image region...
Embedding shape: (768,)

[STEP 3] Retrieval - Finding similar landmarks...
Top-5 Retrieved:
  1. Angkor Wat (distance: 0.00)
  2. Angkor Wat (distance: 0.33)
  3. Angkor Wat (distance: 0.35)
  ...

[STEP 4] Synthesis - Generating travel caption...
Final Caption:
Step into a world of ancient grandeur as you gaze upon this 
intricately crafted stone wall, reminiscent of the majestic 
Angkor Wat complex in Cambodia...
```

## Custom Usage

### Process Your Own Image

```python
from PIL import Image
from modules.dam.inference import DAMInference
from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever

# Load models
dam = DAMInference(device="cuda")
embedder = VisualEmbedder(device="cuda")
retriever = VectorRetriever.load("data/vector_index", use_gpu=True)

# Load your image
image = Image.open("your_image.jpg").convert("RGB")

# Step 1: Embed image
vector = embedder.embed_image(image)

# Step 2: Retrieve landmarks
distances, metadata = retriever.search(vector, top_k=5)

# Step 3: Synthesize caption with DAM (combines visual + knowledge)
final_caption = dam.synthesize_with_knowledge(
    image=image,
    mask=None,
    knowledge_items=metadata,
    style="informative"
)

print(final_caption)
```

### Process Specific Region

```python
# Option 1: Using bounding box
caption = dam.generate_caption(
    image=image,
    box=(100, 100, 400, 400)  # (x1, y1, x2, y2)
)

# Option 2: Using binary mask
import numpy as np
mask = np.zeros((height, width), dtype=np.uint8)
mask[100:400, 100:400] = 1  # Region of interest
caption = dam.generate_caption(image=image, mask=mask)

# Then continue with embedding → retrieval → synthesis
```

## Technical Specifications

### Models

| Component | Model | Size | Precision | VRAM |
|-----------|-------|------|-----------|------|
| DAM | nvidia/DAM-3B (incl. Llama-3.2-3B) | 3B | FP16 | ~6GB |
| CLIP | openai/clip-vit-large-patch14 | 427M | FP16 | ~1GB |
| **Total** | - | - | - | **~7GB** |

**Previous (deprecated):**
| LLM | Qwen/Qwen2.5-7B-Instruct | 7B | FP16 | ~14GB |
| Total (with Qwen) | - | - | - | ~21GB |

### Dataset

- **Source:** Wikimedia Commons
- **Landmarks:** 10 famous locations worldwide
- **Images:** ~244 total (~24 per landmark)
- **Vector Index:** 244 CLIP embeddings (768-dim each)

### Performance (RTX 4080 16GB)

**Optimized Pipeline (DAM LLM):**
- **DAM inference:** ~2-3s per image
- **CLIP embedding:** ~0.1s per image
- **FAISS retrieval:** <0.01s for top-5
- **DAM synthesis:** ~1.5-2s per caption
- **Total pipeline:** ~5-10s per image
- **VRAM usage:** ~7GB

**Previous (Qwen):**
- LLM synthesis: ~103-127s per caption
- Total pipeline: ~130-140s per image
- VRAM usage: ~21GB

**Improvement:** ~13.7x faster, 14GB less VRAM

## Development Status

- ✅ Module 1: DAM inference (complete)
- ✅ Module 2: RAG retrieval (complete - 244 images indexed)
- ✅ Module 3: LLM synthesis (complete - optimized with DAM LLM)
- ✅ Performance optimization (13.7x speedup)
- ⏳ Gradio/FastAPI demo (planned)
- ⏳ Documentation expansion (planned)

## Known Limitations

1. **English-only output** - DAM-3B trained primarily on English data
2. **Limited landmarks** - Currently 10 landmarks (~244 images)
3. **GPU required** - Models optimized for CUDA inference
4. **VRAM requirement** - Need ~7GB for optimized pipeline

## Troubleshooting

### Out of Memory (OOM)

The optimized pipeline requires ~7GB VRAM. If you encounter OOM:

```python
# Option 1: Reduce batch size in retrieval
retriever.search(vector, top_k=3)  # Instead of 5

# Option 2: Use smaller max_new_tokens
dam.synthesize_with_knowledge(..., max_new_tokens=150)  # Instead of 200
```

### Legacy Qwen Pipeline

If you need the old Qwen-based synthesis (slower but different style):

```python
from modules.synthesis.local_synthesizer import LocalLLMSynthesizer

synthesizer = LocalLLMSynthesizer(device="cuda")
caption = synthesizer.synthesize(
    dam_caption=dam.generate_caption(image),
    retrieved_knowledge=metadata,
    style="informative"
)
```

## Citation

If you use this project, please cite:

```bibtex
@software{dam_rag_2026,
  title={DAM-RAG Travel Captioner},
  author={tucpd},
  year={2026},
  url={https://github.com/tucpd/DAM-RAG.git}
}

@article{dam2025,
  title={Describe Anything Model},
  author={NVIDIA Research},
  journal={arXiv preprint arXiv:2504.16072},
  year={2025}
}
```

## License

[Specify your license here]

## Acknowledgments

- **DAM-3B:** NVIDIA Research - [Paper](https://arxiv.org/abs/2504.16072) | [Code](https://github.com/NVlabs/describe-anything)
- **CLIP:** OpenAI - [Paper](https://arxiv.org/abs/2103.00020)
- **Qwen2.5:** Alibaba Cloud - [Model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Wikimedia Commons:** Community-contributed images and metadata