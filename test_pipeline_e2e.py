"""
End-to-End Pipeline Test
Test đầy đủ: Image + Region → DAM → Retrieval → Synthesis
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables từ .env
load_dotenv()

from modules.dam.inference import DAMInference
from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever
# from modules.synthesis.llm_synthesizer import LLMSynthesizer  # Gemini
from modules.synthesis.local_synthesizer import LocalLLMSynthesizer  # Local LLM

def test_end_to_end_pipeline(image_path):
    """
    Test pipeline đầy đủ
    """
    print("="*70)
    print("END-TO-END PIPELINE TEST")
    print("="*70)
    
    # =================================================================
    # Setup
    # =================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # =================================================================
    # 1. Load models
    # =================================================================
    print("\n[1/5] Loading DAM model...")
    dam = DAMInference(device=device)
    print("DAM loaded")
    
    print("\n[2/5] Loading CLIP embedder...")
    embedder = VisualEmbedder(device=device)
    print("CLIP embedder loaded")
    
    print("\n[3/5] Loading FAISS retriever...")
    retriever = VectorRetriever.load("data/vector_index", use_gpu=True if device == "cuda" else False)
    print(f"FAISS retriever loaded ({retriever.index.ntotal} vectors, {len(retriever.metadata)} metadata)")
    
    print("\n[4/5] Loading LLM synthesizer...")
    synthesizer = LocalLLMSynthesizer(device=device)
    print("Local LLM synthesizer loaded")
    
    # =================================================================
    # 2. Load test image
    # =================================================================
    print("\n[5/5] Loading test image...")
    
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image_path}")
    print(f"  Size: {image.size}")
    
    # =================================================================
    # PIPELINE EXECUTION
    # =================================================================
    print("\n" + "="*70)
    print("PIPELINE EXECUTION")
    print("="*70)
    
    # Step 1: DAM - Generate caption
    print("\n[STEP 1] DAM - Generating detailed caption...")
    dam_caption = dam.generate_caption(
        image=image,
        mask=None  # Full image
    )
    print(f"\nDAM Caption:\n{dam_caption}")
    
    # Step 2: Embedding - Convert image to vector
    print("\n[STEP 2] Embedding image region...")
    # Embed toàn bộ ảnh (hoặc có thể embed region cụ thể)
    image_embedding = embedder.embed_image(image)
    print(f"Embedding shape: {image_embedding.shape}")
    print(f"  Norm: {np.linalg.norm(image_embedding):.4f}")
    
    # Step 3: Retrieval - Tìm landmarks tương tự
    print("\n[STEP 3] Retrieving similar landmarks...")
    distances, results = retriever.search(image_embedding, top_k=3)
    
    print("\nTop-3 retrieved landmarks:")
    for i, (dist, meta) in enumerate(zip(distances, results), 1):
        name = meta.get('name', meta.get('landmark', 'Unknown'))
        location = meta.get('location', '')
        if location:
            print(f"  {i}. {name} ({location}) - Distance: {dist:.4f}")
        else:
            print(f"  {i}. {name} - Distance: {dist:.4f}")
    
    # Step 4: Synthesis - Tạo travel caption
    print("\n[STEP 4] Synthesizing final travel caption...")
    print("(Using local LLM...)")
    
    final_result = synthesizer.synthesize(
        dam_caption=dam_caption,
        retrieved_knowledge=results,
        style="informative"
    )
    
    # =================================================================
    # RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("FINAL TRAVEL CAPTION")
    print("="*70)
    print(final_result['caption'])
    
    print("\n" + "="*70)
    print("METADATA")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Retrieved landmarks: {', '.join(final_result['retrieved_landmarks'])}")
    print(f"Style: {final_result['style']}")
    
    print("\nPipeline completed successfully!")
    
    return final_result


def test_with_custom_region(image_path):
    """
    Test với vùng ảnh cụ thể (sử dụng box)
    """
    print("\n" + "="*70)
    print("TEST WITH CUSTOM REGION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    dam = DAMInference(device=device)
    embedder = VisualEmbedder(device=device)
    retriever = VectorRetriever.load("data/vector_index", use_gpu=True if device == "cuda" else False)
    synthesizer = LocalLLMSynthesizer(device=device)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Define box (x1, y1, x2, y2) - center region
    w, h = image.size
    box = [w//4, h//4, 3*w//4, 3*h//4]
    
    print(f"\nImage: {image_path} ({w}x{h})")
    print(f"Region: box {box}")
    
    # Pipeline
    print("\n[1/4] DAM caption...")
    caption = dam.generate_caption(image=image, box=box)
    print(f"Caption: {caption[:100]}...")
    
    print("\n[2/4] Embedding region...")
    embedding = embedder.embed_region(image, box)
    
    print("\n[3/4] Retrieval...")
    _, results = retriever.search(embedding, top_k=2)
    
    print("\n[4/4] Synthesis...")
    final = synthesizer.synthesize_simple(caption, results)
    
    print("\n" + "="*70)
    print(final)
    print("="*70)


if __name__ == "__main__":
    import sys
    # Đường dẫn ảnh
    image_path = "data/knowledge_base/Angkor_Wat/Angkor_Wat_14993680.jpg"
    
    # Run tests
    try:
        # Test 1: Full image
        result = test_end_to_end_pipeline(image_path)
        
        # Test 2: Custom region (optional)
        # test_with_custom_region(image_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

