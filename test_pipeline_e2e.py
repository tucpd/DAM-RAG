"""
End-to-End Pipeline Test
Test đầy đủ: Image + Region → DAM → Retrieval → Synthesis
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from time import time

# Load environment variables từ .env
load_dotenv()

from modules.dam.inference import DAMInference
from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever
# from modules.synthesis.local_synthesizer import LocalLLMSynthesizer  # Replaced by DAM LLM 

def test_end_to_end_pipeline(image_path):
    """
    Test pipeline đầy đủ
    """
    print("="*70)
    print("END-TO-END PIPELINE TEST")
    print("="*70)
    
    start_time_set = time()
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
    
    # print("\n[4/5] Loading LLM synthesizer...")
    # synthesizer = LocalLLMSynthesizer(device=device)
    # print("Local LLM synthesizer loaded")
    
    # =================================================================
    # 2. Load test image
    # =================================================================
    print("\n[4/4] Loading test image...")
    
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image_path}")
    print(f"  Size: {image.size}")
    
    end_time_set = time()
    print(f"\nSetup completed in {end_time_set - start_time_set:.2f} seconds")
    
    # =================================================================
    # PIPELINE EXECUTION
    # =================================================================
    print("\n" + "="*70)
    print("PIPELINE EXECUTION")
    print("="*70)
    start_time_exe = time()
    
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
    start_time_cap = time()
    print("\n[STEP 4] Synthesizing final travel caption...")
    print("(Using DAM LLM - Llama-3.2-3B)")
    
    final_caption = dam.synthesize_with_knowledge(
        image=image,
        mask=None,  # Full image
        knowledge_items=results,
        style="informative",
        max_new_tokens=200,
        temperature=0.3
    )
    end_time_cap = time()
    print(f"\nCaption synthesized in {end_time_cap - start_time_cap:.2f} seconds")
    
    # =================================================================
    # RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("FINAL TRAVEL CAPTION")
    print("="*70)
    print(final_caption)
    
    print("\n" + "="*70)
    print("METADATA")
    print("="*70)
    print(f"Image: {image_path}")
    landmark_names = [item.get('name', item.get('landmark', 'Unknown')) for item in results]
    print(f"Retrieved landmarks: {', '.join(landmark_names)}")
    print(f"Style: informative")
    print(f"DAM Caption (initial): {dam_caption[:100]}...")
    
    print("\nPipeline completed successfully!")
    end_time_exe = time()
    print(f"\nExecution completed in {end_time_exe - start_time_exe:.2f} seconds")
    
    return {
        'caption': final_caption,
        'dam_caption': dam_caption,
        'retrieved_landmarks': landmark_names,
        'style': 'informative'
    }


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
    from glob import glob
    
    # Danh sách ảnh test từ các địa điểm khác nhau
    test_images = [
        "data/knowledge_base/Ha_Long_Bay/img_0004.jpg",
        "data/knowledge_base/Taj_Mahal/img_0005.jpg",
        "data/knowledge_base/Eiffel_Tower/img_0003.jpg",
        "data/knowledge_base/Great_Wall_of_China/img_0002.jpg",
        "data/knowledge_base/Angkor_Wat/img_0001.jpg",
    ]
    
    # Lọc ra những ảnh có tồn tại
    import os
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print("Không tìm thấy ảnh nào trong danh sách!")
        sys.exit(1)
    
    print("="*70)
    print(f"TESTING WITH {len(available_images)} IMAGES")
    print("="*70)
    for i, img in enumerate(available_images, 1):
        landmark_name = img.split('/')[-2].replace('_', ' ')
        print(f"  {i}. {landmark_name}: {img}")
    print()
    
    # Run tests
    results = []
    total_start = time()
    
    for idx, image_path in enumerate(available_images, 1):
        landmark_name = image_path.split('/')[-2].replace('_', ' ')
        
        print(f"\n{'='*70}")
        print(f"TEST {idx}/{len(available_images)}: {landmark_name}")
        print(f"{'='*70}\n")
        
        try:
            start_time = time()
            result = test_end_to_end_pipeline(image_path)
            end_time = time()
            
            results.append({
                'landmark': landmark_name,
                'image': image_path,
                'time': end_time - start_time,
                'success': True,
                'caption': result['caption'][:100] + '...' if len(result['caption']) > 100 else result['caption']
            })
            
        except Exception as e:
            print(f"\nError testing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'landmark': landmark_name,
                'image': image_path,
                'time': 0,
                'success': False,
                'error': str(e)
            })
    
    total_end = time()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"\nTotal time: {total_end - total_start:.2f} seconds")
    print(f"Average time per image: {(total_end - total_start) / len(results):.2f} seconds")
    
    print("\n" + "-"*70)
    print("DETAILED RESULTS:")
    print("-"*70)
    for i, r in enumerate(results, 1):
        if r['success']:
            print(f"\n{i}. {r['landmark']}")
            print(f"   Time: {r['time']:.2f}s")
            print(f"   Caption: {r['caption']}")
        else:
            print(f"\n{i}. {r['landmark']}")
            print(f"   Status: FAILED")
            print(f"   Error: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)

