"""
Evaluation Script for LLaVA-1.6 Zero-shot Baseline
"""

import torch
import json
import numpy as np
import sys
import argparse
from pathlib import Path
from PIL import Image
from time import time
from datetime import datetime

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ImportError:
    print("Please install transformers: pip install transformers accelerate")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.retrieval.embedder import VisualEmbedder
from evaluation.evaluate import collect_test_images, compute_clipscore, compute_ku_score

# ============================================================
# Baseline: LLaVA-1.6 Zero-shot
# ============================================================

def run_llava_zero_shot(processor, model, image):
    """
    LLaVA-1.6 Zero-shot: Prompt LLaVA to describe the landmark without retrieved knowledge
    """
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this image and the landmark in it in detail, including historical and cultural context if applicable. ASSISTANT:"
    
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
        
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    del inputs
    del output
    
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:")[-1].strip()
    return caption


def run_evaluation(device="cuda", output_file=None):
    project_root = Path(__file__).parent.parent
    if output_file is None:
        output_file = project_root / "evaluation" / "llava_results.json"
    else:
        output_file = Path(output_file)
        if not output_file.is_absolute():
            output_file = project_root / output_file
            
    print("=" * 70)
    print("LLaVA-1.6 ZERO-SHOT EVALUATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # 1. Collect test data
    print("\n[1/4] Collecting test images from data/tests/...")
    test_images = collect_test_images()
    print(f"  Total test images: {len(test_images)}")

    landmarks = set(item['landmark'] for item in test_images)
    print(f"  Total landmarks: {len(landmarks)}")

    # 2. Load models
    print("\n[2/4] Loading models...")
    t0 = time()

    print("  Loading CLIP embedder (for CLIPScore)...")
    embedder = VisualEmbedder(device=device)

    print("  Loading LLaVA-1.6 (Zero-shot)...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    
    # Use 4-bit quantization to fit easily in 16GB VRAM
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        print("  Using 4-bit quantization (bitsandbytes) to save VRAM...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b-hf", 
            quantization_config=quantization_config,
            device_map="auto"
        )
    except ImportError:
        print("  [Warning] bitsandbytes not found, loading in float16 (might cause OOM).")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b-hf", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    print(f"  Models loaded in {time() - t0:.1f}s")

    # 3. Run evaluation
    print("\n[3/4] Running LLaVA-1.6 on test images...")

    results = []

    for idx, item in enumerate(test_images):
        img_path = item['image_path']
        landmark = item['landmark']
        metadata = item['metadata']
        
        print(f"\n  [{idx+1}/{len(test_images)}] {landmark} - {Path(img_path).name}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"    Error loading image: {e}")
            continue

        t1 = time()
        caption = run_llava_zero_shot(processor, model, image)
        inference_time = time() - t1

        clip_score = compute_clipscore(embedder, image, caption)
        ku_score = compute_ku_score(caption, metadata)

        results.append({
            'landmark': landmark,
            'image_path': img_path,
            'caption': caption,
            'clipscore': clip_score,
            'ku_score': ku_score,
            'time': inference_time,
        })

        print(f"    LLaVA-1.6: CLIPScore={clip_score:.2f}  KU={ku_score:.2f}  Time={inference_time:.2f}s")
        
        # Clear cache to prevent OOM across loop iterations
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Aggregate Results
    print("\n[4/4] Aggregating results...")

    clip_scores = [r['clipscore'] for r in results]
    ku_scores = [r['ku_score'] for r in results]
    times = [r['time'] for r in results]

    summary = {
        'num_samples': len(results),
        'clipscore_mean': float(np.mean(clip_scores)),
        'clipscore_std': float(np.std(clip_scores)),
        'ku_score_mean': float(np.mean(ku_scores)),
        'ku_score_std': float(np.std(ku_scores)),
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
    }

    print("\nResults Summary")
    print("=" * 70)
    print(f"{'Method':<20} {'CLIPScore':>10} {'KU-Score':>10} {'Time (s)':>10} {'Ret Acc':>10}")
    print("-" * 70)
    print(f"{'LLaVA-1.6-ZeroShot':<20} {summary['clipscore_mean']:>10.2f} {summary['ku_score_mean']:>10.4f} {summary['time_mean']:>10.2f} {'N/A':>10}")
    print("-" * 70)

    # Save full results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        'meta': {
            'date': datetime.now().isoformat(),
            'device': device,
            'num_landmarks': len(landmarks),
            'num_test_images': len(test_images),
        },
        'summary': {'LLaVA-1.6-ZeroShot': summary},
        'detailed_results': {'LLaVA-1.6-ZeroShot': results},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-1.6 Zero-shot Evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="evaluation/llava_results.json", help="Output file path")
    args = parser.parse_args()
    run_evaluation(device=args.device, output_file=args.output)
