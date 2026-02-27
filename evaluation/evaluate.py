"""
Evaluation Script for DAM-RAG Pipeline
Chay cac thuc nghiem: RA@k, CLIPScore, KU-Score 
So sanh baselines: DAM-Only, DAM-RAG, Text-Query RAG
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
import random
import os
import sys
import re
import argparse
import unicodedata
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from time import time
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever
from modules.dam.inference import DAMInference


# ============================================================
# Name Normalization
# ============================================================

# Mapping cac ten khac nhau cua cung 1 landmark 
# (Wikipedia name -> directory name)
LANDMARK_ALIASES = {
    'great pyramid of giza': 'pyramids of giza',
    'pyramid of giza': 'pyramids of giza',
    'aurora': 'aurora borealis',
    'aurora (astronomy)': 'aurora borealis',
    'northern lights': 'aurora borealis',
    'fushimi inari-taisha': 'fushimi inari taisha',
    'fushimi inari shrine': 'fushimi inari taisha',
    "giant's causeway": 'giant causeway',
    'giants causeway': 'giant causeway',
    'mont-saint-michel': 'mont saint michel',
    'ha long bay': 'ha long bay',
    'imperial city, hue': 'imperial city hue',
    'imperial city hue': 'imperial city hue',
    'plitvice lakes national park': 'plitvice lakes',
}


def normalize_landmark_name(name):
    """
    Chuan hoa ten landmark de so sanh:
    1. Loai bo diacritics (Ha Long -> Ha Long, Hue -> Hue)
    2. Loai bo ky tu dac biet (apostrophe, hyphen, comma, parentheses)
    3. Lowercase
    4. Ap dung alias mapping
    5. Loai bo stop words (of, the, a)
    """
    if not name:
        return ''
    
    # 1. Unicode normalize - loai bo diacritics
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    # 2. Loai bo noi dung trong ngoac (vi du: "Aurora (astronomy)" -> "Aurora")
    ascii_name = re.sub(r'\([^)]*\)', '', ascii_name)
    
    # 3. Thay the ky tu dac biet bang space
    ascii_name = re.sub(r"['\-,_./]", ' ', ascii_name)
    
    # 4. Lowercase va loai bo whitespace thua
    ascii_name = ' '.join(ascii_name.lower().split())
    
    # 5. Ap dung alias mapping
    if ascii_name in LANDMARK_ALIASES:
        ascii_name = LANDMARK_ALIASES[ascii_name]
    
    # 6. Loai bo stop words
    stop_words = {'of', 'the', 'a', 'an', 'in', 'at', 'by', 'on'}
    tokens = [w for w in ascii_name.split() if w not in stop_words]
    
    return ' '.join(tokens)


def landmark_names_match(name1, name2):
    """
    So sanh 2 ten landmark sau khi chuan hoa.
    Returns True neu 2 ten la cung 1 landmark.
    """
    n1 = normalize_landmark_name(name1)
    n2 = normalize_landmark_name(name2)
    
    if not n1 or not n2:
        return False
    
    # Exact match after normalization
    if n1 == n2:
        return True
    
    # Substring match (1 ten chua trong ten kia)
    if n1 in n2 or n2 in n1:
        return True
    
    # Token overlap: >60% tokens trung nhau
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2)
        min_len = min(len(tokens1), len(tokens2))
        if min_len > 0 and overlap / min_len >= 0.6:
            return True
    
    return False


# ============================================================
# Metric Functions
# ============================================================

def compute_clipscore(embedder, image, caption_text):
    """
    Tinh CLIPScore = max(100 * cos(CLIP_img, CLIP_txt), 0)
    
    Args:
        embedder: VisualEmbedder instance (da co CLIP model)
        image: PIL Image
        caption_text: str
    
    Returns:
        float: CLIPScore
    """
    # Embed image
    img_inputs = embedder.processor(
        images=image, return_tensors="pt"
    ).to(embedder.device)
    img_features = embedder.model.get_image_features(**img_inputs)
    img_features = F.normalize(img_features, p=2, dim=-1)

    # Embed text
    txt_inputs = embedder.processor(
        text=caption_text, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(embedder.device)
    txt_features = embedder.model.get_text_features(**txt_inputs)
    txt_features = F.normalize(txt_features, p=2, dim=-1)

    # Cosine similarity
    cos_sim = (img_features @ txt_features.T).item()
    score = max(100.0 * cos_sim, 0.0)
    return score


def compute_ku_score(caption, metadata):
    """
    Tinh Knowledge Utilization Score
    Kiem tra xem caption co chua cac fact tu metadata khong
    
    Facts duoc kiem tra: name, location/country, year_built, 
    unesco_status, style
    
    Args:
        caption: str - generated caption
        metadata: dict - metadata cua landmark
    
    Returns:
        float: KU-Score (0.0 - 1.0)
    """
    caption_lower = caption.lower()
    facts_checked = 0
    facts_found = 0

    # Fact 1: Landmark name
    name = metadata.get('name', '')
    if name:
        facts_checked += 1
        if name.lower() in caption_lower:
            facts_found += 1

    # Fact 2: Location / Country
    location = metadata.get('location', '')
    country = metadata.get('country', '')
    if location or country:
        facts_checked += 1
        loc_found = False
        if location and location.lower() in caption_lower:
            loc_found = True
        if country and country.lower() in caption_lower:
            loc_found = True
        if loc_found:
            facts_found += 1

    # Fact 3: Year built
    year_built = str(metadata.get('year_built', ''))
    if year_built and year_built.strip():
        facts_checked += 1
        # Tim so nam trong caption
        if year_built in caption:
            facts_found += 1
        else:
            # Thu tim century format ("12th century", "19th century"...)
            import re
            year_match = re.search(r'\d{3,4}', year_built)
            if year_match:
                year_num = year_match.group()
                if year_num in caption:
                    facts_found += 1

    # Fact 4: UNESCO status
    unesco = metadata.get('unesco_status', '')
    if unesco and 'heritage' in unesco.lower():
        facts_checked += 1
        if 'unesco' in caption_lower or 'heritage' in caption_lower or 'world heritage' in caption_lower:
            facts_found += 1

    # Fact 5: Architectural style
    style = metadata.get('style', '')
    if style and style.strip():
        facts_checked += 1
        if style.lower() in caption_lower:
            facts_found += 1

    if facts_checked == 0:
        return 0.0

    return facts_found / facts_checked


def compute_retrieval_accuracy(retriever, embedder, test_images, top_k_list=[1, 3, 5]):
    """
    Tinh Retrieval Accuracy @ k
    Cho moi anh query, kiem tra xem landmark dung co trong top-k retrieval khong
    
    Args:
        retriever: VectorRetriever instance
        embedder: VisualEmbedder instance
        test_images: list of dicts {'image_path': str, 'landmark': str}
        top_k_list: list of k values
    
    Returns:
        dict: {k: accuracy} e.g. {1: 0.85, 3: 0.95, 5: 0.98}
    """
    max_k = max(top_k_list)
    correct = {k: 0 for k in top_k_list}
    total = 0

    for item in tqdm(test_images, desc="Computing RA@k"):
        try:
            image = Image.open(item['image_path']).convert('RGB')
            embedding = embedder.embed_image(image)
            distances, results = retriever.search(embedding, top_k=max_k)

            true_landmark = item['landmark'].replace('_', ' ')

            for k in top_k_list:
                top_k_names = [
                    r.get('name', r.get('landmark', ''))
                    for r in results[:k]
                ]
                if any(landmark_names_match(true_landmark, name)
                       for name in top_k_names):
                    correct[k] += 1

            total += 1
        except Exception as e:
            print(f"  Error processing {item['image_path']}: {e}")
            continue

    accuracies = {k: correct[k] / total if total > 0 else 0.0 for k in top_k_list}
    return accuracies


# ============================================================
# Data Collection
# ============================================================

def collect_test_images(tests_dir="data/tests", metadata_dir="data/metadata"):
    """
    Thu thap danh sach anh test tu thu muc data/tests/
    (da duoc tach truoc boi split_test_data.py)
    
    Returns:
        list of dicts: [{'image_path', 'landmark', 'metadata'}, ...]
    """
    tests_path = Path(tests_dir)
    metadata_path = Path(metadata_dir)
    test_images = []

    if not tests_path.exists():
        print(f"Test directory not found: {tests_dir}")
        return test_images

    for landmark_dir in sorted(tests_path.iterdir()):
        if not landmark_dir.is_dir():
            continue

        landmark_name = landmark_dir.name
        meta_dir = metadata_path / landmark_name

        # Load landmark_info.json de lay metadata tong hop
        landmark_info = {}
        info_file = meta_dir / "landmark_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                landmark_info = json.load(f)

        # Lay tat ca anh trong thu muc test cua landmark nay
        img_files = sorted(
            list(landmark_dir.glob("*.jpg")) + list(landmark_dir.glob("*.png"))
        )

        for img_file in img_files:
            test_images.append({
                'image_path': str(img_file),
                'landmark': landmark_name,
                'metadata': landmark_info,
            })

    return test_images


# ============================================================
# Baseline: DAM-Only (Perception Agent only, no knowledge)
# ============================================================

def run_dam_only(dam, image, mask=None):
    """
    Baseline DAM-Only: Chi dung DAM caption, khong co knowledge
    """
    caption = dam.generate_caption(image=image, mask=mask)
    return caption


# ============================================================
# Method: DAM-RAG (Full pipeline)
# ============================================================

def run_dam_rag(dam, embedder, retriever, image, mask=None, top_k=3):
    """
    DAM-RAG pipeline: DAM caption + CLIP retrieval + DAM synthesis
    """
    # Step 1: Embed image
    embedding = embedder.embed_image(image)

    # Step 2: Retrieve knowledge
    distances, results = retriever.search(embedding, top_k=top_k)

    # Step 3: Synthesize with knowledge
    caption = dam.synthesize_with_knowledge(
        image=image,
        mask=mask,
        knowledge_items=results,
        style="informative",
        max_new_tokens=200,
        temperature=0.3
    )
    return caption, results


# ============================================================
# Baseline: Text-Query RAG
# ============================================================

def run_text_query_rag(dam, embedder, retriever, image, mask=None, top_k=3):
    """
    Text-Query RAG: DAM caption -> text embed -> retrieve -> synthesize
    Dung DAM caption lam query text thay vi image embedding
    """
    # Step 1: Generate initial DAM caption
    dam_caption = dam.generate_caption(image=image, mask=mask)

    # Step 2: Embed caption text (dung CLIP text encoder)
    txt_inputs = embedder.processor(
        text=dam_caption, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(embedder.device)

    with torch.no_grad():
        txt_features = embedder.model.get_text_features(**txt_inputs)
        txt_features = F.normalize(txt_features, p=2, dim=-1)

    text_embedding = txt_features.cpu().numpy()[0]

    # Step 3: Retrieve using text embedding
    distances, results = retriever.search(text_embedding, top_k=top_k)

    # Step 4: Synthesize with knowledge
    caption = dam.synthesize_with_knowledge(
        image=image,
        mask=mask,
        knowledge_items=results,
        style="informative",
        max_new_tokens=200,
        temperature=0.3
    )
    return caption, results, dam_caption


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation(
    top_k=3,
    device="cuda",
    output_file="evaluation/results.json"
):
    """
    Chay toan bo evaluation pipeline
    Test images duoc doc tu data/tests/ (da tach truoc)
    """
    print("=" * 70)
    print("DAM-RAG EVALUATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Top-k retrieval: {top_k}")

    # --------------------------------------------------------
    # 1. Collect test data (tu data/tests/)
    # --------------------------------------------------------
    print("\n[1/6] Collecting test images from data/tests/...")
    test_images = collect_test_images()
    print(f"  Total test images: {len(test_images)}")

    landmarks = set(item['landmark'] for item in test_images)
    print(f"  Total landmarks: {len(landmarks)}")

    # --------------------------------------------------------
    # 2. Load models
    # --------------------------------------------------------
    print("\n[2/6] Loading models...")
    t0 = time()

    print("  Loading DAM-3B...")
    dam = DAMInference(device=device)

    print("  Loading CLIP embedder...")
    embedder = VisualEmbedder(device=device)

    print("  Loading FAISS retriever...")
    retriever = VectorRetriever.load("data/vector_index", use_gpu=False)
    print(f"  FAISS loaded: {retriever.index.ntotal} vectors")

    print(f"  Models loaded in {time() - t0:.1f}s")

    # --------------------------------------------------------
    # 3. Retrieval Accuracy
    # --------------------------------------------------------
    print("\n[3/6] Computing Retrieval Accuracy (RA@k)...")
    ra_scores = compute_retrieval_accuracy(
        retriever, embedder, test_images, top_k_list=[1, 3, 5]
    )
    print(f"  RA@1: {ra_scores[1]:.4f}")
    print(f"  RA@3: {ra_scores[3]:.4f}")
    print(f"  RA@5: {ra_scores[5]:.4f}")

    # --------------------------------------------------------
    # 4. Run methods & compute metrics
    # --------------------------------------------------------
    print("\n[4/6] Running methods on test images...")

    results_per_method = {
        'DAM-Only': [],
        'DAM-RAG': [],
        'Text-Query-RAG': [],
    }

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

        # --- DAM-Only ---
        t1 = time()
        dam_only_caption = run_dam_only(dam, image)
        dam_only_time = time() - t1

        dam_only_clip = compute_clipscore(embedder, image, dam_only_caption)
        dam_only_ku = compute_ku_score(dam_only_caption, metadata)

        results_per_method['DAM-Only'].append({
            'landmark': landmark,
            'image_path': img_path,
            'caption': dam_only_caption,
            'clipscore': dam_only_clip,
            'ku_score': dam_only_ku,
            'time': dam_only_time,
        })

        # --- DAM-RAG (Ours) ---
        t1 = time()
        dam_rag_caption, retrieved = run_dam_rag(dam, embedder, retriever, image, top_k=top_k)
        dam_rag_time = time() - t1

        dam_rag_clip = compute_clipscore(embedder, image, dam_rag_caption)
        dam_rag_ku = compute_ku_score(dam_rag_caption, metadata)

        # Kiem tra retrieval co dung landmark khong
        top1_name = retrieved[0].get('name', '') if retrieved else ''
        true_name = landmark.replace('_', ' ')
        retrieval_correct = landmark_names_match(true_name, top1_name)

        results_per_method['DAM-RAG'].append({
            'landmark': landmark,
            'image_path': img_path,
            'caption': dam_rag_caption,
            'clipscore': dam_rag_clip,
            'ku_score': dam_rag_ku,
            'time': dam_rag_time,
            'retrieval_correct': retrieval_correct,
            'retrieved_names': [r.get('name', '') for r in retrieved],
        })

        # --- Text-Query RAG ---
        t1 = time()
        tqr_caption, tqr_retrieved, tqr_dam_caption = run_text_query_rag(
            dam, embedder, retriever, image, top_k=top_k
        )
        tqr_time = time() - t1

        tqr_clip = compute_clipscore(embedder, image, tqr_caption)
        tqr_ku = compute_ku_score(tqr_caption, metadata)

        tqr_top1_name = tqr_retrieved[0].get('name', '') if tqr_retrieved else ''
        tqr_correct = landmark_names_match(true_name, tqr_top1_name)

        results_per_method['Text-Query-RAG'].append({
            'landmark': landmark,
            'image_path': img_path,
            'caption': tqr_caption,
            'clipscore': tqr_clip,
            'ku_score': tqr_ku,
            'time': tqr_time,
            'retrieval_correct': tqr_correct,
            'retrieved_names': [r.get('name', '') for r in tqr_retrieved],
            'text_query': tqr_dam_caption,
        })

        # In ket qua nhanh
        print(f"    DAM-Only:       CLIPScore={dam_only_clip:.2f}  KU={dam_only_ku:.2f}  Time={dam_only_time:.2f}s")
        print(f"    DAM-RAG:        CLIPScore={dam_rag_clip:.2f}  KU={dam_rag_ku:.2f}  Time={dam_rag_time:.2f}s  Ret={retrieval_correct}")
        print(f"    Text-Query-RAG: CLIPScore={tqr_clip:.2f}  KU={tqr_ku:.2f}  Time={tqr_time:.2f}s  Ret={tqr_correct}")

    # --------------------------------------------------------
    # 5. Aggregate Results
    # --------------------------------------------------------
    print("\n[5/6] Aggregating results...")

    summary = {}
    for method, results in results_per_method.items():
        if not results:
            continue

        clip_scores = [r['clipscore'] for r in results]
        ku_scores = [r['ku_score'] for r in results]
        times = [r['time'] for r in results]

        method_summary = {
            'num_samples': len(results),
            'clipscore_mean': float(np.mean(clip_scores)),
            'clipscore_std': float(np.std(clip_scores)),
            'ku_score_mean': float(np.mean(ku_scores)),
            'ku_score_std': float(np.std(ku_scores)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
        }

        # Retrieval accuracy cho methods co retrieval
        if 'retrieval_correct' in results[0]:
            ret_correct = sum(1 for r in results if r.get('retrieval_correct', False))
            method_summary['retrieval_accuracy'] = ret_correct / len(results)

        summary[method] = method_summary

    # --------------------------------------------------------
    # 6. Print & Save
    # --------------------------------------------------------
    print("\n[6/6] Results Summary")
    print("=" * 70)
    print(f"{'Method':<20} {'CLIPScore':>10} {'KU-Score':>10} {'Time (s)':>10} {'Ret Acc':>10}")
    print("-" * 70)

    for method, s in summary.items():
        ret_acc = f"{s.get('retrieval_accuracy', 0):.4f}" if 'retrieval_accuracy' in s else "N/A"
        print(f"{method:<20} {s['clipscore_mean']:>10.2f} {s['ku_score_mean']:>10.4f} {s['time_mean']:>10.2f} {ret_acc:>10}")

    print("-" * 70)

    # RA@k
    print(f"\nRetrieval Accuracy (Visual-Only CLIP):")
    print(f"  RA@1: {ra_scores[1]:.4f}")
    print(f"  RA@3: {ra_scores[3]:.4f}")
    print(f"  RA@5: {ra_scores[5]:.4f}")

    # Text-Query retrieval accuracy
    tqr_results = results_per_method.get('Text-Query-RAG', [])
    if tqr_results:
        tqr_ra1 = sum(1 for r in tqr_results if r.get('retrieval_correct', False)) / len(tqr_results)
        print(f"\nRetrieval Accuracy (Text-Query):")
        print(f"  RA@1: {tqr_ra1:.4f}")

    # Per-landmark breakdown
    print(f"\nPer-Landmark CLIPScore (DAM-RAG):")
    landmark_clips = defaultdict(list)
    for r in results_per_method.get('DAM-RAG', []):
        landmark_clips[r['landmark']].append(r['clipscore'])

    for lm, scores in sorted(landmark_clips.items()):
        print(f"  {lm.replace('_', ' '):<30} {np.mean(scores):.2f} (n={len(scores)})")

    # Save full results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        'meta': {
            'date': datetime.now().isoformat(),
            'device': device,
            'top_k': top_k,
            'num_landmarks': len(landmarks),
            'num_test_images': len(test_images),
            'faiss_vectors': retriever.index.ntotal,
        },
        'retrieval_accuracy': {str(k): v for k, v in ra_scores.items()},
        'summary': summary,
        'detailed_results': {
            method: results for method, results in results_per_method.items()
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_path}")
    print("=" * 70)

    return full_results


# ============================================================
# Scalability Analysis
# ============================================================

def run_scalability_analysis(
    kb_sizes=[10, 20, 54],
    top_k=3,
    device="cuda",
    output_file="evaluation/scalability_results.json"
):
    """
    Phan tich anh huong cua KB size den retrieval accuracy va CLIPScore
    Test voi cac tap con cua knowledge base
    """
    print("=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    embedder = VisualEmbedder(device=device)

    # Lay danh sach tat ca landmarks tu test set
    tests_path = Path("data/tests")
    all_landmarks = sorted([d.name for d in tests_path.iterdir() if d.is_dir()])
    print(f"Total landmarks available: {len(all_landmarks)}")

    # Load full retriever
    full_retriever = VectorRetriever.load("data/vector_index", use_gpu=False)

    # Thu thap test images
    all_test_images = collect_test_images()

    results = []

    for kb_size in kb_sizes:
        print(f"\n--- KB Size: {kb_size} landmarks ---")

        if kb_size >= len(all_landmarks):
            # Dung full KB
            subset_landmarks = all_landmarks
        else:
            # Chon ngau nhien kb_size landmarks
            random.seed(42)
            subset_landmarks = random.sample(all_landmarks, kb_size)

        # Filter retriever metadata de chi giu landmarks trong subset
        subset_landmark_set = set(l.replace('_', ' ') for l in subset_landmarks)

        # Dem so vectors trong subset
        subset_count = 0
        for meta in full_retriever.metadata:
            name = meta.get('name', meta.get('landmark', ''))
            folder = meta.get('folder', '')
            if name in subset_landmark_set or folder in subset_landmarks:
                subset_count += 1

        # Compute RA@k chi cho test images thuoc subset landmarks
        subset_test = [t for t in all_test_images if t['landmark'] in subset_landmarks]

        if not subset_test:
            print(f"  No test images for this subset, skipping")
            continue

        ra = compute_retrieval_accuracy(
            full_retriever, embedder, subset_test, top_k_list=[1, 3, 5]
        )

        print(f"  Landmarks: {len(subset_landmarks)}")
        print(f"  Test images: {len(subset_test)}")
        print(f"  KB vectors (est): {subset_count}")
        print(f"  RA@1: {ra[1]:.4f}  RA@3: {ra[3]:.4f}  RA@5: {ra[5]:.4f}")

        results.append({
            'kb_size': kb_size,
            'num_landmarks': len(subset_landmarks),
            'num_test_images': len(subset_test),
            'kb_vectors': subset_count,
            'ra_at_1': ra[1],
            'ra_at_3': ra[3],
            'ra_at_5': ra[5],
        })

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'date': datetime.now().isoformat(),
                'total_landmarks': len(all_landmarks),
            },
            'scalability': results,
        }, f, indent=2)

    print(f"\nScalability results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAM-RAG Evaluation")
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "scalability", "retrieval_only"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of test images per landmark (default: 3)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Top-k retrieval (default: 3)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results.json",
        help="Output file path"
    )

    args = parser.parse_args()

    if args.mode == "full":
        run_evaluation(
            top_k=args.top_k,
            device=args.device,
            output_file=args.output,
        )
    elif args.mode == "scalability":
        run_scalability_analysis(
            device=args.device,
            output_file="evaluation/scalability_results.json",
        )
    elif args.mode == "retrieval_only":
        # Chi chay retrieval accuracy
        print("Loading models...")
        embedder = VisualEmbedder(device=args.device)
        retriever = VectorRetriever.load("data/vector_index", use_gpu=False)
        test_images = collect_test_images()
        print(f"Test images: {len(test_images)}")

        ra = compute_retrieval_accuracy(
            retriever, embedder, test_images, top_k_list=[1, 3, 5]
        )
        print(f"RA@1: {ra[1]:.4f}")
        print(f"RA@3: {ra[3]:.4f}")
        print(f"RA@5: {ra[5]:.4f}")
