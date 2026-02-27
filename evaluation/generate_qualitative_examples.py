"""
Generate qualitative examples for the paper.
Chay pipeline DAM-RAG tren 3 landmarks: Ha Long Bay, Taj Mahal, Machu Picchu
Luu caption va metrics vao file JSON
"""

import sys
import json
from pathlib import Path
from PIL import Image
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever
from modules.dam.inference import DAMInference
from evaluation.evaluate import compute_clipscore, compute_ku_score


def main():
    device = "cuda"
    
    # 3 landmarks x 1 anh moi landmark
    examples = [
        {
            "landmark": "Ha_Long_Bay",
            "image": "data/tests/Ha_Long_Bay/Ha_Long_Bay_124576580.jpg",
            "metadata_path": "data/metadata/Ha_Long_Bay/landmark_info.json",
        },
        {
            "landmark": "Taj_Mahal",
            "image": "data/tests/Taj_Mahal/Taj_Mahal_73871228.jpg",
            "metadata_path": "data/metadata/Taj_Mahal/landmark_info.json",
        },
        {
            "landmark": "Machu_Picchu",
            "image": "data/tests/Machu_Picchu/Machu_Picchu_159379095.jpg",
            "metadata_path": "data/metadata/Machu_Picchu/landmark_info.json",
        },
    ]
    
    # Load models
    print("Loading models...")
    dam = DAMInference(device=device)
    embedder = VisualEmbedder(device=device)
    retriever = VectorRetriever.load("data/vector_index", use_gpu=False)
    
    results = []
    
    for ex in examples:
        print(f"\n{'='*60}")
        print(f"Landmark: {ex['landmark']}")
        print(f"Image:    {ex['image']}")
        
        image = Image.open(ex['image']).convert('RGB')
        
        # Load metadata
        with open(ex['metadata_path'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 1. Perception Agent (DAM-Only)
        print("\n[Perception Agent]")
        t0 = time()
        perception_caption = dam.generate_caption(image=image, mask=None)
        perception_time = time() - t0
        print(f"  Caption ({len(perception_caption.split())} words): {perception_caption}")
        
        # 2. Knowledge Agent
        print("\n[Knowledge Agent]")
        embedding = embedder.embed_image(image)
        distances, retrieved = retriever.search(embedding, top_k=3)
        
        top1_name = retrieved[0].get('name', 'N/A') if retrieved else 'N/A'
        print(f"  Top-1: {top1_name}")
        for i, r in enumerate(retrieved):
            print(f"  Result {i+1}: {r.get('name', 'N/A')} (dist={distances[i]:.2f})")
        
        # Gathered knowledge
        knowledge_summary = {
            'name': retrieved[0].get('name', ''),
            'location': retrieved[0].get('location', ''),
            'country': retrieved[0].get('country', ''),
            'year_built': retrieved[0].get('year_built', ''),
            'style': retrieved[0].get('style', ''),
            'unesco_status': retrieved[0].get('unesco_status', ''),
        }
        
        # 3. Narration Agent (DAM-RAG with longer caption)
        print("\n[Narration Agent]")
        t0 = time()
        narration_caption = dam.synthesize_with_knowledge(
            image=image,
            mask=None,
            knowledge_items=retrieved,
            style="informative",
            max_new_tokens=300,  # Longer to ensure 80+ words
            temperature=0.3,
        )
        narration_time = time() - t0
        word_count = len(narration_caption.split())
        print(f"  Caption ({word_count} words): {narration_caption}")
        
        # Compute metrics
        clip_dam_only = compute_clipscore(embedder, image, perception_caption)
        clip_dam_rag = compute_clipscore(embedder, image, narration_caption)
        ku_dam_only = compute_ku_score(perception_caption, metadata)
        ku_dam_rag = compute_ku_score(narration_caption, metadata)
        
        print(f"\n[Metrics]")
        print(f"  DAM-Only:  CLIPScore={clip_dam_only:.2f}  KU={ku_dam_only:.2f}")
        print(f"  DAM-RAG:   CLIPScore={clip_dam_rag:.2f}  KU={ku_dam_rag:.2f}")
        
        result = {
            'landmark': ex['landmark'],
            'image_file': Path(ex['image']).name,
            'image_path': ex['image'],
            'perception_agent': {
                'caption': perception_caption,
                'word_count': len(perception_caption.split()),
                'time': perception_time,
                'clipscore': clip_dam_only,
                'ku_score': ku_dam_only,
            },
            'knowledge_agent': {
                'top1_name': top1_name,
                'retrieved': knowledge_summary,
                'distances': [float(d) for d in distances[:3]],
            },
            'narration_agent': {
                'caption': narration_caption,
                'word_count': word_count,
                'time': narration_time,
                'clipscore': clip_dam_rag,
                'ku_score': ku_dam_rag,
            },
        }
        results.append(result)
    
    # Save results
    output_path = Path("evaluation/qualitative_examples.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    
    # Print summary for paper
    print(f"\n{'='*60}")
    print("SUMMARY FOR PAPER")
    print(f"{'='*60}")
    for r in results:
        print(f"\n--- {r['landmark']} ({r['image_file']}) ---")
        print(f"Perception: {r['perception_agent']['caption'][:100]}...")
        print(f"Knowledge:  {r['knowledge_agent']['top1_name']}")
        print(f"Final:      {r['narration_agent']['caption'][:100]}...")
        print(f"Metrics:    CLIPScore={r['narration_agent']['clipscore']:.1f}  KU={r['narration_agent']['ku_score']:.2f}")


if __name__ == "__main__":
    main()
