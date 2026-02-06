"""
Build Vector Index
Tạo FAISS index
"""

import torch
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever


def collect_all_images(knowledge_base_dir: str = "data/knowledge_base"):
    """
    Thu thập tất cả ảnh từ knowledge base
    
    Returns:
        List of dicts: {'image_path': str, 'landmark': str, 'metadata': dict}
    """
    kb_path = Path(knowledge_base_dir)
    all_images = []
    
    for landmark_dir in kb_path.iterdir():
        if not landmark_dir.is_dir():
            continue
        
        landmark_name = landmark_dir.name
        
        # Load metadata
        metadata_file = landmark_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
        else:
            metadata_list = []
        
        # Tìm tất cả ảnh
        for img_file in landmark_dir.glob("*.jpg"):
            # Tìm metadata tương ứng
            img_metadata = None
            for meta in metadata_list:
                if Path(meta.get('image', '')).name == img_file.name:
                    img_metadata = meta
                    break
            
            if img_metadata is None:
                img_metadata = {
                    'landmark': landmark_name,
                    'image': str(img_file)
                }
            
            all_images.append({
                'image_path': str(img_file),
                'landmark': landmark_name,
                'metadata': img_metadata
            })
    
    return all_images


def build_vector_index(
    knowledge_base_dir: str = "data/knowledge_base",
    output_dir: str = "data/vector_index",
    device: str = "cuda",
    batch_size: int = 16
):
    """
    Build vector index từ tất cả ảnh
    """
    print("="*70)
    print("BUILD VECTOR INDEX FROM IMAGES")
    print("="*70)
    
    # 1. Collect all images
    print("\n[1/4] Collecting images...")
    all_images = collect_all_images(knowledge_base_dir)
    print(f"Found {len(all_images)} images")
    
    # Group by landmark
    landmarks_count = {}
    for img in all_images:
        landmark = img['landmark']
        landmarks_count[landmark] = landmarks_count.get(landmark, 0) + 1
    
    print("\nImages per landmark:")
    for landmark, count in sorted(landmarks_count.items()):
        print(f"  {landmark}: {count} images")
    
    # 2. Load embedder
    print("\n[2/4] Loading CLIP embedder...")
    embedder = VisualEmbedder(device=device)
    print("CLIP loaded")
    
    # 3. Embed all images
    print(f"\n[3/4] Embedding {len(all_images)} images...")
    print(f"Batch size: {batch_size}")
    
    embeddings = []
    metadata_list = []
    failed_count = 0
    
    for i in tqdm(range(0, len(all_images), batch_size), desc="Batches"):
        batch = all_images[i:i + batch_size]
        
        # Load images
        images = []
        batch_metadata = []
        
        for item in batch:
            try:
                img = Image.open(item['image_path']).convert('RGB')
                images.append(img)
                
                # Build metadata for retrieval
                meta = item['metadata'].copy()
                meta['name'] = item['landmark'].replace('_', ' ')
                meta['image_path'] = item['image_path']
                batch_metadata.append(meta)
                
            except Exception as e:
                print(f"\n⚠️  Failed to load {item['image_path']}: {e}")
                failed_count += 1
                continue
        
        if not images:
            continue
        
        # Embed batch
        try:
            batch_embeddings = embedder.embed_images_batch(images)
            embeddings.extend(batch_embeddings)
            metadata_list.extend(batch_metadata)
        except Exception as e:
            print(f"\nFailed to embed batch: {e}")
            failed_count += len(images)
    
    print(f"\nEmbedded {len(embeddings)} images")
    if failed_count > 0:
        print(f"Failed: {failed_count} images")
    
    # 4. Build FAISS index
    print("\n[4/4] Building FAISS index...")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    # Create retriever and add vectors
    retriever = VectorRetriever(
        embed_dim=embeddings_array.shape[1],
        use_gpu=True if device == "cuda" else False
    )
    
    retriever.add_vectors(embeddings_array, metadata_list)
    
    # Save
    retriever.save(output_dir)
    
    print(f"\nFull index saved to: {output_dir}")
    print(f"   Total vectors: {retriever.index.ntotal}")
    print(f"   Total metadata: {len(retriever.metadata)}")
    
    return retriever


def test_retrieval(retriever, test_image_path: str, top_k: int = 5):
    """Test retrieval với một ảnh query"""
    print("\n" + "="*70)
    print("TEST RETRIEVAL")
    print("="*70)
    
    print(f"\nQuery image: {test_image_path}")
    
    # Load and embed query image
    embedder = VisualEmbedder(device="cuda")
    query_img = Image.open(test_image_path).convert('RGB')
    query_embedding = embedder.embed_image(query_img)
    
    # Search
    distances, results = retriever.search(query_embedding, top_k=top_k)
    
    print(f"\nTop-{top_k} results:")
    for i, (dist, meta) in enumerate(zip(distances, results), 1):
        landmark = meta.get('name', meta.get('landmark', 'Unknown'))
        img_path = Path(meta.get('image_path', '')).name
        print(f"  {i}. {landmark} - {img_path} (Distance: {dist:.4f})")


if __name__ == "__main__":
    import sys
    
    # Build vector index
    retriever = build_vector_index(
        knowledge_base_dir="data/knowledge_base",
        output_dir="data/vector_index",
        device="cuda",
        batch_size=16
    )
    
    # Test với ảnh Angkor Wat
    test_images = [
        "data/knowledge_base/Angkor_Wat/img_0000.jpg",
        "data/knowledge_base/Taj_Mahal/img_0000.jpg",
        "data/knowledge_base/Eiffel_Tower/img_0000.jpg"
    ]
    
    for test_img in test_images:
        if Path(test_img).exists():
            test_retrieval(retriever, test_img, top_k=5)
