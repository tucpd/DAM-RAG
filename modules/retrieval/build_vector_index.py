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


def collect_all_images(
    images_dir: str = "data/images",
    metadata_dir: str = "data/metadata"
):
    """
    Thu thap tat ca anh tu knowledge base
    
    Args:
        images_dir: Thu muc chua anh (data/images/{Landmark}/)
        metadata_dir: Thu muc chua metadata (data/metadata/{Landmark}/)
    
    Returns:
        List of dicts: {'image_path': str, 'landmark': str, 'metadata': dict}
    """
    images_path = Path(images_dir)
    metadata_path = Path(metadata_dir)
    all_images = []
    
    # Tim tat ca landmark folders trong thu muc metadata
    if not metadata_path.exists():
        print(f"Metadata directory not found: {metadata_dir}")
        return all_images
    
    landmark_dirs = sorted([
        d for d in metadata_path.iterdir() if d.is_dir()
    ])
    
    for landmark_meta_dir in landmark_dirs:
        landmark_name = landmark_meta_dir.name
        
        # Thu muc chua anh tuong ung
        img_dir = images_path / landmark_name
        
        # Load metadata tu landmark folder
        metadata_map = {}  # page_id hoac image_name -> metadata
        
        # Uu tien metadata.jsonl (format moi, co nhieu thong tin hon)
        jsonl_file = landmark_meta_dir / "metadata.jsonl"
        json_file = landmark_meta_dir / "metadata.json"
        
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        meta = json.loads(line)
                        # Map theo image_path (ten file)
                        img_path = meta.get('image_path', meta.get('local_path', ''))
                        if img_path:
                            img_name = Path(img_path).name
                            metadata_map[img_name] = meta
                        # Map theo page_id
                        page_id = meta.get('page_id', '')
                        if page_id:
                            metadata_map[f"pid_{page_id}"] = meta
                    except json.JSONDecodeError:
                        continue
        
        elif json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    metadata_list = json.load(f)
                    for meta in metadata_list:
                        img_path = meta.get('image', meta.get('image_path', ''))
                        if img_path:
                            img_name = Path(img_path).name
                            metadata_map[img_name] = meta
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Load landmark_info.json (thong tin chi tiet tu Wikipedia)
        landmark_info = {}
        info_file = landmark_meta_dir / "landmark_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                try:
                    landmark_info = json.load(f)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Tim tat ca anh trong thu muc images/Landmark_Name/
        if not img_dir.exists():
            continue
        
        img_files = sorted(
            list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        )
        
        for img_file in img_files:
            img_name = img_file.name
            
            # Tim metadata tuong ung
            img_metadata = metadata_map.get(img_name)
            
            # Thu tim theo page_id trong ten file (format: Landmark_pageID.jpg)
            if img_metadata is None:
                parts = img_name.replace('.jpg', '').replace('.png', '').rsplit('_', 1)
                if len(parts) == 2:
                    pid_key = f"pid_{parts[1]}"
                    img_metadata = metadata_map.get(pid_key)
            
            # Neu khong co metadata, tao metadata co ban tu landmark_info
            if img_metadata is None:
                img_metadata = {
                    'landmark': landmark_name.replace('_', ' '),
                    'name': landmark_name.replace('_', ' '),
                    'image_path': str(img_file),
                }
                # Gop thong tin tu landmark_info
                for key in ['location', 'country', 'year_built', 'architect', 
                            'style', 'height', 'unesco_status', 'visitors_per_year',
                            'significance', 'description', 'coordinates']:
                    if key in landmark_info and landmark_info[key]:
                        img_metadata[key] = landmark_info[key]
            
            all_images.append({
                'image_path': str(img_file),
                'landmark': landmark_name,
                'metadata': img_metadata
            })
    
    return all_images


def build_vector_index(
    images_dir: str = "data/images",
    metadata_dir: str = "data/metadata",
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
    all_images = collect_all_images(images_dir, metadata_dir)
    print(f"Found {len(all_images)} images")
    
    if not all_images:
        print("No images found! Check data/images/ and data/metadata/ directories.")
        return None
    
    # Group by landmark
    landmarks_count = {}
    for img in all_images:
        landmark = img['landmark']
        landmarks_count[landmark] = landmarks_count.get(landmark, 0) + 1
    
    print("\nImages per landmark:")
    for landmark, count in sorted(landmarks_count.items()):
        print(f"  {landmark}: {count} images")
    print(f"\nTotal landmarks: {len(landmarks_count)}")
    
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
                # Dam bao co truong 'name'
                if 'name' not in meta:
                    meta['name'] = item['landmark'].replace('_', ' ')
                meta['image_path'] = item['image_path']
                batch_metadata.append(meta)
                
            except Exception as e:
                print(f"\nFailed to load {item['image_path']}: {e}")
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
    
    print(f"\nIndex saved to: {output_dir}")
    print(f"   Total vectors: {retriever.index.ntotal}")
    print(f"   Total metadata: {len(retriever.metadata)}")
    print(f"   Total landmarks: {len(landmarks_count)}")
    
    # In thong ke metadata quality
    has_description = sum(1 for m in metadata_list if m.get('description'))
    has_year = sum(1 for m in metadata_list if m.get('year_built'))
    has_location = sum(1 for m in metadata_list if m.get('location'))
    print(f"\n   Metadata quality:")
    print(f"     With description: {has_description}/{len(metadata_list)}")
    print(f"     With year_built: {has_year}/{len(metadata_list)}")
    print(f"     With location: {has_location}/{len(metadata_list)}")
    
    return retriever


def test_retrieval(retriever, test_image_path: str, top_k: int = 5):
    """Test retrieval với một ảnh query"""
    print("\n" + "="*70)
    print("TEST RETRIEVAL")
    print("="*70)
    
    print(f"\nQuery image: {test_image_path}")
    
    embedder = VisualEmbedder(device="cuda")
    query_img = Image.open(test_image_path).convert('RGB')
    query_embedding = embedder.embed_image(query_img)
    
    # Search
    distances, results = retriever.search(query_embedding, top_k=top_k)
    
    print(f"\nTop-{top_k} results:")
    for i, (dist, meta) in enumerate(zip(distances, results), 1):
        landmark = meta.get('name', meta.get('landmark', 'Unknown'))
        location = meta.get('location', '')
        year = meta.get('year_built', '')
        img_path = Path(meta.get('image_path', '')).name
        
        info_str = f"{landmark}"
        if location:
            info_str += f" ({location})"
        if year:
            info_str += f" [{year}]"
        
        print(f"  {i}. {info_str} - {img_path} (Distance: {dist:.4f})")


if __name__ == "__main__":
    import sys
    
    # Build vector index
    retriever = build_vector_index(
        images_dir="data/images",
        metadata_dir="data/metadata",
        output_dir="data/vector_index",
        device="cuda",
        batch_size=16
    )
