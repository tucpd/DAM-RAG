"""
Build RAG Pipeline
Script để xây dựng retrieval pipeline từ dataset có sẵn
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from modules.retrieval.embedder import VisualEmbedder
from modules.retrieval.retriever import VectorRetriever


def create_sample_dataset():
    """
    Tạo sample dataset về landmarks nổi tiếng
    Trong thực tế sẽ dùng dataset lớn hơn như Google Landmarks Dataset
    """
    dataset = [
        {
            "name": "Angkor Wat",
            "description": "Angkor Wat is a temple complex in Cambodia and the largest religious monument in the world. It was originally constructed as a Hindu temple dedicated to the god Vishnu for the Khmer Empire, gradually transforming into a Buddhist temple towards the end of the 12th century. Built in the early 12th century (1113-1150) by King Suryavarman II.",
            "location": "Siem Reap, Cambodia",
            "year_built": "1113-1150",
            "style": "Khmer architecture",
            "category": "temple"
        },
        {
            "name": "Eiffel Tower",
            "description": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair. It stands at 330 meters (1,083 ft) tall.",
            "location": "Paris, France",
            "year_built": "1887-1889",
            "style": "Structural expressionism",
            "category": "tower"
        },
        {
            "name": "Taj Mahal",
            "description": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, India. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house.",
            "location": "Agra, India",
            "year_built": "1631-1648",
            "style": "Mughal architecture",
            "category": "mausoleum"
        },
        {
            "name": "Colosseum",
            "description": "The Colosseum is an oval amphitheatre in the centre of Rome, Italy. Built of travertine limestone, tuff (volcanic rock), and brick-faced concrete, it was the largest amphitheatre ever built at the time and held 50,000 to 80,000 spectators. Construction began under the emperor Vespasian (r. 69-79 AD) and was completed in 80 AD under his successor Titus.",
            "location": "Rome, Italy",
            "year_built": "70-80 AD",
            "style": "Ancient Roman architecture",
            "category": "amphitheatre"
        },
        {
            "name": "Great Wall of China",
            "description": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China. The best-known sections were built during the Ming dynasty (1368-1644). The wall stretches over 21,000 km (13,000 miles) from east to west.",
            "location": "Northern China",
            "year_built": "7th century BC - 1644 AD",
            "style": "Chinese military architecture",
            "category": "fortification"
        },
        {
            "name": "Machu Picchu",
            "description": "Machu Picchu is a 15th-century Inca citadel located in the Eastern Cordillera of southern Peru. It is situated on a mountain ridge 2,430 meters (7,970 ft) above sea level. Most archaeologists believe that Machu Picchu was constructed as an estate for the Inca emperor Pachacuti (1438-1472).",
            "location": "Cusco Region, Peru",
            "year_built": "1450",
            "style": "Inca architecture",
            "category": "citadel"
        },
        {
            "name": "Statue of Liberty",
            "description": "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. The copper statue was a gift from the people of France to the people of the United States, designed by French sculptor Frédéric Auguste Bartholdi. It was dedicated on October 28, 1886. The statue is 46 meters (151 ft) tall.",
            "location": "New York, USA",
            "year_built": "1884-1886",
            "style": "Neoclassical",
            "category": "statue"
        },
        {
            "name": "Sydney Opera House",
            "description": "The Sydney Opera House is a multi-venue performing arts centre in Sydney. Designed by Danish architect Jørn Utzon, the building was formally opened on 20 October 1973. The building comprises multiple performance venues, which together host well over 1,500 performances annually. It is one of the most famous and distinctive buildings of the 20th century.",
            "location": "Sydney, Australia",
            "year_built": "1959-1973",
            "style": "Expressionist modernism",
            "category": "performing arts centre"
        },
    ]
    
    return dataset


def build_index_from_dataset(
    dataset: list,
    embedder: VisualEmbedder,
    retriever: VectorRetriever,
    images_dir: Path
):
    """
    Build vector index từ dataset
    
    Trong thực tế sẽ embed ảnh thật, nhưng demo này chỉ dùng text để generate embeddings
    """
    print(f"\nBuilding index từ {len(dataset)} landmarks...")
    
    # Vì không có ảnh thật, tôi sẽ dùng text embeddings từ CLIP text encoder
    # Trong thực tế sẽ dùng ảnh thật
    
    text_inputs = []
    for item in dataset:
        # Tạo text description cho embedding
        text = f"{item['name']}, {item['category']}. {item['description']}"
        text_inputs.append(text)
    
    # CLIP text embeddings (placeholder - trong thực tế dùng image embeddings)
    print("Đang tạo embeddings từ text descriptions...")
    vectors = []
    
    import torch
    with torch.no_grad():
        for text in tqdm(text_inputs):
            # Dùng CLIP text encoder
            inputs = embedder.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(embedder.device)
            
            text_features = embedder.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            vectors.append(text_features.cpu().numpy()[0])
    
    vectors = np.array(vectors)
    
    # Add to retriever
    retriever.add_vectors(vectors, dataset)
    
    print(f"✓ Đã build index với {len(dataset)} landmarks")
    
    return retriever


def main():
    print("="*80)
    print("BUILD RAG PIPELINE")
    print("="*80)
    
    # 1. Tạo sample dataset
    print("\n[1] Tạo sample landmark dataset...")
    dataset = create_sample_dataset()
    print(f"✓ Dataset: {len(dataset)} landmarks")
    
    # 2. Load embedder
    print("\n[2] Load Visual Embedder (CLIP)...")
    embedder = VisualEmbedder(
        model_name="openai/clip-vit-large-patch14",
        device="cuda"
    )
    
    # 3. Khởi tạo retriever
    print("\n[3] Khởi tạo Vector Retriever (FAISS)...")
    retriever = VectorRetriever(
        embed_dim=embedder.embed_dim,
        use_gpu=True
    )
    
    # 4. Build index
    print("\n[4] Build index...")
    images_dir = Path("data/knowledge_base/images")
    retriever = build_index_from_dataset(dataset, embedder, retriever, images_dir)
    
    # 5. Test retrieval
    print("\n[5] Test retrieval...")
    
    # Test query: "ancient temple"
    test_queries = [
        "ancient temple with towers",
        "modern building with unique design",
        "stone monument in Europe"
    ]
    
    for query_text in test_queries:
        print(f"\n Query: '{query_text}'")
        
        # Embed query
        inputs = embedder.processor(
            text=[query_text],
            return_tensors="pt",
            padding=True
        ).to(embedder.device)
        
        with torch.no_grad():
            query_embedding = embedder.model.get_text_features(**inputs)
            query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()[0]
        
        # Search
        distances, results = retriever.search(query_embedding, top_k=3)
        
        print("  Top-3 kết quả:")
        for i, (dist, meta) in enumerate(zip(distances, results)):
            print(f"    {i+1}. {meta['name']} ({meta['location']}) - Distance: {dist:.4f}")
    
    # 6. Save index
    print("\n[6] Lưu index...")
    save_dir = "data/vector_index"
    retriever.save(save_dir)
    
    # Lưu dataset
    dataset_file = Path(save_dir) / "landmarks_dataset.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Đã lưu index tại: {save_dir}")
    print("\n✓ RAG Pipeline build hoàn tất!")


if __name__ == "__main__":
    main()
