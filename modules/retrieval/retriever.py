"""
Vector Retriever
Module xử lý retrieval sử dụng FAISS vector database
"""

import faiss
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class VectorRetriever:
    """
    Vector Retriever sử dụng FAISS để tìm kiếm nearest neighbors
    
    FAISS (Facebook AI Similarity Search) là thư viện cực nhanh cho similarity search
    Hỗ trợ billions vectors, GPU acceleration
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        use_gpu: bool = True
    ):
        """
        Args:
            embed_dim: Dimension của embedding vectors
            use_gpu: True để sử dụng GPU
        """
        self.embed_dim = embed_dim
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Khởi tạo FAISS index (Flat L2 - exact search)
        self.index = faiss.IndexFlatL2(embed_dim)
        
        # Move to GPU nếu có
        if self.use_gpu:
            print("Sử dụng GPU cho FAISS")
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,  # GPU 0
                self.index
            )
        
        # Metadata storage
        self.metadata = []
        
        print(f"FAISS index initialized (dim={embed_dim}, GPU={self.use_gpu})")
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Thêm vectors và metadata vào index
        
        Args:
            vectors: Array of vectors (num_vectors, embed_dim)
            metadata: List of metadata dicts (phải cùng length với vectors)
        """
        assert len(vectors) == len(metadata), "Số vectors và metadata phải bằng nhau"
        assert vectors.shape[1] == self.embed_dim, f"Vector dim phải là {self.embed_dim}"
        
        # Ensure float32
        vectors = vectors.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        print(f"Đã thêm {len(vectors)} vectors vào index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[float], List[Dict]]:
        """
        Tìm kiếm top-K nearest neighbors
        
        Args:
            query_vector: Query embedding vector (embed_dim,)
            top_k: Số kết quả trả về
            
        Returns:
            (distances, metadata_list)
        """
        # Ensure shape và type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get metadata
        results_metadata = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results_metadata.append(self.metadata[idx])
        
        return distances[0].tolist(), results_metadata
    
    def save(self, save_dir: str):
        """
        Lưu index và metadata
        
        Args:
            save_dir: Thư mục để lưu
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu FAISS index (chuyển về CPU trước)
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        faiss.write_index(cpu_index, str(save_path / "faiss.index"))
        
        # Lưu metadata
        with open(save_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu index tại: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, use_gpu: bool = True) -> 'VectorRetriever':
        """
        Load index và metadata từ disk
        
        Args:
            load_dir: Thư mục chứa index
            use_gpu: True để load lên GPU
            
        Returns:
            VectorRetriever đã được load
        """
        load_path = Path(load_dir)
        
        # Load FAISS index
        cpu_index = faiss.read_index(str(load_path / "faiss.index"))
        embed_dim = cpu_index.d
        
        # Create retriever
        retriever = cls(embed_dim=embed_dim, use_gpu=use_gpu)
        
        # Replace index
        if use_gpu and faiss.get_num_gpus() > 0:
            retriever.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                cpu_index
            )
        else:
            retriever.index = cpu_index
        
        # Load metadata
        with open(load_path / "metadata.json", 'r', encoding='utf-8') as f:
            retriever.metadata = json.load(f)
        
        print(f"Đã load index: {retriever.index.ntotal} vectors")
        return retriever


def main():
    """
    Test retriever
    """
    print("="*80)
    print("VECTOR RETRIEVER TEST")
    print("="*80)
    
    # Create dummy data
    embed_dim = 768
    num_samples = 100
    
    print(f"\nTạo {num_samples} dummy vectors...")
    vectors = np.random.randn(num_samples, embed_dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize
    
    metadata = [
        {"id": i, "name": f"landmark_{i}", "description": f"Description of landmark {i}"}
        for i in range(num_samples)
    ]
    
    # Create retriever
    print("\nKhởi tạo retriever...")
    retriever = VectorRetriever(embed_dim=embed_dim, use_gpu=True)
    
    # Add vectors
    print("\nThêm vectors vào index...")
    retriever.add_vectors(vectors, metadata)
    
    # Test search
    print("\nTest search...")
    query = vectors[0]  # Use first vector as query
    
    distances, results = retriever.search(query, top_k=5)
    
    print("\nTop-5 kết quả:")
    for i, (dist, meta) in enumerate(zip(distances, results)):
        print(f"{i+1}. Distance: {dist:.4f}, Name: {meta['name']}")
    
    # Test save/load
    print("\nTest save/load...")
    save_dir = "data/test_index"
    retriever.save(save_dir)
    
    retriever_loaded = VectorRetriever.load(save_dir, use_gpu=True)
    
    # Test search again
    distances2, results2 = retriever_loaded.search(query, top_k=5)
    print(f"\nLoad thành công, kết quả giống nhau: {np.allclose(distances, distances2)}")


if __name__ == "__main__":
    main()
