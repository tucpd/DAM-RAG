"""
Visual Embedder
Module để embed ảnh vùng thành vector cho retrieval
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from pathlib import Path

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Cần cài transformers: pip install transformers")


class VisualEmbedder:
    """
    Visual Embedder sử dụng CLIP để embed ảnh thành vector
    
    CLIP là model multimodal mạnh mẽ, đã được train trên 400M image-text pairs
    Rất phù hợp cho việc embed ảnh landmarks/architecture
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: CLIP model name từ Hugging Face
            device: Device để chạy (None = auto detect)
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Đang load CLIP model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load CLIP
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embed_dim = self.model.config.projection_dim
        
        print(f"✓ CLIP model loaded, embed_dim: {self.embed_dim}")
    
    @torch.no_grad()
    def embed_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed một ảnh thành vector
        
        Args:
            image: Ảnh input
            normalize: True để normalize vector về unit length
            
        Returns:
            Embedding vector (numpy array)
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process qua CLIP processor
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Get image features
        image_features = self.model.get_image_features(**inputs)
        
        # Normalize if needed
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy()[0]
        
        return embedding
    
    @torch.no_grad()
    def embed_images_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed nhiều ảnh cùng lúc (batched)
        
        Args:
            images: List các ảnh
            batch_size: Batch size cho inference
            normalize: True để normalize vectors
            
        Returns:
            Array of embeddings (num_images, embed_dim)
        """
        all_embeddings = []
        
        # Process theo batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Load và convert images
            pil_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil_images.append(Image.open(img).convert("RGB"))
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)
            
            # Process batch
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)
            
            # Get features
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize if needed
            if normalize:
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Append to results
            all_embeddings.append(image_features.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                print(f"Đã embed {min(i+batch_size, len(images))}/{len(images)} ảnh")
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        return embeddings
    
    def embed_region(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        box: Optional[tuple] = None,
        mask: Optional[Union[np.ndarray, Image.Image]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed một vùng cụ thể trong ảnh
        
        Args:
            image: Ảnh input
            box: Bounding box (x1, y1, x2, y2) để crop
            mask: Binary mask - sẽ crop theo bounding box của mask
            normalize: True để normalize vector
            
        Returns:
            Embedding vector
        """
        # Load ảnh
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Crop region
        if mask is not None:
            # Tìm bounding box của mask
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            
            # Find bounding box
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                box = (x1, y1, x2, y2)
        
        # Crop image nếu có box
        if box is not None:
            x1, y1, x2, y2 = box
            image = image.crop((x1, y1, x2, y2))
        
        # Embed cropped region
        return self.embed_image(image, normalize=normalize)


def main():
    """
    Test embedder
    """
    print("="*80)
    print("VISUAL EMBEDDER TEST")
    print("="*80)
    
    embedder = VisualEmbedder()
    
    # Test với ảnh sample
    test_image = "data/images/test_simple.jpg"
    
    if Path(test_image).exists():
        print(f"\nĐang embed ảnh: {test_image}")
        embedding = embedder.embed_image(test_image)
        
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"✓ Sample values: {embedding[:5]}")
    else:
        print(f"Không tìm thấy ảnh test: {test_image}")


if __name__ == "__main__":
    main()
