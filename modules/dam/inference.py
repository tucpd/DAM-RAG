"""
DAM Inference Module
Module chính để load và chạy inference DAM-3B model
"""

import torch
from PIL import Image
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path

try:
    from dam import DescribeAnythingModel, disable_torch_init
except ImportError:
    raise ImportError("DAM package chưa được cài đặt. Chạy: pip install git+https://github.com/NVlabs/describe-anything")


class DAMInference:
    """
    Class để load và inference DAM-3B model
    
    DAM (Describe Anything Model) sinh caption chi tiết cho vùng được chọn trong ảnh.
    Hỗ trợ input dạng: mask (chính xác nhất)
    
    Output: Caption chi tiết bằng tiếng Anh
    """
    
    def __init__(
        self,
        model_path: str = "nvidia/DAM-3B",
        conv_mode: str = "v1",
        prompt_mode: str = "full+focal_crop",
        device: Optional[str] = None,
    ):
        """
        Khởi tạo DAM model
        
        Args:
            model_path: Path model trên Hugging Face (nvidia/DAM-3B hoặc nvidia/DAM-3B-Video)
            conv_mode: Conversation mode (mặc định: v1)
            prompt_mode: Prompt mode (mặc định: full+focal_crop)
            device: Device để chạy model (None = auto detect)
        """
        self.model_path = model_path
        self.conv_mode = conv_mode
        self.prompt_mode = prompt_mode
        
        # Auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Đang load DAM model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Conv mode: {conv_mode}, Prompt mode: {prompt_mode}")
        
        # Disable torch init để load nhanh hơn
        disable_torch_init()
        
        # Load DAM model sử dụng DescribeAnythingModel
        self.dam = DescribeAnythingModel(
            model_path=model_path,
            conv_mode=conv_mode,
            prompt_mode=prompt_mode,
        ).to(self.device)
        
        print("✓ DAM model đã được load thành công!")
        
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Optional[Union[np.ndarray, Image.Image]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        query: str = "<image>\nDescribe the masked region in detail.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        streaming: bool = False,
    ) -> Union[str, list]:
        """
        Sinh caption chi tiết cho vùng được chọn
        
        Args:
            image: Ảnh input (có thể là path, PIL Image, hoặc numpy array)
            mask: Binary mask cho vùng cần mô tả (H, W) hoặc (H, W, 1)
            box: Bounding box (x1, y1, x2, y2) - sẽ được convert thành mask
            query: Query text (mặc định: mô tả vùng mask chi tiết)
            max_new_tokens: Số token tối đa cho caption
            temperature: Temperature cho sampling (0 = greedy)
            top_p: Top-p sampling value
            num_beams: Số beams cho beam search
            streaming: True để trả về generator (từng token), False để trả về full text
            
        Returns:
            Caption chi tiết bằng tiếng Anh (str hoặc generator nếu streaming=True)
            
        Note:
            - Nếu không có mask hoặc box, sẽ mô tả toàn bộ ảnh
            - Mask là binary image (0-255)
        """
        
        # Load và chuẩn bị ảnh
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Prepare mask
        if mask is not None:
            if isinstance(mask, np.ndarray):
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            else:
                mask_pil = mask
        
        elif box is not None:
            # Convert box to mask
            x1, y1, x2, y2 = box
            w, h = image_pil.size
            mask_np = np.zeros((h, w), dtype=np.uint8)
            mask_np[y1:y2, x1:x2] = 255
            mask_pil = Image.fromarray(mask_np)
        
        else:
            # No mask - create full image mask (DAM requires mask)
            w, h = image_pil.size
            mask_pil = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255)
        
        # Generate description using DAM
        output = self.dam.get_description(
            image_pil=image_pil,
            mask_pil=mask_pil,
            query=query,
            streaming=streaming,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        
        if streaming:
            return output  # Generator
        else:
            return output.strip()  # String
    
    def batch_generate(
        self,
        images: list,
        masks: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Sinh caption cho nhiều ảnh cùng lúc
        
        Args:
            images: List các ảnh
            masks: List các mask tương ứng (None để mô tả toàn ảnh)
            **kwargs: Các tham số khác cho generate_caption
            
        Returns:
            List caption
        """
        if masks is None:
            masks = [None] * len(images)
        
        results = []
        for img, mask in zip(images, masks):
            caption = self.generate_caption(img, mask, **kwargs)
            results.append(caption)
        
        return results
