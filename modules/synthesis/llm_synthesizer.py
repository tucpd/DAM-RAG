"""
LLM Synthesizer
Sử dụng LLM để tổng hợp DAM caption + RAG knowledge thành travel caption
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables từ file .env
load_dotenv()


class LLMSynthesizer:
    """
    Synthesizer dùng Gemini API để tạo travel caption
    Kết hợp:
    - DAM caption: Mô tả chi tiết hình ảnh
    - RAG knowledge: Thông tin lịch sử, kiến trúc, văn hóa từ knowledge base
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Args:
            api_key: Gemini API key (nếu None sẽ đọc từ env GEMINI_API_KEY)
            model_name: Tên model (models/gemini-2.5-flash, models/gemini-2.5-pro, etc.)
            temperature: Temperature cho generation (0-1)
            max_tokens: Số token tối đa cho output
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _build_prompt(
        self,
        dam_caption: str,
        retrieved_knowledge: List[Dict],
        style: str = "informative"
    ) -> str:
        """
        Xây dựng prompt cho LLM
        
        Args:
            dam_caption: Caption từ DAM (mô tả hình ảnh chi tiết)
            retrieved_knowledge: List các knowledge snippets từ RAG
            style: Phong cách caption (informative, casual, poetic)
        """
        # Build knowledge context
        knowledge_text = ""
        for i, item in enumerate(retrieved_knowledge, 1):
            knowledge_text += f"\n{i}. {item['name']} ({item['location']})"
            if 'description' in item:
                knowledge_text += f"\n   {item['description'][:200]}..."
            if 'year_built' in item:
                knowledge_text += f"\n   Built: {item['year_built']}"
            if 'style' in item:
                knowledge_text += f"\n   Style: {item['style']}"
            knowledge_text += "\n"
        
        # Build prompt based on style
        if style == "informative":
            prompt = f"""You are a professional travel guide writer. Create an engaging and informative caption for a travel photo.

IMAGE DESCRIPTION (from visual analysis):
{dam_caption}

RELATED LANDMARKS INFORMATION:
{knowledge_text}

TASK:
Write a travel caption (80-200 words) that:
1. Describes what is visually present in the image (based on the description above)
2. If the image matches one of the landmarks, provide historical/cultural context
3. Use an informative, engaging tone suitable for travelers
4. Keep it concise and interesting
5. Write in English

Focus on accuracy and helpfulness for travelers. Do not make up information not provided above.

TRAVEL CAPTION:"""
        
        elif style == "casual":
            prompt = f"""You are a friendly travel blogger. Create a casual, engaging caption for this photo.

WHAT I SEE:
{dam_caption}

BACKGROUND INFO:
{knowledge_text}

Write a short, friendly caption (80-150 words) like you're sharing with friends. Keep it casual but informative.

CAPTION:"""
        
        else:  # poetic
            prompt = f"""You are a poetic travel writer. Create an evocative caption for this image.

IMAGE:
{dam_caption}

CONTEXT:
{knowledge_text}

Write a beautiful, evocative caption (80-200 words) that captures the essence and wonder of this place.

CAPTION:"""
        
        return prompt
    
    def synthesize(
        self,
        dam_caption: str,
        retrieved_knowledge: List[Dict],
        style: str = "informative"
    ) -> Dict[str, str]:
        """
        Tổng hợp caption cuối cùng
        
        Args:
            dam_caption: Caption từ DAM
            retrieved_knowledge: List knowledge từ RAG (top-K results)
            style: Phong cách caption
        
        Returns:
            Dict với keys: 'caption', 'dam_caption', 'retrieved_landmarks'
        """
        # Build prompt
        prompt = self._build_prompt(dam_caption, retrieved_knowledge, style)
        
        # Generate
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        
        final_caption = response.text.strip()
        
        # Build result
        landmark_names = [item['name'] for item in retrieved_knowledge]
        
        return {
            'caption': final_caption,
            'dam_caption': dam_caption,
            'retrieved_landmarks': landmark_names,
            'style': style
        }
    
    def synthesize_simple(
        self,
        dam_caption: str,
        retrieved_knowledge: List[Dict]
    ) -> str:
        """
        Version đơn giản, chỉ trả về caption text
        """
        result = self.synthesize(dam_caption, retrieved_knowledge)
        return result['caption']


if __name__ == "__main__":
    # Test với sample data
    synthesizer = LLMSynthesizer()
    
    dam_caption = "A majestic ancient stone temple with multiple tiers and intricate carvings, featuring five distinctive towers rising against a blue sky, surrounded by tropical vegetation."
    
    retrieved_knowledge = [
        {
            "name": "Angkor Wat",
            "location": "Siem Reap, Cambodia",
            "description": "Angkor Wat is a temple complex and the largest religious monument in the world, originally constructed as a Hindu temple dedicated to Vishnu in the early 12th century.",
            "year_built": "1113-1150",
            "style": "Khmer architecture"
        },
        {
            "name": "Borobudur",
            "location": "Java, Indonesia",
            "description": "9th-century Mahayana Buddhist temple with stone platforms and circular terraces.",
            "year_built": "750-850",
            "style": "Javanese Buddhist architecture"
        }
    ]
    
    result = synthesizer.synthesize(dam_caption, retrieved_knowledge)
    
    print("="*60)
    print("FINAL TRAVEL CAPTION:")
    print("="*60)
    print(result['caption'])
    print("\n" + "="*60)
    print(f"Based on: {', '.join(result['retrieved_landmarks'])}")
