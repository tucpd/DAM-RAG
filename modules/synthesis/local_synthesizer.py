"""
Local LLM Synthesizer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional


class LocalLLMSynthesizer:
    """
    Synthesizer dùng local LLM để tạo travel caption
    Model: Qwen2.5-7B-Instruct (hoặc Llama-3.1-8B-Instruct)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 300,
        temperature: float = 0.7
    ):
        """
        Args:
            model_name: Tên model trên HuggingFace
            device: cuda hoặc cpu
            max_new_tokens: Số token tối đa sinh ra
            temperature: Temperature cho generation
        """
        print(f"Loading local LLM: {model_name}")
        print(f"Device: {device}")
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully")
    
    def _build_prompt(
        self,
        dam_caption: str,
        retrieved_knowledge: List[Dict],
        style: str = "informative"
    ) -> str:
        """Xây dựng prompt cho LLM"""
        
        # Build knowledge context
        knowledge_text = ""
        for i, item in enumerate(retrieved_knowledge, 1):
            name = item.get('name', item.get('landmark', 'Unknown'))
            location = item.get('location', '')
            
            if location:
                knowledge_text += f"\n{i}. {name} ({location})"
            else:
                knowledge_text += f"\n{i}. {name}"
            
            if 'description' in item:
                knowledge_text += f"\n   {item['description'][:200]}..."
            if 'year_built' in item:
                knowledge_text += f"\n   Built: {item['year_built']}"
            if 'style' in item:
                knowledge_text += f"\n   Style: {item['style']}"
            knowledge_text += "\n"
        
        if style == "informative":
            system_prompt = "You are a professional travel guide writer. Create engaging and informative captions for travel photos."
            
            user_prompt = f"""Based on the following information, write a travel caption (80-200 words):

IMAGE DESCRIPTION:
{dam_caption}

RELATED LANDMARKS:
{knowledge_text}

Write an informative, engaging caption suitable for travelers. Focus on what is visually present and provide historical/cultural context if the image matches a known landmark. Keep it concise and interesting. Write in English only.

TRAVEL CAPTION:"""
        
        else:  # casual or other
            system_prompt = "You are a friendly travel blogger."
            user_prompt = f"""Write a casual travel caption based on:

IMAGE: {dam_caption}

CONTEXT: {knowledge_text}

Write a short, friendly caption (80-150 words) like sharing with friends.

CAPTION:"""
        
        return system_prompt, user_prompt
    
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
            retrieved_knowledge: List knowledge từ RAG
            style: Phong cách caption
        
        Returns:
            Dict với keys: 'caption', 'dam_caption', 'retrieved_landmarks'
        """
        # Build prompt
        system_prompt, user_prompt = self._build_prompt(
            dam_caption, retrieved_knowledge, style
        )
        
        # Format cho Qwen chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Build result
        landmark_names = [item['name'] for item in retrieved_knowledge]
        
        return {
            'caption': generated_text,
            'dam_caption': dam_caption,
            'retrieved_landmarks': landmark_names,
            'style': style
        }
    
    def synthesize_simple(
        self,
        dam_caption: str,
        retrieved_knowledge: List[Dict]
    ) -> str:
        """Version đơn giản, chỉ trả về caption text"""
        result = self.synthesize(dam_caption, retrieved_knowledge)
        return result['caption']


if __name__ == "__main__":
    # Test
    synthesizer = LocalLLMSynthesizer()
    
    dam_caption = "A majestic ancient stone temple with multiple tiers and intricate carvings, featuring five distinctive towers rising against a blue sky."
    
    retrieved_knowledge = [
        {
            "name": "Angkor Wat",
            "location": "Siem Reap, Cambodia",
            "description": "The largest religious monument in the world, built in the 12th century.",
            "year_built": "1113-1150",
            "style": "Khmer architecture"
        }
    ]
    
    print("\n" + "="*70)
    print("Generating caption...")
    print("="*70)
    
    result = synthesizer.synthesize(dam_caption, retrieved_knowledge)
    
    print("\n" + "="*70)
    print("FINAL CAPTION:")
    print("="*70)
    print(result['caption'])
