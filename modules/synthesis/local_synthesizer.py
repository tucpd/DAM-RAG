"""
Local LLM Synthesizer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import warnings


class LocalLLMSynthesizer:
    """
    Synthesizer dùng local LLM để tạo travel caption
    Model: Qwen2.5-7B-Instruct (hoặc Llama-3.1-8B-Instruct)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 200,
        temperature: float = 0.3,
        use_flash_attn: bool = False,
        use_compile: bool = True
    ):
        """
        Args:
            model_name: Tên model trên HuggingFace
            device: cuda hoặc cpu
            max_new_tokens: Số token tối đa sinh ra
            temperature: Temperature cho generation
            use_flash_attn: Sử dụng Flash Attention 2
            use_compile: Sử dụng torch.compile
        """
        print(f"Loading local LLM: {model_name}")
        print(f"Device: {device}")
        print(f"Optimizations: Flash-Attn={use_flash_attn}, Compile={use_compile}")
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_flash_attn = use_flash_attn
        self.use_compile = use_compile
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model với tối ưu
        load_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Thêm Flash Attention 2 nếu có
        if use_flash_attn and device == "cuda":
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("Enabling Flash Attention 2...")
            except Exception:
                warnings.warn("Flash Attention 2 not available, using default attention")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Apply torch.compile
        if use_compile and device == "cuda":
            print("Applying torch.compile (first run will be slow)...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
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
            system_prompt = (
                "You are a friendly, knowledgeable travel guide who writes clear, natural captions "
                "for travel photos. Your tone is conversational and helpful, like chatting with a traveler."
            )
    
            user_prompt = f"""Write a natural, engaging travel caption (80-200 words) based on this information:

IMAGE DESCRIPTION:
{dam_caption}

RELATED INFORMATION:
{knowledge_text}

Guidelines:
- Write in a natural, conversational tone — as if you're speaking directly to a traveler.
- Keep it concise, interesting, and easy to read.
- Avoid overly flowery, poetic, or exaggerated language.
- Focus on what is actually visible in the photo first, then naturally weave in relevant historical or cultural context.
- Sound genuine and helpful, not like a brochure.
- Write in English only.

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
        
        # Generate với tối ưu config
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,  # KV cache cho tốc độ
            }
            
            # Chế độ sampling tùy theo temperature
            if self.temperature > 0.01:
                generation_config.update({
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 50
                })
            else:
                # Greedy decoding nhanh hơn
                generation_config["do_sample"] = False
            
            outputs = self.model.generate(
                **inputs,
                **generation_config
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
