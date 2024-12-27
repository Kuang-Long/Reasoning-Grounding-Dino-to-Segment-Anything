import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class Llava:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=0):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor


        """
        使用 LLaVA 產生圖片描述或回答：
        - 如果 is_question = None: 回傳完整詳細描述
        - 如果 is_question = True: 簡短回答問題，聚焦視覺特徵
        - 如果 is_question = False: 描述該物件的視覺特徵，不包含品牌與專有名詞
        """

    def chat(self, image, prompt, is_question=None, max_new_tokens=500):
        
        conversation_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": """Describe this image in extreme detail. 
                    Your description should include:
                    1. All visible objects (animals, items, and background elements).
                    2. Their relative positions (e.g., \"on the left\", \"next to\", \"behind\").
                    3. Any notable colors, patterns, or physical features."""},
                ],
            },
        ]

        conversation_object = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                        (f"Describe the {prompt} based only on its visual characteristics, "
                         "such as color, shape, size, and unique features. Avoid using brand names, models, or any specific nouns.")}
                ],
            },
        ]

        conversation_question = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                        (f"Answer this as short as possible: Focus only on visual details of the {prompt}, "
                         "such as its color, material, or design patterns. Do not include brand or specific names.")}
                ],
            },
        ]

        if is_question is None:
            prompt_template = self.processor.apply_chat_template(conversation_full, add_generation_prompt=True)
        elif is_question:
            prompt_template = self.processor.apply_chat_template(conversation_question, add_generation_prompt=True)
        else:
            prompt_template = self.processor.apply_chat_template(conversation_object, add_generation_prompt=True)

        inputs = self.processor(
            images=image, text=prompt_template, return_tensors="pt"
        ).to(self.device, torch.float16)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
        )
        output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        output = str(output).split('assistant', 1)[-1].strip()
        return output