import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class Llava2:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=0):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        """Loads the Llava model and processor."""
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor

    def chat(self, image, inp, is_question = None, max_new_tokens=300):
        """Generates a detailed description of an image."""
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""Analyze the image and provide a description relevant to the following prompt: 
                        '{inp}'
                    """},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device, torch.float16)

        output = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=0.1,
            top_k=30,
            top_p=0.9
        )
        output = self.processor.decode(
            output[0][2:], skip_special_tokens=True
        )
        output = str(output).split('assistant')[1]
        if output.startswith('\n'):
            output = output[1:]
        return output

# Usage
if __name__ == "__main__":
    image = Image.open('trump.png').convert('RGB')
    llava = Llava2()
    prompt = "trump"
    print(llava.chat(image, prompt, is_question=False))
    prompt = "what's the thing may bark"
    print(llava.chat(image, prompt, is_question=True))
