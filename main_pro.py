from PIL import Image
from models import Llava, Llama, QuestionDetector, BoundingBoxSAM
import re

def main(inp, image_path, image_url):
    image = Image.open(image_path).convert('RGB')
    llava = Llava()
    llama = Llama()
    qd = QuestionDetector()

    # 步驟1：使用LLaVA描述圖片
    description = llava.chat(image, inp, is_question=None)
    print('Description:', description)

    # 步驟2：使用LLaMA從描述中萃取物件清單
    extract_system_prompt = """You are a helpful assistant that reads a description of an image and extracts all distinct mentioned objects.
        Only output a numbered list of objects, one per line, no extra text, no explanations.
        If no objects are mentioned, write nothing.
        """
    extract_user_prompt = f"""
        Description:
        \"\"\"{description}\"\"\"


        List all distinct mentioned objects as instructed:
        Format:
        1. object1
        2. object2
    ...
    """
    extract_prompt = extract_system_prompt + "\n\n" + extract_user_prompt
    extracted = llama.chat(extract_prompt)
    print('Extracted objects (raw):', extracted)

    # 使用正則解析物件列表
    pattern = r"^\d+\.\s*(.*)$"
    candidates = re.findall(pattern, extracted.strip(), flags=re.IGNORECASE | re.MULTILINE)
    candidates = [c.strip() for c in candidates if c.strip()]
    print("Candidates:", candidates)

    # 若 candidates 為空嘗試一次重試
    if not candidates:
        extract_user_prompt_retry = f"""
            Below is the image description again. Just list the objects, nothing else.

            Description:
            \"\"\"{description}\"\"\"


            Format:
            1. object1
            2. object2
            ...
        """
        extract_prompt_retry = extract_system_prompt + "\n\n" + extract_user_prompt_retry
        extracted_retry = llama.chat(extract_prompt_retry)
        print('Extracted objects (retry raw):', extracted_retry)
        candidates = re.findall(pattern, extracted_retry.strip(), flags=re.IGNORECASE | re.MULTILINE)
        candidates = [c.strip() for c in candidates if c.strip()]
        print("Candidates after retry:", candidates)

    if not candidates:
        candidates = []

    # 步驟3：依據使用者的prompt選擇最符合的候選物件
    # 此階段僅告訴LLaMA要選出最符合描述的物件，無特定範例（泛用）
    refine_system_prompt = """You have a list of candidate objects from an image and a user request (prompt).
        Your job is to pick the single best matching object name from the candidate list, based strictly on the prompt.
        Only output the object name itself or "none" if no suitable match.
        Do not add any explanations, just output the single best matching object or "none".
        """
    refine_user_prompt = f"""
        Candidate objects: {', '.join(candidates)}
        User prompt: "{inp}"

        Which candidate best matches the user's request?
        If no candidate matches, output "none".
    """
    refine_prompt = refine_system_prompt + "\n\n" + refine_user_prompt
    final_ans = llama.chat(refine_prompt)
    print("Refined final answer:", final_ans)

    final_object = final_ans.strip()
    # 確保 final_object 在 candidates 中或為 "none"
    if final_object.lower() not in [c.lower() for c in candidates] and final_object.lower() != "none":
        # 若不在名單中也不是none，嘗試簡單近似匹配
        lowered_candidates = [c.lower() for c in candidates]
        if final_object.lower() not in lowered_candidates:
            approximate = [c for c in candidates if final_object.lower() in c.lower()]
            if approximate:
                final_object = approximate[0]
            else:
                final_object = "none"

    print(f"Final object for BoundingBoxSAM: {final_object}")

    # 步驟4：使用 BoundingBoxSAM 執行分割
    
    image_name = str(image_path).split('/')
    image_name = image_name[len(image_name) - 1]
    token = "e68a3624c8ab6b42c4de6f9b5cfb8b67"  # 替換為您的 token
    bbox_sam = BoundingBoxSAM(token=token)
    bbox_sam.process_image(image_url, image_path, final_object, output_path=f'output_images/{image_name}')

# Example usage
if __name__ == "__main__":
    url = 'https://www.dropbox.com/scl/fi/51chdqy8d2ngsd0j17ebc/car.jpg?rlkey=fe1tfaa559amebgyg5n6pw661&st=g7hugr0g&dl=1'
    image_path = '/home/kl/llm/Reasoning-Grounding-Dino-to-Segment-Anything/input_images/car.jpg'
    token = "e68a3624c8ab6b42c4de6f9b5cfb8b67"  # 替換為您的 token
    bbox_sam = BoundingBoxSAM(token=token)

    bbox_sam.process_image(url, image_path, 'wheels', output_path=f'output_images/111.png')
    # while(True):
    #     inp = input('Prompt: ')
    #     image_path = input('Image path: ')
    #     image_url = input('url: ')
    #     if image_url.endswith('0'):
    #         image_url = image_url[:len(image_url)-1] + '1'
    #     main(inp, image_path, image_url)