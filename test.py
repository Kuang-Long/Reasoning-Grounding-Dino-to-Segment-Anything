import re
import json
import torch
from PIL import Image
from utils import draw_masks
from models import Llava, Llama, BoundingBoxSAM

def main(inp, image_path, image_url, token):
    image = Image.open(image_path).convert('RGB')

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
    output_path=f'test_result/seg/{image_name}'
    output_mask_path=f'test_result/mask/{image_name}'
    output = bbox_sam.process_image(image_url, image_path, final_object)
    if output is not None:
        output, masks = output
    else:
        output = Image.open(image_path).convert("RGB")
        masks = None
    output.save(output_path)
    draw_masks(masks, output_mask_path, image_path)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_first_text_element(file_path):
    """
    Reads a JSON file and returns the first element of the 'text' field as a string.
    
    Args:
        file_path (str): The path to the JSON file.

    Returns:
        str: The first element of the 'text' field if it exists, otherwise None.
    """
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        # Check if 'text' field exists and is a list
        if 'text' in json_data and isinstance(json_data['text'], list) and len(json_data['text']) > 0:
            return json_data['text'][0]
        else:
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None

# Example usage
if __name__ == "__main__":
    llava = Llava()
    llama = Llama()
    
    token = '6e59b41ca95415ac095a39569557a72e'
    bbox_sam = BoundingBoxSAM(token=token)

    with open('file_names_only_jpg.txt') as file:
        with open('url.txt') as ufile:
            i = 1
            for line, url in zip(file, ufile):
                filename = line.strip()
                image_path = f'test/{filename}'
                image_url = url
                filename = filename.split('.')[0] + '.json'
                inp = get_first_text_element(f'test/{filename}')
                print('input =', inp)
                print('url =', image_url)
                main(inp, image_path, image_url, token)
                i = i+1
    # while(True):
    #     inp = input('Prompt: ')
    #     image_path = input('Image path: ')
    #     image_url = input('url: ')
    #     if image_url.endswith('0'):
    #         image_url = image_url[:len(image_url)-1] + '1'
    #     main(inp, image_path, image_url)