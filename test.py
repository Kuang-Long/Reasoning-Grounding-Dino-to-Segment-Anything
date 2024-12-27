import re
import json
import dropbox
from PIL import Image
from utils import draw_masks
from models import Llava, Llama, QuestionDetector, BoundingBoxSAM

def main(inp, image_path, image_url, token):
    image = Image.open(image_path).convert('RGB')
    llava = Llava()
    llama = Llama()

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
    bbox_sam = BoundingBoxSAM(token=token)
    output = bbox_sam.process_image(image_url, image_path, final_object)
    if output is not None:
        output, masks = output
    else:
        output = Image.open(image_path).convert("RGB")
        masks = None
    output.save(output_path)
    draw_masks(masks, output_mask_path, image_path)

def get_shared_link(file_path):
    try:
        # Check if a shared link already exists
        result = dbx.sharing_list_shared_links(path=file_path)
        if result.links:
            # If a link exists, retrieve and modify it
            existing_link = result.links[0].url
            modified_link = existing_link.replace('dl=0', 'dl=1')
            return modified_link
        else:
            # If no link exists, create a new one
            link_metadata = dbx.sharing_create_shared_link_with_settings(file_path)
            new_link = str(link_metadata.url).replace('dl=0', 'dl=1')
            return new_link
    except dropbox.exceptions.ApiError as e:
        print("Error handling shared link:", e)
        return None

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
    # Replace this with your access token
    ACCESS_TOKEN = 'sl.u.AFYhqLg4aAwLrKSn_XC5FdRA6JUS09MzVydSGgyOyx3LwlIZALQpP5D_pRzUI5nEgdbQNJIj1RzT7TKH9K-wRqzCS_i_aFBNj2YBpatGl5znkMK5RjppLKcWExQ6aH-Or8RAxD3eYevffiWD__V-GhchT_x03AWVbJ8Agt86a2-DHLhRn3xS6cFtS3l5h5PHzgkCWgPSPKIDpUZtaq4q5csC_GceJGJ7rSYTeoBCEFb33uO3sTSTlJD8TBhCf5prNEqesVxZRHLNXuQu74PvcOIjT-whMbQ2GorLFom2DDSi4y40GSuwLcwBsm4aAoiWnBcOptQ50iJx3OhWOgH-GO05BQoykjO5nV83ph8TSijGfw-AWPbDx5bKBvm_ww1OT1BiC8YijHIwYnLukSndh-wxad9E5Xw1-ScWwpvJ6qbzgJyvyVgSRtNU9HKjURog--8Tbkq4ZuQMajOO3qvdQcTr8VTEb5Yvvdw-IqsSa6HdhixbI4oFzSpuEzD4ySIk-jzU6W0SKZUvR_7otnQdUDhF0A_t46MCQgCE3_PITu2UXQnnwsfbPE05HAsR5XVM8Z8sBj8Z0z-TQ0XiEY4w6X2qJ92XVOawLivmtA4x6mrJuD0r0ub0JPAxsvU9cnAGv2FpvXalBYs1T1Cho0Koi4K2UV8E30qwaV1UMEMYQNKJ84xCyemKPKXOjaNSiMxWobSJB6U-T1FJmhoRT7c8mRg5MCL_AIVHrcvSgb6jgTPmNOpoGDGQByCb3JaRbc8EzQkcS2flApA2qb-OrwAFmi8Ve-E1F4-t56dz56ihs7xF3T5-tM-ItPl-55-Seyail-nPu1Zv4IWLWKlsXRxv5cOWaH6XQ353NyT2VaVAMYCT2-SlF8RvmII8ihiy-MgB5Ovfrw5dji8hJG4hF849qxDn9bDciDeVWtAtY0f1CEaT-f54HU2GtsqE7656c1RPoFU6Fuad18pe-RBqq9IUmjRupR7FCo_OOm5WVflAus3yQAIg-GAw-aRCnUii8xCOgnaG3sXN7iEiL86jOl-sdz61iJAi3_n0Ev3Nm6p4wYf4GqA4774c50-F7b-7ieI66DiJtxxwqB5y12wrH5ge60KllUguUTBDFsXR3aXWD5BhivgfKcuvX6lDZrZRb2yxTLV_wHsa9v_k6bgOePnyKtGL7czhoEI_J51dUlG_lEwBrGfDnksZmzUHKBvgpUeDnulRxNvZxHxOQBMXZphm4VhFbxYtY-OgaIWCOalNVPPc_y9nzaxuzj1TjpxazomeRuVYuE-xN4ymdWFIXX21asd2'

    # Initialize Dropbox client
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    token = '3ef4060368982c4cb1fb2487d584199c'
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
                if i == 130:
                    token = '8e7bd19bd9e73006c3f1e8be715c07da'
                elif i == 330:
                    token = '7dd0b01b5dafa9bc6de0431326a99c8d'
                main(inp, image_path, image_url, token)
                i = i+1
    # while(True):
    #     inp = input('Prompt: ')
    #     image_path = input('Image path: ')
    #     image_url = input('url: ')
    #     if image_url.endswith('0'):
    #         image_url = image_url[:len(image_url)-1] + '1'
    #     main(inp, image_path, image_url)