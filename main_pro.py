import re 
import torch
from PIL import Image
from utils import draw_masks
from models import Llava, Llama, BoundingBoxSAM

def main(inp, image_path, image_url):
    image = Image.open(image_path).convert('RGB')

    # Step 1: Use LLaVA to describe the image
    description = llava.chat(image, inp, is_question=None)
    print('Description:', description)

    # Step 2: Use LLaMA to extract a list of objects from the description
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

    # Use regex to parse the object list
    pattern = r"^\d+\.\s*(.*)$"
    candidates = re.findall(pattern, extracted.strip(), flags=re.IGNORECASE | re.MULTILINE)
    candidates = [c.strip() for c in candidates if c.strip()]
    print("Candidates:", candidates)

    # Retry if candidates list is empty
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

    # Step 3: Use the user's prompt to select the best matching candidate object
    refine_system_prompt = """You have a list of candidate objects from an image and a user request (prompt).
        Your job is to pick the single best matching object name from the candidate list, based strictly on the prompt.
        Only output the object name itself or \"none\" if no suitable match.
        Do not add any explanations, just output the single best matching object or \"none\".
        """
    refine_user_prompt = f"""
        Candidate objects: {', '.join(candidates)}
        User prompt: \"{inp}\"

        Which candidate best matches the user's request?
        If no candidate matches, output \"none\".
    """
    refine_prompt = refine_system_prompt + "\n\n" + refine_user_prompt
    final_ans = llama.chat(refine_prompt)
    print("Refined final answer:", final_ans)

    final_object = final_ans.strip()
    if final_object.lower() not in [c.lower() for c in candidates] and final_object.lower() != "none":
        lowered_candidates = [c.lower() for c in candidates]
        if final_object.lower() not in lowered_candidates:
            approximate = [c for c in candidates if final_object.lower() in c.lower()]
            if approximate:
                final_object = approximate[0]
            else:
                final_object = "none"

    print(f"Final object for BoundingBoxSAM: {final_object}")

    # Use BoundingBoxSAM for segmentation
    image_name = str(image_path).split('/')
    image_name = image_name[len(image_name) - 1]
    output_path=f'output_images/{image_name}'
    output_mask_path=f'output_images/masks/{image_name}'
    output, masks = bbox_sam.process_image(image_url, image_path, final_object)
    output.save(output_path)
    draw_masks(masks, output_mask_path)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Example usage
if __name__ == "__main__":
    llava = Llava()
    llama = Llama()
    
    token = "your token"  # replace with token
    bbox_sam = BoundingBoxSAM(token=token)
    while(True):
        inp = input('Prompt: ')
        image_path = input('Image path: ')
        image_url = input('url: ')
        if image_url.endswith('0'):
            image_url = image_url[:len(image_url)-1] + '1'
        main(inp, image_path, image_url)
