from PIL import Image
from models import Llava
from models import Llama
from models import QuestionDetector
from models import BoundingBoxSAM

def main(inp, image_path, image_url):
    # load image
    image = Image.open(image_path).convert('RGB')
    
    image_name = str(image_path).split('/')
    image_name = image_name[len(image_name) - 1]

    # initialize models
    llava = Llava()
    llama = Llama()
    qd = QuestionDetector()

    # Generate image description with Llava
    description = llava.chat(image, inp, None, max_new_tokens=300)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Output description:\n', description)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    
    is_question = qd.is_question(inp)
    ans = llava.chat(image, inp=inp, is_question=is_question)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    if is_question:
        print(ans[0])
        print(ans[1])
    else:
        print(ans)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    
    if not is_question:
        question = f"{ans}"
        template = "what are the main objects explicitly described in this: {question} and related to " + f"{inp}"
        ans = llama.chat(question, template)
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('llama:\n' + ans)
        print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
    # Image segmentation with langsam
    bbox_sam = BoundingBoxSAM(token=token)
    if is_question:
        bbox_sam.process_image(image_url, image_path, ans[0], output_path=f'output_images/{image_name}')
    else:
        bbox_sam.process_image(image_url, image_path, ans, output_path=f'output_images/{image_name}')

# Example usage
if __name__ == "__main__":
    # use ur token here
    token = "your token" 
    while(True):
        inp = input('Prompt: ')
        image_path = input('Image path: ')
        image_url = input('url: ')
        if image_url.endswith('0'):
            image_url = image_url[:len(image_url)-1] + '1'
        main(inp, image_path, image_url)