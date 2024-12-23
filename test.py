from PIL import Image
from models import Llava
from models import Llama
from models import QuestionDetector
from models import LangSAM

def main(inp, image_path):
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
    langsam = LangSAM(sam_type="sam2.1_hiera_small")
    output = langsam.run_inference(image, 'small lens')

    output.save(f'output_images/{image_name}')

# Example usage
if __name__ == "__main__":
    while(True):
        inp = input('Prompt: ')
        image_path = input('Image path: ')
        main(inp, image_path)