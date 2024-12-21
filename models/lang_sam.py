import numpy as np
from PIL import Image

from models import GDINO
from models import SAM
from utils import draw_image

class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None):
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path)
        self.gdino = GDINO()
        self.gdino.build_model()

    def predict(
        self,
        image_pil: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for a given image and text prompt using GDINO and SAM models.

        Parameters:
            image_pil (Image.Image): Input image.
            text_prompt (str): Text prompt corresponding to the image.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            dict: Result containing masks and other outputs for the image.
            Output format:
            {
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }
        """

        gdino_results = self.gdino.predict([image_pil], [text_prompt], box_threshold, text_threshold)
        result = gdino_results[0]

        processed_result = {
            **result,
            "masks": [],
            "mask_scores": [],
        }

        if result["labels"]:
            processed_result["boxes"] = result["boxes"].cpu().numpy()
            processed_result["scores"] = result["scores"].cpu().numpy()
            sam_image = np.asarray(image_pil)
            sam_boxes = processed_result["boxes"]

            print(f"Predicting masks for the given image")
            masks, mask_scores, _ = self.sam.predict_batch([sam_image], xyxy=[sam_boxes])
            processed_result.update(
                {
                    "masks": masks[0],
                    "mask_scores": mask_scores[0],
                }
            )
            print(f"Predicted masks for the image")

        return processed_result

    def run_inference(self, image: Image.Image, text_prompt: str, box_threshold=0.3, text_threshold=0.25) -> Image.Image:
        # Model inference
        print("Running inference...")
        results = self.predict(
            image_pil=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # If no objects are detected, return the original image
        if results["masks"] is None or len(results["masks"]) == 0:
            print("No masks detected. Returning original image.")
            return image

        # Draw the results
        image_array = np.asarray(image)
        output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

        return output_image
    
if __name__ == "__main__":
    model = LangSAM()
    image = Image.open("./assets/food.jpg")
    prompt = "food"
    output = model.predict(image, prompt)
    print(output)