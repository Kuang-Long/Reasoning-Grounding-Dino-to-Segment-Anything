import torch
import numpy as np
import supervision as sv
from PIL import Image

MIN_AREA = 100

def get_device_type() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
def draw_image(image_rgb, masks, xyxy, probs, labels):
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    return annotated_image

def draw_masks(masks, image_path):
    # Ensure masks are binary
    masks = (masks > 0).astype(np.uint8)

    # Combine all masks into a single composite mask
    composite_mask = np.max(masks, axis=0)  # Combine masks (element-wise max ensures binary output)

    # Scale the mask to 0-255 for black-and-white image
    bw_image_array = (composite_mask * 255).astype(np.uint8)

    # Convert to a PIL image for saving/displaying
    bw_image = Image.fromarray(bw_image_array, mode="L")  # 'L' mode for grayscale

    bw_image.save(image_path)
    return bw_image