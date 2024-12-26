import time
import requests
import numpy as np
from PIL import Image
from models import SAM
from utils import draw_image

class BoundingBoxSAM:
    def __init__(self, token: str, sam_type="sam2.1_hiera_small"):
        self.headers = {
            "Content-Type": "application/json",
            "Token": token
        }
        self.sam_type = sam_type
        self.sam_model = SAM()
        self.sam_model.build_model(sam_type=self.sam_type)

    def send_detection_request(self, image_url: str, prompt: str, model="GroundingDino-1.6-Pro"):
        body = {
            "image": image_url,
            "prompts": [
                {"type": "text", "text": prompt},
            ],
            "model": model,
            "targets": ["bbox"]
        }

        resp = requests.post('https://api.deepdataspace.com/tasks/detection', json=body, headers=self.headers)
        if resp.status_code == 200:
            json_resp = resp.json()
            task_uuid = json_resp["data"]["task_uuid"]
            print(f"Task UUID: {task_uuid}")
            return task_uuid
        else:
            raise Exception(f"Error sending request: {resp.status_code} - {resp.text}")

    def poll_detection_result(self, task_uuid: str, max_retries=60):
        retry_count = 0
        while retry_count < max_retries:
            resp = requests.get(f'https://api.deepdataspace.com/task_statuses/{task_uuid}', headers=self.headers)
            if resp.status_code != 200:
                raise Exception(f"Error fetching task status: {resp.status_code}")

            json_resp = resp.json()
            status = json_resp["data"]["status"]

            if status not in ["waiting", "running"]:
                if status == "failed":
                    raise Exception(f"Task failed: {json_resp}")
                elif status == "success":
                    print("Detection successful.")
                    return json_resp["data"]["result"]

            time.sleep(1)
            retry_count += 1

        raise Exception("Max retries reached without a successful result.")

    def run_sam_inference(self, image_path: str, bbox: list[list[float]]):
        # Load the image
        image_pil = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image_pil)

        # Prepare results
        results = []
        
        # Predict masks for each bounding box
        for box in bbox:
            box_np = np.array(box, dtype=np.float32)
            print(box_np)
            masks, scores, logits = self.sam_model.predict(image_array, box_np)
            results.append({
                "box": box,
                "masks": masks,
                "scores": scores,
                "logits": logits
            })

        if not any(result["masks"].size > 0 for result in results):
            print("No masks detected.")
            return image_pil

        # Draw results
        all_masks = np.array(np.concatenate([result["masks"] for result in results], axis=0))
        all_boxes = np.array([result["box"] for result in results])
        all_scores = np.array([result["scores"] for result in results])
        all_scores = np.array([p.item() for p in all_scores], dtype=np.float32)


        output_image = draw_image(
            image_array,
            all_masks,
            all_boxes,
            all_scores,
            np.array([""] * len(all_boxes))
        )

        return Image.fromarray(np.uint8(output_image)).convert("RGB")

    def process_image(self, image_url: str, local_image_path: str, prompt: str, output_path: str):
        try:
            # Step 1: Send detection request
            task_uuid = self.send_detection_request(image_url, prompt)

            task_uuid= task_uuid
            # Step 2: Poll for detection result
            detection_result = self.poll_detection_result(task_uuid)

            # Step 3: Extract bounding box
            objects = detection_result.get("objects", [])
            if not objects:
                print("No objects detected.")
                return None

            # bbox = objects[0]["bbox"]
            bbox = [obj["bbox"] for obj in objects]
            print(f"Bounding Box: {bbox}")

            # Step 4: Run SAM inference
            output_image = self.run_sam_inference(local_image_path, bbox)

            # Step 5: Show or save the output
            output_image.save(output_path)
            return output_image

        except Exception as e:
            print(f"Error: {e}")
            return None