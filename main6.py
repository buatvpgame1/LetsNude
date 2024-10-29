import requests
import torch
import cv2
import time
import base64
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

app = Flask(__name__)

# API URLs and credentials
API_URL = "https://api.neuramare.com/v1/image-to-image-generation-job"
IMGBB_API_URL = "https://api.imgbb.com/1/upload"
IMGBB_API_KEY = "8e21d6385abe388116eb21b8d5a5e649"  # Replace with your ImgBB API key
NEURAMARE_ACCESS_TOKEN = "ecdeebe2-8a3d-4d6d-9c69-5ed715e0d874"  # Neuramare API access token

# Load Mask R-CNN model pre-trained on COCO dataset
device = torch.device("cpu")  # Ensure CPU-only processing
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to segment clothing using Mask R-CNN, excluding hands, glasses, and watches
def detect_clothing(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Convert image to RGB format and create a tensor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

    # Get predictions from Mask R-CNN
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Initialize an empty mask for the clothing area
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process only the "person" label (label ID 1 in COCO dataset) and clothing (if applicable)
    for i, label in enumerate(predictions["labels"]):
        # Exclude labels for hands (40), glasses (39), and watches (41)
        if label.item() in {1, 42, 43, 44}:  # Person and clothing
            if predictions["scores"][i] > 0.5:  # Filter by confidence
                # Retrieve the mask for the detected object
                object_mask = predictions["masks"][i, 0].cpu().numpy()
                object_mask = (object_mask > 0.5).astype(np.uint8) * 255
                object_mask = cv2.resize(object_mask, (width, height))

                # Add the object mask to the overall mask
                mask = cv2.bitwise_or(mask, object_mask)

    return image, mask, width, height

# Updated function to unmask only the head and neck area based on the clothing mask
def unmask_face(image, clothing_mask, width_factor=0.5, upper_height_factor=0.5, lower_height_factor=1.2):
    # Detect faces in the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)  # Adjust size to detect smaller faces if needed
    )

    # Initialize mask for unmasking head
    head_mask = np.zeros(clothing_mask.shape, dtype=np.uint8)

    for (x, y, w, h) in faces:
        # Calculate bounding box for head with adjustable factors
        x1 = max(0, x - int(w * width_factor))                # Expand or shrink width on left
        x2 = min(clothing_mask.shape[1], x + w + int(w * width_factor))  # Expand or shrink width on right
        y1 = max(0, y - int(h * upper_height_factor))         # Move top boundary up or down
        y2 = y + int(h * lower_height_factor)                 # Adjust bottom boundary for neck

        # Draw a rectangle on the head mask to cover just the head and neck area
        cv2.rectangle(head_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # Smooth the mask edges for better blending
    head_mask = cv2.GaussianBlur(head_mask, (5, 5), 0)

    # Apply the mask to unmask the head and neck region in the clothing mask
    final_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(head_mask))

    return final_mask

def remove_bare_hands(image, clothing_mask):
    # Convert image to HSV color space for skin detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for skin tone
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin areas
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_mask = cv2.medianBlur(skin_mask, 5)

    # Use bitwise operations to remove bare hands (skin) from the clothing mask
    no_hands_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(skin_mask))

    return no_hands_mask

def apply_mask(image, mask):
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.addWeighted(image, 0.2, mask_colored, 1, 0)
    return masked_image

# Function to upload an image to ImgBB and get the URL
def upload_image_to_imgbb(file_path):
    with open(file_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "key": IMGBB_API_KEY,
        "image": image_base64,
        "expiration": 600
    }

    response = requests.post(IMGBB_API_URL, data=payload)
    if response.status_code == 200:
        data = response.json().get("data", {})
        return data.get("url")
    else:
        print(f"Error uploading image: {response.status_code} - {response.text}")
        return None

# Function to send API request to Neuramare using image URLs
def send_image_edit_request(image_url, mask_url, prompt, width, height):
    payload = {
        "input": {
            "prompt": prompt,
            "init_img_url": image_url,
            "mask_url": mask_url,
            "width": width,
            "height": height,
            "negative_prompt": "",
            "num_inference_steps": 80,
            "guidance_scale": 13.5,
            "strength": 1,
            "scheduler": "DPM++ 2M Karras",
        }
    }

    headers = {
        "Authorization": f"{NEURAMARE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 202:
        result = response.json().get("result", {}).get("job_id")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Poll the API for the job result
def poll_job_status(job_id):
    poll_url = f"{API_URL}/{job_id}"
    headers = {
        "Authorization": f"{NEURAMARE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(30):
        response = requests.get(poll_url, headers=headers)
        if response.status_code == 200:
            result = response.json().get("result")
            status = result.get("status")

            if status == "COMPLETED":
                return result
            elif status == "FAILED":
                print("Job failed")
                return None

        time.sleep(4)

    print("Job did not complete within 30 attempts")
    return None


@app.route("/process_image", methods=["POST"])
def process_image():
    data = request.json
    image_base64 = data.get("image_base64")
    user_id = data.get("user_id")
    prompt = "nude, very wide waist, very super huge big round tits, normal nipples, normal pose, realistic, white natural skin, flat belly, smooth skin, no muscle definition, realistic, realistic, realistic"

    # Decode base64 image
    if image_base64:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        image.save("downloaded_image0.jpg")  # Save for further processing

    image, clothing_mask, width, height = detect_clothing("downloaded_image0.jpg")
    final_mask = unmask_face(image, clothing_mask, width_factor=0.5, upper_height_factor=0.5, lower_height_factor=1.2)
    final_mask = remove_bare_hands(image, final_mask)
    masked_image_path = "final_masked_image0.png"
    cv2.imwrite(masked_image_path, final_mask)

    image_url = upload_image_to_imgbb("downloaded_image0.jpg")
    mask_url = upload_image_to_imgbb(masked_image_path)

    if image_url and mask_url:
        job_id = send_image_edit_request(image_url, mask_url, prompt, width, height)

        if job_id:
            result = poll_job_status(job_id)

            if result:
                output_image_url = result.get('output_original_url')
                if output_image_url:
                    response = requests.get(output_image_url)
                    if response.status_code == 200:
                        with open("output_image.png", "wb") as f:
                            f.write(response.content)

                        with open("output_image.png", "rb") as img_file:
                            return jsonify({"user_id": user_id, "output_image_url": output_image_url})

    return jsonify({"error": "Failed to process the image."}), 500


# Run the Flask application
if __name__ == "__main__":
    app.run(port=5000)  # You can specify the port as needed
