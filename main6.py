import requests
import torch
import cv2
import json
import time
import base64
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import torchvision
from torchvision.transforms import functional as F

# API URLs and credentials
API_URL = "https://api.neuramare.com/v1/image-to-image-generation-job"
IMGBB_API_URL = "https://api.imgbb.com/1/upload"
IMGBB_API_KEY = "8e21d6385abe388116eb21b8d5a5e649"  # imgbb API key
NEURAMARE_ACCESS_TOKEN = "ecdeebe2-8a3d-4d6d-9c69-5ed715e0d874"  # Neuramare API access token

# Load Mask R-CNN model pre-trained on COCO dataset
device = torch.device("cpu")  # Ensure CPU-only processing
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Load OpenCV face detection model for unmasking the head area
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_image_counts = {}

# Function to detect skin regions using HSV color segmentation
def detect_skin(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin areas
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin_mask = cv2.medianBlur(skin_mask, 5)  # Smooth edges

    return skin_mask

# Function to segment clothing using Mask R-CNN, focusing on clothing areas within the "person" mask
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

    # Process only the "person" label (label ID 1 in COCO dataset)
    for i, label in enumerate(predictions["labels"]):
        if label.item() == 1 and predictions["scores"][i] > 0.5:  # Filter by person label and confidence
            # Retrieve the mask for the person
            person_mask = predictions["masks"][i, 0].cpu().numpy()
            person_mask = (person_mask > 0.5).astype(np.uint8) * 255
            person_mask = cv2.resize(person_mask, (width, height))

            # Detect skin regions and exclude them from the person mask
            skin_mask = detect_skin(image)
            clothing_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(skin_mask))

            # Define the approximate clothing area as the lower portion of the detected person mask
            mask = cv2.bitwise_or(mask, clothing_mask)

    return image, mask, width, height

# Function to unmask the face area
def unmask_face(image, mask):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)  # Adjust size to detect smaller faces
    )

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x - int(w * 0.1)), max(0, y - int(h * 0.2))
        x2, y2 = min(mask.shape[1], x + w + int(w * 0.1)), min(mask.shape[0], y + h + int(h * 0.2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

def apply_mask(image, mask):
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.addWeighted(image, 0.2, mask_colored, 1, 0)
    return masked_image

# Function to upload an image to imgbb and get the URL
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

    for attempt in range(15):
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

    print("Job did not complete within 15 attempts")
    return None

# Updated handle_image function
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id not in user_image_counts:
        user_image_counts[user_id] = 0

    if user_image_counts[user_id] >= 5:
        await update.message.reply_text("You have used all your 5 image generations. Please come back later.")
        return

    user_image_counts[user_id] += 1
    await update.message.reply_text("Processing your image, please wait...")

    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive('1.jpg')

    asyncio.create_task(process_image(update, context, '1.jpg'))

async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE, image_path: str) -> None:
    prompt = "nude, very super huge big round tits, normal nipples, realistic, white natural skin"

    image, clothing_mask, width, height = detect_clothing(image_path)
    final_mask = unmask_face(image, clothing_mask)
    masked_image = apply_mask(image, final_mask)

    masked_image_path = "final_masked_image.png"
    cv2.imwrite(masked_image_path, masked_image)

    image_url = upload_image_to_imgbb(image_path)
    mask_url = upload_image_to_imgbb(masked_image_path)

    if image_url and mask_url:
        job_id = send_image_edit_request(image_url, mask_url, prompt, width, height)

        if job_id:
            result = poll_job_status(job_id)

            if result:
                output_image_url = result['output_original_url']
                response = requests.get(output_image_url)
                if response.status_code == 200:
                    with open("output_image.png", "wb") as f:
                        f.write(response.content)

                    with open("output_image.png", "rb") as img_file:
                        await update.message.reply_photo(img_file)

                    return

    await update.message.reply_text("Failed to process the image.")

# Function to handle the /start command, sending a welcome message
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_message = (
        f"Hello {user.mention_html()}! 👋\n\n"
        "Welcome to the Image Editing Bot! 🖼️ You can upload an image, and I'll process it based on your requests.\n"
        "Each user has up to 5 image editing requests.\n\n"
        "Please send an image to start!"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")

# Initialize and run the Telegram bot application
def main():
    application = ApplicationBuilder().token("7525407455:AAGrGbEI7m_mK03Rwh_WllLEBeaxOuIvaPc").build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start the bot
    application.run_polling()

# Run the bot
if __name__ == "__main__":
    main()
