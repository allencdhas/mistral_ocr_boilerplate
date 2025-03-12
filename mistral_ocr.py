import cv2
import pytesseract
import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Set your Mistral API Key
API_KEY = "YOUR_MISTRAL_API_KEY"

# Initialize Mistral client
client = MistralClient(api_key=API_KEY)
model = "mistral-large-latest"  # Change model as required

# Configure Tesseract path (adjust this if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows users

def extract_text_from_image(image_path):
    """Extracts text from an image using pytesseract."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def process_text_with_mistral(text):
    """Sends the extracted text to Mistral AI for processing."""
    messages = [ChatMessage(role="user", content=text)]
    response = client.chat(model=model, messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    image_path = "sample_image.png"  # Replace with your image path
    extracted_text = extract_text_from_image(image_path)
    
    if extracted_text:
        print("Extracted Text:", extracted_text)
        processed_text = process_text_with_mistral(extracted_text)
        print("\nProcessed by Mistral:", processed_text)
    else:
        print("No text detected in the image.")
