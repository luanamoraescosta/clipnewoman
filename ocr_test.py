import pytesseract
import cv2
import os
import pandas as pd

# Configuring the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image
def extract_text_from_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Convert the image to grayscale (optional, improves OCR accuracy)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract text from the image
    text = pytesseract.image_to_string(gray_image)
    
    return text

# Function to process images inside a folder and its subfolders
def process_images_in_subfolders(root_folder, output_file):
    data = []  # List to store extracted data

    # Walk through the main folder and all its subfolders
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            image_path = os.path.join(foldername, filename)
            
            # Check if the file is an image
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                print(f"Processing: {image_path}")
                text = extract_text_from_image(image_path)
                
                # Only add the result if the text was successfully extracted
                if text is not None:
                    # Add data to the list
                    data.append([image_path, text])

    # Create a pandas DataFrame and save it to a CSV file
    df = pd.DataFrame(data, columns=['image_path', 'extracted_text'])
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"CSV file '{output_file}' created successfully.")

root_folder = "D:\\The Delineator Images"  
output_file = 'output_all_images.csv'  
process_images_in_subfolders(root_folder, output_file)