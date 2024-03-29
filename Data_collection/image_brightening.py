import cv2
import os

folders = ['/train/Cleaned/Happy_Cleaned', 
           '/train/Cleaned/surprised_Cleaned', 
           '/train/Cleaned/Neutral_Cleaned', 
           '/train/Cleaned/Focused_Cleaned'
           ]

def clahe_processing(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    image_clahe = clahe.apply(img)
    return image_clahe

def process_images(directory):
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        
        # Check if the file is an image
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
           
            clahe_image = clahe_processing(image)
            cv2.imwrite(os.path.join(directory,'brightened', 'brightened_' + filename), clahe_image)
            print(f"Brightened image saved to {os.path.join(directory, 'brightened','brightened_'+ filename)}")
                
            
def iterate_directories():
    for folder in folders:
        process_images(os.getcwd() + folder)         
        
iterate_directories()