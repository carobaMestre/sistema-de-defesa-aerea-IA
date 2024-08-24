import os
import cv2
import numpy as np
from pathlib import Path
import concurrent.futures

def preprocess_image(orig_folder, dest_folder, file_name, resize_dim=(224,224), normalize=True):
    try:
        # Loading the image
        img = cv2.imread(str(orig_folder / file_name))
        
        # Checking if the image was read correctly
        if img is None:
            print(f"Couldn't read the image {file_name}")
            return
        
        # Converting the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Normalizing the image
        normalized = equalized / 255.0 if normalize else equalized

        # Resizing the image
        resized = cv2.resize(normalized, resize_dim)

        # Converting the image to the format expected by the CNN input
        input_img = np.expand_dims(resized, axis=0)
        input_img = input_img.reshape(input_img.shape[0], input_img.shape[1], input_img.shape[2], 1)

        # Saving the preprocessed image to the destination folder
        cv2.imwrite(str(dest_folder / file_name), resized * 255.0)
        
    except Exception as e:
        print(f"Error processing the image {file_name}: {e}")
        
if __name__ == '__main__':
    print("Starting the image preprocessing...")
    # Defining the source and destination folder paths using pathlib
    orig_folder = Path('C:/Users/vitor/Desktop/identificacao-drones/data/modelo/originais')
    dest_folder = Path('C:/Users/vitor/Desktop/identificacao-drones/data/modelo/preprocessado')

    # Creating the destination folder if it doesn't exist
    #dest_folder.mkdir(parents=True, exist_ok=True)

    # Defining the resizing and normalization options
    resize_dim = (300, 300)
    normalize = True

    # List of file names to be processed
    file_names = [f for f in os.listdir(orig_folder) if f.endswith(('.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'))]
    print("file_names: ", file_names)
    # Processing the images in parallel
    num_workers = os.cpu_count() # Number of available processors
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        results = [executor.submit(preprocess_image, orig_folder, dest_folder, file_name, resize_dim, normalize) for file_name in file_names]
    print("Finished the image preprocessing... Freaks!")
    # Checking for errors during processing
    for result in concurrent.futures.as_completed(results):
        if result.exception() is not None:
            print(result.exception())
