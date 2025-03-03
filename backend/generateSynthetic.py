import random
import cv2
import numpy as np
import os

# Function to adjust brightness and add noise
random.seed(42)

def process_image(image_path):
    # Read the image into a numpy array
    image = cv2.imread(image_path)

    # Reduce brightness and exposure
    alpha = random.uniform(0.4, 0.5)
    beta = random.randint(-20, -15)
    darker_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # alpha <1 darkens, beta <0 reduces brightness

    # # Add noise
    # noise = np.random.normal(0, 0.4, darker_image.shape).astype(np.uint8)  # Gaussian noise
    # noisy_image = cv2.add(darker_image, noise)

    # return noisy_image
    return darker_image

def main():
    # Define input and output folders
    input_folder = "C://Users//wenji//OneDrive//Desktop//Y3S2//ATAP//sample images//Huge Image Dataset//JPEGImages"
    output_folder = "C://Users//wenji//OneDrive//Desktop//Y3S2//ATAP//sample images//low_light_synthetic//JPEGImages_lowlight_synthetic"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Ensure the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_img = process_image(input_path)

            # Save the modified image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)

    print("Processing complete. Edited images saved in:", output_folder)

if __name__ == "__main__":
    main()