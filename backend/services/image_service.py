import cv2
import numpy as np
import math
from scipy import signal
import os


def enhance_image(img_np):
    """
    Enhance the input image using the LIME algorithm.
    """
    if img_np is None:
        raise ValueError("Invalid image provided")
    
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB) #only works with np array

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert back to BGR color space (numpy)
    enhanced_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_img

# Converts image into RGB channels
def img_to_rgb(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        raise ValueError("Invalid image provided")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # print("rgb image shape: ", rgb_img.shape)
    return rgb_img


def compute_rgbmax(img):
    """
    Compute the RGBmax matrix from the input image.
    The RGBmax matrix is calculated by keeping, for each pixel,
    only the maximum value of the RGB triplet.

    Args:
        img (str): Path to the input image or the image itself.
    
    Returns:
        numpy.ndarray: The RGBmax matrix (2D array) containing the maximum RGB value for each pixel.
    """
    
    rgb_img = img_to_rgb(img)
    # print("rgb image shape: ", rgb_img.shape)
    rgbmax_matrix = np.max(rgb_img, axis=2)  # Shape: (height, width)
    # print("rgb max shape: ", rgbmax_matrix.shape)
    return rgbmax_matrix


def gaussian_kernel(H=2, sigma=2):
    """Returns a 2D Gaussian kernel array."""
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(H-1)/2)**2+(y-(H-1)/2)**2))/(2*sigma**2)), (H, H))
    kernel /= np.sum(kernel) # normalise kernel
    return kernel

# Define the 2D FIR filter
def fir_filter(size, cutoff=0.99):
    # Create a 1D lowpass FIR filter
    fir_filter_1d = signal.firwin(size, cutoff)
    
    # Create a 2D filter by taking the outer product of the 1D filter with itself
    fir_filter_2d = np.outer(fir_filter_1d, fir_filter_1d)
    
    return fir_filter_2d

def create_weighting_kernel(image_shape: tuple, sigma: float = 2, H: int = 3): # TODO
    """Creates the weighting kernel for the image."""
    
    # Step 1: Create the FIR filter
    fir = fir_filter(H)
    
    # Step 2: Apply the Gaussian kernel
    gaussian = gaussian_kernel(H, sigma)
    
    # Apply the FIR filter followed by the Gaussian kernel
    kernel = np.multiply(fir, gaussian)
    
    # Step 3: Resize the kernel to match the image size
    kernel_resized = cv2.resize(kernel, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return kernel_resized

def hdr_brightness(img_gray):
    if img_gray is None:
        raise ValueError("Invalid image provided")
    image_shape = img_gray.shape
    weighting_kernel = create_weighting_kernel(image_shape)
    
    rgbmax_matrix = compute_rgbmax(img_gray)
    filtered_image = rgbmax_matrix * weighting_kernel
    n = filtered_image.size  # Total number of pixels
    rms_filtered_rgmmax = np.sqrt(np.sum(filtered_image**2) / n)

    return rms_filtered_rgmmax
