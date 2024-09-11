import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_noise(image, noise_sigma):
    noisy_image = image.astype(np.float32) + np.random.normal(0, noise_sigma, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def denoise_image(image, h, templateWindowSize, searchWindowSize):
    denoised_image = cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    return denoised_image

if __name__ == "__main__":

    image = cv2.imread("/Users/kanishknaithani/Desktop/medical mini project/Skeleton.jpg", cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError("Image not found. Please check the path and filename.")

    noise_sigma = 5 
    noisy_image = add_noise(image, noise_sigma)

    h_value = 12  
    templateWindowSize = 5
    searchWindowSize = 11
    denoised_image = denoise_image(noisy_image, h_value, templateWindowSize, searchWindowSize)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image using NLM')
    plt.imshow(denoised_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()