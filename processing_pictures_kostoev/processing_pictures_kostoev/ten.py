import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path)
image_array = np.array(image)


def add_gaussian_noise(image, mean=0, var=0.1):
    sigma = var ** 0.5
    
    noisy_image = image + np.random.normal(mean, sigma, image.shape)
    return np.clip(noisy_image, 0, 255)

def add_salt_and_pepper_noise(image, salt_prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    num_pepper = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def median_filter(image):
    from scipy.ndimage import median_filter
    return median_filter(image, size=3)

gaussian_noisy_image = add_gaussian_noise(image_array)
sp_noisy_image = add_salt_and_pepper_noise(image_array)

gaussian_recovered = median_filter(gaussian_noisy_image)
sp_recovered = median_filter(sp_noisy_image)

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs[0, 0].imshow(image_array, cmap='gray')
axs[0, 0].set_title('Original image')
axs[0, 1].imshow(gaussian_noisy_image.astype(np.uint8), cmap='gray')
axs[0, 1].set_title('Gaussian noise')
axs[0, 2].imshow(gaussian_recovered.astype(np.uint8), cmap='gray')
axs[0, 2].set_title('Gaussian recovery')

axs[1, 0].imshow(sp_noisy_image.astype(np.uint8), cmap='gray')
axs[1, 0].set_title('Salt and pepper')
axs[1, 1].imshow(sp_recovered.astype(np.uint8), cmap='gray')
axs[1, 1].set_title('Salt and pepper recovery')
axs[1, 2].axis('off')  

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
