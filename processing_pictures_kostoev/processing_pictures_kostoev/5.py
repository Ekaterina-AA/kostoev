import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path).convert('L')

def normalize_histogram(image, mean, std_dev):
    """Reducing the histogram to a normal distribution."""
    image_array = np.array(image).astype(np.float32)
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)
    normalized_image = mean + std_dev * image_array
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized_image)

def exponential_distribution(image):
    """Converting a histogram to an exponential distribution."""
    image_array = np.array(image).astype(np.float32)
    exp_image = 255 * (1 - np.exp(-image_array / 255))
    exp_image = np.clip(exp_image, 0, 255).astype(np.uint8)
    return Image.fromarray(exp_image)

def cauchy_distribution(image):
    """Reducing the histogram to the Cauchy distribution."""
    image_array = np.array(image).astype(np.float32)
    cauchy_image = (np.arctan(image_array / 255) + np.pi / 2) / np.pi * 255
    cauchy_image = np.clip(cauchy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(cauchy_image)

mean = 128
std_dev = 50

normalized_image = normalize_histogram(image, mean, std_dev)
exp_image = exponential_distribution(image)
cauchy_image = cauchy_distribution(image)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(normalized_image, cmap='gray')
axs[1].set_title('Normal dist')
axs[2].imshow(exp_image, cmap='gray')
axs[2].set_title('Exponential dist')
axs[3].imshow(cauchy_image, cmap='gray')
axs[3].set_title('Cauchy distribution')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
