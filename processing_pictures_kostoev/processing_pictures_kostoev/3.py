import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path)

def adjust_brightness(image, factor):
    return Image.eval(image, lambda x: min(255, int(x * factor)))

brightness_factor = 2
brightened_image = adjust_brightness(image, brightness_factor)

def threshold_transformation(image, threshold):
    image_array = np.array(image)
    image_array[image_array > threshold] = 255
    image_array[image_array <= threshold] = 0
    return Image.fromarray(image_array)

threshold = 122
threshold_image = threshold_transformation(image, threshold)

def logarithmic_transformation(image):
    c = 255 / np.log(1 + np.max(np.array(image)))
    log_transformed = c * np.log(1 + np.array(image))
    log_transformed = np.array(np.clip(log_transformed, 0, 255), dtype=np.uint8)
    return Image.fromarray(log_transformed)

log_image = logarithmic_transformation(image)

def power_law_transformation(image, gamma):
    c = 255 / np.max(np.array(image)) ** gamma
    power_law_transformed = c * np.array(image) ** gamma
    power_law_transformed = np.array(np.clip(power_law_transformed, 0, 255), dtype=np.uint8)
    return Image.fromarray(power_law_transformed)

gamma_value = 2.0
power_law_image = power_law_transformation(image, gamma_value)

fig, axs = plt.subplots(2, 3, figsize=(15, 5))
axs[0,0].imshow(brightened_image)
axs[0,0].set_title('adjusted_brightness')
axs[0,1].imshow(log_image)
axs[0,1].set_title('logarithmic_transformation')
axs[0,2].imshow(power_law_image)
axs[0,2].set_title('power_law_transformation')
axs[1,0].imshow(threshold_image)
axs[1,0].set_title('threshold_transformation')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

