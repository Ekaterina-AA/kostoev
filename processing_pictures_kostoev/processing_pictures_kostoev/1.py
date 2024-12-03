import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path)

print(f"size: {image.size}")  
print(f"format: {image.format}") 
print(f"mode: {image.mode}")  
plt.imshow(image)
plt.axis('off') 
plt.show()


image_array = np.asarray(image)

def find_regions(image_array, threshold):
    mask = np.abs(image_array - threshold) < 10  
    return mask

if image.mode != 'L':
    image_gray = image.convert('L')  
else:
    image_gray = image

image_gray_array = np.asarray(image_gray)

threshold_value = 169

regions_mask = find_regions(image_gray_array, threshold_value)

plt.imshow(regions_mask, cmap='gray')
plt.title('Regions above threshold')
plt.axis('off') 
plt.show()

