import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path)

def invert_colors(image):
    return Image.eval(image, lambda x: 255 - x)

def to_grayscale(image):
    return image.convert("L")

def rotate_image(image):
    return image.rotate(90, expand=True)

def crop_image(image, box):
    return image.crop(box)

def zero_pixels_below_threshold(image, threshold):
    data = np.array(image)
    data[data < threshold] = 0
    return Image.fromarray(data)

image_inverted = invert_colors(image)
image_gray = to_grayscale(image)
image_rotated = rotate_image(image)
image_cropped = crop_image(image, (50, 50, 200, 200))  
image_zeroed = zero_pixels_below_threshold(image, 100)  

image_inverted.save('inverted_image.jpg')
image_gray.save('grayscale_image.jpg')
image_rotated.save('rotated_image.jpg')
image_cropped.save('cropped_image.jpg')
image_zeroed.save('zeroed_image.jpg')

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs[0, 0].imshow(image_inverted)
axs[0, 0].set_title('inverted_image')
axs[0, 1].imshow(image_gray, cmap='gray')
axs[0, 1].set_title('grayscale_image')
axs[0, 2].imshow(image_rotated)
axs[0, 2].set_title('rotated_image')
axs[1, 0].imshow(image_cropped)
axs[1, 0].set_title('cropped_image')
axs[1, 1].imshow(image_zeroed)
axs[1, 1].set_title('zeroed_image')
axs[1, 2].imshow(image)
axs[1, 2].set_title('original_image')


for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

