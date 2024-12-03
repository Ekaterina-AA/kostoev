import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift


cutoff = 122
order = 2

def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask[np.sqrt((x - ccol)**2 + (y - crow)**2) <= cutoff] = 1
    return mask

def butterworth_low_pass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    d = np.sqrt((x - crow)**2 + (y - ccol)**2)
    return 1 / (1 + (d / cutoff)**(2 * order))

def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    d = np.sqrt((x - crow)**2 + (y - ccol)**2)
    return np.exp(-(d**2) / (2 * (cutoff**2)))

def apply_filter(image, filter_func, *args):
    image_fft = fft2(image)
    filter_mask = filter_func(image.shape, *args)
    filtered_fft = image_fft * filter_mask
    filtered_image = np.abs(ifft2(filtered_fft))
    return filtered_image

image_path = 'picture2.jpg'  
image = Image.open(image_path).convert('L')
image_array = np.array(image)

ideal_filtered = apply_filter(image_array, ideal_low_pass_filter, cutoff)
butter_filtered = apply_filter(image_array, butterworth_low_pass_filter, cutoff, order)
gaussian_filtered = apply_filter(image_array, gaussian_low_pass_filter, cutoff)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(image_array, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(ideal_filtered, cmap='gray')
axs[1].set_title('ideal_low_pass_filter')
axs[2].imshow(butter_filtered, cmap='gray')
axs[2].set_title('butterworth_low_pass_filter')
axs[3].imshow(gaussian_filtered, cmap='gray')
axs[3].set_title('gaussian_low_pass_filter')

for ax in axs.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()






def ideal_high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask[np.sqrt((x - ccol)**2 + (y - crow)**2) <= cutoff] = 0
    return mask

def butterworth_high_pass_filter(shape, cutoff, order):
    return 1 - butterworth_low_pass_filter(shape, cutoff, order)

def gaussian_high_pass_filter(shape, cutoff):
    return 1 - gaussian_low_pass_filter(shape, cutoff)

ideal_hp_filtered = apply_filter(image_array, ideal_high_pass_filter, cutoff)
butter_hp_filtered = apply_filter(image_array, butterworth_high_pass_filter, cutoff, order)
gaussian_hp_filtered = apply_filter(image_array, gaussian_high_pass_filter, cutoff)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(image_array, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(ideal_hp_filtered, cmap='gray')
axs[1].set_title('ideal_high_pass_filter')
axs[2].imshow(butter_hp_filtered, cmap='gray')
axs[2].set_title('butterworth_high_pass_filter')
axs[3].imshow(gaussian_hp_filtered, cmap='gray')
axs[3].set_title('gaussian_high_pass_filter')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

