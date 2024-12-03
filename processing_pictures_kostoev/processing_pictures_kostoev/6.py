import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'picture2.jpg'
image = Image.open(image_path)

from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

def laplacian_sharpening(image):
    """Sharpness improvement based on Laplacian."""
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    sharpened_image = convolve(image, laplacian_kernel)
    return np.clip(sharpened_image + image, 0, 255)

def sobel_edge_detection(image):
    """Application of the Sobel operator."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    
    sobel_image = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.clip(sobel_image, 0, 255)

def averaging_filter(image):
    """Smoothing using a 5x5 averaging filter."""
    kernel = np.ones((5, 5)) / 25
    smoothed_image = convolve(image, kernel)
    return smoothed_image

def arithmetic_operations(original_image, processed_image):
    """Arithmetic Image Transformations."""
    return np.clip(original_image + processed_image - 128, 0, 255)

image_gray = Image.open(image_path).convert('L')
image_array = np.array(image_gray)

sharpened_image = laplacian_sharpening(image_array)
sobel_image = sobel_edge_detection(image_array)
smoothed_image = averaging_filter(image_array)
arithmetic_result = arithmetic_operations(image_array, smoothed_image)

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs[0, 0].imshow(image_gray, cmap='gray')
axs[0, 0].set_title('Original image')
axs[0, 1].imshow(sharpened_image.astype(np.uint8), cmap='gray')
axs[0, 1].set_title('Laplacian')
axs[0, 2].imshow(sobel_image.astype(np.uint8), cmap='gray')
axs[0, 2].set_title('Sobel')
axs[1, 0].imshow(smoothed_image.astype(np.uint8), cmap='gray')
axs[1, 0].set_title('averaging filter')
axs[1, 1].imshow(arithmetic_result.astype(np.uint8), cmap='gray')
axs[1, 1].set_title('Arithmetic Image Transformations.')
axs[1, 2].axis('off')  

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

rng = np.random.default_rng()
image_gray = image.convert('L')

astro = color.rgb2gray(image)
astro = color.rgb2gray(image_gray)

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.0

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()