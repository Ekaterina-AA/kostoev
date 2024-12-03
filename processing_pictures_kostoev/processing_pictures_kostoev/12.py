import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.restoration import richardson_lucy
from skimage.filters import threshold_otsu
from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image_path = 'picture2.jpg'
image = Image.open(image_path)
image_array = np.array(image)
grey = image.convert('L') 


def lucy_richardson_deconvolution(image):
    rng = np.random.default_rng()
    grey_picture = color.rgb2gray(image)

    psf = np.ones((5, 5)) / 25
    grey_picture = conv2(grey_picture, psf, 'same')
    noisy = grey_picture.copy()
    noisy += (rng.poisson(lam=25, size=grey_picture.shape) - 10) / 255.0
    deconvolved_RL = restoration.richardson_lucy(noisy, psf, num_iter=30)
    return deconvolved_RL, noisy

lucy_restored, noisy = lucy_richardson_deconvolution(image)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()
ax[0].imshow(grey)
ax[0].set_title('Original image')
ax[1].imshow(noisy)
ax[1].set_title('Noisy image')
ax[2].imshow(lucy_restored, vmin=noisy.min(), vmax=noisy.max())
ax[2].set_title('Restoration using Richardson-Lucy')
plt.show()
for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

def otsu_thresholding(image):
   
    if image.ndim == 3:
        image = np.mean(image, axis=2)  
    thresh_value = threshold_otsu(image)
    binary_image = image > thresh_value
    return binary_image.astype(np.uint8)  

binary_image_otsu = otsu_thresholding(image_array)

plt.figure(figsize=(5, 5))
plt.imshow(binary_image_otsu, cmap='gray')
plt.title('Binarization using Otsu\'s method')
plt.axis('off')
plt.tight_layout()
plt.show()






