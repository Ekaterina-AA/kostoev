import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def add_noise(image):
    noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def inverse_filter(blurred, kernel):
    kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
    blurred_ft = np.fft.fft2(blurred)
    epsilon = 1e-1
    recovered_ft = blurred_ft / (kernel_ft + epsilon)
    recovered = np.fft.ifft2(recovered_ft)
    return np.abs(recovered)



def wiener_filter(blurred, kernel, noise_var, estimated_noise):
    kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
    blurred_ft = np.fft.fft2(blurred)

    wiener_filter = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + noise_var/estimated_noise)
    recovered_ft = wiener_filter * blurred_ft
    recovered = np.fft.ifft2(recovered_ft)
    return np.abs(recovered)

def tikhonov_filter(blurred, kernel, lam):
    kernel_ft = np.fft.fft2(kernel, s=blurred.shape)
    blurred_ft = np.fft.fft2(blurred)

    tikhonov_filter = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + lam)
    recovered_ft = tikhonov_filter * blurred_ft
    recovered = np.fft.ifft2(recovered_ft)
    return np.abs(recovered)

image = cv2.imread('picture2.jpg', cv2.IMREAD_GRAYSCALE)
blurred = add_noise(image)

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]]) / 255.0

noise_var = 20
estimated_noise = 1
lambda_param = 0.1

inverse_recovered = inverse_filter(blurred, kernel)
wiener_recovered = wiener_filter(blurred, kernel, noise_var, estimated_noise)
tikhonov_recovered = tikhonov_filter(blurred, kernel, lambda_param)

plt.figure(figsize=(12, 8))
plt.subplot(3, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('Blurred Image')
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Inverse Filter')
plt.imshow(inverse_recovered, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Wiener Filter')
plt.imshow(wiener_recovered, cmap='gray')
plt.axis('off')


plt.subplot(1, 4, 4)
plt.title('Tikhonov Filter')
plt.imshow(tikhonov_recovered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

