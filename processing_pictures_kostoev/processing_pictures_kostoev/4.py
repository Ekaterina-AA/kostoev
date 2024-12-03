import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('picture2.jpg', cv.IMREAD_GRAYSCALE)


def plot_histogram(image):
    plt.hist(image.flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

def equalize_histogram(image):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
 
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[image]
     
   #plt.plot(cdf_normalized, color = 'b')
   #plt.hist(img2.flatten(),256,[0,256], color = 'r')
   #plt.xlim([0,256])
   #plt.legend(('cdf','histogram'), loc = 'upper left')
   #plt.show()
    
    return Image.fromarray(img2)

plot_histogram(img) 
equalized_image = equalize_histogram(img)


plt.imshow(equalized_image, cmap='gray')
plt.title('equalized_image')
plt.axis('off') 
plt.show()

plt.imshow(img, cmap='gray')
plt.title('not_equalized_image')
plt.axis('off') 
plt.show()




