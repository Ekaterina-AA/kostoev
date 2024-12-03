import numpy as np
import matplotlib.pyplot as plt
   

a = 1  
b = 1

x = np.linspace(-5, 5, 1000) 

f1 = np.exp(-a**2 * x**2)
f2 = 1 / (1 + (b**2 * x**2))
f3 = np.sin(a * x) / (1 + (b * x**2))

def plot_fourier_transform(f_x, title):
    F_k = np.fft.fft(f_x)
    k = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k, np.real(F_k))
    plt.title(f'Real part of the Fourier image {title}')
    plt.xlabel('Frequency k')
    plt.ylabel('Real part')
    plt. xlim(-5,5)

    plt.subplot(1, 2, 2)
    plt.plot(k, np.imag(F_k))
    plt.title(f'Imaginary part of the Fourier image {title}')
    plt.xlabel('Frequency k')
    plt.ylabel('Imaginary part')
    plt. xlim(-5,5)

    plt.tight_layout()
    plt.show()

plot_fourier_transform(f1, 'e^(-a^2 * x^2)')
plot_fourier_transform(f2, '1/(1 + b^2 * x^2)')
plot_fourier_transform(f3, 'sin(ax)/(1 + b * x^2)')
