import numpy as np
import math
from scipy.signal import convolve2d
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

MAX_PIX_LEVEL = 255
GRAYSCALE = 1
PI = math.pi
I = 1j


def read_image(filename, representation):
    """
    opens image as matrix
    :param filename: name of image
    :param representation: 1 if grayscale, 2 if RGB
    :return: np.array float64 of given image
    """
    image = imread(filename)
    image_float = image.astype(np.float64) / MAX_PIX_LEVEL
    if representation == GRAYSCALE:
        image_float_gray = rgb2gray(image_float)
        return image_float_gray
    return image_float


def DFT(signal):
    """
    compute the 1-D DFT of a signal
    :param signal: representation of the signal in image spectrum
    :return: representation of the signal in fourier spectrum
    """
    n = len(signal)
    omega = np.exp((-2*PI*I) / n)
    omega_van = np.vander([omega], n, True).flatten()
    matrix_van = np.vander(omega_van, n, True)
    fourier_signal = np.dot(matrix_van, signal)
    return fourier_signal


def IDFT(fourier_signal):
    """
    compute the 1-D inverse DFT
    :param fourier_signal: representation of the signal in fourier spectrum
    :return: representation of the signal in the image spectrum
    """
    n = len(fourier_signal)
    omega = np.exp((2 * PI * I) / n)
    omega_van = np.vander([omega], n, True).flatten()
    matrix_van = np.vander(omega_van, n, True)
    matrix_van = matrix_van / n
    signal = np.dot(matrix_van, fourier_signal)
    return signal


def DFT2(image):
    """
    coumpute the 2-D DFT of an image
    :param image: image in image spectrum
    :return: image in fourier spectrum
    """
    return DFT(DFT(image.T).T)


def IDFT2(fourier_image):
    """
    coumpute the 2-D inverse DFT of an imag
    :param fourier_image: image in fourier spectrum
    :return: image in image spectrum
    """
    return IDFT(IDFT(fourier_image.T).T)


def conv_der(im):
    """
    computing the derivative of an image using convolution
    :param im: image
    :return: the magnitude of the derivative of x and of y of the image
    """
    a = np.array([-1, 0, 1])

    y_size, x_size = im.shape
    dx = np.zeros(im.shape)
    dy = np.zeros(im.shape)
    for i in range(y_size):
        dx[i,:] = np.convolve(im[i,:], a, 'same')

    for j in range(x_size):
        dy[:,j] = np.convolve(im[:,j], a, 'same')

    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def fourier_der(im):
    """
    computing the derivative of an image using fourier
    :param im: image
    :return: the magnitude of the derivative of x and of y of the image
    """
    y_size, x_size = im.shape
    x_fourier = DFT2(im)
    x_fourier = np.fft.fftshift(x_fourier)
    x_coef = (2 * PI) / x_size
    u = np.arange(np.ceil(-x_size/2), np.ceil(x_size/2))
    for i in range(y_size):
        x_fourier[i,:] *= (u*x_coef)
    dx = IDFT2(x_fourier)

    y_fourier = DFT2(im)
    y_fourier = np.fft.fftshift(y_fourier)
    y_coef = (2 * PI) / y_size
    v = np.arange(np.ceil(-y_size/2), np.ceil(y_size/2))
    for j in range(x_size):
        y_fourier[:,j] *= (v*y_coef)
    dy = IDFT2(y_fourier)

    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def get_gaussian_matrix(size):
    """
    helper function, to build a 2D gaussian to blur with (size*size), used as kernel
    :param size: the size of each axis for the gaussian
    :return: the gaussian, normalized, to be used as kernel
    """
    x = np.array([1])
    convo_multiplier = np.array([1, 2, 1])
    counter = 1
    while counter < size:
        x = np.convolve(x, convo_multiplier)
        counter += 2
    gaussian = x*x[:,None]
    gaus_sum = np.sum(gaussian)
    normalized_gaus = gaussian / gaus_sum
    return normalized_gaus


def blur_spatial(im, kernel_size):
    """
    bluring an image using 2d convolution
    :param im: image
    :param kernel_size: the size of the blurring kernel
    :return: blurred image
    """
    gausian_mat = get_gaussian_matrix(kernel_size)
    return convolve2d(im, gausian_mat, mode='same', boundary='fill', fillvalue=0)


def blur_fourier(im, kernel_size):
    """
    bluring an image using fourier
    :param im: image
    :param kernel_size:the size of the blurring kernel, before being padded with zeros
    :return: blurred image
    """
    y_size, x_size = im.shape
    g = np.zeros(im.shape).astype(np.complex128)
    gaussian_mat = get_gaussian_matrix(kernel_size)

    #add g_fourier to center of g
    g_y, g_x = gaussian_mat.shape
    start_y = (y_size//2) - (g_y//2)
    end_y = (y_size//2) + (g_y//2 + 1)
    start_x = (x_size//2) - (g_x//2)
    end_x = (x_size//2) + (g_x//2 + 1)
    g[start_y:end_y,start_x:end_x] = gaussian_mat

    g_fourier = DFT2(g)
    f_fourier = DFT2(im)
    point_multiplied = g_fourier*f_fourier
    return np.real(np.fft.fftshift(IDFT2(point_multiplied)))

# im = read_image('frog.jpg',1)
# plt.imshow(im,cmap='gray')
# plt.show()
# plt.imshow(conv_der(im),cmap='gray')
# plt.show()
# plt.imshow(fourier_der(im),cmap='gray')
# plt.show()

im2 = read_image('view.jpg',1)
plt.imshow(im2,cmap='gray')
plt.show()
plt.imshow(blur_fourier(im2,5),cmap='gray')
plt.show()
plt.imshow(blur_spatial(im2,5),cmap='gray')
plt.show()