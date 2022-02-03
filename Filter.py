from obraz import  Picture
import copy
import datetime
import numpy as np
import utilities as util
from PIL import Image


class Filter(Picture):
    def __init__(self, image):
        self.image = image
        self.data = self.image.load()

    def convolute(self, data, kernel):
        """
        Calculates convolution of data and kernel matrix
        :param data: 3D matrix of pixels values in shape (band, y, x)
        :param kernel: 2D kernel matrix
        :return:
        """
        m, n = kernel.shape
        z, y, x = data.shape

        x = x - m + 1
        y = y - m + 1
        processed_data = np.zeros((z, y, x), dtype=np.int8)

        for k in range(z):
            for i in range(y):
                for j in range(x):
                    processed_data[k][i][j] = np.sum(data[k][i:i + m, j:j + m] * kernel)

        return np.abs(processed_data)

    def average(self, kernel_size=5):
        """
        Average filtration
        :return:
        """
        image = copy.deepcopy(self.image)
        data = self.matrix(image)
        kernel = np.ones((kernel_size, kernel_size))*2


        if np.sum(kernel) != 0:
            kernel = kernel/np.sum(kernel)

        new_data = self.convolute(data, kernel)

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        new_image.show()

        self.new_image = new_image
        return self.new_image

    def gaussian(self, kernel_size):
        image = copy.deepcopy(self.image)
        data = self.matrix(image)
        pass

    def roberts(self):
        image = self.greyscale()
        data = self.matrix(image)
        kernel = np.zeros((3, 3))
        kernel[0, 1] = -1
        kernel[1, 0] = 1
        kernel[1, 2] = 0
        kernel[2, 1] = 0

        new_data = self.convolute(data, kernel)

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        new_image.show()

        self.new_image = new_image
        return self.new_image

    def prewitt(self, axis='vertical', sobel=False):
        image = self.greyscale()
        data = self.matrix(image)
        kernel = np.zeros((3, 3))
        kernel[0, 0] = -1
        kernel[1, 0] = -1
        kernel[2, 0] = -1
        kernel[0, 2] = 1
        kernel[1, 2] = 1
        kernel[2, 2] = 1

        if sobel:
            kernel[1, 0] = -2
            kernel[1, 2] = 2

        if axis == 'horizontal':
            kernel = kernel.T

        new_data = self.convolute(data, kernel)

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        new_image.show()

        self.new_image = new_image
        return self.new_image

    def laplace(self):
        image = self.greyscale()
        data = self.matrix(image)
        kernel = np.ones((3, 3)) * (-1)
        kernel[1, 1] = 8

        new_data = self.convolute(data, kernel)
        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        new_image.show()

        self.new_image = new_image
        return self.new_image