import copy
import datetime

import numpy as np
import utilities as util
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

class Picture:
    def __init__(self, directory=None, image=None):
        """

        :param directory: directory of an image, if None image must be defined
        :param image: PIL.Image object, used only if directory is None
        """
        if directory is None:
            self.image = image
        elif image is None:
            self.directory = directory
            self.image = Image.open(self.directory)
            self.cvimage = cv2.imread(self.directory, 0)
        self.data = self.image.load()

# Utilities functions---------------------------------------------------------------------------------------------------
    def show(self):
        """Shows image"""
        self.image.show()

    def save(self, name, quality):
        """
        Saves image to file with given name
        :param name: filename with extension
        :return:
        """
        self.new_image.save(fp=f'output\\{name}', dpi=quality)

    def matrix(self, image=None):
        """
        Creates 3 dimensional matrix with pixel values in form (R, G, B)
        :param image: image instance
        :return: numpy 3D array
        """
        if image is None:
            image = self.image

        red, green, blue = image.split()
        matrix = np.array([np.array(red), np.array(green), np.array(blue)]).astype(int)

        return matrix

    def fit(self, data):
        """
        Fits outliing values to range <0;255>
        :param data: numpy 3D array
        :return: fitted pixel values
        """

        data[data > 255] = 255
        data[data < 0] = 0
        data = np.uint8(data)
        data = np.dstack((data[0], data[1], data[2]))

        return data

    def histogram(self, image=None, show=False, name='hist'):
        """
        Calculates histogram of image
        :param image:
        :param show:
        :return:
        """
        sns.set()
        sns.set_style('whitegrid')
        if image is None:
            image = self.image
        data = self.matrix(image)
        unique = []
        count = []
        for i in range(data.shape[0]):
            unique_band, count_band = np.unique(data[i], return_counts=True)
            unique.append(unique_band)
            count.append(count_band)

        if show:
            fig, ax = plt.subplots(3, 1, figsize=(8, 6))
            band = ['Kanał czerwony', 'Kanał zielony', 'Kanał niebieski']
            band_color = ['r', 'g', 'b']
            plt.suptitle('Histogram')
            for i in range(ax.shape[0]):
                plt.sca(ax[i])
                plt.xlim((0, 255))
                plt.bar(x=unique[i], height=count[i], width=1.3, color=band_color[i], linewidth=0.2)
                ax[i].set_title(band[i])
                ax[i].set_ylabel('Liczba pikseli')
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.37, hspace=0.40, top=0.90, bottom=0.05)
            plt.savefig(f'output\\{name}.png', dpi=400)

        return unique, count

    def histogram_2(self):
        """Funkcja z zajęć"""
        img2 = self.image
        pixels = img2.load()
        histR = []
        histG = []
        histB = []

        for a in range(256):
            histR.append(0)
            histG.append(0)
            histB.append(0)

        for y in range(img2.height):
            for x in range(img2.width):
                histR[pixels[x, y][0]] += 1
                histG[pixels[x, y][1]] += 1
                histB[pixels[x, y][2]] += 1

        return [histR, histG, histB]

    def cumulative_distribution(self):
        """
        Calculates cumulative distributions of an image for each band
        :return: list o list with cumulative distribtion for each band
        """
        histograms = self.histogram_2()

        sumR = 0
        sumG = 0
        sumB = 0
        cum_hR = []
        cum_hG = []
        cum_hB = []
        for i in range(len(histograms[0])):
            sumR += histograms[0][i]
            sumG += histograms[1][i]
            sumB += histograms[2][i]
            cum_hR.append(sumR)
            cum_hG.append(sumG)
            cum_hB.append(sumB)

        return [cum_hR, cum_hG, cum_hB]

# Transformation functions----------------------------------------------------------------------------------------------
    def flip(self, show=False, axis='vertical'):
        """
        Reflects image in given axis
        :param show:
        :param axis: axis to reflect image
        :return:
        """
        new_image = copy.deepcopy(self.image)
        flip_data = new_image.load()

        if axis == 'vertical':
            for i in range(self.image.width):
                for j in range(self.image.height):
                    flip_data[i, j] = self.data[-i, j]
        elif axis == 'horizontal':
            for i in range(self.image.width):
                for j in range(self.image.height):
                    flip_data[i, j] = self.data[i, -j]
        elif axis == 'both':
            for i in range(self.image.width):
                for j in range(self.image.height):
                    flip_data[i, j] = self.data[-i, -j]
        else:
            print('Niby jak mam to zrobić. Podaj poprawną metodę')
            exit()
        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def rotate(self, show=False):
        """
        Rotates image clockwise
        :param show:
        :return:
        """
        rotate_image = copy.deepcopy(self.image)
        rotate_data = np.zeros((3, rotate_image.width, rotate_image.height))

        for k in range(rotate_data.shape[0]):
            x = 0
            for i in range(rotate_data.shape[1]):
                y = 0
                for j in range(rotate_data.shape[2]):
                    rotate_data[k][i, j] = self.data[x, -y][k]
                    y+=1
                x+=1

        new_data = self.fit(rotate_data)
        new_image = Image.fromarray(new_data, 'RGB')
        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def negative(self, show=False):
        """
        Creates negative of an image
        :param show:
        :return:
        """
        start_time = datetime.datetime.now()

        neg_image = copy.deepcopy(self.image)
        neg_data = neg_image.load()
        for i in range(self.image.width):
            for j in range(self.image.height):
                neg_data[i, j] = (255 - self.data[i, j][0], 255 - self.data[i, j][1], 255 - self.data[i, j][2])
        end_time = datetime.datetime.now()
        print(end_time - start_time)
        if show:
            neg_image.show()
        self.new_image = neg_image
        return self.new_image

    def negative_LUT(self, show=False):
        """
        Creates neagative of an image. (LUT version)
        :param show:
        :return:
        """
        start_time = datetime.datetime.now()
        neg_image = copy.deepcopy(self.image)
        neg_data = neg_image.load()
        neg_values = [255-x for x in range(0, 256)]
        for i in range(self.image.width):
            for j in range(self.image.height):
                neg_data[i, j] = (neg_values[self.data[i, j][0]], neg_values[self.data[i, j][1]],
                                  neg_values[self.data[i, j][2]])
        end_time = datetime.datetime.now()
        print(end_time - start_time)

        if show:
            neg_image.show()
        self.new_image = neg_image
        return self.new_image


    def greyscale2(self, show=False):
        """
        Creates greyscaled image
        :param show:
        :return:
        """
        start = datetime.datetime.now()
        grey_image = copy.deepcopy(self.image)
        grey_data = grey_image.load()
        for i in range(self.image.width):
            for j in range(self.image.height):
                val = np.round((self.data[i, j][0]+self.data[i, j][1]+self.data[i, j][2])/3).astype(int)
                grey_data[i, j] = (val, val, val)

        end = datetime.datetime.now()
        print(end - start)
        if show:
            grey_image.show()

        self.new_image = grey_image
        return self.new_image

    def greyscale(self, show=False):
        """
        Creates greyscaled image. Improved version, faster
        :param show:
        :return:
        """
        start = datetime.datetime.now()
        new_image = copy.deepcopy(self.image)
        new_data = self.matrix(new_image)
        average = (new_data[0] + new_data[1] + new_data[2])/3
        for i in range(3):
            new_data[i] = average

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')
        end = datetime.datetime.now()
        print(end - start)

        if show:
            new_image.show()

        self.new_image = new_image
        return self.new_image

    def sepia(self, fill=15, show=False):
        """
        Creates sepia of an image
        :param show:
        :return: zwraca obiekt image
        """

        sepia_image = copy.deepcopy(self.image)
        sepia_data = sepia_image.load()
        grey = self.greyscale()

        sepia_data_matrix = self.matrix(grey)
        sepia_data_matrix[0] = sepia_data_matrix[0] + 2 * fill
        sepia_data_matrix[1] = sepia_data_matrix[1] + fill

        sepia_data = self.fit(sepia_data_matrix)
        sepia_image = Image.fromarray(sepia_data, 'RGB')
        if show:
            sepia_image.show()
        self.new_image = sepia_image
        return self.new_image

    def binarization(self, treshold, show=False):
        """
        Binarization of an image
        :param parameter: treshold to differ black and white pixels
        :param show:
        :return:
        """
        grey_image = self.greyscale()
        grey_data = grey_image.load()
        bin_image = copy.deepcopy(grey_image)
        bin_data = bin_image.load()
        bin_matrix = self.matrix(bin_image)

        for i in range(self.image.width):
            for j in range(self.image.height):
                if grey_data[i, j][0] > treshold:
                    val = 255
                else:
                    val = 0

                bin_data[i, j] = (val, val, val)

        if show:
            bin_image.show()
        self.new_image = bin_image
        return self.new_image

    def brightness(self, value, show=False):
        """
        Increases brightness in an image with given value
        :param value: Value to increase brightness
        :param show:
        :return:
        """
        new_image = copy.deepcopy(self.image)
        new_data = self.matrix(new_image)
        new_data = new_data + value
        new_data = self.fit(new_data)

        new_image = Image.fromarray(new_data, 'RGB')
        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def contrast(self, left_treshold, right_treshold=None, show=False):
        """
        Change contrast of an image
        :param left_treshold:
        :param right_treshold:
        :param show:
        :return:
        """
        new_image = copy.deepcopy(self.image)
        new_data = self.matrix(new_image)

        if right_treshold is None:
            right_treshold = 255-left_treshold

        new_data = (new_data - left_treshold)/(right_treshold - left_treshold) * 255
        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def find_tresholds(self, q=50):
        """
        Find tresholds to use in auto_contrast function
        :param q:
        :return:
        """
        img2 = self.image
        pixels = img2.load()
        histR = []
        histG = []
        histB = []

        for a in range(256):
            histR.append(0)
            histG.append(0)
            histB.append(0)

        for y in range(img2.height):
            for x in range(img2.width):
                histR[pixels[x, y][0]] += 1
                histG[pixels[x, y][1]] += 1
                histB[pixels[x, y][2]] += 1

        for i in range(256):
            if histR[i] > q:
                p1r = i
                break
        for i in range(256):
            if histG[i] > q:
                p1g = i
                break
        for i in range(256):
            if histB[i] > q:
                p1b = i
                break
        for i in range(255, 0, -1):
            if histR[i] > q:
                p2r = i
                break
        for i in range(255, 0, -1):
            if histG[i] > q:
                p2g = i
                break
        for i in range(255, 0, -1):
            if histB[i] > q:
                p2b = i
                break

        return [[p1r, p1g, p1b], [p2r, p2g, p2b]]

    def auto_contrast(self, useful=False, show=False, treshold=50, scale=2):
        """
        Automatic change of contrast
        :param useful: if True contrast tresholds are defined within useful fragment of histogram
        :param show:
        :param treshold: Number of minimal pixels of pixel value to bo considered as treshold.
                        Use only when useful set to False
        :param scale: Scale of standard deviation in useful mode
        :return:
        """
        if not useful:
            new_image = copy.deepcopy(self.image)
            new_data = self.matrix(new_image)

            tresholds = self.find_tresholds(treshold)

            for i in range(new_data.shape[0]):
                minimum = tresholds[0][i]
                maximum = tresholds[1][i]
                new_data[i] = (new_data[i] - minimum)/(maximum - minimum) * 255

        else:
            new_image = copy.deepcopy(self.image)
            new_data = self.matrix(new_image)

            for i in range(new_data.shape[0]):
                mean = new_data[i].mean()
                std = new_data[i].std()
                minimum = mean - scale*std
                maximum = mean + scale*std

                new_data[i] = (new_data[i] - minimum) / (maximum - minimum) * 255

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def hist_equalize(self, show=False):
        """
        Equlize histogram of an image
        :param show:
        :return:
        """
        cum_distribution = self.cumulative_distribution()

        N = self.image.width * self.image.height

        new_image = copy.deepcopy(self.image)
        new_data = self.matrix(new_image)

        equalR = []
        equalG = []
        equalB = []
        for j in range(len(cum_distribution[0])):
            equalR.append(int(cum_distribution[0][j] * 255 / N))
            equalG.append(int(cum_distribution[1][j] * 255 / N))
            equalB.append(int(cum_distribution[2][j] * 255 / N))

        red = new_data[0].flatten()
        green = new_data[1].flatten()
        blue = new_data[2].flatten()
        for i in range(red.shape[0]):
            red[i] = equalR[int(red[i])]
            green[i] = equalG[int(green[i])]
            blue[i] = equalB[int(blue[i])]

        new_data[0] = red.reshape(self.image.height, self.image.width)
        new_data[1] = green.reshape(self.image.height, self.image.width)
        new_data[2] = blue.reshape(self.image.height, self.image.width)

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image

    def add(self, add_image, show=True):
        new_image = copy.deepcopy(self.image)
        add_data = self.matrix(add_image)
        new_data = self.matrix(new_image)[:, :1498, :1198]
        new_data = new_data - add_data/20

        new_data = self.fit(new_data)
        new_image = Image.fromarray(new_data, 'RGB')

        if show:
            new_image.show()
        self.new_image = new_image
        return self.new_image