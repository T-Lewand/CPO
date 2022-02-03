from obraz import *
from Filter import Filter
from utilities import *
import os
from classifier import MLClassifier
directory = os.getcwd()
print(directory)
os.chdir(directory[0:-6])
# pictures_list = ['Lenna.bmp', '4.2.06.tiff', 'Bahamy.tif', 'false_color.tif', 'true_color.tif', 'Take2.png']
images = list_files('output\\')
print(images)
for i in images:
    if i =='gimp':
        continue
    print(i)
    image = Picture(f'output\\{i}')
    image.histogram(show=True, name=f'{i[:-4]}_hist')






exit()
# image.flip(show=True, axis='both')
# image.save('flipped2.tif')
# image.rotate(show=True)
# image.save('rotated.tif')
# image.negative(show=True)
# image.save('negative.tif')
# image.greyscale(show=True)
# image.save('greyscale.tif')
# image.sepia(show=True)
# image.save('sepia.tif')
# image.binarization(treshold=50, show=True)
# image.save('binarization50.tif')
# image.brightness(value=70, show=True)
# image.save('brightnes70.tif')
# contrast = image.contrast(left_treshold=40, show=False)

# image.save('contrast40.tif')
# image.auto_contrast(show=True, treshold=25)

# image.save('auto_contrast25.tif')
# contrast = image.hist_equalize(show=True)
# image.histogram(contrast, show=True)
# image.save('hist_equalize.tif')
# filter = Filter(image.image)
# filtrated = filter.laplace()
# filtrated.save('Laplace.tif')
# filtrated = filter.prewitt(sobel=True, axis='horizontal')
# filtrated.save('sobel.tif')
# filtrated = filter.average()
# filtrated.save('average.tif')
#image.kolejne()
#image.hist_equalize()
#image.auto_contrast(show=True, useful=False, treshold=100)
#image.auto_contrast(show=True, useful=True, scale=3)


