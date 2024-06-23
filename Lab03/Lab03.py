import numpy as np 
import pandas as pd
import cv2 
import os
from matplotlib import pyplot as plt 
from pylab import imread
from skimage.color import rgb2gray

def imshows(ImageData, LabelData, rows, cols, gridType = False):
  # Convert ImageData and LabelData to List
  from matplotlib import pyplot as plt
  ImageArray = list(ImageData)
  LabelArray = list(LabelData)
  if(rows == 1 & cols == 1):
    fig = plt.figure(figsize=(20,20))
  else:
    fig = plt.figure(figsize=(cols*8,rows*5))
        
  for i in range(1, cols * rows + 1):
      fig.add_subplot(rows, cols, i)
      image = ImageArray[i - 1]
      # If the channel number is less than 3, we display as grayscale image
      # otherwise, we display as color image
      if (len(image.shape) < 3):
          plt.imshow(image, plt.cm.gray)
          plt.grid(gridType)
      else:
          plt.imshow(image)
          plt.grid(gridType)
      plt.title(LabelArray[i - 1])
  plt.show()

def ShowThreeImages(IM1, IM2, IM3):
    imshows([IM1, IM2, IM3], ["Image 1","Image 2", "Image 3"], 1, 3)
def ShowTwoImages(IM1, IM2):
    imshows([IM1, IM2], ["Image 1","Image 2"], 1, 2)
def ShowOneImage(IM):
    imshows([IM], ["Image"], 1, 1)
def ShowListImages(listImage, row, col):
    listCaption = []
    for i in range(len(listImage)):
        listCaption.append(str(i))
    imshows(listImage,listCaption,row,col)

## Read Image 
#image_color = imread("Sample03/tom.jpg")
## Convert Image into Gray
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
## Display Image
#ShowTwoImages(image_color, image_gray)

kernel_3_3 = np.ones((3, 3), np.float32) / 9
kernel_5_5 = np.ones((5, 5), np.float32) / 25

#image_filter_3_3_01 = cv2.filter2D(image_color, -1, kernel_3_3)
#image_filter_3_3_02 = cv2.filter2D(image_filter_3_3_01, -1, kernel_3_3)

#image_filter_5_5_01 = cv2.filter2D(image_color, -1, kernel_5_5)
#image_filter_5_5_02 = cv2.filter2D(image_filter_5_5_01, -1, kernel_5_5)

#ShowThreeImages(image_color, image_filter_3_3_01, image_filter_3_3_02)
#ShowThreeImages(image_color, image_filter_5_5_01, image_filter_5_5_02)

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

#blur_measurement = variance_of_laplacian(image_color)
#blur_measurement_3_3_01 = variance_of_laplacian(image_filter_3_3_01)
#blur_measurement_3_3_02 = variance_of_laplacian(image_filter_3_3_02)
#print("Blur Measurement of image_color:", blur_measurement)
#print("Blur Measurement of image_filter_3_3_01:", blur_measurement_3_3_01)
#print("Blur Measurement of image_filter_3_3_02:", blur_measurement_3_3_02)
#print()

#text = "Blurry measurement"
#fm = blur_measurement
#image_color_text = image_color.copy()
#cv2.putText(image_color_text, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#text = "Blurry measurement"
#fm = blur_measurement_3_3_01
#image_filter_3_3_01_text = image_filter_3_3_01.copy()
#cv2.putText(image_filter_3_3_01_text, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#text = "Blurry measurement"
#fm = blur_measurement_3_3_02
#image_filter_3_3_02_text = image_filter_3_3_02.copy()
#cv2.putText(image_filter_3_3_02_text, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#ShowThreeImages(image_color_text, image_filter_3_3_01_text, image_filter_3_3_02_text)

#blur_measurement = variance_of_laplacian(image_color)
#blur_measurement_5_5_01 = variance_of_laplacian(image_filter_5_5_01)
#blur_measurement_5_5_02 = variance_of_laplacian(image_filter_5_5_02)
#print("Blur Measurement of image_color:", blur_measurement)
#print("Blur Measurement of image_filter_5_5_01:", blur_measurement_5_5_01)
#print("Blur Measurement of image_filter_5_5_02:", blur_measurement_5_5_02)
#print()

#text = "Blurry measurement"
#fm = blur_measurement_5_5_01
#image_filter_5_5_01_text = image_filter_5_5_01.copy()
#cv2.putText(image_filter_5_5_01_text, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#text = "Blurry measurement"
#fm = blur_measurement_5_5_02
#image_filter_5_5_02_text = image_filter_5_5_02.copy()
#cv2.putText(image_filter_5_5_02_text, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#ShowThreeImages(image_color_text, image_filter_5_5_01_text, image_filter_5_5_02_text)

#image_gray_filter = cv2.filter2D(image_gray, -1, kernel_5_5)
#ShowTwoImages(image_gray, image_gray_filter)
#print("Blur Measurement of image_gray:", variance_of_laplacian(image_gray))
#print("Blur Measurement of image_gray_filter:", variance_of_laplacian(image_gray_filter))
#print()

#image_gray_filter_color = cv2.cvtColor(image_gray_filter, cv2.COLOR_GRAY2RGB)
#text = "Blurry measurement"
#fm = variance_of_laplacian(image_gray_filter)
#cv2.putText(image_gray_filter_color, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#ShowTwoImages(image_gray, image_gray_filter_color)

# Read Image 
image_color = imread("Sample03/(84).jpg")
# Convert Image into Gray
image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
# Display Image
ShowTwoImages(image_color, image_gray)

#def max_rgb_filter(image):
#    # split the image into its BGR components
#    (B, G, R) = cv2.split(image)
#    # find the maximum pixel intensity values for each
#    # (x, y)-coordinate,, then set all pixel values less
#    # than M to zero
#    M = np.maximum(np.maximum(R, G), B)
#    R[R < M] = 0
#    G[G < M] = 0
#    B[B < M] = 0
#    # merge the channels back together and return the image
#    return cv2.merge([B, G, R])

#image_color_rgbmax = max_rgb_filter(image_color)
#ShowTwoImages(image_color, image_color_rgbmax)

#def SegmentColorImageByMask(IM, Mask):    
#    Mask = Mask.astype(np.uint8)
#    result = cv2.bitwise_and(IM, IM, mask = Mask)
#    return result

#image_maxR_mask = image_gray < 0
#image_maxG_mask = image_gray < 0
#image_maxB_mask = image_gray < 0

#R = image_color_rgbmax[:,:,0]
#G = image_color_rgbmax[:,:,1]
#B = image_color_rgbmax[:,:,2]

#image_maxR_mask[(G == 0) & (B == 0)] = 1
#image_maxG_mask[(R == 0) & (B == 0)] = 1
#image_maxB_mask[(G == 0) & (R == 0)] = 1

#image_maxR = SegmentColorImageByMask(image_color, image_maxR_mask)
#image_maxG = SegmentColorImageByMask(image_color, image_maxG_mask)
#image_maxB = SegmentColorImageByMask(image_color, image_maxB_mask)

#ShowThreeImages(image_maxR_mask, image_maxG_mask, image_maxB_mask)
#ShowThreeImages(image_maxR, image_maxG, image_maxB)

## Read Image 
#image_color = imread("Sample03/windowxp.jpg")
## Convert Image into Gray
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
## Display Image
#ShowTwoImages(image_color, image_gray)

kernel_sharpen_01 = np.array([[-1,-1,-1], 
                              [-1, 9,-1], 
                              [-1,-1,-1]])

kernel_sharpen_02 = np.array(([0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]), dtype="int")

image_color_sharpen_01 = cv2.filter2D(image_color, -1, kernel_sharpen_02)
image_color_sharpen_02 = cv2.filter2D(image_color_sharpen_01, -1, kernel_sharpen_02)
image_color_sharpen_02 = cv2.filter2D(image_color_sharpen_02, -1, kernel_sharpen_02)
#image_color_sharpen_01 = cv2.filter2D(image_color, -1, kernel_sharpen_01)
#image_gray_sharpen_01 = cv2.filter2D(image_gray, -1, kernel_sharpen_01)

#image_color_sharpen_02 = cv2.filter2D(image_color, -1, kernel_sharpen_02)
#image_gray_sharpen_02 = cv2.filter2D(image_gray, -1, kernel_sharpen_02)

#print("Sharpen Measurement of image_color:", variance_of_laplacian(image_color))
#print("Sharpen Measurement of image_color_sharpen_01:", variance_of_laplacian(image_color_sharpen_01))
#print("Sharpen Measurement of image_color_sharpen_02:", variance_of_laplacian(image_color_sharpen_02))
#print()

ShowThreeImages(image_color, image_color_sharpen_01, image_color_sharpen_02)

#print("Sharpen Measurement of image_gray:", variance_of_laplacian(image_gray))
#print("Sharpen Measurement of image_gray_sharpen_01:", variance_of_laplacian(image_gray_sharpen_01))
#print("Sharpen Measurement of image_gray_sharpen_01:", variance_of_laplacian(image_gray_sharpen_02))
#print()

#ShowThreeImages(image_gray, image_gray_sharpen_01, image_gray_sharpen_02)

## Read Image 
#image_color = imread("Sample03/animal.jpg")
## Convert Image into Gray
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
## Display Image
#ShowTwoImages(image_color, image_gray)

## construct the Laplacian kernel used to detect edge-like
## regions of an image
## construct the Sobel x-axis kernel
#kernel_sobelX = np.array((
#    [-1, 0, 1],
#    [-2, 0, 2],
#    [-1, 0, 1]), dtype="int")
 
## construct the Sobel y-axis kernel
#kernel_sobelY = np.array((
#    [-1, -2, -1],
#    [0, 0, 0],
#    [1, 2, 1]), dtype="int")

#image_color_edge_sobelX = cv2.filter2D(image_color, -1, kernel_sobelX)
#image_color_edge_sobelY = cv2.filter2D(image_color, -1, kernel_sobelY)
#image_color_edge = image_color_edge_sobelX + image_color_edge_sobelY

#ShowTwoImages(image_color, image_color_edge)
#ShowTwoImages(image_color_edge_sobelX, image_color_edge_sobelY)

#image_gray_edge_sobelX = cv2.filter2D(image_gray, -1, kernel_sobelX)
#image_gray_edge_sobelY = cv2.filter2D(image_gray, -1, kernel_sobelY)
#image_gray_edge = image_gray_edge_sobelX + image_gray_edge_sobelY

#ShowTwoImages(image_gray, image_gray_edge)
#ShowTwoImages(image_gray_edge_sobelX, image_gray_edge_sobelY)

## Read Image 
#image_color = imread("Sample03/product.jpg")
## Convert Image into Gray
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
## Display Image
#ShowTwoImages(image_color, image_gray)

## Use function OpenCV 
#image_color_blur_01 = cv2.blur(image_color, (15, 15))
#image_gray_blur_01 = cv2.blur(image_gray, (15, 15))

#image_color_blur_02 = cv2.GaussianBlur(image_color, (15, 15), 0)
#image_gray_blur_02 = cv2.GaussianBlur(image_gray, (15, 15), 0)

#ShowThreeImages(image_color, image_color_blur_01, image_color_blur_02)
#ShowThreeImages(image_gray, image_gray_blur_01, image_gray_blur_02)

#image_color_blur_03 = cv2.medianBlur(image_color, 15)
#image_gray_blur_03 = cv2.medianBlur(image_gray, 15)
#ShowThreeImages(image_color, image_color_blur_03, image_gray_blur_03)

## Read Image 
#image_color = imread("Sample03/pattern.jpg")
## Convert Image into Gray
#image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
## Display Image
#ShowTwoImages(image_color, image_gray)

#image_color_blur = cv2.bilateralFilter(image_color, 20, 100, 100)
#ShowTwoImages(image_color, image_color_blur)

#def gaussian_kernel(size, sigma=1):
#    size = int(size) // 2
#    x, y = np.mgrid[-size:size+1, -size:size+1]
#    normal = 1 / (2.0 * np.pi * sigma**2)
#    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#    return g

#kernel = gaussian_kernel(15)
#image_color_gaussion = cv2.filter2D(image_color, -1, kernel)
#ShowTwoImages(image_color, image_color_gaussion)

#def read_and_check_images_from_folder(folder):
#    images = []
#    names = []
#    blur_check = 500
#    for filename in os.listdir(folder):
#        img = imread(os.path.join(folder, filename))
#        if img is not None:
#            blur_measurement = variance_of_laplacian(img)
#            if (blur_measurement < blur_check):
#                text = "Blurry Image"
#            else:
#                text = "Good Image"
#            cv2.putText(img, "{}".format(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#            images.append(img)
#            names.append(filename)
#    path = "Image_BlurDetection_Output"
#    os.mkdir(path)
#    os.chdir(path)
#    for i in range(len(images)):
#        plt.imsave(names[i], images[i])

#read_and_check_images_from_folder("Image_Input")