from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt

inpath = "/home/anachronicnomad/Desktop/2020_03_05/"
filenames = [f for f in listdir(inpath) if isfile(join(inpath, f))]
ODIR = '/home/anachronicnomad/Desktop/clipLimit64_grid32/clip64_grid32/'
ODIR4color = '/home/anachronicnomad/Desktop/clipLimit64_grid32/clip64_grid32_4c/'
ODIR5color = '/home/anachronicnomad/Desktop/clipLimit64_grid32/clip64_grid32_5c/'

for fname in filenames:
    FILENAME = inpath + fname
    img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=64, tileGridSize=(32,32))
    equalized = clahe.apply(img)

    # Truncate all image values below midpoint threshold
    ret,truncd = cv2.threshold(equalized,127,255,cv2.THRESH_TRUNC)

    # Take 5x5 pixel normalized average across image
    blur = cv2.GaussianBlur(truncd,(5,5),0)

    # Apply binary threshold, Otsu threshold
    ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Filter morphology - close gaps
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    ## Contour to find bone region
    #ret4 = cv2.cvtColor(ret3, cv2.COLOR_GRAY2BGR)
    ret4 = closing.copy()
    contours, hierarchy = cv2.findContours(ret4,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_TC89_L1)



    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    print(max_index)

    color_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    #out = cv2.drawContours(color_img, contours, max_index, (0,0,255), 3)
    out = cv2.drawContours(color_img, contours, -1, (0,0,255), 2)

    ## isolate bone
    bone = contours[max_index]

    fill_color = [0,0,0]
    mask_value = 255

    stencil = np.zeros(color_img.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, [bone], mask_value)

    sel = (stencil != mask_value)
    color_img[sel] = fill_color

    BASE_IMG = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    ## write out the equalized img
    OFILE = ODIR + fname
    cv2.imwrite(OFILE, BASE_IMG)

    ## Apply Color Quantization
    img = BASE_IMG.copy()
    Z = img.reshape((-1,2))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,
                                cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # write out
    OFILE = ODIR4color + fname
    cv2.imwrite(OFILE, res2)

    ## color quant with K+1
    K = 5
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,
                                cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # write out
    OFILE = ODIR5color + fname
    cv2.imwrite(OFILE, res2)

