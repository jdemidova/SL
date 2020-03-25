import csv
import os
import random

import sklearn
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from CV_Skin_Lesions import SLImageAnalysis


# def statistics_threshold(gray_image, seg_image):
#     hist, bins = np.histogram(gray_image.ravel(), 256, [0, 256])
#     a = 127
#     bin_type, bin_image = cv.threshold(gray_image, a, 255, cv.THRESH_BINARY_INV)
#     right_counter = 0
#     for i in range(gray_image.shape[0]):
#         for j in range(gray_image.shape[1]):
#             if bin_image[i][j] == seg_image[i][j]:
#                 right_counter += 1
#     right = right_counter / gray_image.size
#     return right, bin_image

# # адаптивная cv.adaptiveThreshold и параметр cv.ADAPTIVE_THRESH_MEAN_C
# def statistic_adaptive_mean(gray_image, seg_image, block_size, const):
#     right = 0.0
#     bin_image = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
#                                                      cv.THRESH_BINARY, block_size, const)
#     right_counter = 0
#     for i in range(gray_image.shape[0]):
#         for j in range(gray_image.shape[1]):
#             if bin_image[i][j] == seg_image[i][j]:
#                 right_counter += 1
#     right = right_counter / gray_image.size
#     return right, bin_image


# # адаптивная cv.adaptiveThreshold и параметр cv.ADAPTIVE_THRESH_GAUSSIAN_C
# def statistic_adaptive_gaussian(gray_image, seg_image, block_size, const):
#     right = 0.0
#     bin_image = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv.THRESH_BINARY, block_size, const)
#     right_counter = 0
#     for i in range(gray_image.shape[0]):
#         for j in range(gray_image.shape[1]):
#             if bin_image[i][j] == seg_image[i][j]:
#                 right_counter += 1
#     right = right_counter / gray_image.size
#     bin_image_c = cv.cvtColor(bin_image, cv.COLOR_GRAY2BGR)
#     kernel = np.ones((5, 5), np.uint8)
#     bin_image_c = cv.morphologyEx(bin_image_c, cv.MORPH_CLOSE, kernel)
#     bin_image_c = cv.morphologyEx(bin_image_c, cv.MORPH_OPEN, kernel)
#     c, hierarchy = cv.findContours(bin_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     cv.drawContours(bin_image_c, c, -1, (0, 255, 0), 3)
#     return right, bin_image_c


# метод Оцу
def statistic_Otsu(gray_image, seg_image):
    bin_otsu_inv, bin_image_inv = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    bin_otsu, bin_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    h, w = gray_image.shape[:2]
    mask_inv = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.ones((h + 2, w + 2), np.uint8) * 255
    cv.floodFill(bin_image, mask, (0, 0), 255)
    cv.floodFill(bin_image, mask, (w - 1, 0), 255)
    cv.floodFill(bin_image, mask, (0, h - 1), 255)
    cv.floodFill(bin_image, mask, (w - 1, h - 1), 255)
    cv.floodFill(bin_image_inv, mask_inv, (0, 0), 0)
    cv.floodFill(bin_image_inv, mask_inv, (w - 1, 0), 0)
    cv.floodFill(bin_image_inv, mask_inv, (0, h - 1), 0)
    cv.floodFill(bin_image_inv, mask_inv, (w - 1, h - 1), 0)
    kernel = np.ones((13, 13), np.uint8)
    bin_image_inv = cv.morphologyEx(bin_image_inv, cv.MORPH_OPEN, kernel)
    bin_image_inv = cv.morphologyEx(bin_image_inv, cv.MORPH_CLOSE, kernel)
    bin_image = cv.morphologyEx(bin_image, cv.MORPH_OPEN, kernel)
    bin_image = cv.morphologyEx(bin_image, cv.MORPH_CLOSE, kernel)
    f = sklearn.metrics.f1_score(seg_image, bin_image, average='micro')
    # print(f)
    f_inv = sklearn.metrics.f1_score(seg_image, bin_image_inv, average='micro')
    # print(f_inv)
    if f_inv > f:
        img = bin_image_inv
        var = f_inv
    else:
        img = bin_image
        var = f
    bin_image_c = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    c, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    cv.drawContours(bin_image_c, c, -1, (0, 255, 0), 1, cv.LINE_AA, hierarchy, 1)
    return var, bin_image_c


if __name__ == '__main__':
    absolute_path = os.path.join(os.getcwd(), 'data', 'metadata_global.csv')
    reader = csv.reader(open(absolute_path))
    sortedlist = sorted(reader, key=lambda row: row[1])
    counter_melanoma, counter_nevus = 0, 0
    res1, count = 0, 0
    for k in sortedlist:
        if k[6] == 'melanoma':
            counter_melanoma += 1
        else:
            if k[6] == 'nevus':
                counter_nevus += 1
        obj = SLImageAnalysis(k[1])
        img = obj.gray_image
        seg = obj.binary_image
        obj.patient_age = k[2]
        obj.diagnosis = k[6]
        res, otsu = statistic_Otsu(img, seg)
        absolute_path = os.path.join(os.getcwd(), 'data', 'images')
        # cv.imwrite(os.path.join(absolute_path, obj.name + '_new.png'), otsu)
        res1 += res
        print(str(count) + " " + obj.name + " " + obj.diagnosis + " " + obj.patient_age + " " + str(res))
        count += 1
        if count > 624:
            break
    res1 /= (counter_melanoma + counter_nevus)
    print("nevus images: " + str(counter_nevus))
    print("melanoma images: " + str(counter_melanoma))
    print("average res1: " + str(res1))

    # plt.imshow(image_otsu, cmap='gray')
    # plt.show()

