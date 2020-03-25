import os
import math
import numpy as np
import cv2 as cv
import csv
from matplotlib import pyplot as plt
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd


class SLImageAnalysis:
    def __init__(self, name):
        self.name = name
        absolute_path = os.path.join(os.getcwd(), 'data', 'images', self.name + ".jpg")
        self.image = cv.imread(absolute_path)
        self.gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.sizeY = self.image.shape[0]
        self.sizeX = self.image.shape[1]
        self.diagnosis = ""
        self.patient_age = 0.0
        self.binary_image = self.binarization()
        self.contour = self.contour_init()
        self.area, self.perimeter, self.GD, self.SD, self.CRC, self.irA, \
            self.irB, self.irC, self.irD = self.contour_features()
        self.gray_hist_var, self.gray_hist_m3, self.gray_hist_m4, self.gray_hist_median, self.gray_hist_mean,\
            self.gray_hist_max = self.histogram_gray_features()
        self.BGR_hist_var, self.BGR_hist_m3, self.BGR_hist_m4, self.BGR_hist_median, self.BGR_hist_mean, \
            self.BGR_hist_max = self.histogram_BGR_features()

    def binarization(self):
        # удаление волос DoG-фильтром???
        # какая-то бинаризация стандартной функцией пороговой обработки
        # пока что просто берем бинарное изображение из датасета!
        absolute_path = os.path.join(os.getcwd(), 'data', 'images', self.name + "_segmentation.png")
        bin_img = cv.imread(absolute_path, cv.IMREAD_GRAYSCALE)
        return bin_img

    # точки контура объекта на бинарном изображении
    def contour_init(self):
        c, hierarchy = cv.findContours(self.binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return c

    # геом. признаки выделенной области
    # площадь, периметр, наиб. диаметр, наим. диаметр, индекс округлости,
    # индексы неравномерности A, B, C, D - всего 9 признаков
    def contour_features(self):
        img_size = self.sizeX * self.sizeY
        area = cv.contourArea(self.contour[0]) / img_size
        perimeter = cv.arcLength(self.contour[0], True) / img_size
        rect = cv.minAreaRect(self.contour[0])
        greatest_diameter = max(rect[1][0], rect[1][1]) / img_size
        shortest_diameter = min(rect[1][0], rect[1][1]) / img_size
        CRC_index = 4.0 * area * math.pi / (perimeter ** 2.0)
        irA = perimeter / area
        irB = perimeter / greatest_diameter
        irC = perimeter * (1.0 / shortest_diameter + 1.0)
        irD = greatest_diameter - shortest_diameter
        return area, perimeter, greatest_diameter, shortest_diameter, \
            CRC_index, irA, irB, irC, irD

    def contour_drawing(self):
        image = self.image
        contours = self.contour
        cv.drawContours(image, contours, -1, (0, 255, 0), 3)
        rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0, 0, 255), 2)
        return image

    # нормализованные RGB-гистограммы пораженной области
    # по ОУ - доля пикселей с данным значением в области объекта
    # по ОХ - значения пикселя R, G или B
    def histograms_BGR(self):
        img = self.image
        img_bin = self.binary_image
        res = [None] * 3
        for i in range(3):
            hist_arr = cv.calcHist([img], [i], img_bin, [256], [0, 256])
            res[i] = np.true_divide(hist_arr, self.area * self.sizeY * self.sizeX)
        return res

    # нормализованная гистограмма яркости пикселей в области объекта
    def histogram_gray(self):
        img = self.gray_image
        img_bin = self.binary_image
        hist_obj = cv.calcHist([img], [0], img_bin, [256], [0, 256])
        res = np.true_divide(hist_obj, self.area * self.sizeY * self.sizeX)
        return res

    # возвращает вектор со свойствами заданной гистограммы:
    # дисперсия, коэффициент асимметрии, коэффициент эксцесса,
    # медиана, среднее, максимум
    def histogram_gray_features(self):
        hist = self.histogram_gray()
        return np.var(hist), st.moment(hist, 3)[0], st.moment(hist, 4)[0], \
            np.median(hist), np.mean(hist), np.amax(hist)

    def histogram_BGR_features(self):
        hists = self.histograms_BGR()
        # некрасиво, исправить?
        v = [None] * 3
        m3 = [None] * 3
        m4 = [None] * 3
        med = [None] * 3
        mn = [None] * 3
        mx = [None] * 3
        for i in range(3):
            v[i] = np.var(hists[i])
            m3[i] = st.moment(hists[i], 3)[0]
            m4[i] = st.moment(hists[i], 4)[0]
            med[i] = np.median(hists[i])
            mn[i] = np.mean(hists[i])
            mx[i] = np.amax(hists[i])
        return v, m3, m4, med, mn, mx

    def get_all_features(self):
        return self.name, self.diagnosis, self.patient_age, \
            self.area, self.perimeter, self.GD, self.SD, self.CRC, self.irA,\
            self.irB, self.irC, self.irD, \
            self.gray_hist_var, self.gray_hist_m3, self.gray_hist_m4, self.gray_hist_median, \
            self.gray_hist_mean, self.gray_hist_max, \
            self.BGR_hist_var[0], self.BGR_hist_m3[0], self.BGR_hist_m4[0], self.BGR_hist_median[0],\
            self.BGR_hist_mean[0], self.BGR_hist_max[0], \
            self.BGR_hist_var[1], self.BGR_hist_m3[1], self.BGR_hist_m4[1], self.BGR_hist_median[1],\
            self.BGR_hist_mean[1], self.BGR_hist_max[1],\
            self.BGR_hist_var[2], self.BGR_hist_m3[2], self.BGR_hist_m4[2], self.BGR_hist_median[2],\
            self.BGR_hist_mean[2], self.BGR_hist_max[2]


def write_features():
    absolute_path = os.path.join(os.getcwd(), 'data', 'metadata_global.csv')
    reader = csv.reader(open(absolute_path))
    sortedlist = sorted(reader, key=lambda row: row[1])
    counter_melanoma_test, counter_nevus_test, counter_melanoma_learn, counter_nevus_learn, \
        counter_melanoma, counter_nevus = 0, 0, 0, 0, 0, 0
    arr, arr_test, arr_learn = [], [], []
    for k in sortedlist:
        if k[6] == 'melanoma':
            counter_melanoma += 1
        else:
            if k[6] == 'nevus':
                counter_nevus += 1
            else:
                continue
        obj = SLImageAnalysis(k[1])
        obj.diagnosis = k[6]
        obj.patient_age = k[2]
        data = obj.get_all_features()
        s = ""
        for i in data:
            s += str(i) + ","
        if bool(random.getrandbits(1)):
            arr_test.append(s.split(","))
            if k[6] == 'nevus':
                counter_nevus_test += 1
            else:
                counter_melanoma_test += 1
        else:
            arr_learn.append(s.split(","))
            if k[6] == 'nevus':
                counter_nevus_learn += 1
            else:
                counter_melanoma_learn += 1
        arr.append(s.split(","))
    absolute_path_test = os.path.join(os.getcwd(), 'data', 'all_features_test.csv')
    absolute_path_learn = os.path.join(os.getcwd(), 'data', 'all_features_learn.csv')
    path = os.path.join(os.getcwd(), 'data', 'all_features.csv')
    with open(absolute_path_test, "w", newline='') as file_test:
        writer = csv.writer(file_test, delimiter=',')
        for line in arr_test:
            writer.writerow(line)
    with open(absolute_path_learn, "w", newline='') as file_learn:
        writer = csv.writer(file_learn, delimiter=',')
        for line in arr_learn:
            writer.writerow(line)
    with open(path, "w", newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for line in arr:
            writer.writerow(line)
    print("nevus images: " + str(counter_nevus))  # 616
    print("   nevus learning images: " + str(counter_nevus_learn))  # 306
    print("   nevus testing images: " + str(counter_nevus_test))  # 310
    print("melanoma images: " + str(counter_melanoma))  # 632
    print("   melanoma learning images: " + str(counter_melanoma_learn))  # 328
    print("   melanoma testing images: " + str(counter_melanoma_test))  # 303


def random_forest_classification():
    learn_path = os.path.join(os.getcwd(), 'data', 'all_features_learn.csv')
    reader = csv.reader(open(learn_path))
    print("Read learning data...")
    train, train_labels = np.empty((0, 34), dtype=float), np.empty((0,), dtype=str)
    col = []
    for k in reader:
        if k[1] == 'melanoma' or k[1] == 'nevus':
            train = np.append(train, [k[2:]], axis=0)
            train_labels = np.append(train_labels, k[1])
        else:
            col = k[2:]
            continue
    train_copy = np.empty((np.shape(train)[0], 34), dtype=float)
    for i in range(np.shape(train)[0]):
        train_copy[i] = np.ndarray.astype(train[i], float)

    test_path = os.path.join(os.getcwd(), 'data', 'all_features_test.csv')
    reader = csv.reader(open(test_path))
    print("Read testinging data...")
    test, test_labels = np.empty((0, 34), dtype=float), np.empty((0,), dtype=str)
    for k in reader:
        if k[1] == 'melanoma' or k[1] == 'nevus':
            test = np.append(test, [k[2:]], axis=0)
            test_labels = np.append(test_labels, k[1])
        else:
            continue
    test_copy = np.empty((np.shape(test)[0], 34), dtype=float)
    for i in range(np.shape(test)[0]):
        test_copy[i] = np.ndarray.astype(test[i], float)

    print("Classifier init...")
    model = RandomForestClassifier(n_estimators=6000, bootstrap=True, max_features='sqrt')
    print("Training the model...")
    model.fit(train_copy, train_labels)
    print("Training results:")
    print("   Feature importances: ")
    print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), col), reverse=True))
    n_nodes, max_depths = [], []
    # Stats about the trees in random forest
    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)
    print(f'   Average number of nodes {int(np.mean(n_nodes))}')
    print(f'   Average maximum depth {int(np.mean(max_depths))}')
    # Training predictions (to demonstrate overfitting)
    train_predictions = model.predict(train_copy)[:]
    train_probs = model.predict_proba(train_copy)[:]
    # Testing predictions (to determine performance)
    test_predictions = model.predict(test_copy)[:]
    test_probs = model.predict_proba(test_copy)[:]
    print("  Training data class probabilities: " + str(model.classes_))
    print(train_probs)
    # print("  Training data predictions: ")
    # print(train_predictions)
    print("  Testing data class probabilities: " + str(model.classes_))
    print(test_probs)
    # print("  Testing data predictions: ")
    # print(test_predictions)


if __name__ == '__main__':
    print("main")
    # write_features()
    random_forest_classification()







    # absolute_path = os.path.join(os.getcwd(), 'data', 'metadata_global_sorted.csv')
    # writer = csv.writer(open(absolute_path), delimiter=',')
    # for i in sortedlist:
    #     writer.writerow(sortedlist[i][1], sortedlist[i][2], sortedlist[i][6])
    # obj = SLImageAnalysis("ISIC_0001100")
    # print("Возраст: " + str(obj.patient_age))
    # print("Диагноз: " + str(obj.diagnosis))
    # img = obj.contour_drawing()
    # img_b = obj.binary_image
    # height, width = img_b.shape[:2]
    # print("area: " + str(obj.area))
    # print("perimeter: " + str(obj.perimeter))
    # print("GD: " + str(obj.GD))
    # print("SD: " + str(obj.SD))
    # print("CRC: " + str(obj.CRC))
    # print("IrA: " + str(obj.irA))
    # print("IrB: " + str(obj.irB))
    # print("IrC: " + str(obj.irC))
    # print("IrD: " + str(obj.irD))
    # cv.namedWindow('window1', cv.WINDOW_NORMAL)
    # cv.resizeWindow('window1', width, height)
    # cv.imshow('window1', img)
    # cv.namedWindow('window2', cv.WINDOW_NORMAL)
    # cv.resizeWindow('window2', width, height)
    # cv.imshow('window2', img_b)
    # h = obj.histogram_gray()
    # plt.plot(h, color='black')
    # plt.xlim([0, 256])
    # plt.show()
    # hgbr = obj.histograms_BGR()
    # plt.plot(hgbr[0], color='b')
    # plt.plot(hgbr[1], color='g')
    # plt.plot(hgbr[2], color='r')
    # plt.xlim([0, 256])
    # plt.show()
    # print("Gray histogram features: ")
    # h_f = obj.histogram_gray_features()
    # print("Дисперсия: " + str(h_f[0]))
    # print("Момент 3го порядка: " + str(h_f[1]))
    # print("Момент 4го порядка: " + str(h_f[2]))
    # print("Медиана: " + str(h_f[3]))
    # print("Среднее: " + str(h_f[4]))
    # print("Максимум: " + str(h_f[5]))
    # print()
    # print("BGR histogram features: ")
    # hgbr_f = obj.histogram_BGR_features()
    # print("Дисперсия B: " + str(hgbr_f[0][0]))
    # print("Момент 3го порядка B: " + str(hgbr_f[1][0]))
    # print("Момент 4го порядка B: " + str(hgbr_f[2][0]))
    # print("Медиана B: " + str(hgbr_f[3][0]))
    # print("Среднее B: " + str(hgbr_f[4][0]))
    # print("Максимум B: " + str(hgbr_f[5][0]))
    # print()
    # print("Дисперсия G: " + str(hgbr_f[0][1]))
    # print("Момент 3го порядка G: " + str(hgbr_f[1][1]))
    # print("Момент 4го порядка G: " + str(hgbr_f[2][1]))
    # print("Медиана G: " + str(hgbr_f[3][1]))
    # print("Среднее G: " + str(hgbr_f[4][1]))
    # print("Максимум G: " + str(hgbr_f[5][1]))
    # print()
    # print("Дисперсия R: " + str(hgbr_f[0][2]))
    # print("Момент 3го порядка R: " + str(hgbr_f[1][2]))
    # print("Момент 4го порядка R: " + str(hgbr_f[2][2]))
    # print("Медиана R: " + str(hgbr_f[3][2]))
    # print("Среднее R: " + str(hgbr_f[4][2]))
    # print("Максимум R: " + str(hgbr_f[5][2]))
    # print(obj.patient_age)
    # print(obj.diagnosis)
    # cv.waitKey(0)
