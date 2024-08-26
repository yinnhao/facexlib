import cv2
import numpy as np
def draw_histogram(image):
    histogram = np.zeros((256, 256, 3), dtype=np.uint8)

    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        for j in range(256):
            cv2.line(histogram, (j, 256), (j, 256 - int(hist[j])), (255, 255, 255) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255))
    
    return histogram

import matplotlib.pyplot as plt

def draw_brightness_histogram(image, mask=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])
    hist = hist.flatten()  # 将直方图转换为一维数组

    plt.figure()
    plt.title("Brightness Histogram")
    plt.xlabel("Brightness Value")
    plt.ylabel("Pixel Count")
    plt.bar(range(256), hist, width=1.0, color='black')
    plt.xlim([0, 256])
    # plt.show()
    plt.savefig("histogram_2.jpg")

def draw_waveform(image):
    height, width = image.shape[:2]
    waveform = np.zeros((256, width, 3), dtype=np.uint8)  # 初始化空白图像

    for x in range(width):
        column = image[:, x, 0]  # 提取当前列的亮度值 (灰度图像)
        for y in range(height):
            intensity = column[y]
            waveform[255-intensity, x] += 1  # 将亮度值映射到波形图上
    
    waveform = cv2.normalize(waveform, None, 0, 255, cv2.NORM_MINMAX)
    waveform = cv2.applyColorMap(waveform, cv2.COLORMAP_HOT)  # 颜色映射便于观察

    return waveform

# image = cv2.imread('input_image.jpg')
# 



image = cv2.imread('/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img/cftl_no_005_24s.png')
cv2.imwrite('input_image.jpg', image)
histogram = draw_histogram(image)
cv2.imwrite('histogram.jpg', histogram)

draw_brightness_histogram(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
waveform = draw_waveform(cv2.merge([gray_image, gray_image, gray_image]))
waveform_and_input_img = np.vstack((image, waveform))
cv2.imwrite('waveform.jpg', waveform_and_input_img)