import cv2
import numpy as np

def skin_detect(Red, Green, Blue):
    skinflag = 0
    if Red - Green >= 10.0 / 255.0:
        if Green > Blue:
            Sum = Red + Green + Blue
            T1 = 156 * Red - 52 * Sum
            T2 = 156 * Green - 52 * Sum
            if T1 * T1 + T2 * T2 >= (Sum * Sum) / 16:
                T3 = 10000 * Green * Sum
                Lower = -7760 * Red * Red + 5601 * Red * Sum + 1766 * Sum * Sum
                if T3 > Lower:
                    Upper = -13767 * Red * Red + 10743 * Red * Sum + 1552 * Sum * Sum
                    if T3 < Upper:
                        skinflag = 1
    return skinflag

def detect_skin_in_image(image):
    # 将图像转换为float32类型，并将其归一化到0到1之间
    image = image.astype(np.float32) / 255.0

    # 创建一个与输入图像大小相同的全零数组，用来保存检测到的皮肤区域
    skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 遍历图像的每个像素
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            B, G, R = image[y, x]
            if skin_detect(R, G, B):
                skin_mask[y, x] = 255

    return skin_mask

def visualize_skin_detection(image, skin_mask, alpha=0.4):
    # 创建一个红色的掩码图像
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[skin_mask == 255] = [0, 0, 255]  # 红色

    # 将掩码叠加在原图上，使用alpha参数控制透明度
    result = cv2.addWeighted(image, 1.0, mask, alpha, 0)
    return result

def main(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 检测皮肤区域
    skin_mask = detect_skin_in_image(image)

    # 可视化皮肤检测结果
    result = visualize_skin_detection(image, skin_mask)

    # # 显示原图和结果图
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Skin Detection', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果图
    cv2.imwrite('skin_detection_result.png', result)

if __name__ == "__main__":
    main("/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img/cftl_no_005_24s.png")
