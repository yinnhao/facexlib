import cv2
import numpy as np

def skin_detect_vectorized(image):
    # 将图像转换为float32类型，并将其归一化到0到1之间
    image = image.astype(np.float32) / 255.0
    R, G, B = image[:,:,2], image[:,:,1], image[:,:,0]

    # 条件1：R - G >= 10 / 255
    cond1 = (R - G) >= 10.0 / 255.0
    
    # 条件2：G > B
    cond2 = G > B
    
    # 计算 Sum, T1, T2
    Sum = R + G + B
    T1 = 156 * R - 52 * Sum
    T2 = 156 * G - 52 * Sum
    
    # 条件3：T1^2 + T2^2 >= (Sum^2) / 16
    cond3 = T1**2 + T2**2 >= (Sum**2) / 16
    
    # 计算 T3, Lower, Upper
    T3 = 10000 * G * Sum
    Lower = -7760 * R**2 + 5601 * R * Sum + 1766 * Sum**2
    Upper = -13767 * R**2 + 10743 * R * Sum + 1552 * Sum**2
    
    # 条件4和条件5：Lower < T3 < Upper
    cond4 = T3 > Lower
    cond5 = T3 < Upper
    
    # 最终的皮肤检测掩码
    # skin_mask = cond1 & cond2 & cond3 & cond4 & cond5
    skin_mask = cond1 & cond3 & cond4 & cond5
    
    # 将布尔掩码转换为uint8类型
    skin_mask = (skin_mask * 255).astype(np.uint8)
    
    return skin_mask

def visualize_skin_detection(image, skin_mask, alpha=0.4):
    # 创建一个红色的掩码图像
    mask = image.copy()
    mask[skin_mask == 255] = [255, 144, 30]  

    # 将掩码叠加在原图上，使用alpha参数控制透明度
    result = cv2.addWeighted(image, 0.6, mask, alpha, 0)
    
    return result

def main(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 检测皮肤区域
    skin_mask = skin_detect_vectorized(image)

    # 可视化皮肤检测结果
    result = visualize_skin_detection(image, skin_mask)

    # 显示原图和结果图
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Skin Detection', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果图
    cv2.imwrite('skin_detection_result_np.png', result)

if __name__ == "__main__":
    main("/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img/cftl_no_005_24s.png")
