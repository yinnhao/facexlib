import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from facexlib.detection import init_detection_model
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor, imwrite
from facexlib.utils.tra_skin_detect_numpy import skin_detect_vectorized
import matplotlib.pyplot as plt

def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None,
                 model_rootpath=None,
                 parse_model='parsenet',
                 target_size=1600, 
                 max_size=2150,
                 use_origin_size=False):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = upscale_factor
        self.use_origin_size = use_origin_size
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []
        self.face_masks = []
        self.det_faces_enlarge = []
        self.skin_masks = []
        self.face_hists = []

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # init face detection model
        self.face_det = init_detection_model(det_model, half=False, device=self.device, model_rootpath=model_rootpath, target_size=target_size, max_size=max_size)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(model_name=parse_model, device=self.device, model_rootpath=model_rootpath)

    def set_upscale_factor(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def read_image(self, img):
        """img can be image path or cv2 loaded image."""
        # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
        if isinstance(img, str):
            img = cv2.imread(img)

        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img = img[:, :, 0:3]

        self.input_img = img

    def get_face_boxs(self,
                      only_keep_largest=False,
                      only_center_face=False,
                      resize=None,
                      blur_ratio=0.01,
                      eye_dist_threshold=None):
        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = min(h, w) / resize
            h, w = int(h / scale), int(w / scale)
            input_img = cv2.resize(self.input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        with torch.no_grad():
            bboxes = self.face_det.detect_faces(input_img, 0.97, use_origin_size=self.use_origin_size) * scale
        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue
            self.det_faces.append(bbox[0:5])
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
        return len(self.det_faces)
    
    def get_crop_face(self):
        input_img = self.input_img
        h, w = input_img.shape[:2]
        for box in self.det_faces_enlarge:
            cropped_face = input_img[box[1]:box[3], box[0]:box[2], :]
            # cv2.imwrite("/data/yh/FACE_2024/facexlib/result/face_crop.jpg", cropped_face)
            self.cropped_faces.append(cropped_face)

    def get_face_enlarge(self):
        def expand_box(x0, y0, x1, y1, ratio1, ratio2):
            # 计算box的中心点
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            
            # 计算原来的宽度和高度
            width = x1 - x0
            height = y1 - y0
            
            # 计算新的宽度和高度
            new_width = width * ratio1
            new_height = height * ratio2
            
            # 计算新的box的左上和右下坐标
            new_x0 = center_x - new_width / 2
            new_y0 = center_y - new_height / 2
            new_x1 = center_x + new_width / 2
            new_y1 = center_y + new_height / 2
            
            return new_x0, new_y0, new_x1, new_y1


        input_img = self.input_img
        h, w = input_img.shape[:2]
        for box in self.det_faces:
            box = np.array(box)
            box[0], box[1], box[2], box[3] = expand_box(box[0], box[1], box[2], box[3], 1.7, 1.3)
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)
            box[2] = min(box[2], w - 1)
            box[3] = min(box[3], h - 1)
            box = box.astype(int)
            self.det_faces_enlarge.append(box)

    def get_face_parsing(self, soft_mask=False, mask_type="face_mask", parsing_method='bisenet'):
        '''
        功能:获取人脸的分割mask
        soft_mask: 是否对mask进行高斯模糊处理，使得边界变得模糊
        mask_type: face_mask (完整的人脸) or skin_mask (皮肤区域, 不包括眼睛、嘴巴、鼻子)
        parsing_method: bisenet or traditional [传统的肤色检测方法(在RGB空间根据一些颜色特征)]
        
        '''
        input_img = self.input_img
        h, w = input_img.shape[:2]
        for face_id, cropped_face in enumerate(self.cropped_faces):
                big_mask = np.zeros(input_img.shape[:2])
                
                face_input = cv2.resize(cropped_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                if parsing_method == 'bisenet':
                    face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                    normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    face_input = torch.unsqueeze(face_input, 0).to(self.device)
                    with torch.no_grad():
                        out = self.face_parse(face_input)[0]
                    out = out.argmax(dim=1).squeeze().cpu().numpy()

                    mask = np.zeros(out.shape)
                    if mask_type == "face_mask":
                        MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                    elif mask_type == "skin_mask":
                        MASK_COLORMAP = [0] * 19
                        MASK_COLORMAP[1] = 255
                    for idx, color in enumerate(MASK_COLORMAP):
                        mask[out == idx] = color
                    if soft_mask:
                        #  blur the mask
                        # mask = cv2.GaussianBlur(mask, (101, 101), 11)
                        # mask = cv2.GaussianBlur(mask, (101, 101), 11)
                        mask = cv2.GaussianBlur(mask, (5, 5), 11)

                elif parsing_method == 'traditional':
                    mask = skin_detect_vectorized(face_input)
                    
                mask = mask / 255.
                # cv2.imwrite('/data/yh/FACE_2024/facexlib/result/mask_small.png', (mask * 255).astype(np.uint8))
                mask = cv2.resize(mask, (cropped_face.shape[1], cropped_face.shape[0]))
                # cv2.imwrite('/data/yh/FACE_2024/facexlib/result/mask_origin.png', (mask * 255).astype(np.uint8))
                box = self.det_faces_enlarge[face_id]
                big_mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = mask[:, :]
                # inv_soft_mask = mask[:, :, None]
                if mask_type == "face_mask":
                    self.face_masks.append(big_mask)
                elif mask_type == "skin_mask":
                    self.skin_masks.append(big_mask)



    def get_face_landmarks_5(self,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = min(h, w) / resize
            h, w = int(h / scale), int(w / scale)
            input_img = cv2.resize(self.input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        with torch.no_grad():
            bboxes = self.face_det.detect_faces(input_img, 0.97, use_origin_size=self.use_origin_size) * scale
        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]

        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces with face template.
        """
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            cv2.imwrite('/data/yh/FACE_2024/facexlib/result/input_img.png', input_img)
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def draw_res_to_input_image(self, draw_box=False, draw_mask=False, mask_type="face_mask", draw_hist=False, draw_contrast=False):
        def plt2cv(hist):
            '''
            将 matplotlib 绘制的图像转换为 OpenCV 可用的格式
            :param hist: 直方图数据
            '''
            plt.figure()
            plt.title("Brightness Histogram")
            plt.xlabel("Brightness Value")
            plt.ylabel("Pixel Count")
            plt.bar(range(256), hist, width=1.0, color='black')
            plt.xlim([0, 256])
            plt.draw()
            waveform_img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            waveform_img = waveform_img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return waveform_img
        
        def calculate_percentile_values(hist, lower_percentile=1, upper_percentile=99):
            '''
            从直方图中计算亮度值的 1% 和 99% 分位点
            '''
            # 计算累积直方图（CDF）
            cdf = hist.cumsum()
            cdf_normalized = cdf / (cdf.max() + 1e-8)  # 归一化 CDF

            # 获取 1% 和 99% 分位点的亮度值
            lower_value = np.searchsorted(cdf_normalized, lower_percentile / 100.0)
            upper_value = np.searchsorted(cdf_normalized, upper_percentile / 100.0)

            return lower_value, upper_value
        
        input_img = self.input_img
        vis_mask = input_img.copy()
        if mask_type == "face_mask":
            masks = self.face_masks
        elif mask_type == "skin_mask":
            masks = self.skin_masks

        face_nums = len(masks)
        # 如果需要绘制 mask，则将 mask 绘制到 vis_img 上
        if draw_mask:
            for i in range(face_nums):
                # draw mask
                mask = masks[i]
                mask = np.round(mask * 255).astype(np.uint8)
                index = np.where(mask == 255)
                vis_mask[index[0], index[1], :] = [255, 144, 30]
            
            # add mask to origin img
            vis_img = cv2.addWeighted(input_img, 0.4, vis_mask, 0.6, 0)
        else:
            vis_img = input_img.copy()

        for i in range(face_nums):
            # draw box
            if draw_box:
                box = self.det_faces[i].astype(int)
                cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                box_enlarge = self.det_faces_enlarge[i].astype(int)
                cv2.rectangle(vis_img, (box_enlarge[0], box_enlarge[1]), (box_enlarge[2], box_enlarge[3]), (0, 0, 255), 2)
            # draw hist
            if draw_hist:
                # 获取直方图
                hist = self.face_hists[i]
                hist_np = plt2cv(hist)
                # print(hist_np.shape)
                # print(vis_img.shape)
                hist_h, hist_w, _ = hist_np.shape
                base_h, base_w, _ = vis_img.shape
                # 将直方图绘制到 vis_img 上。位置为 box 的左下角
                box = self.det_faces[i].astype(int)
                x, y = (box[0], box[3])
                if y + hist_h > base_h or x + hist_w > base_w:
                    y = base_h - hist_h
                    x = base_w - hist_w
                vis_img[y:y + hist_h, x:x + hist_w, :] = cv2.addWeighted(vis_img[y:y + hist_h, x:x + hist_w, :], 0.4, hist_np, 0.6, 0)
            # 将对比度信息写在图像上进行可视化：1% 和 99% 分位点，以及亮度范围
            if draw_contrast:
                low_value, up_value = calculate_percentile_values(hist)
                text_lower = f"1% Percentile: {low_value}"
                text_upper = f"99% Percentile: {up_value}"
                text_range = f"Range: {up_value - low_value + 1}"
                # 在图像上绘制文本
                x, y = box[0], box[1]
                cv2.putText(vis_img, text_lower, (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, text_upper, (x, y+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, text_range, (x, y+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
        return vis_img

    def get_face_histogram(self, mask_type='face_mask'):
        '''
        获取face区域的直方图，并保存下来
        '''
        input_img = self.input_img
        if mask_type == "face_mask":
            masks = self.face_masks
        elif mask_type == "skin_mask":
            masks = self.skin_masks
        for mask in masks:
            mask = mask.astype(np.uint8)
            gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])
            hist = hist.flatten()  # 将直方图转换为一维数组
            self.face_hists.append(hist)
    

    def paste_restore_faces(self):
        upsample_img = self.input_img
        face_pad = upsample_img.copy()
        for box, mask, face in zip(self.det_faces_enlarge, self.face_masks, self.restored_faces):
            face_pad[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = face[:, :, :]
            mask = np.expand_dims(mask, 2)
            upsample_img = mask * face_pad + (1 - mask) * upsample_img
        upsample_img = upsample_img.astype(np.uint8)
        return upsample_img

    def paste_faces_to_input_image(self, save_path=None, upsample_img=None, soft_mask=True):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            # simply resize the background
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # Add an offset to inverse affine matrix, for more precise back alignment
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                mask = np.zeros(out.shape)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(MASK_COLORMAP):
                    mask[out == idx] = color
                if soft_mask:
                    #  blur the mask
                    mask = cv2.GaussianBlur(mask, (101, 101), 11)
                    mask = cv2.GaussianBlur(mask, (101, 101), 11)
                # remove the black borders
                # thres = 10
                # mask[:thres, :] = 0
                # mask[-thres:, :] = 0
                # mask[:, :thres] = 0
                # mask[:, -thres:] = 0
                mask = mask / 255.
                cv2.imwrite('/data/yh/FACE_2024/facexlib/result/mask_small.png', (mask * 255).astype(np.uint8))
                mask = cv2.resize(mask, restored_face.shape[:2])
                mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
                cv2.imwrite('/data/yh/FACE_2024/facexlib/result/mask.png', (mask * 255).astype(np.uint8))
                inv_soft_mask = mask[:, :, None]
                self.face_masks.append(inv_soft_mask)
                pasted_face = inv_restored

            else:  # use square parse maps
                mask = np.ones(self.face_size, dtype=np.float32)
                inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
                # remove the black borders
                inv_mask_erosion = cv2.erode(
                    inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
                pasted_face = inv_mask_erosion[:, :, None] * inv_restored
                total_face_area = np.sum(inv_mask_erosion)  # // 3
                # compute the fusion edge based on the area of face
                w_edge = int(total_face_area**0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                if len(upsample_img.shape) == 2:  # upsample_img is gray image
                    upsample_img = upsample_img[:, :, None]
                inv_soft_mask = inv_soft_mask[:, :, None]

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []
        self.det_faces_enlarge = []
        self.face_masks = []
        self.skin_masks = []
        self.face_hists = []
        
