import argparse
import cv2
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection


def main(args):
    # initialize model
    det_net = init_detection_model(args.model_name, half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.5, use_origin_size=False)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        print(bboxes)
        if args.output_txt:
            txt_path = args.save_path.replace('.png', '.txt')
            with open(txt_path, 'w') as f:
                for bbox in bboxes:
                    # f.write(' '.join(map(str, bbox)) + '\n')
                    # f.write(' '.join('face', str(bbox[4]), map(str, bbox[:4])) + '\n')
                    f.write(f'face {bbox[4]} {int(bbox[0])} {int(bbox[1])} {int(bbox[2]) + 1} {int(bbox[3])+1}\n')
        visualize_detection(img, bboxes, args.save_path)

# python inference_detection.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_detect.png
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png')
    parser.add_argument('--save_path', type=str, default='/data/yh/FACE_2024/facexlib/result/yuexia3_madong_face.png')
    parser.add_argument(
        '--model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--output_txt', action='store_true')
    args = parser.parse_args()

    main(args)
