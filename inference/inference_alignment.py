import argparse
import cv2
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.visualization import visualize_alignment


def main(args):
    # initialize model
    align_net = init_alignment_model(args.model_name, device=args.device)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        landmarks = align_net.get_landmarks(img)
        if args.to68:
            landmarks = landmark_98_to_68(landmarks)
        visualize_alignment(img, [landmarks], args.save_path)

# python inference_alignment.py --img_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_cvwarp_00.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_align.png
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    parser.add_argument('--model_name', type=str, default='awing_fan')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--to68', action='store_true')
    args = parser.parse_args()

    main(args)
