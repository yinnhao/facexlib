import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ffpipe import video_infer
# from video_process_lib import lib_numpy
import cupy as cp
from torch.utils.dlpack import to_dlpack,from_dlpack
from cupy import fromDlpack
import torch
# from onnx2trt.trt_python_api import load_cuda_engine
# import tensorrt as trt
import argparse
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection

def dump_img(x, path):
    y = (cp.clip(x, 0, 1) * 255).round().astype(cp.uint8)
    y = cp.asnumpy(y)
    y = np.transpose(y[[2, 1, 0], :, :], (1, 2, 0))
    cv2.imwrite(path, y)

def dump_yuv(x, path):
    y = cp.asnumpy(x)
    y.tofile(path)
DEBUG = False
class InferDET(video_infer):
    def __init__(self, file_name, save_name, encode_params, model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p", decode_out_pix_fmt="yuv420p" ,use_fp16=True, **decode_param_dict) -> None:
        super().__init__(file_name, save_name, encode_params, model, scale, in_pix_fmt, out_pix_fmt, decode_out_pix_fmt, **decode_param_dict)
        self.model_name = model
        self.det_net = init_detection_model(self.model_name, half=use_fp16)
        self.use_fp16 = use_fp16
    
    def forward(self, x):
        # y = lib_numpy.yuv_to_1_domain(x, 16.0, 235.0, 240.0)
        # y = lib_numpy.yuv2rgb_709_repalce(y)
        # y = y[:, :, ::-1] # BGR
        # y[:] = np.round(y * 255)
        y = x[:,:,::-1].copy()
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(y, 0.97)
        print(bboxes)
        y = visualize_detection(y.astype(np.uint8), bboxes, return_img=True)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test.jpg", y)
        
        return y




if __name__ == '__main__':
    # /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR0357_709.mp4
    # /data/yh/FACE_2024/facexlib/SDR0357_709_1s.mp4
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Your program description')

    # 添加命令行参数
    parser.add_argument('--file_name', type=str, help='Input file name')
    parser.add_argument('--save_name', type=str, help='Output file name')
    parser.add_argument('--det_model', type=str, default='retinaface_resnet50', help='detection model name')
    parser.add_argument('--qp', type=int, default=16, help='qp value')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    file_name = args.file_name
    save_name = args.save_name
    det_model_name = args.det_model
    qp = args.qp
    # encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
    # encode_params = ("libx265", "x265-params", "qp={}".format(str(qp)))
    encode_params = ("libx264", "x264opts", "qp={}".format(str(qp)))

    infer = InferDET(file_name, save_name, encode_params, in_pix_fmt="rgb24", out_pix_fmt="rgb24", 
                           decode_out_pix_fmt="yuv420p", model=det_model_name, scale=1)
    infer.infer()
    
    
    