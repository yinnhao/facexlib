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
from face_process import FaceEnhancer
from basicsr.archs.rrdbnet_arch import RRDBNet
def dump_img(x, path):
    y = (cp.clip(x, 0, 1) * 255).round().astype(cp.uint8)
    y = cp.asnumpy(y)
    y = np.transpose(y[[2, 1, 0], :, :], (1, 2, 0))
    cv2.imwrite(path, y)

def dump_yuv(x, path):
    y = cp.asnumpy(x)
    y.tofile(path)

DEBUG = False
class FaceInfer(video_infer):
    def __init__(self, file_name, save_name, encode_params, model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p", decode_out_pix_fmt="yuv420p" 
                 ,use_fp16=True, target_size=512, max_size=1024, use_origin_size=False, task='parsing', **decode_param_dict) -> None:
        super().__init__(file_name, save_name, encode_params, model, scale, in_pix_fmt, out_pix_fmt, decode_out_pix_fmt, **decode_param_dict)
        
        self.face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
        self.task = task
    
    def forward(self, x):
        
        y = x[:,:,::-1].copy()

        if self.task == 'enhance':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
            model_path = "/data/yh/SR2023/Real-ESRGAN/weights/RealESRGAN_x1plus_590000.pth"
            face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size, 
                                        enhance_model=model, enhance_model_path=model_path)
            vis_img = face_enhancer.enhance_face(y)
        
        elif self.task == 'parsing':
            face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
            vis_img = face_enhancer.get_face_parsing(y, mask_type='face_mask')
        
        elif self.task == 'skin':
            face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
            vis_img = face_enhancer.get_face_parsing(y, mask_type='skin_mask')
        elif self.task == 'analyze':
            face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
            vis_img = face_enhancer.analyze_face(y, mask_type='skin_mask')
        else:
            raise ValueError("task should be one of 'enhance', 'parsing', 'skin'")

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        return vis_img




if __name__ == '__main__':
    # /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR0357_709.mp4
    # /data/yh/FACE_2024/facexlib/SDR0357_709_1s.mp4
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Your program description')

    # 添加命令行参数
    parser.add_argument('--video_path', type=str, help='Input file name')
    parser.add_argument('--save_path', type=str, help='Output file name')
    parser.add_argument('--qp', type=int, default=16, help='qp value')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--max_size', type=int, default=1024)
    parser.add_argument('--use_origin_size', action='store_true')
    parser.add_argument('--task', type=str, default='parsing', help='enhance | parsing | skin | analyze')
    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    video_path = args.video_path
    save_path = args.save_path
    

    target_size = args.target_size
    max_size = args.max_size
    use_origin_size = args.use_origin_size

    qp = args.qp
    # encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
    # encode_params = ("libx265", "x265-params", "qp={}".format(str(qp)))
    encode_params = ("libx264", "x264opts", "qp={}".format(str(qp)))

    infer = FaceInfer(video_path, save_path, encode_params, in_pix_fmt="rgb24", out_pix_fmt="rgb24", 
                           decode_out_pix_fmt="yuv420p", scale=1, target_size=target_size, max_size=max_size, 
                           use_origin_size=use_origin_size, task=args.task)
    infer.infer()
    
    
    