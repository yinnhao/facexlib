import cv2
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.utils.real_esrganer import RealESRGANer
from torchvision.transforms.functional import normalize
import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
# from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
# from gfpgan.archs.gfpganv1_arch import GFPGANv1
# from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FaceEnhancer():
    """
    """

    def __init__(self, upscale=1, device=None, target_size=512, max_size=1024, use_origin_size=False, enhance_model=None, enhance_model_path=None):
        self.upscale = upscale
        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath=f'{ROOT_DIR}/facexlib/weights',
            parse_model='bisenet',
            target_size=target_size, 
            max_size=max_size, 
            use_origin_size=use_origin_size)

        if enhance_model:
            self.face_enhancer = RealESRGANer(
                scale=1,
                model_path=enhance_model_path,
                model=enhance_model,
                tile=0,
                tile_pad=40,
                pre_pad=0,
                half=True
            )
        

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face(save_cropped_path='/data/yh/FACE_2024/facexlib/result')

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            cv2.imwrite('/data/yh/FACE_2024/facexlib/result/face_crop_restored.png', restored_face)
            self.face_helper.add_restored_face(restored_face)
        
        if not has_aligned and paste_back:
            bg_img = img

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, soft_mask=False)
            vis_img = self.face_helper.paste_masks_to_input_image()
            # cv2.imwrite('/data/yh/FACE_2024/facexlib/result/face_parse.png', vis_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img, vis_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
        
    @torch.no_grad()
    def get_face_parsing(self, img, mask_type='face_mask'):
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_boxs(only_center_face=False, eye_dist_threshold=5)
        self.face_helper.get_face_enlarge()
        self.face_helper.get_crop_face()
        self.face_helper.get_face_parsing(mask_type=mask_type)
        vis_img = self.face_helper.draw_res_to_input_image(draw_box=True, mask_type=mask_type)
        return vis_img  
    
    @torch.no_grad()
    def enhance_face(self, img):
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_boxs(only_center_face=False, eye_dist_threshold=5)
        self.face_helper.get_face_enlarge()
        self.face_helper.get_crop_face()
        self.face_helper.get_face_parsing(soft_mask=True)
        for cropped_face in self.face_helper.cropped_faces:
            restore_crop_face, img_mode = self.face_enhancer.enhance(cropped_face)
            self.face_helper.add_restored_face(restore_crop_face)
        enhanced_res = self.face_helper.paste_restore_faces()
        return enhanced_res
    
    def analyze_face(self, img, mask_type='face_mask'):
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_boxs(only_center_face=False, eye_dist_threshold=5)
        self.face_helper.get_face_enlarge()
        self.face_helper.get_crop_face()
        self.face_helper.get_face_parsing(mask_type=mask_type)
        self.face_helper.get_face_histogram(mask_type=mask_type)
        vis_img = self.face_helper.draw_res_to_input_image(draw_box=True, draw_mask=False, mask_type=mask_type, draw_hist=True, draw_contrast=True)
        return vis_img 


def main(args):
    # initialize model
    target_size = args.target_size
    max_size = args.max_size
    use_origin_size = args.use_origin_size
    task = args.task

    img = cv2.imread(args.img_path)
    vis_img = None
    if task == 'enhance':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
        model_path = "/data/yh/SR2023/Real-ESRGAN/weights/RealESRGAN_x1plus_590000.pth"
        face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size, 
                                    enhance_model=model, enhance_model_path=model_path)
        vis_img = face_enhancer.enhance_face(img)
        

    elif task == 'parsing':
        face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
        vis_img = face_enhancer.get_face_parsing(img, mask_type='face_mask')
    
    elif task == 'skin':
        face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
        vis_img = face_enhancer.get_face_parsing(img, mask_type='skin_mask')
    
    elif task == 'analyze':
        face_enhancer = FaceEnhancer(target_size=target_size, max_size=max_size, use_origin_size=use_origin_size)
        vis_img = face_enhancer.analyze_face(img, mask_type='skin_mask')
    
    cv2.imwrite(args.save_path, vis_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select/SDR2822_709-0227.png')
    parser.add_argument('--save_path', type=str, default='/data/yh/FACE_2024/facexlib/result/SDR2822_709-0227_skin_analyze.png')
    parser.add_argument(
        '--model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--output_txt', action='store_true')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--max_size', type=int, default=1024)
    parser.add_argument('--use_origin_size', action='store_true')
    parser.add_argument('--task', type=str, default='parsing', help='parsing | detection | enhance | skin | analyze')
    args = parser.parse_args()

    main(args)