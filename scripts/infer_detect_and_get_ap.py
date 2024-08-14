import os
import subprocess


def infer_folder(input_dir, output_dir, **kwargs):
    target_size = kwargs['target_size']
    max_size = kwargs['max_size']
    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        # 拼接输入和输出文件的路径
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}.png')

        # 执行命令
        subprocess.run([
            'python', '../inference/inference_detection.py',
            '--img_path', input_path,
            '--save_path', output_path,
            '--half',
            '--output_txt',
            '--target_size', f'{target_size}',
            '--max_size', f'{max_size}',
            '--use_origin_size'
        ])

def get_ap(gt_dir, input_dir, dr_dir, out_dir, iou_dic):
    iou_list = []
    for k, v in iou_dic.items():
        iou_list.append(k)
        iou_list.append(str(v))

    command = [
        'python', 'map.py',
        '--gt_path', gt_dir,
        '--dr_path', dr_dir,
        '--img_path', input_dir,
        '--out_path', out_dir,
        '--set-class-iou', *iou_list,
    ]
    subprocess.run(command)
    
    
   

if __name__ == '__main__':
    
    # 设置缩放尺寸
    target_size = 1600
    max_size = 2150

    # 设置输入路径
    input_dir = '/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img'
    # 设置输出路径
    output_dir = '/data/yh/FACE_2024/facexlib/result/val_set_ori_size'
    # 真实标签路径
    gt_dir = '/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/label/txt'
    # ap log路径，包含相关可视化结果
    ap_log_path = '/data/yh/FACE_2024/facexlib/result/val_set_ori_size/ap_log'
    # 创建输出目录，如果不存在的话
    os.makedirs(output_dir, exist_ok=True)
    
    # 推理检测
    kwargs = {
                'target_size': target_size,
                'max_size': max_size
             }
    infer_folder(input_dir, output_dir, **kwargs)
    # 根据检测结果，计算AP
    iou_dic = {
                'face': 0.7
              }
    
    get_ap(gt_dir, input_dir, output_dir, ap_log_path, iou_dic)
