# ![icon](assets/icon_small.png) FaceXLib

[![PyPI](https://img.shields.io/pypi/v/facexlib)](https://pypi.org/project/facexlib/)
[![download](https://img.shields.io/github/downloads/xinntao/facexlib/total.svg)](https://github.com/xinntao/facexlib/releases)
[![Open issue](https://img.shields.io/github/issues/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/facexlib)](https://github.com/xinntao/facexlib/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/facexlib.svg)](https://github.com/xinntao/facexlib/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/facexlib/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/facexlib/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/publish-pip.yml)

[English](README.md) **|** [简体中文](README_CN.md)

---

**facexlib** aims at providing ready-to-use **face-related** functions based on current SOTA open-source methods. <br>
Only PyTorch reference codes are available. For training or fine-tuning, please refer to their original repositories listed below. <br>
Note that we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use.

If facexlib is helpful in your projects, please help to :star: this repo. Thanks:blush: <br>
Other recommended projects: &emsp; :arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) &emsp; :arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN) &emsp; :arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR)

---

## :sparkles: Functions

| Function | Sources  | Original LICENSE |
| :--- | :---:        |     :---:      |
| [Detection](facexlib/detection/README.md) | [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MIT |
| [Alignment](facexlib/alignment/README.md) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [Recognition](facexlib/recognition/README.md) | [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) | MIT |
| [Parsing](facexlib/parsing/README.md) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [Matting](facexlib/matting/README.md) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [Headpose](facexlib/headpose/README.md) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [Tracking](facexlib/tracking/README.md) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [Assessment](facexlib/assessment/README.md) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [Utils](facexlib/utils/README.md) | Face Restoration Helper | - |

## :eyes: Demo and Tutorials

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

```bash
pip install facexlib
```

### Pre-trained models

It will **automatically** download pre-trained models at the first inference. <br>
If your network is not stable, you can download in advance (may with other download tools), and put them in the folder: `PACKAGE_ROOT_PATH/facexlib/weights`.

## :scroll: License and Acknowledgement

This project is released under the MIT license. <br>

## :e-mail: Contact

If you have any question, open an issue or email `xintao.wang@outlook.com`.


## 原代码库的推理命令行
```shell
# 1. detection
python inference_detection.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_detect.png
# 2. matting
python inference_matting.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_mat.png
# 3. alignment（关键点） 需要人脸小图
python inference_alignment.py --img_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_cvwarp_00.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_align.png
# 4. parsing （人脸解析分割成更精细的部分） 需要人脸小图
# bisenet: 效果比较好
python inference_parsing.py --input /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_cvwarp_00.png --output /data/yh/FACE_2024/facexlib/result
# parsenet: 效果不好
python inference_parsing_parsenet.py --input /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_cvwarp_00.png --output /data/yh/FACE_2024/facexlib/result
```

## 自定义接口命令行

### 图片推理命令

#### 1. 人脸检测
```shell
python inference/inference_detection.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_detect.png --half --output_txt --target_size 512 --max_size 1024
```
参数介绍：
- `--half`：是否使用half精度
- `--output_txt`：是否输出txt文件， txt文件与图片同名，内容为检测到的人脸信息，每一行为一个人脸的信息，包括人脸的置信度和坐标，坐标为左上角和右下角的坐标，例如'face 1.0 937 185 1150 467'
- `--target_size`：将图像的短边缩放到的目标大小
- `--max_size`：缩放后的最大尺寸，如果缩放完之后，图像的长边大于max_size，则再次将其缩放到max_size
如果不缩放需要使用'--use_origin_size'参数

脚本：scripts/run_detection_folder.sh

#### 2. 人脸分割（使用face detection + face parsing）
```shell
python inference/face_process.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_parsing.jpg --half --output_txt --target_size 512 --max_size 1024 --task parsing
```

使用`inference/face_process.py`， 指定`--task`为`parsing`

脚本：scripts/run_parsing_folder.sh

#### 3. 皮肤分割 （使用face detection + face parsing, 19类区域，1号区域就是皮肤）

```shell
python inference/face_process.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_skin.jpg --half --output_txt --target_size 512 --max_size 1024 --task skin
```

使用`inference/face_process.py`, 指定`--task`为`skin`

脚本：scripts/run_skin_folder.sh


#### 4. 人脸增强（face detection + face parsing）

```shell
python inference/face_process.py --img_path /mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src/yuexia3_madong_face.png --save_path /data/yh/FACE_2024/facexlib/result/yuexia3_madong_face_enhance.png --half --output_txt --target_size 512 --max_size 1024 --task enhance
```
使用`inference/face_process.py`, 指定`--task`为`enhance`

脚本：scripts/run_enhance_folder.sh


### 视频推理命令行

#### 1. 人脸检测
```shell

python video_inference_detection.py --video_path SDR0357_709_1s.mp4 --save_path SDR0357_709_1s_res.mp4
```


#### 2. 人脸分割（使用face detection + face parsing）
```shell
python inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_parsing.mp4 --task parsing --qp 20
```

#### 3. 皮肤分割 （使用face detection + face parsing, 19类区域，1号区域就是皮肤）

```shell
python inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_skin.mp4 --task skin --qp 20
```

#### 4. 人脸增强（face detection + face parsing）

```shell
python inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_enhance.mp4 --task enhance --qp 12
```

人脸分割、皮肤分割和人脸增强都是使用的`video_face_process.py`, 通过--task指定