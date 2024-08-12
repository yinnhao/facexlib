# input=/mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src
# output=/data/yh/FACE_2024/facexlib/result/yuexia_det
input=/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img
output=/data/yh/FACE_2024/facexlib/result/val_set_det_conf_0.5
mkdir -p $output
for file in `ls $input`
do 
python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt
done