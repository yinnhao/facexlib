input=/mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src
output=/data/yh/FACE_2024/facexlib/result/yuexia_det
mkdir -p $output
for file in `ls $input`
do 
python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}_det.png
done