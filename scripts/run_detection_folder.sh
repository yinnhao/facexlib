# input=/mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src
# output=/data/yh/FACE_2024/facexlib/result/yuexia_det
input=/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img
output=/data/yh/FACE_2024/facexlib/result/val_set_ori_size
mkdir -p $output
for file in `ls $input`
do 
python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt --target_size 512 --max_size 1024
# python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt --target_size 1600 --max_size 2150 
# python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt --target_size 1600 --max_size 2150 --use_origin_size
done

