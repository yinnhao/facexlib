# input=/mnt/ec-data2/ivs/1080p/zyh/testset/sr/yuexia_src
# input=/mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select
# output=/data/yh/FACE_2024/facexlib/result/yuexia_det
input=/mnt/ec-data2/ivs/1080p/zyh/dataset/face_detection/val_set/img
output=/data/yh/FACE_2024/facexlib/result/val_512_skin
mkdir -p $output
for file in `ls $input`
do 
# python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt --target_size 512 --max_size 1024
# python ../inference/inference_detection.py --img_path $input/$file --save_path $output/${file%.*}.png --half --output_txt --target_size 1600 --max_size 2150 
python ../inference/face_process.py --img_path $input/$file --save_path $output/${file%.*}.jpg --half --output_txt --target_size 512 --max_size 1024 --task skin
done

