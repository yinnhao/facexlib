input=/mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin
output=/mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/face_enhance
mkdir -p $output
for file in `ls $input`
do 
python ../inference/video_face_process.py --video_path $input/$file --save_path $output/${file%.*}_enh.mp4 --qp 12 --task enhance
# break
done